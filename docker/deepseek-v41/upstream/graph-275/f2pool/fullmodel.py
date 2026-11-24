#!/usr/bin/env python3
"""Exclusive full-model F2Pool test on the benchmark host. Start only for public test prompts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
GRAPH = HERE.parents[1]
HOST = REPO.parent
NATIVE_CONFIG = os.environ.get('PEARL_NATIVE_IDLE_CONFIG')
RUN = Path(NATIVE_CONFIG).resolve().parent if NATIVE_CONFIG else HOST / 'live-f2pool-verification'
STATE = RUN / 'serving'
OLD = HOST / 'graph-production-275'
PROTO = REPO / 'work/pearl-inference-20261123/prototype'
ORIGINAL = Path('/mnt/hot/dsv41_state/production')
PREFIX = 'dsv41-native-' if NATIVE_CONFIG else 'dsv41-f2pool-'
BROKER = PREFIX+'fence'
CANDIDATE = PREFIX+'model'
GUARD = PREFIX+'guard'
RESTORE = PREFIX+'restore'
GATEWAY = PREFIX+'gateway'
SAMPLER = PREFIX+'sampler'
SOCKET = Path(os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')) / BROKER / 'broker.sock'
SMOKES = ('smoke.json', 'smoke-structured.json', 'smoke-tools.json', 'smoke-vision.json')
DOCKER = ('docker', '--context', 'shared-operator')
PYTHON = '/usr/bin/python3'
GPU_TEMP_CUTOFF_C = 92


def run(*args, timeout=25):
    return subprocess.check_output([str(a) for a in args], text=True, timeout=timeout).strip()


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0


def unit(name, *command, properties=(), env=(), output=None):
    args = ['systemd-run', '--user', '--collect', '--unit=' + name, '-p', 'KillMode=control-group',
            '-p', 'UMask=0077']
    for item in properties:
        args += ['-p', item]
    if output:
        args += ['-p', 'StandardOutput=append:' + str(output), '-p', 'StandardError=append:' + str(output)]
    args.extend('--setenv=' + item for item in env)
    if NATIVE_CONFIG:
        args.append('--setenv=PEARL_NATIVE_IDLE_CONFIG='+NATIVE_CONFIG)
    run(*args, *command)


def api(port, key, path, payload=None):
    headers = {'Content-Type':'application/json', 'Authorization':'Bearer ' + key.read_text().strip()}
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(f'http://127.0.0.1:{port}' + path, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read()


def write(name, data):
    target = STATE / name
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(target)


def cards():
    rows = run('nvidia-smi', '--query-gpu=index,power.limit,temperature.gpu',
               '--format=csv,noheader,nounits').splitlines()
    data = [tuple(float(n.strip()) for n in row.split(',')) for row in rows]
    if [r[0] for r in data] != list(range(4)) or any(r[1] != 275 for r in data):
        raise RuntimeError('Expected four original 275 W GPUs')
    return [int(r[2]) for r in data]


def namespace_state():
    script = GRAPH.parent / 'full-canary/host-run/miners/check_restored_limits.py'
    return json.loads(run(PYTHON, script))


def original_state():
    api(8000, ORIGINAL/'api-key', '/health')
    info = json.loads(api(8000, ORIGINAL/'api-key', '/server_info'))
    return dict(container=run(*DOCKER,'inspect','-f','{{.Id}} {{.State.Status}}','dsv41'),
                capacity=info['max_total_num_tokens'], decode=info['cuda_graph_config']['decode']['backend'],
                gpu_quota=[v['actual_hard_mib'] for v in namespace_state()['rows']], paid=run('systemctl','show','pearl-f2pool','-p','ActiveState','--value'))


def check_artifacts():
    lock = json.loads((GRAPH/'build-lock.json').read_text())
    files = {HOST/'graph-canary-code/lib'/name: digest for name, digest in lock['tested_library_sha256'].items()}
    for name, digest in json.loads((GRAPH/'measurement-setup.json').read_text())['overlay'].items():
        files[OLD/'pkg'/name.removeprefix('python/')] = digest
    for line in (GRAPH/'staged-launcher.sha256').read_text().splitlines():
        digest, path = line.split()
        files[OLD/path] = digest
    for path, digest in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Tested artifact changed: {path.name}')


def before_stop():
    state = original_state()
    if state['paid'] not in ('failed','inactive') or state['gpu_quota'] != ['max']*4:
        raise RuntimeError('Standalone paid fleet or original GPU quotas differ from tested input state')
    processes = run('nvidia-smi','--query-compute-apps=process_name','--format=csv,noheader').splitlines()
    if any(Path(x.strip()).name == 'bpm' for x in processes):
        raise RuntimeError('A direct miner still owns a GPU')
    if max(cards()) >= GPU_TEMP_CUTOFF_C or not state['container'].endswith(' running'):
        raise RuntimeError('Original state or temperatures unsuitable for cutover')
    return state


def broker_status():
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(2)
        sock.connect(str(SOCKET))
        sock.sendall(b'STATUS\n')
        with sock.makefile('r') as f:
            return json.loads(f.readline())


def ensure_idle():
    # Reuse the production load accounting and HTTP active/counter drain window.
    source = GRAPH.parent/'full-canary/host-run/idle.py'
    run(PYTHON, source, timeout=300)


def check_private(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('Gateway token must be an owner-only regular file')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', path.read_text().strip()):
        raise RuntimeError('Invalid private gateway token')


def start(token, minutes):
    if not 1 <= minutes <= 1440:
        raise ValueError('Choose a recovery window of 1–1440 minutes')
    token = token.absolute()
    check_private(token)
    check_artifacts()
    if STATE.exists() or any(active(n) for n in (CANDIDATE,BROKER,GUARD,RESTORE+'.timer',RESTORE+'.service')):
        raise RuntimeError('Expected fresh serving state and unused private units')
    if not active(GATEWAY):
        raise RuntimeError('Start the public-input gateway first')
    if not RUN.is_dir() or RUN.lstat().st_uid != os.getuid() or RUN.lstat().st_mode & 0o077:
        raise RuntimeError('Expected private run directory')
    with socket.socket() as check:
        check.bind(('127.0.0.1',18002))
    baseline = before_stop()
    ensure_idle()
    if baseline != before_stop():
        raise RuntimeError('Original state changed over idle period')
    STATE.mkdir(mode=0o700)
    (STATE/'shapes').mkdir(mode=0o700)
    write('before.json',baseline)
    original_cgroup = Path(namespace_state()['cgroup'])
    write('instance.json', {'cgroup':str(original_cgroup),'before_inode':original_cgroup.stat().st_ino,
                            'before_smokes':[(ORIGINAL/name).stat().st_mtime_ns for name in SMOKES]})
    with (STATE/'cutover.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            context = ['--setenv=PEARL_NATIVE_IDLE_CONFIG='+NATIVE_CONFIG] if NATIVE_CONFIG else []
            run('systemd-run','--user','--collect',*context,'--unit='+RESTORE,'--on-active='+str(minutes)+'min',
                '--timer-property=AccuracySec=1s',PYTHON,HERE,'restore')
            if not active(RESTORE+'.timer'):
                raise RuntimeError('Recovery timer did not arm')
            native_args = ['--native-permit',SOCKET.with_name('idle.permit'),'--timeline',RUN/'broker.jsonl'] if NATIVE_CONFIG else []
            unit(BROKER,PYTHON,PROTO/'broker.py','--socket',SOCKET,'--inference-only-ranks','4',*native_args,
                 properties=('RuntimeDirectory='+BROKER,'RuntimeDirectoryMode=0700',
                             *(['OnFailure='+RESTORE+'.service'] if NATIVE_CONFIG else [])))
            for _ in range(40):
                if SOCKET.exists():
                    break
                time.sleep(.1)
            status = broker_status()
            if len(status['idle']) != 4 or status['drained'] != []:
                raise RuntimeError('Expected four scheduler ranks and zero physical miners')
            # Check again after creating the timer and socket. Any new real request aborts cutover.
            sys.path.insert(0,str(REPO/'scripts'))
            import dsv41_load as load
            rows=load.parse_loads(json.loads(load.fetch(load.LOAD_PATH)))
            busy,_=load.activity_metrics(load.fetch('/metrics'))
            if busy or not load.is_idle(rows):
                raise RuntimeError('Traffic arrived after idle qualification')
            write('cutover.json',{'requested_ns':time.time_ns()})
            unit(GUARD,PYTHON,HERE,'guard',properties=('OnFailure='+RESTORE+'.service',))
            run(*DOCKER,'stop','dsv41',timeout=120)
            if (STATE/'guard.json').exists() or not active(GUARD):
                raise RuntimeError('Independent thermal guard tripped before model startup')
            env = ('CANARY_PORT=18002','CANARY_REQUIRE_API_KEY=1','CANARY_STATE='+str(STATE),
                   'PEARL_SHAPE_DIR='+str(STATE/'shapes'),'PEARL_BROKER_SOCKET='+str(SOCKET),
                   'PEARL_PUBLIC_REPLAY=1','PEARL_WOB_MODULE=checkpoint_graph','PEARL_GRAPH_VERIFY=1',
                   'PEARL_GPU_PREPARE=1','PEARL_REPLAY_POOL_PORT=31990',
                   'PEARL_REPLAY_POOL_HOST=127.0.0.1','PEARL_REPLAY_GATEWAY_TOKEN_FILE='+str(token))
            unit(CANDIDATE,OLD/'env-run',HERE,'launch-oci',env=env+('RAYON_NUM_THREADS=2',) if NATIVE_CONFIG else env,
                 properties=('MemoryMax=365G','MemorySwapMax=0','TimeoutStopSec=70','OnFailure='+RESTORE+'.service'),
                 output=RUN/'logs/fullmodel.log')
            print('Private full-model startup initiated; fresh startup smokes and graph profile appear in status.',flush=True)
        except BaseException:
            if (STATE/'cutover.json').exists():
                run('systemctl','--user','start','--no-block',RESTORE+'.service')
            else:
                for name in (BROKER,RESTORE+'.timer'):
                    if active(name):
                        run('systemctl','--user','stop',name)
            raise


def launch_oci():
    # The tested environment wrapper establishes all OCI sys.path, CUDA, and dynamic-linker
    # settings. A new interpreter applies this branch's authenticated pool client first.
    paths = [str(PROTO)] + [p for p in os.environ['PYTHONPATH'].split(os.pathsep) if p != str(PROTO)]
    if NATIVE_CONFIG:
        paths.insert(0,str(RUN/'pkg'))
    env = {**os.environ,'PYTHONPATH':os.pathsep.join(paths),'SGLANG_LISTEN_ADDRESS':'127.0.0.1'}
    env.pop('API_KEY',None)
    os.execve(sys.executable,[sys.executable,str(OLD/'boot/boot.py'),'run'],env)


def guard():
    while True:
        try:
            temps = cards()
            peak = max(temps)
            write('temperature.json', {'timestamp_ns':time.time_ns(), 'temperatures_c':temps, 'max_c':peak})
            if peak >= GPU_TEMP_CUTOFF_C:
                raise RuntimeError(f'GPU temperature reached the {GPU_TEMP_CUTOFF_C} C cutoff')
        except Exception as error:
            try:
                write('guard.json',dict(timestamp_ns=time.time_ns(),reason=str(error),
                                        temperatures_c=locals().get('temps',[])))
            finally:
                run('systemctl','--user','start','--no-block',RESTORE+'.service')
            return
        time.sleep(1)


def verify():
    baseline = json.loads((STATE/'before.json').read_text())
    restart = json.loads((STATE/'restart.json').read_text())
    since = restart['time_ns']
    for _ in range(550):
        files = [ORIGINAL/name for name in SMOKES]
        if all(p.exists() and p.lstat().st_uid==0 and stat.S_ISREG(p.lstat().st_mode) and p.lstat().st_mtime_ns>=since for p in files):
            now = original_state()
            if now != baseline:
                raise RuntimeError('Restored original identity, profile, quota or paid fleet state differs from baseline')
            if len(cards()) != 4:
                raise RuntimeError('Missing GPU')
            reply = json.loads(api(8000,ORIGINAL/'api-key','/v1/chat/completions',
                dict(model='deepseek-v4-flash',messages=[dict(role='user',content='What is 19 + 23? Reply only with the number.')],
                     chat_template_kwargs={'thinking':False},temperature=0,max_tokens=96)))
            if reply['choices'][0]['message']['content'].strip().rstrip('.') != '42':
                raise RuntimeError('Restored public arithmetic check failed')
            write('restored.json',{'smokes':list(SMOKES),'fresh_original_start':restart['fresh'],'arithmetic_42':True,'state':now,'verified_ns':time.time_ns()})
            return
        time.sleep(2)
    raise RuntimeError('Original startup smokes did not finish before verification deadline')


def restore():
    if not (STATE/'cutover.json').is_file():
        raise RuntimeError('No private full-model cutover to restore')
    with (STATE/'cutover.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (STATE/'restored.json').exists():
            return
        for name in (SAMPLER,CANDIDATE,GATEWAY,BROKER,GUARD):
            if active(name):
                run('systemctl','--user','stop',name,timeout=130)
        if not (STATE/'restart.json').exists():
            instance = json.loads((STATE/'instance.json').read_text())
            cgroup = Path(instance['cgroup'])
            current = run(*DOCKER,'inspect','-f','{{.Id}} {{.State.Status}}','dsv41')
            # A rejected stop may leave the original process untouched. In that case a
            # new boot receipt will never be written; verify the same instance instead.
            unchanged = (current.endswith(' running') and cgroup.is_dir()
                         and cgroup.stat().st_ino == instance['before_inode'])
            since = min(instance['before_smokes']) if unchanged else json.loads((STATE/'cutover.json').read_text())['requested_ns']
            write('restart.json', {'time_ns':since, 'fresh':not unchanged})
        run(*DOCKER,'start','dsv41',timeout=120)
        verify()
        if active(RESTORE+'.timer'):
            run('systemctl','--user','stop',RESTORE+'.timer')
        print('Original DeepSeek restored and verified',flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest='command',required=True)
    create=sub.add_parser('start')
    create.add_argument('--token-file',type=Path,required=True)
    create.add_argument('--window-minutes',type=int,default=30)
    for cmd in ('status','restore','verify','guard','launch-oci','renew'):
        sub.add_parser(cmd)
    args=cli.parse_args()
    if args.command=='start':
        start(args.token_file,args.window_minutes)
    elif args.command=='launch-oci':
        launch_oci()
    elif args.command=='guard':
        guard()
    elif args.command=='restore':
        restore()
    elif args.command=='verify':
        verify()
    elif args.command=='renew':
        if not active(CANDIDATE) or not active(GUARD) or not active(RESTORE+'.timer'):
            raise RuntimeError('Can renew only while candidate, thermal guard and recovery timer are active')
        run('systemctl','--user','restart',RESTORE+'.timer')
    else:
        data={'candidate':active(CANDIDATE),'gateway':active(GATEWAY),'guard':active(GUARD),
              'failback_timer':active(RESTORE+'.timer'),'smokes':all((STATE/name).exists() for name in SMOKES),
              'restored':(STATE/'restored.json').exists(),'guard_tripped':(STATE/'guard.json').exists()}
        if data['smokes'] and data['candidate']:
            info=json.loads(api(18002,STATE/'api-key','/server_info'))
            data['profile']={k:info.get(k) for k in ('tp_size','ep_size','speculative_algorithm','context_length','cuda_graph_config')}
            data['broker']=broker_status()
        print(json.dumps(data),flush=True)


if __name__ == '__main__':
    main()
