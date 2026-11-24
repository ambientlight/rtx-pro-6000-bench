#!/usr/bin/env python3
"""Serve the owner-authorized interactive Pearl trial on the incumbent's loopback address."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time
import urllib.request

import fullmodel as m

HERE = Path(__file__).resolve()
DIR = HERE.parent
PUBLIC = 'dsv41-f2pool-entry'
METRICS = 'dsv41-f2pool-hour-metrics'
END = 'dsv41-f2pool-hour-end'
INTERNAL_PORT = 18002
HOUR = 3600
SCOPE = 'operator-authorized real inference'
JOURNAL = m.HOST/'candidate-unpacked/rootfs/opt/content-free/journal_entrypoint.py'


def read_json(path):
    return json.loads(path.read_text())


def public_get(port, path):
    # Match the original trusted loopback backend's authentication behavior.
    with urllib.request.urlopen(f'http://127.0.0.1:{port}'+path, timeout=5) as response:
        return response.read()


def wait_for(predicate, seconds, label):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if (m.STATE/'guard.json').exists() or (m.STATE/'restart.json').exists():
            raise RuntimeError('Independent rollback initiated during '+label)
        if predicate():
            return
        time.sleep(.25)
    raise TimeoutError(label+' timed out')


def gateway():
    return read_json(m.RUN/'receipts/gateway.json')


def recovery_timer(unit, seconds, *args):
    m.run('systemd-run','--user','--collect','--unit='+unit,'--on-active='+str(seconds)+'s',
          '--timer-property=AccuracySec=100ms',*args)
    if not m.active(unit+'.timer'):
        raise RuntimeError('Missing recovery timer '+unit)


def request_restore():
    m.run('systemctl','--user','start','--no-block',m.RESTORE+'.service')


def verify_upstream():
    state = gateway()
    if not state.get('job') or not state.get('connected') or state.get('armed'):
        raise RuntimeError('Expected a connected disarmed real inference gateway')
    if not m.active(m.GATEWAY) or (m.RUN/'armed').exists():
        raise RuntimeError('Unexpected gateway service/arm state')
    return state['run_id']


def start():
    token = m.RUN/'gateway-token'
    m.check_private(token)
    m.check_artifacts()
    if not JOURNAL.is_file() or not (DIR/'paid_idle.py').is_file():
        raise RuntimeError('Install the pinned content-free supervisor and paid idle worker harness')
    if not m.RUN.is_dir() or m.RUN.stat().st_uid != os.getuid() or m.RUN.stat().st_mode & 0o077:
        raise RuntimeError('Expected owner-only trial state')
    units = (m.CANDIDATE, m.GUARD, m.BROKER, m.RESTORE+'.timer',m.RESTORE+'.service',END+'.timer',PUBLIC,METRICS)
    if m.STATE.exists() or any(m.active(name) for name in units):
        raise RuntimeError('The previous trial state or units remain active')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', INTERNAL_PORT))
    run_id = verify_upstream()
    baseline = m.before_stop()
    models = json.loads(public_get(8000,'/v1/models'))
    if [x['id'] for x in models['data']] != ['deepseek-v4-flash']:
        raise RuntimeError('Incumbent public model list differs from the requested backend')
    m.ensure_idle()
    if m.before_stop() != baseline or verify_upstream() != run_id:
        raise RuntimeError('Original deployment or gateway changed during drain')
    m.STATE.mkdir(mode=0o700)
    (m.STATE/'shapes').mkdir(mode=0o700)
    (m.STATE/'api-key').symlink_to(m.ORIGINAL/'api-key')
    m.write('before.json',baseline)
    cg = Path(m.namespace_state()['cgroup'])
    m.write('instance.json', dict(cgroup=str(cg),before_inode=cg.stat().st_ino,
            before_smokes=[(m.ORIGINAL/name).stat().st_mtime_ns for name in m.SMOKES]))
    m.write('trial.json', dict(gateway_run_id=run_id, scope=SCOPE, duration_s=HOUR, port=8000,
            internal_port=INTERNAL_PORT, incumbent_model='deepseek-v4-flash', backend_requires_auth=False))
    try:
        # This fallback also recovers if the process preparing the hour dies.
        recovery_timer(m.RESTORE, 80*60, m.PYTHON,HERE,'restore')
        with (m.STATE/'cutover.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            sys.path.insert(0,str(m.REPO/'scripts'))
            import dsv41_load as load
            rows=load.parse_loads(json.loads(load.fetch(load.LOAD_PATH)))
            busy,_=load.activity_metrics(load.fetch('/metrics'))
            if busy or not load.is_idle(rows):
                raise RuntimeError('New inference arrived after the initial drain')
            m.write('cutover.json', {'requested_ns':time.time_ns()})
            failure = 'OnFailure='+m.RESTORE+'.service'
            m.unit(m.GUARD,m.PYTHON,m.HERE,'guard',properties=(failure,))
            m.run(*m.DOCKER,'stop','dsv41', timeout=120)
            m.unit(m.BROKER,m.PYTHON,DIR/'paid_idle.py','serve','--socket',m.SOCKET,
                   '--state',m.RUN/'idle',properties=('RuntimeDirectory='+m.BROKER,'RuntimeDirectoryMode=0700',failure),
                   output=m.RUN/'logs/idle-startup.log')
        def fleet_ready():
            if not m.active(m.BROKER):
                raise RuntimeError('Idle fleet supervisor exited')
            path=m.RUN/'idle/status.json'
            return path.exists() and read_json(path).get('ready')
        wait_for(fleet_ready, 180, 'four-worker F2Pool startup')
        status = m.broker_status()
        if status['drained'] != [True]*4 or status['watched'] != [True]*4 or status['idle'] != [False]*4:
            raise RuntimeError('Expected all four paid workers held and fenced before capture')
        with (m.STATE/'cutover.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not m.active(m.GUARD) or (m.STATE/'guard.json').exists() or (m.STATE/'restart.json').exists():
                raise RuntimeError('Recovery initiated during worker warmup')
            env = ('CANARY_PORT='+str(INTERNAL_PORT),'CANARY_REQUIRE_API_KEY=0','CANARY_CONTENT_FREE=1',
                   'CANARY_STATE='+str(m.STATE),'PEARL_SHAPE_DIR='+str(m.STATE/'shapes'),
                   'PEARL_BROKER_SOCKET='+str(m.SOCKET),'PEARL_PUBLIC_REPLAY=1',
                   'PEARL_WOB_MODULE=checkpoint_graph','PEARL_GRAPH_VERIFY=1','PEARL_GPU_PREPARE=1',
                   'PEARL_REPLAY_POOL_PORT=31990','PEARL_REPLAY_POOL_HOST=127.0.0.1',
                   'PEARL_REPLAY_GATEWAY_TOKEN_FILE='+str(token))
            m.unit(m.CANDIDATE,m.PYTHON,'-S',JOURNAL,'--',m.OLD/'env-run',m.HERE,'launch-oci',
                   properties=('MemoryMax=365G','MemorySwapMax=0','TimeoutStopSec=70',failure),
                   env=env,output=m.RUN/'logs/hour-safe-model.log')
    except BaseException:
        if (m.STATE/'cutover.json').exists():
            request_restore()
        elif m.active(m.RESTORE+'.timer'):
            m.run('systemctl','--user','stop',m.RESTORE+'.timer')
        raise
    print('Four paid workers held. Interactive joined DeepSeek is loading on its private port.',flush=True)


def prepare():
    if not m.active(m.CANDIDATE):
        raise RuntimeError('Joined model is down')
    if not all((m.STATE/name).is_file() for name in m.SMOKES):
        return False
    state=gateway()
    if state['run_id'] != read_json(m.STATE/'trial.json')['gateway_run_id']:
        raise RuntimeError('Joined attribution gateway restarted')
    return (state['connected'] and not state['armed'] and state['authorized_clients']==4
            and all(read_json(m.STATE/name)=={'status':'passed'} for name in m.SMOKES))


def responses_smoke():
    payload = dict(model="deepseek-v4-flash", stream=True, store=False, temperature=0,
                   input=[dict(role="user",content="What is 19 + 23? Reply only with the number.")],
                   max_output_tokens=96, chat_template_kwargs={"thinking":False})
    req = urllib.request.Request(f"http://127.0.0.1:{INTERNAL_PORT}/v1/responses",
            data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    text = []
    usage = {}
    with urllib.request.urlopen(req,timeout=60) as stream:
        for line in stream:
            if not line.startswith(b"data:") or line[5:].strip()==b"[DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("type") == "response.output_text.delta":
                text.append(event["delta"])
            if event.get("type") == "response.completed":
                usage = event["response"]["usage"]
    if "".join(text).strip().rstrip(".") != "42" or not usage.get("output_tokens"):
        raise RuntimeError("Direct streaming Responses readiness check failed")
    m.write("responses-smoke.json",dict(status="passed",stream=True,store=False,
            input_tokens=usage["input_tokens"],output_tokens=usage["output_tokens"]))


def open_hour():
    if (m.STATE/'window.json').exists():
        raise RuntimeError('The measurement window is already recorded')
    try:
        wait_for(prepare, 950, 'four startup smoke checks and authenticated joined ranks')
        info=json.loads(public_get(INTERNAL_PORT,'/server_info'))
        if (info['tp_size'],info['ep_size'],info['speculative_algorithm'],info['served_model_name'],
                info['cuda_graph_config']['decode']['backend'],info['cuda_graph_config']['prefill']['backend']) != (
                4,4,'DSPARK','deepseek-v4-flash','full','disabled'):
            raise RuntimeError('Unexpected effective serving or capture profile')
        responses_smoke()
        if not m.active(METRICS):
            m.unit(METRICS, m.PYTHON,DIR/'hour_metrics.py','collect','--run-dir',m.RUN,
                    '--port',str(INTERNAL_PORT),'--key-file',m.STATE/'api-key','--shape-dir',m.STATE/'shapes',
                    '--gateway-receipt',m.RUN/'receipts/gateway.json','--fleet-dir',m.RUN/'idle',
                    '--window-file',m.STATE/'window.json',output=m.RUN/'logs/hour-metrics.log')
        def metrics_ready():
            if not m.active(METRICS):
                raise RuntimeError('Measurement collector stopped before taking its baseline')
            return (m.RUN/'metrics/ready.json').is_file()
        wait_for(metrics_ready, 90, 'passive collector baseline')
        with (m.STATE/'cutover.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not prepare() or not m.active(m.GUARD) or (m.STATE/'guard.json').exists() or (m.STATE/'restart.json').exists():
                raise RuntimeError('Startup prerequisites changed before ingress')
            old_job=gateway()['job']['id']
            (m.RUN/'armed').touch(mode=0o600,exist_ok=False)
            def rotated():
                state=gateway()
                return state['armed'] and state['connected'] and state['job'] and state['job']['id'] != old_job
            wait_for(rotated, 10, 'arm transition')
            # Preserve the exact existing loopback bind. Existing LiteLLM/VM ingress remains upstream.
            m.unit(PUBLIC,m.PYTHON,DIR/'hour_ingress.py',
                   properties=('OnFailure='+m.RESTORE+'.service',))
            def public_ready():
                try:
                    return json.loads(public_get(8000,'/v1/models'))['data'][0]['id'] == 'deepseek-v4-flash'
                except OSError:
                    return False
            wait_for(public_ready, 10, 'incumbent address availability')
            recovery_timer(END,HOUR,m.PYTHON,HERE,'restore')
            start_ns=time.monotonic_ns()
            epoch=time.time()
            m.write('window.json',dict(start_utc_epoch=epoch,end_utc_epoch=epoch+HOUR,
                    start_monotonic_ns=start_ns,end_monotonic_ns=start_ns+HOUR*10**9,
                    scope=SCOPE,duration_s=HOUR, active_workers=4,active_joined_ranks=4))
    except BaseException:
        if (m.STATE/'cutover.json').exists():
            request_restore()
        raise
    print((m.STATE/'window.json').read_text().strip(),flush=True)


def close_ingress():
    if not m.active(PUBLIC):
        return
    m.run('systemctl','--user','kill','--kill-whom=main','--signal=SIGUSR1',PUBLIC)
    # The shared connection can still return model tokens after new requests are frozen.
    sys.path.insert(0,str(m.REPO/'scripts'))
    import dsv41_load as load
    deadline=time.monotonic()+90
    began=time.monotonic()
    retired=False
    while time.monotonic()<deadline and not (m.STATE/'guard.json').exists() and m.active(m.CANDIDATE):
        try:
            busy,_=load.activity_metrics(public_get(INTERNAL_PORT,'/metrics').decode())
            rows=load.parse_loads(json.loads(public_get(INTERNAL_PORT,load.LOAD_PATH)))
            if not busy and load.is_idle(rows):
                retired=True
                time.sleep(.2)
                break
        except (OSError, ValueError):
            break
        time.sleep(.25)
    m.write('drain.json',{'inflight_drained':retired,'elapsed_seconds':time.monotonic()-began})


def restore():
    if not (m.STATE/'cutover.json').exists():
        raise RuntimeError('No trial cutover has occurred')
    (m.RUN/'armed').unlink(missing_ok=True)
    window_path=m.STATE/'window.json'
    boundary=window_path.exists() and time.monotonic_ns()>=read_json(window_path)['end_monotonic_ns']-2_000_000_000
    if m.active(METRICS):
        stop=m.RUN/'metrics/stop'
        stop.touch(mode=0o600,exist_ok=True)
    holding=None
    try:
        if boundary and not (m.STATE/'guard.json').exists():
            # Hold paid workers across the closing edge while already-running inference returns.
            try:
                holding=socket.socket(socket.AF_UNIX)
                holding.settimeout(2)
                holding.connect(str(m.SOCKET))
                holding.sendall(b'LEASE trial-hour-deadline\n')
                holding.recv(128)
            except OSError:
                pass
            close_ingress()
        if m.active(PUBLIC):
            m.run('systemctl','--user','stop',PUBLIC,timeout=20)
        if m.active(METRICS) and not (m.STATE/'guard.json').exists():
            until=time.monotonic()+3
            while not (m.RUN/'metrics/finished.json').exists() and time.monotonic()<until:
                time.sleep(.05)
    finally:
        try:
            m.restore()
        finally:
            if holding is not None:
                holding.close()
    if m.active(METRICS):
        m.run('systemctl','--user','stop',METRICS,timeout=15)
    if m.active(END+'.timer'):
        m.run('systemctl','--user','stop',END+'.timer')
    print('Interactive trial ended',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('start','open','restore','status'))
    cmd=parser.parse_args().command
    if cmd=='start':
        start()
    elif cmd=='open':
        open_hour()
    elif cmd=='restore':
        restore()
    else:
        print(json.dumps(dict(ingress=m.active(PUBLIC),candidate=m.active(m.CANDIDATE),
                fleet=m.active(m.BROKER),metrics=m.active(METRICS),guard=m.active(m.GUARD),
                window=read_json(m.STATE/'window.json') if (m.STATE/'window.json').exists() else None,
                timer=m.active(END+'.timer'),restored=(m.STATE/'restored.json').exists())))


if __name__=='__main__':
    main()
