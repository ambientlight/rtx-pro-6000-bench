"""Run four request-preemptible K780 workers on the existing Pearl F2Pool account."""
import argparse
import asyncio
from collections import OrderedDict
from contextlib import AsyncExitStack, suppress
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from fullmodel import GPU_TEMP_CUTOFF_C
from gateway import HOST, PORT, PREFLIGHT, MAX_FRAME, digest, frame, message, parse_job, secret

HERE = Path(__file__).resolve()
PROTO = HERE.parents[5]/'work/pearl-inference-20261123/prototype'
BINARY = Path('/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-sglang-prototype/target/cargo/release/bpm')
BINARY_SHA = 'ce5c50ce53418fa1e1ffad5ea4bd0d486b2bfa83327686fd58b624e067ad9108'


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    temporary.replace(path)


def gpu_info():
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,power.limit,temperature.gpu',
                                    '--format=csv,noheader,nounits'], text=True, timeout=4)
    cards = [line.split(', ') for line in rows.splitlines()]
    if [int(c[0]) for c in cards] != list(range(4)) or any(float(c[3]) != 275 for c in cards):
        raise RuntimeError('Four cards at 275 W are required')
    return cards


def prewarm_cards():
    cards = gpu_info()
    if min(int(c[2]) for c in cards) < 850:
        raise RuntimeError('Insufficient free memory for prewarm')
    if max(int(c[4]) for c in cards) >= GPU_TEMP_CUTOFF_C:
        raise RuntimeError(f'GPU temperature reached the {GPU_TEMP_CUTOFF_C} C cutoff before prewarm')
    return cards


def owns_connection(pid, local_port, relay_port):
    """Authenticate the originating local process by its connected TCP socket inode."""
    local = f'0100007F:{local_port:04X}'
    remote = f'0100007F:{relay_port:04X}'
    with Path('/proc/net/tcp').open() as tcp:
        for line in tcp:
            cols = line.split()
            if cols[1:4] == [local, remote, '01']:
                inode = f'socket:[{cols[9]}]'
                break
        else:
            return False
    for fd in Path(f'/proc/{pid}/fd').iterdir():
        with suppress(FileNotFoundError):
            if os.readlink(fd) == inode:
                return True
    return False


class Relay:
    """Forward one bpm process's verified synthetic shares and count matching remote responses."""
    def __init__(self, rank, account, journal):
        self.rank, self.account, self.journal = rank, account, journal
        self.pid = None
        self.active = False
        self.tasks = set()
        # Archived runs and the idle-rate collector use these cooling counters.
        self.state = dict(worker=f'joinedidle{rank}', connected=False, authorized=False, connections=0,
                          notifications=0, submitted=0, accepted=0, rejected=0, unknown=0,
                          accepted_work_macs=0., target=None, work=None, cooling=False, cooling_seconds=0.)
    def event(self, kind, **data):
        print(json.dumps(dict(event=kind,rank=self.rank,monotonic_ns=time.monotonic_ns(),
                              utc_epoch=time.time(),**data),separators=(',',':')), file=self.journal,flush=True)

    async def accept(self, lr, lw):
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            if not self.pid or self.active or not owns_connection(self.pid, lw.get_extra_info('peername')[1],
                                                                 lw.get_extra_info('sockname')[1]):
                return
            self.active = True
            try:
                await self.session(lr,lw)
            except (OSError, ValueError, TimeoutError, EOFError, KeyError, TypeError) as e:
                self.event('connection_end', issue=type(e).__name__)
            finally:
                self.active = False
                self.state.update(connected=False, authorized=False, target=None)
        finally:
            lw.close()
            self.tasks.discard(task)

    async def session(self, lr, lw):
        auth=await message(lr,6)
        worker=self.state['worker']
        if (auth.get('method')!='mining.authorize' or type(auth.get('id')) is not int
                or auth.get('params',{}).get('worker')!=worker):
            raise ValueError('Unexpected local authorization')
        ur,uw=await asyncio.wait_for(asyncio.open_connection(HOST,PORT,limit=MAX_FRAME),10)
        self.state['connections']+=1
        generation=self.state['connections']
        pending={}
        jobs=OrderedDict()
        auth_id=auth['id']
        tasks=[]
        async def send(w,obj):
            w.write(frame(obj))
            await asyncio.wait_for(w.drain(),5)
        async def upstream():
            while True:
                obj=await message(ur,120)
                method=obj.get('method')
                key=obj.get('id')
                if method=='mining.notify':
                    job=parse_job(obj['params'],generation)
                    alias=digest(job.id.encode()+job.header+job.target.to_bytes(32,'big'))
                    jobs[alias]=job
                    if len(jobs)>128:
                        jobs.popitem(last=False)
                    target=f'{job.target:064x}'
                    self.state.update(target=target,connected=self.state['authorized'])
                    self.state['notifications']+=1
                    await send(lw,dict(method=method,params=dict(job_id=alias,cert_version=3,
                               target=target,header=job.header.hex())))
                elif method=='client.reconnect':
                    raise ConnectionError('Pool requested reconnection')
                elif method is None and type(key) is int:
                    ok=obj.get('result') is True and obj.get('error') is None
                    if key==auth_id:
                        if self.state['authorized'] or not ok or obj.get('type')!='v2':
                            raise ValueError('F2Pool negotiation failed')
                        self.state.update(authorized=True,connected=bool(jobs))
                        self.event('authorized',generation=generation)
                        await send(lw,dict(id=key,result=True,error=None,type='v2'))
                    elif key in pending:
                        item=pending.pop(key)
                        kind='accepted' if ok else 'rejected'
                        self.state[kind]+=1
                        if ok:
                            self.state['accepted_work_macs']+=item['work_macs']
                        self.event(kind,**item)
                        await send(lw,dict(id=key,result=ok,error=None if ok else 'pool declined'))
        async def downstream():
            while True:
                obj=await message(lr,None)
                key=obj.get('id')
                params=obj['params']
                if (not self.state['connected'] or obj.get('method')!='mining.submit' or type(key) is not int
                        or key==auth_id or key in pending or len(pending)>=128):
                    raise ValueError('Unexpected local submission')
                job=jobs.get(params.get('job_id'))
                if job is None:
                    await send(lw,dict(id=key,result=False,error='expired job'))
                    continue
                proof=params['plain_proof']
                if not isinstance(proof,str):
                    raise ValueError('Invalid witness')
                item=dict(generation=generation,rpc_id=key,target=f'{job.target:064x}',
                          proof_sha256=digest(proof.encode()),work_macs=(1<<256)/(job.target+1))
                pending[key]=item
                self.state['submitted']+=1
                self.event('submitted',**item)
                await send(uw,dict(id=key,method='mining.submit',params=dict(job_id=job.id,plain_proof=proof)))
        try:
            await send(uw,dict(id=auth_id,method='mining.authorize',params={
                'wallet':self.account,'worker':worker,'password':'x','pass':'x','agent':'bpm/0.2.0','type':'v2'}))
            tasks=[asyncio.create_task(upstream()),asyncio.create_task(downstream())]
            done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                t.result()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            for item in pending.values():
                self.state['unknown']+=1
                self.event('unknown',**item)
            uw.close()


async def rpc(path,line='STATUS'):
    async with asyncio.timeout(6):
        rd,wr=await asyncio.open_unix_connection(path)
        try:
            wr.write((line+'\n').encode())
            await wr.drain()
            raw=(await rd.readline()).decode().strip()
            return json.loads(raw) if line=='STATUS' else raw
        finally:
            wr.close()
            await wr.wait_closed()


async def warm(broker,workers,relays):
    # Four temporary rank owners are permitted only during the exclusive service cutover.
    for port in (8000,18002):
        with socket.socket() as test:
            test.settimeout(.5)
            if test.connect_ex(('127.0.0.1',port))==0:
                raise RuntimeError('Warm the paid workers only after stopping the previous endpoint')
    async with asyncio.timeout(100):
        while True:
            with suppress(OSError):
                status=await rpc(broker)
                if all(status['watched']) and all(status['drained']):
                    break
            await asyncio.sleep(.05)
        channels=[]
        try:
            for rank in range(4):
                rd,wr=await asyncio.open_unix_connection(broker)
                channels.append((rd,wr))
                wr.write(f'IDLE {rank}\n'.encode())
                await wr.drain()
                if await rd.readline()!=b'OK\n':
                    raise ValueError('Boot rank rejected')
            while True:
                rows=await asyncio.gather(*(rpc(w) for w in workers))
                if all(r.state['connected'] for r in relays) and all(s['completed_slices']>=3 for s in rows):
                    break
                await asyncio.sleep(.1)
        finally:
            for _,wr in channels:
                wr.close()
            await asyncio.gather(*(wr.wait_closed() for _,wr in channels))
        while True:
            status=await rpc(broker)
            if all(status['drained']) and not any(status['idle']) and all(status['watched']):
                return
            await asyncio.sleep(.03)


def read_work(stream, relay):
    while True:
        start=stream.tell()
        line=stream.readline()
        if not line.endswith('\n'):
            stream.seek(start)
            return
        event=json.loads(line)
        if event['event'] in ('stats','complete'):
            relay.state['work']=dict(macs=round(event['seconds']*event['TMAC_wall_s']*1e12),
                                    monotonic_ns=time.monotonic_ns(),native_utc_epoch=event['time'])


async def serve(args):
    os.umask(0o077)
    if hashlib.sha256(BINARY.read_bytes()).hexdigest()!=BINARY_SHA:
        raise ValueError('Unexpected K780 miner binary')
    account=json.loads(secret(PREFLIGHT))['f2pool_mining_account']
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',account):
        raise ValueError('Malformed configured mining account')
    cards=await asyncio.to_thread(prewarm_cards)
    if subprocess.run(['systemctl','is-active','--quiet','pearl-f2pool']).returncode==0:
        raise RuntimeError('Existing paid supervisor is already running')
    ctl=args.socket.parent
    if ctl.stat().st_uid!=os.getuid() or ctl.stat().st_mode & 0o077:
        raise ValueError('Use a private RuntimeDirectory')
    args.state.mkdir(mode=0o700)
    workers=[ctl/f'gpu{i}.sock' for i in range(4)]
    children=[]
    async with AsyncExitStack() as resources:
        events=resources.enter_context((args.state/'events.jsonl').open('x',buffering=1))
        relays=[Relay(i,account,events) for i in range(4)]
        logs=[]
        try:
            for i,r in enumerate(relays):
                server=await asyncio.start_server(r.accept,'127.0.0.1',0,limit=MAX_FRAME)
                await resources.enter_async_context(server)
                rankdir=ctl/f'rank{i}'
                rankdir.mkdir(mode=0o700)
                log=rankdir/'events.jsonl'
                log.touch(mode=0o600)
                logs.append(resources.enter_context(log.open()))
                proc=await asyncio.create_subprocess_exec(str(BINARY),'--pool',
                    f'127.0.0.1:{server.sockets[0].getsockname()[1]}','--wallet','localrank',
                    '--worker',r.state['worker'],'--variant','780','--k','65536','--kernel-variant','v1',
                    '--slice-tiles','188','--power-limit','275','--max-temp',str(GPU_TEMP_CUTOFF_C),'--device','0',
                    '--control-socket',str(workers[i]),'--require-watch','--json','--log',str(log),
                    cwd=rankdir,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':cards[i][1],'RAYON_NUM_THREADS':'2'})
                r.pid=proc.pid
                children.append(proc)
            cli=[word for p in workers for word in ('--worker',str(p))]
            children.append(await asyncio.create_subprocess_exec(sys.executable,str(PROTO/'broker.py'),
                '--socket',str(args.socket),'--timeline',str(args.state/'broker-events.jsonl'),'--watch-workers',*cli))
            await warm(args.socket,workers,relays)
            print('Four authenticated K780 workers are resident, watched and drained',flush=True)
            while True:
                if any(p.returncode is not None for p in children):
                    raise RuntimeError(f'Idle fleet process exited: {[p.returncode for p in children]}')
                cards=await asyncio.to_thread(gpu_info)
                temps=[int(c[4]) for c in cards]
                broker,*stats=await asyncio.gather(*(rpc(p) for p in [args.socket,*workers]))
                if any(s['faulted'] for s in stats):
                    raise RuntimeError('Lost watched worker hold')
                for stream,r in zip(logs,relays):
                    read_work(stream,r)
                save(args.state/'status.json',dict(ready=True,utc_epoch=time.time(),monotonic_ns=time.monotonic_ns(),
                     broker=broker,workers=stats,temperatures_c=temps,ranks=[r.state for r in relays]))
                await asyncio.sleep(.5)
        finally:
            for p in reversed(children):
                if p.returncode is None:
                    with suppress(ProcessLookupError):
                        p.terminate()
            try:
                await asyncio.wait_for(asyncio.gather(*(p.wait() for p in children)),4)
            except TimeoutError:
                for p in children:
                    if p.returncode is None:
                        with suppress(ProcessLookupError):
                            p.kill()
                await asyncio.gather(*(p.wait() for p in children))
            tasks=[t for r in relays for t in r.tasks]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)


async def main(args):
    task=asyncio.create_task(serve(args))
    loop=asyncio.get_running_loop()
    for sig in (signal.SIGTERM,signal.SIGINT):
        loop.add_signal_handler(sig,task.cancel)
    with suppress(asyncio.CancelledError):
        await task


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('serve',))
    p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
