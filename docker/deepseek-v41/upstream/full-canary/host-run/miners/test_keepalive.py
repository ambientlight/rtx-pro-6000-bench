import argparse,json,sys,tempfile,threading,time
from pathlib import Path
from types import SimpleNamespace
bench=Path('/mnt/hot/ambientlight/repos/rtx-pro-6000-bench')
sys.path[:0]=[str(bench/'docker/deepseek-v41/upstream/full-canary'), str(bench/'work/pearl-inference-20261123/prototype')]
import public_pool as peer
from replay_pool import ReplayPool

def until(fn,timeout=5):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if got:=fn():return got
        time.sleep(.01)
    raise TimeoutError(fn)

def main(wait):
    with tempfile.TemporaryDirectory() as td:
        dst=Path(td)/'peer.json'
        args=SimpleNamespace(model=Path('/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017'),verifier=Path('/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-v1/target/v1-build/release/examples/verify_receipt'),port=0,share_target=None,sustained=True,resume_receipt=False,receipt=dst)
        if wait<120: peer.KEEPALIVE_SECONDS=.06
        with peer.Peer(args) as p:
            worker=threading.Thread(target=p.serve_forever,daemon=True);worker.start()
            clients=[ReplayPool(*p.server_address) for _ in range(4)]
            try:
                first=[until(c.snapshot) for c in clients]
                time.sleep(wait)
                assert all(c.snapshot() is job for c,job in zip(clients,first,strict=True))
                status=json.loads(dst.read_text())
                assert status['negotiations']==4 and status['keepalive_frames']>=4 and status['rejected_frames']==0
                for c,job in zip(clients,first,strict=True):
                    assert c.submit(job,'decode',b'invalid local fixture')
                until(lambda:sum(c.counters()['decode']['reject'] for c in clients)==4)
            finally:
                for c in clients: c.close()
                p.shutdown();worker.join()
            status=json.loads(dst.read_text())
        args.resume_receipt=True
        with peer.Peer(args) as p:
            resumed=json.loads(dst.read_text())
            assert resumed['rejected_frames']==4 and resumed['accepted_total']==0
            assert resumed['job']==status['job'] and resumed['negotiations']==status['negotiations']
            assert len(resumed['peer_restarts'])==1
        print(json.dumps(dict(seconds_idle=wait,keepalive_seconds=peer.KEEPALIVE_SECONDS,keepalive_frames=status['keepalive_frames'],quiet_rank_connections=4,job_identity_preserved=True,rejections=4,prior_counts_preserved=True, generation=[j.generation for j in first])))
if __name__=='__main__':
    main(float(sys.argv[1]))
