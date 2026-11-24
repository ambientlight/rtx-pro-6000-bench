import json,sys,tempfile,threading,socket
from types import SimpleNamespace
from pathlib import Path
bench=Path("/mnt/hot/ambientlight/repos/rtx-pro-6000-bench")
sys.path[:0]=[str(bench/"docker/deepseek-v41/upstream/full-canary"),str(bench/"work/pearl-inference-20261123/prototype")]
import public_pool as peer
from projection_wire import send,receive
kw=dict(model=Path("/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017"),verifier=Path("/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-v1/target/v1-build/release/examples/verify_receipt"),port=0,share_target=None)
with tempfile.TemporaryDirectory() as td:
 result={}
 for sustained in (False,True):
  receipt=Path(td)/f"peer-{sustained}.json"
  with peer.Peer(SimpleNamespace(**kw,sustained=sustained,receipt=receipt)) as p:
   t=threading.Thread(target=p.serve_forever,daemon=True);t.start()
   try:
    with socket.create_connection(p.server_address) as sock,sock.makefile("rb") as stream:
     send(sock,dict(id=1,method="mining.authorize",params=dict(type="v2")))
     assert receive(stream)["result"] and receive(stream)["params"]["target"] == f"{(peer.SHARE_TARGET>>16 if sustained else peer.SHARE_TARGET):064x}"
     send(sock,dict(id=2,method="mining.submit",params=dict(job_id=p.job["job_id"],plain_proof="bogus")))
     assert receive(stream)["result"] is False
    p.shutdown(); t.join()
    if sustained:
     # Isolate bounded accounting independently of the wire-verification test.
     for i in range(520):
      p.record(dict(tp_rank=i%4,proof_rows=16 if i%2 else 6,proof_sha256="test-only"))
    q=json.loads(receipt.read_text())
    assert q["negotiations"]==q["rejected_frames"]==1
    if sustained:
     assert len(q["accepted"])==16 and q["accepted_total"]==520 and sum(sum(c.values()) for c in q["verified_by_rank_rows"].values())==520 and not p.at_limit()
    else:
     assert q["share_limit"]==512 and "accepted_total" not in q
    result['sustained' if sustained else 'default']=dict(authorization=q["negotiations"],invalid_submission_rejected=q["rejected_frames"],counter_fixture_total=q.get("accepted_total"),records=len(q["accepted"]),target=q["job"]["target"])
   finally:p.shutdown();t.join()
 print(json.dumps(result))
