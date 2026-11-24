"""Exercise content-free proof accounting in isolation; no GPU or serving process."""
import json,os,sys,tempfile,types
from pathlib import Path
sys.path.insert(0,os.environ["GCOMP"])
for module,names in [("check_http_projection",["matrix_opening"]),("checkpoint_projection",["checkpoint_weight","quantize_rows"]),("projection_wire",["SHARE_TARGET","receive","send","verify_submission"])]:
    stub=types.ModuleType(module)
    for name in names:setattr(stub,name,None)
    sys.modules[module]=stub
from public_pool import Peer
with tempfile.TemporaryDirectory() as folder:
    d=Path(folder)
    p=Peer.__new__(Peer)
    p.receipt=d/"ledger.json";p.accepted_log=d/"accepted.jsonl";p.sustained=True
    p.results={"started_at":1,"accepted_total":0,"accepted":[],"verified_by_rank_rows":{str(rank):{} for rank in range(4)}}
    for i in range(22):
        p.record({"job_id":"test","proof_sha256":f"{i:064x}","tp_rank":i%4,"proof_rows":6,"received_monotonic_ns":100+i,"ack_monotonic_ns":600+i,"synthetic_extra":"dummy"})
    lines=[json.loads(l) for l in p.accepted_log.read_text().splitlines()]
    assert len(p.results["accepted"])==16 and json.loads(p.receipt.read_text())["accepted_total"]==22
    assert len(lines)==22 and all(0<row["received_monotonic_ns"]<row["ack_monotonic_ns"] for row in lines)
    assert not any("synthetic_extra" in row for row in lines)
print("22 ACK events audited with a 16-sample cumulative receipt and no payload retention")
