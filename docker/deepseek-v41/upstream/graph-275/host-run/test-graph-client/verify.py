import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,os.environ["GCOMP"])
import graph_compare as g

class MockService:
    def __init__(self):
        self.n=0
        self.eager=0
        self.extra=0
        self.active_polls=0
        self.out=0

    @contextlib.contextmanager
    def request(self,base,path,body=None,timeout=None):
        if path=="/server_info":
            blob=json.dumps(dict(tp_size=4,ep_size=4,served_model_name="public",speculative_algorithm="DSPARK",disable_cuda_graph=False,
                                 cuda_graph_config={"decode":{"backend":"full"}})).encode()
        elif path=="/metrics":
            as_text=f'sglang:http_requests_active{{endpoint="/generate",method="POST"}} {int(self.active_polls>0)}\n'
            self.active_polls=max(0,self.active_polls-1)
            as_text+=f'sglang:http_requests_total{{endpoint="/generate",method="POST"}} {self.n}\n'
            for rank in range(4):
                as_text+=f'sglang:cuda_graph_passes_total{{tp_rank="{rank}",mode="decode_cuda_graph"}} {self.n*3}\n'
                as_text+=f'sglang:cuda_graph_passes_total{{tp_rank="{rank}",mode="decode_none"}} {self.n*self.eager}\n'
            blob=as_text.encode()
        elif path=="/generate":
            self.n+=1+self.extra
            self.active_polls=1
            n=body["sampling_params"]["max_new_tokens"]
            events=[dict(output_ids=[self.out]*k,meta_info=dict(completion_tokens=k,prompt_tokens=len(body["input_ids"]),cached_tokens=0,
                    decode_throughput=100,spec_verify_ct=3,spec_accept_length=1,num_retractions=0)) for k in (1,n)]
            blob=b''.join(b'data: '+json.dumps(row).encode()+b'\n\n' for row in events)+b'data: [DONE]\n\n'
        else:
            raise RuntimeError(path)
        yield io.BytesIO(blob)

def test():
    with tempfile.TemporaryDirectory() as tmp:
        q=Path(tmp)
        s=MockService()
        gpu=[{"index":str(i),"power.limit":"275.00","clocks.current.memory":"13365","uuid":f"u{i}"} for i in range(4)]
        args=SimpleNamespace(base="http://fake",arm="switch",mode="fixed-period",inputs=256,outputs=8,pairs=3,gap=.01,period=.04,
                             peer_ledger=None,accepted_log=None,shape_dir=None,receipt=q/"switch.json")
        with patch.object(g,"request",s.request),patch.object(g,"gpu_snapshot",return_value=gpu),patch.object(g,"exact_prompt",side_effect=lambda u,m,n,id: [ord(id[-1])]*n):
            control=g.run_trace(args)
            args.arm="joined";s.out=1;args.receipt=q/"joined.json"
            candidate=g.run_trace(args)
            v=g.compare_traces(q/"switch.json",q/"joined.json")
            assert v["control"]["horizon_seconds"]==v["candidate"]["horizon_seconds"]
            assert v["exact_outputs"]==0 and v["control"]["replay_passes_by_rank"]=={str(i):9 for i in range(4)}
            s.eager=1
            try:g.run_trial("u","bad",1,inputs=256,outputs=8,receipt=q/"eager.json")
            except RuntimeError:pass
            else:raise AssertionError("Missed graph fallback")
            assert json.loads((q/"eager.json").read_text())["qualification_issues"]
            s.eager=0;s.extra=1
            try:g.run_trial("u","bad",1,inputs=256,outputs=8)
            except RuntimeError:pass
            else:raise AssertionError("Missed concurrent user")
            s.extra=0;args.mode="fixed-gap";args.receipt=q/"g.json"
            gap=g.run_trace(args)
            assert (gap["requested_horizon_end_monotonic_ns"]-gap["runs"][-1]["request"]["stream_closed_monotonic_ns"])==10_000_000
        events=[(1,"a"),(101,"a"),(111,"b"),(301,"c")]
        p=q/"acks.jsonl"
        p.write_text("".join(json.dumps(dict(peer_started_at=1,job_id="j",received_monotonic_ns=t,proof_rows=6,tp_rank=0,proof_sha256=h))+"\n" for t,h in events))
        result=g.local_proofs(p,100,200,{"started_at":1,"job":{"job_id":"j"}})
        assert result["verified_by_real_rows"]=={"6":{"accepted":2,"unique":1,"by_rank":{"0":1}}}
        return dict(scope="CPU-only simulated response and metric fixtures",matched=True,positive_all_rank_replays=True,zero_cache=True,
                    fallbacks_rejected=True,extra_http_traffic_rejected=True,duplicate_and_old_proofs_excluded=True,
                    fixed_horizon_equal=True,post_completion_gap_verified=True)
print(json.dumps(test(),indent=2))
