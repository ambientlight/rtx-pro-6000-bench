from pathlib import Path
import copy, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"deployment-source/docker/deepseek-v41/upstream/full-canary"))
import graph_compare as g
ranks = [str(i) for i in range(4)]
def snapshot(num):
    return {"shapes": {r:[dict(mode="TARGET_VERIFY",real_rows=6,executed_rows=6,calls=num,path=p)
        for p in ("target_graph","target_graph_mined")] for r in ranks},
        "counters": {r: {"0": {"verify": {"verified_steps":num,"tickets_scanned":num*960}, "eager":{"prefill":{
            "mined_calls":num,"mined_rows":2048*num}}, "eager_verify":{"prefill":{"verified_steps":num,"tickets_scanned":81_920*num}}}}
            for r in ranks}}
before, after = snapshot(10), snapshot(14)
runs=[{"request":{"input_tokens":2048},"graph_delta_by_rank":{r:{"decode_cuda_graph":1} for r in ranks}} for _ in range(4)]
expected = {"decode":4*4*6*5120*2048,"prefill":4*4*2048*5120*2048}
assert g.graph_work(before,after,runs,{r:{'decode_cuda_graph':4} for r in ranks})["independently_scanned_projection_mac_equivalent"] == expected
bad=copy.deepcopy(after)
bad["counters"]["2"]["0"]["verify"]["tickets_scanned"]-=1
assert g.graph_work(before,bad,runs,{r:{'decode_cuda_graph':4} for r in ranks})["independently_scanned_projection_mac_equivalent"] == dict(expected,decode=None)
bad["counters"]["2"]["0"]["verify"]["verified_steps"]-=1
assert g.pending_scans(bad)
print("Verified rank deltas, compact/wide MAC accounting, stale/mismatched classification")
