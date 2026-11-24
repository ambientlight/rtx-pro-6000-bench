"""Read back native graph counters around short public inference on the original service."""
import json, re, statistics, sys, threading, time
from pathlib import Path
from throughput import native_stream, exact_prompt
from qualify import http_counters, gpu_snapshot, request
base="http://127.0.0.1:8000"
def passes():
    with request(base,"/metrics") as res: text=res.read().decode()
    result={}
    for line in text.splitlines():
        if line.startswith("sglang:cuda_graph_passes_total{"):
            mode=re.search(r'\bmode="([^"]+)"',line)[1]
            rank=re.search(r'\btp_rank="([^"]+)"',line)[1]
            if mode.startswith('decode_'):
                result[rank,mode]=float(line.rsplit(' ',1)[1])
    return result
with request(base,"/server_info") as res: info=json.load(res)
before=http_counters(base)
assert before is not None and before['active']==0, "existing inference active"
gpus=gpu_snapshot()
assert len(gpus)==4 and all(float(g['power.limit'])==275 for g in gpus)
epoch=time.time_ns()
trials=[]
for i in range(4):
    name=f"prod-graph-275-control-reference-{i}"
    ids=exact_prompt(base,info["served_model_name"],2048,name)
    prior=passes()
    row=native_stream(base,ids,256,threading.Barrier(1),cache_salt=f"{name}-{epoch}")
    post=passes()
    delta={str(key):v-prior.get(key,0) for key,v in sorted(post.items()) if key[1].startswith('decode_')}
    assert all(delta.get(str((str(k),'decode_cuda_graph')),0)>0 for k in range(4))
    assert not any(v>0 for k,v in delta.items() if 'decode_cuda_graph' not in k)
    trials.append(dict(warmup=(i==0),graph_deltas=delta,**row))
    print(i,round(row['native_decode_tps'],1),delta,flush=True)
for _ in range(30):
    after=http_counters(base)
    if after and after["active"]==0 and after["total"]>=before["total"]+4: break
    time.sleep(.1)
print(before,after,flush=True)
assert after['active']==0 and after['total']-before['total']==4
out=dict(profile={k:info.get(k) for k in ('version','tp_size','ep_size','served_model_name','speculative_algorithm','context_length','max_total_num_tokens','cuda_graph_config')},gpu_before=gpus,gpu_after=gpu_snapshot(),before=before,after=after,trials=trials)
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')
