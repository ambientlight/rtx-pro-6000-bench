import json
from pathlib import Path

root=Path(__file__).resolve().parent
q=Path('/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/docker/deepseek-v41/upstream/full-canary')
def read(path): return json.loads(path.read_text())
def read_totals(snap):
    out={}
    for rank, rows in snap['ranks'].items():
        modes={}
        for r in rows:
            if r['path']!='eager_wo_b' or r['real_rows']!=r['executed_rows']: continue
            m=modes.setdefault(r['mode'],{'calls':0,'rows':0})
            m['calls']+=r['calls']; m['rows']+=r['real_rows']*r['calls']
        out[rank]=modes
    return out
frames=[]
for line in (root/'projection-counters.jsonl').read_text().splitlines():
    s=json.loads(line)
    events=[s['monotonic']+(t*1e-9-s['utc_epoch']) for t in s['mtimes'].values()]
    frames.append((min(events),max(events), read_totals(s)))
runs=read(q/'pearl-handbook.json')['runs']
if (q/'window-pearl.json').exists(): runs.append(read(q/'window-pearl.json'))
idle=read(q/'mining-evidence.json')['idle']['sliced_total_tmac_s']
results=[]
for r in runs:
    start=r['start_monotonic']; end=r['end_monotonic']
    before=[f for f in frames if f[1]<start]
    nearby=[f for f in frames if abs(f[1]-end)<2 and abs(f[0]-end)<2]
    if not before or len(nearby)!=1: continue
    a=before[-1][2]; b=nearby[0][2]
    assert len(a)==len(b)==4 and a.keys()==b.keys()
    ttft=r.get('ttft_s',r.get('first_text_seconds'))
    request={'input_tokens':r['input_tokens'],'output_tokens':r.get('output_tokens',(r.get('usage') or {}).get('output_tokens')),
             'all_idle_delta_seconds_after_http':round(nearby[0][1]-end,5),'phases':{}}
    for mode,seconds in [('EXTEND',ttft),('TARGET_VERIFY',end-start-ttft)]:
        diff={rank:{k:b[rank].get(mode,{}).get(k,0)-a[rank].get(mode,{}).get(k,0) for k in ('calls','rows')} for rank in sorted(b)}
        assert all(x['rows']>=0 and x['calls']>=0 for x in diff.values())
        mac=sum(x['rows'] for x in diff.values())*2048*5120
        tps=mac/seconds/1e12
        request['phases'][mode]=dict(by_rank=diff,wall_seconds=seconds,logical_macs=mac,logical_tmac_s=tps,percent_of_measured_idle_mac_rate=100*tps/idle)
    results.append(request)
ret=dict(shape_source='Differences of four rank-local idle-flush counters, restricted to the layer 0 eager_wo_b path with real_rows == executed_rows. End flush must fall within 2s of HTTP completion on every rank; last pre-admission flush provides the initial count.', logical_macs_per_real_row_per_tp_rank=2048*5120, checkpoint_weight_shard=[5120,2048],idle_sliced_worker_tmac_s=idle, timing='Prefill denominator is HTTP start to first streamed text, verify denominator first text to terminal status. This boundary can include generating or buffering the first token. Each timing includes other DeepSeek layers and engine overhead.', interpretation='Logical real-model MACs in the eligible projection. Some projections may lack job/queue availability. This counts mathematical work, not candidate hashes, accepted shares, credited hashrate or coin income.',rows=results)
(q/'projection-work.json').write_text(json.dumps(ret,indent=2)+'\n')
print([(r['input_tokens'],{k:round(v['logical_tmac_s'],5) for k,v in r['phases'].items()}) for r in results])
