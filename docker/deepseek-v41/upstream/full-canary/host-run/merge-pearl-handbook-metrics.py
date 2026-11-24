from pathlib import Path
import json,sys
p=Path('docker/deepseek-v41/upstream/full-canary')
sys.path.insert(0,str(p))
from throughput import write,single_request_metric
r=Path('/mnt/hot/ambientlight/sglang-sm120-runtime/full-canary')
path=p/'pearl-handbook.json'
result=json.loads(path.read_text())
assert result['complete'] and len(result['runs'])==3
def edges(path, metric):
    name=metric+'_count'
    last=None
    for line in path.read_text().splitlines():
        point=json.loads(line)
        if last is not None:
            count=point['metrics'][name]-last['metrics'][name]
            if count:
                assert count==1
                yield last,point
        last=point
tpot=list(edges(r/'pearl-handbook-metric-edges.jsonl','request_time_per_output_token_seconds'))
uncached=list(edges(r/'pearl-handbook-token-all-edges.jsonl','uncached_prompt_tokens_histogram'))
if min(len(tpot),len(uncached))<3: raise RuntimeError('incomplete metric edges')
for row,(tb,ta),(cb,ca) in zip(result['runs'],tpot[:3],uncached[:3],strict=True):
    end=row['end_monotonic']
    for point in (ta,ca):
        assert abs(point['monotonic']-end)<3, (end,point)
    before,after=tb['metrics']|cb['metrics'],ta['metrics']|ca['metrics']
    row['server_metrics_before']=before
    row['server_metrics_after']=after
    gap=single_request_metric(before,after,'request_time_per_output_token_seconds')
    completed=single_request_metric(before,after,'generation_tokens_histogram')
    uncached=single_request_metric(before,after,'uncached_prompt_tokens_histogram')
    assert gap>0 and completed==row['output_tokens'] and 0<=uncached<=row['input_tokens']
    row['native_tpot_histogram_decode_tps']=1/gap
    native_cached=int(row['input_tokens']-uncached)
    cached=row.get('cached_tokens')
    if cached is None:
        row['cached_tokens']=native_cached
        row['cached_tokens_source']='native isolated histogram delta'
    else:
        assert cached==native_cached
        row['cached_tokens_source']='responses.usage'
    row['client_effective_prefill_tps']=uncached/row['ttft_s']
    row['histogram_observation']={'method':'first aggregate histogram edge recording completion in the isolated serving process', 'stream_tpot_edge_monotonic':ta['monotonic'], 'token_histogram_edge_monotonic':ca['monotonic']}
write(path,result)
