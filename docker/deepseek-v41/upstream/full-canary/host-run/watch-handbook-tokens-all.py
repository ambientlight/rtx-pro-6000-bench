import json,os,re,sys,time
from pathlib import Path
p=Path('docker/deepseek-v41/upstream/full-canary');sys.path.insert(0,str(p))
from qualify import request
names=('uncached_prompt_tokens_histogram', 'generation_tokens_histogram')
def snap():
 out={}
 with request('http://127.0.0.1:18000','/metrics') as resp:
  for line in resp.read().decode().splitlines():
   m=re.match(r'^sglang:(\w+)_(sum|count)\{([^}]+)\}\s+([^ ]+)',line)
   if m and m[1] in names:
    k=m[1]+'_'+m[2];out[k]=out.get(k,0)+float(m[4])
 return out
r=Path('/mnt/hot/ambientlight/sglang-sm120-runtime/full-canary')
with (r/'pearl-handbook-token-all-edges.jsonl').open('w') as f:
 last=None
 while True:
  data=snap()
  if data!=last:
   print(json.dumps({'monotonic':time.monotonic(),'utc':time.time(),'metrics':data}),file=f,flush=True);last=data
  if json.loads((p/'pearl-handbook.json').read_text()).get('complete'):break
  time.sleep(1)
