import json, time, pathlib, urllib.request
target=pathlib.Path('/tmp/pearl-upstream-native/samegpu-dir').read_text().strip()
url='http://127.0.0.1:31874'
for _ in range(120):
    try:
        with urllib.request.urlopen(url+'/v1/models',timeout=2) as s:
            if json.load(s)['data'][0]['id']=='sm120-dsv41-canary': break
    except OSError: time.sleep(1)
else: raise RuntimeError('GPU2 canary boot failed')
times=[]
for _ in range(4):
    payload={'input_ids':[4]*16,'sampling_params':{'temperature':0,'max_new_tokens':24,'ignore_eos':True}}
    req=urllib.request.Request(url+'/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    ts=time.monotonic()
    with urllib.request.urlopen(req,timeout=300) as res:
        v=json.load(res); assert v['meta_info']['completion_tokens']==len(v['output_ids'])==24
    times.append((time.monotonic()-ts)*1000)
pathlib.Path(target,'warm.json').write_text(json.dumps(times))
