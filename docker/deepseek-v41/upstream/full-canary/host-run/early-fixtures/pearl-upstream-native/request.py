import json, pathlib, time, urllib.request, sys
model=sys.argv[1]; path=pathlib.Path(sys.argv[2])
base='http://127.0.0.1:31874'
def req(url,payload=None):
    body=json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(urllib.request.Request(base+url,data=body,headers={'Content-Type':'application/json'}), timeout=480) as r:
        return (r.status,json.load(r))
models=req('/v1/models'); info=req('/server_info')
first=time.monotonic()
completion=req('/v1/completions',dict(model=model,prompt='ping',temperature=0,max_tokens=4,ignore_eos=True))
elapsed=(time.monotonic()-first)*1000
native=req('/generate',dict(input_ids=[4,4,4,4],sampling_params=dict(temperature=0,max_new_tokens=4,ignore_eos=True)))
assert models[0] == completion[0] == native[0] == 200
assert completion[1]['usage']['completion_tokens']==4 and native[1]['meta_info']['completion_tokens']==4
if model=='sm120-tiny': assert completion[1]['choices'][0]['text'].strip()=='ready ready ready ready' and native[1]['output_ids']==[5]*4
out=dict(model=model, get_models=models,completion=completion,native=native,first_request_ms=elapsed)
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
