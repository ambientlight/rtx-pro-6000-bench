import concurrent.futures
import json
import pathlib
import socket
import sys
import time
import urllib.request

path=pathlib.Path('/tmp/pearl-upstream-native/control-dir').read_text().strip()
url='http://127.0.0.1:31874/generate'
def state(name):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(3)
        s.connect(f'{path}/{name}.sock')
        s.sendall(b'STATUS\n')
        return json.loads(s.makefile('r').readline())
def submit():
    payload={'input_ids':[4]*16, 'sampling_params':{'temperature':0,'max_new_tokens':24,'ignore_eos':True}}
    request=urllib.request.Request(url, data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=300) as http:
        result=json.load(http)
        assert len(result['output_ids'])==result['meta_info']['completion_tokens']==24
        return http.status,result['output_ids']
def sample():
    before=state('worker'); assert before['completed_slices']>0 and not before['holds']
    t0=time.monotonic_ns()
    held=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut=ex.submit(submit)
        while not fut.done():
            broker=state('broker'); worker=state('worker')
            if worker['holds'] and not worker['gpu_busy'] and broker['drained']==[True]:
                held.append((time.monotonic_ns(),worker['completed_slices']))
            time.sleep(.002)
        status,ids=fut.result()
    response_at=time.monotonic_ns()
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        after=state('worker'); broker=state('broker')
        if after['release_ns']>=t0 and after['first_gpu_seen_ns']>=after['release_ns'] and after['first_slice_end_ns']>=after['first_gpu_seen_ns'] and after['completed_slices']>held[-1][1] and not any(after[k] for k in ['faulted','holds']) and broker['idle']==[True] and not broker['claims']: break
        time.sleep(.001)
    else: raise AssertionError((after,broker))
    assert len(held)>=2 and len({n for _,n in held})==1
    return dict(http_status=status,output_ids=ids,held_observations=len(held),held_completed_slices=held[0][1],before_completed_slices=before['completed_slices'],after_completed_slices=after['completed_slices'],observed_held_window_ms=(held[-1][0]-held[0][0])/1e6,http_send_to_first_observed_drain_ms=(held[0][0]-t0)/1e6,http_duration_ms=(response_at-t0)/1e6,release_to_observed_gpu_entry_ms=(after['first_gpu_seen_ns']-after['release_ns'])/1e6,broker_final=broker)
rows=[sample() for i in range(4)]
pathlib.Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+'\n')
