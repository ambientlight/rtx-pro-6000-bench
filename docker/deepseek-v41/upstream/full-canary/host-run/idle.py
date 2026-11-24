import json,sys,time
from pathlib import Path
sys.path.insert(0,'/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/scripts')
import dsv41_load as load
last=None
since=None
window=load.DrainWindow(60,[0])
while True:
    rows=load.parse_loads(json.loads(load.fetch(load.LOAD_PATH)))
    active,counts=load.activity_metrics(load.fetch('/metrics'))
    now=time.monotonic()
    if active or last!=counts or not load.is_idle(rows):
        since=None
    if since is None and not active and load.is_idle(rows):
        since=now
    last=counts
    sched=window.observe(rows,now)
    if sched and since is not None and now-since>=60:
        print('60s idle interval ended',time.time(),flush=True)
        break
    print('idle seconds',round(now-since) if since is not None else 0,'active HTTP',active,flush=True)
    time.sleep(2)
