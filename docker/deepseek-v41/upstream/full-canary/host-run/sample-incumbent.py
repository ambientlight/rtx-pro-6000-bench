import json,time
from pathlib import Path
base=Path(__file__).resolve().parent
scope=Path('/sys/fs/cgroup/dsv41.slice/docker-087c42c0a93e58dc2ab902b13c1719a03ba6a2a261a3b685129d7743df5e3130.scope')
while not (base/'experiment-complete').exists():
    if (scope/'memory.current').exists():
        stat=dict(line.split() for line in (scope/'memory.stat').read_text().splitlines())
        data=dict(time=time.time(),available_gib=round(int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))/2**20,1), cgroup_gib=round(int((scope/'memory.current').read_text())/2**30,1))
        data.update({out:round(int(stat[s])/2**30,1) for s,out in [('anon','anon_gib'),('file','file_gib'),('unevictable','locked_gib')]})
        print(json.dumps(data),flush=True)
    time.sleep(6)
