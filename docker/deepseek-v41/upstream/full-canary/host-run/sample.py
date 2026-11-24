import json,subprocess,time
from pathlib import Path
d=Path(__file__).parent
cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[-1].lstrip('/')
while not (d/'experiment-complete').exists():
    unit='dsv41-sm120-full-candidate.service'; field=subprocess.run(['systemctl','--user','show','-p','ControlGroup','--value',unit],capture_output=True,text=True).stdout.strip()
    rec={'time':time.time(),'available_gib':round(int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/2**20,1)}
    if field:
        p=Path('/sys/fs/cgroup')/field.lstrip('/')
        try:
            st=dict(line.split() for line in (p/'memory.stat').read_text().splitlines())
            rec.update(cgroup_gib=round(int((p/'memory.current').read_text())/2**30,1),anon_gib=round(int(st['anon'])/2**30,1),file_gib=round(int(st['file'])/2**30,1),locked_gib=round(int(st['unevictable'])/2**30,1))
        except FileNotFoundError:pass
    v=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    rec['free_gpu_mib']=[int(x) for x in v.stdout.split() if x.isdigit()]
    print(json.dumps(rec),flush=True)
    time.sleep(6)
