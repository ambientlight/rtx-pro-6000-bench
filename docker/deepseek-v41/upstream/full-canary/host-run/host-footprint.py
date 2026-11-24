import json,time
from pathlib import Path
run=Path(__file__).resolve().parent
receipt=Path('/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/docker/deepseek-v41/upstream/full-canary/full-run.json')
results=json.loads(receipt.read_text())
samples=[json.loads(line) for line in (run/'footprint.jsonl').read_text().splitlines()]
original=[json.loads(line) for line in (run/'footprint-incumbent.jsonl').read_text().splitlines()]
for arm,start,stop,group in [
        ('stock','candidate-started-at','stock-stopped-at',samples),
        ('pearl','pearl-started-at','eager-control-started-at',samples),
        ('pearl-control','eager-control-started-at','stock-throughput-started-at',samples),
        ('stock-throughput','stock-throughput-started-at','incumbent-restore-at',samples),
        ('incumbent-unlocked','incumbent-restore-at',None,original)]:
    t0=(run/start).stat().st_mtime
    t1=(run/stop).stat().st_mtime if stop else time.time()
    rows=[v for v in group if t0<=v['time']<=t1 and 'anon_gib' in v]
    if rows:
        results['profiles'].setdefault(arm,{})['host_footprint_gib']={
            'sample_count':len(rows),
            'sample_interval_seconds':6,
            'sampled_cgroup_high_water':max(v['cgroup_gib'] for v in rows),
            'sampled_anon_high_water':max(v['anon_gib'] for v in rows),
            'sampled_file_high_water':max(v['file_gib'] for v in rows),
            'sampled_unevictable_high_water':max(v['locked_gib'] for v in rows),
            'minimum_host_available':min(v['available_gib'] for v in rows)}
results['memory_method']='Linux cgroup v2 memory.current and memory.stat sampled from the transient candidate unit or original Docker scope. file includes charged page cache, and unevictable overlaps other categories. Independent component maxima can occur at different times. Candidate user units had a 365 GiB memory.max including file cache.'
receipt.write_text(json.dumps(results,indent=2)+'\n')
