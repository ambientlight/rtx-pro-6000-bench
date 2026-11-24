"""Bound the physical GPU telemetry strictly to each benchmark's measurement window."""
import csv
import datetime as dt
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path('/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/docker/deepseek-v41/upstream/full-canary')
raw_path = Path(__file__).parent / 'gpu-telemetry.csv'
tz = ZoneInfo('America/Los_Angeles')
clock_offset = time.time() - time.monotonic()

def utc(epoch):
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat(timespec='milliseconds')

def stats(vals):
    return dict(min=min(vals),median=statistics.median(vals),max=max(vals)) if vals else None

def sample_group(rows):
    active = [r for r in rows if r['util'] > 0]
    high = [r for r in rows if r['util'] >= 80]
    return dict(
        samples=len(rows),active_samples=len(active),high_util_samples=len(high),
        memory_clock_mhz_when_active=stats([r['memory_clock'] for r in active]),
        memory_clock_mhz_counts_when_active=dict(sorted(Counter(r['memory_clock'] for r in active).items())),
        memory_clock_mhz_above_80pct_gpu=stats([r['memory_clock'] for r in high]),
        sm_clock_mhz_when_active=stats([r['sm'] for r in active]),
        power_draw_w_when_active=stats([r['watts'] for r in active]),
        power_limit_w=sorted(set(r['limit'] for r in rows)),
        util_pct=stats([r['util'] for r in rows]),peak_vram_mib=max((r['used'] for r in rows),default=None)
    )

def parse_samples():
    samples=[]
    with raw_path.open() as file:
        for items in csv.reader(file, skipinitialspace=True):
            if len(items) != 10:
                continue
            ts,index,gpu,pstate,watts,limit,sm,mem,util,used = items
            try:
                stamp=dt.datetime.strptime(ts,'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=tz).timestamp()
                samples.append(dict(time=stamp,index=int(index),uuid=gpu,watts=float(watts),limit=float(limit),sm=int(sm),memory_clock=int(mem),util=int(util),used=int(used)))
            except ValueError:
                continue
    return samples

def phase(path, samples):
    doc=json.loads(path.read_text())
    if path.name.startswith('window-') and 'start_monotonic' in doc:
        start,end = doc['start_monotonic'] + clock_offset, doc['end_monotonic'] + clock_offset
        intervals=[(start,end)]
        complete=True
    elif doc.get('phase') in ('native','handbook','eager-control','prefix') and 'start_utc_epoch' in doc:
        start,end = doc['start_utc_epoch'], doc.get('end_utc_epoch')
        intervals=[]
        for batch in doc.get('runs',[]):
            for r in batch.get('requests',[batch]):
                if 'start_monotonic' in r and 'end_monotonic' in r:
                    intervals.append((r['start_monotonic'] + clock_offset,r['end_monotonic'] + clock_offset))
        complete=doc.get('complete',False)
    else:
        return
    if not end:
        if not intervals:
            return
        end=max(b for _,b in intervals)
    rows=[r for r in samples if start <= r['time'] <= end]
    in_requests=[r for r in rows if any(a <= r['time'] <= b for a,b in intervals)]
    result=dict(start_utc=utc(start),end_utc=utc(end),interval_complete=complete,
                receipt=path.name,first_sample_utc=utc(rows[0]['time']) if rows else None,
                last_sample_utc=utc(rows[-1]['time']) if rows else None,
                missing_leading_s=round(max(0,samples[0]['time']-start),3) if samples else None,
                elapsed_all_phase_s=round(end-start,3),all_phase={str(g):sample_group([r for r in rows if r['index']==g]) for g in range(4)})
    if intervals:
        result['completed_request_intervals']=len(intervals)
        result['within_recorded_requests']={str(g):sample_group([r for r in in_requests if r['index']==g]) for g in range(4)}
    if doc.get('status') == 'failed':
        result.update(attempted_request_intervals=1,completed_request_intervals=0,
                      service_outcome=doc.get('server_error','failed'),timing_basis=doc['timing_basis'],
                      interval_complete=True)
    temps=Path(__file__).parent/'thermal-telemetry.csv'
    if temps.exists():
        observed={i:[] for i in range(4)}
        with temps.open() as f:
            for ts,index,c in csv.reader(f,skipinitialspace=True):
                try:
                    t=dt.datetime.strptime(ts,'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=tz).timestamp()
                    if start <= t <= end and (not intervals or any(a <= t <= b for a,b in intervals)):
                        observed[int(index)].append(float(c))
                except ValueError:
                    continue
        if any(observed.values()):
            result['temperature_c_by_gpu']={str(i):{'samples':len(v),'stats':stats(v)} for i,v in observed.items()}
    return result

def main():
    rows=parse_samples()
    ids={str(r['index']):r['uuid'] for r in rows}
    phases={}
    for p in sorted(root.glob('*.json')):
        if p.name == 'hardware-throughput.json':
            continue
        result=phase(p,rows)
        if result:
            phases[p.stem]=result
    out=dict(interval_s=2, source='nvidia-smi board telemetry; independent physical per-card samples, host America/Los_Angeles timestamps converted to UTC',first_sample_utc=utc(rows[0]['time']) if rows else None,index_to_uuid=ids,
             columns_note='Power is instantaneous reported board draw. Utilization is the sampled NVIDIA GPU utilization percentage. Within-recorded-requests excludes tokenization/setup between requests. A max is a sample maximum. Handbook phase includes prompt generation and tokenization. Short request intervals can fall between samples.',
             phases=phases)
    dest=root/'hardware-throughput.json'
    tmp=dest.with_suffix('.part')
    tmp.write_text(json.dumps(out,indent=2)+'\n')
    tmp.replace(dest)

if __name__=='__main__':
    main()
