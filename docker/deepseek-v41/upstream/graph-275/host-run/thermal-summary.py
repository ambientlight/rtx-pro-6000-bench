import csv
from datetime import datetime
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
clock = json.loads((root / 'receipts/timebase.json').read_text())
offset = clock['realtime_ns'] - clock['monotonic_ns']

def load(name):
    boards = {i: [] for i in range(4)}
    with (root / name).open() as fp:
        for row in csv.reader(fp, skipinitialspace=True):
            epoch = datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
            boards[int(row[1])].append((int(epoch * 1e9) - offset, row[2:]))
    return boards

def summarize(samples, reasons, start, end):
    out = []
    for rank in range(4):
        rows = [(t, r) for t, r in samples[rank] if start <= t <= end]
        counters = [r for t, r in reasons[rank] if start <= t <= end]
        if not rows:
            continue
        # Per-board time average and energy from trapezoids inside the trace.
        seconds = (rows[-1][0] - rows[0][0]) / 1e9
        joules = sum((float(a[2]) + float(b[2])) / 2 * (t1 - t0) / 1e9
                     for (t0, a), (t1, b) in zip(rows, rows[1:]))
        out.append(dict(rank=rank, samples=len(rows), inside_sample_span_s=seconds,
            temp_max_c=max(int(r[1]) for _, r in rows),
            power_mean_w=round(joules/seconds,2) if seconds else None,
            power_limits_w=sorted({float(r[3]) for _, r in rows}),
            sm_clocks_median_mhz=statistics.median(int(r[4]) for _, r in rows),
            peak_memory_mib=max(int(r[7]) for _,r in rows),
            hw_thermal_throttle_delta_us=(int(counters[-1][0])-int(counters[0][0])) if len(counters)>1 else None,
            sw_thermal_throttle_delta_us=(int(counters[-1][1])-int(counters[0][1])) if len(counters)>1 else None))
    return out

if __name__ == '__main__':
    import sys
    records, counters = load('gpus.csv'), load('thermal.csv')
    for path in map(Path,sys.argv[1:]):
        trace=json.loads(path.read_text())
        if not trace.get('complete'):
            raise ValueError(f'incomplete trace {path}')
        rows=summarize(records,counters,trace['requested_horizon_start_monotonic_ns'],trace['requested_horizon_end_monotonic_ns'])
        path.with_name(path.stem+'-hardware.json').write_text(json.dumps(rows,indent=2)+'\n')
