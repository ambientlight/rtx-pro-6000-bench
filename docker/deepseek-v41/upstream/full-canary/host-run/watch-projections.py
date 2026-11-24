"""Capture the aggregate shape counters that ranks flush once they go idle."""
import json,time
from pathlib import Path
base=Path(__file__).resolve().parent
last={}
while not (base/'proof-arm-done').exists():
    files=sorted((base/'pearl-shapes').glob('*-tp*.json'))
    if len(files)==4:
        starts={f.name:f.stat().st_mtime_ns for f in files}
        if set(last)!=set(starts) or all(starts[k]!=last[k] for k in starts):
            data={f.stem.rsplit('-',1)[1]:json.loads(f.read_text()) for f in files}
            if starts=={f.name:f.stat().st_mtime_ns for f in files}:
                row=dict(monotonic=time.monotonic(),utc_epoch=time.time(),mtimes=starts,ranks=data)
                with (base/'projection-counters.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
                last=starts
    time.sleep(.1)
