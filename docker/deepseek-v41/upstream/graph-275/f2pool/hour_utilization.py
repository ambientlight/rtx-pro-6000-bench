"""Observe NVML's per-GPU utilization during the remaining measured trial window."""
import argparse
from bisect import bisect_left, bisect_right
from datetime import datetime
import json
import os
from pathlib import Path
import signal
from statistics import mean
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def collect(run):
    os.umask(0o077)
    window = read(run/"serving/window.json")
    end = window["end_monotonic_ns"]
    if time.monotonic_ns() >= end:
        raise ValueError("The trial has ended")
    folder = run/"metrics"
    command = ["stdbuf","-oL","nvidia-smi","--query-gpu=index,utilization.gpu,timestamp",
               "--format=csv,noheader,nounits","--loop-ms=1000"]
    stopping = False
    with (folder/"gpu-utilization.jsonl").open("x",buffering=1) as out:
        proc = subprocess.Popen(command,stdout=subprocess.PIPE,text=True,bufsize=1)
        def stop(*unused):
            nonlocal stopping
            stopping = True
            proc.terminate()
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        cards = {}
        try:
            for line in proc.stdout:
                now = time.monotonic_ns()
                if stopping or now >= end or (run/"serving/restart.json").exists():
                    break
                parts = line.strip().split(",")
                if len(parts) != 3 or not parts[0].strip().isdigit():
                    continue
                rank = int(parts[0])
                if rank == 0:
                    cards = {}
                value = parts[1].strip()
                if rank not in range(4) or not value.isdigit() or not 0 <= int(value) <= 100:
                    continue
                stamp = datetime.strptime(parts[2].strip(),"%Y/%m/%d %H:%M:%S.%f").timestamp()
                cards[rank] = (int(value),now,round(stamp*1000))
                if len(cards) != 4:
                    continue
                row = dict(first_ns=cards[0][1],last_ns=now,
                           util_pct=[cards[i][0] for i in range(4)],
                           nvidia_epoch_ms=[cards[i][2] for i in range(4)])
                try:
                    fleet = read(run/"idle/status.json")
                    row.update(fleet_ns=fleet["monotonic_ns"],serving=fleet["broker"]["serving"],
                               cooling=[card["cooling"] for card in fleet["ranks"]],
                               drained=fleet["broker"]["drained"])
                except (FileNotFoundError,ValueError,KeyError):
                    pass
                print(json.dumps(row,separators=(",",":")),file=out)
                cards = {}
            if not stopping and time.monotonic_ns() < end and not (run/"serving/restart.json").exists():
                raise RuntimeError("Utilization source ended before the trial")
        finally:
            proc.terminate()
            try:
                proc.wait(3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.endswith("\n")]


def stats(values):
    if not values:
        return {"sweeps":0}
    return dict(sweeps=len(values),mean_pct=round(mean(values),3),min_pct=min(values),max_pct=max(values),
                sweeps_at_most_5_pct=sum(n<=5 for n in values),
                sweeps_at_least_90_pct=sum(n>=90 for n in values),
                sweeps_from_72_through_73_pct=sum(72<=n<=73 for n in values))


def overall(rows):
    return {"fleet":stats([mean(row["util_pct"]) for row in rows]),
            "by_gpu":[stats([row["util_pct"][i] for row in rows]) for i in range(4)]}


def thermal_state(fleet, times, start, stop, gpu):
    # Bound the whole hardware lookback using independently accumulated thermal-hold seconds.
    a, b = bisect_right(times,start)-1, bisect_left(times,stop)
    if a < 0 or b >= len(fleet) or any(times[i+1]-times[i] > 4_000_000_000 for i in range(a,b)):
        return None
    left, right = fleet[a]["ranks"][gpu],fleet[b]["ranks"][gpu]
    span = (times[b]-times[a])/1e9
    held = right["cooling_seconds"]-left["cooling_seconds"]
    if abs(held) < .005 and not left["cooling"] and not right["cooling"]:
        return "released"
    if abs(held-span) < .035 and left["cooling"] and right["cooling"]:
        return "held"
    return None


def report(run):
    folder = run/"metrics"
    window = read(run/"serving/window.json")
    start, stop = window["start_monotonic_ns"], window["end_monotonic_ns"]
    rows = [row for row in lines(folder/"gpu-utilization.jsonl")
            if start <= row["first_ns"] <= row["last_ns"] <= stop]
    if not rows:
        raise ValueError("No complete sweeps within the observed trial")
    events = [r for r in lines(run/"idle/broker-events.jsonl") if r["event"]=="state"]
    stamps = [r["monotonic_ns"] for r in events]
    fleet = [r["idle"] for r in lines(folder/"series.jsonl") if "idle" in r]
    times = [r["source_monotonic_ns"] for r in fleet]
    modes = {k:[] for k in ("idle","serving","transition_or_unknown")}
    thermal = {k:[[] for i in range(4)] for k in ("held","released","mixed_or_unknown")}
    for row in rows:
        # NVML reports a past sampling period of up to one second. A two-second
        # guard allows one period for kernel activity and another for counter refresh.
        left, right = row["first_ns"]-2_000_000_000, row["last_ns"]
        a,b = bisect_right(stamps,left)-1,bisect_right(stamps,right)-1
        mode = "transition_or_unknown"
        if a >= 0 and a == b and events[a]["watched"]==4:
            if events[a]["serving"] and events[a]["held"]==4:
                mode = "serving"
            if not events[a]["serving"] and events[a]["held"]==0:
                mode = "idle"
        modes[mode].append(row)
        if mode=="idle":
            for gpu,pct in enumerate(row["util_pct"]):
                key=thermal_state(fleet,times,left,right,gpu) or "mixed_or_unknown"
                thermal[key][gpu].append(pct)
    dt = [(b["last_ns"]-a["last_ns"])/1e9 for a,b in zip(rows,rows[1:])]
    span = (rows[-1]["last_ns"]-rows[0]["last_ns"])/1e9
    measured = (stop-start)/1e9
    same_series = sum(interval for interval in dt if 0 < interval < 1.5)
    interior = [point for point in fleet
                if rows[0]["first_ns"] <= point["source_monotonic_ns"] <= rows[-1]["last_ns"]]
    cooling = None
    if len(interior) >= 2:
        first, last = interior[0],interior[-1]
        seconds = (last["source_monotonic_ns"] - first["source_monotonic_ns"])/1e9
        held = [b["cooling_seconds"] - a["cooling_seconds"]
                for a,b in zip(first["ranks"],last["ranks"],strict=True)]
        if seconds > 0 and all(0 <= value <= seconds+.035 for value in held):
            cooling = dict(seconds=seconds,held_seconds_by_gpu=held,
                           pct_of_interval_held_by_gpu=[100*value/seconds for value in held])
    return dict(definition="Percent of NVIDIA's past sampling period with one or more kernels executing.",
        weighting="Arithmetic means of one-second query sweeps. Fleet mean gives each GPU equal weight.",
        stable_state_method="An unchanged broker state with all workers watched and held for serving or released by the broker for idle throughout the preceding two seconds. Thermal status additionally requires unchanged cumulative hold time or continuous accrual across bracketing fleet snapshots; unclassified sweeps remain in the overall mean.",
        start_epoch_ms=min(rows[0]["nvidia_epoch_ms"]),end_epoch_ms=max(rows[-1]["nvidia_epoch_ms"]),
        sweep_count=len(rows),first_to_last_s=span,continuous_observed_s=same_series,
        interior_cumulative_thermal_holds=cooling,
        actual_trial_s=measured,pct_of_actual_trial_span=100*same_series/measured,
        max_adjacent_s=max(dt,default=None),overall=overall(rows),
        by_stable_broker_state={k:overall(v) for k,v in modes.items()},
        by_idle_thermal_state_by_gpu={k:[stats(card) for card in v] for k,v in thermal.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("collect","report"))
    parser.add_argument("--run-dir",required=True,type=Path)
    args = parser.parse_args()
    if args.mode=="collect":
        collect(args.run_dir)
    else:
        print(json.dumps(report(args.run_dir),indent=2,allow_nan=False))
