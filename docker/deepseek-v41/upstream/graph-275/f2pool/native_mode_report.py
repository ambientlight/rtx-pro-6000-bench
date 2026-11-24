"""Compare exact K780 counters over equal sustained idle windows and summarize Responses samples."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median

from native_check import logs
from native_report import RANKS, TILE_MACS, lines, timings
from throughput import write


def nearest(items, when):
    return min(items, key=lambda p: abs(p["time"] - when))


def mining_result(ranks, telemetry):
    rates = []
    segments = []
    starts, ends = [], []
    for samples in ranks:
        a, b = samples[0], samples[-1]
        rates.append((b["tmac"]-a["tmac"])/(b["elapsed"]-a["elapsed"]))
        segments.append([(q["tmac"]-p["tmac"])/(q["elapsed"]-p["elapsed"]) for p,q in zip(samples,samples[1:])])
        starts.append(a["time"])
        ends.append(b["time"])
    windows = [sum(v) for v in zip(*segments,strict=True)]
    gpu = []
    for rank in RANKS:
        s = [v for t,i,v in telemetry if i==rank and max(starts)<=t<=min(ends)]
        if len(s) < 2:
            raise RuntimeError("Missing GPU telemetry over the measured interval")
        gpu.append(dict(mean_w=mean(v[2] for v in s), mean_sm_mhz=mean(v[3] for v in s),
                        mean_temp_c=mean(v[1] for v in s), peak_c=max(v[1] for v in s),
                        mean_util_pct=mean(v[0] for v in s), sw_thermal_us=s[-1][5]-s[0][5],
                        hw_thermal_us=s[-1][6]-s[0][6]))
    return dict(total_tmac_s=sum(rates), rank_tmac_s=rates, windows_tmac_s=windows, samples_by_rank=gpu,
                common_seconds=min(ends)-max(starts), start_utc=max(starts), end_utc=min(ends))


def embedded_points(records, broker, control, last_request, warm):
    # Compare each 30-second standalone counter interval to the same elapsed duration
    # on its embedded rank. Every sample must fall in one uninterrupted four-rank lease.
    closes = {x["epoch"]-1:x["monotonic_ns"] for x in broker if x["event"]=="native_demand"}
    for release in broker:
        if release["event"]!="native_release" or release["monotonic_ns"]<last_request-1e9:
            continue
        epoch = release["epoch"]
        points = [[e for e in records[i] if e["event"]=="native_work" and e["graphs"]==139 and
                   e["epoch"]==epoch and e["monotonic_ns"]<closes.get(epoch,float("inf"))] for i in RANKS]
        if any(not p for p in points):
            continue
        t0 = max(p[0]["time"] for p in points) + warm
        if any(p[-1]["time"] < t0+b[-1]["elapsed"]-b[0]["elapsed"] for p,b in zip(points,control)):
            continue
        aligned = []
        for p,b in zip(points, control):
            selected = [nearest(p,t0+x["elapsed"]-b[0]["elapsed"]) for x in b]
            aligned.append([dict(time=e["time"],elapsed=e["monotonic_ns"]/1e9,tmac=e["tiles"]*TILE_MACS/1e12)
                            for e in selected])
        return epoch, aligned
    raise ValueError("Waiting for an uninterrupted post-benchmark native idle window")


def summary(inference):
    rows = inference["rows"]
    summary = []
    for size in sorted({r["size"] for r in rows}):
        group = [r for r in rows if r["size"]==size]
        summary.append(dict(target=size, n=len(group), input_range=[min(r["input_tokens"] for r in group),
            max(r["input_tokens"] for r in group)], output_tokens=group[0]["output_tokens"],
            ttft_s=median(r["ttft_ms"] for r in group)/1000, prefill_tps=median(r["server_prefill_tps"] for r in group),
            client_decode_tps=median(r["client_decode_tps"] for r in group),
            cached_tokens=sum(r["cached_tokens"] for r in group)))
    return summary


def report(bench, run, out):
    read = lambda path: json.loads(path.read_text())
    capture = read(bench/"standalone.json")
    control = capture["counters_by_rank"]
    # Independent process startup can place the final snapshot on opposite sides
    # of the deadline. Compare the same number of whole 30s windows on every rank.
    count = min(map(len,control))
    control = [p[:count] for p in control]
    with (bench/"gpu.csv").open() as stream:
        telemetry = [(datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(),int(row[1]),list(map(float,row[2:])))
                     for row in csv.reader(stream) if len(row)==9]
    serving = read(bench/"non-joined.json")
    joined = read(bench/"joined.json")
    if not serving.get("complete"):
        raise ValueError("Non-joined inference benchmark is incomplete")
    indexed = {r["prompt_sha256"]:r for r in serving["rows"]}
    for old in joined["rows"]:
        new = indexed[old["prompt_sha256"]]
        if any(old[k]!=new[k] for k in ("input_tokens","output_tokens","cached_tokens","budget")):
            raise ValueError("Inference inputs or decode budgets differ")
    requests = serving["rows"] + read(bench/"transitions.json")["rows"]
    records, broker = logs(run), lines(run/"broker.jsonl")
    epoch, embedded = embedded_points(records,broker,control,max(r["end_ns"] for r in requests),capture["warm_seconds"])
    standalone = mining_result(control,telemetry)
    native = mining_result(embedded,telemetry)
    for field in ("accepted","rejected"):
        native[field] = [sum(e["event"]=="share" and e["accepted"]==(field=="accepted") and
            native["start_utc"]<=e["time"]<=native["end_utc"] for e in records[i]) for i in RANKS]
        a,b = (nearest(capture["samples"],standalone[k]) for k in ("start_utc","end_utc"))
        standalone[field] = [hi-lo for hi,lo in zip(b[field],a[field])]
    timing = timings(requests,broker,records)
    stats = dict(mode="non-joined", k780_slice_tiles=188, idle_epoch=epoch, standalone_binary_sha256=capture["binary_sha256"],
                 standalone=standalone, embedded=native, embedded_pct_standalone=100*native["total_tmac_s"]/standalone["total_tmac_s"],
                 inference={s["mode"]:summary(s) for s in (joined,serving)}, paired_prompts=len(joined["rows"]),
                 timing=timing, requests_for_timing=len(requests))
    write(out,stats)
    return stats


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bench","run","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    stats = report(**vars(parser.parse_args()))
    print(f"Embedded K780 achieved {stats['embedded_pct_standalone']:.2f}% of standalone")
