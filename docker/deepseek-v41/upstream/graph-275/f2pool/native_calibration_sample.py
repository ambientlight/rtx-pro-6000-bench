"""Measure paid K780 throughput and request handoffs in one native service run."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from native_check import logs, rows
from native_mode_report import embedded_points, mining_result
from native_report import RANKS, timings
from throughput import write


def sample(run, requests, telemetry, seconds=120, warm=20):
    run = run.resolve(strict=True)
    requests = json.loads(requests.read_text())["rows"]
    if not requests or any(row["http"] != 200 for row in requests):
        raise ValueError("Expected successful timed requests")
    if seconds < 30 or seconds % 30 or warm < 0:
        raise ValueError("Duration must be a positive multiple of 30 seconds and warmup must be nonnegative")

    with telemetry.open() as stream:
        complete = csv.reader(line for line in stream if line.endswith("\n"))
        gpu = [
            (datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), int(row[1]), list(map(float, row[2:])))
            for row in complete if len(row) == 9
        ]
    events = logs(run)
    broker = rows(run / "broker.jsonl")
    offsets = [[dict(elapsed=t) for t in range(0, seconds + 1, 30)] for _ in RANKS]
    epoch, points = embedded_points(events, broker, offsets, max(r["end_ns"] for r in requests), warm)
    window = mining_result(points, gpu)
    for label, accepted in (("accepted", True), ("rejected", False)):
        window[label] = [
            sum(e["event"] == "share" and e["accepted"] == accepted
                and window["start_utc"] <= e["time"] <= window["end_utc"] for e in events[rank])
            for rank in RANKS
        ]

    handoff = timings(requests, broker, events)
    if any(metric["n"] != len(requests) for metric in handoff.values()):
        raise ValueError("Expected a complete cut-in and resumption for every timed request")
    startup = [next(e for e in events[rank] if e["event"] == "native_rank") for rank in RANKS]
    return dict(run=run.name, epoch=epoch, window=window, handoff=handoff,
                profiles=[e.get("slice_profile", "unreported") for e in startup])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "requests", "telemetry", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--warm", type=int, default=20, help="Idle seconds after the final handoff before measuring")
    args = parser.parse_args()
    write(args.out, sample(args.run, args.requests, args.telemetry, args.seconds, args.warm))
