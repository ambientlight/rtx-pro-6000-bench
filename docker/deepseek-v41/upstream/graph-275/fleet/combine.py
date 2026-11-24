#!/usr/bin/env python3
"""Correlate clock-monotonic client intervals with four actual TP rank and worker transitions."""
import argparse
import json
from pathlib import Path


def correlate(events,requests):
    broker=[event for event in events if event["type"]=="broker"]
    snapshots=[e["state"] for e in events if e["type"] in ("start","stop","workers")]
    results=[]
    for i,request in enumerate(requests):
        start,end=request["start_monotonic_ns"],request["end_monotonic_ns"]
        upper=requests[i+1]["start_monotonic_ns"] if i+1<len(requests) else events[-1]["monotonic_ns"]
        before=[s for s in snapshots if s["monotonic_ns"] < start]
        drained=next((e for e in broker if start <= e["monotonic_ns"] < upper and all(e["state"]["drained"])),None)
        release=next((s for s in snapshots if start<s["monotonic_ns"]<upper
                      and not s["broker"]["claims"] and all(s["broker"]["idle"])
                      and all(w["release_ns"]>=start and w["first_gpu_seen_ns"]>=w["release_ns"] for w in s["workers"])),None)
        result=dict(trial=request["trial"],http_wall_ms=request["wall_ms"],saw_drain=drained is not None,saw_all_resumed=release is not None)
        if before:
            result["active_mining_before_http"]=not before[-1]["broker"]["serving"] and all(w["watched"] and not w["holds"] and w["completed_slices"] > 0 for w in before[-1]["workers"])
        if drained:
            result["http_send_to_observed_all_drained_ms"]=(drained["monotonic_ns"]-start)/1e6
        if release:
            workers=release["workers"]
            release_at=min(w["release_ns"] for w in workers)
            last_seen=max(w["first_gpu_seen_ns"] for w in workers)
            result.update(first_release_to_all_observed_gpu_entry_ms=(last_seen-release_at)/1e6,
                          http_return_to_last_observed_gpu_entry_ms=(last_seen-end)/1e6,
                          worker_release_to_entry_ms=[(w["first_gpu_seen_ns"]-w["release_ns"])/1e6 for w in workers],
                          healthy_all=all(not w["faulted"] and w["watched"] for w in workers))
            held=[s for s in snapshots if start<s["monotonic_ns"]<release_at and all(s["broker"]["drained"]) and all(w["holds"] and not w["gpu_busy"] for w in s["workers"])]
            cursors={tuple(w["completed_slices"] for w in s["workers"]) for s in held}
            result.update(confirmed_hold_samples=len(held),stationary_four_cursors=len(held)>=2 and len(cursors)==1)
        results.append(result)
    return dict(requests=results, observer='persistent broker STATUS at nominal 1 ms; GPU entry is timestamped by the worker seeing its GPU-mapped start flag; HTTP return includes transport',
                polling_summary=next(({k:e[k] for k in ("samples","maximum_poll_gap_ns")} for e in events if e["type"]=="stop"),None))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("observer","proof_receipt","output"):
        p.add_argument(name,type=Path)
    args=p.parse_args()
    data=correlate([json.loads(line) for line in args.observer.read_text().splitlines()],
                   json.loads(args.proof_receipt.read_text())["requests"])
    args.output.write_text(json.dumps(data,indent=2)+"\n")
