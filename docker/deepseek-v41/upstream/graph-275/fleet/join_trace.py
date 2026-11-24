"""Join a scheduled trace to independent four-worker cursor and local proof receipts."""
import argparse
import asyncio
import json
import math
import statistics
from pathlib import Path

from combine import correlate
from miners import snapshot, tiles


def summarize(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(n=len(ordered), median=statistics.median(ordered),
                p95=ordered[math.ceil(.95*len(ordered))-1], maximum=ordered[-1])

TILE_MACS = 160 * 256 * 65536


def bracket(samples, instant):
    before = next((s for s in reversed(samples) if s["monotonic_ns"] <= instant), None)
    after = next((s for s in samples if s["began_ns"] >= instant), None)
    if before is None or after is None:
        raise ValueError(f"no full worker snapshots bracketing {instant}")
    return before, after


def bounds(samples, start_ns, end_ns):
    if end_ns <= start_ns:
        raise ValueError("nonpositive horizon")
    earlier_start, later_start = bracket(samples, start_ns)
    earlier_end, later_end = bracket(samples, end_ns)

    def delta_tiles(left, right):
        return [max(0, tiles(b["completed_slices"]) - tiles(a["completed_slices"]))
                for a,b in zip(left["workers"], right["workers"], strict=True)]
    low = delta_tiles(later_start, earlier_end)
    high = delta_tiles(earlier_start, later_end)
    duration = (end_ns - start_ns)/1e9
    return dict(start_ns=start_ns, end_ns=end_ns, duration_s=duration,
                boundary_uncertainty_ms=[(s1["monotonic_ns"]-s0["began_ns"])/1e6 for s0,s1 in
                    [(earlier_start,later_start),(earlier_end,later_end)]],
                lower_tiles_by_gpu=low, upper_tiles_by_gpu=high,
                lower_tmac_s=sum(low)*TILE_MACS/duration/1e12, upper_tmac_s=sum(high)*TILE_MACS/duration/1e12)


def join(trace, events, run):
    samples = [e["state"] for e in events if e["type"] in ("start","workers","stop","live_receipt")]
    if events[-1]["type"] not in ("stop","live_receipt"):
        raise ValueError("observer needs a stop or explicitly sampled live receipt")
    if not trace.get("complete",True):
        raise ValueError("client trace is still running")
    if any(len(s["workers"]) != 4 or not all(w["watched"] and not w["faulted"] for w in s["workers"])
           or not all(s["broker"]["watched"]) for s in samples):
        raise ValueError("four-rank worker supervision interrupted")
    if any(any(x["completed_slices"]>y["completed_slices"] for x,y in zip(a["workers"], b["workers"]))
           for a,b in zip(samples,samples[1:])):
        raise ValueError("worker cursor reset")
    requests = []
    for ordinal, trial in enumerate(trace["runs"]):
        req = trial["request"]
        requests.append(dict(trial=ordinal,start_monotonic_ns=req["start_monotonic_ns"],
                             end_monotonic_ns=req["stream_closed_monotonic_ns"],wall_ms=req["stream_wall_s"]*1000))
    result = correlate(events, requests)
    for receipt, trial in zip(result["requests"], trace["runs"], strict=True):
        req = trial["request"]
        first = req["first_token_monotonic_ns"]
        last = req["last_token_monotonic_ns"]
        decoder = bounds(samples, first, last)
        # A parked snapshot before the first token and a post-last-token release
        # timestamp bound the full stream, including very short outputs that
        # would span few internal samples. Observed broker releases invalidate it.
        before = [s for s in samples if s['monotonic_ns']<=first]
        before = before[-1] if before else None
        through = [s for s in samples if first<s['monotonic_ns'] and s['began_ns']<last]
        after = next((s for s in samples if s['began_ns']>=last), None)
        def held(s):
            return s is not None and all(s["broker"]["drained"]) and all(w["holds"] and not w["gpu_busy"] for w in s["workers"])
        def cursor(s):
            return tuple(w["completed_slices"] for w in s["workers"])
        release_times_ok = after is not None and before is not None and all(
            y["release_ns"]==x["release_ns"] or y["release_ns"]>=last
            for x,y in zip(before["workers"],after["workers"],strict=True))
        no_mid_release = all(all(e["state"]["drained"]) for e in events if e["type"]=="broker"
                             and first<=e["monotonic_ns"]<last)
        confirmed = (held(before) and all(held(s) and cursor(s)==cursor(before) for s in through if s["monotonic_ns"]<=last)
                     and release_times_ok and no_mid_release)
        receipt.update(decode_direct_parked_through_last_token=confirmed, decode_cursor_bounds=decoder,
                       pair=trial["pair"], arm=trial["arm"])
    start, end = (trace[k] for k in ("requested_horizon_start_monotonic_ns","requested_horizon_end_monotonic_ns"))
    direct = bounds(samples, start, end)
    timeline = [(events[0]["state"]["monotonic_ns"], events[0]["state"]["broker"])]
    timeline.extend((e["monotonic_ns"],e["state"]) for e in events if e["type"]=="broker")
    timeline.append((events[-1]["state"]["monotonic_ns"],events[-1]["state"]["broker"]))
    idle_ns = sum(max(0, min(end,hi)-max(start,lo)) for (lo,br),(hi,_) in zip(timeline,timeline[1:])
                  if not br["serving"] and not br["claims"] and all(br["idle"]))
    direct["broker_permitted_idle_s"] = idle_ns/1e9
    direct["broker_permitted_idle_fraction"] = idle_ns/(end-start)
    prewarm = json.loads((run/"prewarm.json").read_text())
    direct["idle_calibration_tmac_s"] = prewarm["calibration"]["total_tmac_s"]
    direct["idle_calibration_ratio_bounds"] = [direct[k]/direct["idle_calibration_tmac_s"] for k in ("lower_tmac_s","upper_tmac_s")]
    direct["pure_idle_rate_fraction_while_permitted_bounds"] = (
        [v/direct["broker_permitted_idle_fraction"] for v in direct["idle_calibration_ratio_bounds"]]
        if idle_ns else None)
    ledger = run/"peer"/"ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    eligible = [r for r in records if start<=r["received_ns"]<end]
    if len({row["sha256"] for row in records})!=len(records):
        raise ValueError("duplicates in verified share ledger")
    target = prewarm["target_hex"]
    count = len(eligible)
    direct.update(unique_local_verified_shares_by_rank=[sum(r["rank"]==i for r in eligible) for i in range(4)],
                  unique_local_verified_shares=count, pool_base_target=target,
                  local_share_work_th_s=count*(2**256/(int(target,16)+1))/direct["duration_s"]/1e12,
                  share_time_basis="local proof receive time, deduplicated, independently verified; CPU submission can follow a completed GPU slice",
                  external_pool_acks=0)
    timing = {k:summarize([r[k] for r in result["requests"] if k in r]) for k in
              ("http_send_to_observed_all_drained_ms","first_release_to_all_observed_gpu_entry_ms",
               "http_return_to_last_observed_gpu_entry_ms")}
    timing.update(requests=len(requests), parked_decode=sum(bool(r["decode_direct_parked_through_last_token"]) for r in result["requests"]),
                  stationary_cursors=sum(bool(r.get("stationary_four_cursors")) for r in result["requests"]))
    result.update(fleet_run=str(run), direct_horizon=direct, trace_mode=trace["trace_mode"], switching_summary=timing,
                  observer_closed=events[-1]["type"]=="stop", direct_work_basis="complete 160x256x65536 K780 output tiles across four monotonic worker cursors; lower/upper totals bracket the exact requested client horizon")
    return result


def join_handbook(trace, events):
    if not trace.get("complete") or events[-1]["type"] != "stop":
        raise ValueError("both handbook and observer must be final")
    requests=[]
    for row in trace["runs"]:
        start=round(row["start_monotonic"]*1e9)
        end=round(row["end_monotonic"]*1e9)
        requests.append(dict(trial=row["target_input_tokens"],start_monotonic_ns=start,end_monotonic_ns=end,
                             wall_ms=row["e2e_s"]*1000))
    result=correlate(events,requests)
    states=[e["state"] for e in events if e["type"] in ("start","stop","workers")]
    for row,req,out in zip(trace["runs"],requests,result["requests"],strict=True):
        first=req["start_monotonic_ns"]+round(row["ttft_s"]*1e9)
        end=req["end_monotonic_ns"]
        out["input_tokens"]=row["input_tokens"]
        out["output_tokens"]=row["output_tokens"]
        # These handbook streams timestamp the first client-visible delta and
        # the terminal response. Device lease can resume before terminal return.
        interior=[s for s in states if first<=s["began_ns"] and s["monotonic_ns"]<=end]
        held=[s for s in interior if all(s["broker"]["drained"])
              and all(w["watched"] and not w["faulted"] and w["holds"] and not w["gpu_busy"] for w in s["workers"])]
        out["client_output_interval_held_samples"]=len(held)
        out["client_output_interval_total_samples"]=len(interior)
        out["client_output_interval_held_four_cursor_sets"]=len({tuple(w["completed_slices"] for w in s["workers"]) for s in held})
        out["first_client_delta_to_terminal_s"]=(end-first)/1e9
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for argument in ("trace","observer","fleet_run","output"):
        p.add_argument(argument,type=Path)
    p.add_argument("--live",action="store_true",help="emit a provisional receipt from a still running observer")
    p.add_argument("--handbook",action="store_true",help="correlate original long-prompt handbook streaming timestamps")
    a=p.parse_args()
    with a.observer.open() as stream:
        events=[json.loads(row) for row in stream if row.endswith("\n")]
    if a.live and events[-1]["type"]!="stop":
        state=asyncio.run(snapshot())
        events.append(dict(type="live_receipt",monotonic_ns=state["monotonic_ns"],state=state))
    trace=json.loads(a.trace.read_text())
    data=join_handbook(trace,events) if a.handbook else join(trace,events,a.fleet_run)
    with a.output.open("x") as out:
        json.dump(data,out,indent=2)
        out.write("\n")
