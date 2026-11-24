"""Reduce the native run to aggregate, source-attributed verification and timing evidence."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

TILE_MACS = 160*256*65536
RANKS = range(4)


def read(path):
    return json.loads(path.read_text())


def lines(path):
    with path.open() as file:
        return [json.loads(line) for line in file if line.endswith("\n")]


def quantiles(values):
    if not values:
        return dict(n=0)
    values = sorted(values)
    return dict(n=len(values), median=statistics.median(values),
                p95=values[math.ceil(len(values)*.95)-1], p99=values[math.ceil(len(values)*.99)-1], max=values[-1])


def live_shares(events):
    pending = {}
    replies = set()
    accepted = rejected = unknown = 0
    for row in events:
        event=row["event"]
        if event=="submit_unknown":
            unknown += row["count"]
        if event not in ("share_submitted","share"):
            continue
        key=(row["connection_id"],row["request_id"])
        if event=="share_submitted":
            if key in pending:
                raise ValueError("Repeated submission on the same pool connection")
            pending[key]=row["proof_sha256"]
        else:
            if key in replies or pending.get(key)!=row["proof_sha256"]:
                raise ValueError("Pool acknowledgement failed submission attribution")
            replies.add(key)
            if row["accepted"]:
                accepted+=1
            else:
                rejected+=1
    return dict(submitted=len(pending),accepted=accepted,rejected=rejected,unknown=unknown,
                unacknowledged=len(pending)-len(replies))


def idle_rate(events, broker, cutoff):
    # Require an uninterrupted four-rank idle epoch. Each counter interval must be wholly inside
    # the common epoch; independent 1 s sampling introduces a small skew between the four spans.
    released={x["epoch"]:x["monotonic_ns"] for x in broker if x["event"]=="native_release"}
    closed={x["epoch"]-1:x["monotonic_ns"] for x in broker if x["event"]=="native_demand"}
    candidates=[]
    for epoch, start in released.items():
        finish=min(closed.get(epoch,cutoff),cutoff)
        points=[[r for r in events[i] if r["event"]=="native_work" and r["epoch"]==epoch and
                 r["graphs"]==139 and start <= r["monotonic_ns"] <= finish] for i in RANKS]
        if any(len(p)<2 for p in points):
            continue
        common=(min(p[-1]["monotonic_ns"] for p in points)-max(p[0]["monotonic_ns"] for p in points))/1e9
        if common<=0:
            continue
        rates=[(p[-1]["tiles"]-p[0]["tiles"])*TILE_MACS/(p[-1]["monotonic_ns"]-p[0]["monotonic_ns"])/1e3 for p in points]
        candidates.append(dict(common_seconds=common,tmac_s_by_rank=rates,total_tmac_s=sum(rates)))
    return max(candidates,key=lambda c:c["common_seconds"]) if candidates else None


def bench(rows):
    return dict(requests=len(rows),http_200=sum(r["http"]==200 for r in rows),
                outputs=sum(r["output_tokens"] for r in rows),
                median_client_decode_tps=statistics.median(r["client_decode_tps"] for r in rows),
                median_server_decode_tps=statistics.median(r["native_server_decode_tps"] for r in rows if r.get("native_server_decode_tps")),
                ttft_ms=quantiles([r["ttft_ms"] for r in rows]),
                per_length=[dict(input_range=[min(r["input_tokens"] for r in rows if r["notes"]==notes),
                                               max(r["input_tokens"] for r in rows if r["notes"]==notes)],
                                 median_ttft_ms=statistics.median(r["ttft_ms"] for r in rows if r["notes"]==notes),
                                 median_uncached_input_per_ttft=statistics.median(
                                     (r["input_tokens"]-r["cached_tokens"])*1000/r["ttft_ms"]
                                     for r in rows if r["notes"]==notes and r["cached_tokens"] is not None))
                            for notes in sorted({r["notes"] for r in rows})])


def timings(requests, broker, events):
    demand={e["epoch"]:e["monotonic_ns"] for e in broker if e["event"]=="native_demand"}
    drained={e["epoch"]:e["monotonic_ns"] for e in broker if e["event"]=="native_drained"}
    release={e["epoch"]:e["monotonic_ns"] for e in broker if e["event"]=="native_release"}
    seen={i:{e["epoch"]:e["gpu_seen_ns"] for e in events[i] if e["event"]=="native_reentry"} for i in RANKS}
    measures={k:[] for k in ("demand_to_all_drained_ms","http_start_to_all_drained_ms",
                "eligible_to_first_gpu_ms","eligible_to_all_gpus_ms","client_terminal_to_first_gpu_ms")}
    for req in requests:
        start,end=req["start_ns"],req["end_ns"]
        # Require one uniquely paired admission edge before the first output token.
        edges=[ep for ep,t in demand.items() if start<=t<=start+req["ttft_ms"]*1e6 and ep in drained]
        if len(edges)!=1:
            continue
        ep=edges[0]
        measures["demand_to_all_drained_ms"].append((drained[ep]-demand[ep])/1e6)
        measures["http_start_to_all_drained_ms"].append((drained[ep]-start)/1e6)
        next_epoch = ep+1
        if next_epoch not in release:
            continue
        times=[seen[i][next_epoch] for i in RANKS if next_epoch in seen[i]]
        if len(times)==4:
            measures["eligible_to_first_gpu_ms"].append((min(times)-release[next_epoch])/1e6)
            measures["eligible_to_all_gpus_ms"].append((max(times)-release[next_epoch])/1e6)
            measures["client_terminal_to_first_gpu_ms"].append((min(times)-end)/1e6)
    return {k:quantiles(v) for k,v in measures.items()}


def report(run):
    stock=read(run/"stock.json")["rows"]
    joined=read(run/"native.json")["rows"]
    stress=read(run/"transitions.json")["rows"]
    if [(r["prompt_sha256"],r["budget"],r["input_tokens"]) for r in stock] != [
            (r["prompt_sha256"],r["budget"],r["input_tokens"]) for r in joined]:
        raise ValueError("Production comparison used different inputs or output budgets")
    before,after=(read(run/name) for name in ("before-public.json","after-public.json"))
    events={int(p.name.split("-")[1]):lines(p) for p in (run/"logs").glob("native-*.jsonl")}
    broker=lines(run/"broker.jsonl")
    phase={phase:{i:after["phases"][str(i)][phase+"_scan"].get("tickets_scanned",0)-
                       before["phases"][str(i)][phase+"_scan"].get("tickets_scanned",0) for i in RANKS}
           for phase in ("prefill","decode")}
    setup=[r["gpu_setup_us"]/1000 for es in events.values() for r in es if r["event"]=="native_job"]
    power_rows=list(csv.reader((run/"gpu.csv").read_text().splitlines(),skipinitialspace=True))
    temps={i:max(int(r[3]) for r in power_rows if len(r)==7 and int(r[1])==i) for i in RANKS}
    return dict(ownership=read(run/"ownership.json"),stage=read(run/"staged.json"),stock=bench(stock),joined=bench(joined),
                matched_token_budgets=True,output_hash_matches=sum(a["output_sha256"]==b["output_sha256"] for a,b in zip(stock,joined)),
                transition_requests=len(stress),transition_http_200=sum(r["http"]==200 for r in stress),
                timings=timings(joined+stress,broker,events), idle=idle_rate(events,broker,after["monotonic_ns"]),
                direct_pool={i:live_shares(events[i]) for i in RANKS},gpu_job_setup_ms=quantiles(setup),
                tickets_by_rank=phase,inference_pool={k:after["pool"][k]-before["pool"][k] for k in ("submitted","accepted","rejected")},
                peak_c_by_rank=temps,profile=read(run/"native-profile.json"),native_failures=sum(
                    row["event"]=="native_error" for es in events.values() for row in es))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run",type=Path)
    cli.add_argument("out",type=Path)
    args=cli.parse_args()
    args.out.write_text(json.dumps(report(args.run),indent=2)+"\n")
