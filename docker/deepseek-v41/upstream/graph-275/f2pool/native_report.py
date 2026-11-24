"""Reduce the native run to aggregate, source-attributed verification and timing evidence."""
import argparse
import csv
from datetime import datetime
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
        candidates.append(dict(epoch=epoch, common_seconds=common,
                               start_utc_epoch=max(p[0]["time"] for p in points),
                               end_utc_epoch=min(p[-1]["time"] for p in points),
                               tmac_s_by_rank=rates,total_tmac_s=sum(rates)))
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


def idle_hardware(rows, idle):
    samples = {i:[] for i in RANKS}
    for row in rows:
        if len(row) != 7:
            continue
        when = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        if idle["start_utc_epoch"] <= when <= idle["end_utc_epoch"]:
            samples[int(row[1])].append(list(map(float, row[2:])))
    if any(len(values) < 2 for values in samples.values()):
        return None
    return {rank:dict(samples=len(values),mean_util_pct=statistics.mean(v[0] for v in values),
                      max_temp_c=max(v[1] for v in values),mean_power_w=statistics.mean(v[2] for v in values),
                      sw_thermal_counter_us=values[-1][3]-values[0][3],
                      hw_thermal_counter_us=values[-1][4]-values[0][4]) for rank,values in samples.items()}


def paired_summary(rows):
    held = {r["index"]:r for r in rows if r["mode"] == "held"}
    enabled = {r["index"]:r for r in rows if r["mode"] == "enabled"}
    if not held or len(rows) != 2*len(held) or held.keys() != enabled.keys():
        raise ValueError("Incomplete native idle comparison")
    changes = []
    for i,a in held.items():
        b = enabled[i]
        if (a["prompt_sha256"] != b["prompt_sha256"] or a["budget"] != b["budget"]
                or a["cached_tokens"] != 0 or b["cached_tokens"] != 0):
            raise ValueError("Warm or unmatched native idle comparison")
        changes.append(100 * (b["client_decode_tps"]/a["client_decode_tps"] - 1))
    return dict(pairs=len(held), median_pair_decode_change_pct=statistics.median(changes),
                range_pair_decode_change_pct=[min(changes),max(changes)],
                output_hash_matches=sum(a["output_sha256"] == enabled[i]["output_sha256"] for i,a in held.items()),
                arms={name:{k:statistics.median(r[k] for r in group.values())
                            for k in ("client_decode_tps","ttft_ms")} for name,group in (("held",held),("enabled",enabled))})


def phase_summary(rows, baseline):
    phases = {}
    for phase in ("prefill","decode"):
        seconds = sum(r["seconds"][phase] for r in rows)
        macs = [sum(r["mac_equivalent"][phase][i] for r in rows) for i in (0,1)]
        rates = [n/seconds/1e12 for n in macs]
        phases[phase] = dict(seconds=seconds, mac_equivalent=macs, tmac_s=rates,
                             pct_of_standalone=[v*100/baseline for v in rates],
                             tickets=sum(r["scanned_tickets"][phase] for r in rows))
    return dict(requests=len(rows), work=phases,
                serving=[dict(input_range=[min(r["input_tokens"] for r in group),max(r["input_tokens"] for r in group)],
                              requests=len(group), **{k:statistics.median(r[k] for r in group) for k in
                                      ("server_prefill_tps","client_decode_tps","ttft_ms")})
                         for group in ([r for r in rows if r["input_tokens"] < 2048],
                                       [r for r in rows if r["input_tokens"] >= 2048]) if group])


def historical(run, graph_dir, baseline):
    from workload import scan_summary
    from hour_metrics import prefill_bounds

    doc = read(run/"handbook.json")
    old = [read(graph_dir/"measurements"/name)["runs"] for name in ("switch-handbook.json","joined-handbook.json")]
    if not doc["complete"]:
        raise ValueError("Historical replay incomplete")
    rows = []
    for row, control, previous in zip(doc["runs"], *old, strict=True):
        if any(row["prompt_sha256"] != other["prompt_sha256"] or row["output_budget"] != other["output_budget"]
               or other["cached_tokens"] != 0 for other in (row, control, previous)):
            raise ValueError("Unmatched historical replay")
        a,b = row["projection_before"],row["projection_after"]
        work = scan_summary(a,b,row["graph_delta_by_rank"])
        bounds = prefill_bounds(a,b,work)
        decode = work["reconciled_mac_equivalent"]["decode"]
        rows.append(dict(input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
                         effective_prefill_tps=row["client_effective_prefill_tps"], ttft_s=row["ttft_s"],
                         decode_tps=row["client_decode_tps"], previous_switch_decode_tps=control["client_decode_tps"],
                         previous_joined_decode_tps=previous["client_decode_tps"], previous_switch_ttft_s=control["ttft_s"],
                         previous_joined_ttft_s=previous["ttft_s"],
                         decode_delta_pct=100*(row["client_decode_tps"]/control["client_decode_tps"]-1),
                         prompt_sha256=row["prompt_sha256"],
                         prefill_pct_of_standalone=[100*v/1e12/row["ttft_s"]/baseline for v in bounds],
                         decode_pct_of_standalone=100*decode/1e12/(row["e2e_s"]-row["ttft_s"])/baseline))
    return rows


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
    graph_dir=Path(__file__).resolve().parents[1]
    baseline=read(graph_dir/"direct-calibration.json")["calibration"]
    last=max(row["monotonic_ns"] for batch in events.values() for row in batch if row["event"]=="native_work")
    sustained=idle_rate(events,broker,last)
    sustained["pct_of_standalone"]=100*sustained["total_tmac_s"]/baseline["total_tmac_s"]
    sustained["hardware"]=idle_hardware(power_rows,sustained)
    additional={}
    for filename,key in (("proof-check","independent_verification"),("overlap","concurrent_disconnect")):
        if (run/(filename+".json")).exists():
            additional[key]=read(run/(filename+".json"))
    if (run/"paired.json").exists():
        additional["same_build"]=paired_summary(read(run/"paired.json")["rows"])
    if (run/"phases.json").exists():
        additional["phase_benchmark"]=phase_summary(read(run/"phases.json")["rows"],baseline["total_tmac_s"])
    if (run/"handbook.json").exists():
        additional["historical"]=historical(run,graph_dir,baseline["total_tmac_s"])
    pool_sessions=[read(p) for p in (run/"receipts").glob("gateway*.json")]
    return dict(ownership=read(run/"ownership.json"),stage=read(run/"staged.json"),stock=bench(stock),joined=bench(joined),
                matched_token_budgets=True,output_hash_matches=sum(a["output_sha256"]==b["output_sha256"] for a,b in zip(stock,joined)),
                transition_requests=len(stress),transition_http_200=sum(r["http"]==200 for r in stress),
                timings=timings(joined+stress,broker,events), idle=idle_rate(events,broker,after["monotonic_ns"]),
                direct_pool={i:live_shares(events[i]) for i in RANKS},gpu_job_setup_ms=quantiles(setup),
                tickets_by_rank=phase,inference_pool={k:after["pool"][k]-before["pool"][k] for k in ("submitted","accepted","rejected")},
                qualification_inference_pool={k:sum(int(session[k]) for session in pool_sessions)
                                             for k in ("submitted","accepted","rejected","unknown")},
                comparison_standalone=dict(seconds=baseline["seconds"],total_tmac_s=baseline["total_tmac_s"]),
                sustained_idle=sustained, **additional,
                peak_c_by_rank=temps,profile=read(run/"native-profile.json"),native_failures=sum(
                    row["event"]=="native_error" for es in events.values() for row in es))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run",type=Path)
    cli.add_argument("out",type=Path)
    args=cli.parse_args()
    args.out.write_text(json.dumps(report(args.run),indent=2)+"\n")
