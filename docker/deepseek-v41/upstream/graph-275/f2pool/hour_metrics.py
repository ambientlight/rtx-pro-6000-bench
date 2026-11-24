"""Passively measure a real-user joined Pearl trial using only aggregate telemetry."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
import urllib.request

from workload import RejectRedirect, graph, scan_summary

RANKS = ("0", "1", "2", "3")
HIST = {"time_to_first_token_seconds", "request_time_per_output_token_seconds",
        "inter_token_latency_seconds", "e2e_request_latency_seconds"}
COUNTERS = {"num_requests_total", "num_aborted_requests_total", "prompt_tokens_total",
            "generation_tokens_total", "cached_tokens_total", "uncached_prompt_tokens_histogram_sum",
            "spec_verify_calls_total", "num_retracted_requests_total"}
GPUS = ("index", "temperature.gpu", "power.draw", "power.limit", "clocks.current.sm",
        "clocks.current.memory", "clocks_event_reasons_counters.sw_thermal_slowdown",
        "clocks_event_reasons_counters.hw_thermal_slowdown")
BASE_RATE = json.loads((Path(__file__).resolve().parents[1]/"direct-calibration.json").read_text())["calibration"]["total_tmac_s"]
PHASE = ("submitted","accepted","rejected","unknown","accepted_expected_macs")
IDLE = ("connected","submitted","accepted","rejected","unknown","accepted_work_macs","target","cooling","cooling_seconds")
LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}), RejectRedirect())


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    temp.replace(path)


def parse_metrics(text):
    totals, graphs, hist, observed = {}, {r:{} for r in RANKS}, {}, set()
    active, http = graph.activity_metrics(text)
    cohorts = {s:{"tokens":{},"hist":{}} for s in ("streaming","nonstreaming")}
    live, response_status, responses_active = {}, {}, 0
    cohort_counts = {"num_requests_total","prompt_tokens_total","generation_tokens_total"}
    realtime = {"prefill_compute","prefill_cache","decode"}
    for line in text.splitlines():
        if not line.startswith("sglang:"):
            continue
        try:
            sample, raw = line.rsplit(None, 1)
            metric = sample[7:].split("{",1)[0]
            name, _, suffix = metric.rpartition("_")
            if (metric not in COUNTERS and metric != "cuda_graph_passes_total"
                    and metric not in ("http_requests_active","http_responses_total","realtime_tokens_total","prefill_effective_tokens_total")
                    and name not in HIST and name != "per_stage_req_latency_seconds"):
                continue
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Nonfinite or negative metric")
            labels = dict(re.findall(r'(\w+)="([^"]*)"', sample))
            if metric in ("http_responses_total","http_requests_active"):
                if labels.get("endpoint")=="/v1/responses" and labels.get("method")=="POST":
                    if metric=="http_requests_active":
                        responses_active+=int(value)
                    elif re.fullmatch(r"[1-5][0-9]{2}",labels.get("status_code","")):
                        code=labels["status_code"]
                        response_status[code]=response_status.get(code,0)+int(value)
                continue
            if metric == "cuda_graph_passes_total":
                rank, mode = labels["tp_rank"], labels["mode"]
                if rank in RANKS and mode == "decode_cuda_graph":
                    graphs[rank][mode] = graphs[rank].get(mode, 0) + int(value)
                continue
            if labels.get("tp_rank","0") != "0":
                continue
            if metric in ("realtime_tokens_total","prefill_effective_tokens_total"):
                mode = labels.get("mode")
                selected = realtime if metric=="realtime_tokens_total" else {"input","device_hit","host_hit","storage_hit"}
                if mode in selected:
                    label = metric+"."+mode
                    live[label] = live.get(label,0)+int(value)
                continue
            stream_label = {"true":"streaming","false":"nonstreaming"}.get(labels.get("is_streaming"))
            if stream_label and metric in cohort_counts:
                ct=cohorts[stream_label]["tokens"]
                ct[metric]=ct.get(metric,0)+value
            if stream_label and name in HIST and suffix in ("sum","count","bucket"):
                ch=cohorts[stream_label]["hist"].setdefault(name,{})
                bucket = None
                if suffix=="bucket":
                    numeric = float(labels["le"])
                    if math.isfinite(numeric):
                        bucket = "le_"+str(numeric)
                field = bucket if suffix=="bucket" else suffix
                if field:
                    ch[field]=ch.get(field,0)+value
            if metric in COUNTERS:
                observed.add(metric)
                totals[metric] = totals.get(metric, 0) + value
                continue
            if name == "per_stage_req_latency_seconds":
                if labels.get("stage") != "prefill_forward" or suffix not in ("count","sum"):
                    continue
                key = "prefill_forward_seconds"
            elif name in HIST and suffix in ("sum","count","bucket"):
                if name in ("time_to_first_token_seconds","request_time_per_output_token_seconds") and labels.get("is_streaming") == "false":
                    continue
                key = name
            else:
                continue
            if suffix == "bucket":
                bound = float(labels["le"])
                if not math.isfinite(bound):
                    continue
                suffix = "le_" + str(bound)
            data = hist.setdefault(key,{})
            data[suffix] = data.get(suffix, 0) + value
        except (KeyError, ValueError) as err:
            raise ValueError("Invalid recognized SGLang metric") from err
    if "num_requests_total" not in observed:
        raise ValueError("Missing completed request metric")
    endpoint_totals = {}
    for (endpoint, method), count in http:
        endpoint_totals[endpoint] = endpoint_totals.get(endpoint, 0) + count
    return dict(active=active,http=endpoint_totals,totals=totals,hist=hist,graphs=graphs,
                cohorts=cohorts,scheduler_progress=live,responses_status=response_status,
                responses_active=responses_active)


def server(args):
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/metrics",
        headers={"Authorization":"Bearer " + args.key_file.read_text().strip()})
    with LOOPBACK.open(req, timeout=2) as stream:
        return parse_metrics(stream.read().decode())


def hardware():
    raw = subprocess.check_output(["nvidia-smi","--query-gpu="+",".join(GPUS),
                                   "--format=csv,noheader,nounits"], text=True, timeout=2,
                                  stderr=subprocess.DEVNULL)
    result = []
    for line in raw.splitlines():
        fields = line.split(",")
        if len(fields) != len(GPUS):
            raise ValueError("Missing GPU metric")
        result.append([float(x) if x.strip() not in ("[N/A]","N/A") else None for x in fields])
    if [g[0] for g in result] != [0,1,2,3]:
        raise ValueError("Missing GPU index")
    return result


def pool(path):
    raw = read(path)
    job = raw.get("job") or {}
    phases = {phase:{k:row[k] for k in PHASE if k in row} for phase,row in raw.get("phases",{}).items()
              if phase in ("prefill","decode")}
    return {**{k:raw.get(k) for k in ("run_id","connected","authorized_clients","armed","submitted",
            "accepted","rejected","unknown","stale","pool_stale","notifications","invalid","duplicate")},
            "target":job.get("target"), "phases":phases}


def idle(path):
    raw = read(path/"status.json")
    ranks, workers = raw["ranks"], raw["workers"]
    if len(ranks) != 4 or len(workers) != 4:
        raise ValueError("Missing direct worker status")
    sources = []
    for rank in ranks:
        row = {k:rank.get(k) for k in IDLE}
        work = rank.get("work")
        row["work"] = {k:work[k] for k in ("macs","monotonic_ns") if k in work} if work else None
        sources.append(row)
    return dict(source_monotonic_ns=raw["monotonic_ns"],ranks=sources,
                workers=[{k:w.get(k) for k in ("completed_slices","holds","watched","faulted",
                         "first_gpu_seen_ns","release_ns")} for w in workers],
                broker={k:raw["broker"].get(k) for k in ("serving","claims","idle","drained","watched")})


def take(args):
    sample = {"began_ns":time.monotonic_ns(), "utc_epoch":time.time(), "errors":{}}
    for key, fn, params in (("server",server,(args,)),("gateway",pool,(args.gateway_receipt,)),
            ("idle",idle,(args.fleet_dir,)),("projection",graph.projection_snapshot,(args.shape_dir,)),
            ("gpu",hardware,())):
        try:
            sample[key] = fn(*params)
        except (OSError, TimeoutError, ValueError, KeyError, subprocess.SubprocessError) as e:
            sample["errors"][key] = type(e).__name__
    sample["end_ns"] = time.monotonic_ns()
    return sample


def complete(row):
    return not row["errors"] and len(row["projection"].get("counters",{})) == 4


def counter_diff(old, new):
    diff = {k:new.get(k,0)-old.get(k,0) for k in old.keys()|new.keys()}
    if any(v < -1e-8 for v in diff.values()):
        raise ValueError("Monotonic counter reset")
    return diff


def histogram(before, after):
    d = counter_diff(before, after)
    n, total = d.get("count",0), d.get("sum",0)
    bounds = sorted((float(k[3:]),v) for k,v in d.items() if k.startswith("le_"))
    return dict(observations=int(n),sum_s=total,mean_s=total/n if n else None,
                p95_bucket_upper_s=next((bound for bound,count in bounds if n and count>=math.ceil(.95*n)),None))


def by_cohort(before,after,quiet_boundaries,aborted):
    cohorts={}
    for which,final in after.get("cohorts",{}).items():
        init=before.get("cohorts",{}).get(which,{"tokens":{},"hist":{}})
        tokens=counter_diff(init["tokens"],final["tokens"])
        h={name:histogram(init["hist"].get(name,{}),row) for name,row in final["hist"].items()}
        n=int(tokens.get("num_requests_total",0))
        generated=int(tokens.get("generation_tokens_total",0))
        ttft,e2e,tpot=(h.get(name,{}) for name in ("time_to_first_token_seconds","e2e_request_latency_seconds","request_time_per_output_token_seconds"))
        matched=bool(quiet_boundaries and not aborted and n and ttft.get("observations")==e2e.get("observations")==n)
        duration=e2e.get("sum_s",0)-ttft.get("sum_s",0)
        # Samples are from the same tokenizer lifetime clock. One first token per
        # finished request is removed from the numerator, including singleton outputs.
        weighted=(generated-n)/duration if matched and generated>n and duration>0 else None
        tpot_mean=tpot.get("mean_s")
        cohorts[which]=dict(native_completed=n,prompt_tokens=int(tokens.get("prompt_tokens_total",0)),generated_tokens=generated,
                ttft=ttft,e2e=e2e,tpot=tpot,populations_matched=matched,
                fallback_basis="Sum of generated tokens after one first token per finished request, divided by summed native decode seconds.",
                decode_seconds_from_matched_requests=duration if weighted is not None else None,
                token_weighted_decode_tps_from_e2e_minus_ttft=weighted,
                reciprocal_mean_request_tpot_tps=1/tpot_mean if tpot_mean else None)
    return cohorts


def serving_diff(before, after, quiet_boundaries=False):
    counters = counter_diff(before["totals"],after["totals"])
    h = {name:histogram(before["hist"].get(name,{}),after["hist"].get(name,{}))
         for name in set(before["hist"]) | set(after["hist"])}
    fresh = counters.get("uncached_prompt_tokens_histogram_sum")
    cached = None if fresh is None else counters.get("prompt_tokens_total",0)-fresh
    if fresh is not None and (cached < 0 or cached != counters.get("cached_tokens_total",0)):
        raise ValueError("Completed-request token and cache counters disagree")
    interval = h.get("inter_token_latency_seconds",{})
    tpot = h.get("request_time_per_output_token_seconds",{})
    prefill = h.get("prefill_forward_seconds",{})
    prefill_rate = None
    if (prefill.get("sum_s") and prefill["observations"] == counters.get("num_requests_total",0)
            and not counters.get("num_retracted_requests_total") and fresh is not None):
        prefill_rate = fresh / prefill["sum_s"]
    http_counts=counter_diff(before["http"],after["http"])
    return dict(response_endpoint_only=bool(http_counts.get("/v1/responses") and not any(v for k,v in http_counts.items() if k!="/v1/responses")),
                responses_http_status=counter_diff(before.get("responses_status",{}),after.get("responses_status",{})),
                responses_http_active_at_edges=[before.get("responses_active"),after.get("responses_active")],
                native_tokenizer_cohorts=by_cohort(before,after,quiet_boundaries and not (before["active"] or after["active"]),
                                                  counters.get("num_aborted_requests_total",0)),
                scheduler_progress=counter_diff(before.get("scheduler_progress",{}),after.get("scheduler_progress",{})),
                completed_requests=counters.get("num_requests_total",0),aborted_requests=counters.get("num_aborted_requests_total",0),
                prompt_tokens=counters.get("prompt_tokens_total",0),fresh_prompt_tokens=fresh,cached_prompt_tokens=cached,
                output_tokens=counters.get("generation_tokens_total",0),http_by_endpoint=counter_diff(before["http"],after["http"]),
                active_at_boundaries=[before["active"],after["active"]],histograms=h,
                token_weighted_inter_token_tps=1/interval["mean_s"] if interval.get("mean_s") else None,
                reciprocal_mean_streaming_request_tpot_tps=1/tpot["mean_s"] if tpot.get("mean_s") else None,
                completed_request_prefill_tokens_per_s=prefill_rate)


def distribution(samples):
    if not samples:
        return {"observations":0}
    vals = sorted(samples)
    return dict(observations=len(vals),median_ms=statistics.median(vals),p95_ms=vals[math.ceil(.95*len(vals))-1],max_ms=vals[-1])


def handoffs(path, start, end, samples):
    with path.open() as fp:
        events = [json.loads(line) for line in fp if line.endswith("\n")]
    states = [e for e in events if e["event"]=="state" and e["monotonic_ns"]<=end]
    origin = next((e for e in reversed(states) if e["monotonic_ns"]<=start), None)
    if not origin:
        raise ValueError("Missing broker state at the measurement boundary")
    timeline = [dict(origin,monotonic_ns=start)] + [e for e in states if e["monotonic_ns"]>start]
    permits, resumes = 0., []
    for i,left in enumerate(timeline):
        right = timeline[i+1]["monotonic_ns"] if i+1<len(timeline) else end
        if not left["serving"] and left["watched"]==4:
            permits += (right-left["monotonic_ns"])/1e9
        if i and not left["serving"] and timeline[i-1]["serving"]:
            resumes.append((left["monotonic_ns"],next((r["monotonic_ns"] for r in timeline[i+1:]
                            if r["serving"]),end)))
    observations = sorted(set(tuple((w.get("release_ns"),w.get("first_gpu_seen_ns")) for w in s["idle"]["workers"])
           for s in samples if "idle" in s and all(w.get("first_gpu_seen_ns") for w in s["idle"]["workers"])))
    starts = []
    for edge, until in resumes:
        for workers in observations:
            if all(edge<=release<=seen<until for release,seen in workers):
                starts.append((max(seen for _,seen in workers)-edge)/1e6)
                break
    delays = [r["elapsed_ns"]/1e6 for r in events if r["event"]=="admitted" and r["required_drain"]
              and start<=r["monotonic_ns"]<=end]
    return dict(broker_idle_permitted_s=permits,demand_to_all_worker_drain_ms=distribution(delays),
                idle_transitions=len(resumes),idle_to_all_gpu_reentry_ms=distribution(starts),
                reentry_note="Includes transitions for which all four workers retained their GPU start stamps at sampling time.")


def thermal(samples):
    rows = [s for s in samples if "gpu" in s]
    out = []
    for rank in range(4):
        temp = [s["gpu"][rank][1] for s in rows if s["gpu"][rank][1] is not None]
        watts = seconds = 0.
        for a,b in zip(rows,rows[1:]):
            dt = (b["end_ns"]-a["end_ns"])/1e9
            p,q = a["gpu"][rank][2],b["gpu"][rank][2]
            if 0 < dt < 10 and p is not None and q is not None:
                watts += (p+q)*.5*dt
                seconds += dt
        def delta(k):
            vals = [s["gpu"][rank][k] for s in rows if s["gpu"][rank][k] is not None]
            return vals[-1]-vals[0] if len(vals)>=2 and vals[-1]>=vals[0] else None
        out.append(dict(gpu=rank,max_temperature_c=max(temp,default=None),energy_sample_span_s=seconds,
                        energy_wh=watts/3600 if seconds else None,mean_power_w=watts/seconds if seconds else None,
                        sw_thermal_slowdown_s=delta(6)/1e6 if delta(6) is not None else None,
                        hw_thermal_slowdown_s=delta(7)/1e6 if delta(7) is not None else None))
    return out


def prefill_bounds(before, after, result):
    # Both layouts cover exactly 5120*2048 MACs per physical activation row;
    # splitting a logical singleton into two adds exactly one physical row.
    full = result["reconciled_mac_equivalent"].get("prefill")
    if full is not None:
        return [full,full]
    bound = [0,0]
    for r in RANKS:
        raw=result["tickets"]["prefill"][r]
        lo,hi=raw*65_536,raw*262_144
        record=result["rank_verifier_delta"][r]
        jobs,scan=record.get("prefill_jobs",{}),record.get("prefill_scan",{})
        calls=jobs.get("mined_calls",0)
        if (not graph.pending_scans(before) and not graph.pending_scans(after)
                and scan.get("verified_steps",0)==calls and not scan.get("verification_errors")):
            def singletons(doc):
                return sum(row["calls"] for row in doc["shapes"][r] if row["mode"]=="EXTEND"
                           and row["path"]=="eager_wo_b" and row["real_rows"] % 2)
            extras=singletons(after)-singletons(before)
            rows=jobs.get("mined_rows",0)
            tight_lo=max(lo,rows*5120*2048)
            tight_hi=min(hi,(rows+min(calls,extras))*5120*2048)
            if tight_lo <= tight_hi:
                lo,hi=tight_lo,tight_hi
        bound[0]+=lo
        bound[1]+=hi
    return bound


def direct_ledger(path, start, end):
    by_rank = [{k:0 for k in ("submitted","accepted","rejected","unknown","accepted_work_macs")} for _ in RANKS]
    seen = set()
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                continue
            item = json.loads(line)
            kind = item["event"]
            if not (start <= item["monotonic_ns"] < end and kind in ("submitted","accepted","rejected","unknown")):
                continue
            rank = item["rank"]
            token = (kind, rank, item["generation"], item["rpc_id"])
            if token in seen:
                raise ValueError("Duplicate direct pool receipt")
            seen.add(token)
            by_rank[rank][kind] += 1
            if kind == "accepted":
                by_rank[rank]["accepted_work_macs"] += (1<<256)/(int(item["target"],16)+1)
    counts = {k:sum(r[k] for r in by_rank) for k in by_rank[0]}
    return dict(by_rank=by_rank, **counts)


def summarize(folder, fleet_dir):
    window, first, last = (read(folder/n) for n in ("window.json","baseline.json","closing.json"))
    with (folder/"series.jsonl").open() as f:
        samples = [json.loads(line) for line in f if line.endswith("\n")]
    start, end = window["start_monotonic_ns"], min(window["end_monotonic_ns"],last["end_ns"])
    span = (end-start)/1e9
    summary = dict(requested_window=window, observed_s=span, first_ns=first["end_ns"],last_ns=last["end_ns"],
                   boundaries_relative_to_window_ms=[(first["end_ns"]-start)/1e6,(last["end_ns"]-window["end_monotonic_ns"])/1e6],
                   sample_count=len(samples), sample_errors={},issues={},baseline_direct_tmac_s=BASE_RATE)
    for s in samples:
        for k in s["errors"]:
            summary["sample_errors"][k]=summary["sample_errors"].get(k,0)+1
    def section(key, fn):
        try:
            summary[key]=fn()
        except (ValueError, KeyError, OSError, TypeError, IndexError, ZeroDivisionError) as err:
            summary["issues"][key] = type(err).__name__ + ": "+str(err)[:160]
    boundary_idle=all(s["idle"]["broker"].get("serving") is False and
                      all(s["idle"]["broker"].get("idle",[])) for s in (first,last))
    section("serving",lambda:serving_diff(first["server"],last["server"],boundary_idle))
    # A test can end mid-generation. Preserve separate estimates for the longest
    # observed interior with matching quiet boundaries and expose its coverage.
    quiet=[s for s in [first,*samples,last] if "server" in s and "idle" in s and not s["server"]["active"]
           and s["idle"]["broker"].get("serving") is False and s["idle"]["broker"].get("idle")==[True]*4]
    if len(quiet)>1 and not boundary_idle:
        section("serving_idle_to_idle_interval",lambda:dict(
            start_ns=quiet[0]["end_ns"],end_ns=quiet[-1]["end_ns"],
            **serving_diff(quiet[0]["server"],quiet[-1]["server"],True)))
    def mining():
        graphs=graph.graph_deltas(first["server"],last["server"])
        work=scan_summary(first["projection"],last["projection"],graphs)
        return {**work,"prefill_mac_bounds":prefill_bounds(first["projection"],last["projection"],work)}
    section("joined_work",mining)
    def upstream():
        a,b=first["gateway"],last["gateway"]
        if a["run_id"]!=b["run_id"]:
            raise ValueError("Gateway session restarted")
        fields=("submitted","accepted","rejected","unknown","invalid","duplicate","stale","pool_stale")
        out={k:b[k]-a[k] for k in fields if a.get(k) is not None and b.get(k) is not None}
        out["phases"]={p:{k:r[k]-a["phases"].get(p,{}).get(k,0) for k in r}
                       for p,r in b["phases"].items()}
        return out
    section("joined_pool",upstream)
    def direct():
        ranks=[]
        max_short_slices = 0
        for a,b,u,v in zip(first["idle"]["ranks"],last["idle"]["ranks"],
                           first["idle"]["workers"],last["idle"]["workers"],strict=True):
            r={k:b[k]-a[k] for k in ("submitted","accepted","rejected","unknown","accepted_work_macs") if a.get(k) is not None and b.get(k) is not None}
            r["completed_slices"]=v["completed_slices"]-u["completed_slices"]
            if r["completed_slices"]<0 or r.get("accepted",0)<0:
                raise ValueError("Direct-worker counters reset")
            max_short_slices += (r["completed_slices"]+138)//139
            if b.get("cooling_seconds") is not None and a.get("cooling_seconds") is not None:
                r["thermal_hold_seconds"]=b["cooling_seconds"]-a["cooling_seconds"]
            seen = {}
            index = len(ranks)
            for sample in samples:
                if "idle" not in sample:
                    continue
                point = sample["idle"]["ranks"][index].get("work")
                if point and start <= point["monotonic_ns"] <= end:
                    seen[point["monotonic_ns"]] = point["macs"]
            if len(seen) >= 2:
                lo,hi=min(seen),max(seen)
                r["native_counter_interior"] = dict(first_ns=lo,last_ns=hi,seconds=(hi-lo)/1e9,
                    macs=seen[hi]-seen[lo],tmac_s=(seen[hi]-seen[lo])/(hi-lo)/1e3)
            ranks.append(r)
        slices=sum(r["completed_slices"] for r in ranks)
        # With k780/K65536 and 188-tile slices, every final slice has 168 tiles.
        # Each new pool job restarts at tile zero. Consecutive short tails therefore
        # need at least 139 completed slices, including when a job rotates mid-nonce.
        bounds=[tiles*160*256*65536 for tiles in (slices*188-max_short_slices*20,slices*188)]
        accepted_work=sum(r["accepted_work_macs"] for r in ranks) if all("accepted_work_macs" in r for r in ranks) else None
        st, en = (s["idle"]["source_monotonic_ns"] for s in (first,last))
        cursor_span = (en-st)/1e9
        if cursor_span<=0:
            raise ValueError("Worker cursor sample interval empty")
        return dict(ranks=ranks,accepted=sum(r["accepted"] for r in ranks),cursor_span_s=cursor_span,
                    cursor_boundary_offsets_ms=[(st-start)/1e6,(en-end)/1e6],
                    accepted_work_macs=accepted_work,credited_work_tmac_s_over_wall=accepted_work/span/1e12 if accepted_work is not None else None,
                    direct_mac_bounds=bounds,wall_tmac_s_bounds=[x/cursor_span/1e12 for x in bounds],
                    pct_of_pure_direct_over_wall_bounds=[100*x/cursor_span/1e12/BASE_RATE for x in bounds])
    section("idle_mining",direct)
    section("direct_pool_exact_window",lambda:direct_ledger(fleet_dir/"events.jsonl",start,end))
    section("handoff",lambda:handoffs(fleet_dir/"broker-events.jsonl",start,end,samples))
    if "handoff" in summary and "idle_mining" in summary:
        mining = summary["idle_mining"]
        permit = summary["handoff"]["broker_idle_permitted_s"]
        if permit>0:
            mining["tmac_s_during_broker_permitted_idle_bounds"]=[m/permit/1e12 for m in mining["direct_mac_bounds"]]
            mining["pct_of_pure_direct_during_broker_permitted_idle_bounds"]=[100*m/permit/1e12/BASE_RATE for m in mining["direct_mac_bounds"]]
    if "joined_work" in summary and "handoff" in summary:
        j=summary["joined_work"]
        j["joined_mac_equivalent_over_wall_tmac_s"]={p:n/span/1e12 for p,n in j["reconciled_mac_equivalent"].items()}
        active=span-summary["handoff"]["broker_idle_permitted_s"]
        j["broker_serving_s"]=active
        if active>0:
            j["pct_of_historical_direct_rate_during_broker_serving"]={p:100*n/active/1e12/BASE_RATE for p,n in j["reconciled_mac_equivalent"].items()}
    if "joined_work" in summary:
        # A bound using the actual targets sampled at the gateway. The estimator
        # assumes random tickets and immediate preparation of the corresponding job.
        targets = [int(s["gateway"]["target"],16) for s in samples if s.get("gateway",{}).get("target")]
        if targets:
            work = summary["joined_work"]
            decode = sum(work["tickets"]["decode"].values())*65_536
            macs = [decode + x for x in work["prefill_mac_bounds"]]
            lam = [x*(target+1)/(1<<256) for x,target in zip(macs,(min(targets),max(targets)))]
            work["expected_shares_at_hardest_and_easiest_sampled_targets"] = lam
            work["p_at_least_one_estimate_range"] = [-math.expm1(-v) for v in lam]
    section("gpu",lambda:thermal(samples))
    return summary


def collect(args):
    os.umask(0o077)
    folder=args.run_dir/"metrics"
    folder.mkdir(mode=0o700,exist_ok=True)
    stopping=False
    def stop(*unused):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,stop)
    base=last=window=None
    profile_written=False
    with (folder/"series.jsonl").open("x",buffering=1) as series:
        while True:
            sample=take(args)
            # Never persist free-form labels or model traffic. Rank projection files
            # contain shapes and aggregate counters only; store them at boundaries.
            row={k:v for k,v in sample.items() if k!="projection"}
            if "projection" in sample:
                row["verified_tickets"]={r:{p:int(graph.phase_counters(sample["projection"],r)
                        .get(p+"_scan",{}).get("tickets_scanned",0)) for p in ("decode","prefill")} for r in RANKS}
            if window is None and args.window_file.exists():
                window=read(args.window_file)
                if window["end_monotonic_ns"]<=window["start_monotonic_ns"]:
                    raise ValueError("Invalid trial schedule")
                write(folder/"window.json",window)
                if base is None and complete(sample):
                    base=sample
                if base is not None:
                    write(folder/"baseline.json",base)
            if window is None:
                if complete(sample):
                    base=sample
                    if not profile_written:
                        os.environ["CANARY_API_KEY_FILE"]=str(args.key_file)
                        try:
                            profile=graph.profile(f"http://127.0.0.1:{args.port}")
                            if profile["tp_size"]==4 and profile["ep_size"]==4 and profile["speculative_algorithm"]=="DSPARK":
                                write(folder/"profile.json",profile)
                                profile_written=True
                        except (OSError,ValueError,KeyError):
                            pass
                    if (profile_written and sample["gateway"]["connected"] and sample["gateway"]["authorized_clients"]==4
                            and all(r.get("connected") for r in sample["idle"]["ranks"]) and not (folder/"ready.json").exists()):
                        write(folder/"ready.json",dict(monotonic_ns=sample["end_ns"],utc_epoch=time.time(),
                                                       tp_size=4, idle_workers=4, joined_ranks=4))
                write(folder/"startup.json",{"source_errors":sample["errors"],
                      "profile_ready":profile_written,"connected_ranks":sample.get("gateway",{}).get("authorized_clients"),
                      "idle_connected":[r.get("connected") for r in sample.get("idle",{}).get("ranks",())]})
            else:
                print(json.dumps(row,separators=(",",":"), allow_nan=False),file=series)
                write(folder/"latest.json",row)
                if complete(sample):
                    last=sample
                    # An independent restore or host restart can end the test at any instant.
                    write(folder/"closing.json",last)
                if base is None and last is not None:
                    base=last
                    write(folder/"baseline.json",base)
                if sample["end_ns"]>=window["end_monotonic_ns"]:
                    break
            if stopping or (folder/"stop").exists():
                break
            wait=.35 if window is None else min(args.interval,max(0,(window["end_monotonic_ns"]-sample["end_ns"])/1e9))
            time.sleep(wait)
    if window and base and last:
        report=summarize(folder,args.fleet_dir)
        write(folder/"report.json",report)
    write(folder/"finished.json",dict(utc_epoch=time.time(),report=(folder/"report.json").is_file()))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    subs=cli.add_subparsers(dest="action",required=True)
    collect_cli=subs.add_parser("collect")
    for name in ("run-dir","key-file","shape-dir","gateway-receipt","fleet-dir","window-file"):
        collect_cli.add_argument("--"+name,required=True,type=Path)
    collect_cli.add_argument("--port",type=int,default=18002)
    collect_cli.add_argument("--interval",type=float,default=2.)
    offline=subs.add_parser("report")
    offline.add_argument("--run-dir",type=Path,required=True)
    offline.add_argument("--fleet-dir",type=Path,required=True)
    args=cli.parse_args()
    if args.action=="report":
        print(json.dumps(summarize(args.run_dir/"metrics",args.fleet_dir),indent=2,allow_nan=False))
    elif 0.2<=args.interval<=10 and 1<=args.port<=65535:
        urllib.request.install_opener(LOOPBACK)
        collect(args)
    else:
        cli.error("Supply a valid port and sampling interval from 0.2 to 10 seconds")


if __name__=="__main__":
    main()
