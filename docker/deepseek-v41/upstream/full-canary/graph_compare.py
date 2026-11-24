"""Measure real HTTP decode and verify hardware replay on all four production ranks."""
import argparse
import hashlib
import json
import math
import re
import statistics
import struct
import time
import uuid
from pathlib import Path

from qualify import gpu_snapshot, request
from throughput import exact_prompt, handbook, sse, write
from dsv41_load import activity_metrics


def snapshot(base):
    with request(base, "/metrics", timeout=5) as response:
        text = response.read().decode()
    active, endpoints = activity_metrics(text)
    graphs = {}
    for line in text.splitlines():
        if not line.startswith("sglang:cuda_graph_passes_total{"):
            continue
        sample, value = line.rsplit(None, 1)
        labels = dict(re.findall(r'(\w+)="([^"]*)"', sample))
        rank, mode = labels["tp_rank"], labels["mode"]
        counts = graphs.setdefault(rank, {})
        if mode in counts:
            raise ValueError(f"Ambiguous rank {rank} graph series for {mode}")
        counts[mode] = int(float(value))
    return dict(active=active, total=sum(n for _, n in endpoints), graphs=graphs)


def profile(base):
    with request(base, "/server_info", timeout=5) as response:
        info = json.load(response)
    fields = ("served_model_name", "model_path", "version", "tp_size", "ep_size", "context_length",
              "max_total_num_tokens", "mem_fraction_static", "speculative_algorithm", "disable_cuda_graph",
              "cuda_graph_config", "chunked_prefill_size", "speculative_num_draft_tokens",
              "quantization", "kv_cache_dtype", "attention_backend", "moe_runner_backend", "random_seed",
              "speculative_draft_model_path", "speculative_dspark_block_size")
    result = {k: info.get(k) for k in fields}
    if (result["tp_size"], result["ep_size"], result["speculative_algorithm"]) != (4, 4, "DSPARK"):
        raise ValueError("Expected a four-rank DSPARK serving deployment")
    if result["disable_cuda_graph"] or result["cuda_graph_config"]["decode"]["backend"] != "full":
        raise ValueError("The serving profile does not enable the full decode CUDA graph")
    return result


def gpu_issues(gpus, cap):
    if not gpus or sorted(int(g["index"]) for g in gpus) != [0, 1, 2, 3]:
        return ["Cannot verify all four physical GPUs"]
    if any(abs(float(g["power.limit"]) - cap) > 0.01 for g in gpus):
        return [f"Per-card power cap differs from requested {cap} W"]
    return []


def gpu_clocks(gpus):
    return {g["uuid"]: int(g["clocks.current.memory"]) for g in gpus or []}


def graph_deltas(before, after):
    result = {}
    for rank in ("0", "1", "2", "3"):
        old = before["graphs"].get(rank, {})
        new = after["graphs"].get(rank, {})
        result[rank] = {m: new.get(m, 0) - old.get(m, 0) for m in set(old) | set(new)}
    return result


def token_hash(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def percentile(samples, fraction):
    if not samples:
        return None
    return sorted(samples)[math.ceil(fraction * len(samples)) - 1]


def measure_stream(base, ids, output, salt):
    body = dict(input_ids=ids, stream=True, cache_salt=salt,
                sampling_params=dict(temperature=0, max_new_tokens=output, ignore_eos=True))
    trace = []
    old_count = 0
    started = time.monotonic_ns()
    with request(base, "/generate", body) as reply:
        for event in sse(reply):
            stamp = time.monotonic_ns()
            meta = event["meta_info"]
            count = meta["completion_tokens"]
            if count < old_count or count > output:
                raise ValueError("Non-monotonic or excessive generated token count")
            if count > old_count:
                trace.append([stamp, count])
                old_count = count
    finished = time.monotonic_ns()
    if len(trace) < 2 or old_count != output or len(event["output_ids"]) != output or meta["prompt_tokens"] != len(ids):
        raise RuntimeError("Stream did not complete the requested fixed workload")
    first, last = trace[0][0], trace[-1][0]
    gaps_ms = [(nxt[0] - prev[0]) * 1e-6 for prev, nxt in zip(trace, trace[1:])]
    native_prefill_s = (meta.get("prefill_finished_time") or 0) - (meta.get("forward_entry_time") or 0)
    return dict(effective_prefill_tps=(len(ids) - meta["cached_tokens"]) * 1e9 / (first-started),
                native_prefill_tps=(len(ids) - meta["cached_tokens"]) / native_prefill_s if native_prefill_s > 0 else None,
                input_sha256=token_hash(ids), output_sha256=token_hash(event["output_ids"]),
                input_tokens=len(ids), output_tokens=output, cache_salt=salt, start_monotonic_ns=started,
                first_token_monotonic_ns=first, last_token_monotonic_ns=last, stream_closed_monotonic_ns=finished,
                ttft_s=(first-started)*1e-9, send_to_last_token_s=(last-started)*1e-9, stream_wall_s=(finished-started)*1e-9,
                first_event_output_tokens=trace[0][1], token_bearing_events=len(trace),
                client_decode_tps=1e9*(output-trace[0][1])/(last-first),
                native_decode_tps=meta.get("decode_throughput"), sse_gap_ms_p50=percentile(gaps_ms,.50),
                sse_gap_ms_p95=percentile(gaps_ms,.95), sse_gap_ms_p99=percentile(gaps_ms,.99),
                sse_gap_ms_max=max(gaps_ms), token_event_timeline_ns=trace,
                **{k:meta.get(k) for k in ("cached_tokens","spec_verify_ct","spec_accept_length","num_retractions")})


def wait_until(stamp_ns):
    if stamp_ns is not None:
        delay = (stamp_ns-time.monotonic_ns())/1e9
        if delay > 0:
            time.sleep(delay)


def completed_http_snapshot(base, before, expected_requests=1):
    after = snapshot(base)
    expected_total = before["total"] + expected_requests
    deadline = time.monotonic() + 3
    # Prometheus marks the request complete after the final SSE frame; wait for
    # that teardown without accepting an additional request or a counter reset.
    while (before["total"] <= after["total"] <= expected_total
           and (after["active"] or after["total"] < expected_total)
           and time.monotonic() < deadline):
        time.sleep(.03)
        after = snapshot(base)
    return after


def run_trial(base, arm, pair, inputs=2048, outputs=256, receipt=None, cap=275,
              suite="public-graph-v1", ids=None, info=None, scheduled_ns=None, trace_state=None):
    """Run one cold, public, single-stream fixture. Pair number fixes token IDs across arms."""
    if inputs < 256 or outputs < 2:
        raise ValueError("Use at least 256 input tokens and two output tokens")
    info = info or profile(base)
    if ids is None:
        ids = exact_prompt(base, info["served_model_name"], inputs, f"{suite}-{inputs}-{pair}")
    if scheduled_ns is not None and trace_state is None:
        wait_until(scheduled_ns-250_000_000)
    salt = f"{suite}-{arm}-{pair}-{uuid.uuid4().hex}"
    gpus = gpu_snapshot() if trace_state is None else trace_state["gpus"]
    issues = gpu_issues(gpus, cap)
    if issues:
        raise ValueError("; ".join(issues))
    before = snapshot(base) if trace_state is None else trace_state["http"]
    if trace_state is None and before["active"]:
        raise RuntimeError("Existing inference traffic; skip this trial rather than overlap")
    wait_until(scheduled_ns)
    stream = measure_stream(base, ids, outputs, salt)
    # A completed HTTP handler can remain active in the exported metrics after
    # the last SSE frame. During scheduled traces we verify total HTTP isolation
    # after all sends and use immediate per-request graph-counter snapshots.
    after = completed_http_snapshot(base, before) if trace_state is None else snapshot(base)
    gpus_after = gpu_snapshot()
    if trace_state is not None:
        trace_state.update(gpus=gpus_after, http=after)
    deltas = graph_deltas(before, after)
    issues.extend(gpu_issues(gpus_after, cap))
    if trace_state is None and (after["active"] or after["total"]-before["total"] != 1):
        issues.append("HTTP counter delta is not exactly one isolated request")
    for rank, modes in deltas.items():
        if modes.get("decode_cuda_graph", 0) <= 0 or modes.get("decode_none", 0) != 0:
            issues.append(f"Rank {rank} has missing decode graph replay or an eager decode fallback")
    if stream["cached_tokens"] != 0 or stream["num_retractions"] != 0:
        issues.append("The fixture was cached, retracted, or did not expose both counters")
    result = dict(arm=arm, pair=pair, suite=suite, cap_w=cap, scheduled_send_monotonic_ns=scheduled_ns,
                  utc_epoch=time.time(), serving=info, gpus_before=gpus, gpus_after=gpus_after,
                  http_and_graph_before=before, http_and_graph_after=after, graph_delta_by_rank=deltas,
                  http_isolation_basis="individual handler" if trace_state is None else "whole serial trace HTTP delta and individual rank replay",
                  request=stream, qualification_issues=issues)
    if receipt is not None:
        write(receipt, result)
    if issues:
        raise RuntimeError("Invalid trial: "+"; ".join(issues))
    return result



def projection_snapshot(directory):
    """Read locally published, aggregate-only rank snapshots; no scheduler polling."""
    if directory is None:
        return None
    result = {}
    for kind in ("shapes", "counters"):
        files = sorted(directory.glob(f"pearl-{kind}-*-tp*.json"))
        if not files and kind == "counters":
            continue
        by_rank = {}
        for path in files:
            rank = path.stem.rsplit("tp", 1)[1]
            if rank not in ("0","1","2","3") or rank in by_rank:
                raise ValueError(f"Ambiguous rank snapshot in {directory}")
            by_rank[rank] = json.loads(path.read_text())
        if len(by_rank) != 4:
            raise ValueError(f"Expected four rank {kind} files, found {len(by_rank)} in {directory}")
        result[kind] = by_rank
    return result


def phase_counters(doc, rank):
    """Each TP rank is one process and normally owns one device; sum any device partitions."""
    result = {}
    for device in doc.get("counters", {}).get(rank, {}).values():
        phases = {"decode_scan": device.get("verify", {}),
                  "prefill_scan": device.get("eager_verify", {}).get("prefill", {}),
                  "prefill_jobs": device.get("eager", {}).get("prefill", {}),
                  "prefill_submit": device.get("pool", {}).get("prefill", {}),
                  "decode_submit": device.get("pool", {}).get("decode", {})}
        for phase, counters in phases.items():
            totals = result.setdefault(phase, {})
            for key, value in counters.items():
                totals[key] = totals.get(key, 0) + value
    return result


def pending_scans(doc):
    if doc.get("counters", {}).keys() != doc["shapes"].keys():
        return True
    for rank, rows in doc["shapes"].items():
        counters = phase_counters(doc, rank)
        decode_calls = sum(row["calls"] for row in rows if row["mode"] == "TARGET_VERIFY" and row["path"] == "target_graph_mined")
        for phase, submitted in (("decode_scan", decode_calls), ("prefill_scan", counters.get("prefill_jobs", {}).get("mined_calls", 0))):
            scan = counters.get(phase, {})
            if scan.get("verified_steps", 0) + scan.get("verification_errors", 0) != submitted:
                return True
    return False


def drained_projection(directory):
    if directory is None:
        return None
    # Call only outside the measured horizon. Four idle callbacks publish snapshots
    # independently; the verification threads drain after the final graph replay.
    time.sleep(.35)
    deadline = time.monotonic() + 5
    while True:
        snapshot = projection_snapshot(directory)
        if not pending_scans(snapshot) or time.monotonic() >= deadline:
            return snapshot
        time.sleep(.1)


def scanned_prefill_layout(before, after, jobs, scan):
    def observed(shapes):
        return {(r["real_rows"], r["executed_rows"]):r["calls"] for r in shapes
                if r["mode"] == "EXTEND" and r["path"] == "eager_wo_b"}
    old, new = observed(before), observed(after)
    tickets = {"compact": 0, "wide": 0}
    calls = real_rows = executed_rows = 0
    for key in old.keys() | new.keys():
        count = new.get(key, 0) - old.get(key, 0)
        if count < 0:
            raise ValueError("Prefill shape counter reset")
        if not count:
            continue
        real, executed = key
        if real != executed or real <= 0:
            return None
        # torch_forward splits an odd M>1 into M-1 plus one; a single real row
        # uses a two-row quotient/remainder encoding to produce one logical output.
        pieces = [real-1, 1] if real > 1 and real % 2 else [real]
        for rows in pieces:
            physical = max(2, rows)
            layout = "compact" if physical % 16 else "wide"
            tickets[layout] += count * physical * (160 if layout == "compact" else 40)
            calls += count
            real_rows += count * rows
            executed_rows += count * physical
    if (real_rows != jobs.get("mined_rows", 0) or calls != jobs.get("mined_calls", 0) or
            calls != scan.get("verified_steps", 0) or scan.get("verification_errors", 0) or
            sum(tickets.values()) != scan.get("tickets_scanned", 0)):
        return None
    return dict(tickets=tickets, verified_steps=calls, real_rows=real_rows, executed_rows=executed_rows,
                logical_model_macs=real_rows*5120*2048,
                scanned_ticket_mac_equivalent=tickets["compact"]*65_536 + tickets["wide"]*262_144)


def graph_work(before, after, runs, drained_graph_delta):
    if before is None:
        return None
    def shape_totals(items):
        seen = {}
        for row in items:
            if row["mode"] != "TARGET_VERIFY":
                continue
            key = (row["path"], row["real_rows"], row["executed_rows"])
            seen[key] = seen.get(key, 0) + row["calls"]
        return seen
    by_rank = {}
    per_rank_scan = {}
    issues = []
    if pending_scans(before) or pending_scans(after):
        issues.append("Verifier accounting did not fully drain at the trace boundaries")
    total_macs = {"decode": 0, "prefill": 0}
    phase_issues = {"decode": [], "prefill": []}
    prefill_layouts = {}
    prefill_scanned_lower = 0
    prefill_scanned_upper = 0
    for rank in ("0","1","2","3"):
        old = shape_totals(before["shapes"][rank])
        new = shape_totals(after["shapes"][rank])
        summary = {}
        for key in set(old) | set(new):
            path, real, executed = key
            calls = new.get(key, 0) - old.get(key, 0)
            if calls < 0:
                raise ValueError(f"Rank {rank} projection counter reset")
            if not calls:
                continue
            counts = summary.setdefault(path, dict(calls=0, real_rows=0, executed_rows=0))
            counts["calls"] += calls
            counts["real_rows"] += calls * real
            counts["executed_rows"] += calls * executed
            if path == "target_graph_mined" and real != executed:
                issues.append(f"Rank {rank} counted padded verification rows as mined")
        native = drained_graph_delta[rank].get("decode_cuda_graph", 0)
        if summary.get("target_graph", {}).get("calls", 0) != native:
            issues.append(f"Rank {rank} target graph hook pass count differs from native graph counter")
        if summary.get("target_graph_mined", {}).get("calls", 0) == 0:
            issues.append(f"Rank {rank} did not select a job and slot for real graph verification")
        by_rank[rank] = summary
        previous, current = (phase_counters(doc, rank) for doc in (before, after))
        delta = {}
        for phase in current.keys() | previous.keys():
            old_phase, new_phase = previous.get(phase, {}), current.get(phase, {})
            delta[phase] = {k: new_phase.get(k, 0) - old_phase.get(k, 0) for k in new_phase.keys() | old_phase.keys()}
            if any(v < 0 for v in delta[phase].values()):
                raise ValueError(f"Rank {rank} verifier counter reset in {phase}")
        per_rank_scan[rank] = delta
        mined = summary.get("target_graph_mined", {})
        decode = delta.get("decode_scan", {})
        decode_tickets = decode.get("tickets_scanned", 0)
        if (decode_tickets != mined.get("executed_rows", 0) * 160 or
                decode.get("verified_steps", 0) != mined.get("calls", 0) or decode.get("verification_errors", 0)):
            phase_issues["decode"].append(f"Rank {rank} target replay rows, scanned tickets, and completed verifier steps disagree")
        total_macs["decode"] += decode_tickets * 65_536  # 2 x 16 x K=2048 compact tickets.
        prefill, jobs = delta.get("prefill_scan", {}), delta.get("prefill_jobs", {})
        prefill_tickets = prefill.get("tickets_scanned", 0)
        # Each short fixture fits in one unpadded, aligned prefill projection per rank.
        # Exact scanned tickets independently distinguish 2x64 wide from 2x16 compact.
        aligned_inputs = {r["request"]["input_tokens"] for r in runs}
        layout = scanned_prefill_layout(before["shapes"][rank], after["shapes"][rank], jobs, prefill)
        prefill_layouts[rank] = layout
        if layout is not None:
            total_macs["prefill"] += layout["scanned_ticket_mac_equivalent"]
        elif (len(aligned_inputs) == 1 and min(aligned_inputs) % 16 == 0 and
              jobs.get("mined_rows", 0) == min(aligned_inputs) * jobs.get("mined_calls", 0) and
              prefill.get("verified_steps", 0) == jobs.get("mined_calls", 0) and
              prefill_tickets == 40 * jobs.get("mined_rows", 0) and not prefill.get("verification_errors", 0)):
            total_macs["prefill"] += prefill_tickets * 262_144
        else:
            phase_issues["prefill"].append(f"Rank {rank} prefill shapes, job counts, and verified ticket layout disagree")
        prefill_scanned_lower += prefill_tickets * 65_536
        prefill_scanned_upper += prefill_tickets * 262_144
    qualified = {phase: None if phase_issues[phase] or issues else count for phase,count in total_macs.items()}
    return {"shape_delta_by_rank": by_rank, "mined_hook_issues": issues,
            "verifier_delta_by_rank": per_rank_scan, "ticket_accounting_issues": phase_issues,
            "prefill_layout_reconciliation_by_rank": prefill_layouts,
            "independently_scanned_projection_mac_equivalent": qualified,
            "physically_scanned_prefill_mac_bounds_before_layout_reconciliation": [prefill_scanned_lower, prefill_scanned_upper]}



def peer_snapshot(path):
    if path is None:
        return None
    doc = json.loads(path.read_text())
    keys = ("started_at", "job", "accepted_total", "negotiations", "rejected_frames",
            "verifier_sha256", "verified_by_rank_rows")
    return {k: doc[k] for k in keys}


def local_proofs(path, start_ns, end_ns, peer):
    """Count ACKs whose public submission first reached this verifier in the trace."""
    if path is None:
        return None
    first_seen = {}
    received = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["peer_started_at"] != peer["started_at"] or event["job_id"] != peer["job"]["job_id"]:
            raise ValueError("Unexpected peer generation or job in accepted log")
        stamp = event["received_monotonic_ns"]
        if stamp >= end_ns:
            continue
        key = (event["job_id"], event["proof_sha256"])
        if key not in first_seen or stamp < first_seen[key]["received_monotonic_ns"]:
            first_seen[key] = event
        if stamp >= start_ns:
            received.append(event)
    # ACK order can differ from input-frame order when four verifier threads run concurrently.
    by_shape = {}
    for event in received:
        rows = str(event["proof_rows"])
        counts = by_shape.setdefault(rows, dict(accepted=0, unique=0, by_rank={}))
        counts["accepted"] += 1
    unique = 0
    for event in first_seen.values():
        if event["received_monotonic_ns"] < start_ns:
            continue
        counts = by_shape[str(event["proof_rows"])]
        counts["unique"] += 1
        rank = str(event["tp_rank"])
        counts["by_rank"][rank] = counts["by_rank"].get(rank, 0) + 1
        unique += 1
    return dict(verified_by_real_rows=by_shape, duplicate_submissions=len(received)-unique,
                observed_at_monotonic_ns=time.monotonic_ns(),
                basis="ACKed public proofs grouped by receipt time; deduplicated since peer start, local validity only")



def run_trace(args):
    if args.gap < 0 or args.period <= 0 or args.pairs <= 0 or args.inputs < 256 or args.outputs < 2:
        raise ValueError("Invalid trace dimensions")
    if (args.accepted_log is None) != (args.peer_ledger is None):
        raise ValueError("Use peer ledger and accepted log together")
    if args.receipt.exists():
        raise FileExistsError(f"Use a fresh trace receipt path: {args.receipt}")
    info = profile(args.base)
    suite = "public-graph-" + args.mode
    inputs = [exact_prompt(args.base, info["served_model_name"], args.inputs, f"{suite}-{args.inputs}-{i}")
              for i in range(args.pairs)]
    # Warm both the real serving graph and the first-use kernels on an unrelated, uncached prefix.
    warm = run_trial(args.base, args.arm, -1, args.inputs, args.outputs, suite=suite, info=info)
    # Allow the warm-up's four busy-to-idle rank callbacks to publish. Their
    # TP graph totals must match the independently checked in-trace counters.
    projection_before = drained_projection(args.shape_dir)
    before = snapshot(args.base)
    peer_before = peer_snapshot(args.peer_ledger)
    start = time.monotonic_ns()
    lead_ns = 500_000_000
    period_ns = int(args.period * 1e9)
    gap_ns = int(args.gap * 1e9)
    end = start + lead_ns + args.pairs * period_ns if args.mode == "fixed-period" else None
    doc = dict(arm=args.arm, trace_mode=args.mode, suite=suite, cap_w=275, inputs=args.inputs, outputs=args.outputs,
               serving=info, schedule_period_s=args.period if args.mode == "fixed-period" else None,
               post_completion_idle_s=args.gap if args.mode == "fixed-gap" else None, initial_idle_s=lead_ns/1e9,
               requested_horizon_start_monotonic_ns=start, requested_horizon_end_monotonic_ns=end,
               untimed_warmup=warm, http_before=before, peer_before=peer_before, projection_before=projection_before,
               runs=[], complete=False)
    write(args.receipt, doc)
    scheduled = start + lead_ns
    trace_state = dict(gpus=warm["gpus_after"], http=before)
    try:
        for index, ids in enumerate(inputs):
            if args.mode == "fixed-period":
                scheduled = start + lead_ns + index * period_ns
            # Save the per-request receipt on validation failure as well as on success.
            trial_file = args.receipt.with_name(f"{args.receipt.stem}-trial-{index}.json")
            row = run_trial(args.base, args.arm, index, args.inputs, args.outputs, suite=suite, ids=ids,
                            info=info, scheduled_ns=scheduled, receipt=trial_file, trace_state=trace_state)
            doc["runs"].append(row)
            if args.mode == "fixed-gap":
                scheduled = row["request"]["stream_closed_monotonic_ns"] + gap_ns
            write(args.receipt, doc)
        if args.mode == "fixed-gap":
            end = scheduled
            doc["requested_horizon_end_monotonic_ns"] = end
        wait_until(end)
        observed_end = time.monotonic_ns()
        projection_after = drained_projection(args.shape_dir)
        after = completed_http_snapshot(args.base, before, expected_requests=args.pairs)
        peer_after = peer_snapshot(args.peer_ledger)
        if peer_before and (peer_after["job"] != peer_before["job"] or
                            peer_after["started_at"] != peer_before["started_at"]):
            raise ValueError("Verifier changed job or generation during this trace")
        offsets = [(r["request"]["start_monotonic_ns"]-r["scheduled_send_monotonic_ns"])/1e6 for r in doc["runs"]]
        last = doc["runs"][-1]["request"]["stream_closed_monotonic_ns"]
        doc.update(observed_horizon_end_monotonic_ns=observed_end, send_lateness_ms=offsets,
                   horizon_overrun_ms=max(0, last-end)/1e6, http_after=after, peer_after=peer_after,
                   local_verified=local_proofs(args.accepted_log,start,end,peer_after),
                   projection_after=projection_after, inference_graph_work=graph_work(projection_before, projection_after, doc["runs"], graph_deltas(before, after)),
                   counted_inference_requests=after["total"]-before["total"])
        if any(modes.get("decode_none", 0) or modes.get("decode_cuda_graph", 0) <= 0
               for modes in graph_deltas(before, after).values()):
            raise ValueError("Invalid settled four-rank decode replay counters")
        if doc["counted_inference_requests"] != args.pairs or before["active"] or after["active"]:
            raise ValueError("Unrelated inference overlapped the overall trace")
        if args.mode == "fixed-period" and last>end:
            raise ValueError("Trace overran its fixed horizon; repeat both arms with a longer period")
        doc["complete"] = True
    except Exception as exc:
        doc["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write(args.receipt, doc)
    return doc


def run_handbook(args):
    """Replay the historical Responses fixture and certify native decode graph deltas."""
    if args.receipt.exists():
        raise FileExistsError(f"Use a fresh receipt path: {args.receipt}")
    info = profile(args.base)
    doc = dict(arm=args.arm, serving=info, start_utc_epoch=time.time(), runs=[], complete=False)
    write(args.receipt, doc)
    try:
        for size in (32768, 131072, 200000):
            gpus = gpu_snapshot()
            projection_before = drained_projection(args.shape_dir)
            before = snapshot(args.base)
            if before["active"] or gpu_issues(gpus, 275):
                raise ValueError("Unexpected active request, GPU device, or power cap")
            row = handbook(args.base, info["served_model_name"], size, 8192 if size == 32768 else 512)
            projection_after = drained_projection(args.shape_dir)
            after = completed_http_snapshot(args.base, before)
            deltas = graph_deltas(before, after)
            if projection_before is not None:
                row["projection_before"] = projection_before
                row["projection_after"] = projection_after
                # Long, chunked, non-aligned prefills can mix compact and wide jobs.
                # Preserve the raw rank/scan counts; use exact row-normalized
                # MAC accounting only for the measured aligned public traces.
                row["graph_work"] = graph_work(projection_before, projection_after, [{"request":row}], deltas)

            row.update(graph_delta_by_rank=deltas, gpus_before=gpus, gpus_after=gpu_snapshot(),
                       http_and_graph_before=before, http_and_graph_after=after)
            doc["runs"].append(row)
            write(args.receipt, doc)
            print(json.dumps({k: row[k] for k in ("input_tokens","output_tokens","cached_tokens","ttft_s","client_decode_tps","e2e_s")}), flush=True)
            if after["total"] - before["total"] != 1 or after["active"]:
                raise ValueError(f"Unrelated traffic or incomplete handler at {size}")
            if gpu_issues(row["gpus_after"], 275) or any(
                d.get("decode_cuda_graph", 0) <= 0 or d.get("decode_none", 0) for d in deltas.values()
            ):
                raise ValueError(f"GPU power cap change or incomplete four-rank graph replay at {size}")
        doc["complete"] = True
    except Exception as exc:
        doc["error"] = str(exc)
        raise
    finally:
        doc["end_utc_epoch"] = time.time()
        write(args.receipt, doc)
    return doc


def paired_rows(controls, candidates):
    """Summarize matched pairs; reject fixture differences and invalid trials."""
    def index(rows):
        records = {}
        for row in rows:
            if row["qualification_issues"] or row["pair"] in records:
                raise ValueError(f"Invalid or duplicate pair {row['pair']}")
            records[row["pair"]] = row
        return records
    a, b = index(controls), index(candidates)
    if a.keys() != b.keys() or not a:
        raise ValueError("Controls and candidates must have identical nonempty pair sets")
    paired = []
    for pair in sorted(a):
        x,y=a[pair],b[pair]
        xr,yr=x["request"],y["request"]
        if (x["suite"],xr["input_sha256"],xr["output_tokens"],x["cap_w"]) != (
                y["suite"],yr["input_sha256"],yr["output_tokens"],y["cap_w"]):
            raise ValueError(f"Unmatched fixture/cap for pair {pair}")
        if {g["uuid"] for g in x["gpus_before"]} != {g["uuid"] for g in y["gpus_before"]}:
            raise ValueError(f"Physical GPUs changed for pair {pair}")
        changed = {k:[x["serving"].get(k),y["serving"].get(k)] for k in x["serving"]
                   if x["serving"].get(k) != y["serving"].get(k)}
        paired.append(dict(pair=pair,control_decode_tps=xr["client_decode_tps"],candidate_decode_tps=yr["client_decode_tps"],
                           decode_rate_change_pct=100*(yr["client_decode_tps"]/xr["client_decode_tps"]-1),
                           ttft_change_ms=1000*(yr["ttft_s"]-xr["ttft_s"]),
                           token_event_gap_p99_change_ms=yr["sse_gap_ms_p99"]-xr["sse_gap_ms_p99"],
                           exact_output=xr["output_sha256"]==yr["output_sha256"],serving_differences=changed,
                           memory_clocks_mhz_control=[gpu_clocks(x[k]) for k in ("gpus_before","gpus_after")],
                           memory_clocks_mhz_candidate=[gpu_clocks(y[k]) for k in ("gpus_before","gpus_after")]))
    return dict(pairs=paired,median_paired_decode_change_pct=statistics.median(
        r["decode_rate_change_pct"] for r in paired), exact_outputs=sum(r["exact_output"] for r in paired))


def trace_summary(trace):
    rows = [r["request"] for r in trace["runs"]]
    gaps = []
    per_request_graphs = {str(rank): 0 for rank in range(4)}
    for run, row in zip(trace["runs"], rows, strict=True):
        stamps = [ns for ns, _ in row["token_event_timeline_ns"]]
        gaps.extend((b-a)/1e6 for a,b in zip(stamps, stamps[1:]))
        for rank, modes in run["graph_delta_by_rank"].items():
            per_request_graphs[rank] += modes["decode_cuda_graph"]
    drained_graphs = graph_deltas(trace["http_before"], trace["http_after"])
    t0, t1 = (trace[f"requested_horizon_{key}_monotonic_ns"] for key in ("start", "end"))
    return dict(horizon_seconds=(t1-t0)/1e9, total_output_tokens=sum(r["output_tokens"] for r in rows),
                median_ttft_ms=statistics.median(r["ttft_s"] for r in rows)*1e3,
                median_effective_prefill_tps=statistics.median(r["effective_prefill_tps"] for r in rows),
                median_native_prefill_tps=statistics.median(r["native_prefill_tps"] for r in rows) if all(r["native_prefill_tps"] is not None for r in rows) else None,
                median_client_decode_tps=statistics.median(r["client_decode_tps"] for r in rows),
                median_native_decode_tps=statistics.median(r["native_decode_tps"] for r in rows),
                pooled_token_bearing_sse_gap_p95_ms=percentile(gaps,.95), pooled_token_bearing_sse_gap_p99_ms=percentile(gaps,.99),
                replay_passes_by_rank={k:v.get("decode_cuda_graph",0) for k,v in drained_graphs.items()},
                immediate_per_request_replays_by_rank=per_request_graphs, send_lateness_ms=trace["send_lateness_ms"],
                received_verified_inference_proofs=trace.get("local_verified"),
                inference_graph_work=trace.get("inference_graph_work"))


def work_comparison(control, candidate, direct_control, direct_candidate):
    summaries = {}
    for name, trace, fleet in (("control", control, direct_control), ("candidate", candidate, direct_candidate)):
        direct = fleet["direct_horizon"]
        start, end = (trace[f"requested_horizon_{key}_monotonic_ns"] for key in ("start", "end"))
        if (start, end) != (direct["start_ns"], direct["end_ns"]):
            raise ValueError("Direct worker receipt belongs to another trace horizon")
        baseline = direct["idle_calibration_tmac_s"]
        elapsed = (end-start) * 1e-9
        rows = [row["request"] for row in trace["runs"]]
        phase_seconds = {"decode": sum((r["last_token_monotonic_ns"]-r["first_token_monotonic_ns"])*1e-9 for r in rows),
                         "prefill": sum(r["ttft_s"] for r in rows)}
        scanned = (trace.get("inference_graph_work") or {}).get("independently_scanned_projection_mac_equivalent", {})
        # Switch-only is intentionally native; these zeroes describe the arm design.
        macs = {phase: 0 if name == "control" else scanned.get(phase) for phase in phase_seconds}
        parked = fleet["switching_summary"]["parked_decode"] == len(rows)
        credited = {phase: None if count is None or not parked else count/1e12/phase_seconds[phase]/baseline*100
                    for phase, count in macs.items()}
        equivalent = None
        if all(count is not None for count in macs.values()) and parked:
            added = sum(macs.values())/1e12/elapsed
            equivalent = [direct[key]+added for key in ("lower_tmac_s", "upper_tmac_s")]
        summaries[name] = dict(idle_only_reference_tmac_s=baseline, phase_wall_seconds=phase_seconds,
                               verified_projection_macs=macs, phase_percent_of_direct_reference=credited,
                               direct_park_confirmed=parked, timing=fleet["switching_summary"],
                               direct_horizon_tmac_s_bounds=[direct[k] for k in ("lower_tmac_s", "upper_tmac_s")],
                               combined_horizon_tmac_s_bounds=equivalent, unique_local_direct_receipts=direct["unique_local_verified_shares"],
                               direct_receipt_target=direct["pool_base_target"])
    a, b = (summaries[name]["combined_horizon_tmac_s_bounds"] for name in ("control", "candidate"))
    delta = [100*(b[0]/a[1]-1), 100*(b[1]/a[0]-1)] if a and b and min(a)>0 else None
    return dict(arms=summaries, combined_horizon_rate_change_percent_bounds=delta,
                basis="Four-rank completed standalone tile MACs plus independently scanned real projection ticket MAC equivalents. Phase windows use HTTP start to first token and first to last token. Direct local proofs use a separate peer/target from inference proofs.")


def compare_traces(control_path, candidate_path, final_log=None, direct_control=None, direct_candidate=None):
    control, candidate = (json.loads(p.read_text()) for p in (control_path, candidate_path))
    keys = ("trace_mode", "suite", "cap_w", "inputs", "outputs", "initial_idle_s", "schedule_period_s", "post_completion_idle_s")
    if not all(t["complete"] for t in (control, candidate)) or any(control[k] != candidate[k] for k in keys):
        raise ValueError("Need complete trial sets with the same trace and offered schedule")
    if candidate.get("projection_before") is not None:
        candidate["inference_graph_work"] = graph_work(candidate["projection_before"], candidate["projection_after"], candidate["runs"],
                                                       graph_deltas(candidate["http_before"], candidate["http_after"]))
    if final_log is not None:
        candidate["local_verified"] = local_proofs(final_log, candidate["requested_horizon_start_monotonic_ns"],
                                                   candidate["requested_horizon_end_monotonic_ns"], candidate["peer_after"])
    result = paired_rows(control["runs"], candidate["runs"])
    if (direct_control is None) != (direct_candidate is None):
        raise ValueError("Both direct worker observation receipts are required")
    if direct_control is not None:
        result["mining_efficiency"] = work_comparison(control, candidate, json.loads(direct_control.read_text()), json.loads(direct_candidate.read_text()))
    result["trace_mode"] = control["trace_mode"]
    result["control"] = trace_summary(control)
    result["candidate"] = trace_summary(candidate)
    if control["trace_mode"] == "fixed-period":
        if result["control"]["horizon_seconds"] != result["candidate"]["horizon_seconds"]:
            raise ValueError("Fixed-period traces have different horizons")
    return result


def compare_handbooks(control_path, candidate_path):
    control, candidate = (json.loads(p.read_text()) for p in (control_path, candidate_path))
    if not control["complete"] or not candidate["complete"]:
        raise ValueError("Both historical fixture runs must be complete")
    directory = Path(__file__).resolve().parent
    handbook = json.loads((directory / "historical-handbook.json").read_text())
    earlier = {name:json.loads((directory / f"{name}-handbook.json").read_text())["runs"]
               for name in ("incumbent", "stock", "pearl")}
    rows = []
    for index, (a, b, historical) in enumerate(zip(control["runs"], candidate["runs"], handbook["rows"], strict=True)):
        fields = ("target_input_tokens", "input_tokens", "output_budget", "prompt_sha256")
        if any(a[k] != b[k] for k in fields) or a["target_input_tokens"] != historical["target_input_tokens"]:
            raise ValueError("Historical fixtures, budgets, or requested lengths differ")
        result = {k:a[k] for k in fields}
        for name, row in [("switch", a), ("joined", b)] + [(name, records[index]) for name, records in earlier.items()]:
            result[name] = {k:row.get(k) for k in ("input_tokens", "output_tokens", "cached_tokens", "ttft_s",
                            "client_effective_prefill_tps","native_server_decode_tps","client_decode_tps","e2e_s","graph_delta_by_rank")}
        work = graph_work(b["projection_before"], b["projection_after"], [{"request":b}], b["graph_delta_by_rank"])
        result.update(historical_2026_09_15=historical, paired_decode_change_pct=100*(b["client_decode_tps"]/a["client_decode_tps"]-1),
                      paired_ttft_change_pct=100*(b["ttft_s"]/a["ttft_s"]-1),
                      joined_reconciled_work={k:work[k] for k in ("mined_hook_issues", "ticket_accounting_issues",
                                              "prefill_layout_reconciliation_by_rank", "independently_scanned_projection_mac_equivalent")})
        rows.append(result)
    return {"history_profile": handbook["profile"], "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op",required=True)
    trial = sub.add_parser("trial")
    trial.add_argument("--base",default="http://127.0.0.1:8000")
    trial.add_argument("--arm",required=True)
    trial.add_argument("--pair",required=True,type=int)
    trial.add_argument("--inputs",default=2048,type=int)
    trial.add_argument("--outputs",default=256,type=int)
    trial.add_argument("--receipt",required=True,type=Path)
    trace = sub.add_parser("trace")
    trace.add_argument("--base", default="http://127.0.0.1:18000")
    trace.add_argument("--arm", required=True)
    trace.add_argument("--mode", required=True, choices=("fixed-period", "fixed-gap"))
    trace.add_argument("--pairs", type=int, default=4)
    trace.add_argument("--period", type=float, default=10., help="seconds between offered requests in fixed-period mode")
    trace.add_argument("--gap", type=float, default=1., help="idle seconds after stream completion in fixed-gap mode")
    trace.add_argument("--inputs", type=int, default=2048)
    trace.add_argument("--outputs", type=int, default=256)
    trace.add_argument("--receipt", type=Path, required=True)
    trace.add_argument("--peer-ledger", type=Path)
    trace.add_argument("--accepted-log", type=Path)
    trace.add_argument("--shape-dir", type=Path, help="graph-mining arm's per-rank aggregate snapshots")
    historical = sub.add_parser("handbook")
    historical.add_argument("--base", required=True)
    historical.add_argument("--arm", required=True)
    historical.add_argument("--receipt", type=Path, required=True)
    historical.add_argument("--shape-dir", type=Path)
    compare_args = sub.add_parser("compare")
    compare_args.add_argument("--control",required=True,nargs="+",type=Path)
    compare_args.add_argument("--candidate",required=True,nargs="+",type=Path)
    compare_args.add_argument("--receipt",required=True,type=Path)
    handbook_compare = sub.add_parser("compare-handbook")
    handbook_compare.add_argument("--control", type=Path, required=True)
    handbook_compare.add_argument("--candidate", type=Path, required=True)
    handbook_compare.add_argument("--receipt", type=Path, required=True)
    trace_compare = sub.add_parser("compare-traces")
    trace_compare.add_argument("--control", required=True, type=Path)
    trace_compare.add_argument("--candidate", required=True, type=Path)
    trace_compare.add_argument("--receipt", required=True, type=Path)
    trace_compare.add_argument("--candidate-accepted-log", type=Path, help="recount the final drained verifier ledger within the candidate trace horizon")
    trace_compare.add_argument("--control-fleet", type=Path)
    trace_compare.add_argument("--candidate-fleet", type=Path)
    args=parser.parse_args()
    if args.op=="trial":
        row=run_trial(args.base,args.arm,args.pair,args.inputs,args.outputs,receipt=args.receipt)
        print(row["request"]["client_decode_tps"],row["graph_delta_by_rank"])
    elif args.op == "trace":
        result = run_trace(args)
        receipt = {k: result[k] for k in ("complete", "arm", "trace_mode", "counted_inference_requests", "send_lateness_ms", "horizon_overrun_ms", "local_verified")}
        print(json.dumps(receipt))
    elif args.op == "handbook":
        run_handbook(args)
    elif args.op == "compare-handbook":
        write(args.receipt, compare_handbooks(args.control, args.candidate))
    elif args.op == "compare-traces":
        write(args.receipt, compare_traces(args.control, args.candidate, args.candidate_accepted_log, args.control_fleet, args.candidate_fleet))
    else:
        read = lambda paths: [json.loads(p.read_text()) for p in paths]
        write(args.receipt, paired_rows(read(args.control), read(args.candidate)))


if __name__=="__main__":
    main()
