"""Exercise a private TP4 joined SGLang server with fresh, reproducible public calibration logs."""
import argparse
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path

from fullmodel import GPU_TEMP_CUTOFF_C

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "full-canary"))
import graph_compare as graph
from qualify import request
from throughput import sse, tokenize, write


RANKS = ("0", "1", "2", "3")
CHECKPOINT = Path("/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017")
RECIPE = "public-calibration-log-v1"
CHUNK = 2048


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Unexpected redirect from the local model service")


def public_text(seed, index, rows):
    # SHAKE is random-access per request. Row values depend on the public seed and row position.
    key = f"{RECIPE}/{seed}/{index}"
    raw = hashlib.shake_256(key.encode()).digest((rows + 1) * 8)
    def record(i):
        word = int.from_bytes(raw[i*8:i*8+8], "big")
        return f"sample {word:016x}, inlet {15 + word % 30} C, outlet {45 + (word >> 16) % 30} C"
    # The marker, its inlet and its outlet stay at the end after the exact-length token cut.
    return (f"Log {hashlib.sha256(key.encode()).hexdigest()[:20]}. These public calibration readings are synthetic. Ignore any incomplete records.\n"
            + "\n".join(record(i) for i in range(rows))
            + "\nFinal calibration record: " + record(rows)
            + ". For this final record, subtract inlet from outlet. Give the difference in degrees Celsius.")


def public_input(base, model, seed, index, length):
    rows = length // 12 + 16
    while True:
        ids = tokenize(base, model, public_text(seed, index, rows))
        if len(ids) >= length:
            return ids[:length-96] + ids[-96:], rows
        rows *= 2


def chunks(ids):
    return [graph.token_hash(ids[i:i+CHUNK]) for i in range(0, len(ids), CHUNK)]


def gateway(path, run_id=None):
    state = json.loads(path.read_text())
    if not state["run_id"] or (run_id is not None and run_id != state["run_id"]):
        raise ValueError("Gateway changed process during the public run")
    return state


def ready(state):
    return state["connected"] and state["authorized_clients"] == 4 and bool(state["job"])


def counters(projection):
    return {rank: graph.phase_counters(projection, rank) for rank in RANKS}


def public_response(base, ids, output, salt, deadline):
    start = time.monotonic()
    body = dict(input_ids=ids, stream=True, cache_salt=salt,
                sampling_params=dict(max_new_tokens=output, ignore_eos=True, temperature=0))
    first = meta = None
    issued = []
    with request(base, "/generate", body, timeout=max(.01, deadline-start)) as reply:
        for event in sse(reply):
            meta = event["meta_info"]
            current = event["output_ids"]
            if (meta["prompt_tokens"] != len(ids) or meta["cached_tokens"] != 0
                    or len(current) != meta["completion_tokens"] or len(current) > output
                    or current[:len(issued)] != issued):
                raise ValueError("Public request token or cache counts disagree")
            issued = current
            if issued and first is None:
                first = time.monotonic()
            if time.monotonic() >= deadline and len(issued) != output:
                break
    if not issued or meta is None:
        raise RuntimeError("No model-issued tokens were observed")
    return dict(input_tokens=len(ids), cached_tokens=meta["cached_tokens"], output_tokens=len(issued),
                output_ids=issued, output_sha256=graph.token_hash(issued), completed=len(issued) == output,
                ttft_s=first-start, elapsed_s=time.monotonic()-start,
                spec_verify_ct=meta.get("spec_verify_ct"), native_decode_tps=meta.get("decode_throughput"))


class ArmWindow:
    """Disarm even during a long prefill or quiet SSE stream."""
    def __init__(self, arm, receipt, run_id, seconds):
        self.arm, self.receipt, self.run_id, self.seconds = arm, receipt, run_id, seconds
        self.done = threading.Event()
        self.error = None
        self.reason = None

    def __enter__(self):
        self.start = time.monotonic()
        self.deadline = self.start + self.seconds
        self.arm.touch(mode=0o600, exist_ok=False)
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def watch(self):
        try:
            while not self.done.is_set():
                state = gateway(self.receipt, self.run_id)
                if state["accepted"] > 0:
                    self.reason = "pool_accepted"
                    break
                if time.monotonic() >= self.deadline:
                    self.reason = "budget_elapsed"
                    break
                self.done.wait(min(.1, max(0., self.deadline-time.monotonic())))
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()
            self.arm.unlink(missing_ok=True)

    def __exit__(self, *unused):
        self.done.set()
        self.thread.join()
        if self.error:
            raise self.error


def stable_job(before, after):
    # The gateway increments the local revision even if a reconnect or A->B->A
    # transition returns to the same external template, and on every arm edge.
    return before["job"] if (before["armed"] and after["armed"] and ready(before) and ready(after)
                             and before["job"] == after["job"]) else None


def expected_shares(job, macs):
    # The device searches the compact-rounded pool target scaled by the MACs per ticket.
    bits = job["share_bits"]
    mantissa, shift = bits & 0x7fffff, 8 * ((bits >> 24) - 3)
    floor = mantissa << shift if shift >= 0 else mantissa >> -shift
    if not 0 < floor <= int(job["target"], 16):
        raise ValueError("Invalid compact pool target")
    return macs * floor / (1 << 256)


def projection_totals(doc, rank):
    graph_calls = sum(row["calls"] for row in doc["shapes"][rank]
                      if row["mode"] == "TARGET_VERIFY" and row["path"] == "target_graph")
    prefill = graph.phase_counters(doc, rank).get("prefill_jobs", {})
    return graph_calls, sum(n for key, n in prefill.items() if key.endswith("_rows"))


def request_projection(directory, before, graphs, input_tokens):
    # An idle callback can initially present an old, internally drained snapshot.
    # Reconcile its progress with independent HTTP graph metrics and issued prompt rows.
    wanted = {}
    for rank in RANKS:
        calls, rows = projection_totals(before, rank)
        wanted[rank] = (calls + graphs[rank].get("decode_cuda_graph", 0), rows + input_tokens)
    deadline = time.monotonic() + 6
    after = graph.drained_projection(directory)
    while True:
        counts = {rank:projection_totals(after, rank) for rank in RANKS}
        if wanted == counts and not graph.pending_scans(after):
            return after
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Rank counters failed to settle: expected {wanted}, observed {counts}")
        time.sleep(.1)
        after = graph.projection_snapshot(directory)


def aligned_prefill_macs(before, after, rank, jobs, scan):
    # A job change or backpressure skips some chunks. Each remaining mined chunk
    # still has an exact wide-ticket layout if every observed shape is aligned.
    def observed(doc):
        return {(row["real_rows"], row["executed_rows"]):row["calls"] for row in doc["shapes"][rank]
                if row["mode"] == "EXTEND" and row["path"] == "eager_wo_b"}
    old, new = observed(before), observed(after)
    total = 0
    for real, executed in old.keys() | new.keys():
        n = new.get((real,executed), 0) - old.get((real,executed), 0)
        if n < 0 or (n and (real != executed or real <= 0 or real % 16)):
            return None
        total += n * real
    rows = jobs.get("mined_rows", 0)
    if (total == sum(n for key,n in jobs.items() if key.endswith("_rows"))
            and jobs.get("mined_calls", 0) == scan.get("verified_steps", 0)
            and not scan.get("verification_errors", 0) and 40 * rows == scan.get("tickets_scanned", 0)):
        return rows * 40 * 262_144
    return None


def scan_summary(before, after, graphs):
    work = graph.graph_work(before, after, [], graphs)
    counts = work["verifier_delta_by_rank"]
    phases = {"prefill":{}, "decode":{}}
    for rank in RANKS:
        for phase in phases:
            scan = counts[rank].get(phase + "_scan", {})
            phases[phase][rank] = int(scan.get("tickets_scanned", 0))
    # A short greedy request can produce no eligible decode replay. A fully
    # reconciled prefill still counts on its own, independent of decode selection.
    macs = {}
    layouts = work["prefill_layout_reconciliation_by_rank"]
    if not graph.pending_scans(before) and not graph.pending_scans(after):
        prefill = {}
        for rank in RANKS:
            if layouts[rank] is not None:
                prefill[rank] = layouts[rank]["scanned_ticket_mac_equivalent"]
            else:
                data = counts[rank]
                prefill[rank] = aligned_prefill_macs(before, after, rank, data.get("prefill_jobs", {}),
                                                     data.get("prefill_scan", {}))
        if all(n is not None for n in prefill.values()):
            macs["prefill"] = sum(prefill.values())
            work["ticket_accounting_issues"]["prefill"] = []
        issues = work["ticket_accounting_issues"]["decode"]
        if not issues and all(not any(
                row["executed_rows"] != row["real_rows"] for name, row in by_path.items()
                if name == "target_graph_mined") for by_path in work["shape_delta_by_rank"].values()):
            macs["decode"] = sum(phases["decode"].values()) * 65_536
    return dict(tickets=phases, reconciled_mac_equivalent=macs, graph_deltas=graphs,
                shape_delta=work["shape_delta_by_rank"], rank_verifier_delta=counts,
                hook_issues=work["mined_hook_issues"], ticket_issues=work["ticket_accounting_issues"])


def temperature(path):
    try:
        sample = json.loads(path.read_text())
        age = time.time_ns() - sample["timestamp_ns"]
        temps = sample["temperatures_c"]
        if (0 <= age <= 5_000_000_000 and len(temps) == 4
                and all(type(t) is int and 0 <= t <= 120 for t in temps) and max(temps) == sample["max_c"]):
            return sample
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        pass
    return None


def check_admission_temperature(path, window):
    """Wait for a fresh guard sample and close the request window at the cutoff."""
    started = time.monotonic()
    paused = False
    peak = None
    while not window.done.is_set():
        sample = temperature(path)
        if sample is not None:
            peak = sample["max_c"]
            if peak >= GPU_TEMP_CUTOFF_C:
                window.reason = window.reason or "temperature_cutoff"
                window.done.set()
            break
        paused = True
        window.done.wait(.25)
    return dict(paused_s=time.monotonic() - started if paused else 0., peak_c=peak)


def idle_http(base):
    state = graph.snapshot(base)
    if state["active"]:
        raise RuntimeError("The private model endpoint has an overlapping inference request")
    return state


def run(args):
    # Use the same loopback-only transport for the shared native metrics and tokenizer helpers.
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({}), RejectRedirect()))
    os.environ["CANARY_API_KEY_FILE"] = str(args.key_file)
    base = f"http://127.0.0.1:{args.port}"
    profile = graph.profile(base)
    if Path(profile["model_path"]).resolve() != CHECKPOINT.resolve() or profile["chunked_prefill_size"] != CHUNK:
        raise ValueError("Expected the pinned full checkpoint and 2,048-token chunked prefill")
    initial = gateway(args.gateway_receipt)
    if not ready(initial) or initial["armed"] or initial["accepted"] or initial["submitted"] or args.arm_file.exists():
        raise ValueError("Expected a fresh disarmed gateway with four authorized ranks and a live job")
    run_id = initial["run_id"]
    http = idle_http(base)
    rank = graph.drained_projection(args.shape_dir)
    if graph.pending_scans(rank):
        raise ValueError("Startup smoke verification is still in flight")
    # Tokenize the first fixture before arming. Tokenization never asks for a completion.
    ids, rows = public_input(base, profile["served_model_name"], run_id, 0, args.input_tokens)
    details = args.receipt.with_suffix(".requests.jsonl")
    summary = dict(recipe=RECIPE, seed=run_id, chunk_tokens=CHUNK, requested_input_tokens=args.input_tokens,
                   requested_output_tokens=args.output_tokens, budget_s=args.seconds, profile=profile,
                   startup_http=http, startup_rank_counters=counters(rank), initial_gateway=initial,
                   request_log=details.name, requests_admitted=0, completed=0, completed_input_tokens=0,
                   model_issued_tokens=0, stable_job_exposure={}, unattributed_mac_equivalent=0,
                   first_request_monotonic_ns=None, latest_scan_monotonic_ns=None,
                   scanned_tickets={p:{r:0 for r in RANKS} for p in ("prefill","decode")},
                   graph_passes={r:0 for r in RANKS}, estimated_shares_on_stable_jobs=0.,
                   layout_unreconciled_scanned_mac_bounds=[0,0],
                   max_measured_request_s=0., cooldown_s=0., cooldown_events=0,
                   peak_admission_c=None, temperature_guard_used=bool(args.temperature_file),
                   mined_decode_graph_passes={r:0 for r in RANKS},
                   prefill_mined_rows={r:0 for r in RANKS}, prefill_probed_rows={r:0 for r in RANKS})
    seen = set()
    # Refuse to overwrite either receipt, including the per-request audit trail.
    with args.receipt.open("x"):
        pass
    with details.open("x", buffering=1) as journal:
        write(args.receipt, summary)
        with ArmWindow(args.arm_file, args.gateway_receipt, run_id, args.seconds) as window:
            summary["window_start_monotonic_ns"] = round(window.start * 1e9)
            index = 0
            preceding_job = None
            latest_scan = window.start
            try:
                while not window.done.is_set():
                    if time.monotonic() - latest_scan > args.no_scan_seconds and index >= 3:
                        summary["stop_reason"] = "no_verified_ticket_feed"
                        break
                    if index >= args.max_requests:
                        summary["stop_reason"] = "request_limit"
                        break
                    if index and window.deadline - time.monotonic() < summary["max_measured_request_s"] * 1.05:
                        # Keep watching for outstanding acknowledgements before closing the authorized window.
                        window.done.wait(max(0., window.deadline - time.monotonic()) + .2)
                        break
                    peer = gateway(args.gateway_receipt, run_id)
                    if not ready(peer) or not peer["armed"]:
                        window.done.wait(.1)
                        continue
                    if index:
                        ids, rows = public_input(base, profile["served_model_name"], run_id, index, args.input_tokens)
                    chunk_hashes = chunks(ids)
                    if len(set(chunk_hashes)) != len(chunk_hashes) or seen.intersection(chunk_hashes):
                        raise ValueError("Duplicate public prefill chunk")
                    request_hash = graph.token_hash(ids)
                    if args.temperature_file:
                        cooling = check_admission_temperature(args.temperature_file, window)
                        if cooling["peak_c"] is not None:
                            summary["peak_admission_c"] = max(summary["peak_admission_c"] or 0, cooling["peak_c"])
                        if cooling["paused_s"]:
                            summary["cooldown_s"] += cooling["paused_s"]
                            summary["cooldown_events"] += 1
                            # Deliberately idle time cannot diagnose a missing verifier feed.
                            latest_scan += cooling["paused_s"]
                            journal.write(json.dumps(dict(event="cooldown", index=index, **cooling)) + "\n")
                            write(args.receipt, summary)
                    if window.done.is_set():
                        break
                    if index and window.deadline-time.monotonic() < summary["max_measured_request_s"] * 1.05:
                        continue
                    peer = gateway(args.gateway_receipt, run_id)
                    if not ready(peer) or not peer["armed"] or peer["accepted"] or window.done.is_set():
                        continue
                    start_http = idle_http(base)
                    if start_http["total"] != http["total"]:
                        raise RuntimeError("Unexpected inference on the private endpoint between public requests")
                    seen.update(chunk_hashes)
                    summary["requests_admitted"] += 1
                    started_ns = time.monotonic_ns()
                    if index == 0:
                        summary["first_request_monotonic_ns"] = started_ns
                    journal.write(json.dumps(dict(event="send", index=index, input_sha256=request_hash,
                                                  start_monotonic_ns=started_ns, rows=rows,
                                                  chunks_sha256=chunk_hashes, job=peer["job"])) + "\n")
                    response = public_response(base, ids, args.output_tokens, f"{run_id}-{index}", window.deadline)
                    end_http = graph.completed_http_snapshot(base, start_http)
                    if end_http["active"] or end_http["total"] != start_http["total"] + 1:
                        raise RuntimeError("HTTP accounting could not isolate this public inference request")
                    graphs = graph.graph_deltas(start_http, end_http)
                    end_rank = request_projection(args.shape_dir, rank, graphs, len(ids))
                    end_peer = gateway(args.gateway_receipt, run_id)
                    scans = scan_summary(rank, end_rank, graphs)
                    # Keep the first observation after a local job change unassigned while ranks
                    # asynchronously retire old tensors and prepare the new job.
                    current_job = stable_job(peer, end_peer)
                    job = current_job if current_job == preceding_job else None
                    preceding_job = current_job
                    by_phase = scans["reconciled_mac_equivalent"]
                    macs = sum(by_phase.values())
                    for phase in ("prefill", "decode"):
                        if phase not in by_phase:
                            tickets = sum(scans["tickets"][phase].values())
                            summary["layout_unreconciled_scanned_mac_bounds"][0] += tickets * 65_536
                            summary["layout_unreconciled_scanned_mac_bounds"][1] += tickets * (262_144 if phase == "prefill" else 65_536)
                    if job:
                        exposure = summary["stable_job_exposure"].setdefault(job["id"],
                                dict(job=job, requests=0, mac_equivalent=0, macs_by_phase={"prefill":0, "decode":0}, estimated_shares=0.))
                        exposure["requests"] += 1
                        exposure["mac_equivalent"] += macs
                        for phase, amount in by_phase.items():
                            exposure["macs_by_phase"][phase] += amount
                        chance = expected_shares(job, macs)
                        exposure["estimated_shares"] += chance
                        summary["estimated_shares_on_stable_jobs"] += chance
                    else:
                        summary["unattributed_mac_equivalent"] += macs
                    for phase, by_rank in scans["tickets"].items():
                        for r, n in by_rank.items():
                            summary["scanned_tickets"][phase][r] += n
                    if any(n for values in scans["tickets"].values() for n in values.values()):
                        latest_scan = time.monotonic()
                        summary["latest_scan_monotonic_ns"] = round(latest_scan * 1e9)
                    for r in RANKS:
                        summary["graph_passes"][r] += graphs[r].get("decode_cuda_graph", 0)
                        decode = scans["shape_delta"][r].get("target_graph_mined", {})
                        summary["mined_decode_graph_passes"][r] += decode.get("calls", 0)
                        jobs = scans["rank_verifier_delta"][r].get("prefill_jobs", {})
                        summary["prefill_mined_rows"][r] += jobs.get("mined_rows", 0)
                        summary["prefill_probed_rows"][r] += sum(n for key,n in jobs.items() if key.endswith("_rows"))
                    summary["max_measured_request_s"] = max(summary["max_measured_request_s"],
                                                             time.monotonic() - started_ns / 1e9)
                    summary["model_issued_tokens"] += response["output_tokens"]
                    if response["completed"]:
                        summary["completed"] += 1
                        summary["completed_input_tokens"] += len(ids)
                    journal.write(json.dumps(dict(event="result", index=index, response=response, scans=scans,
                                                  end_monotonic_ns=time.monotonic_ns(), end_job=end_peer["job"],
                                                  stable_window=bool(job))) + "\n")
                    index += 1
                    http, rank = end_http, end_rank
                    write(args.receipt, summary)
            except TimeoutError:
                if time.monotonic() + .1 < window.deadline:
                    raise
                summary["stop_reason"] = "budget_elapsed_during_request"
            finally:
                summary["stop_reason"] = window.reason or summary.get("stop_reason", "interrupted")
                summary["last_completed_rank_counters"] = counters(rank)
                summary["end_http"] = http
                write(args.receipt, summary)
    # An upstream response may still be in flight when the final request completes.
    drain_end = time.monotonic() + 27
    while True:
        final = gateway(args.gateway_receipt, run_id)
        if (not final["pending"] and not final["armed"]) or time.monotonic() >= drain_end:
            break
        time.sleep(.1)
    summary["gateway_disarm_observed"] = not final["armed"]
    summary["final_gateway"] = final
    summary["shutdown_rank_counters"] = counters(graph.drained_projection(args.shape_dir))
    summary["stopped_monotonic_ns"] = time.monotonic_ns()
    summary["positive_scan_on_all_ranks"] = {
        p: all(by_rank.values()) for p,by_rank in summary["scanned_tickets"].items()}
    summary["decode_graph_on_all_ranks"] = all(summary["graph_passes"].values())
    summary["poisson_p_at_least_one_on_measured_stable_jobs"] = -math.expm1(-summary["estimated_shares_on_stable_jobs"])
    if final["accepted"]:
        ack = final.get("first_accepted") or final["last_accepted"]
        if (not summary["requests_admitted"] or ack["outcome"] != "accepted"
                or ack["sent_monotonic_ns"] < summary["first_request_monotonic_ns"]):
            raise ValueError("Pool acknowledgement precedes the armed public workload")
        summary["stop_reason"] = "pool_accepted"
        summary["pool_ack"] = ack
    write(args.receipt, summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=18002)
    for option in ("key-file", "shape-dir", "gateway-receipt", "arm-file", "receipt"):
        p.add_argument("--" + option, type=Path, required=True)
    p.add_argument("--seconds", type=int, required=True)
    p.add_argument("--temperature-file", type=Path,
                   help=f"Wait for a fresh guard sample before each request and end the run at {GPU_TEMP_CUTOFF_C} C")
    p.add_argument("--no-scan-seconds", type=int, default=120)
    p.add_argument("--max-requests", type=int, default=1_000_000)
    p.add_argument("--input-tokens", type=int, default=16_384)
    p.add_argument("--output-tokens", type=int, default=8)
    args = p.parse_args()
    if (not 1 <= args.port <= 65535 or args.seconds <= 0 or args.max_requests < 1 or args.no_scan_seconds < 1
            or not 8192 <= args.input_tokens <= 32768 or args.input_tokens % CHUNK
            or not 1 <= args.output_tokens <= 256):
        p.error("Invalid port, duration, request limit or token budget (inputs must be 2,048-aligned)")
    # systemd stops the sampler with SIGTERM on failback. Exit through the arm context.
    def stop(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    result = run(args)
    print(json.dumps({k:result[k] for k in ("stop_reason","completed","positive_scan_on_all_ranks",
                                           "decode_graph_on_all_ranks","model_issued_tokens")}))
    healthy = result["gateway_disarm_observed"] and (result.get("pool_ack") or (
        all(result["positive_scan_on_all_ranks"].values()) and result["decode_graph_on_all_ranks"]))
    return 0 if healthy else 2


if __name__ == "__main__":
    sys.exit(main())
