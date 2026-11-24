"""Verify prefill and DSPARK shares on every TP rank during isolated full-model public requests."""
import argparse
import json
import time
from pathlib import Path

from qualify import request


def rank_shapes(directory):
    shapes = {}
    for file in sorted(directory.glob("pearl-shapes-*-tp*.json")):
        rank = int(file.stem.rsplit("tp", 1)[1])
        if rank in shapes:
            raise ValueError("shape directory contains multiple schedulers for a rank")
        shapes[rank] = {(r["mode"], r["real_rows"], r["executed_rows"]): r["calls"]
                        for r in json.loads(file.read_text()) if r["path"] == "eager_wo_b"}
    return shapes


def run(options):
    base = f"http://127.0.0.1:{options.port}"
    with request(base, "/server_info") as reply:
        info = json.load(reply)
    if (info.get("tp_size"), info.get("ep_size"), info.get("speculative_algorithm")) != (4, 4, "DSPARK"):
        raise ValueError("expected TP4/EP4 full model with DSPARK")
    rows = 16
    width = info.get("speculative_num_draft_tokens")
    if not width or width == rows:
        raise ValueError("prompt and verify widths must distinguish the two proof phases")
    baseline = json.loads(options.pool_receipt.read_text())
    before = rank_shapes(options.shape_dir)
    accepted_before = len(baseline["accepted"])
    result = dict(scope="isolated public requests on full TP4; ongoing peer ACKs correlated by shape",
                  checkpoint=info.get("model_path"), version=info.get("version"), tp=4, ep=4,
                  prefill_rows=rows, spec_verify_rows=width, expected_ranks=[0, 1, 2, 3], requests=[],
                  prior_pool_acknowledgments=accepted_before, verified=[], rank_shapes={}, passed=False)
    for trial in range(options.attempts):
        # A distinct first token prevents the public prefix from turning this into a cache hit.
        ids = [30001 + trial] + [567 + j % 3 for j in range(rows - 1)]
        payload = dict(input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": 16, "ignore_eos": True})
        started = time.time()
        first = time.monotonic_ns()
        with request(base, "/generate", payload) as reply:
            out = json.load(reply)
        end = time.monotonic_ns()
        meta = out["meta_info"]
        if meta.get("completion_tokens") != 16 or len(out["output_ids"]) != 16:
            raise ValueError("unexpected public token count")
        result["requests"].append(dict(trial=trial, first_input_id=ids[0], start_time=started, end_time=time.time(),
                                       start_monotonic_ns=first, end_monotonic_ns=end,
                                       wall_ms=(end - first) / 1e6, prompt_tokens=meta["prompt_tokens"],
                                       cached_tokens=meta.get("cached_tokens"), output_ids=out["output_ids"],
                                       spec_verify_ct=meta.get("spec_verify_ct")))
        deadline = time.monotonic() + 4
        while True:
            pool = json.loads(options.pool_receipt.read_text())
            current_shapes = rank_shapes(options.shape_dir)
            accepted = pool["accepted"][accepted_before:]
            if pool["started_at"] != baseline["started_at"] or len(pool["accepted"]) < accepted_before:
                raise RuntimeError("the local verifier was restarted during the test")
            result["verified"] = [r for r in accepted if r["received_at"] >= result["requests"][0]["start_time"]]
            result["distinct_verified_sha256"] = len({r["proof_sha256"] for r in result["verified"]})
            result["distinct_by_rows"] = {
                str(m): len({r["proof_sha256"] for r in result["verified"] if r["proof_rows"] == m})
                for m in sorted({r["proof_rows"] for r in result["verified"]})
            }
            all_pass = True
            for rank in range(4):
                modes = {}
                for mode, m in (("EXTEND", rows), ("TARGET_VERIFY", width)):
                    shape = (mode, m, m)
                    delta = current_shapes.get(rank, {}).get(shape, 0) - before.get(rank, {}).get(shape, 0)
                    count = sum(r["tp_rank"] == rank and r["proof_rows"] == m for r in result["verified"])
                    modes[mode] = dict(observed_unpadded_calls=delta, independently_verified=count)
                    all_pass = all_pass and delta > 0 and count > 0
                result["rank_shapes"][str(rank)] = modes
            result["passed"] = all_pass and pool["rejected_frames"] == baseline["rejected_frames"]
            temporary = options.receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(options.receipt)
            if result["passed"] or time.monotonic() > deadline:
                break
            time.sleep(.25)
        if result["passed"] and len(result["requests"]) >= options.min_requests:
            break
        # Let the independent serving-rank fence and the real idle miners complete
        # a transition before admitting the next cold-prefix public request.
        time.sleep(.35)
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--shape-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--attempts", type=int, choices=range(1, 17), default=6)
    parser.add_argument("--min-requests", type=int, choices=range(1, 17), default=1)
    args = parser.parse_args()
    if args.min_requests > args.attempts:
        parser.error("--min-requests cannot exceed --attempts")
    if not run(args):
        raise SystemExit("missing verified shares or scheduler phases; see receipt")
