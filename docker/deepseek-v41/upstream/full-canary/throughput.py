"""Full-checkpoint TP4 public prefill, decode and concurrency benchmarks using native scheduler metrics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import statistics
import struct
import threading
import time
from pathlib import Path

from public_handbook import make_prompt
from qualify import gpu_snapshot, http_counters, request


def tokenize(base, model, text):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "chat_template_kwargs": {"thinking": False}}
    with request(base, "/v1/tokenize", body) as reply:
        return json.load(reply)["tokens"]


def nonce(name):
    return hashlib.sha256(("sm120-public-v1:" + name).encode()).hexdigest()[:32]


def exact_prompt(base, model, length, name):
    # Keep the initial role and question, and the final assistant suffix. Cut only middle reference notes.
    count = max(24, length // 65)
    while True:
        ids = tokenize(base, model, make_prompt(nonce(name), count))
        if len(ids) >= length:
            break
        count = round(count * length / len(ids)) + 8
    return ids[: length - 128] + ids[-128:]


def sse(raw):
    for line in raw:
        if not line.startswith(b"data:"):
            continue
        event = line[5:].strip()
        if event == b"[DONE]":
            break
        if event:
            yield json.loads(event)


def native_stream(base, ids, outputs, barrier, cache_salt=None):
    input_hash = hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()
    body = dict(input_ids=ids, stream=True, sampling_params=dict(temperature=0, max_new_tokens=outputs, ignore_eos=True))
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
    barrier.wait(timeout=15)
    start = time.monotonic()
    first = first_count = last = None
    gaps = []
    with request(base, "/generate", body) as reply:
        for event in sse(reply):
            now = time.monotonic()
            meta = event["meta_info"]
            count = meta["completion_tokens"]
            if count and first is None:
                first, first_count = now, count
            if last is not None:
                gaps.append(now - last)
            last = now
    end = time.monotonic()
    if first is None:
        raise RuntimeError("native stream closed before the first output token")
    if len(event["output_ids"]) != count or count != outputs or meta["prompt_tokens"] != len(ids):
        raise RuntimeError(f"expected {len(ids)} input and {outputs} output tokens, "
                           f"received {meta['prompt_tokens']} input and {count} output tokens")
    cached = meta["cached_tokens"]
    prefill = meta.get("prefill_finished_time", 0) - meta.get("forward_entry_time", 0)
    return dict(input_sha256=input_hash, input_tokens=len(ids), cached_tokens=cached, output_tokens=count,
                output_ids_sha256=hashlib.sha256(struct.pack(f"<{count}I", *event["output_ids"])).hexdigest(),
                start_monotonic=start, end_monotonic=end, ttft_s=first - start, e2e_s=end - start,
                client_effective_prefill_tps=(len(ids) - cached) / (first - start),
                native_prefill_s=prefill if prefill > 0 else None,
                native_prefill_tps=(len(ids) - cached) / prefill if prefill > 0 else None,
                native_decode_tps=meta.get("decode_throughput"),
                client_decode_tps=(count - first_count) / (last - first) if last > first else None,
                first_event_output_tokens=first_count, max_sse_gap_ms=max(gaps, default=0) * 1000,
                **{k: meta.get(k) for k in ("queue_time", "spec_accept_rate", "spec_accept_length", "spec_verify_ct", "num_retractions")})


def run_batch(base, model, name, length, outputs, concurrency):
    ids = [exact_prompt(base, model, length, f"{name}-{j}") for j in range(concurrency)]
    counter_before = http_counters(base)
    barrier = threading.Barrier(concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        calls = [pool.submit(native_stream, base, inp, outputs, barrier) for inp in ids]
        rows = [f.result() for f in calls]
    span = max(r["end_monotonic"] for r in rows) - min(r["start_monotonic"] for r in rows)
    medians = {}
    for key in ("ttft_s", "e2e_s", "client_effective_prefill_tps", "native_prefill_tps", "native_decode_tps",
                "client_decode_tps", "spec_accept_length"):
        values = [r[key] for r in rows if r[key] is not None]
        if values:
            medians[key] = statistics.median(values)
    return dict(name=name, input_per_call=length, output_per_call=outputs, concurrency=concurrency,
                http_before=counter_before, http_after=http_counters(base), expected_inference_requests=concurrency,
                batch_wall_s=span, aggregate_output_tps_including_prefill=concurrency * outputs / span,
                aggregate_total_token_tps=concurrency * (length + outputs) / span, request_medians=medians, requests=rows)


def server_histograms(base):
    names = {"request_time_per_output_token_seconds", "time_to_first_token_seconds", "e2e_request_latency_seconds",
             "uncached_prompt_tokens_histogram", "generation_tokens_histogram"}
    totals = {}
    with request(base, "/metrics", timeout=10) as resp:
        for line in resp.read().decode().splitlines():
            match = re.fullmatch(r"sglang:(\w+)_(sum|count)\{([^}]+)\}\s+([^ ]+)", line)
            if not match or match[1] not in names or 'is_streaming="false"' in match[3]:
                continue
            key = match[1] + "_" + match[2]
            totals[key] = totals.get(key, 0) + float(match[4])
    return totals


def single_request_metric(before, after, name):
    skey, ckey = name + "_sum", name + "_count"
    if skey in after and after.get(ckey, 0) - before.get(ckey, 0) == 1:
        return after[skey] - before.get(skey, 0)
    return None


def native_decode_metrics(before, after, outputs):
    tpot = single_request_metric(before, after, "request_time_per_output_token_seconds")
    if tpot and tpot > 0:
        return dict(native_tpot_histogram_decode_tps=1 / tpot, native_server_decode_tps=1 / tpot,
                    native_server_decode_source="request_time_per_output_token_seconds")
    elapsed = single_request_metric(before, after, "e2e_request_latency_seconds")
    first = single_request_metric(before, after, "time_to_first_token_seconds")
    if elapsed is not None and first is not None and elapsed > first and outputs > 1:
        # Earlier SGLang uses these same request timestamps to compute meta.decode_throughput.
        return dict(native_tpot_histogram_decode_tps=None, native_server_decode_tps=(outputs - 1) / (elapsed - first),
                    native_server_decode_source="single-request e2e minus first-token latency")
    return dict(native_tpot_histogram_decode_tps=None, native_server_decode_tps=None, native_server_decode_source=None)


def handbook(base, model, size, budget):
    # Exactly the September task generator, native chat-template calibration and sampling config.
    count = max(1, size // 100)
    for _ in range(5):
        prompt = make_prompt(nonce(f"historical-{size}"), count)
        n = len(tokenize(base, model, prompt))
        if abs(n - size) < max(150, size * .003):
            break
        count = max(1, round(count * size / n))
    else:
        raise ValueError("historical prompt calibration failed")
    payload = dict(model=model, input=[{"role": "user", "content": prompt}], stream=True, store=False,
                   temperature=1.0, top_p=.95, max_output_tokens=budget, chat_template_kwargs={"thinking": False})
    metrics_before = server_histograms(base)
    counters_before = http_counters(base)
    first = terminal = final = None
    output = hashlib.sha256()
    output_chars = 0
    log_at = 0
    start = time.monotonic()
    with request(base, "/v1/responses", payload, timeout=1200) as reply:
        for event in sse(reply):
            kind = event.get("type")
            if kind in ("response.output_text.delta", "response.reasoning_text.delta", "response.reasoning_summary_text.delta") and event.get("delta"):
                if first is None:
                    first = time.monotonic()
                output.update(event["delta"].encode())
                output_chars += len(event["delta"])
                if time.monotonic() >= log_at:
                    print(f"handbook target={size} first={first-start:.2f}s elapsed={time.monotonic()-start:.1f}s output_chars={output_chars}", flush=True)
                    log_at = time.monotonic() + 20
            if kind in ("response.completed", "response.incomplete", "response.failed"):
                terminal = time.monotonic()
                final = event["response"]
    if not first or not final or final["status"] not in ("completed", "incomplete"):
        raise RuntimeError("missing valid terminal Responses event")
    use = final["usage"]
    cached = (use.get("input_tokens_details") or {}).get("cached_tokens")
    n, out = use["input_tokens"], use["output_tokens"]
    metrics_after = server_histograms(base)
    native_uncached = single_request_metric(metrics_before, metrics_after, "uncached_prompt_tokens_histogram")
    cache_source = "responses.usage" if cached is not None else None
    if cached is None and native_uncached is not None and 0 <= native_uncached <= n:
        cached, cache_source = int(n - native_uncached), "native isolated histogram delta"
    return dict(start_monotonic=start, end_monotonic=terminal, target_input_tokens=size, input_tokens=n, cached_tokens=cached, output_tokens=out,
                output_budget=budget, prompt_nonce=nonce(f"historical-{size}"), temperature=1.0, top_p=.95,
                sampling_seed=None, terminal_status=final["status"], incomplete_details=final.get("incomplete_details"),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), output_delta_sha256=output.hexdigest(),
                **native_decode_metrics(metrics_before, metrics_after, out),
                cached_tokens_source=cache_source,
                server_metrics_before=metrics_before, server_metrics_after=metrics_after,
                http_before=counters_before, http_after=http_counters(base), expected_inference_requests=1,
                ttft_s=first - start, e2e_s=terminal - start, client_input_tokens_per_ttft=n / (first - start),
                client_decode_tps=(out - 1) / (terminal - first),
                client_effective_prefill_tps=(n - cached) / (first - start) if cached is not None else None)


def write(path, result):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--phase", choices=("native", "handbook", "prefix", "eager-control"), required=True)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--control-state", choices=("job", "plain"))
    p.add_argument("--handbook-sizes", nargs="+", type=int, default=[32768])
    p.add_argument("--handbook-long-budget", type=int, default=8192, help="output cap for handbook inputs larger than 32K")
    args = p.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    with request(base, "/server_info") as resp:
        info = json.load(resp)
    assert (info["tp_size"], info["ep_size"]) == (4, 4)
    result = dict(profile={k: info.get(k) for k in ("model_path","version","tp_size","ep_size","context_length",
                        "max_total_num_tokens","mem_fraction_static","speculative_algorithm","cuda_graph_config","random_seed")},
                  phase=args.phase, start_utc_epoch=time.time(), gpu_before=gpu_snapshot(), http_before=http_counters(base), runs=[])
    write(args.receipt, result)
    model = info["served_model_name"]
    if args.phase == "native":
        for length in (2048, 8192, 32768):
            for rep in range(args.trials):
                row = run_batch(base, model, f"single-{length}-{rep}", length, 256, 1)
                result["runs"].append(row)
                write(args.receipt, result)
                print(row["name"], row["request_medians"], flush=True)
        for width in (1, 2, 4, 8):
            row = run_batch(base, model, f"parallel-{width}", 2048, 128, width)
            result["runs"].append(row)
            write(args.receipt, result)
            print(row["name"], row["aggregate_output_tps_including_prefill"], flush=True)
    elif args.phase == "eager-control":
        if not args.control_state:
            p.error("--phase eager-control requires --control-state job or plain")
        if args.control_state == "job":
            ids = exact_prompt(base, model, 2048, "control-shape-prime")
            result["unmeasured_warmup"] = native_stream(
                base, ids, 128, threading.Barrier(1), cache_salt="full-checkpoint-control-shape-prime"
            )
            write(args.receipt, result)
        for rep in range(3):
            ids = exact_prompt(base, model, 2048, f"matched-control-{rep}")
            salt = f"full-checkpoint-eager-{args.control_state}-{rep}"
            row = native_stream(base, ids, 128, threading.Barrier(1), cache_salt=salt)
            row.update(name=f"matched-control-{rep}", cache_salt=salt, control_state=args.control_state)
            result["runs"].append(row)
            write(args.receipt, result)
            print(row["name"], row["cached_tokens"], row["native_decode_tps"], flush=True)
    elif args.phase == "prefix":
        prefix = exact_prompt(base, model, 8192, "shared-prefix-cold-v1")
        for i, name in enumerate(("cold", "warm")):
            # Distinct suffixes after exactly 8,192 identical tokens.
            ids = prefix + [9103 + i] + list(range(3170, 3185))
            row = native_stream(base, ids, 1, threading.Barrier(1))
            row.update(name=name, shared_prefix_tokens=len(prefix), suffix_tokens=len(ids) - len(prefix))
            result["runs"].append(row)
            write(args.receipt, result)
            print(name, row["cached_tokens"], row["ttft_s"], flush=True)
            time.sleep(.35)
    else:
        for size in args.handbook_sizes:
            row = handbook(base, model, size, 8192 if size <= 32768 else args.handbook_long_budget)
            result["runs"].append(row)
            write(args.receipt, result)
            print({k:row[k] for k in ("input_tokens","output_tokens","ttft_s","client_decode_tps","e2e_s")}, flush=True)
    expected = sum(row.get("concurrency", 1) for row in result["runs"]) + int("unmeasured_warmup" in result)
    result.update(gpu_after=gpu_snapshot(), end_utc_epoch=time.time(), complete=True, http_after=http_counters(base),
                  expected_inference_requests=expected)
    write(args.receipt, result)


if __name__ == "__main__":
    main()
