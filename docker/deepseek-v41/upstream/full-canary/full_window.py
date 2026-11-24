"""Replay the archived public 524K Responses prefill with its original payload hash."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re
import time

from qualify import gpu_snapshot, request

DIGEST = "ff7319cb893282458cf77fdcef75ca5aebe0b8a5145708fe616bdab0cdd467df"


def payload():
    # Recovered from the saved tokenizer output of the September 16 public
    # fixture. The original run used this exact nonce, repeat count and template.
    notes = "Reference record: the cache stores recent entries; preserve ordering and validate invariants.\n"
    prompt = ("Unique qualification aa7796d3054744618d40121affc09bb9. Read reference notes; do not summarize them.\n"
              + notes * 32758 + "\nThe requested verification number is 42. Reply with exactly 42 and nothing else.")
    data = dict(model="deepseek-v4-flash", input=[dict(role="user", content=prompt)], temperature=0,
                max_output_tokens=64, chat_template_kwargs=dict(thinking=False), stream=True, store=False)
    archived = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()
    if hashlib.sha256(archived).hexdigest() != DIGEST:
        raise ValueError("archived payload checksum mismatch")
    return data


def counters(base):
    names = {
        "sglang:prompt_tokens_total", "sglang:cached_tokens_total", "sglang:generation_tokens_total",
        "sglang:time_to_first_token_seconds_sum", "sglang:time_to_first_token_seconds_count",
        "sglang:queue_time_seconds_sum", "sglang:num_retracted_reqs_total",
        "sglang:per_stage_req_latency_seconds_sum", "sglang:per_stage_req_latency_seconds_count",
        "sglang:prefill_effective_tokens_total", "sglang:http_requests_total",
    }
    with request(base, "/metrics") as resp:
        lines = resp.read().decode().splitlines()
    result = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        sample, value = line.rsplit(None, 1)
        name = sample.split("{", 1)[0]
        if name not in names:
            continue
        rank = re.search(r'tp_rank="(\d+)"', sample)
        if rank and rank.group(1) != "0":
            continue
        if name.endswith("http_requests_total") and not any(path in sample for path in ("/generate", "/v1/responses", "/v1/messages", "/v1/completions", "/v1/chat/completions")):
            continue
        result[sample] = float(value)
    return result


def run(base):
    body = payload()
    start_counters = counters(base)
    hardware = gpu_snapshot()
    first = None
    terminal = None
    answer = []
    transport_error = None
    start = time.monotonic()
    try:
        with request(base, "/v1/responses", body, timeout=1200) as resp:
            for line in resp:
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == b"[DONE]":
                    continue
                event = json.loads(raw)
                if event["type"] == "response.output_text.delta" and event.get("delta"):
                    if first is None:
                        first = time.monotonic()
                    answer.append(event["delta"])
                elif event["type"] in ("response.completed", "response.incomplete", "response.failed", "error"):
                    terminal = event
                    break
    except (OSError, http.client.HTTPException) as exc:
        transport_error = type(exc).__name__
    end = time.monotonic()
    if first is None or not terminal or terminal["type"] != "response.completed":
        return dict(status="failed", payload_sha256=DIGEST, fixture_input_tokens=524177,
                    input_tokens=None, cached_tokens=None, answer_check=False,
                    start_monotonic=start, end_monotonic=end, wall_seconds=end-start,
                    first_text_seconds=first-start if first is not None else None, gpus=hardware,
                    terminal_event=terminal.get("type") if terminal else None, transport_error=transport_error,
                    server_prefill_seconds=None, server_uncached_prefill_tokens_per_second=None)
    usage = terminal["response"]["usage"]
    finish_counters = counters(base)
    delta = {k: v - start_counters.get(k, 0.0) for k, v in finish_counters.items() if v != start_counters.get(k, 0.0)}
    count = usage.get("input_tokens", usage.get("prompt_tokens"))
    cached = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cache_delta = sum(v for k,v in delta.items() if k.split("{")[0] == "sglang:cached_tokens_total")
    if "cached_tokens" in cached:
        cached_count = cached["cached_tokens"]
    elif any(k.startswith("sglang:cached_tokens_total") for k in finish_counters):
        cached_count = int(cache_delta)
    elif any("sglang:prefill_effective_tokens_total" in k and 'mode="device_hit"' in k
             for k in finish_counters):
        cached_count = int(sum(v for k,v in delta.items()
                               if k.startswith("sglang:prefill_effective_tokens_total")
                               and any(f'mode="{tier}_hit"' in k for tier in ("device", "host", "storage"))))
    else:
        cached_count = None
    # This histogram records prefill-finished minus forward-entry for a request
    # without retractions. Attribute it only when exactly one sample advanced.
    stage = {k.split("{", 1)[0]: v for k, v in delta.items()
             if k.startswith("sglang:per_stage_req_latency_seconds_") and 'stage="prefill_forward"' in k}
    prefill = (stage.get("sglang:per_stage_req_latency_seconds_sum")
               if stage.get("sglang:per_stage_req_latency_seconds_count") == 1 else None)
    if any(v > 0 for k, v in delta.items() if k.startswith("sglang:num_retracted_reqs_total")):
        prefill = None
    return dict(payload_sha256=DIGEST, input_tokens=count, start_monotonic=start, end_monotonic=end,
                server_prefill_seconds=prefill,
                server_uncached_prefill_tokens_per_second=(count-cached_count)/prefill if prefill and cached_count is not None else None, cached_tokens=cached_count, usage=usage,
                completion_text="".join(answer), answer_check="".join(answer).strip() == "42",
                first_text_seconds=first-start, wall_seconds=end-start,
                effective_client_input_tokens_per_second=count/(first-start),
                effective_client_uncached_tokens_per_second=(count-cached_count)/(first-start) if cached_count is not None else None,
                metric_deltas=delta, gpus=hardware)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args=parser.parse_args()
    result=run(f"http://127.0.0.1:{args.port}")
    args.receipt.write_text(json.dumps(result,indent=2)+"\n")
    print({k:v for k,v in result.items() if k not in ("gpus","metric_deltas","usage")})
    if not result["answer_check"]:
        raise SystemExit(1)
