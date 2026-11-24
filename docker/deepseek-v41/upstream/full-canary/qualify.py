"""Replay public, greedy HTTP fixtures against an incumbent or reconstructed TP4 engine."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from dsv41_load import activity_metrics


def gpu_snapshot():
    fields = ("index", "uuid", "pstate", "clocks.current.sm", "clocks.current.memory", "power.limit", "memory.used")
    query = ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    try:
        output = subprocess.check_output(query, text=True, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return [dict(zip(fields, (s.strip() for s in row.split(",")), strict=True)) for row in output.splitlines()]


def request(base, path, body=None, timeout=300):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if keyfile := os.environ.get("CANARY_API_KEY_FILE"):
        value = Path(keyfile).read_text().strip()
        if not value:
            raise ValueError("the canary API key file is empty")
        headers["Authorization"] = "Bearer " + value
    spec = urllib.request.Request(base + path, data=data, headers=headers)
    return urllib.request.urlopen(spec, timeout=timeout)


def http_counters(base):
    try:
        with request(base, "/metrics", timeout=10) as reply:
            active, totals = activity_metrics(reply.read().decode())
        return {"utc_epoch": time.time(), "active": active,
                "totals": {str(key): value for key, value in totals}, "total": sum(v for _, v in totals)}
    except (OSError, ValueError):
        return None


def generate(base, name, body):
    start = time.perf_counter()
    with request(base, "/generate", dict(body, stream=True, sampling_params={
        "temperature": 0, "max_new_tokens": 16, "ignore_eos": True,
    })) as response:
        events = []
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            output = json.loads(data)
            if "meta_info" not in output:
                raise RuntimeError("unexpected native SSE response")
            if output["meta_info"]["completion_tokens"]:
                events.append(((time.perf_counter() - start) * 1000, output))
    finish_ms = (time.perf_counter() - start) * 1000
    first_ms, first = events[0]
    last_ms, last = events[-1]
    meta = last["meta_info"]
    ids = last["output_ids"]
    assert len(ids) == meta["completion_tokens"] == 16
    fields = ("cached_tokens", "queue_time", "spec_accept_rate", "spec_accept_length", "spec_verify_ct",
              "spec_num_correct_drafts", "spec_num_proposed_drafts", "decode_throughput")
    decode_tokens = len(ids) - first["meta_info"]["completion_tokens"]
    return dict(name=name, prompt_tokens=meta["prompt_tokens"], output_ids=ids, text=last["text"],
                ttft_ms=round(first_ms, 3), wall_ms=round(finish_ms, 3), sse_events=len(events),
                post_first_stream_tokens_per_s=round(decode_tokens * 1000 / (last_ms - first_ms), 3)
                if last_ms > first_ms else None, **{k: meta[k] for k in fields if k in meta})


def qualify(port):
    base = f"http://127.0.0.1:{port}"
    gpu_before = gpu_snapshot()
    before = http_counters(base)
    with request(base, "/server_info") as result:
        info = json.load(result)
    fields = ("served_model_name", "model_path", "tp_size", "ep_size", "context_length",
              "speculative_algorithm", "speculative_dspark_block_size", "speculative_num_draft_tokens",
              "attention_backend", "moe_runner_backend", "mem_fraction_static", "max_total_num_tokens",
              "version", "cuda_graph_config")
    server = {k: info[k] for k in fields if k in info}
    if server.get("tp_size") != 4 or server.get("ep_size") != 4:
        raise ValueError("expected four-way TP and EP")
    chat = dict(model=server["served_model_name"], messages=[dict(
        role="user", content="What is 19 + 23? Reply only with the number.")],
        temperature=0, max_tokens=24, chat_template_kwargs={"thinking": False})
    with request(base, "/v1/chat/completions", chat) as result:
        response = json.load(result)
    choice = response["choices"][0]
    arithmetic = dict(content=choice["message"]["content"], finish_reason=choice["finish_reason"],
                      completion_tokens=response["usage"]["completion_tokens"])
    arithmetic["passed"] = arithmetic["content"].strip() == "42" and arithmetic["finish_reason"] == "stop"
    rows = [generate(base, "natural", {"text": "The first four powers of two are 1, 2, 4, 8. The next four are"})]
    # Fixed, disjoint first tokens avoid introducing cache flushes into a serving instance.
    for size in (32, 256):
        for n in range(3):
            ids = [1957 + size + n] + [567 + j % 3 for j in range(size - 1)]
            rows.append(generate(base, f"public-{size}-{n}", {"input_ids": ids}))
    summary = {}
    for size in (32, 256):
        group = [r for r in rows if r["name"].startswith(f"public-{size}-")]
        summary[str(size)] = {key: round(statistics.median(r[key] for r in group if r.get(key) is not None), 3)
                              for key in ("ttft_ms", "wall_ms", "cached_tokens", "post_first_stream_tokens_per_s")
                              if any(r.get(key) is not None for r in group)}
    return dict(profile=server, utc_epoch=time.time(), public_arithmetic=arithmetic, native=rows,
                medians_by_prompt_length=summary, outputs_per_native_request=16,
                gpu_before=gpu_before, gpu_after=gpu_snapshot(),
                http_before=before, http_after=http_counters(base), expected_inference_requests=1 + len(rows))


def compare(base, run):
    previous = {item["name"]: item for item in base["native"]}
    result = []
    for item in run["native"]:
        other = previous[item["name"]]
        ids, ref = item["output_ids"], other["output_ids"]
        first = next((i for i, (a, b) in enumerate(zip(ids, ref, strict=True)) if a != b), len(ids))
        result.append(dict(name=item["name"], common_prefix_tokens=first, output_tokens=len(ids),
                           exact_tokens=ids == ref, exact_text=item["text"] == other["text"],
                           incumbent_cached=other.get("cached_tokens"), candidate_cached=item.get("cached_tokens")))
    return dict(native=result, arithmetic_equal=run["public_arithmetic"]["content"] == base["public_arithmetic"]["content"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    results = qualify(args.port)
    if args.baseline:
        results["compare_to_incumbent"] = compare(json.loads(args.baseline.read_text()), results)
    args.receipt.write_text(json.dumps(results, indent=2) + "\n")
    if not results["public_arithmetic"]["passed"]:
        raise SystemExit("public arithmetic check failed; see receipt")
