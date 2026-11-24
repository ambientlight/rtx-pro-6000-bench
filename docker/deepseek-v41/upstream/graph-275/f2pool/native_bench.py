"""Replay matched public streaming Responses requests across stock and native-mining DeepSeek."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "full-canary"))
from public_handbook import make_prompt
from qualify import request
from throughput import native_decode_metrics, server_histograms, sse, write


def measure(base, index, notes, limit, *, cache_salt=None):
    prompt = make_prompt(hashlib.sha256(f"native-pearl-qualification-v2/{index}".encode()).hexdigest()[:24], notes)
    payload = dict(model="deepseek-v4-flash", input=[dict(role="user", content=prompt)], stream=True,
                   store=False, temperature=0, max_output_tokens=limit, chat_template_kwargs={"thinking": False})
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    before = server_histograms(base)
    start = time.monotonic_ns()
    first = final = None
    output = hashlib.sha256()
    with request(base, "/v1/responses", payload, timeout=180) as response:
        for event in sse(response):
            if event["type"] in ("response.output_text.delta", "response.reasoning_text.delta",
                                 "response.reasoning_summary_text.delta") and event.get("delta"):
                if first is None:
                    first = time.monotonic_ns()
                output.update(event["delta"].encode())
            if event["type"] in ("response.completed", "response.incomplete", "response.failed"):
                final = event["response"]
                break
    end = time.monotonic_ns()
    if first is None or not final or final["status"] not in ("completed","incomplete"):
        raise RuntimeError("Responses stream failed before a terminal event")
    usage = final["usage"]
    generated = usage["output_tokens"]
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    after = server_histograms(base)
    return dict(index=index, notes=notes, budget=limit, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                output_sha256=output.hexdigest(), start_ns=start, end_ns=end, http=response.status,
                input_tokens=usage["input_tokens"], cached_tokens=cached, output_tokens=generated,
                status=final["status"], ttft_ms=(first-start)/1e6, client_decode_tps=(generated-1)*1e9/(end-first),
                **native_decode_metrics(before, after, generated))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--transitions", type=int, default=0)
    args = parser.parse_args()
    if args.key_file:
        os.environ["CANARY_API_KEY_FILE"] = str(args.key_file)
    base = f"http://127.0.0.1:{args.port}"
    cases = [(i, 14 if i < 8 else 150, 256) for i in range(10)]
    if args.transitions:
        cases = [(100+i, 2, 12) for i in range(args.transitions)]
    rows = []
    for i, notes, limit in cases:
        result = measure(base, i, notes, limit)
        rows.append(result)
        write(args.out, dict(rows=rows))
        print(i, result["input_tokens"], result["output_tokens"], round(result["client_decode_tps"], 1), flush=True)
        time.sleep(.22 if args.transitions else .7)
    print("median decode", statistics.median(r["client_decode_tps"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
