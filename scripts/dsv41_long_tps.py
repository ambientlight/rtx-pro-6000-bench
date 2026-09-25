#!/usr/bin/env python3
"""Bounded, sequential long-context/long-output TPS probes of the V4.1 canary.

Uses /v1/responses plus the already-running native log capture. Never flushes
caches or changes the deployment. Output tokens come from API usage and native
generated IDs, NOT SSE chunk counts (DSPARK emits multi-token bursts).
Artifacts are private; no authorization headers are saved.
"""

import argparse
import collections
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

from dsv41_load import LOAD_PATH, has_contention, is_idle, parse_loads


MODEL = "deepseek-v4-flash"
DOCKER_TIMESTAMP_PREFIX = re.compile(r"20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z ")
DELTA_TYPES = {"response.output_text.delta", "response.reasoning_text.delta", "response.reasoning_summary_text.delta"}
TOPICS = (
    "admission control",
    "request cancellation",
    "chunked prefill",
    "prefix caching",
    "memory accounting",
    "speculative decoding",
    "fair scheduling",
    "stream backpressure",
    "tool argument validation",
    "incremental parsing",
    "deadline propagation",
    "retry budgets",
    "distributed tracing",
    "token accounting",
    "tenant isolation",
    "model rollouts",
    "crash recovery",
    "overload management",
    "capacity planning",
    "storage tiering",
    "data retention",
    "reproducible benchmarking",
    "regression testing",
    "incident response",
)
TASK = (
    "Write a comprehensive engineering handbook for a hypothetical multi-tenant inference service. "
    "The reference notes below are synthetic design inputs, not measurements of a real system. "
    "Write a substantial chapter for EVERY topic in the numbered list, in order. Each chapter "
    "should contain 400-600 words of original technical prose explaining design decisions, "
    "tradeoffs, a concrete failure scenario, and tests. Do not replace chapters with an outline "
    "or a summary. Do not ask to continue in another response. Avoid counting sequences and "
    "verbatim repetition. No tools are available; output the handbook itself in English.\n"
    + "\n".join(f"{i}. {topic}" for i, topic in enumerate(TOPICS, 1))
)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_frame(lines):
    data = "\n".join(line[5:].lstrip(" ") for line in lines if line.startswith("data:"))
    return json.loads(data) if data and data != "[DONE]" else None


def parse_native_record(line):
    start = line.find('{"timestamp"')
    if start < 0:
        return None
    value = line[start:]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Same transport normalization as dsv4_analyze.py: Docker prefixes
        # underlying ~16 KiB chunks, even inside a logical JSON line. Try the
        # original JSON first so valid application timestamps remain intact.
        return json.loads(DOCKER_TIMESTAMP_PREFIX.sub("", value))


def validate_events(timeline):
    events = [row["event"] for row in timeline]
    if not events or events[-1].get("type") not in ("response.completed", "response.incomplete"):
        raise ValueError("Missing successful/budget-limited terminal event")
    if any(e.get("type") in ("error", "response.failed") for e in events):
        raise ValueError("API error in stream")
    if [e.get("sequence_number") for e in events] != list(range(len(events))):
        raise ValueError("Non-contiguous SSE sequence")
    final = events[-1]["response"]
    if final["status"] == "incomplete":
        if final.get("incomplete_details", {}).get("reason") != "max_output_tokens":
            raise ValueError("Unexpected incomplete reason")
    elif final["status"] != "completed":
        raise ValueError("Unexpected terminal status")
    streamed = "".join(e["delta"] for e in events if e.get("type") == "response.output_text.delta")
    assembled = "".join(
        part.get("text", "")
        for item in final.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    if streamed != assembled:
        raise ValueError("Streamed text and final text differ")
    return final, streamed


def measurements(timeline, elapsed, native):
    final, output = validate_events(timeline)
    times = [
        row["seconds"] for row in timeline if row["event"].get("type") in DELTA_TYPES and row["event"].get("delta")
    ]
    if not times:
        raise ValueError("No model output delta")
    usage, meta = final["usage"], native["out"]["meta_info"]
    n = usage["output_tokens"]
    if n != meta["completion_tokens"] or n != len(native["out"]["output_ids"]):
        raise ValueError("API usage/native usage/generated ID count disagree")
    if usage["input_tokens"] != meta["prompt_tokens"]:
        raise ValueError("API/native input token counts disagree")
    if meta.get("id") != final["id"]:
        raise ValueError("Native record belongs to a different request")
    tps = meta.get("decode_throughput")
    if not tps or tps <= 0 or n <= 1:
        raise ValueError("Insufficient exact server decode timing")
    cached = meta.get("cached_tokens", 0)
    result = {
        "response_id": final["id"],
        "status": final["status"],
        "incomplete_details": final.get("incomplete_details"),
        "input_tokens": usage["input_tokens"],
        "output_tokens": n,
        "cached_tokens": cached,
        "ttft_client_s": times[0],
        "e2e_client_s": timeline[-1]["seconds"],
        "stream_eof_client_s": elapsed,
        "e2e_server_s": meta["e2e_latency"],
        "decode_server_tps": tps,
        "decode_server_s": (n - 1) / tps,
        "ttft_server_s": meta["e2e_latency"] - (n - 1) / tps,
        "effective_prefill_tps": (usage["input_tokens"] - cached) / times[0],
        "e2e_output_tps": n / timeline[-1]["seconds"],
        "largest_output_burst_gap_s": max((b - a for a, b in zip(times, times[1:])), default=0),
        "output_bursts": len(times),
        "num_retractions": meta.get("num_retractions"),
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "spec_accept_length": meta.get("spec_accept_length"),
        "queue_time_s": meta.get("queue_time"),
    }
    # A lightweight warning screen, not an accuracy/quality evaluation.
    runs = [len(m[0]) for m in re.finditer(r"([^\s])\1+", output)]
    lines = [line.strip() for line in output.splitlines() if len(line.split()) >= 10]
    repeats = sum(n - 1 for n in collections.Counter(lines).values())
    result["quality_screen"] = {
        "characters": len(output),
        "longest_character_run": max(runs, default=1),
        "duplicate_long_line_fraction": repeats / len(lines) if lines else 0,
        "raw_protocol_marker": "｜DSML｜" in output or "<｜tool" in output,
    }
    return result, output


class Client:
    def __init__(self, args):
        self.args = args
        self.headers = {"Content-Type": "application/json"}
        if args.key_file is not None:
            secret = args.key_file.read_text().strip()
            if not secret or "\n" in secret or "\r" in secret:
                raise ValueError("Invalid key file")
            self.headers["Authorization"] = "Bearer " + secret

    def open(self, path, payload=None, timeout=None):
        req = urllib.request.Request(
            self.args.base_url.rstrip("/") + path,
            headers=self.headers,
            data=json.dumps(payload).encode() if payload is not None else None,
        )
        return urllib.request.urlopen(req, timeout=timeout or self.args.timeout)

    def json(self, path, payload=None, timeout=30):
        with self.open(path, payload, timeout) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

    def idle(self):
        state = parse_loads(self.json(LOAD_PATH))
        if not is_idle(state):
            raise RuntimeError("Endpoint is busy; no additional test request was started")
        return state


def make_prompt(nonce, count):
    lines = [f"Independent benchmark identity: {nonce}\n", TASK, "\nSynthetic reference notes:\n"]
    for i in range(count):
        topic = TOPICS[i % len(TOPICS)]
        variant = (
            f"Tenant {i % 137} has a {17 + i % 211} second deadline and a burst of {1 + i % 31} requests. "
            f"Track service epoch {1000 + i}, pending bytes {4096 + (i * 8191) % 65521}, "
            f"and cancellation budget {2 + i % 11}. "
        )
        concern = (
            "Separate queue delay from processing time and propagate cancellation to every owner.",
            "Bound resident state, account for in-flight work, and release resources after the terminal event.",
            "Preserve event ordering under partial writes; document retries and reject ambiguous replay.",
            "Compare uncached latency with steady-state throughput using versioned, reproducible fixtures.",
            "Consider rolling upgrades, worker loss, slow consumers, and fairness between large and small jobs.",
        )[i % 5]
        lines.append(f"Record {i}: {topic}. {variant}{concern}\n")
    lines.append("\nEnd of notes. Now write the full handbook as requested, starting with chapter 1.\n" + TASK)
    return "".join(lines)


def calibrated_prompt(client, target):
    nonce, count = uuid.uuid4().hex, max(1, target // 100)
    for _ in range(5):
        text = make_prompt(nonce, count)
        result = client.json(
            "/v1/tokenize",
            {
                "model": client.args.model,
                "messages": [{"role": "user", "content": text}],
                "chat_template_kwargs": {"thinking": False},
            },
            timeout=120,
        )
        actual = result["count"]
        if abs(actual - target) < max(150, target * 0.003):
            return text, actual
        count = max(1, round(count * target / actual))
    raise RuntimeError(f"Could not calibrate input: target {target}, got {actual}")


class Observer:
    def __init__(self, client, folder):
        self.client, self.folder = client, folder
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with (self.folder / "observer.jsonl").open("w") as out:
            while not self.stop.is_set():
                sample = {"utc": utc_now()}
                try:
                    sample["load"] = parse_loads(self.client.json(LOAD_PATH, timeout=5))
                    proc = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=True,
                    )
                    sample["gpus"] = [
                        dict(
                            zip(
                                ("index", "memory_mib", "utilization_pct", "power_w", "temperature_c"),
                                map(float, line.split(",")),
                            )
                        )
                        for line in proc.stdout.splitlines()
                    ]
                except Exception as error:
                    sample["observer_error"] = type(error).__name__
                self.samples.append(sample)
                out.write(json.dumps(sample) + "\n")
                out.flush()
                self.stop.wait(5)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=15)
        return {
            "load_contention_observed": any(
                has_contention(sample.get("load", []))
                for sample in self.samples
            ),
            "observer_errors": sum("observer_error" in sample for sample in self.samples),
            "gpu_peak_memory_mib": {
                str(i): max(
                    (
                        gpu["memory_mib"]
                        for sample in self.samples
                        for gpu in sample.get("gpus", [])
                        if gpu["index"] == i
                    ),
                    default=None,
                )
                for i in range(4)
            },
        }


def collect_native(path, offset, rid, folder):
    """Only retain this synthetic request's record, not unrelated user prompts."""
    found, warnings = None, []
    others = set()
    deadline = time.monotonic() + 15
    with path.open() as source:
        source.seek(offset)
        while time.monotonic() < deadline:
            while True:
                position = source.tell()
                line = source.readline()
                if not line or not line.endswith("\n"):
                    source.seek(position)
                    break
                start = line.find('{"timestamp"')
                if start >= 0:
                    try:
                        event = parse_native_record(line)
                    except json.JSONDecodeError:
                        continue
                    other_id = event.get("rid", "")
                    if (
                        event.get("event") == "request.received"
                        and other_id != rid
                        and not str(other_id).startswith("HEALTH_CHECK_")
                    ):
                        others.add(str(other_id))
                    if event.get("event") == "request.finished" and other_id == rid:
                        found = event
                        break
                elif any(
                    s in line
                    for s in (
                        "memory allocation failed with OOM",
                        "OutOfMemoryError",
                        "Traceback (most recent call last)",
                        "state was deleted in TokenizerManager",
                    )
                ):
                    warnings.append(line.rstrip())
            if found:
                break
            time.sleep(0.5)
    save(folder / "native-warnings.json", warnings)
    if found is None:
        raise RuntimeError("Exact native completion record unavailable; refusing to substitute SSE chunk counts")
    save(folder / "native-finished.json", found)
    return found, {
        "allocator_retry_warning_lines": sum("memory allocation failed with OOM" in line for line in warnings),
        "fatal_or_deleted_state_lines": sum("memory allocation failed with OOM" not in line for line in warnings),
        "other_native_request_ids": sorted(others),
    }


def stream_request(client, payload, folder):
    save(folder / "request.json", payload)
    timeline, frame = [], []
    started = time.perf_counter()
    try:
        with (folder / "response.sse").open("wb") as raw_out, (folder / "timeline.jsonl").open("w") as timed_out:
            with client.open("/v1/responses", payload) as response:
                if response.status != 200 or "text/event-stream" not in response.headers.get("Content-Type", ""):
                    raise RuntimeError("Expected HTTP 200 event stream")
                for raw in response:
                    now = time.perf_counter() - started
                    if now > client.args.timeout:
                        raise TimeoutError("Bounded benchmark deadline exceeded")
                    raw_out.write(raw)
                    line = raw.decode().rstrip("\r\n")
                    if line:
                        frame.append(line)
                        continue
                    event = parse_frame(frame)
                    frame = []
                    if event is not None:
                        row = {"seconds": now, "event": event}
                        timeline.append(row)
                        timed_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        timed_out.flush()
    except urllib.error.HTTPError as error:
        (folder / "http-error.txt").write_bytes(error.read())
        raise RuntimeError(f"HTTP {error.code}; see private artifact") from None
    elapsed = time.perf_counter() - started
    save(folder / "stream-timing.json", {"eof_seconds": elapsed})
    return timeline, elapsed


def identity(container):
    fmt = '{"id":{{json .Id}},"image_id":{{json .Image}},"started_at":{{json .State.StartedAt}},"running":{{json .State.Running}},"restart_count":{{json .RestartCount}}}'
    return json.loads(subprocess.check_output(["docker", "inspect", "--format", fmt, container], text=True, timeout=10))


def report(root, results):
    save(root / "results.json", results)
    lines = [
        "# V4.1 long-call TPS benchmark",
        "",
        "Sequential concurrency 1, unique uncached prefixes, temperature 1.0 / top-p 0.95, thinking off.",
        "No cache flush, server reconfiguration, or ignored EOS. One synthetic handbook request per input size.",
        "",
        "| Actual input | Actual output | Client TTFT (s) | Decode tok/s | Total (s) | Cached input | Allocator retries |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['input_tokens']:,} | {r['output_tokens']:,} | {r['ttft_client_s']:.2f} | {r['decode_server_tps']:.2f} | {r['e2e_client_s']:.2f} | {r['cached_tokens']:,} | {r['allocator_retry_warning_lines']} |"
        )
    lines += [
        "",
        "Decode tok/s is SGLang's (actual completion tokens − 1) / (last-token time − first-token time), cross-checked against API usage and generated token IDs. SSE bursts are not treated as tokens.",
        "",
        "Client total time ends at receipt of the terminal SSE event (not TCP close). Natural EOS may end a request before its requested output cap.",
        "",
        "Effective prefill tok/s in results.json is uncached input / client TTFT; it includes HTTP, tokenization, queueing and first-token overhead, not pure GPU prefill time.",
        "",
        "Budget-limited response.incomplete is expected, not an API failure. Inspect results.json for EOS/length status, speculation, overlap, memory retry, and heuristic quality details. Observed synthetic TPS is not a general workload guarantee.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def analyze_existing(root):
    """Read-only to the deployment: recover metrics without resending inference."""
    deployment = json.loads((root / "deployment.json").read_text())
    results = []
    for folder in sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
        timeline = [json.loads(line) for line in (folder / "timeline.jsonl").read_text().splitlines()]
        final, _ = validate_events(timeline)
        before = json.loads((folder / "before.json").read_text())
        native, log_info = collect_native(
            Path(deployment["capture"]) / "dsv4-native.log", before["native_offset"], final["id"], folder
        )
        timing_file = folder / "stream-timing.json"
        elapsed = json.loads(timing_file.read_text())["eof_seconds"] if timing_file.exists() else None
        result, output = measurements(timeline, elapsed, native)
        result.update(json.loads((folder / "observer-summary.json").read_text()))
        result.update(log_info)
        request = json.loads((folder / "request.json").read_text())
        result.update(
            {
                "target_input_tokens": int(folder.name),
                "requested_output_tokens": request["max_output_tokens"],
                "folder": str(folder),
            }
        )
        (folder / "output.txt").write_text(output)
        save(folder / "result.json", result)
        results.append(result)
        print(json.dumps(result), flush=True)
    report(root, results)
    save(root / "analysis.json", {"analyzed_utc": utc_now(), "requests": len(results), "resubmitted_inference": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--key-file", type=Path, help="Optional backend credential")
    parser.add_argument("--capture", type=Path, default=Path("/mnt/hot/dsv41_dumps/current"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/hot/dsv41_state/benchmarks"))
    parser.add_argument("--container", default="dsv41")
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 131072, 200000])
    parser.add_argument("--output-tokens", type=int, default=8192)
    parser.add_argument("--context-limit", type=int, default=524288)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--analyze-existing", type=Path, help="Rebuild metrics from a previous run; sends no inference")
    args = parser.parse_args()
    os.umask(0o077)
    if args.analyze_existing:
        analyze_existing(args.analyze_existing)
        return
    if not 1 <= len(args.contexts) <= 4 or not 256 <= args.output_tokens <= 32768:
        parser.error("Use 1–4 contexts and an output budget of 256–32768 tokens")
    if any(n < 4096 or n * 1.005 + args.output_tokens > args.context_limit for n in args.contexts):
        parser.error("Input/output sizes must fit the configured context with calibration margin")
    root = args.output_root / ("long-tps-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    root.mkdir(parents=True, mode=0o700)
    print("Evidence:", root, flush=True)
    capture = args.capture.resolve(strict=True)
    original_identity = identity(args.container)
    captured_identity = json.loads((capture / "wire/config.json").read_text())["deployment"]
    if original_identity["id"] != captured_identity["id"] or not original_identity["running"]:
        raise RuntimeError("Capture does not match the running deployment")
    save(root / "deployment.json", {"started_utc": utc_now(), "capture": str(capture), **original_identity})
    client, results = Client(args), []
    try:
        client.json("/health")
        for target in args.contexts:
            client.idle()
            folder = root / str(target)
            folder.mkdir(mode=0o700)
            text, tokenized = calibrated_prompt(client, target)
            client.idle()
            if identity(args.container) != original_identity or args.capture.resolve() != capture:
                raise RuntimeError("Deployment/capture changed during the benchmark")
            native_path = capture / "dsv4-native.log"
            offset = native_path.stat().st_size
            save(
                folder / "before.json",
                {
                    "utc": utc_now(),
                    "tokenized_input": tokenized,
                    "native_offset": offset,
                    "capture": json.loads((capture / "wire/status.json").read_text()),
                },
            )
            payload = {
                "model": args.model,
                "input": [{"role": "user", "content": text}],
                "stream": True,
                "store": False,
                "temperature": 1.0,
                "top_p": 0.95,
                "max_output_tokens": args.output_tokens,
                "chat_template_kwargs": {"thinking": False},
            }
            print(
                json.dumps(
                    {
                        "event": "starting",
                        "target_input": target,
                        "tokenized_input": tokenized,
                        "output_budget": args.output_tokens,
                        "utc": utc_now(),
                    }
                ),
                flush=True,
            )
            observer = Observer(client, folder)
            observer.thread.start()
            try:
                timeline, elapsed = stream_request(client, payload, folder)
            finally:
                observed = observer.finish()
                save(folder / "observer-summary.json", observed)
            final, _ = validate_events(timeline)
            native, log_info = collect_native(native_path, offset, final["id"], folder)
            result, output = measurements(timeline, elapsed, native)
            result.update(observed)
            result.update(log_info)
            result.update(
                {"target_input_tokens": target, "requested_output_tokens": args.output_tokens, "folder": str(folder)}
            )
            (folder / "output.txt").write_text(output)
            save(folder / "result.json", result)
            save(folder / "after-capture.json", json.loads((capture / "wire/status.json").read_text()))
            results.append(result)
            report(root, results)
            print(json.dumps({"event": "finished", **result}), flush=True)
            # Stop further probes if these measurements could be contended or the server changed.
            if result["other_native_request_ids"] or result["load_contention_observed"]:
                raise RuntimeError("Other inference traffic observed; results flagged and further probes stopped")
            if result["fatal_or_deleted_state_lines"] or identity(args.container) != original_identity:
                raise RuntimeError("Server error/restart observed; further probes stopped")
            client.idle()
            client.json("/health")
        save(root / "completion.json", {"completed": True, "finished_utc": utc_now(), "requests": len(results)})
    except Exception as error:
        save(
            root / "completion.json",
            {"completed": False, "finished_utc": utc_now(), "error": str(error), "requests": len(results)},
        )
        raise


if __name__ == "__main__":
    main()
