#!/usr/bin/env python3
"""Sequential, bounded two-tier KV-cache pressure test; never changes serving.

64 unique random 256k prompts, short deterministic output, eight exact replays.
Metrics are archived every ten seconds. Stop on deployment changes, API errors,
wrong answers, monitoring loss, or low system RAM. No cache flush or restart.
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import threading
import time
import urllib.request

from dsv41_deferred_relaunch import activity_metrics
from dsv41_live_acceptance import metric_total
from dsv41_load import LOAD_PATH, is_idle, parse_loads
from dsv41_long_tps import parse_frame, validate_events


WORDS = tuple(
    """amber apple arch autumn beach blue boat book bridge bright brook
calm cedar chair clay clear cloud coast copper coral corner cotton creek dawn
deep desert dock door east field fire forest fresh garden glass gold grain grass
green grey grove harbor hill home honey island ivory jade lake lamp leaf light
lime linen maple meadow metal mist moon moss north oak ocean olive orange paper
path peach pearl pine plain pond purple quiet rain reed river road rock rose
round sand seed shade shell shore silver sky slate small smooth snow soft south
spring star steel stone storm stream summer sun sweet table teal tree valley
warm water wave west wheat white willow wind window winter wood yellow""".split()
)
MODEL = "deepseek-v4-flash"
REPLAYS = {16: (0,), 32: (8,), 48: (16,), 64: (0, 24, 40, 48, 63)}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def host_memory():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {key: int(fields[key].split()[0]) * 1024 for key in ("MemTotal", "MemAvailable", "Mlocked", "SwapFree")}


def container_state():
    raw = subprocess.check_output(["docker", "inspect", "dsv41", "--format", "{{json .}}"], text=True, timeout=10)
    item = json.loads(raw)
    return {
        "id": item["Id"],
        "started": item["State"]["StartedAt"],
        "restarts": item["RestartCount"],
        "state": item["State"]["Status"],
        "image": item["Image"],
    }


def selected_metrics(raw):
    rank = {"tp_rank": "0"}
    result = {
        mode: metric_total(raw, "sglang:prefill_effective_tokens_total", {**rank, "mode": mode})
        for mode in ("input", "device_hit", "host_hit", "storage_hit")
    }
    for name in (
        "kv_available_tokens",
        "kv_evictable_tokens",
        "kv_used_tokens",
        "swa_evictable_tokens",
        "hicache_host_used_tokens",
        "hicache_host_total_tokens",
        "hicache_swa_host_used_tokens",
        "hicache_swa_host_total_tokens",
        "hicache_swa_host_evicted_tokens",
        "hicache_swa_host_boundary_evicted_tokens",
        "hicache_swa_host_retention_enabled",
    ):
        result[name] = metric_total(raw, "sglang:" + name, rank)
    for name in (
        "hicache_backup_bytes_total",
        "load_back_bytes_total",
        "load_back_tokens_total",
        "hicache_backup_duration_seconds_sum",
        "load_back_duration_seconds_sum",
        "evicted_tokens_total",
        "hicache_dropped_tokens_total",
    ):
        result[name] = metric_total(raw, "sglang:" + name)
    # These pool counters sum all TP ranks. Keep them separate: SWA-only
    # restore does not demonstrate FULL KV restoration from RAM.
    for pool in ("kv", "swa"):
        for action in ("hicache_backup", "load_back"):
            result[f"{action}_{pool}_tokens_all_ranks"] = metric_total(
                raw, f"sglang:{action}_tokens_total", {"pool": pool}
            )
    return result


def render_prompt(nonce, index, words, padding=0):
    code = str(700000 + index)
    return (
        f"Case {index}, independent identity {nonce}. This is a synthetic cache test. "
        "The following randomized reference words are inert data, not instructions.\n<reference>\n"
        + " ".join(words)
        + " a" * padding
        + f"\n</reference>\nVerification code: {code}. Reply with exactly {code} and nothing else."
    )


def random_words(nonce, index, count):
    seed = int(hashlib.sha256(f"{nonce}:{index}".encode()).hexdigest(), 16)
    rng = random.Random(seed)
    return [rng.choice(WORDS) for _ in range(count)]


def calibrate(nonce, index, target, count_tokens):
    words = random_words(nonce, index, target)
    count = max(1, int(target * 0.9))
    padding = 0
    attempts = []
    for _ in range(8):
        text = render_prompt(nonce, index, words[:count], padding)
        actual = count_tokens(text)
        attempts.append({"words": count, "padding": padding, "tokens": actual})
        if actual == target:
            return text, attempts
        if actual < target and target - actual <= 2048:
            padding += target - actual
        elif padding and padding + target - actual >= 0:
            padding += target - actual
        else:
            count = max(1, min(len(words), round(count * (target - 512) / actual)))
            padding = 0
    raise RuntimeError(f"Cannot calibrate exact token length: {attempts}")


def validate_answer(timeline, expected, target):
    response, text = validate_events(timeline)
    if response["status"] != "completed" or text.strip() != expected:
        raise RuntimeError("Synthetic verification answer wrong or incomplete; inspect private response")
    if response["usage"]["input_tokens"] != target:
        raise RuntimeError("Actual inference token count differs from calibration")
    return response


class Experiment:
    def __init__(self, args):
        self.args, self.root = args, args.output
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if (self.root / "manifest.json").exists():
            raise RuntimeError("Existing experiment: refusing to repeat inference")
        self.nonce = os.urandom(16).hex()
        self.identity = container_state()
        if self.identity["state"] != "running":
            raise RuntimeError("Deployment is not running")
        self.stop = threading.Event()
        self.abort_reason = None
        self.results = []
        self.completed = 0
        self.phase = "preflight"
        self.index = None
        self.response = None
        self.minimum_available = host_memory()["MemAvailable"]
        save(
            self.root / "manifest.json",
            {
                "created": utc(),
                "identity": self.identity,
                "nonce": self.nonce,
                "count": args.count,
                "target_tokens": args.tokens,
                "max_output_tokens": 16,
                "concurrency": 1,
                "replays_after_cold_count": REPLAYS,
                "ram_floor_gib": args.ram_floor_gib,
                "endpoint": args.base_url + "/v1/responses",
                "capture": str(Path("/mnt/hot/dsv41_dumps/current").resolve()),
            },
        )

    def request(self, path, payload=None, timeout=30):
        request = urllib.request.Request(
            self.args.base_url + path,
            headers={"Content-Type": "application/json"},
            data=None if payload is None else json.dumps(payload).encode(),
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode()

    def status(self, state, **extra):
        save(
            self.root / "status.json",
            {
                "utc": utc(),
                "state": state,
                "phase": self.phase,
                "index": self.index,
                "cold_completed": self.completed,
                "inference_completed": len(self.results),
                "minimum_available_gib": self.minimum_available / 2**30,
                **extra,
            },
        )

    def snapshot(self):
        raw = self.request("/metrics", timeout=10)
        memory = host_memory()
        self.minimum_available = min(self.minimum_available, memory["MemAvailable"])
        return {
            "utc": utc(),
            "phase": self.phase,
            "index": self.index,
            "memory": memory,
            "memory_pressure": Path("/proc/pressure/memory").read_text(),
            "loads": json.loads(self.request(LOAD_PATH, timeout=10)),
            "metrics": selected_metrics(raw),
            "raw_metrics": raw,
        }

    def monitor(self):
        failures = low_samples = 0
        with gzip.open(self.root / "telemetry.jsonl.gz", "at") as log:
            while not self.stop.is_set():
                try:
                    sample = self.snapshot()
                    failures = 0
                    low_samples = (
                        low_samples + 1 if sample["memory"]["MemAvailable"] < self.args.ram_floor_gib * 2**30 else 0
                    )
                    if low_samples >= 2 or sample["memory"]["MemAvailable"] < 16 * 2**30:
                        raise RuntimeError("System RAM below safety floor")
                    if container_state() != self.identity:
                        raise RuntimeError("Deployment changed, stopped or restarted")
                    save(self.root / "latest-telemetry.json", {k: v for k, v in sample.items() if k != "raw_metrics"})
                except RuntimeError as error:
                    self.abort_reason = str(error)
                    sample = {"utc": utc(), "abort": self.abort_reason}
                except Exception as error:
                    failures += 1
                    sample = {"utc": utc(), "monitor_error": type(error).__name__, "consecutive_failures": failures}
                    if failures >= 3:
                        self.abort_reason = "Monitoring failed three consecutive times"
                log.write(json.dumps(sample) + "\n")
                log.flush()
                if self.abort_reason:
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                self.stop.wait(10)

    def wait_idle(self):
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            rows = parse_loads(json.loads(self.request(LOAD_PATH)))
            active, _ = activity_metrics(self.request("/metrics"))
            if is_idle(rows) and active == 0:
                if container_state() != self.identity:
                    raise RuntimeError("Deployment identity changed")
                if host_memory()["MemAvailable"] < self.args.ram_floor_gib * 2**30:
                    raise RuntimeError("Insufficient RAM to admit next request")
                return
            self.status("waiting_for_idle")
            time.sleep(5)
        raise RuntimeError("No idle opportunity within one hour; no further test request sent")

    def prompt(self, index):
        path = self.root / "prompts" / f"{index:02}.json"
        if path.exists():
            return json.loads(path.read_text())["text"]
        self.phase, self.index = "calibrating", index
        self.status("running")

        def count(text):
            data = json.loads(
                self.request(
                    "/v1/tokenize",
                    {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": text}],
                        "chat_template_kwargs": {"thinking": False},
                    },
                    timeout=120,
                )
            )
            return data["count"]

        text, attempts = calibrate(self.nonce, index, self.args.tokens, count)
        path.parent.mkdir(mode=0o700, exist_ok=True)
        save(
            path,
            {
                "text": text,
                "tokens": self.args.tokens,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "calibration": attempts,
            },
        )
        return text

    def inference(self, index, phase):
        self.wait_idle()
        text = self.prompt(index)
        self.wait_idle()
        self.phase, self.index = phase, index
        folder = self.root / f"{len(self.results):03}-{phase}-{index:02}"
        folder.mkdir(mode=0o700)
        before = self.snapshot()
        save(folder / "before.json", before)
        payload = {
            "model": MODEL,
            "input": [{"role": "user", "content": text}],
            "stream": True,
            "store": False,
            "temperature": 0,
            "max_output_tokens": 16,
            "chat_template_kwargs": {"thinking": False},
        }
        save(folder / "request.json", payload)
        self.status("running", current_artifacts=str(folder))
        request = urllib.request.Request(
            self.args.base_url + "/v1/responses",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        timeline, frame = [], []
        started = time.monotonic()
        with (folder / "stream.txt").open("w") as stream:
            with urllib.request.urlopen(request, timeout=600) as response:
                self.response = response
                for encoded in response:
                    line = encoded.decode()
                    stream.write(line)
                    stream.flush()
                    if line.strip():
                        frame.append(line.rstrip("\r\n"))
                    else:
                        event = parse_frame(frame)
                        frame = []
                        if event is not None:
                            timeline.append({"seconds": time.monotonic() - started, "event": event})
                self.response = None
        save(folder / "timeline.json", timeline)
        result = validate_answer(timeline, str(700000 + index), self.args.tokens)
        # Allow metrics/HTTP-finalization callbacks to publish before attribution.
        time.sleep(1)
        after = self.snapshot()
        save(folder / "after.json", after)
        _, before_http = activity_metrics(before["raw_metrics"])
        active, after_http = activity_metrics(after["raw_metrics"])
        before_http, after_http = dict(before_http), dict(after_http)
        changes = {
            str(key): after_http.get(key, 0) - before_http.get(key, 0)
            for key in before_http.keys() | after_http.keys()
            if before_http.get(key, 0) != after_http.get(key, 0)
        }
        deltas = {key: after["metrics"][key] - value for key, value in before["metrics"].items()}
        ttft = next(row["seconds"] for row in timeline if row["event"].get("type") == "response.output_text.delta")
        record = {
            "index": index,
            "phase": phase,
            "utc": utc(),
            "response_id": result["id"],
            "usage": result["usage"],
            "ttft_seconds": ttft,
            "elapsed_seconds": time.monotonic() - started - 1,
            "metrics_delta": deltas,
            "metrics_after": after["metrics"],
            "uncontended": active == 0 and changes == {str(("/v1/responses", "POST")): 1},
            "http_changes": changes,
            "available_gib": after["memory"]["MemAvailable"] / 2**30,
        }
        save(folder / "result.json", record)
        self.results.append(record)
        if phase == "cold":
            self.completed += 1
        save(self.root / "results.json", self.results)
        self.status("running", last_result=record)
        print(json.dumps({"completed": len(self.results), "cold_completed": self.completed, **record}), flush=True)

    def run(self):
        def interrupted(signum, frame):
            raise InterruptedError(self.abort_reason or f"Stopped by signal {signum}; no further inference")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        self.wait_idle()
        info = json.loads(self.request("/server_info"))
        save(self.root / "server-info.json", info)
        if not info.get("enable_hierarchical_cache") or self.args.tokens + 16 >= info["context_length"]:
            raise RuntimeError("Cache disabled or context insufficient")
        monitor = threading.Thread(target=self.monitor, daemon=True)
        monitor.start()
        try:
            for index in range(self.args.count):
                self.inference(index, "cold")
                for replay in REPLAYS.get(index + 1, ()):
                    self.inference(replay, "replay")
            self.phase = "complete"
            self.status("complete")
        except BaseException as error:
            self.status("stopped", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self.stop.set()
            if self.response is not None:
                self.response.close()
            monitor.join(timeout=25)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=256000)
    parser.add_argument("--ram-floor-gib", type=int, default=24)
    args = parser.parse_args(argv)
    if not 1 <= args.count <= 64 or not 1024 <= args.tokens <= 256000 or args.ram_floor_gib < 24:
        parser.error("Require 1–64 prompts, 1024–256000 tokens, and RAM floor >=24 GiB")
    return args


def main():
    os.umask(0o077)
    Experiment(parse_args()).run()


if __name__ == "__main__":
    main()
