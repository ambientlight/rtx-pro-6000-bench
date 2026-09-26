#!/usr/bin/env python3
"""Read /v1/loads safely and share passive-monitoring utilities.

The CLI optionally waits for a stable, empty scheduler. It never starts/stops
containers or sends inference requests. Keep the historical observer's list
shape and num_reqs alias, but read the modern API.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request


LOAD_PATH = "/v1/loads"
BASE_URL = "http://127.0.0.1:8000"
INFERENCE = {"/v1/messages", "/v1/responses", "/v1/chat/completions", "/v1/completions", "/generate", "/encode"}
COUNTER = re.compile(r"^sglang:http_requests_(active|total)\{([^}]+)\}\s+([^ ]+)(?:\s+\d+)?$")
QUEUE_FIELDS = {"waiting", "grammar", "paused", "retracted", "prealloc_ready"}
ACTIVE_TOKEN_FIELDS = (
    "num_used_tokens",
    "num_total_tokens",
    "num_active_tokens",
    "num_waiting_uncached_tokens",
)


def run(*command, timeout=30):
    return subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=timeout)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fetch(path):
    with urllib.request.urlopen(BASE_URL + path, timeout=10) as response:
        return response.read().decode()


def activity_metrics(text):
    values = {"active": {}, "total": {}}
    for line in text.splitlines():
        match = COUNTER.fullmatch(line)
        if not match:
            continue
        kind, labels, number = match.groups()
        labels = dict(re.findall(r'(\w+)="([^"]*)"', labels))
        endpoint = labels.get("endpoint")
        if endpoint not in INFERENCE:
            continue
        value = float(number)
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError("Invalid HTTP activity counter")
        key = (endpoint, labels.get("method"))
        if key in values[kind]:
            raise ValueError("Duplicate HTTP activity counter")
        values[kind][key] = int(value)
    if not values["active"] or values["active"].keys() != values["total"].keys():
        raise ValueError("Missing or incomplete HTTP inference metrics")
    return sum(values["active"].values()), tuple(sorted(values["total"].items()))


def mem_available():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) * 1024


def counter(value):
    if type(value) is not int or value < 0:
        raise ValueError("Load counters must be nonnegative integers")
    return value


def parse_loads(payload, *, now=None, max_age=30):
    """Validate full, fresh rank snapshots; never interpret missing data as idle."""
    rows = payload.get("loads") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty /v1/loads envelope")
    now = time.time() if now is None else now
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid scheduler load row")
        rank = counter(row.get("dp_rank"))
        if rank in seen:
            raise ValueError("Duplicate DP rank in load response")
        seen.add(rank)
        stamp = row.get("timestamp")
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or not -5 <= now - stamp <= max_age:
            raise ValueError("Missing, stale or future scheduler load timestamp")
        running = counter(row.get("num_running_reqs"))
        waiting = counter(row.get("num_waiting_reqs"))
        for field in ACTIVE_TOKEN_FIELDS:
            counter(row.get(field))
        queues = row.get("queues")
        if not isinstance(queues, dict) or not QUEUE_FIELDS <= queues.keys():
            raise ValueError("Missing scheduler queue counters; request the full /v1/loads response")
        for value in queues.values():
            counter(value)
        disagg = row.get("disaggregation", {})
        if not isinstance(disagg, dict):
            raise ValueError("Invalid disaggregation counters")
        for key, value in disagg.items():
            if key.endswith("_queue_reqs"):
                counter(value)
        result.append({**row, "num_reqs": running + waiting})
    return result


def queued(row):
    return (
        row.get("num_waiting_reqs", 0) > 0
        or any(row.get("queues", {}).values())
        or any(v for k, v in row.get("disaggregation", {}).items() if k.endswith("_queue_reqs"))
    )


def is_idle(rows):
    if not rows:
        raise ValueError("An empty load result cannot establish idleness")
    # This runtime excludes the currently chunked prefill from running/waiting
    # request counts. Its active/remaining tokens must drain too. Cumulative
    # prefill totals and cached-prefix occupancy are not activity indicators.
    return all(
        row["num_reqs"] == 0 and not queued(row) and not any(row[k] for k in ACTIVE_TOKEN_FIELDS)
        for row in rows
    )


def has_contention(rows):
    # Sum across DP ranks, rather than missing one request on each of two ranks.
    return sum(row["num_reqs"] for row in rows) > 1 or any(queued(row) for row in rows)


class DrainWindow:
    def __init__(self, seconds, expected_ranks):
        self.seconds, self.expected_ranks = seconds, set(expected_ranks)
        self.since = None
        self.last_stamps = None

    def observe(self, rows, now):
        if {row["dp_rank"] for row in rows} != self.expected_ranks:
            self.since = self.last_stamps = None
            raise ValueError("Incomplete or unexpected DP ranks during drain")
        if not is_idle(rows):
            self.since = self.last_stamps = None
            return False
        stamps = {row["dp_rank"]: row["timestamp"] for row in rows}
        if self.last_stamps is not None and any(stamps[k] <= self.last_stamps[k] for k in stamps):
            self.since = self.last_stamps = None
            raise ValueError("Scheduler snapshot did not advance during drain")
        self.last_stamps = stamps
        if self.since is None:
            self.since = now
        return now - self.since >= self.seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--idle-seconds", type=float, default=20)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--dp-ranks", type=int, nargs="+", default=[0])
    parser.add_argument("--log-file", type=Path, required=True)
    args = parser.parse_args()
    if args.idle_seconds < 5 or not 1 <= args.interval <= 30 or any(n < 0 for n in args.dp_ranks):
        parser.error("Require >=5 idle seconds, a 1-30 second poll interval and nonnegative DP ranks")
    window = DrainWindow(args.idle_seconds, args.dp_ranks)
    # Exclusive creation prevents overwriting earlier deployment evidence.
    os.umask(0o077)
    with args.log_file.open("x") as log:
        while True:
            with urllib.request.urlopen(args.base_url.rstrip("/") + LOAD_PATH, timeout=15) as response:
                payload = json.load(response)
            rows = parse_loads(payload)
            ready = window.observe(rows, time.monotonic())
            log.write(json.dumps({"observed_at": time.time(), "ready": ready, "response": payload}) + "\n")
            log.flush()
            print(
                json.dumps(
                    {
                        "running": sum(r["num_running_reqs"] for r in rows),
                        "waiting": sum(r["num_waiting_reqs"] for r in rows),
                        "active_tokens": sum(r["num_active_tokens"] for r in rows),
                        "pending_prefill_tokens": sum(r["num_waiting_uncached_tokens"] for r in rows),
                        "other_queued": any(queued(r) for r in rows),
                        "drained": ready,
                    }
                ),
                flush=True,
            )
            if ready:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
