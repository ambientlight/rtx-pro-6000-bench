"""Observe one supervised worker across real /generate requests to a TP1 canary."""

import argparse
from contextlib import contextmanager, nullcontext
import concurrent.futures
import json
import random
from pathlib import Path
import socket
import time
import urllib.request


def status(path):
    with socket.socket(socket.AF_UNIX) as channel:
        channel.settimeout(3)
        channel.connect(str(path))
        channel.sendall(b"STATUS\n")
        with channel.makefile() as reader:
            return json.loads(reader.readline())


def generate(url, token_id, stream=False):
    payload = {
        "input_ids": [token_id] * 16,
        "stream": stream,
        "sampling_params": {"temperature": 0, "max_new_tokens": 24, "ignore_eos": True},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic_ns()
    with urllib.request.urlopen(request, timeout=300) as http:
        if stream:
            first = count = None
            for line in http:
                if not line.startswith(b"data:"):
                    continue
                event = line[5:].strip()
                if event == b"[DONE]":
                    break
                value = json.loads(event)
                count = value["meta_info"]["completion_tokens"]
                if count and first is None:
                    first = time.monotonic_ns()
            if count != 24 or first is None:
                raise RuntimeError("Streaming fixture did not return 24 tokens")
            return {
                "tokens": count,
                "first_token_ms": (first - start) / 1e6,
                "duration_ms": (time.monotonic_ns() - start) / 1e6,
            }
        value = json.load(http)
        count = value["meta_info"]["completion_tokens"]
        if http.status != 200 or len(value["output_ids"]) != count or count != 24:
            raise RuntimeError("Canary generation did not complete 24 tokens")
        return count


def sample(args):
    before = status(args.worker)
    if before["completed_slices"] == 0 or before["holds"] or before["faulted"]:
        raise RuntimeError("Start a warm, supervised, actively mining canary worker")

    start = time.monotonic_ns()
    held = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(generate, args.url, args.token_id)
        while not pending.done():
            broker = status(args.broker)
            worker = status(args.worker)
            if broker["drained"] == [True] and worker["holds"] and not worker["gpu_busy"]:
                held.append((time.monotonic_ns(), worker["completed_slices"]))
            time.sleep(0.002)
        tokens = pending.result()
    response = time.monotonic_ns()
    if len(held) < 2 or len({counter for _, counter in held}) != 1:
        raise RuntimeError(f"Expected a stationary mining cursor while drained: {held}")

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        after = status(args.worker)
        broker = status(args.broker)
        if (
            start <= after["release_ns"] <= after["first_gpu_seen_ns"] <= after["first_slice_end_ns"]
            and after["completed_slices"] > held[-1][1]
            and not after["faulted"]
            and not after["holds"]
            and broker["idle"] == [True]
            and broker["claims"] == 0
            and broker["watched"] == [True]
        ):
            break
        time.sleep(0.001)
    else:
        raise RuntimeError(f"No resumed worker slice after generation: {after} {broker}")
    return {
        "completion_tokens": tokens,
        "held_observations": len(held),
        "held_completed_slices": held[0][1],
        "before_completed_slices": before["completed_slices"],
        "after_completed_slices": after["completed_slices"],
        "observed_held_window_ms": (held[-1][0] - held[0][0]) / 1e6,
        "http_send_to_first_observed_drain_ms": (held[0][0] - start) / 1e6,
        "http_duration_ms": (response - start) / 1e6,
        "release_to_observed_gpu_entry_ms": (
            after["first_gpu_seen_ns"] - after["release_ns"]
        ) / 1e6,
        "broker_final": broker,
    }


@contextmanager
def parked(path):
    # An independent worker hold composes with the broker's per-request hold.
    with socket.socket(socket.AF_UNIX) as channel:
        channel.settimeout(5)
        channel.connect(str(path))
        with channel.makefile("rwb", buffering=0) as wire:
            wire.write(b"HOLD\n")
            if wire.readline() != b"DRAINED\n":
                raise RuntimeError("Independent test hold failed")
            try:
                yield
            finally:
                wire.write(b"RELEASE\n")
                if wire.readline() != b"RELEASED\n":
                    raise RuntimeError("Independent test release failed")


def paired(args):
    order = random.Random(1017)
    rows = []
    for _ in range(args.samples):
        arms = ["parked", "mining"]
        order.shuffle(arms)
        row = {"order": arms}
        for arm in arms:
            before = status(args.worker)["completed_slices"]
            deadline = time.monotonic() + 5
            while status(args.worker)["completed_slices"] < before + 2:
                if time.monotonic() > deadline:
                    raise RuntimeError("Mining did not resume between test arms")
                time.sleep(0.001)
            with parked(args.worker) if arm == "parked" else nullcontext():
                row[arm] = generate(args.url, args.token_id, stream=True)
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="A dedicated synthetic-model TP1 endpoint")
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--token-id", type=int, default=4)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--paired-ttft", action="store_true",
                        help="Randomize paired streaming requests with the worker parked or mining before arrival")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    result = paired(args) if args.paired_ttft else [sample(args) for _ in range(args.samples)]
    print(json.dumps(result, indent=2))
