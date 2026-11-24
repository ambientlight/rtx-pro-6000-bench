#!/usr/bin/env python3
"""Measure an already running prototype worker. No pool connection or inference is created."""

import argparse
import json
import math
import random
import socket
import time


def connect(path):
    sock = socket.socket(socket.AF_UNIX)
    sock.settimeout(5)
    try:
        sock.connect(path)
    except Exception:
        sock.close()
        raise
    return sock


def line(sock):
    with sock.makefile("rb") as source:
        value = source.readline(4096)
    if not value.endswith(b"\n"):
        raise RuntimeError("incomplete worker reply")
    return value


def status(path):
    with connect(path) as sock:
        sock.sendall(b"STATUS\n")
        return json.loads(line(sock))


def hold(path):
    sock = connect(path)
    try:
        sock.sendall(b"HOLD\n")
        if line(sock) != b"DRAINED\n":
            raise RuntimeError("worker did not acknowledge drain")
    except Exception:
        sock.close()
        raise
    return sock


def release(sock):
    try:
        sock.sendall(b"RELEASE\n")
        if line(sock) != b"RELEASED\n":
            raise RuntimeError("worker did not acknowledge release")
    finally:
        sock.close()


def quantiles(values):
    s = sorted(values)
    return {
        label: round(s[math.ceil(len(s) * q) - 1], 4)
        for label, q in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1.0)]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("socket")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    previous = status(args.socket)
    if previous["faulted"] or previous["holds"]:
        raise RuntimeError("worker already held or faulted")
    # Verify nested admissions: while any lease remains, no new slices can commit.
    first = hold(args.socket)
    before = status(args.socket)["completed_slices"]
    second = hold(args.socket)
    release(first)
    try:
        time.sleep(0.012)
        during = status(args.socket)
        assert during["holds"] == 1 and not during["gpu_busy"] and during["completed_slices"] == before
    finally:
        release(second)
    rng = random.Random(args.seed)
    samples = []
    for _ in range(args.samples):
        time.sleep(rng.uniform(0, 0.013))
        start = time.monotonic_ns()
        sock = hold(args.socket)
        drained = time.monotonic_ns()
        time.sleep(rng.uniform(0, 0.002))
        sent = time.monotonic_ns()
        release(sock)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            s = status(args.socket)
            if s["release_ns"] >= sent and s["first_slice_end_ns"] >= s["release_ns"]:
                break
            time.sleep(0.0005)
        else:
            raise TimeoutError("worker never completed first slice")
        assert (
            s["release_ns"]
            <= s["first_slice_begin_ns"]
            <= s["first_gpu_seen_ns"]
            <= s["first_slice_end_ns"]
        )
        samples.append(
            [
                (drained - start) / 1e6,
                (s["first_slice_begin_ns"] - sent) / 1e6,
                (s["first_gpu_seen_ns"] - sent) / 1e6,
                (s["first_slice_end_ns"] - sent) / 1e6,
            ]
        )
    rows = zip(*samples)
    print(
        json.dumps(
            {
                "samples": len(samples),
                "nested_hold_checked": True,
                "seed": args.seed,
                "hold_to_drained_ms": quantiles(next(rows)),
                "release_to_first_slice_host_begin_ms": quantiles(next(rows)),
                "release_to_gpu_start_observed_ms": quantiles(next(rows)),
                "release_to_first_slice_completed_ms": quantiles(next(rows)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
