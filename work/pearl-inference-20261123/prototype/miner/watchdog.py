#!/usr/bin/env python3
"""Test initial parking and permit revocation and recovery after supervisor loss."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from measure import connect, hold, line, release, status


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise TimeoutError("watchdog check timed out")


def test(worker, gpu, power_limit, max_temp):
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "watchdog.sock")
        with subprocess.Popen(
            [
                str(worker.resolve()),
                "--control-socket",
                path,
                "--require-watch",
                "--bench-rounds",
                "3",
                "--bench-job",
                "71",
                "--bench-nonce",
                "1001",
                "--kernel-variant",
                "v1",
                "--k",
                "65536",
                "--slice-tiles",
                "188",
                "--power-limit",
                str(power_limit),
                "--max-temp",
                str(max_temp),
                "--device",
                "0",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "RAYON_NUM_THREADS": "2"},
            stdout=subprocess.PIPE,
            text=True,
        ) as proc:
            try:
                wait_for(lambda: Path(path).exists())
                initial = status(path)
                assert initial["require_watch"] and not initial["watched"] and not initial["gpu_busy"]
                time.sleep(0.1)
                assert status(path)["completed_slices"] == 0

                def arm():
                    held = hold(path)
                    watcher = connect(path)
                    watcher.sendall(b"WATCH\n")
                    assert line(watcher) == b"WATCHING\n"
                    assert status(path)["watched"] and not status(path)["gpu_busy"]
                    release(held)
                    return watcher

                with connect(path) as rejected:
                    rejected.sendall(b"WATCH\n")
                    assert rejected.recv(32) == b""
                watcher = arm()
                # Sample observed in-flight operations before the first completed
                # mining slice. GPU context/engine/job setup use separate permits.
                cold_holds = []
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    row = status(path)
                    if row["completed_slices"]:
                        break
                    if not row["gpu_busy"]:
                        time.sleep(0.0002)
                        continue
                    arrival = time.monotonic_ns()
                    lease = hold(path)
                    drained = time.monotonic_ns()
                    after = status(path)
                    cold_holds.append(
                        {
                            "hold_to_drained_ms": (drained - arrival) / 1e6,
                            "completed_slices_at_ack": after["completed_slices"],
                        }
                    )
                    release(lease)
                wait_for(lambda: status(path)["completed_slices"] >= 3)
                before_drop = status(path)["completed_slices"]
                watcher.close()
                wait_for(lambda: not status(path)["watched"] and not status(path)["gpu_busy"])
                parked = status(path)["completed_slices"]
                time.sleep(0.15)
                assert status(path)["completed_slices"] == parked
                with arm():
                    wait_for(lambda: status(path)["completed_slices"] > parked)
                    out, _ = proc.communicate(timeout=25)
                assert proc.returncode == 0
                bench = json.loads(out)
                return {
                    "early_inflight_holds": cold_holds,
                    "started_parked": True,
                    "required_initial_hold": True,
                    "rearmed": True,
                    "completed_before_drop": before_drop,
                    "completed_when_parked": parked,
                    "frozen_while_disconnected_ms": 150,
                    "worker": bench,
                }
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker", type=Path)
    p.add_argument("gpu")
    p.add_argument("--power-limit", type=int, default=280)
    p.add_argument("--max-temp", type=int, default=92)
    args = p.parse_args()
    print(json.dumps(test(args.worker, args.gpu, args.power_limit, args.max_temp), indent=2))
