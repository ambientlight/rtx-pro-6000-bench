#!/usr/bin/env python3
"""Check real Stratum job changes and wire proofs while the GPU producer is held."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from types import SimpleNamespace

from measure import hold, release, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker", type=Path)
    parser.add_argument("replay", type=Path, help="miner tuning/replay-job-changes.py")
    parser.add_argument("verifier", type=Path, help="miner release/examples/verify_receipt")
    parser.add_argument("gpu", help="GPU UUID")
    parser.add_argument("--power-limit", type=int, default=280)
    parser.add_argument("--max-temp", type=int, default=92)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("job_replay", args.replay)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    cfg = SimpleNamespace(
        interval=2, jobs=5, target=fixture.INITIAL_TARGET * 256, paired_tip=False, transition=False
    )
    with tempfile.TemporaryDirectory() as temp, socket.socket() as server:
        folder = Path(temp)
        control = str(folder / "worker.sock")
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(15)
        cmd = [
            str(args.worker.resolve()),
            "--pool",
            f"127.0.0.1:{server.getsockname()[1]}",
            "--wallet",
            "loopback-fixture",
            "--worker",
            "sliced",
            "--device",
            "0",
            "--kernel-variant",
            "v1",
            "--variant",
            "780",
            "--k",
            "65536",
            "--control-socket",
            control,
            "--slice-tiles",
            "188",
            "--power-limit",
            str(args.power_limit),
            "--max-temp",
            str(args.max_temp),
            "--log",
            str(folder / "events.jsonl"),
            "--seconds",
            "10",
            "--json",
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "RAYON_NUM_THREADS": "2"}
        with (folder / "stdout").open("w") as out:
            proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)
            try:
                with server.accept()[0] as peer, ThreadPoolExecutor(1) as threads:
                    result = threads.submit(fixture.serve, peer, proc, cfg, 42, folder)
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline and not Path(control).exists():
                        time.sleep(0.01)
                    time.sleep(0.4)
                    lease = hold(control)
                    before = status(control)["completed_slices"]
                    try:
                        # A full fixture notification interval must elapse while paused.
                        time.sleep(2.5)
                        assert status(control)["completed_slices"] == before
                    finally:
                        release(lease)
                    _, count, _, late_tip, tips = result.result(timeout=35)
                assert count > 0 and len(tips) == cfg.jobs
                verified = subprocess.check_output(
                    [str(args.verifier.resolve()), *sorted(folder.glob("proof-*.json"))],
                    text=True,
                    env=env,
                )
                assert verified.strip() == f"{count} pool proofs verified"
                logs = [json.loads(t) for t in (folder / "events.jsonl").read_text().splitlines()]
                last = next(r for r in reversed(logs) if r["event"] == "complete")
                jobs = [r["job_id"] for r in logs if r["event"] == "sliced_job"]
                assert jobs[-1] == f"work-{cfg.jobs - 1}" and last["accepted"] == count
                print(
                    json.dumps(
                        {
                            "jobs_issued": len(tips),
                            "jobs_opened": jobs,
                            "held_through_refresh": True,
                            "wire_proofs_verified": count,
                            "previous_tip_inflight": late_tip,
                            "local_stale": last["stale"],
                            "rounds": last["rounds"],
                        },
                        indent=2,
                    )
                )
            except Exception:
                print((folder / "stdout").read_text()[-1500:])
                raise
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()


if __name__ == "__main__":
    main()
