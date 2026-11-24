#!/usr/bin/env python3
"""Measure a single K780 worker at several tile widths against identical full nonce controls."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from measure import status

HERE = Path(__file__).resolve().parent


def admission(args, width):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "RAYON_NUM_THREADS": "2"}
    with tempfile.TemporaryDirectory() as d:
        sock = str(Path(d) / "worker.sock")
        # Full nonces finish naturally and the worker verifies sampled proofs before exit.
        cmd = [
            str(args.worker.resolve()),
            "--control-socket",
            sock,
            "--bench-rounds",
            "24",
            "--bench-job",
            "71",
            "--bench-nonce",
            "1001",
            "--kernel-variant",
            "v1",
            "--k",
            "65536",
            "--slice-tiles",
            str(width),
            "--power-limit",
            str(args.power_limit),
            "--max-temp",
            str(args.max_temp),
            "--device",
            "0",
        ]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env) as worker:
            try:
                deadline = time.monotonic() + 15
                while not (Path(sock).exists() and status(sock)["completed_slices"] > 2):
                    if worker.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("worker failed to reach warm mining")
                    time.sleep(0.01)
                output = subprocess.check_output(
                    [
                        sys.executable,
                        str(HERE / "measure.py"),
                        sock,
                        "--samples",
                        str(args.samples),
                        "--seed",
                        str(args.seed),
                    ],
                    text=True,
                    timeout=60,
                )
                bench_text, _ = worker.communicate(timeout=60)
                if worker.returncode:
                    raise RuntimeError(f"worker exited {worker.returncode}")
                return {"handoff": json.loads(output), "worker": json.loads(bench_text)}
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()


def main(args):
    runs = []
    for width in args.tiles:
        result = admission(args, width)
        timed = subprocess.check_output(
            [
                sys.executable,
                str(HERE / "speed.py"),
                str(args.worker.resolve()),
                args.gpu,
                "--slice-tiles",
                str(width),
                "--pairs",
                str(args.pairs),
                "--rounds",
                str(args.rounds),
                "--power-limit",
                str(args.power_limit),
                "--max-temp",
                str(args.max_temp),
            ],
            text=True,
            timeout=300,
        )
        result["matched_throughput"] = json.loads(timed)
        runs.append(result)
        print(
            f"width {width}: drain p99 {result['handoff']['hold_to_drained_ms']['p99']} ms, "
            f"retained {result['matched_throughput']['sliced_fraction_of_full']:.2%}",
            file=sys.stderr,
            flush=True,
        )
    digests = {r["worker"]["hit_digest"] for r in runs}
    assert len(digests) == 1, "changing the tile width changed the hits for a fixed offline job"
    return {
        "width_order": args.tiles,
        "k": 65536,
        "samples_per_width": args.samples,
        "pairs_per_width": args.pairs,
        "nonces_per_arm_per_pair": args.rounds,
        "seed": args.seed,
        "runs": runs,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker", type=Path)
    p.add_argument("gpu")
    p.add_argument("--tiles", type=int, nargs="+", default=[188, 376, 752])
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=314159)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--power-limit", type=int, default=280)
    p.add_argument("--max-temp", type=int, default=92)
    print(json.dumps(main(p.parse_args()), indent=2))
