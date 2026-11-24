#!/usr/bin/env python3
"""Pair unchanged-depth-2 and sliced-depth-1 idle rates with matching nonces."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("worker", type=Path)
p.add_argument("gpu")
p.add_argument("--rounds", type=int, default=8)
p.add_argument("--pairs", type=int, default=4)
p.add_argument("--slice-tiles", type=int, default=188)
p.add_argument("--power-limit", type=int, default=280)
p.add_argument("--max-temp", type=int, default=92)
args = p.parse_args()
env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "RAYON_NUM_THREADS": "2"}
pairs = []
for i in range(args.pairs):
    order = ("full", "sliced") if i % 2 == 0 else ("sliced", "full")
    sample = {"order": order, "header": 109 + i, "first_nonce_control": 1001 + 16 * i}
    for mode in order:
        with tempfile.TemporaryDirectory() as d:
            cmd = [
                str(args.worker.resolve()),
                "--bench-rounds",
                str(args.rounds),
                "--bench-job",
                str(sample["header"]),
                "--bench-nonce",
                str(sample["first_nonce_control"] + (8 if mode == "sliced" else 0)),
                "--device",
                "0",
                "--k",
                "65536",
                "--kernel-variant",
                "v1",
                "--power-limit",
                str(args.power_limit),
                "--max-temp",
                str(args.max_temp),
            ]
            if mode == "sliced":
                # Stock benchmarks run eight warmups before scoring; offset the
                # sliced nonce so both arms score identical committed operands.
                cmd += [
                    "--control-socket",
                    str(Path(d) / "worker.sock"),
                    "--slice-tiles",
                    str(args.slice_tiles),
                ]
            result = json.loads(subprocess.check_output(cmd, text=True, env=env))
            if result["verified_proofs"] <= 0:
                raise RuntimeError("no proof verified")
            sample[mode] = {
                k: result[k]
                for k in ("rounds", "hits", "hit_digest", "verified_proofs", "seconds", "tmac_s")
            }
    assert sample["sliced"]["hit_digest"] == sample["full"]["hit_digest"]
    pairs.append(sample)
ratio = sum(v["full"]["seconds"] for v in pairs) / sum(v["sliced"]["seconds"] for v in pairs)
print(
    json.dumps(
        {
            "gpu": args.gpu,
            "k": 65536,
            "slice_tiles": args.slice_tiles,
            "sliced_fraction_of_full": ratio,
            "pairs": pairs,
        },
        indent=2,
    )
)
