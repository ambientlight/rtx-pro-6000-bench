"""Pair complete warm eager projection calls on one checkpoint shard with CPU/GPU preparation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from checkpoint_projection import K, N, Projection
from gpu_prepare.activation import GpuPrepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--pairs", type=int, default=12)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    if args.pairs < 2:
        p.error("supply at least two pairs")
    torch.cuda.set_device(0)
    stream = torch.cuda.current_stream()
    os.environ.pop("PEARL_GPU_PREPARE", None)
    witnesses = []
    def witness(phase, raw):
        witnesses.append(hashlib.sha256(raw).hexdigest())

    def run(adapter, x, phase):
        begin = time.perf_counter_ns()
        value = adapter.torch_forward(x, phase)
        stream.synchronize()
        ready = time.perf_counter_ns()
        # Drain the proof slot after timing. Both arms start every call with an empty queue.
        with adapter._proof_slots, adapter._proof_slots:
            assert len(witnesses) == 1, "expected exactly one verified witness in this public fixture"
            return value, witnesses.pop(), (ready - begin) / 1e6

    rows = []
    with Projection(args.model, args.build, args.rank, 0, on_witness=witness) as adapter:
        gpu = GpuPrepare(args.build, adapter.op)
        for m in (1, 6, 2048):
            gen = torch.Generator().manual_seed(219 + m)
            phase = "prefill" if m == 2048 else "decode"
            obs = {"cpu": [], "gpu": []}
            deltas = []
            for i in range(-2, args.pairs):
                x = torch.randn((m, K), generator=gen).to(device=0, dtype=torch.bfloat16)
                stream.synchronize()
                order = ("cpu", "gpu") if i % 2 == 0 else ("gpu", "cpu")
                result = {}
                for mode in order:
                    adapter.gpu_prepare = gpu if mode == "gpu" else None
                    result[mode] = run(adapter, x, phase)
                assert torch.equal(result["cpu"][0], result["gpu"][0])
                assert result["cpu"][1] == result["gpu"][1]
                if i >= 0:
                    for mode in order:
                        obs[mode].append(result[mode][2])
                    deltas.append(result["gpu"][2] - result["cpu"][2])
            rows.append(dict(rows=m, output_coordinates_per_pair=m*N, identical_witness_pairs=args.pairs,
                             cpu_median_ms=float(np.median(obs["cpu"])), gpu_median_ms=float(np.median(obs["gpu"])),
                             paired_delta_median_ms=float(np.median(deltas)), samples_ms=obs))
        assert adapter.busy_skips == 0
    args.receipt.write_text(json.dumps(dict(scope="single-rank native prototype eager callback; fixed public header, "
        "changed synthetic BF16 activations per pair, 2 warmup pairs, alternating order, existing production F2Pool worker. "
        "Wall time through serving-stream completion; includes Python/Torch quantization, activation preparation, "
        "noisy GEMM, scaling and proof scheduling. Proof verification drained outside the timed interval. Excludes TP reduction and full server.",
        model=args.model.name, rank=args.rank, gpu=torch.cuda.get_device_name(0),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"), cases=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
