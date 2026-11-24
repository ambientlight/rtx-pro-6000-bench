"""Run the SM120 operator with the real DeepSeek layer-0 weights and seeded inputs.

The four TP partial outputs are summed on CPU for evaluation. No model request,
NCCL collective, live share, or customer activations are used by this runner.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import time

import numpy as np

from checkpoint_projection import K, TP, Projection, e4m3_lookup


def bf16_float(x):
    u = x.view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def fp8_activations(x):
    blocks = x.reshape(len(x), -1, 32)
    maxima = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True), 1e-4)
    scale = np.exp2(np.ceil(np.log2(maxima / 448)))
    scaled = np.minimum(np.abs(blocks / scale), 448.)
    lut = e4m3_lookup()[:127]
    upper = np.searchsorted(lut, scaled)
    lower = np.maximum(0, upper - 1)
    below, above = scaled - lut[lower], lut[upper] - scaled
    take_upper = (above < below) | ((below == above) & ((upper & 1) == 0))
    rounded = lut[np.where(take_upper, upper, lower)]
    return (np.copysign(rounded, blocks) * scale).reshape(x.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="Four physical GPUs or one device for a sequential TP4 replay")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 6, 2048])
    ap.add_argument("--receipt", type=Path)
    args = ap.parse_args()
    if len(args.devices) not in (1, 4) or any(i <= 0 for i in args.rows):
        ap.error("expected one or four GPU devices and positive rows")
    cases = []
    with contextlib.ExitStack() as stack:
        workers = [stack.enter_context(Projection(args.model, args.build, rank,
                                                  args.devices[rank % len(args.devices)]))
                   for rank in range(TP)]
        for m in args.rows:
            phase = "decode" if m <= 6 else "prefill"
            generator = np.random.default_rng(17000 + m)
            full_x = bf16_float(generator.standard_normal((m, TP * K), dtype=np.float32))
            idx = np.unique(np.linspace(0, m - 1, min(m, 16), dtype=int))
            baseline = measured = None
            shards = []
            for rank in range(TP):
                worker = workers[rank]
                x = np.ascontiguousarray(full_x[:, rank*K:(rank+1)*K])
                projected, stats = worker.forward(x, phase, check=True)
                ref = fp8_activations(x[idx]) @ worker.reference_weight.T
                if measured is None:
                    measured, baseline = projected[idx].copy(), ref
                else:
                    measured += projected[idx]
                    baseline += ref
                shards.append(dict(rank=rank, device=args.devices[rank % len(args.devices)], calls=stats))
            v = measured.astype(np.float64).ravel()
            r = baseline.astype(np.float64).ravel()
            err = float(np.linalg.norm(v-r) / np.linalg.norm(r))
            case = dict(phase=phase, real_rows=m, fp8_reference_rows=len(idx),
                        fp8_reference_outputs=len(r), fp8_relative_l2=err,
                        fp8_cosine=float(np.dot(r,v) / (np.linalg.norm(r)*np.linalg.norm(v))),
                        shards=shards)
            if not np.isfinite(err):
                raise ValueError("nonfinite projection")
            print(f"{phase} rows={m} relative_L2={err:.6f} cosine={case['fp8_cosine']:.6f}", flush=True)
            cases.append(case)
    if args.receipt:
        doc = dict(unix_time=time.time(), layer="layers.0.attn.wo_b", model=args.model.name,
                   source="public checkpoint, synthetic BF16 activations; artificial job",
                   comparison="E4M3 activations with per-token 32-wide E8M0 scales and dequantized checkpoint weights, FP32 reference accumulation",
                   reduction="CPU sum of four per-shard predictions", cuda_visible=os.getenv("CUDA_VISIBLE_DEVICES"),
                   cases=cases)
        args.receipt.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
