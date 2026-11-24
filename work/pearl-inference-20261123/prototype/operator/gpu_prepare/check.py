"""Check and time key/shape/activation rotation and the captured preparation kernels."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from checkpoint_projection import HEADER, K, N, checkpoint_weight, quantize_rows
from pearl_operator import Operator
from gpu_prepare.activation import GpuPrepare


def samples_gpu(fn, stream, repeats):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        begin.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        times.append(begin.elapsed_time(end))
    return times


def summary(values):
    return dict(median=statistics.median(values), p95=float(np.percentile(values, 95)),
                min=min(values), max=max(values), samples=len(values))


def check_retirement(op, gpu, bt, nb, stream):
    # Retire the originating weight while its projection and proof copy are in flight.
    # Each pending opening independently retains the Rust weight tree.
    m = 6
    q = torch.from_numpy(np.random.default_rng(515).integers(-64, 65, (m, K), dtype=np.int8)).cuda()
    with op.prepare_weight(HEADER, bt.ctypes.data, nb.ctypes.data, N, True) as weight:
        db = torch.from_numpy(bt).cuda()
        dn = torch.from_numpy(nb).cuda()
        na, pending = gpu.prepare(q, weight, stream)
        _, damaged = gpu.prepare(q, weight, stream)
        out = torch.empty((m, N), dtype=torch.int32, device=q.device)
        words = torch.empty((pending.ticket_count, 16), dtype=torch.int32, device=q.device)
        op.launch(q.data_ptr(), db.data_ptr(), na.data_ptr(), dn.data_ptr(), out.data_ptr(),
                  words.data_ptr(), m, N, True, stream.cuda_stream)
        copy = torch.cuda.Stream(device=q.device)
        copy.wait_stream(stream)
        with torch.cuda.stream(copy):
            pending.stage(copy)
            damaged.stage(copy)
    with pending, damaged:
        copy.synchronize()
        host = words.cpu().numpy()
        stats = pending.scan(host.ctypes.data, 0x1e400000)
        assert stats["first_hit"] is not None
        size = len(pending.proof(stats["first_hit"], 0x1e400000))
        damaged.host_meta[0] ^= 1
        try:
            damaged.scan(host.ctypes.data, 0x1e400000)
        except ValueError as exc:
            if "independent Rust verification" not in str(exc):
                raise
        else:
            raise AssertionError("corrupt root was accepted")
    return dict(closed_weight_proof_bytes=size, corrupt_root_rejected=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=32)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("use at least one repetition")
    op = Operator(args.build)
    gpu = GpuPrepare(args.build, op)
    rng = np.random.default_rng(20261123)
    bt, _ = quantize_rows(checkpoint_weight(args.model, args.rank))
    nb = np.empty_like(bt)
    results = []
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for m in (2, 6, 10, 16, 2048):
            compact = m % 16 != 0
            host = rng.integers(-64, 65, (m, K), dtype=np.int8)
            q = torch.from_numpy(host.copy()).cuda()
            material = torch.empty(64, dtype=torch.uint8, device="cuda")
            meta = torch.empty_like(material)
            noise = torch.empty_like(q)
            scratch = torch.empty(gpu.scratch_bytes(m), dtype=torch.uint8, device="cuda")
            def launch():
                gpu.launch(q, material, meta, noise, scratch, N, stream)

            def compare(weight):
                reference_noise = np.empty_like(host)
                with weight.prepare(host.ctypes.data, reference_noise.ctypes.data, m) as ref:
                    expected = (C.c_uint8 * 64)()
                    assert gpu.reference_material(ref.ctx, expected) == 0
                    assert bytes(meta.cpu().numpy()) == bytes(expected)
                    np.testing.assert_array_equal(noise.cpu().numpy(), reference_noise)
                    return bytes(expected).hex()

            with op.prepare_weight(HEADER, bt.ctypes.data, nb.ctypes.data, N, compact) as w:
                material.copy_(gpu.material(w, q.device))
                for _ in range(4):
                    launch()
                initial = compare(w)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    launch()
                graph.replay()
                assert compare(w) == initial
                host[0, 0] = -63 if host[0, 0] == 64 else int(host[0, 0]) + 1
                q.copy_(torch.from_numpy(host))
                graph.replay()
                updated = compare(w)
                assert updated != initial
                times = samples_gpu(launch, stream, args.repeats)
                graph_times = samples_gpu(graph.replay, stream, args.repeats)
                cpu = []
                na = np.empty_like(host)
                for _ in range(args.repeats):
                    start = time.perf_counter_ns()
                    with w.prepare(host.ctypes.data, na.ctypes.data, m):
                        pass
                    cpu.append((time.perf_counter_ns() - start) / 1e6)
            header2 = bytearray(HEADER)
            header2[17] ^= 0x80
            with op.prepare_weight(bytes(header2), bt.ctypes.data, nb.ctypes.data, N, compact) as w:
                material.copy_(gpu.material(w, q.device))
                graph.replay()
                rotated = compare(w)
                assert rotated != updated
            case = dict(rows=m, compact=compact, scratch_bytes=scratch.numel(), checked_noise_bytes=m*K,
                        root_and_seed_exact=True, changed_input_exact=True, rotated_job_graph_exact=True,
                        cuda_eager_ms=summary(times), cuda_graph_ms=summary(graph_times), cpu_prepare_ms=summary(cpu))
            results.append(case)
            print(json.dumps(case), flush=True)
        retirement = check_retirement(op, gpu, bt, nb, stream)
    args.receipt.write_text(json.dumps(dict(
        model=args.model.name, rank=args.rank, gpu=torch.cuda.get_device_name(0),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"), rayon_threads=os.getenv("RAYON_NUM_THREADS"),
        measurement="CUDA events on one stream with resident buffers and existing production F2Pool worker. CPU timings are cached-weight Rust activation preparation including noise and release. Quantization, proof processing, transfers and GEMM excluded.",
        retired_generation=retirement, cases=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
