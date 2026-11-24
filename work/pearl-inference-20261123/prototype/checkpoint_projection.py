"""Offline, eager, public-replay wo_b adapter. No external pool connection.

Quantize each rank's dequantized FP8 weight to symmetric INT7 per output row,
and each activation to INT7 per input row. Consume the corrected integer
accumulators in the local projection. Collect tickets against an artificial job.
"""
import contextlib
import ctypes as C
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "operator"))
from pearl_operator import Operator

K, N, TP = 2048, 5120, 4
HEADER_BITS = 0x1D00FFFF
HEADER = struct.pack("<I32s32sII", 0, bytes([1]) * 32, bytes([2]) * 32, 0x66666666, HEADER_BITS)


def e4m3_lookup():
    code = np.arange(256, dtype=np.int32)
    exp, mantissa = (code >> 3) & 15, code & 7
    magnitude = np.where(exp == 0, np.ldexp(mantissa.astype("f"), -9),
                         np.ldexp(1 + mantissa.astype("f") / 8, exp - 7))
    magnitude[(exp == 15) & (mantissa == 7)] = np.nan
    return np.where(code & 128, -magnitude, magnitude).astype("f")


def read_tensor(model, prefix, filename, name, shape, dtype):
    path = model / filename
    with path.open("rb") as f:
        length, = struct.unpack("<Q", f.read(8))
        if length > 100_000_000:
            raise ValueError("invalid safetensors header length")
        meta = json.loads(f.read(length))[prefix + "." + name]
    begin, end = meta["data_offsets"]
    if meta["dtype"] != dtype or meta["shape"] != list(shape) or end - begin != int(np.prod(shape)):
        raise ValueError(f"unsupported {prefix}.{name} format {meta['shape']} {meta['dtype']}")
    return np.memmap(path, mode="r", offset=8 + length + begin, dtype=np.uint8, shape=shape)


def checkpoint_weight(model, rank):
    if rank not in range(TP):
        raise ValueError("expected a TP4 rank")
    model = Path(model)
    prefix = "layers.0.attn.wo_b"
    weights = json.loads((model / "model.safetensors.index.json").read_bytes())["weight_map"]
    raw = read_tensor(model, prefix, weights[prefix + ".weight"], "weight", (N, TP * K), "F8_E4M3")
    raw_scale = read_tensor(model, prefix, weights[prefix + ".scale"], "scale", (N // 32, TP * K // 32), "F8_E8M0")
    shard = raw[:, rank * K:(rank + 1) * K]
    scales = raw_scale[:, rank * K // 32:(rank + 1) * K // 32]
    if np.any(scales == 255):
        raise ValueError("nonfinite checkpoint scale")
    w = e4m3_lookup()[shard]
    grouped = w.reshape(N // 32, 32, K // 32, 32)
    grouped *= np.exp2(scales.astype(np.float32) - 127)[:, None, :, None]
    if not np.isfinite(w).all():
        raise ValueError("nonfinite checkpoint weight")
    return w


def quantize_rows(data):
    scale = np.max(np.abs(data), axis=1, keepdims=True) / 64
    scale[scale == 0] = 1
    units = np.clip(np.rint(data / scale), -64, 64).astype(np.int8)
    return np.ascontiguousarray(units), scale


class Cuda:
    def __init__(self, device):
        self.device = device
        self.lib = C.CDLL(os.environ.get("PEARL_CUDART", "libcudart.so.13"))
        ptr, size, integer = C.c_void_p, C.c_size_t, C.c_int
        for name, args in {
            "cudaSetDevice": [integer], "cudaMalloc": [C.POINTER(ptr), size],
            "cudaFree": [ptr], "cudaMemcpyAsync": [ptr, ptr, size, integer, ptr],
            "cudaStreamSynchronize": [ptr],
        }.items():
            getattr(self.lib, name).argtypes = args
            getattr(self.lib, name).restype = integer
        self.lib.cudaGetErrorString.argtypes = [integer]
        self.lib.cudaGetErrorString.restype = C.c_char_p
        self.call("cudaSetDevice", device)

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name}: {self.lib.cudaGetErrorString(code).decode()}")

    @contextlib.contextmanager
    def allocation(self, size):
        ptr = C.c_void_p()
        self.call("cudaMalloc", C.byref(ptr), size)
        try:
            yield ptr.value
        finally:
            self.call("cudaFree", ptr)


class Projection:
    def __init__(self, model, build, rank, device, *, header=HEADER, share_bits=None, on_witness=None,
                 replay_pool=None):
        self.reference_weight = checkpoint_weight(model, rank)
        self.header = bytes(header)
        self.share_bits = share_bits
        self.on_witness = on_witness
        self.bt, self.scale_b = quantize_rows(self.reference_weight)
        self.cuda = Cuda(device)
        self.op = Operator(build)
        self.gpu_prepare = None
        if os.environ.get("PEARL_GPU_PREPARE") == "1":
            from gpu_prepare.activation import GpuPrepare
            self.gpu_prepare = GpuPrepare(build, self.op)
        self.pool_layouts = None
        if replay_pool is not None:
            from pool_layouts import PoolLayouts
            self.pool_layouts = PoolLayouts(replay_pool, self.op, self.bt)
        self._persistent = contextlib.ExitStack()
        self._device_bt = None
        self._layouts = {}
        self._proof_slots = threading.BoundedSemaphore(2)
        self.busy_skips = 0
        self.eager_counts = {} if os.environ.get("PEARL_GRAPH_VERIFY") == "1" else None
        self.eager_verify = {}
        self.eager_verify_lock = threading.Lock()
        self._proofs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pearl-public-proof")
        self._torch_scale = self._proof_stream = None

    def close(self):
        self._proofs.shutdown(wait=True)
        self.cuda.call("cudaSetDevice", self.cuda.device)
        if self.pool_layouts is not None:
            self.pool_layouts.close()
        self._persistent.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _device_weight(self, stream):
        if self._device_bt is None:
            with contextlib.ExitStack() as staging:
                device_bt = staging.enter_context(self.cuda.allocation(self.bt.nbytes))
                self.cuda.call("cudaMemcpyAsync", device_bt, self.bt.ctypes.data, self.bt.nbytes, 1, stream)
                self.cuda.call("cudaStreamSynchronize", stream)
                self._persistent.enter_context(staging.pop_all())
                self._device_bt = device_bt
        return self._device_bt

    def _weights(self, compact, stream):
        if compact in self._layouts:
            return self._layouts[compact]
        # The fixed public fixture header and quantized checkpoint belong to this instance.
        # A changed job/quantization requires a new instance; each pattern has its own key.
        nb = np.empty_like(self.bt)
        with contextlib.ExitStack() as staging:
            weight = staging.enter_context(self.op.prepare_weight(
                self.header, self.bt.ctypes.data, nb.ctypes.data, N, compact))
            device_bt = self._device_weight(stream)
            device_nb = staging.enter_context(self.cuda.allocation(nb.nbytes))
            self.cuda.call("cudaMemcpyAsync", device_nb, nb.ctypes.data, nb.nbytes, 1, stream)
            self.cuda.call("cudaStreamSynchronize", stream)
            self._persistent.enter_context(staging.pop_all())
        self._layouts[compact] = weight, device_bt, device_nb
        return self._layouts[compact]

    def _verify(self, ref, words, bits, phase, on_witness):
        with ref:
            stats = ref.scan(words.ctypes.data, bits)
            first = stats["first_hit"]
            stats["verified_witness_bytes"] = 0
            if first is not None:
                witness = ref.proof(first, bits)
                stats["verified_witness_bytes"] = len(witness)
                if on_witness is not None:
                    on_witness(phase, witness)
            return stats

    def _finish_proof(self, future, phase):
        try:
            result = future.result()
            if self.eager_counts is not None and phase != "decode_graph":
                with self.eager_verify_lock:
                    counts = self.eager_verify.setdefault(phase, Counter())
                    counts["verified_steps"] += 1
                    counts["tickets_scanned"] += result["tickets"]
                    counts["winning_tickets"] += result["hits"]
                    counts["verified_proofs"] += int(bool(result["verified_witness_bytes"]))
            print(json.dumps({"pearl_public_verified": result, "phase": phase}), flush=True)
        except Exception as exc:
            if self.eager_counts is not None and phase != "decode_graph":
                with self.eager_verify_lock:
                    self.eager_verify.setdefault(phase, Counter())["verification_errors"] += 1
            print(json.dumps({"pearl_public_verification_failed": type(exc).__name__, "phase": phase}), flush=True)
        finally:
            self._proof_slots.release()

    def _plain_output(self, q, device_bt, stream, scale, dtype):
        import torch
        out = torch.empty((len(q), N), dtype=torch.int32, device=q.device)
        self.op.launch_plain(q.data_ptr(), device_bt, out.data_ptr(), len(q), N, stream.cuda_stream)
        return (out.float() * scale * self._torch_scale).to(dtype)

    def _count_eager(self, phase, path, rows):
        if self.eager_counts is not None:
            counts = self.eager_counts.setdefault(phase, Counter())
            counts[path + "_calls"] += 1
            counts[path + "_rows"] += rows

    def torch_forward(self, x, phase):
        import torch
        if x.ndim != 2 or x.shape[1] != K or len(x) == 0:
            raise ValueError("expected nonempty local [rows,2048] activation")
        if len(x) > 1 and len(x) % 2:
            return torch.cat([self.torch_forward(x[:-1], phase), self.torch_forward(x[-1:], phase)])
        stream = torch.cuda.current_stream(x.device)
        if self._torch_scale is None:
            self._torch_scale = torch.from_numpy(self.scale_b.T.copy()).to(x.device)
            self._proof_stream = torch.cuda.Stream(device=x.device)
        values = x.float()
        scale = values.abs().amax(dim=1, keepdim=True) / 64
        scale = torch.where(scale == 0, 1., scale)
        q = torch.round(values / scale).clamp(-64, 64).to(torch.int8).contiguous()
        m = max(2, len(q))
        compact = phase != "prefill" or m % 16 != 0
        if self.pool_layouts is not None:
            layout = self.pool_layouts.select(compact, stream)
            device_bt = self._device_weight(stream.cuda_stream)
            if layout is None:
                self._count_eager(phase, "no_job", len(x))
                return self._plain_output(q, device_bt, stream, scale, x.dtype)
            job, weight, noise_tensor = layout
            device_nb, bits = noise_tensor.data_ptr(), job.share_bits
            on_witness = functools.partial(self.pool_layouts.pool.submit, job)
        else:
            bits = self.share_bits if self.share_bits is not None else (0x1E400000 if compact else 0x1E100000)
            weight, device_bt, device_nb = self._weights(compact, stream.cuda_stream)
            on_witness = self.on_witness
        if not self._proof_slots.acquire(blocking=False):
            self.busy_skips += 1
            self._count_eager(phase, "backpressure", len(x))
            return self._plain_output(q, device_bt, stream, scale, x.dtype)
        ref = None
        submitted = False
        try:
            if len(x) == 1:
                q = torch.cat([torch.div(q, 8, rounding_mode="floor"), q.remainder(8)])
            if self.gpu_prepare is None:
                on_host = q.cpu().numpy()
                noise = np.empty_like(on_host)
                ref = weight.prepare(on_host.ctypes.data, noise.ctypes.data, m)
                na = torch.from_numpy(noise).to(x.device)
            else:
                na, ref = self.gpu_prepare.prepare(q, weight, stream)
            out = torch.empty((m, N), dtype=torch.int32, device=x.device)
            transcripts = torch.empty((ref.ticket_count, 16), dtype=torch.int32, device=x.device)
            self.op.launch(q.data_ptr(), device_bt, na.data_ptr(), device_nb, out.data_ptr(),
                           transcripts.data_ptr(), m, N, compact, stream.cuda_stream)
            if len(x) == 1:
                out = (8 * out[0] + out[1]).unsqueeze(0)
            y = (out.float() * scale * self._torch_scale).to(x.dtype)
            self._proof_stream.wait_stream(stream)
            with torch.cuda.stream(self._proof_stream):
                if self.gpu_prepare is not None:
                    ref.stage(self._proof_stream)
                host = torch.empty(transcripts.shape, dtype=torch.int32, pin_memory=True)
                host.copy_(transcripts, non_blocking=True)
                transcripts.record_stream(self._proof_stream)
                done = self._proof_stream.record_event()
            future = self._proofs.submit(self._verify_copy, ref, host, done, bits, phase, on_witness)
            ref, submitted = None, True
            future.add_done_callback(lambda f: self._finish_proof(f, phase))
            self._count_eager(phase, "mined", len(x))
            return y
        finally:
            if not submitted:
                self._proof_slots.release()
            if ref is not None:
                ref.close()

    def _verify_copy(self, ref, host, event, bits, phase, on_witness):
        try:
            event.synchronize()
            return self._verify(ref, host.numpy(), bits, phase, on_witness)
        finally:
            ref.close()

    def forward(self, x, phase, stream=0, check=False):
        self.cuda.call("cudaSetDevice", self.cuda.device)
        if x.ndim != 2 or x.shape[1] != K or x.shape[0] == 0 or not np.isfinite(x).all():
            raise ValueError("expected finite, nonempty [m,2048] activations")
        x = np.ascontiguousarray(x, dtype=np.float32)
        # A last odd row gets its own genuine one-token projection, with both planes consumed.
        if len(x) > 1 and len(x) % 2:
            y0, r0 = self.forward(x[:-1], phase, stream, check)
            y1, r1 = self.forward(x[-1:], phase, stream, check)
            return np.concatenate([y0, y1]), r0 + r1
        a, scale_a = quantize_rows(x)
        singleton = len(x) == 1
        if singleton:
            a = np.concatenate([a // 8, a % 8])
        m = len(a)
        compact = phase != "prefill" or m % 16 != 0
        share_bits = self.share_bits if self.share_bits is not None else (0x1E400000 if compact else 0x1E100000)
        na = np.empty_like(a)
        out = np.empty((m, N), np.int32)
        words = np.empty((m // 2 * (N // (16 if compact else 64)), 16), np.uint32)
        started = time.perf_counter()
        weight, device_bt, device_nb = self._weights(compact, stream)
        weights_ready = time.perf_counter()
        ref = weight.prepare(a.ctypes.data, na.ctypes.data, m)
        try:
            prepared = time.perf_counter()
            with contextlib.ExitStack() as stack:
                inputs = (a, na)
                buffers = [stack.enter_context(self.cuda.allocation(arr.nbytes))
                           for arr in (*inputs, out, words)]
                for ptr, arr in zip(buffers, inputs):
                    self.cuda.call("cudaMemcpyAsync", ptr, arr.ctypes.data, arr.nbytes, 1, stream)
                self.op.launch(buffers[0], device_bt, buffers[1], device_nb,
                               buffers[2], buffers[3], m, N, compact, stream)
                for ptr, arr in zip(buffers[2:], (out, words)):
                    self.cuda.call("cudaMemcpyAsync", arr.ctypes.data, ptr, arr.nbytes, 2, stream)
                self.cuda.call("cudaStreamSynchronize", stream)
            calculated = time.perf_counter()
            stats = self._verify(ref, words, share_bits, phase, self.on_witness)
        finally:
            ref.close()
        verified = time.perf_counter()
        if check:
            # Each dot can be accumulated exactly in FP64 and then compared in INT32.
            # Use a spread of output coordinates when the prefill call is large.
            idx = np.arange(N) if m <= 16 else np.unique(np.linspace(0, N - 1, 64, dtype=int))
            expected = a.astype(np.float64) @ self.bt[idx].astype(np.float64).T
            np.testing.assert_array_equal(out[:, idx], expected)
            stats["exact_integer_outputs_checked"] = int(expected.size)
        if singleton:
            out = (8 * out[0] + out[1])[None, :]
        y = (out.astype(np.float32) * scale_a) * self.scale_b.T
        stats.update(phase=phase, input_rows=len(x), proof_rows=m, compact=compact,
                     weight_cache_ms=1e3 * (weights_ready - started),
                     host_prepare_ms=1e3 * (prepared - weights_ready),
                     gpu_with_copies_alloc_ms=1e3 * (calculated - prepared),
                     host_verify_ms=1e3 * (verified - calculated))
        return y, [stats]


@functools.lru_cache(maxsize=1)
def _candidate(model, build, rank, device):
    pool = None
    if os.environ.get("PEARL_REPLAY_POOL_PORT"):
        from replay_pool import ReplayPool
        pool = ReplayPool.from_env()
    try:
        return Projection(model, build, rank, device, replay_pool=pool)
    except BaseException:
        if pool is not None:
            pool.close()
        raise


def project(layer, x, bias, forward_batch):
    """Opt-in eager SGLang callback. The caller still owns the original TP collective."""
    import torch
    if os.environ.get("PEARL_PUBLIC_REPLAY") != "1":
        raise RuntimeError("checkpoint prototype requires an isolated public replay")
    if bias is not None or not x.is_cuda or torch.cuda.is_current_stream_capturing():
        raise ValueError("prototype expects eager CUDA row-parallel wo_b without bias")
    mode = forward_batch.forward_mode
    phase = "decode" if (mode.is_decode() or mode.is_target_verify()) else "prefill"
    device = x.device.index
    if hasattr(layer, "tp_rank"):
        rank = layer.tp_rank
    else:
        from sglang.srt.distributed.utils import get_group_rank_size
        rank, _ = get_group_rank_size(layer.tp_group)
    adapter = _candidate(os.environ["PEARL_CHECKPOINT_DIR"], os.environ["PEARL_OPERATOR_BUILD"],
                         int(rank), device)
    with torch.cuda.device(device):
        return adapter.torch_forward(x.detach(), phase)
