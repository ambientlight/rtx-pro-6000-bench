"""Public TP4 SM120 canary. Capture the calibrated layer-zero projection in DSPARK's target graph."""

import ctypes as C
import functools
import os
import threading
from collections import Counter

import torch

import checkpoint_projection as eager
from gpu_prepare.activation import DeferredOpening
from pearl_operator import _bind

# Normal target prefill retains the already-qualified eager adapter.
project = eager.project
_by_device = {}


class Workspace:
    def __init__(self, rows, prep, device):
        self.q = torch.empty((rows, eager.K), dtype=torch.int8, device=device)
        self.meta = torch.empty(64, dtype=torch.uint8, device=device)
        self.noise = torch.empty_like(self.q)
        self.scratch = torch.empty(prep.scratch_bytes(rows), dtype=torch.uint8, device=device)
        self.words = torch.empty((rows // 2 * (eager.N // 16), 16), dtype=torch.int32, device=device)


class GraphVerify:
    def __init__(self, adapter, build, device):
        if adapter.gpu_prepare is None or adapter.pool_layouts is None:
            raise ValueError("Graph verify requires resident GPU preparation and a loopback job peer")
        self.adapter = adapter
        self.device = device
        self.workspaces = {}
        self.current_job = None
        self.before_stream = None
        self.completed = torch.cuda.Event()
        self.material = torch.zeros(64, dtype=torch.uint8, device=device)
        self.noise_b = torch.empty((eager.N, eager.K), dtype=torch.int8, device=device)
        self.enabled = torch.zeros(1, dtype=torch.uint8, device=device)
        self.weight_scale = torch.from_numpy(adapter.scale_b.T.copy()).to(device)
        self.bt = adapter._device_weight(torch.cuda.current_stream(device).cuda_stream)
        p, i = C.c_void_p, C.c_int
        self.prepare = _bind(C.CDLL(str(build / "libpearl_prepare.so")), "pearl_activation_prepare_graph",
                             [p]*7 + [C.c_size_t, i, i, p, p], i)
        self.gemm = _bind(C.CDLL(str(build / "libpearl_wo_b.so")), "pearl_wo_b_i7_graph",
                          [p]*7 + [i, i, p], i)
        self.stats = Counter()
        self.stats_lock = threading.Lock()

    def capture(self, x):
        rows = x.shape[0]
        if x.ndim != 2 or x.shape[1] != eager.K or rows not in range(6, 49, 6):
            raise ValueError("Expected one to eight real six-row DSPARK verify slots")
        stream = torch.cuda.current_stream(self.device)
        ws = self.workspaces.get(rows)
        if ws is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Projection buffers were not allocated in graph warmup")
            ws = self.workspaces[rows] = Workspace(rows, self.adapter.gpu_prepare, self.device)
        if not torch.cuda.is_current_stream_capturing():
            # Dry/warm forwards consume the calibrated plain projection and never open proofs.
            self.enabled.zero_()
            self.adapter.pool_layouts.select(True, stream)

        values = x.float()
        scale = values.abs().amax(dim=1, keepdim=True) / 64
        scale = torch.where(scale == 0, 1., scale)
        ws.q.copy_(torch.round(values / scale).clamp(-64, 64))
        output = torch.empty((rows, eager.N), dtype=torch.int32, device=self.device)
        flag = self.enabled.data_ptr()
        rc = self.prepare(ws.q.data_ptr(), self.material.data_ptr(), self.material.data_ptr()+32,
                          ws.meta.data_ptr(), ws.meta.data_ptr()+32, ws.noise.data_ptr(), ws.scratch.data_ptr(),
                          ws.scratch.numel(), rows, eager.N, stream.cuda_stream, flag)
        if rc:
            raise RuntimeError(f"Graph activation launch failed ({rc})")
        rc = self.gemm(ws.q.data_ptr(), self.bt, ws.noise.data_ptr(), self.noise_b.data_ptr(),
                       output.data_ptr(), ws.words.data_ptr(), flag, rows, eager.N, stream.cuda_stream)
        if rc:
            raise RuntimeError(f"Graph projection launch failed ({rc})")
        return (output.float() * scale * self.weight_scale).to(x.dtype)

    def begin(self, rows, executed, eligible):
        stream = torch.cuda.current_stream(self.device)
        if self.before_stream is not None and self.before_stream != stream.cuda_stream:
            stream.wait_event(self.completed)
        ws = self.workspaces[executed]
        # The flag, job material, input snapshot, and graph run are ordered on the same stream.
        if not eligible:
            self.enabled.zero_()
            return "target_graph_ineligible", None
        layout = self.adapter.pool_layouts.select(True, stream)
        if layout is None:
            self.enabled.zero_()
            return "target_graph_no_job", None
        if not self.adapter._proof_slots.acquire(blocking=False):
            self.enabled.zero_()
            return "target_graph_backpressure", None
        job, weight, noise = layout
        prep = self.adapter.gpu_prepare
        ref = None
        try:
            if self.current_job is not job:
                self.material.copy_(prep.material(weight, self.device))
                self.noise_b.copy_(noise)
                self.current_job = job
            with weight._lock:
                owner = prep._retain(weight.ctx)
            if not owner:
                raise RuntimeError("Could not retain graph job weight")
            ref = DeferredOpening(prep, owner, ws.q, ws.meta, len(ws.words))
            self.enabled.fill_(1)
            return "target_graph_mined", (ws, job, ref)
        except BaseException:
            if ref is not None:
                ref.close()
            self.adapter._proof_slots.release()
            raise

    def end(self, proof, success):
        stream = torch.cuda.current_stream(self.device)
        try:
            if proof is None:
                return
            ws, job, ref = proof
            submitted = False
            try:
                if success:
                    # The next graph replay may immediately overwrite ws. Queue D2H on
                    # the serving stream before returning. The worker waits on its event.
                    ref.stage(stream)
                    host = torch.empty(ws.words.shape, dtype=torch.int32, pin_memory=True)
                    host.copy_(ws.words, non_blocking=True)
                    done = stream.record_event()
                    future = self.adapter._proofs.submit(
                        self.adapter._verify_copy, ref, host, done, job.share_bits, "decode",
                        functools.partial(self.adapter.pool_layouts.pool.submit, job))
                    submitted = True
                    future.add_done_callback(self._finished)
            finally:
                if not submitted:
                    ref.close()
                    self.adapter._proof_slots.release()
        finally:
            self.completed.record(stream)
            self.before_stream = stream.cuda_stream

    def _finished(self, future):
        try:
            result = future.result()
        except Exception:
            with self.stats_lock:
                self.stats["verification_errors"] += 1
        else:
            with self.stats_lock:
                self.stats["verified_steps"] += 1
                self.stats["tickets_scanned"] += result["tickets"]
                self.stats["winning_tickets"] += result["hits"]
                self.stats["verified_proofs"] += int(bool(result["verified_witness_bytes"]))
        finally:
            self.adapter._finish_proof(future, "decode_graph")


def capture_project(layer, x, bias, batch):
    from pathlib import Path
    if os.environ.get("PEARL_PUBLIC_REPLAY") != "1" or bias is not None or isinstance(x, tuple) or not x.is_cuda:
        raise ValueError("Graph projection requires an isolated public GPU replay without bias")
    if not batch.forward_mode.is_target_verify():
        raise ValueError("Graph capture expects plain target-verification activations")
    device = x.device.index
    graph = _by_device.get(device)
    if graph is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Projection was not initialized during pre-capture warmup")
        rank = getattr(layer, "tp_rank", None)
        if rank is None:
            from sglang.srt.distributed.utils import get_group_rank_size
            rank, _ = get_group_rank_size(layer.tp_group)
        build = os.environ["PEARL_OPERATOR_BUILD"]
        adapter = eager._candidate(os.environ["PEARL_CHECKPOINT_DIR"], build, int(rank), device)
        graph = _by_device[device] = GraphVerify(adapter, Path(build), x.device)
    return graph.capture(x)


def begin_graph(device, rows, executed, eligible):
    graph = _by_device[device]
    path, proof = graph.begin(rows, executed, eligible)
    return path, (graph, proof)


def end_graph(state, success):
    graph, proof = state
    graph.end(proof, success)


def counters():
    result = {}
    for device, graph in _by_device.items():
        with graph.stats_lock, graph.adapter.eager_verify_lock:
            result[str(device)] = {"verify": dict(graph.stats), "eager": graph.adapter.eager_counts,
                                   "eager_verify": {k: dict(v) for k, v in graph.adapter.eager_verify.items()},
                                   "pool": graph.adapter.pool_layouts.pool.counters()}
    return result
