"""Experimental SGLang hooks for host-local Pearl arbitration; disabled by default."""
import asyncio
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import socket
import uuid

SOCKET = os.environ.get("PEARL_BROKER_SOCKET")
SHAPE_DIR = os.environ.get("PEARL_SHAPE_DIR")
_OPERATOR_MODULE = os.environ.get("PEARL_WOB_MODULE")
OPERATOR = importlib.import_module(_OPERATOR_MODULE) if _OPERATOR_MODULE else None
OBSERVE = bool(SHAPE_DIR or OPERATOR)
TIMEOUT = 6.0  # broker's default worker-drain deadline is five seconds
_rank = None


def validate_config(args):
    if SOCKET and (args.dp_size != 1 or args.pp_size != 1):
        raise ValueError("Pearl broker handoff requires one host-local TP group (DP1, PP1)")
    if OPERATOR is None:
        return
    graph = args.cuda_graph_config
    if graph.decode.backend != "disabled" or graph.prefill.backend != "disabled" or args.enable_torch_compile:
        raise ValueError("Pearl public projection requires eager prefill and decode. "
                         "Set --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled "
                         "and disable torch.compile.")


async def _rpc(path, line, expected="OK"):
    async with asyncio.timeout(TIMEOUT):
        reader, writer = await asyncio.open_unix_connection(path)
        try:
            writer.write((line + "\n").encode("ascii"))
            await writer.drain()
            reply = await reader.readline()
            if reply != (expected + "\n").encode("ascii"):
                raise RuntimeError(f"Pearl {line.split()[0]} failed: {reply[:160]!r}")
        finally:
            writer.close()
            await writer.wait_closed()


async def _ordered_rpc(path, line, expected="OK"):
    # An incoming cancellation may not roll back a command until the broker
    # has processed it. Otherwise a late DISPATCH could strand a claim.
    op = asyncio.create_task(_rpc(path, line, expected))
    try:
        return await asyncio.shield(op)
    except asyncio.CancelledError:
        await op
        raise


class Admission:
    """Own admission until every wire dispatch has acquired its own child lease."""

    def __init__(self, path):
        self.path = path
        self.token = uuid.uuid4().hex
        self.reader = self.writer = None
        self.drained = False

    async def start(self):
        # A dedicated socket owns this parent. The broker observes EOF even
        # before DRAINED, so tokenizer death or cancellation frees just this lease.
        async with asyncio.timeout(TIMEOUT):
            self.reader, self.writer = await asyncio.open_unix_connection(self.path)
            self.writer.write(f"LEASE {self.token}\n".encode("ascii"))
            await self.writer.drain()

    async def wait(self):
        if not self.drained:
            async with asyncio.timeout(TIMEOUT):
                reply = await self.reader.readline()
            if reply != b"DRAINED\n":
                raise RuntimeError(f"Pearl LEASE failed: {reply[:160]!r}")
            self.drained = True

    async def close(self):
        if self.writer is None:
            return
        try:
            if self.drained:
                async with asyncio.timeout(TIMEOUT):
                    self.writer.write(f"CANCEL {self.token}\n".encode("ascii"))
                    await self.writer.drain()
                    if await self.reader.readline() != b"OK\n":
                        raise RuntimeError("Pearl admission cancellation failed")
        finally:
            self.writer.close()
            await self.writer.wait_closed()
            self.writer = self.reader = None

    async def dispatch(self, send, payload, items):
        await self.wait()
        children = []
        try:
            for item in items:
                child = uuid.uuid4().hex
                children.append(child)
                # Fields survive both the msgspec wire codec and TP broadcast.
                # Opaque tokens are generated locally and never disclose client rids.
                item.pearl_dispatch_id = child
                await _ordered_rpc(self.path, f"DISPATCH {child} {self.token}")
        except BaseException:
            for child in children:
                await _ordered_rpc(self.path, f"DONE {child}")
            raise
        # An ambiguous send exception leaves the child held. Releasing it here
        # could race a message already committed to the scheduler IPC queue.
        send(payload)


class Rank:
    """One ordered persistent stream owned by the scheduler on one TP rank."""

    def __init__(self, index, path, shape_dir=None):
        self.index = index
        self.path = path
        self.shape_dir = Path(shape_dir) if shape_dir else None
        self.counts = Counter()
        self.stream = None
        self.active = None

    def _command(self, line):
        if not self.path:
            return
        if self.stream is None:
            stream = socket.socket(socket.AF_UNIX)
            stream.settimeout(TIMEOUT)
            try:
                stream.connect(self.path)
            except BaseException:
                stream.close()
                raise
            self.stream = stream
        self.stream.sendall((line + "\n").encode("ascii"))
        # Exactly one command is outstanding on this stream.
        data = bytearray()
        while b"\n" not in data and len(data) < 256:
            fragment = self.stream.recv(256)
            if not fragment:
                break
            data.extend(fragment)
        if data != b"OK\n":
            self.close()
            raise RuntimeError(f"Pearl rank command failed: {data[:160]!r}")

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def receive(self, items):
        if not items:
            return
        if self.active is not True:
            self._command(f"BUSY {self.index}")
            self.active = True
        if self.path:
            for item in items:
                child = item.pearl_dispatch_id
                if not child:
                    raise RuntimeError("tokenized request reached scheduler without a Pearl admission")
                self._command(f"TAKE {child} {self.index}")

    def idle(self, synchronize):
        if self.active is False:
            return
        if self.path:
            # The forward, sampler, transfers and NCCL may use different streams.
            # Synchronize once on this rank's actual device per busy-to-idle edge.
            synchronize()
        self._command(f"IDLE {self.index}")
        self.active = False
        if self.shape_dir and self.counts:
            out = self.shape_dir / f"pearl-shapes-{os.getpid()}-tp{self.index}.json"
            tmp = out.with_suffix(".tmp")
            rows = [dict(mode=mode, path=path, real_rows=real, executed_rows=executed, calls=n)
                    for (mode, path, real, executed), n in self.counts.items()]
            tmp.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
            os.replace(tmp, out)


def scheduler_rank(index):
    global _rank
    if _rank is None:
        _rank = Rank(index, SOCKET, SHAPE_DIR)
    elif _rank.index != index:
        raise RuntimeError("Pearl prototype expects one TP scheduler per process")
    return _rank


def real_rows(batch):
    # DP attention padding can replace the visible forward mode and batch size.
    if getattr(batch, "_original_forward_mode", None) is not None:
        return None
    mode = batch.forward_mode.name
    if mode not in ("EXTEND", "DECODE", "TARGET_VERIFY"):
        return None
    if mode == "EXTEND":
        count = batch.extend_num_tokens
        unpadded = getattr(batch, "num_token_non_padded_cpu", None)
        return min(count, unpadded) if count is not None and unpadded is not None else count
    original = getattr(batch, "_original_batch_size", None)
    batch_size = batch.batch_size if original is None else original
    if mode == "DECODE":
        return batch_size
    if mode == "TARGET_VERIFY":
        info = batch.spec_info
        layout = getattr(info, "ragged_verify_layout", None)
        if layout is not None:
            # Device-only ragged layouts can omit the host total. Skip that
            # measurement instead of counting rounded capture-tier padding.
            return layout.total_verify_tokens
        width = getattr(info, "draft_token_num", None)
        return batch_size * width if width is not None else None
    return None


def _record(batch, path, rows, executed):
    if SHAPE_DIR and _rank is not None and _rank.active and rows is not None and 0 < rows <= executed:
        _rank.counts[batch.forward_mode.name, path, rows, executed] += 1


def _capturing():
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
    return get_is_capture_mode()


def project(layer, x, bias, batch):
    # Called only for layer 0's wo_b. Keep warmup and graph capture on the
    # unchanged kernel; public replay candidates must be eager and unpadded.
    if _rank is not None and _rank.active and not _capturing():
        rows = real_rows(batch)
        _record(batch, "eager_wo_b", rows, x.shape[0])
        if OPERATOR is not None and rows == x.shape[0] and rows > 0:
            return OPERATOR.project(layer, x, bias, batch)
    return layer.quant_method.apply(layer, x, bias=bias)


def before_replay(runner, batch):
    # Python in a captured forward does not execute at replay. Count the real
    # incoming target batch on the host, independently of graph padding.
    if runner.model_runner.is_draft_worker or _rank is None or not _rank.active:
        return
    if OPERATOR is not None:
        raise RuntimeError("Pearl public projection requires eager execution for both phases")
    rows = real_rows(batch)
    executed = runner._replay_graph_key.size if runner.ragged_verify_mode else runner.bs * runner.captured_req_width
    _record(batch, "target_graph", rows, executed)
