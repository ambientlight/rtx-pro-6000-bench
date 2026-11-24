"""Optional process-local Stratum v2 transport for public, loopback replays."""
import asyncio
import base64
from collections import deque
import contextlib
from dataclasses import dataclass
import gzip
import ipaddress
import json
import os
from pathlib import Path
import re
import threading
import time

MAX_FRAME = 256 * 1024  # Deployed Pearl Stratum limit including newline.
MAX_WITNESS = 1024 * 1024
PHASES = ("prefill", "decode")
# rank=128, K=2048, largest layout opens two rows and 64 columns.
MAX_FACTOR = 2 * 64 * 2048


def compact(target):
    """Floor a positive uint256 to Bitcoin compact (never relax the target)."""
    if not 0 < target < 1 << 256:
        raise ValueError("invalid uint256 target")
    size = (target.bit_length() + 7) // 8
    shift = 8 * (size - 3)
    mantissa = target >> shift if shift >= 0 else target << -shift
    if mantissa & 0x800000:
        mantissa >>= 8
        size += 1
    return (size << 24) | mantissa


@dataclass(frozen=True)
class Job:
    id: str
    header: bytes
    share_bits: int
    generation: int
    target: int


def parse_job(params, generation):
    if not isinstance(params, dict):
        raise ValueError("missing job")
    job_id, version = params.get("job_id"), params.get("cert_version")
    header, target = params.get("header"), params.get("target")
    if type(version) is not int or version != 3 or not isinstance(job_id, str) or not job_id:
        raise ValueError("expected nonempty job id and certificate v3")
    if not isinstance(header, str) or not re.fullmatch(r"[0-9a-fA-F]{152}", header):
        raise ValueError("expected 76-byte header")
    if not isinstance(target, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", target):
        raise ValueError("expected 32-byte target")
    target = int(target, 16)
    bits = compact(target)
    shift = 8 * ((bits >> 24) - 3)
    mantissa = bits & 0x7fffff
    floor = mantissa << shift if shift >= 0 else mantissa >> -shift
    if floor * MAX_FACTOR >= 1 << 256:
        raise ValueError("scaled share target overflows prefill uint256")
    return Job(job_id, bytes.fromhex(header), bits, generation, target)


def frame(message):
    data = json.dumps(message, separators=(",", ":")).encode() + b"\n"
    if len(data) > MAX_FRAME:
        raise ValueError("Stratum frame exceeds 256 KiB")
    return data


def submission_frame(serial, job, raw, phase=None):
    wire = base64.b64encode(gzip.compress(raw, compresslevel=1, mtime=0)).decode("ascii")
    params = dict(job_id=job.id, plain_proof=wire)
    if phase is not None:
        params["phase"] = phase
    return frame(dict(id=serial, method="mining.submit", params=params))


class ReplayPool:
    """Own one session per process. Producers only share short memory-only sections."""

    @classmethod
    def from_env(cls):
        port = os.environ.get("PEARL_REPLAY_POOL_PORT")
        if port is None:
            return None
        token_file = os.environ.get("PEARL_REPLAY_GATEWAY_TOKEN_FILE")
        token = None
        if token_file:
            token = Path(token_file).read_text().strip()
            if not re.fullmatch(r"[0-9a-fA-F]{64}", token):
                raise ValueError("invalid local gateway credential")
        return cls(os.environ.get("PEARL_REPLAY_POOL_HOST", "127.0.0.1"), int(port), token=token)

    def __init__(self, host, port, *, capacity=8, token=None):
        host = "127.0.0.1" if host == "localhost" else host
        address = ipaddress.ip_address(host)  # Numeric loopback only; no DNS.
        if not address.is_loopback or not 0 < port < 65536 or capacity < 1:
            raise ValueError("expected loopback endpoint and a positive queue capacity")
        self._host, self._port, self._capacity = str(address), port, capacity
        self._token = token
        self._lock = threading.Lock()
        self._closed = False
        self._job = self._loop = self._main = self._wake = None
        self._writer = None
        self._queue = deque()
        self._pending = {}  # Outstanding request ID -> phase. Bounded with queue.
        self._counts = {p: dict(enqueued=0, ack=0, reject=0, stale=0, drop=0) for p in PHASES}
        self._thread = threading.Thread(target=self._run, name="pearl-loopback-pool", daemon=True)
        self._thread.start()

    def snapshot(self):
        return self._job

    def counters(self):
        with self._lock:
            return {phase: values.copy() for phase, values in self._counts.items()}

    def submit(self, job, phase, witness):
        """Queue a completed witness. All compression, framing and I/O run elsewhere."""
        if phase not in PHASES:
            raise ValueError("unsupported projection phase")
        with self._lock:
            if self._closed or job is None or job is not self._job:
                self._counts[phase]["stale"] += 1
                return False
            if (not isinstance(witness, bytes) or not 0 < len(witness) <= MAX_WITNESS
                    or len(self._queue) + len(self._pending) >= self._capacity):
                self._counts[phase]["drop"] += 1
                return False
            self._queue.append((job, phase, witness))
            self._counts[phase]["enqueued"] += 1
            self._loop.call_soon_threadsafe(self._wake.set)
            return True

    def _replace(self, job):
        with self._lock:
            if self._closed:
                job = None
            if job == self._job:
                return
            for _, phase, _ in self._queue:
                self._counts[phase]["stale"] += 1
            for phase in self._pending.values():
                self._counts[phase]["stale"] += 1
            self._queue.clear()
            self._pending.clear()
            self._job = job

    def close(self):
        with self._lock:
            self._closed = True
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._shutdown)
        self._replace(None)
        self._thread.join()

    def _shutdown(self):
        if self._writer is not None:
            self._writer.close()
        if self._main is not None:
            self._main.cancel()

    def _run(self):
        with contextlib.suppress(asyncio.CancelledError):
            asyncio.run(self._reconnect())

    async def _reconnect(self):
        with self._lock:
            if self._closed:
                return
            self._loop = asyncio.get_running_loop()
            self._main = asyncio.current_task()
            self._wake = asyncio.Event()
        generation = 0
        try:
            while not self._closed:
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(self._host, self._port, limit=MAX_FRAME), 2)
                    generation += 1
                    self._writer = writer
                    try:
                        if not self._closed:
                            await self._session(reader, writer, generation)
                    finally:
                        self._writer = None
                        self._replace(None)
                        writer.close()
                        with contextlib.suppress(OSError):
                            await asyncio.wait_for(writer.wait_closed(), 1)
                except (OSError, ValueError, TimeoutError):
                    if not self._closed:
                        await asyncio.sleep(.1)
        finally:
            self._replace(None)
            with self._lock:
                self._main = self._loop = self._wake = None

    async def _session(self, reader, writer, generation):
        writer.write(frame(dict(id=1, method="mining.authorize", params={
            "wallet": "fixture", "worker": "public-projection", "pass": self._token or "x", "password": self._token or "x",
            "agent": "pearl-sglang-prototype", "type": "v2"})))
        await asyncio.wait_for(writer.drain(), 2)
        sender = asyncio.create_task(self._send(writer))
        receiver = asyncio.create_task(self._receive(reader, generation))
        try:
            await asyncio.gather(sender, receiver)
        finally:
            for task in (sender, receiver):
                task.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)

    async def _receive(self, reader, generation):
        last_job = None
        authorized = False
        auth_deadline = time.monotonic() + 10
        while True:
            timeout = 120 if authorized and last_job else max(.001, auth_deadline - time.monotonic())
            raw = await asyncio.wait_for(reader.readline(), timeout)
            if not raw.endswith(b"\n") or len(raw) > MAX_FRAME:
                raise ValueError("missing or oversized incoming Stratum frame")
            msg = json.loads(raw)
            if not isinstance(msg, dict):
                raise ValueError("invalid Stratum message")
            method, serial = msg.get("method"), msg.get("id")
            if method == "client.reconnect":
                raise ConnectionError("local fixture requested reconnect")
            if method == "mining.notify":
                last_job = parse_job(msg.get("params"), generation)
                if authorized:
                    self._replace(last_job)
            elif method is None and type(serial) is int:
                ok = msg.get("result") is True and msg.get("error") is None
                if serial == 1:
                    if authorized or not ok or msg.get("type") != "v2":
                        raise ValueError("Stratum v2 authorization failed")
                    authorized = True
                    if last_job:
                        self._replace(last_job)
                else:
                    with self._lock:
                        phase = self._pending.pop(serial, None)
                        if phase:
                            self._counts[phase]["ack" if ok else "reject"] += 1

    async def _send(self, writer):
        serial = 1
        while True:
            self._wake.clear()
            with self._lock:
                if self._queue:
                    item = self._queue.popleft()
                    serial += 1
                    self._pending[serial] = item[1]
                else:
                    item = None
            if item is None:
                await self._wake.wait()
                continue
            job, phase, witness = item
            try:
                phase_args = (phase,) if self._token else ()
                outgoing = await asyncio.to_thread(submission_frame, serial, job, witness, *phase_args)
            except ValueError:
                with self._lock:
                    if self._pending.pop(serial, None) is not None:
                        self._counts[phase]["drop"] += 1
                continue
            # While compression was off-thread, the reader may have rotated the job.
            if job is self._job:
                writer.write(outgoing)
                await asyncio.wait_for(writer.drain(), 2)
