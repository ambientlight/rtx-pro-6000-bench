"""Forward checkpoint-verified openings from authorized inference to the configured F2Pool account."""
import argparse
import asyncio
import base64
from collections import OrderedDict
import gzip
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[5] / "work/pearl-inference-20261123/prototype"))
from replay_pool import MAX_FRAME, MAX_WITNESS, frame, parse_job

HOST, PORT = "pearl.f2pool.com", 5500
PREFLIGHT = Path("/mnt/hot/ambientlight/repos/mining/pool-trial/five-gpu/f2pool/preflight.json")
VERIFIER_SHA = "3986db322ca4df28c5896b8fbce97707e668d731a0e23912e8e7b177b6ed263c"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def secret(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW)) as f:
        s = os.fstat(f.fileno())
        if (not stat.S_ISREG(s.st_mode) or s.st_mode & 0o077 or
                s.st_uid not in (0, os.getuid()) or s.st_size > 1 << 20):
            raise ValueError("credential must be a private regular file owned by this user or root")
        return f.read().strip()


async def message(reader, timeout):
    line = await asyncio.wait_for(reader.readline(), timeout)
    if not line.endswith(b"\n") or len(line) > MAX_FRAME:
        raise ValueError("invalid Stratum frame")
    msg = json.loads(line)
    if not isinstance(msg, dict):
        raise ValueError("invalid Stratum object")
    return msg


def public_verifier(model, binary):
    # Load checkpoint shard mappings once; the verifier operates on actual opened bytes.
    sys.path.insert(0, str(HERE.parents[2] / "full-canary"))
    from public_pool import checkpoint_weight, inspect_opening, quantize_rows
    if digest(binary.read_bytes()) != VERIFIER_SHA:
        raise ValueError("wrong independent verifier build")
    shards = [memoryview(quantize_rows(checkpoint_weight(model, rank))[0]).cast("B")
              for rank in range(4)]

    def verify(msg, job):
        phase = msg["params"].get("phase")
        if phase not in ("prefill", "decode"):
            raise ValueError("missing public inference phase")
        details = inspect_opening(msg, shards)
        rows = details["proof_rows"]
        if phase == "decode" and rows not in range(6, 49, 6):
            raise ValueError("wrong DSPARK projection shape")
        if phase == "prefill" and not (2 <= rows <= 2048 and rows % 2 == 0):
            raise ValueError("wrong prefill projection shape")
        compressed = base64.b64decode(msg["params"]["plain_proof"], validate=True)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as zipped:
            raw = zipped.read(MAX_WITNESS + 1)
        if not 0 < len(raw) <= MAX_WITNESS:
            raise ValueError("oversized opening")
        b64 = base64.b64encode(raw)
        receipt = dict(header=job.header.hex(), cert_version=3, share_nbits=job.share_bits,
                       pool_base_target=f"{job.target:064x}", plain_proof=b64.decode(),
                       proof_sha256=digest(b64))
        # The canonical CLI takes a filename. A memfd never writes secret-derived data to disk.
        with os.fdopen(os.memfd_create("pearl-public-share"), "w+") as f:
            json.dump(receipt, f, separators=(",", ":"))
            f.flush()
            f.seek(0)
            subprocess.run([str(binary), f"/proc/self/fd/{f.fileno()}"], pass_fds=(f.fileno(),),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=15)
        return dict(phase=phase, tp_rank=details["tp_rank"], proof_rows=rows, proof_sha256=digest(raw),
                    opened_weight_bytes=details["opened_weight_bytes"])
    return verify


class Gateway:
    def __init__(self, account, worker, token, verify, path, arm_file=None):
        self.account, self.worker, self.token, self.verify = account, worker, token, verify
        self.path, self.arm_file = path, arm_file
        self.armed = arm_file is None
        self.up = self.job = None
        self.job_id = None
        self.generation = self.revision = 0
        self.next_id = 2
        self.clients = set()
        self.pending = {}
        self.unconfirmed = OrderedDict()
        self.seen = OrderedDict()
        self.state = dict(run_id=secrets.token_hex(16), started_utc_epoch=time.time(),
                          endpoint=f"{HOST}:{PORT}", scope="joined inference submissions only; direct workers use separate pool connections",
                          phases={phase: dict(submitted=0, accepted=0, rejected=0, unknown=0,
                                              accepted_expected_macs=0) for phase in ("prefill", "decode")},
                          verifier_sha256=VERIFIER_SHA, submitted=0, accepted=0, rejected=0,
                          stale=0, unknown=0, invalid=0, duplicate=0, disarmed=0,
                          pool_stale=0, notifications=0, connections=0, first_accepted=None,
                          last_accepted=None, last_submission=None)
        self.save()

    def save(self):
        info = None
        if self.job:
            info = dict(id=self.job_id, generation=self.generation, target=f"{self.job.target:064x}",
                        share_bits=self.job.share_bits, header_sha256=digest(self.job.header),
                        upstream_job_sha256=digest(self.job.id.encode()))
        self.state.update(utc_epoch=time.time(), connected=self.up is not None and bool(self.job),
                          armed=self.armed, generation=self.generation, job=info,
                          authorized_clients=len(self.clients), pending=len(self.pending))
        with tempfile.NamedTemporaryFile("w", dir=self.path.parent, delete=False) as f:
            json.dump(self.state, f, indent=2)
            f.write("\n")
        os.replace(f.name, self.path)

    def notify(self, writer):
        if self.job:
            writer.write(frame(dict(method="mining.notify", params=dict(
                job_id=self.job_id, cert_version=3, header=self.job.header.hex(),
                target=f"{self.job.target:064x}"))))

    def refresh(self):
        self.revision += 1
        self.job_id = f"public-{self.generation}-{self.revision}"
        for writer in tuple(self.clients):
            try:
                self.notify(writer)
            except OSError:
                writer.close()
        self.save()

    def install(self, job):
        self.state["notifications"] += 1
        if job != self.job:
            self.job = job
            self.refresh()
        else:
            self.save()
            # Genuine repeated notifies also refresh the rank's read watchdog.
            for writer in tuple(self.clients):
                self.notify(writer)

    async def watch_arm(self):
        heartbeat = time.monotonic() + 30
        while True:
            await asyncio.sleep(.2)
            enabled = self.arm_file is None or self.arm_file.is_file()
            if enabled != self.armed:
                self.armed = enabled
                self.refresh()
            if time.monotonic() >= heartbeat:
                for writer in tuple(self.clients):
                    try:
                        writer.write(frame(dict(method="client.show_message", params=["gateway alive"])))
                    except OSError:
                        writer.close()
                heartbeat = time.monotonic() + 30

    def disconnect(self):
        self.up = self.job = self.job_id = None
        self.unconfirmed.clear()
        for writer in tuple(self.clients):
            writer.close()
        for key in tuple(self.pending):
            self.finish(key, "unknown")
        self.save()

    def finish(self, serial, outcome):
        item = self.pending.pop(serial, None)
        if item:
            future, record = item
        else:
            record = self.unconfirmed.pop(serial, None)
            if record is None:
                return
            self.state["unknown"] -= 1
            self.state["phases"][record["phase"]]["unknown"] -= 1
            future = None
        self.state[outcome] += 1
        phase = self.state["phases"][record["phase"]]
        phase[outcome] += 1
        settled = dict(record, outcome=outcome, reply_monotonic_ns=time.monotonic_ns())
        self.state["last_settled"] = settled
        if outcome == "accepted":
            phase["accepted_expected_macs"] += (1 << 256) // (int(record["target"], 16) + 1)
            self.state["last_accepted"] = settled
            if self.state["first_accepted"] is None:
                self.state["first_accepted"] = settled
        self.save()
        if future is not None and not future.done():
            future.set_result(outcome == "accepted")

    async def upstream_session(self, reader, writer):
        self.generation += 1
        self.state["connections"] += 1
        self.next_id = 2
        writer.write(frame(dict(id=1, method="mining.authorize", params=dict(
            wallet=self.account, worker=self.worker, password="x", agent="bpm/0.2.0", type="v2",
            **{"pass": "x"}))))
        await asyncio.wait_for(writer.drain(), 3)
        offered = None
        auth_deadline = time.monotonic() + 15
        try:
            while True:
                msg = await message(reader, 120 if self.up and self.job else max(.01, auth_deadline-time.monotonic()))
                if msg.get("method") == "mining.notify":
                    offered = parse_job(msg.get("params"), self.generation)
                    if self.up:
                        self.install(offered)
                elif msg.get("method") == "client.reconnect":
                    raise ConnectionError("pool requested reconnect")
                elif msg.get("method") is None and type(msg.get("id")) is int:
                    accepted = msg.get("result") is True and msg.get("error") is None
                    if msg["id"] == 1:
                        if self.up or not accepted or msg.get("type") != "v2":
                            raise ValueError("F2Pool v2 authorization failed")
                        self.up = writer
                        if offered:
                            self.install(offered)
                    else:
                        if not accepted and (msg["id"] in self.pending or msg["id"] in self.unconfirmed):
                            error = msg.get("error")
                            if (isinstance(error, list) and error and error[0] == 21
                                    or "stale" in str(error).lower() or "job not found" in str(error).lower()):
                                self.state["pool_stale"] += 1
                        self.finish(msg["id"], "accepted" if accepted else "rejected")
        finally:
            self.disconnect()

    async def connect(self):
        while True:
            writer = None
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(HOST, PORT, limit=MAX_FRAME), 10)
                await self.upstream_session(reader, writer)
            except (OSError, TimeoutError, ValueError) as e:
                self.state["last_connection_issue"] = type(e).__name__
                self.save()
            finally:
                if writer:
                    writer.close()
            await asyncio.sleep(3)

    async def submit(self, msg):
        params = msg.get("params")
        if not isinstance(params, dict) or not isinstance(params.get("plain_proof"), str):
            self.state["invalid"] += 1
            self.save()
            return False
        job, local_id, upstream = self.job, self.job_id, self.up
        if not job or not upstream or params.get("job_id") != local_id:
            self.state["stale"] += 1
            self.save()
            return False
        if not self.armed:
            self.state["disarmed"] += 1
            self.save()
            return False
        try:
            details = await asyncio.to_thread(self.verify, msg, job)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError, EOFError, AssertionError, IndexError,
                KeyError, TypeError, OverflowError, struct.error):
            self.state["invalid"] += 1
            self.save()
            return False
        if not self.armed or self.job is not job or self.job_id != local_id or self.up is not upstream:
            self.state["stale"] += 1
            self.save()
            return False
        key = (job.header, details["proof_sha256"])
        if key in self.seen:
            self.state["duplicate"] += 1
            self.save()
            return False
        serial = self.next_id
        try:
            wire = frame(dict(id=serial, method="mining.submit",
                              params=dict(job_id=job.id, plain_proof=params["plain_proof"])))
        except ValueError:
            self.state["invalid"] += 1
            self.save()
            return False
        self.seen[key] = None
        if len(self.seen) > 8192:
            self.seen.popitem(last=False)
        self.next_id += 1
        future = asyncio.get_running_loop().create_future()
        record = dict(details, generation=self.generation, local_job=local_id,
                      upstream_job_sha256=digest(job.id.encode()), request_id=serial,
                      header_sha256=digest(job.header), target=f"{job.target:064x}", share_bits=job.share_bits,
                      sent_monotonic_ns=time.monotonic_ns())
        self.pending[serial] = (future, record)
        self.state["submitted"] += 1
        self.state["phases"][record["phase"]]["submitted"] += 1
        self.state["last_submission"] = record
        self.save()
        try:
            upstream.write(wire)
            await asyncio.wait_for(upstream.drain(), 3)
            return await asyncio.wait_for(asyncio.shield(future), 25)
        except (TimeoutError, OSError):
            # A closed/replaced session may have recycled the numeric request ID.
            if self.pending.get(serial, (None,))[0] is future:
                self.finish(serial, "unknown")
                if self.up is upstream:
                    self.unconfirmed[serial] = record
                    if len(self.unconfirmed) > 256:
                        self.unconfirmed.popitem(last=False)
            return future.result() if future.done() else False

    async def local(self, reader, writer):
        try:
            auth = await message(reader, 10)
            params = auth.get("params", {})
            supplied = params.get("pass") if isinstance(params, dict) else None
            valid = (auth.get("method") == "mining.authorize" and type(auth.get("id")) is int
                     and isinstance(supplied, str) and supplied.isascii()
                     and hmac.compare_digest(self.token, supplied) and params.get("type") == "v2"
                     and len(self.clients) < 4)
            writer.write(frame(dict(id=auth.get("id"), result=valid, error=None, type="v2")))
            if not valid:
                return
            self.clients.add(writer)
            self.notify(writer)
            self.save()
            await asyncio.wait_for(writer.drain(), 3)
            while True:
                req = await message(reader, None)
                if req.get("method") != "mining.submit" or type(req.get("id")) is not int:
                    raise ValueError("invalid local message")
                ok = await self.submit(req)
                writer.write(frame(dict(id=req["id"], result=ok, error=None if ok else "unaccepted")))
                await asyncio.wait_for(writer.drain(), 3)
        except (OSError, TimeoutError, ValueError):
            pass
        finally:
            self.clients.discard(writer)
            writer.close()
            self.save()


async def run(args):
    account = json.loads(secret(args.account_file))["f2pool_mining_account"]
    token = secret(args.token_file)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", account) or account.lower().startswith("prl1"):
        raise ValueError("invalid configured F2Pool mining account")
    if not re.fullmatch(r"[A-Za-z0-9]{1,32}", args.worker) or not re.fullmatch(r"[0-9a-fA-F]{64}", token):
        raise ValueError("invalid verification worker or local gateway credential")
    if args.arm_file and args.arm_file.exists():
        raise ValueError("new run requires an absent arm marker")
    if args.receipt.exists():
        raise ValueError("new run requires a fresh receipt")
    verify = public_verifier(args.model, args.verifier.resolve())
    gw = Gateway(account, args.worker, token, verify, args.receipt, args.arm_file)
    server = await asyncio.start_server(gw.local, "127.0.0.1", args.port, limit=MAX_FRAME)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    tasks = [asyncio.create_task(gw.connect()), asyncio.create_task(gw.watch_arm())]
    async with server:
        await stop.wait()
        # Python 3.12 waits for active streams when the server context exits.
        # Resident SGLang ranks reconnect when their current stream closes.
        server.close()
        for writer in tuple(gw.clients):
            writer.close()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    gw.disconnect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--account-file", type=Path, default=PREFLIGHT)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--verifier", required=True, type=Path)
    p.add_argument("--token-file", required=True, type=Path)
    p.add_argument("--arm-file", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--worker", default="joinedverify")
    p.add_argument("--port", type=int, default=31990)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
