"""Loopback-only public Stratum job for direct K780 work. Verify before acknowledging."""
import asyncio
import base64
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from contextlib import suppress

MINER = Path("/mnt/hot/ambientlight/repos/mining")
FIXTURE = MINER / "blackwell_pearl_miner-v1/tuning/replay-job-changes.py"
VERIFIER = MINER / "blackwell_pearl_miner-sglang-prototype/target/cargo/release/examples/verify_receipt"

def fixture():
    spec = importlib.util.spec_from_file_location("direct_job_fixture", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Peer:
    def __init__(self, folder):
        self.folder = Path(folder)
        template = fixture()
        self.target = template.INITIAL_TARGET * 32
        self.bits = template.compact(self.target)
        self.job = template.job(51175, 1, self.target, False)
        self.seen = set()
        self.authorized = set()
        self.sessions = set()
        self.failures = []
        self.shares = [0]*4
        self.stop = asyncio.Event()
        self.clients = set()
        self.folder.mkdir(mode=0o700)
        (self.folder/"job.json").write_text(json.dumps(self.job,indent=2)+"\n")

    def verify(self, payload):
        if payload["job_id"] != self.job["job_id"]:
            raise ValueError("stale local job")
        packed = base64.b64decode(payload["plain_proof"], validate=True)
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as src:
            raw = src.read(64*1024*1024+1)
        if len(raw)>64*1024*1024:
            raise ValueError("oversized proof")
        encoded = base64.b64encode(raw)
        fingerprint = hashlib.sha256(encoded).hexdigest()
        record = dict(header=self.job["header"],cert_version=3,share_nbits=self.bits,
                      pool_base_target=self.job["target"],plain_proof=encoded.decode(),proof_sha256=fingerprint)
        with tempfile.NamedTemporaryFile(mode="w",dir=self.folder) as tmp:
            json.dump(record,tmp,separators=(",",":"));tmp.flush()
            out=subprocess.check_output([str(VERIFIER),tmp.name],text=True,timeout=15,
                         env={**os.environ,"RAYON_NUM_THREADS":"2"},stderr=subprocess.STDOUT)
            if out.strip()!="1 pool proofs verified":
                raise ValueError("unexpected independent verifier result")
        return fingerprint

    async def client(self, reader, writer):
        task=asyncio.current_task()
        self.clients.add(task)
        rank=None
        async def send(message):
            writer.write((json.dumps(message,separators=(",",":"))+"\n").encode())
            await writer.drain()
        try:
            while not self.stop.is_set():
                try:
                    line=await asyncio.wait_for(reader.readline(),20)
                except TimeoutError:
                    await send({"method":"client.show_message","params":["public direct 275 W local job unchanged"]})
                    continue
                if not line: break
                req=json.loads(line)
                arrived=time.monotonic_ns()
                mid=req["id"]
                if req["method"]=="mining.authorize":
                    name=req["params"]["worker"]
                    if rank is not None or name not in [f"graph275-{i}" for i in range(4)]:
                        raise ValueError("unexpected worker identity")
                    candidate=int(name[-1])
                    if candidate in self.sessions:
                        raise ValueError("worker reconnected during fixed-job measurement")
                    rank=candidate
                    self.sessions.add(rank)
                    self.authorized.add(rank)
                    await send({"id":mid,"result":True,"error":None,"type":"v2"})
                    await send({"method":"mining.notify","params":self.job})
                elif req["method"]=="mining.submit" and rank is not None:
                    digest=await asyncio.to_thread(self.verify, req["params"])
                    unique=digest not in self.seen
                    if unique:
                        self.seen.add(digest)
                        self.shares[rank]+=1
                        with (self.folder/"ledger.jsonl").open("a") as log:
                            log.write(json.dumps(dict(rank=rank,received_ns=arrived,verified_ns=time.monotonic_ns(),sha256=digest),separators=(",",":"))+"\n")
                    if not unique:
                        self.failures.append("duplicate public witness")
                    await send({"id":mid,"result":unique,"error":None if unique else "duplicate public witness"})
                else:
                    raise ValueError("unexpected local peer method")
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
            if not self.stop.is_set():
                self.failures.append(f"{type(e).__name__}: {str(e)[:120]}")
        finally:
            if rank is not None:
                self.authorized.discard(rank)
            self.clients.discard(task)
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()

    async def __aenter__(self):
        self.server=await asyncio.start_server(self.client,"127.0.0.1",0,limit=16*1024*1024)
        self.port=self.server.sockets[0].getsockname()[1]
        return self
    async def __aexit__(self,*unused):
        self.stop.set()
        self.server.close()
        await self.server.wait_closed()
        tasks=list(self.clients)
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        (self.folder/"summary.json").write_text(json.dumps(dict(target=self.job["target"],verified_unique_by_rank=self.shares,
            errors=self.failures),indent=2)+"\n")
