"""In-memory Stratum checks. No network socket, serving GPU, pool account or live job is needed."""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest

from gateway import Gateway, digest, frame, parse_job


class Writer:
    def __init__(self):
        self.frames = asyncio.Queue()
        self.closed = False

    def write(self, data):
        if self.closed:
            raise ConnectionError("closed")
        self.frames.put_nowait(json.loads(data))

    async def drain(self):
        pass

    def close(self):
        self.closed = True


def public_job(job_id="upstream-id"):
    return dict(job_id=job_id, header="01" * 76, target=f"{2**204 - 1:064x}", cert_version=3)


def req(gate, origin=None):
    return dict(id=7, method="mining.submit", params=dict(job_id=origin or gate.job_id,
                plain_proof="fixture", phase="decode"))


def verified(msg, job):
    return dict(proof_sha256=digest(b"public-witness"), phase=msg["params"]["phase"],
                tp_rank=2, proof_rows=6, opened_weight_bytes=32768)


class TestGateway(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.gate = Gateway("dummy-account", "joinedverify", "token", verified, self.path/"public.json")
        self.writer = Writer()
        self.gate.up = self.writer
        self.gate.generation = 1
        self.gate.install(parse_job(public_job(), 1))

    async def test_matching_upstream_ack_and_evidence(self):
        t = asyncio.create_task(self.gate.submit(req(self.gate)))
        out = await self.writer.frames.get()
        self.assertEqual(out["params"], {"job_id": "upstream-id", "plain_proof": "fixture"})
        self.assertEqual(self.gate.state["accepted"], 0)
        self.gate.finish(out["id"]+9, "accepted")
        self.assertFalse(t.done())
        self.gate.finish(out["id"], "accepted")
        self.assertTrue(await t)
        self.assertFalse(await self.gate.submit(req(self.gate)))
        receipt = json.loads((self.path/"public.json").read_text())
        self.assertEqual(receipt["accepted"], 1)
        self.assertEqual(receipt["duplicate"], 1)
        self.assertEqual(receipt["last_accepted"]["tp_rank"], 2)
        self.assertEqual(receipt["last_accepted"]["target"], public_job()["target"])
        first = receipt["last_accepted"]
        self.gate.install(parse_job(dict(public_job("second"), header="02" * 76), 1))
        later = asyncio.create_task(self.gate.submit(req(self.gate)))
        second = await self.writer.frames.get()
        self.gate.finish(second["id"], "accepted")
        self.assertTrue(await later)
        self.assertEqual(self.gate.state["first_accepted"], first)
        self.assertNotEqual(self.gate.state["last_accepted"], first)
        self.assertNotIn("plain_proof", json.dumps(receipt))

    async def test_stale_under_verification_and_arm_edges(self):
        gate = self.gate
        gate.armed = False
        first_id = gate.job_id
        self.assertFalse(await gate.submit(req(gate)))
        gate.armed = True
        gate.refresh()
        self.assertFalse(await gate.submit(req(gate, first_id)))

        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        def slow(msg, job):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(4):
                raise RuntimeError("test timed out")
            return verified(msg, job)
        gate.verify = slow
        t = asyncio.create_task(gate.submit(req(gate)))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            gate.install(parse_job(public_job("new"), 1))
        finally:
            release.set()
        self.assertFalse(await t)
        self.assertEqual(gate.state["stale"], 2)
        self.assertEqual(gate.state["disarmed"], 1)
        self.assertEqual(gate.state["submitted"], 0)

    async def test_invalid_proof_or_unknown_upstream_status(self):
        def fail(msg, job):
            raise ValueError("false proof")
        self.gate.verify = fail
        self.assertFalse(await self.gate.submit(req(self.gate)))
        self.assertEqual(self.gate.state["invalid"], 1)
        self.gate.verify = verified
        t = asyncio.create_task(self.gate.submit(req(self.gate)))
        await self.writer.frames.get()
        self.gate.disconnect()
        self.assertFalse(await t)
        self.assertEqual(self.gate.state["unknown"], 1)
        self.assertEqual(self.gate.state["accepted"], 0)
        self.assertFalse(self.gate.state["connected"])

    async def test_real_auth_protocol_job_before_ack_and_reject(self):
        gate = self.gate
        gate.disconnect()
        upstream, reader = Writer(), asyncio.StreamReader()
        run = asyncio.create_task(gate.upstream_session(reader, upstream))
        auth = await upstream.frames.get()
        self.assertEqual(auth["method"], "mining.authorize")
        reader.feed_data(frame(dict(method="mining.notify", params=public_job())))
        await asyncio.sleep(.01)
        self.assertIsNone(gate.job)
        reader.feed_data(frame(dict(id=1, result=True, error=None, type="v2")))
        for _ in range(100):
            if gate.job:
                break
            await asyncio.sleep(.001)
        self.assertEqual(gate.job.share_bits, parse_job(public_job(), gate.generation).share_bits)
        candidate = asyncio.create_task(gate.submit(req(gate)))
        submit = await upstream.frames.get()
        reader.feed_data(frame(dict(id=submit["id"], result=False, error=[21, "job not found", None])))
        self.assertFalse(await candidate)
        self.assertEqual(gate.state["rejected"], 1)
        self.assertEqual(gate.state["pool_stale"], 1)
        self.assertIsNone(gate.state["last_accepted"])
        reader.feed_eof()
        with self.assertRaises(ValueError):
            await run

    async def test_local_auth_gate_and_notification(self):
        for token, admitted in (("wrong",False),("token",True)):
            reader, writer = asyncio.StreamReader(), Writer()
            reader.feed_data(frame(dict(method="mining.authorize", id=1, params={"pass":token,"type":"v2"})))
            t = asyncio.create_task(self.gate.local(reader, writer))
            self.assertEqual((await writer.frames.get())["result"], admitted)
            if admitted:
                job = (await writer.frames.get())["params"]
                self.assertEqual(job["job_id"], self.gate.job_id)
                self.assertEqual(job["header"], self.gate.job.header.hex())
            reader.feed_eof()
            await t
            self.assertEqual(len(self.gate.clients), 0)


if __name__ == "__main__":
    unittest.main()
