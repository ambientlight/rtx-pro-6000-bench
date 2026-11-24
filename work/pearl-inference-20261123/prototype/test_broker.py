"""Exercise broker races over real Unix sockets, without GPUs or a pool."""
import asyncio
import contextlib
from pathlib import Path
import tempfile
import unittest

from broker import Fleet


class Socket:
    @classmethod
    async def open(cls, path):
        self = cls()
        self.reader, self.writer = await asyncio.open_unix_connection(path)
        return self

    async def send(self, message):
        self.writer.write((message + "\n").encode())
        await self.writer.drain()
        return (await self.reader.readline()).decode().strip()

    async def close(self):
        self.writer.close()
        await self.writer.wait_closed()


class FakeMiner:
    def __init__(self):
        self.held = asyncio.Event()
        self.drained = asyncio.Event()
        self.drained.set()
        self.released = asyncio.Event()
        self.holds = 0
        self.watch_armed = asyncio.Event()
        self.watch_lost = asyncio.Event()
        self.watcher = None

    async def client(self, reader, writer):
        try:
            first = await reader.readline()
            if first == b"WATCH\n":
                assert self.held.is_set()
                self.watcher = writer
                self.watch_lost.clear()
                self.watch_armed.set()
                writer.write(b"WATCHING\n")
                await writer.drain()
                await reader.read(1)
                self.watch_armed.clear()
                self.watch_lost.set()
                return
            assert first == b"HOLD\n"
            self.holds += 1
            self.released.clear()
            self.held.set()
            await self.drained.wait()
            writer.write(b"DRAINED\n")
            await writer.drain()
            if await reader.readline() == b"RELEASE\n":
                self.held.clear()
                self.released.set()
                writer.write(b"RELEASED\n")
                await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


class BrokerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        d = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.miners = [FakeMiner(), FakeMiner()]
        paths = [d / f"worker{i}" for i in range(2)]
        self.servers = [await asyncio.start_unix_server(m.client, str(p)) for m, p in zip(self.miners, paths)]
        self.fleet = Fleet(paths, timeout=1, watch_workers=True)
        self.broker_path = d / "broker"
        self.servers.append(await asyncio.start_unix_server(self.fleet.client, str(self.broker_path)))
        self.fleet.start()
        self.sockets = []
        self.ranks = [await self.connect() for _ in range(2)]
        self.control = await self.connect()
        for m in self.miners:
            await asyncio.wait_for(m.held.wait(), 1)
        for i, s in enumerate(self.ranks):
            self.assertEqual(await s.send(f"IDLE {i}"), "OK")
        for m in self.miners:
            await asyncio.wait_for(m.released.wait(), 1)

    async def connect(self):
        s = await Socket.open(str(self.broker_path))
        self.sockets.append(s)
        return s

    async def asyncTearDown(self):
        for s in self.servers:
            s.close()
        for s in self.sockets:
            await s.close()
        await self.fleet.stop()
        for s in self.servers:
            await s.wait_closed()

    async def test_all_ranks_drain_then_all_kernels_finish(self):
        self.miners[1].drained.clear()
        admit = asyncio.create_task(self.control.send("ADMIT batch"))
        await asyncio.wait_for(self.miners[1].held.wait(), 1)
        self.assertFalse(admit.done())
        self.miners[1].drained.set()
        self.assertEqual(await admit, "DRAINED")
        for child in ("child1", "child2"):
            self.assertEqual(await self.control.send(f"DISPATCH {child} batch"), "OK")
        self.assertEqual(await self.control.send("CANCEL batch"), "OK")
        self.assertEqual(len(self.fleet.claims), 2)
        for i, s in enumerate(self.ranks):
            self.assertEqual(await s.send(f"BUSY {i}"), "OK")
            for child in ("child1", "child2"):
                self.assertEqual(await s.send(f"TAKE {child} {i}"), "OK")
        self.assertFalse(self.fleet.claims)
        self.assertEqual(await self.ranks[0].send("IDLE 0"), "OK")
        self.assertTrue(self.fleet.serving)
        self.assertEqual(await self.ranks[1].send("IDLE 1"), "OK")
        await asyncio.wait_for(asyncio.gather(*(m.released.wait() for m in self.miners)), 1)

    async def test_late_idle_and_disconnect(self):
        self.assertEqual(await self.control.send("ADMIT first"), "DRAINED")
        self.assertEqual(await self.control.send("ADMIT second"), "DRAINED")
        self.assertEqual(await self.control.send("DISPATCH first"), "OK")
        # An ingress cancellation after dispatch cannot give away an active GPU.
        self.assertEqual(await self.control.send("CANCEL first"), "OK")
        self.assertIn("first", self.fleet.claims)
        self.assertEqual(await self.control.send("DONE first"), "OK")
        self.assertEqual(await self.ranks[0].send("IDLE 0"), "OK")
        self.assertTrue(self.fleet.serving)
        await self.ranks[1].close()
        for _ in range(100):
            if self.fleet.ranks[1] is None:
                break
            await asyncio.sleep(0.001)
        self.assertIsNone(self.fleet.ranks[1])
        self.assertEqual(await self.control.send("CANCEL second"), "OK")
        self.assertTrue(self.fleet.serving)
        replacement = await self.connect()
        self.assertEqual(await replacement.send("IDLE 1"), "OK")
        await asyncio.wait_for(asyncio.gather(*(m.released.wait() for m in self.miners)), 1)

    async def test_watch_survives_idle_and_shutdown_revokes_idle_grants(self):
        for miner in self.miners:
            self.assertTrue(miner.watch_armed.is_set())
            self.assertFalse(miner.held.is_set())
        self.miners[0].watcher.close()
        await asyncio.wait_for(self.miners[0].watch_lost.wait(), 1)
        await asyncio.wait_for(self.miners[0].watch_armed.wait(), 1)
        await asyncio.wait_for(self.miners[0].released.wait(), 1)
        self.assertEqual([m.holds for m in self.miners], [2, 1])
        await self.fleet.stop()
        await asyncio.wait_for(asyncio.gather(*(m.watch_lost.wait() for m in self.miners)), 1)

    async def test_tokenizer_loss_during_drain_and_after_dispatch(self):
        self.miners[0].drained.clear()
        ingress = await self.connect()
        ingress.writer.write(b"LEASE lost-before-drain\n")
        await ingress.writer.drain()
        await asyncio.wait_for(self.miners[0].held.wait(), 1)
        await ingress.close()
        for _ in range(100):
            if "lost-before-drain" not in self.fleet.claims:
                break
            await asyncio.sleep(0.001)
        self.assertNotIn("lost-before-drain", self.fleet.claims)
        self.miners[0].drained.set()
        ingress = await self.connect()
        self.assertEqual(await ingress.send("LEASE lost-after-dispatch"), "DRAINED")
        self.assertEqual(await self.control.send("DISPATCH child lost-after-dispatch"), "OK")
        await ingress.close()
        self.assertIn("child", self.fleet.claims)
        for i, rank in enumerate(self.ranks):
            self.assertEqual(await rank.send(f"BUSY {i}"), "OK")
            self.assertEqual(await rank.send(f"TAKE child {i}"), "OK")
            self.assertEqual(await rank.send(f"IDLE {i}"), "OK")
        await asyncio.wait_for(asyncio.gather(*(m.released.wait() for m in self.miners)), 1)


class InferenceOnlyBrokerTest(unittest.IsolatedAsyncioTestCase):
    async def test_four_scheduler_ranks_with_no_direct_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'broker.sock'
            fleet = Fleet([], inference_only_ranks=4)
            async with await asyncio.start_unix_server(fleet.client, path):
                sockets = [await Socket.open(path) for _ in range(5)]
                try:
                    control, *ranks = sockets
                    self.assertEqual(fleet.status()['drained'], [])
                    self.assertEqual(await control.send('ADMIT live'), 'DRAINED')
                    self.assertEqual(await control.send('DISPATCH live'), 'OK')
                    for i, rank in enumerate(ranks):
                        self.assertIn('live', fleet.claims)
                        self.assertEqual(await rank.send(f'BUSY {i}'), 'OK')
                        self.assertEqual(await rank.send(f'TAKE live {i}'), 'OK')
                    self.assertNotIn('live', fleet.claims)
                    for i, rank in enumerate(ranks):
                        self.assertTrue(fleet.serving)
                        self.assertEqual(await rank.send(f'IDLE {i}'), 'OK')
                    self.assertFalse(fleet.serving)
                finally:
                    for conn in sockets:
                        await conn.close()


class NativePermitTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_gpu_events_and_canceled_admission(self):
        import fcntl
        import struct

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"permit"
            fleet = Fleet([], inference_only_ranks=4, native_permit=path)
            try:
                with path.open("rb", buffering=0) as first, path.open("rb", buffering=0) as last:
                    fleet.idle[:] = [True]*4
                    fleet.notify()
                    self.assertEqual(struct.unpack("<Q", first.read(8))[0], 1)
                    for worker in (first,last):
                        fcntl.flock(worker, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fleet.claims["one"] = None
                    fleet.notify()
                    self.assertFalse(fleet.all_held)
                    self.assertEqual(struct.unpack("<Q", last.read(8))[0], 2)
                    del fleet.claims["one"]
                    fleet.notify()
                    self.assertEqual(fleet.native.epoch, 2)
                    fleet.claims["two"] = None
                    fleet.notify()
                    drain = asyncio.create_task(fleet.drained("two"))
                    fcntl.flock(first, fcntl.LOCK_UN)
                    await asyncio.sleep(.01)
                    self.assertFalse(drain.done())
                    fcntl.flock(last, fcntl.LOCK_UN)
                    self.assertEqual(await asyncio.wait_for(drain,1), "DRAINED")
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(last,fcntl.LOCK_SH | fcntl.LOCK_NB)
                    del fleet.claims["two"]
                    fleet.idle[3] = False
                    fleet.notify()
                    self.assertTrue(fleet.all_held)
                    fleet.idle[3] = True
                    fleet.notify()
                    self.assertEqual(fleet.native.epoch,3)
                    fcntl.flock(last,fcntl.LOCK_SH | fcntl.LOCK_NB)
            finally:
                await fleet.stop()
