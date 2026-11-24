"""Local request and TP-rank ownership broker for the Pearl/SGLang prototype."""
import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

ID = re.compile(r"[A-Za-z0-9._:-]{1,192}\Z")


class Fleet:
    def __init__(self, workers, timeout=5, watch_workers=False, inference_only_ranks=None, timeline=None):
        self.timeout = timeout
        self.watch_workers = watch_workers
        self.claims = {}  # None before dispatch, then the set of ranks that took it.
        rank_count = len(workers) if inference_only_ranks is None else inference_only_ranks
        self.ranks = [None] * rank_count
        self.idle = [False] * rank_count
        self.changed = asyncio.Event()
        self.workers = [Worker(self, path) for path in workers]
        self.tasks = []
        self.timeline = timeline
        self.previous_state = None
        self.record_state()

    def record(self, event, **data):
        if self.timeline is not None:
            try:
                self.timeline.write(json.dumps(dict(event=event, monotonic_ns=time.monotonic_ns(), **data)) + "\n")
            except OSError:
                # A measurement filesystem failure must never delay a reservation.
                print("Broker timeline stopped after a write error", file=sys.stderr, flush=True)
                self.timeline = None

    def record_state(self):
        if self.timeline is None:
            return
        current = self.serving, sum(w.held for w in self.workers), sum(w.watched for w in self.workers)
        if current != self.previous_state:
            self.record("state", serving=current[0], held=current[1], watched=current[2])
            self.previous_state = current

    def notify(self):
        self.record_state()
        event, self.changed = self.changed, asyncio.Event()
        event.set()

    @property
    def serving(self):
        return bool(self.claims) or not all(self.idle)

    def start(self):
        self.tasks = [asyncio.create_task(worker.run()) for worker in self.workers]
        if self.watch_workers:
            self.tasks.extend(asyncio.create_task(worker.watch()) for worker in self.workers)

    async def stop(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def drained(self, rid):
        start = time.monotonic_ns()
        required_drain = not all(w.held for w in self.workers)
        async with asyncio.timeout(self.timeout):
            while rid in self.claims and not all(w.held for w in self.workers):
                await self.changed.wait()
        if rid not in self.claims:
            raise ValueError("reservation was canceled")
        self.record("admitted", elapsed_ns=time.monotonic_ns() - start, required_drain=required_drain)
        return "DRAINED"

    def status(self):
        return dict(serving=self.serving, claims=len(self.claims), idle=self.idle,
                    drained=[w.held for w in self.workers], errors=[w.error for w in self.workers],
                    watch_required=self.watch_workers, watched=[w.watched for w in self.workers],
                    watch_errors=[w.watch_error for w in self.workers])

    async def client(self, reader, writer):
        rank = None
        lease = None
        try:
            while line := await reader.readline():
                args = line.decode("ascii").split()
                if args == ["STATUS"]:
                    answer = json.dumps(self.status(), separators=(",", ":"))
                elif len(args) == 2 and args[0] in ("BUSY", "IDLE"):
                    index = int(args[1])
                    if not 0 <= index < len(self.ranks) or (rank is not None and index != rank):
                        raise ValueError("invalid rank")
                    if self.ranks[index] not in (None, writer):
                        raise ValueError("rank already connected")
                    rank = index
                    self.ranks[rank] = writer
                    self.idle[rank] = args[0] == "IDLE"
                    self.notify()
                    answer = "OK"
                elif len(args) == 3 and args[0] == "DISPATCH":
                    child, parent = args[1:]
                    if not ID.fullmatch(child) or child == parent or parent not in self.claims:
                        raise ValueError("child dispatch requires a distinct id and admitted parent")
                    if self.claims[parent] is not None:
                        raise ValueError("parent is no longer an admission claim")
                    if child in self.claims:
                        raise ValueError("duplicate child id")
                    self.claims[child] = set()
                    answer = "OK"
                elif len(args) == 3 and args[0] == "TAKE":
                    rid = args[1]
                    if rank is None or rank != int(args[2]) or self.idle[rank]:
                        raise ValueError("TAKE requires an active rank")
                    if rid in self.claims:
                        taken = self.claims[rid]
                        if taken is None:
                            raise ValueError("TAKE before DISPATCH")
                        taken.add(rank)
                        if len(taken) == len(self.ranks):
                            del self.claims[rid]
                            self.notify()
                    answer = "OK"
                elif len(args) == 2 and ID.fullmatch(args[1]):
                    cmd, rid = args
                    if cmd == "LEASE":
                        if lease is not None or rid in self.claims:
                            raise ValueError("lease id already in use")
                        self.claims[rid] = None
                        lease = rid
                        self.notify()
                        drained = asyncio.create_task(self.drained(rid))
                        disconnected = asyncio.create_task(reader.readline())
                        try:
                            await asyncio.wait((drained, disconnected),
                                               return_when=asyncio.FIRST_COMPLETED)
                            if disconnected.done():
                                data = disconnected.result()
                                if data not in (b"", ("CANCEL " + rid + "\n").encode()):
                                    raise ValueError("wait for DRAINED before dispatch")
                                return
                            answer = drained.result()
                        finally:
                            for task in (disconnected, drained):
                                task.cancel()
                            await asyncio.gather(disconnected, drained, return_exceptions=True)
                    elif cmd == "ADMIT":
                        self.claims.setdefault(rid, None)
                        self.notify()
                        try:
                            answer = await self.drained(rid)
                        except TimeoutError:
                            if self.claims.get(rid) is None:
                                self.claims.pop(rid, None)
                                self.notify()
                            raise
                    elif cmd == "DISPATCH":
                        if rid not in self.claims:
                            raise ValueError("dispatch without admission")
                        if self.claims[rid] is None:
                            self.claims[rid] = set()
                        answer = "OK"
                    elif cmd in ("CANCEL", "DONE"):
                        if cmd == "DONE" or self.claims.get(rid) is None:
                            self.claims.pop(rid, None)
                        self.notify()
                        answer = "OK"
                    else:
                        raise ValueError("unknown command")
                else:
                    raise ValueError("invalid command")
                writer.write((answer + "\n").encode("ascii"))
                await writer.drain()
        except (ValueError, TimeoutError) as exc:
            writer.write(("ERROR " + (str(exc) or "drain timed out") + "\n").encode("ascii", "replace"))
            with contextlib.suppress(ConnectionError):
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if lease is not None and self.claims.get(lease) is None:
                self.claims.pop(lease, None)
                self.notify()
            if rank is not None and self.ranks[rank] is writer:
                self.ranks[rank] = None
                self.idle[rank] = False
                self.notify()
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


class Worker:
    def __init__(self, fleet, path):
        self.fleet = fleet
        self.path = path
        self.held = False
        self.error = None
        self.watched = False
        self.watch_error = None

    @property
    def needs_hold(self):
        return self.fleet.serving or (self.fleet.watch_workers and not self.watched)

    async def watch(self):
        # A managed miner starts parked. Arm it under HOLD before permitting idle work.
        while True:
            while not self.held:
                await self.fleet.changed.wait()
            writer = None
            try:
                async with asyncio.timeout(self.fleet.timeout):
                    reader, writer = await asyncio.open_unix_connection(str(self.path))
                    writer.write(b"WATCH\n")
                    await writer.drain()
                    if await reader.readline() != b"WATCHING\n":
                        raise ConnectionError("worker did not arm its watchdog")
                self.watched, self.watch_error = True, None
                self.fleet.notify()
                await reader.read(1)
                raise ConnectionError("worker watchdog disconnected")
            except (OSError, TimeoutError) as exc:
                self.watch_error = str(exc) or "worker watchdog timed out"
            finally:
                self.watched = False
                self.fleet.notify()
                if writer is not None:
                    writer.close()
                    with contextlib.suppress(ConnectionError):
                        await writer.wait_closed()
            await asyncio.sleep(0.05)

    async def run(self):
        while True:
            while not self.needs_hold:
                await self.fleet.changed.wait()
            writer = monitor = None
            try:
                async with asyncio.timeout(self.fleet.timeout):
                    reader, writer = await asyncio.open_unix_connection(str(self.path))
                    writer.write(b"HOLD\n")
                    await writer.drain()
                    if await reader.readline() != b"DRAINED\n":
                        raise ConnectionError("worker did not confirm drain")
                self.held, self.error = True, None
                self.fleet.notify()
                # Also detect worker loss while the fleet stays continuously busy.
                monitor = asyncio.create_task(reader.read(1))
                while self.needs_hold:
                    change = asyncio.create_task(self.fleet.changed.wait())
                    try:
                        await asyncio.wait((change, monitor), return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        change.cancel()
                    if monitor.done():
                        raise ConnectionError("worker disconnected while held")
                self.held = False
                self.fleet.notify()
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
                async with asyncio.timeout(self.fleet.timeout):
                    writer.write(b"RELEASE\n")
                    await writer.drain()
                    if await reader.readline() != b"RELEASED\n":
                        raise ConnectionError("worker did not confirm release")
            except (OSError, TimeoutError) as exc:
                self.held = False
                self.error = str(exc) or "worker timed out"
                self.fleet.notify()
                await asyncio.sleep(0.05)  # Reconnect a failed control transport.
            finally:
                self.held = False
                if monitor is not None:
                    monitor.cancel()
                    await asyncio.gather(monitor, return_exceptions=True)
                if writer is not None:
                    writer.close()
                    with contextlib.suppress(ConnectionError):
                        await writer.wait_closed()


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--socket", required=True, type=Path)
    p.add_argument("--timeline", type=Path, help="fresh aggregate-only transition log")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", action="append", type=Path, default=[])
    mode.add_argument("--inference-only-ranks", type=int, choices=(4,),
                      help="TP4 rank fencing with no standalone mining workers")
    p.add_argument("--timeout", type=float, default=5)
    p.add_argument("--watch-workers", action="store_true",
                   help="arm the lifetime watchdog on workers started with --require-watch")
    args = p.parse_args()
    parent = args.socket.parent.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
        p.error("broker socket directory must be private to the current user")
    if len(set(args.worker)) != len(args.worker) or args.timeout <= 0:
        p.error("use distinct worker endpoints and a positive timeout")
    if args.socket.exists() or args.socket.is_symlink():
        p.error("broker socket path is already in use")
    if args.inference_only_ranks and args.watch_workers:
        p.error("--watch-workers needs standalone workers")
    timeline = args.timeline.open("x", buffering=1) if args.timeline else None
    fleet = Fleet(args.worker, args.timeout, watch_workers=args.watch_workers,
                  inference_only_ranks=args.inference_only_ranks, timeline=timeline)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    server = await asyncio.start_unix_server(fleet.client, str(args.socket), limit=512)
    os.chmod(args.socket, 0o600)
    fleet.start()
    try:
        async with server:
            await stop.wait()
    finally:
        await fleet.stop()
        args.socket.unlink(missing_ok=True)
        if timeline is not None:
            timeline.close()


if __name__ == "__main__":
    asyncio.run(main())
