"""Measure the original four paid K780 workers during an uninterrupted idle lease."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from paid_idle import BINARY_SHA, rpc, save


async def capture(control, state, out, warm, seconds):
    deadline = time.monotonic() + 110
    while not (state / "status.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Waiting for the four authenticated standalone workers")
        await asyncio.sleep(.2)
    channels = []
    samples = []
    try:
        for rank in range(4):
            reader, writer = await asyncio.open_unix_connection(control / "broker.sock")
            channels.append(writer)
            writer.write(f"IDLE {rank}\n".encode())
            await writer.drain()
            if await reader.readline() != b"OK\n":
                raise RuntimeError("Could not release all four standalone miners")
        start = time.time()
        while time.time() < start + warm + seconds + 32:
            fleet = json.loads((state / "status.json").read_text())
            if time.time() - fleet["utc_epoch"] > 5 or not all(r["authorized"] for r in fleet["ranks"]):
                raise RuntimeError("Stale or disconnected standalone fleet")
            stats = await asyncio.gather(*(rpc(control / f"gpu{i}.sock") for i in range(4)))
            if any(s["holds"] or s["faulted"] or not s["watched"] for s in stats):
                raise RuntimeError("A standalone worker lost the idle lease")
            samples.append(dict(time=time.time(), slices=[s["completed_slices"] for s in stats],
                accepted=[r["accepted"] for r in fleet["ranks"]], rejected=[r["rejected"] for r in fleet["ranks"]]))
            await asyncio.sleep(1)
        # The pinned miner records cumulative actual MACs every 30s, excluding paused
        # seconds from its displayed rate. Snapshots are timestamped independently by rank.
        ranks = []
        for i in range(4):
            stats = []
            for line in (control / f"rank{i}/events.jsonl").read_text().splitlines():
                e = json.loads(line)
                if e["event"] == "stats" and e["time"] >= start + warm:
                    stats.append(dict(time=e["time"], elapsed=e["seconds"],
                                      tmac=(e["seconds"]-e["paused_seconds"])*e["TMAC_wall_s"]))
            if len(stats) < 2 or stats[-1]["time"]-stats[0]["time"] < seconds-32:
                raise RuntimeError(f"Insufficient exact K780 counters on rank {i}")
            ranks.append(stats)
        save(out, dict(binary_sha256=BINARY_SHA, started=start, warm_seconds=warm, target_seconds=seconds,
                        counters_by_rank=ranks, samples=samples))
    finally:
        for w in channels:
            w.close()
        await asyncio.gather(*(w.wait_closed() for w in channels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "state", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--warm", type=int, default=60)
    parser.add_argument("--seconds", type=int, default=180)
    args = parser.parse_args()
    asyncio.run(capture(**vars(args)))
