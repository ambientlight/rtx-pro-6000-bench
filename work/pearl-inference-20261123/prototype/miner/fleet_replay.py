#!/usr/bin/env python3
"""Exercise four real sliced CUDA workers via the local broker and synthetic scheduler-rank messages."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

from measure import quantiles

HERE = Path(__file__).resolve().parent
now = time.monotonic_ns


async def command(channel, message, expected):
    reader, writer = channel
    writer.write((message + "\n").encode())
    await writer.drain()
    answer = (await asyncio.wait_for(reader.readline(), 5)).decode().strip()
    if answer != expected:
        raise RuntimeError(f"{message}: {answer!r}, expected {expected!r}")


async def exchange(path):
    reader, writer = await asyncio.open_unix_connection(str(path))
    try:
        writer.write(b"STATUS\n")
        await writer.drain()
        return json.loads(await asyncio.wait_for(reader.readline(), 5))
    finally:
        writer.close()
        await writer.wait_closed()


def smoke(url):
    # Public synthetic prompt. Only elapsed times and token counts are retained.
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "What is 2 plus 2? Answer with one digit."}],
        "temperature": 0,
        "max_tokens": 16,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = now()
    first = None
    meta = {}
    with urlopen(request, timeout=40) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            obj = json.loads(data)
            if "error" in obj:
                raise RuntimeError("generation failed")
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content")):
                    first = now()
            if obj.get("usage"):
                meta = obj["usage"]
    if first is None or not meta.get("completion_tokens"):
        raise RuntimeError(f"generation emitted no stream content or usage, counts={meta}")
    return {
        "first_text_or_reasoning_ms": (first - started) / 1e6,
        "http_complete_ms": (now() - started) / 1e6,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
    }


async def until_resumed(paths, start):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        rows = await asyncio.gather(*(exchange(p) for p in paths))
        if all(
            r["release_ns"] >= start and r["first_slice_end_ns"] >= r["release_ns"] and not r["faulted"]
            for r in rows
        ):
            return rows
        await asyncio.sleep(0.0005)
    raise TimeoutError(f"workers did not complete resumed slices: {rows}")


async def main(args):
    if len(set(args.gpu)) != 4 or len(args.gpu) != 4:
        raise ValueError("provide four distinct GPU identifiers")
    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        broker_path = folder / "api.sock"
        paths = [folder / f"gpu{i}.sock" for i in range(4)]
        workers = []
        broker = None
        clients = []
        samples = []
        raced = 0
        http_result = None
        recovery = None
        try:
            for i, (gpu, path) in enumerate(zip(args.gpu, paths)):
                out = open(folder / f"gpu{i}.out", "w")
                try:
                    workers.append(
                        subprocess.Popen(
                            [
                                str(args.worker.resolve()),
                                "--bench-rounds",
                                str(args.rounds),
                                "--bench-job",
                                str(71 + i),
                                "--bench-nonce",
                                "1001",
                                "--kernel-variant",
                                "v1",
                                "--k",
                                "65536",
                                "--control-socket",
                                str(path),
                                "--slice-tiles",
                                str(args.slice_tiles),
                                "--device",
                                "0",
                                "--power-limit",
                                str(args.power_limit),
                                "--max-temp",
                                str(args.max_temp),
                            ]
                            + (["--require-watch"] if args.watch_workers else []),
                            stdout=out,
                            stderr=subprocess.STDOUT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "RAYON_NUM_THREADS": "2"},
                        )
                    )
                finally:
                    out.close()
            deadline = time.monotonic() + 30
            while not all(p.exists() for p in paths):
                if time.monotonic() > deadline or any(w.poll() is not None for w in workers):
                    raise RuntimeError("worker startup failed")
                await asyncio.sleep(0.02)
            argv = [sys.executable, str(HERE.parent / "broker.py"), "--socket", str(broker_path)]
            if args.watch_workers:
                argv += ["--watch-workers"]
            for p in paths:
                argv += ["--worker", str(p)]
            broker = subprocess.Popen(argv, stdout=subprocess.DEVNULL)
            for _ in range(200):
                if broker_path.exists():
                    break
                await asyncio.sleep(0.02)
            for _ in range(5):
                clients.append(await asyncio.open_unix_connection(str(broker_path)))
            ranks = clients[:4]
            api = clients[4]

            async def mode(state):
                await asyncio.gather(*(command(ch, f"{state} {i}", "OK") for i, ch in enumerate(ranks)))

            async def dispatch(rid):
                await command(api, "DISPATCH " + rid, "OK")
                await mode("BUSY")
                await asyncio.gather(
                    *(command(ch, f"TAKE {rid} {i}", "OK") for i, ch in enumerate(ranks))
                )

            async def assert_held():
                rows = await asyncio.gather(*(exchange(p) for p in paths))
                assert all(r["holds"] >= 1 and not r["gpu_busy"] and not r["faulted"] for r in rows)
                return [r["completed_slices"] for r in rows]

            # Establish an initially drained fleet before the warm-idle measurement.
            await mode("BUSY")
            await command(api, "ADMIT warmup", "DRAINED")
            await dispatch("warmup")
            warm = now()
            await mode("IDLE")
            await until_resumed(paths, warm)
            for i in range(args.samples):
                await asyncio.sleep(rng.uniform(0, 0.013))
                rid = f"req-{i}"
                arrival = now()
                await command(api, "ADMIT " + rid, "DRAINED")
                drained = now()
                before = await assert_held()
                await dispatch(rid)
                if i % 23 == 0:
                    # Overlapping arrival while other foreground work is still active.
                    await command(api, "ADMIT nested-" + rid, "DRAINED")
                    await dispatch("nested-" + rid)
                if i == 0 and args.smoke_url:
                    http_result = await asyncio.to_thread(smoke, args.smoke_url)
                await asyncio.sleep(rng.uniform(0, 0.002))
                assert await assert_held() == before
                if i % 11 == 0:
                    # Race a new admission against the fourth rank's IDLE.
                    await asyncio.gather(*(command(ranks[j], f"IDLE {j}", "OK") for j in range(3)))
                    await asyncio.gather(
                        command(ranks[3], "IDLE 3", "OK"), command(api, "ADMIT raced-" + rid, "DRAINED")
                    )
                    await assert_held()
                    await dispatch("raced-" + rid)
                    raced += 1
                end = now()  # before issuing the four simulated rank IDLE events, conservative.
                await mode("IDLE")
                rows = await until_resumed(paths, end)
                assert all(
                    r["first_slice_begin_ns"] <= r["first_gpu_seen_ns"] <= r["first_slice_end_ns"]
                    for r in rows
                )
                samples.append(
                    [
                        (drained - arrival) / 1e6,
                        (max(r["first_slice_begin_ns"] for r in rows) - end) / 1e6,
                        (max(r["first_gpu_seen_ns"] for r in rows) - end) / 1e6,
                        (max(r["first_slice_end_ns"] for r in rows) - end) / 1e6,
                    ]
                )
            if args.restart_broker:
                # At idle each hold has been cleanly released. Kill only our fixture's broker.
                broker.kill()
                await asyncio.to_thread(broker.wait, timeout=5)
                for _, writer in clients:
                    writer.close()
                clients.clear()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    rows = await asyncio.gather(*(exchange(p) for p in paths))
                    if all(not r["watched"] and not r["gpu_busy"] and not r["faulted"] for r in rows):
                        break
                    await asyncio.sleep(0.001)
                else:
                    raise RuntimeError("workers did not park after the broker died")
                counts = [r["completed_slices"] for r in rows]
                await asyncio.sleep(0.150)
                rows = await asyncio.gather(*(exchange(p) for p in paths))
                assert [r["completed_slices"] for r in rows] == counts
                broker_path = folder / "replacement.sock"
                argv[argv.index("--socket") + 1] = str(broker_path)
                broker = subprocess.Popen(argv, stdout=subprocess.DEVNULL)
                for _ in range(250):
                    if broker_path.exists():
                        break
                    await asyncio.sleep(0.01)
                for _ in range(5):
                    clients.append(await asyncio.open_unix_connection(str(broker_path)))
                ranks, api = clients[:4], clients[4]
                await mode("BUSY")
                await command(api, "ADMIT recovered", "DRAINED")
                await dispatch("recovered")
                started = now()
                await mode("IDLE")
                rows = await until_resumed(paths, started)
                assert all(r["watched"] and r["require_watch"] and not r["faulted"] for r in rows)
                recovery = {
                    "idle_broker_killed": True,
                    "parked_ms_checked": 150,
                    "frozen_slices": counts,
                    "resumed_slices": [r["completed_slices"] for r in rows],
                    "all_watches_rearmed": True,
                }
            # Remain connected while workers finish and verify their own sample proofs.
            await asyncio.gather(*(asyncio.to_thread(w.wait, timeout=95) for w in workers))
            if any(w.returncode for w in workers):
                raise RuntimeError("worker exited with errors")
            stats = [
                json.loads((folder / f"gpu{i}.out").read_text().splitlines()[-1]) for i in range(4)
            ]
            cols = zip(*samples)
            return {
                "gpu": args.gpu,
                "watch_workers": args.watch_workers,
                "samples": args.samples,
                "raced_resumes": raced,
                "seed": args.seed,
                "admit_to_four_drained_ms": quantiles(next(cols)),
                "rank_idle_to_last_first_slice_host_begin_ms": quantiles(next(cols)),
                "rank_idle_to_last_gpu_start_observed_ms": quantiles(next(cols)),
                "rank_idle_to_last_first_slice_completed_ms": quantiles(next(cols)),
                "stock_http_under_external_hold": http_result,
                "broker_recovery": recovery,
                "workers": stats,
            }
        except BaseException:
            for i in range(len(workers)):
                print(f"worker {i}: " + (folder / f"gpu{i}.out").read_text()[-1200:], file=sys.stderr)
            raise
        finally:
            for _, writer in clients:
                writer.close()
            if broker and broker.poll() is None:
                broker.terminate()
            for w in workers:
                if w.poll() is None:
                    w.terminate()
            for p in workers + ([broker] if broker else []):
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker", type=Path)
    p.add_argument("--gpu", required=True, action="append")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=523)
    p.add_argument("--rounds", type=int, default=64)
    p.add_argument("--slice-tiles", type=int, default=188)
    p.add_argument(
        "--restart-broker",
        action="store_true",
        help="kill the supervised idle fixture broker and recover using a fresh one",
    )
    p.add_argument(
        "--watch-workers",
        action="store_true",
        help="start parked workers and require a live broker supervisory connection",
    )
    p.add_argument("--power-limit", type=int, default=280)
    p.add_argument("--max-temp", type=int, default=92)
    p.add_argument(
        "--smoke-url",
        help="optional stock local SGLang origin; issues one short synthetic request inside the held window",
    )
    args = p.parse_args()
    if args.restart_broker and not args.watch_workers:
        p.error("--restart-broker requires --watch-workers")
    print(json.dumps(asyncio.run(main(args)), indent=2))
