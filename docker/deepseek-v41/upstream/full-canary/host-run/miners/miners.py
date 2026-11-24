#!/usr/bin/env python3
"""Serve four watched offline K780 workers under a private systemd user unit."""
import asyncio
from contextlib import ExitStack, suppress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CONTROL = Path(os.environ["XDG_RUNTIME_DIR"]) / "dsv41-sm120-full-miners"
BROKER = CONTROL / "broker.sock"
WORKERS = tuple(CONTROL / f"gpu{i}.sock" for i in range(4))
SOURCE = Path("/mnt/hot/ambientlight/sglang-sm120-runtime/deployment-source/work/pearl-inference-20261123/prototype")
BINARY = Path("/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-sglang-prototype/target/cargo/release/bpm")


async def call(path, command="STATUS"):
    async with asyncio.timeout(8):
        reader, writer = await asyncio.open_unix_connection(str(path))
        try:
            writer.write((command+"\n").encode())
            await writer.drain()
            data = (await reader.readline()).decode().strip()
            return json.loads(data) if command == "STATUS" else data
        finally:
            writer.close()
            await writer.wait_closed()


async def snapshot():
    broker, *workers = await asyncio.gather(*(call(p) for p in (BROKER, *WORKERS)))
    return {"monotonic_ns": time.monotonic_ns(), "broker": broker, "workers": workers}


async def wait_for(predicate, timeout=60):
    async with asyncio.timeout(timeout):
        while True:
            try:
                state = await snapshot()
                if predicate(state):
                    return state
            except (OSError, ValueError):
                pass
            await asyncio.sleep(0.012)


async def mode(channel, state, rank):
    reader, writer = channel
    writer.write(f"{state} {rank}\n".encode())
    await writer.drain()
    if await asyncio.wait_for(reader.readline(), 8) != b"OK\n":
        raise RuntimeError(f"broker declined {state} on rank {rank}")


async def warm():
    initial = await wait_for(lambda s: all(s["broker"]["drained"]) and all(s["broker"]["watched"]))
    channels = await asyncio.gather(*(asyncio.open_unix_connection(str(BROKER)) for _ in WORKERS))
    try:
        await asyncio.gather(*(mode(ch, "IDLE", i) for i,ch in enumerate(channels)))
        mining = await wait_for(lambda s: all(w["completed_slices"] >= 3 and not w["faulted"] for w in s["workers"]))
        sent = time.monotonic_ns()
        await asyncio.gather(*(mode(ch, "BUSY", i) for i,ch in enumerate(channels)))
        held = await wait_for(lambda s: all(s["broker"]["drained"]) and all(w["holds"] and not w["gpu_busy"] for w in s["workers"]))
        await asyncio.sleep(0.08)
        stable = await snapshot()
        assert [w["completed_slices"] for w in held["workers"]] == [w["completed_slices"] for w in stable["workers"]]
    finally:
        for _, writer in channels:
            writer.close()
        await asyncio.gather(*(w.wait_closed() for _,w in channels))
    final = await wait_for(lambda s: all(s["broker"]["drained"]) and s["broker"]["idle"] == [False]*4)
    return {"fresh_parked":initial,"all_mining":mining,"busy_signal_ns":sent,
            "fleet_held":held,"checked_stable":stable,"parked_with_no_test_clients":final}


def notify_ready():
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.sendto(b"READY=1\nSTATUS=4 K780 workers warm, watched and held until real TP4 ranks request idle", address.replace("@","\0",1) if address.startswith("@") else address)


async def monitor(children, log):
    previous = None
    last = 0
    while True:
        if any(p.returncode is not None for p in children):
            raise RuntimeError(f"fleet process exited, codes={[p.returncode for p in children]}")
        now = time.monotonic()
        data = await snapshot()
        status = data["broker"]
        key = (status["claims"], tuple(status["idle"]), tuple(status["drained"]),
               tuple(status["watched"]), tuple((w["release_ns"],w["first_gpu_seen_ns"] > 0,w["faulted"]) for w in data["workers"]))
        if key != previous or now - last >= 10:
            log.write(json.dumps(data, separators=(",",":"))+"\n")
            previous, last = key, now
        await asyncio.sleep(0.025 if any(status["idle"]) or status["claims"] else 0.10)


async def serve():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = HERE / f"run-{stamp}-{os.getpid()}"
    run.mkdir(mode=0o700)
    (HERE/"current-run.txt").write_text(str(run)+"\n")
    cards = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"],text=True)
    cards = [line.split(", ") for line in cards.splitlines()]
    assert [int(i) for i,_,_ in cards] == list(range(4))
    if min(int(free) for _,_,free in cards) < 850:
        raise RuntimeError(f"insufficient free GPU memory to initialize offline workers: {cards}")
    assert CONTROL.stat().st_uid == os.getuid() and CONTROL.stat().st_mode & 0o077 == 0
    (run/"cards.json").write_text(json.dumps(cards)+"\n")
    with ExitStack() as handles:
        processes = []
        try:
            for i,(_,uuid,_) in enumerate(cards):
                log = handles.enter_context((run/f"gpu{i}.log").open("wt"))
                p = await asyncio.create_subprocess_exec(str(BINARY),"--bench-rounds","10000",
                    "--bench-job",str(71+i),"--bench-nonce","1001","--kernel-variant","v1","--k","65536",
                    "--control-socket",str(WORKERS[i]),"--slice-tiles","188","--device","0",
                    "--power-limit","280","--max-temp","92","--require-watch",
                    env={**os.environ,"CUDA_VISIBLE_DEVICES":uuid,"RAYON_NUM_THREADS":"2"},
                    stdout=log,stderr=subprocess.STDOUT)
                processes.append(p)
            args = [word for path in WORKERS for word in ("--worker",str(path))]
            log = handles.enter_context((run/"broker.log").open("wt"))
            broker = await asyncio.create_subprocess_exec(sys.executable,str(SOURCE/"broker.py"),
                    "--socket",str(BROKER),"--watch-workers",*args,stdout=log,stderr=subprocess.STDOUT)
            processes.append(broker)
            receipt = await warm()
            (run/"prewarm.json").write_text(json.dumps(receipt,indent=2)+"\n")
            notify_ready()
            log = handles.enter_context((run/"handoff.jsonl").open("wt",buffering=1))
            await monitor(processes,log)
        finally:
            for p in reversed(processes):
                if p.returncode is None:
                    with suppress(ProcessLookupError):
                        p.terminate()
            try:
                async with asyncio.timeout(2):
                    await asyncio.gather(*(p.wait() for p in processes))
            except TimeoutError:
                for process in processes:
                    if process.returncode is None:
                        with suppress(ProcessLookupError):
                            process.kill()
                await asyncio.gather(*(p.wait() for p in processes))


if __name__ == "__main__":
    os.umask(0o077)
    if sys.argv[1:] == ["status"]:
        print(json.dumps(asyncio.run(snapshot()),indent=2))
    elif sys.argv[1:] == ["serve"]:
        asyncio.run(serve())
    else:
        raise SystemExit("use serve or status")
