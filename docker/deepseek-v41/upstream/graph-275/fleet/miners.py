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
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
from direct_peer import Peer

CONTROL = Path(os.environ["XDG_RUNTIME_DIR"]) / "dsv41-pearl-graph275"
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
    began = time.monotonic_ns()
    broker, *workers = await asyncio.gather(*(call(p) for p in (BROKER, *WORKERS)))
    return {"began_ns": began, "monotonic_ns": time.monotonic_ns(), "broker": broker, "workers": workers}


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
        # This is the startup-only controlled idle calibration. Real scheduler
        # sockets connect only after this fixture relinquishes the four ranks.
        await asyncio.sleep(1)
        calibrate_from = await snapshot()
        await asyncio.sleep(6)
        calibrate_to = await snapshot()
        if not all(healthy_active(s) for s in (calibrate_from, calibrate_to)):
            raise RuntimeError("calibration was interrupted before all four rank leases stayed idle")
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
    return {"fresh_parked":initial,"all_mining":mining,"calibration":calibration(calibrate_from,calibrate_to),"busy_signal_ns":sent,
            "fleet_held":held,"checked_stable":stable,"parked_with_no_test_clients":final}


def tiles(slices):
    # 139 slices per full NONCE, with 168 tiles in its final slice.
    rounds, partial = divmod(slices, 139)
    return rounds * 26112 + partial * 188

def healthy_active(state):
    broker = state["broker"]
    return (len(state["workers"])==4 and broker["idle"] == [True]*4
            and not broker["serving"] and not broker["claims"]
            and all(w["watched"] and not w["faulted"] and not w["holds"] for w in state["workers"]))

def calibration(start, end):
    seconds = (end["monotonic_ns"]-start["monotonic_ns"])/1e9
    counts=[tiles(y["completed_slices"])-tiles(x["completed_slices"]) for x,y in zip(start["workers"],end["workers"])]
    if seconds<=0 or min(counts)<=0 or any(x["release_ns"] != y["release_ns"] for x,y in zip(start["workers"],end["workers"])):
        raise RuntimeError("incomplete or interrupted idle calibration")
    rates=[c*160*256*65536/seconds/1e12 for c in counts]
    return dict(seconds=seconds, tiles_by_gpu=counts, tmac_s_by_gpu=rates, total_tmac_s=sum(rates),
                start_ns=start["monotonic_ns"],end_ns=end["monotonic_ns"],
                start_cursors=[w["completed_slices"] for w in start["workers"]],
                end_cursors=[w["completed_slices"] for w in end["workers"]])

def notify_ready():
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.sendto(b"READY=1\nSTATUS=4 K780 workers warm, watched and held until real TP4 ranks request idle", address.replace("@","\0",1) if address.startswith("@") else address)


async def monitor(children, log, peer):
    previous = None
    last = 0
    while True:
        if peer.failures:
            raise RuntimeError(f"local verifier failed: {peer.failures[-1]}")
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


async def guarded_warm(endpoint):
    if endpoint == "offline":
        # The caller explicitly stopped the incumbent. Check that inference
        # cannot enter via its loopback socket while the mock ranks own TP4.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8000)) == 0:
                raise RuntimeError("incumbent still listening during exclusive calibration")
        return await warm()
    sys.path.insert(0,"/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/scripts")
    from dsv41_load import activity_metrics
    def counters():
        with urllib.request.urlopen(endpoint.rstrip("/")+"/metrics", timeout=1.0) as response:
            return activity_metrics(response.read().decode())
    baseline=await asyncio.to_thread(counters)
    if baseline[0]:
        raise RuntimeError("incumbent is serving a request; calibration cancelled")
    async def watch():
        while True:
            if await asyncio.to_thread(counters) != baseline:
                raise RuntimeError("incumbent request arrived during idle-only calibration")
            await asyncio.sleep(0.075)
    trial, guard=asyncio.create_task(warm()), asyncio.create_task(watch())
    try:
        await asyncio.wait((trial, guard), return_when=asyncio.FIRST_COMPLETED)
        if guard.done():
            guard.result()
        return trial.result()
    finally:
        for task in (trial, guard):
            task.cancel()
        await asyncio.gather(trial, guard, return_exceptions=True)


async def serve(endpoint):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = HERE / f"run-{stamp}-{os.getpid()}"
    run.mkdir(mode=0o700)
    (HERE/"current-run.txt").write_text(str(run)+"\n")
    cards = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free,power.limit","--format=csv,noheader,nounits"],text=True)
    cards = [line.split(", ") for line in cards.splitlines()]
    if len(cards)!=4 or [int(row[0]) for row in cards] != list(range(4)) or any(float(row[3])!=275 for row in cards):
        raise RuntimeError(f"expected four 275 W devices: {cards}")
    if subprocess.run(["systemctl","is-active","--quiet","pearl-f2pool.service"]).returncode == 0:
        raise RuntimeError("pre-existing paid fleet is active; leave it untouched")
    if min(int(row[2]) for row in cards) < 850:
        raise RuntimeError(f"insufficient free GPU memory to initialize offline workers: {cards}")
    assert CONTROL.stat().st_uid == os.getuid() and CONTROL.stat().st_mode & 0o077 == 0
    (run/"cards.json").write_text(json.dumps(cards)+"\n")
    async with Peer(run/"peer") as peer:
        await workers(run, cards, peer, endpoint)

async def workers(run, cards, peer, endpoint):
    with ExitStack() as handles:
        processes = []
        try:
            for i,(_,uuid,_,_) in enumerate(cards):
                workdir=run/f"gpu{i}"
                workdir.mkdir(mode=0o700)
                log = handles.enter_context((run/f"gpu{i}.log").open("wt"))
                p = await asyncio.create_subprocess_exec(str(BINARY),"--pool",f"127.0.0.1:{peer.port}",
                    "--wallet","local-public-fixed-job","--worker",f"graph275-{i}","--variant","780",
                    "--log",str(workdir/"events.jsonl"),"--json","--kernel-variant","v1","--k","65536",
                    "--control-socket",str(WORKERS[i]),"--slice-tiles","188","--device","0",
                    "--power-limit","275","--max-temp","92","--require-watch",
                    env={**os.environ,"CUDA_VISIBLE_DEVICES":uuid,"RAYON_NUM_THREADS":"2"},
                    cwd=workdir,stdout=log,stderr=subprocess.STDOUT)
                processes.append(p)
            args = [word for path in WORKERS for word in ("--worker",str(path))]
            log = handles.enter_context((run/"broker.log").open("wt"))
            broker = await asyncio.create_subprocess_exec(sys.executable,str(SOURCE/"broker.py"),
                    "--socket",str(BROKER),"--watch-workers",*args,stdout=log,stderr=subprocess.STDOUT)
            processes.append(broker)
            receipt = await guarded_warm(endpoint)
            if peer.failures or peer.authorized != set(range(4)):
                raise RuntimeError(f"local direct peer incomplete after calibration: {peer.failures}")
            receipt["local_verified_by_rank_at_warm"] = list(peer.shares)
            receipt["target_hex"] = peer.job["target"]
            (run/"prewarm.json").write_text(json.dumps(receipt,indent=2)+"\n")
            notify_ready()
            log = handles.enter_context((run/"handoff.jsonl").open("wt",buffering=1))
            await monitor(processes,log,peer)
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
    elif len(sys.argv) == 3 and sys.argv[1] == "serve":
        asyncio.run(serve(sys.argv[2]))
    elif sys.argv[1:] in (["park"], ["resume"]):
        verb = "ADMIT" if sys.argv[1] == "park" else "DONE"
        reply=asyncio.run(call(BROKER, verb+" graph275-between-windows"))
        if reply != ("DRAINED" if verb=="ADMIT" else "OK"):
            raise SystemExit(reply)
        print(reply)
    else:
        raise SystemExit("use serve or status")
