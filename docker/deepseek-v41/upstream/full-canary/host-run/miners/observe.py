#!/usr/bin/env python3
"""Record four-worker ownership transitions while the canary sends real HTTP requests."""
import argparse
import asyncio
import json
import signal
import time

from miners import BROKER, WORKERS, snapshot


async def observe(output):
    stop = asyncio.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(signum,stop.set)
    # A persistent socket avoids connect/thread scheduling on the timing path.
    br,bw = await asyncio.open_unix_connection(str(BROKER))
    last_key = None
    pending = False
    n = 0
    max_gap = 0
    prev = time.monotonic_ns()
    started = prev
    last_held = 0
    try:
        with open(output,"x",buffering=1) as log:
            def write(row):
                log.write(json.dumps(row,separators=(",",":"))+"\n")
            write({"type":"start","monotonic_ns":started,"state":await snapshot()})
            while not stop.is_set():
                bw.write(b"STATUS\n")
                await bw.drain()
                state=json.loads(await asyncio.wait_for(br.readline(),3))
                stamp=time.monotonic_ns()
                max_gap=max(max_gap,stamp-prev)
                n+=1
                prev=stamp
                key=(state["claims"],tuple(state["idle"]),tuple(state["drained"]),tuple(state["watched"]))
                if key!=last_key:
                    write({"type":"broker","monotonic_ns":stamp,"state":state})
                    pending = pending or all(state["idle"])
                    if all(state["drained"]) or all(state["idle"]):
                        write({"type":"workers","reason":"edge","state":await snapshot()})
                    last_key=key
                if all(state["drained"]) and state["serving"] and stamp - last_held > 25_000_000:
                    write({"type":"workers","reason":"hold-cursor","state":await snapshot()})
                    last_held = stamp
                if pending and not state["serving"]:
                    sample=await snapshot()
                    if all(w["release_ns"] and w["first_gpu_seen_ns"] >= w["release_ns"] and not w["holds"] for w in sample["workers"]):
                        write({"type":"workers","reason":"gpu-entry","state":sample})
                        pending=False
                await asyncio.sleep(0.001)
            write({"type":"stop","monotonic_ns":time.monotonic_ns(),"samples":n,"maximum_poll_gap_ns":max_gap,"state":await snapshot()})
    finally:
        bw.close()
        await bw.wait_closed()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("new_output")
    asyncio.run(observe(p.parse_args().new_output))
