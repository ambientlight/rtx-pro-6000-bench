#!/usr/bin/env python3
"""Paired external-admission, warm-prefix TTFT test on an unchanged local SGLang."""
import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "miner"))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from broker import Fleet
from dsv41_load import LOAD_PATH, activity_metrics, is_idle, parse_loads
from fleet_replay import command, exchange, until_resumed

now = time.monotonic_ns
PROMPT = ("Synthetic prefix for a streaming timing test. " * 64 +
          "What is two plus two? Reply with one digit.")

def utc():
    return datetime.now(timezone.utc).isoformat()

def snapshot(url):
    with urlopen(url + LOAD_PATH, timeout=3) as r:
        idle = is_idle(parse_loads(json.load(r)))
    with urlopen(url + "/metrics", timeout=3) as r:
        metrics = r.read().decode()
        active, totals = activity_metrics(metrics)
        cached = [int(float(line.rsplit(" ",1)[1])) for line in metrics.splitlines()
                  if line.startswith("sglang:cached_tokens_total{")]
    return idle, active, sum(v for _, v in totals), sum(cached) if cached else None

async def completed_request_snapshot(base, previous_total):
    # /v1/loads publishes scheduler snapshots asynchronously. The snapshot may
    # still describe this exact request immediately after its final stream byte.
    deadline = time.monotonic() + 4
    while True:
        idle, active, total, cached = await asyncio.to_thread(snapshot, base)
        if total > previous_total + 1 or active > 1:
            raise RuntimeError(f"outside inference during trial (HTTP active={active}, total {previous_total}->{total})")
        if idle and not active and total == previous_total + 1:
            return total, cached
        if time.monotonic() >= deadline:
            raise RuntimeError(f"stock server did not settle (idle={idle}, HTTP active={active}, total {previous_total}->{total})")
        await asyncio.sleep(.08)

def stream(conn, encoded, started):
    # Reuse the local HTTP/1.1 connection. Empty role/usage/final events have no
    # generated text. SGLang may deliver more than one model token per SSE event.
    post = now()
    conn.request("POST", "/v1/chat/completions", body=encoded,
                 headers={"Content-Type":"application/json"})
    first = kind = usage = None
    done = False
    with conn.getresponse() as r:
        if r.status != 200:
            raise RuntimeError(f"stock HTTP status {r.status}")
        for line in r:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                continue
            msg = json.loads(data)
            if msg.get("error"):
                raise RuntimeError("stock inference stream failed")
            if first is None:
                for choice in msg.get("choices", []):
                    delta = choice.get("delta", {})
                    kind = next((k for k in ("reasoning_content", "content")
                                 if isinstance(delta.get(k),str) and delta[k]), None)
                    if kind:
                        first = now()
                        break
            if msg.get("usage"):
                usage = msg["usage"]
    if not done or first is None or not usage or usage.get("completion_tokens",0) < 1:
        raise RuntimeError("incomplete stream, missing text, or missing token usage")
    return dict(ttft_ms=(first-started)/1e6, http_to_first_text_ms=(first-post)/1e6,
                completion_ms=(now()-started)/1e6, first_delta=kind,
                prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage["completion_tokens"],
                cached_prompt_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"))

@asynccontextmanager
async def fleet(args):
    with tempfile.TemporaryDirectory(prefix="pearl-ttft-") as d:
        folder = Path(d)
        paths = [folder / f"gpu{i}.sock" for i in range(4)]
        broker_path = folder / "broker.sock"
        processes = []
        async with AsyncExitStack() as stack:
            try:
                for i, (gpu, sock) in enumerate(zip(args.gpu,paths)):
                    with (folder / f"gpu{i}.log").open("w") as log:
                        processes.append(subprocess.Popen(
                            [str(args.worker.resolve()), "--bench-rounds","512","--bench-job",str(71+i),
                             "--bench-nonce","1001","--kernel-variant","v1","--k","65536",
                             "--control-socket",str(sock),"--slice-tiles",str(args.slice_tiles),
                             "--device","0","--power-limit","280","--max-temp","92"],
                            env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"RAYON_NUM_THREADS":"2"},
                            stdout=log,stderr=subprocess.STDOUT))
                deadline = time.monotonic()+30
                while not all(p.exists() for p in paths):
                    if time.monotonic() > deadline or any(p.poll() is not None for p in processes):
                        raise RuntimeError("fixture worker startup failed")
                    await asyncio.sleep(.02)
                controller = Fleet(paths)
                server = await asyncio.start_unix_server(controller.client,str(broker_path),limit=512)
                await stack.enter_async_context(server)
                controller.start()
                stack.push_async_callback(controller.stop)
                channels = [await asyncio.open_unix_connection(str(broker_path)) for _ in range(5)]
                for _, writer in channels:
                    stack.callback(writer.close)
                for i, ch in enumerate(channels[:4]):
                    await command(ch,f"BUSY {i}","OK")
                await command(channels[4],"ADMIT init","DRAINED")
                for i, ch in enumerate(channels[:4]):
                    await command(ch,f"IDLE {i}","OK")
                released = now()
                await command(channels[4],"DONE init","OK")
                await until_resumed(paths,released)
                # Capture the executable the OS actually started. A development
                # rebuild can replace the file at args.worker during the trial.
                worker_hashes = {hashlib.sha256(Path(f"/proc/{p.pid}/exe").read_bytes()).hexdigest()
                                 for p in processes}
                if len(worker_hashes) != 1:
                    raise RuntimeError("worker binary changed between GPU launches")
                yield paths, channels[4], worker_hashes.pop()
            finally:
                for proc in processes:
                    if proc.poll() is None:
                        proc.terminate()
                for proc in processes:
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

async def main(args):
    endpoint = urlsplit(args.url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"localhost","127.0.0.1","::1"} or endpoint.path not in ("", "/"):
        raise ValueError("use a local stock SGLang HTTP origin")
    if len(args.gpu) != 4 or len(set(args.gpu)) != 4 or not 2 <= args.pairs <= 24:
        raise ValueError("the trial uses four distinct GPUs and 2–24 pairs")
    base = args.url.rstrip("/")
    rng = random.Random(args.seed)
    orders = [i % 2 == 0 for i in range(args.pairs)]
    rng.shuffle(orders)
    payload = dict(model="deepseek-v4-flash", messages=[dict(role="user",content=PROMPT)],
                   chat_template_kwargs=dict(thinking=False), temperature=0, max_tokens=4,
                   stream=True, stream_options=dict(include_usage=True))
    encoded = json.dumps(payload).encode()
    receipt = dict(start_utc=utc(), pairs_requested=args.pairs, seed=args.seed,
                   model=payload["model"], gpu=args.gpu, slice_tiles=args.slice_tiles,
                   max_output_tokens=4, payload_sha256=hashlib.sha256(encoded).hexdigest(),
                   scope="External broker ingress. HTTP is dispatched after drain in both arms. Stock SGLang and its existing production miners are unchanged. Prototype workers are additional bounded offline miners. Client metric is first nonempty text or reasoning delta, including broker wait. Warm identical request and reused HTTP connection.",
                   warmups=[], pairs=[], stopped_reason=None)
    idle, active, totals, _ = await asyncio.to_thread(snapshot,base)
    if not idle or active:
        receipt["stopped_reason"] = "outside inference active before startup"
        return receipt
    conn = http.client.HTTPConnection(endpoint.hostname,endpoint.port or 80,timeout=30)
    try:
        async with fleet(args) as (paths, api, worker_hash):
            receipt["worker_exe_sha256"] = worker_hash
            receipt["broker_watch_workers"] = False
            index = 0
            for order in (None, *orders):
                pair = {}
                arms = ["preheld","cut_in"] if order is None or order else ["cut_in","preheld"]
                for arm in arms:
                    idle, active, observed, cached_before = await asyncio.to_thread(snapshot,base)
                    if not idle or active or observed != totals:
                        receipt["stopped_reason"] = "outside inference observed before trial"
                        return receipt
                    await asyncio.sleep(rng.uniform(.010,.025))
                    before = await asyncio.gather(*(exchange(p) for p in paths))
                    if any(r["faulted"] or r["holds"] for r in before):
                        raise RuntimeError("fixture worker not mining at start")
                    rid = f"stream-{index}"
                    if arm == "preheld":
                        await command(api,"ADMIT baseline-"+rid,"DRAINED")
                    started = now()
                    await command(api,"ADMIT "+rid,"DRAINED")
                    drained = now()
                    result = await asyncio.to_thread(stream,conn,encoded,started)
                    result["broker_drain_ms"] = (drained-started)/1e6
                    held = await asyncio.gather(*(exchange(p) for p in paths))
                    if any(r["gpu_busy"] or not r["holds"] or r["faulted"] for r in held):
                        raise RuntimeError("fixture worker was not held during inference")
                    if arm == "preheld":
                        await command(api,"DONE baseline-"+rid,"OK")
                    release = now()
                    await command(api,"DONE "+rid,"OK")
                    await until_resumed(paths,release)
                    after, cached_after = await completed_request_snapshot(base, totals)
                    result["server_cached_tokens_delta"] = (cached_after - cached_before
                        if cached_after is not None and cached_before is not None else None)
                    totals=after
                    index+=1
                    pair[arm]=result
                if order is None:
                    receipt["warmups"]=pair
                else:
                    receipt["pairs"].append(dict(index=len(receipt["pairs"]), order=arms,
                                                 **pair, delta_ms=pair["cut_in"]["ttft_ms"]-pair["preheld"]["ttft_ms"]))
    except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
        receipt["stopped_reason"] = str(exc)
    finally:
        conn.close()
    return receipt

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("worker",type=Path)
    p.add_argument("--gpu",action="append",required=True)
    p.add_argument("--url",default="http://127.0.0.1:8000")
    p.add_argument("--pairs",type=int,default=12)
    p.add_argument("--seed",type=int,default=413)
    p.add_argument("--slice-tiles",type=int,default=188)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    receipt=asyncio.run(main(args))
    receipt["end_utc"]=utc()
    pairs=receipt["pairs"]
    if pairs:
        ds=[p["delta_ms"] for p in pairs]
        receipt["summary"]=dict(preheld_median_ms=statistics.median(p["preheld"]["ttft_ms"] for p in pairs),
                                cut_in_median_ms=statistics.median(p["cut_in"]["ttft_ms"] for p in pairs),
                                paired_median_delta_ms=statistics.median(ds), min_delta_ms=min(ds), max_delta_ms=max(ds),
                                cut_in_drain_median_ms=statistics.median(p["cut_in"]["broker_drain_ms"] for p in pairs))
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(pairs=len(receipt["pairs"]),stopped=receipt["stopped_reason"],summary=receipt.get("summary"))))
