"""Compare the two in-process serving modes on uncached streaming Responses."""
import argparse
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from hour_metrics import parse_metrics, serving_diff
from native_bench import measure
from native_check import logs
from native_paired import park
from native_service import BASE, RELEASE, host
from qualify import request
from throughput import handbook, write
from workload import graph


def metrics(base):
    with request(base, "/metrics", timeout=5) as response:
        return parse_metrics(response.read().decode())


def cool_between_trials(run, target):
    control = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))/"dsv41-pearl/broker.sock"
    start = time.monotonic()
    with park(uuid4().hex, broker=control):
        while max(temps := host.cards()) > target:
            if time.monotonic()-start > 120:
                raise TimeoutError("Could not reach the requested benchmark starting temperature")
            time.sleep(1)
        release_ns = time.monotonic_ns()
    # The timed request must cut in on all four mining GPUs after the test lease ends.
    deadline = time.monotonic()+10
    while time.monotonic()<deadline:
        ranks = logs(run)
        if len(ranks)==4 and all(any(e["event"]=="native_reentry" and e["gpu_seen_ns"]>=release_ns
                                      for e in events) for events in ranks.values()):
            return dict(temperatures_c=temps, interval_s=time.monotonic()-start, all_four_reentered=True)
        time.sleep(.03)
    raise RuntimeError("K780 did not resume after benchmark preconditioning")


def benchmark(base, mode, out, resume=False, cool_long_to=None):
    profile = graph.profile(base)
    run = (RELEASE/"current").resolve() if base==BASE else None
    if run and json.loads((run/"ready.json").read_text()).get("mode","joined") != mode:
        raise ValueError("Endpoint mode differs from the requested benchmark")
    if cool_long_to is not None and (not run or not 45 <= cool_long_to < host.GPU_TEMP_CUTOFF_C):
        raise ValueError("Cooling between trials requires the canonical endpoint and a valid GPU temperature")
    result = json.loads(out.read_text()) if resume else dict(mode=mode, profile=profile, rows=[])
    if result["mode"]!=mode or any(result["profile"][k]!=profile[k] for k in
            ("cuda_graph_config","served_model_name","tp_size","speculative_algorithm","mem_fraction_static")):
        raise ValueError("Cannot resume measurements from a different serving configuration")
    # Warm both short-prompt shapes. Every measured request uses a distinct prefix-cache salt.
    for index, notes in enumerate((14, 150)):
        measure(base, 2400+index, notes, 16, cache_salt=f"{out.stem}-warm-{index}")
    cases = [(1350,256),(9420,256)] * 4 + [(200000,512),(131072,512),(32768,8192)]
    already_measured = len(result["rows"])
    if [(r["size"],r["budget"]) for r in result["rows"]]!=cases[:already_measured]:
        raise ValueError("Saved benchmark rows differ from this workload")
    session = uuid4().hex
    for index,(size,budget) in enumerate(cases):
        if index<already_measured:
            continue
        salt = f"{session}-{index}"
        cooling = cool_between_trials(run,cool_long_to) if cool_long_to is not None and size>=32768 else None
        before = metrics(base)
        if before["active"]:
            raise RuntimeError("Overlapping request at benchmark start")
        if size < 10000:
            r = measure(base, 2420+index, 14 if size==1350 else 150, budget, cache_salt=salt)
            row = {k:r[k] for k in ("prompt_sha256","input_tokens","output_tokens","cached_tokens",
                                     "ttft_ms","client_decode_tps","output_sha256","start_ns","end_ns")}
        else:
            r = handbook(base, "deepseek-v4-flash", size, budget, cache_salt=salt)
            row = {k:r[k] for k in ("prompt_sha256","input_tokens","output_tokens","cached_tokens","client_decode_tps")}
            row.update(ttft_ms=1000*r["ttft_s"], output_sha256=r["output_delta_sha256"],
                       start_ns=round(r["start_monotonic"]*1e9), end_ns=round(r["end_monotonic"]*1e9))
        # Tokenizer counters settle shortly after the terminal client event.
        end = time.monotonic()+3
        while True:
            after = metrics(base)
            completed = after["totals"].get("num_requests_total",0)-before["totals"].get("num_requests_total",0)
            if completed or time.monotonic() >= end:
                break
            time.sleep(.03)
        serving = serving_diff(before, after, quiet_boundaries=True)
        if completed!=1 or serving["active_at_boundaries"]!=[0,0] or row["cached_tokens"]!=0:
            raise RuntimeError(f"Unisolated or cached benchmark at input {size}")
        if row["output_tokens"] != budget:
            raise RuntimeError(f"Output stopped before the requested sample at input {size}")
        delta = graph.graph_deltas(before, after)
        if any(count.get("decode_cuda_graph",0)<=0 for count in delta.values()):
            raise RuntimeError("Expected four graph-enabled decoding ranks")
        row.update(size=size,budget=budget,session=run.name if run else None,preconditioning=cooling,
                   server_prefill_tps=serving["completed_request_prefill_tokens_per_s"],
                   server_decode_tps=serving["token_weighted_inter_token_tps"],graphs=delta)
        result["rows"].append(row)
        write(out,result)
        print(mode,size,row["input_tokens"],row["output_tokens"],round(row["ttft_ms"],1),
              round(row["client_decode_tps"],2),flush=True)
        time.sleep(.4)
    result["complete"]=True
    write(out,result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("joined","non-joined"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--base",default="http://127.0.0.1:8000")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--cool-long-to",type=int,help="Park idle K780 between large test prompts until this temperature")
    args=parser.parse_args()
    benchmark(args.base,args.mode,args.out,args.resume,args.cool_long_to)
