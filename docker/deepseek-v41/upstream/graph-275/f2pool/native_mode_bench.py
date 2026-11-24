"""Compare the two in-process serving modes on uncached streaming Responses."""
import argparse
import time
from pathlib import Path
from uuid import uuid4

from hour_metrics import parse_metrics, serving_diff
from native_bench import measure
from qualify import request
from throughput import handbook, write
from workload import graph


def metrics(base):
    with request(base, "/metrics", timeout=5) as response:
        return parse_metrics(response.read().decode())


def benchmark(base, mode, out):
    result = dict(mode=mode, profile=graph.profile(base), rows=[])
    # Warm both short-prompt shapes. Every measured request uses a distinct prefix-cache salt.
    for index, notes in enumerate((14, 150)):
        measure(base, 2400+index, notes, 16, cache_salt=f"{out.stem}-warm-{index}")
    cases = [(1350,256),(9420,256)] * 4 + [(200000,512),(131072,512),(32768,8192)]
    session = uuid4().hex
    for index,(size,budget) in enumerate(cases):
        salt = f"{session}-{index}"
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
        row.update(size=size,budget=budget,server_prefill_tps=serving["completed_request_prefill_tokens_per_s"],
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
    args=parser.parse_args()
    benchmark(args.base,args.mode,args.out)
