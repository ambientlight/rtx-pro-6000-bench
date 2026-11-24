"""Measure native prefill time, decode time and scanned mining work on isolated Responses calls."""
import argparse
import json
import os
from pathlib import Path
import time

from hour_metrics import BASE_RATE, prefill_bounds, server, serving_diff
from native_bench import measure
from workload import graph, request_projection, scan_summary
from throughput import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--shape-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CANARY_API_KEY_FILE"] = str(args.key_file)
    base = f"http://127.0.0.1:{args.port}"
    rows = []
    for index, notes in enumerate((14, 150) * 3):
        before = graph.drained_projection(args.shape_dir)
        left = server(args)
        request = measure(base, 510 + index, notes, 256, cache_salt=f"native-phase-{index}")
        time.sleep(.35)
        right = server(args)
        serving = serving_diff(left, right, quiet_boundaries=True)
        if (serving["completed_requests"] != 1 or serving["active_at_boundaries"] != [0,0]
                or request["cached_tokens"] != 0 or not serving["response_endpoint_only"]):
            raise RuntimeError("Phase measurements require an isolated uncached Responses stream")
        deltas = graph.graph_deltas(left, right)
        after = request_projection(args.shape_dir, before, deltas, request["input_tokens"])
        work = scan_summary(before, after, deltas)
        prefill = prefill_bounds(before, after, work)
        decode = work["reconciled_mac_equivalent"]["decode"]
        times = {p:serving["histograms"][h]["sum_s"] for p,h in
                 (("prefill","prefill_forward_seconds"),("decode","inter_token_latency_seconds"))}
        macs = {"prefill":prefill, "decode":[decode, decode]}
        rows.append(dict(input_tokens=request["input_tokens"], output_tokens=request["output_tokens"],
                         ttft_ms=request["ttft_ms"], client_decode_tps=request["client_decode_tps"],
                         server_prefill_tps=serving["completed_request_prefill_tokens_per_s"],
                         server_decode_tps=serving["token_weighted_inter_token_tps"],
                         seconds=times, mac_equivalent=macs,
                         scanned_tickets={p:sum(r.values()) for p,r in work["tickets"].items()}))
        write(args.out, dict(rows=rows, direct_baseline_tmac_s=BASE_RATE))
        print(json.dumps({k: rows[-1][k] for k in ("input_tokens","server_prefill_tps","client_decode_tps")}), flush=True)


if __name__ == "__main__":
    main()
