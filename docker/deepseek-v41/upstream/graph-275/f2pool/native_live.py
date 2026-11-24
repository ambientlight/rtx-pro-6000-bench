"""Verify the canonical Responses endpoint, four GPU owners and real F2Pool connections after promotion."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request

import fullmodel as host
from native_bench import measure
from native_check import logs, owners
from native_report import live_shares
from native_service import BASE, RELEASE, UNIT
from throughput import sse, write
from workload import graph


def arithmetic():
    body = dict(model="deepseek-v4-flash", store=False, stream=True, temperature=0,
                input="What is 19 + 23? Reply only with the number.", max_output_tokens=96,
                chat_template_kwargs={"thinking": False})
    req = urllib.request.Request(BASE + "/v1/responses", data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    out = []
    finished = False
    with urllib.request.urlopen(req, timeout=45) as response:
        for event in sse(response):
            if event["type"] == "response.output_text.delta":
                out.append(event["delta"])
            elif event["type"] == "response.completed":
                finished = True
        if response.status != 200 or not finished or "".join(out).strip().rstrip(".") != "42":
            raise RuntimeError("Canonical Responses arithmetic check failed")


def phase_totals(directory):
    state = graph.drained_projection(directory)
    return {rank:{phase:graph.phase_counters(state, rank).get(phase+"_scan",{}).get("tickets_scanned",0)
                  for phase in ("prefill","decode")} for rank in state["shapes"]}


def check(out):
    os.environ.pop("CANARY_API_KEY_FILE", None)
    run = (RELEASE / "current").resolve(strict=True)
    profiles = graph.profile(BASE)
    if not host.active(UNIT) or not (run / "ready.json").exists():
        raise RuntimeError("Production startup has not finished")
    ready = json.loads((run / "ready.json").read_text())
    mode = ready.get("mode", "joined")
    joined = mode == "joined"
    if mode not in ("joined", "non-joined") or ready["inference_proofs_enabled"] != joined:
        raise RuntimeError("Serving mode and proof arming disagree")
    state = owners(run, joined=joined)
    if len(state["ranks"]) != 4 or len(state["gpu_processes"]) != 4:
        raise RuntimeError("Expected exactly the four SGLang schedulers on the GPUs")
    gateway_path = run / "receipts/gateway.json"
    shapes = run / "serving/shapes"
    delta = gateway = None
    if joined:
        first = json.loads(gateway_path.read_text())
        if not (first["armed"] and first["connected"] and first["authorized_clients"] == 4):
            raise RuntimeError("Expected four joined ranks with an armed live inference gateway")
        before = phase_totals(shapes)
    elif any(p.exists() for p in (shapes, gateway_path, run/"armed", run/"gateway-token", run/"logs/gateway.log")):
        raise RuntimeError("Found joined proof state in a non-joined run")
    graph_before = graph.snapshot(BASE)
    arithmetic()
    for i in range(4 if joined else 1):
        measure(BASE, 620+i, 32, 96, cache_salt="native-promotion-"+run.name+str(i))
        if not joined:
            break
        after = phase_totals(shapes)
        delta = {r:{p:after[r][p]-before[r][p] for p in before[r]} for r in before}
        if all(min(values.values()) > 0 for values in delta.values()):
            break
    else:
        raise RuntimeError("Inference scan counters did not advance in both phases on every rank")
    graphs = graph.graph_deltas(graph_before, graph.snapshot(BASE))
    if any(v.get("decode_cuda_graph", 0) <= 0 for v in graphs.values()):
        raise RuntimeError("All four ranks must execute graph-backed decoding")
    deadline = time.monotonic()+240
    while time.monotonic() < deadline:
        data = logs(run)
        ledger = {rank:live_shares(records) for rank,records in data.items()}
        if any(row["event"]=="native_error" for records in data.values() for row in records):
            raise RuntimeError("An embedded miner reported a failure")
        if len(ledger)==4 and all(w["accepted"]>0 for w in ledger.values()):
            break
        time.sleep(2)
    else:
        raise TimeoutError("Waiting for a live F2Pool acknowledgement from every native GPU")
    if joined:
        last = json.loads(gateway_path.read_text())
        if first["run_id"] != last["run_id"] or not (last["armed"] and last["connected"] and last["authorized_clients"]==4):
            raise RuntimeError("Production inference gateway lost the qualified session")
        gateway = {k:last[k] for k in ("connected","armed","authorized_clients","submitted","accepted","rejected")}
    owners_now = owners(run, joined=joined)
    if state != owners_now or not host.active(UNIT) or not (RELEASE/"current").samefile(run):
        raise RuntimeError("The production model changed during verification")
    receipt = dict(utc_epoch=time.time(),unit=UNIT,enabled=host.run("systemctl","--user","is-enabled",UNIT),
                   run=run.name, serving=profiles, port=8000, endpoint="/v1/responses", public_fixture_42=True,
                   fresh_smokes=list(host.SMOKES), ownership=state, scanned_ticket_delta_by_rank=delta,
                   inference_gateway=gateway, mode=mode, graph_deltas=graphs,
                   direct_pool=ledger, power_limit_w=275, peak_allowed_c=host.GPU_TEMP_CUTOFF_C, temperatures_c=host.cards(),
                   stock_container=host.run(*host.DOCKER,"inspect","-f","{{.State.Status}}","dsv41"))
    write(out,receipt)
    return receipt


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    result=check(parser.parse_args().out)
    print(f"Verified production Responses, {result['mode']} CUDA graph decoding and four F2Pool-paying scheduler ranks",flush=True)
