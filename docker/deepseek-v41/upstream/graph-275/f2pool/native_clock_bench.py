"""Measure matched cold and burst Responses requests on the live non-joined deployment."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
import uuid

import pynvml as nv
from hour_metrics import serving_diff
from native_bench import measure
from native_check import logs, rows
from native_mode_bench import cool_between_trials, metrics
from native_mode_report import embedded_points
from native_report import timings
from native_service import BASE, RELEASE
from throughput import write
from workload import graph


# Each group starts from sustained mining. The second fixture cuts in during the high-clock hold.
GROUPS = ((14, 14), (150, 150), (540,), (14, 14), (2190,), (14, 14), (150, 150), (14, 14))
CASES = {14: "1.35K", 150: "9.4K", 540: "32K", 2190: "131K"}
PERMIT = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "dsv41-pearl/idle.permit"


class ConcurrentDemand(Exception):
    """Another request crossed a measurement boundary."""


def epoch():
    return struct.unpack("<QQ", PERMIT.read_bytes())[0]


def identity(events):
    result = []
    for rank in range(4):
        names = {e["worker"] for e in events[rank] if e["event"] == "authorized"}
        if len(names) != 1:
            raise RuntimeError("Worker identity changed")
        pid = next(e["pid"] for e in events[rank] if e["event"] == "native_rank")
        result.append([pid, hashlib.sha256(names.pop().encode()).hexdigest()])
    return result


@contextmanager
def record_clocks(path):
    nv.nvmlInit()
    stop = threading.Event()
    history = []
    failures = []
    try:
        cards = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
        if [nv.nvmlDeviceGetPowerManagementLimit(c) for c in cards] != [275000] * 4:
            raise RuntimeError("Expected the four production GPUs at 275 W")
        def sample():
            try:
                with path.open("x", buffering=1) as out:
                    while not stop.is_set():
                        clocks = [nv.nvmlDeviceGetClockInfo(c, nv.NVML_CLOCK_MEM) for c in cards]
                        temps = [nv.nvmlDeviceGetTemperature(c, nv.NVML_TEMPERATURE_GPU) for c in cards]
                        row = (time.monotonic_ns(), clocks, temps)
                        history.append(row)
                        print(json.dumps(row, separators=(",", ":")), file=out)
                        stop.wait(.035)
            except Exception as exc:
                failures.append(exc)
        worker = threading.Thread(target=sample, daemon=True)
        worker.start()
        try:
            yield history
        finally:
            stop.set()
            worker.join(5)
            if failures:
                raise RuntimeError("Clock telemetry failed") from failures[0]
    finally:
        nv.nvmlShutdown()


def cold_start(want, history):
    start = time.monotonic()
    previous = None
    idle_since = start
    while time.monotonic() - start < 180:
        value = epoch()
        if value != previous or value % 2 == 0:
            previous = value
            idle_since = time.monotonic()
        if (time.monotonic() - idle_since >= 6.5 and history
                and history[-1][1] == [want] * 4 and time.monotonic_ns() - history[-1][0] < 1e9):
            return
        time.sleep(.05)
    raise RuntimeError(f"No sustained idle period with four GPUs at {want} MHz")


def request_row(index, notes, position, history, expected):
    before = metrics(BASE)
    if before["active"] or not epoch() % 2:
        raise ConcurrentDemand("Live demand preceded the fixture")
    budget = 256 if notes < 540 else 512
    row = measure(BASE, index, notes, budget, cache_salt="clock-" + uuid.uuid4().hex)
    deadline = time.monotonic() + 3
    while True:
        after = metrics(BASE)
        if after["totals"].get("num_requests_total", 0) > before["totals"].get("num_requests_total", 0):
            break
        if time.monotonic() > deadline:
            raise RuntimeError("Missing completed-request counters")
        time.sleep(.03)
    if (after["active"] or after["totals"].get("num_requests_total", 0) -
            before["totals"].get("num_requests_total", 0) != 1):
        raise ConcurrentDemand("Live demand overlapped the fixture")
    data = serving_diff(before, after, quiet_boundaries=True)
    deltas = graph.graph_deltas(before, after)
    if (data["completed_requests"] != 1 or data["active_at_boundaries"] != [0, 0]
            or data["fresh_prompt_tokens"] != row["input_tokens"] or row["cached_tokens"] != 0
            or not data["completed_request_prefill_tokens_per_s"] or row["output_tokens"] != budget
            or set(deltas) != set("0123")
            or any(d["decode_cuda_graph"] <= 0 for d in deltas.values())):
        raise RuntimeError("Unexpected Responses token budget, cache status, prefill counter or graph execution")
    pre = next(r for r in reversed(history) if r[0] <= row["start_ns"])
    active = [r for r in history if row["start_ns"] <= r[0] <= row["end_ns"]]
    if pre[1] != [expected] * 4 or row["start_ns"] - pre[0] > 250e6 or not active:
        raise RuntimeError("Unexpected start clocks or missing request telemetry")
    high = next((t for t, clocks, _ in active if clocks == [13365] * 4), None)
    high_delay_ms = None if high is None else (high - row["start_ns"]) / 1e6
    if pre[1] == [13365] * 4:
        high_delay_ms = 0.0
    row.update(case=CASES[notes], position=position, before_mhz=pre[1], before_temperature_c=pre[2],
               peak_c=max(max(r[2]) for r in active), server_prefill_tps=data["completed_request_prefill_tokens_per_s"],
               graph_replays=[deltas[str(k)]["decode_cuda_graph"] for k in range(4)],
               all_high_from_http_ms=high_delay_ms)
    return row


def finish(run, requests, history, want):
    last = requests[-1]["end_ns"]
    offsets = [[dict(elapsed=t) for t in (0, 30, 60, 90)] for _ in range(4)]
    deadline = time.monotonic() + 360
    while time.monotonic() < deadline:
        time.sleep(2)
        if time.monotonic_ns() - last < 103e9:
            continue
        events = logs(run)
        broker = rows(run / "broker.jsonl")
        try:
            mining_epoch, aligned = embedded_points(events, broker, offsets, last, 12)
        except ValueError:
            continue
        first = min(p[0]["elapsed"] for p in aligned)*1e9
        end = max(p[-1]["elapsed"] for p in aligned)*1e9
        samples = [r for r in history if first <= r[0] <= end]
        if len(samples) < 900 or any(r[1] != [want]*4 for r in samples):
            raise RuntimeError("Clock changed during the mining comparison")
        rates = [(p[-1]["tmac"]-p[0]["tmac"])/(p[-1]["elapsed"]-p[0]["elapsed"]) for p in aligned]
        t0, t1 = max(p[0]["time"] for p in aligned), min(p[-1]["time"] for p in aligned)
        shares = [[sum(e["event"]=="share" and e["accepted"]==ok and t0<=e["time"]<=t1 for e in events[k])
                   for ok in (True, False)] for k in range(4)]
        return dict(epoch=mining_epoch, rank_tmac_s=rates, fleet_tmac_s=sum(rates), shares=shares,
                    common_seconds=t1-t0, peak_core_c=max(max(r[2]) for r in samples),
                    handoff=timings(requests, broker, events)), identity(events)
    raise RuntimeError("No uninterrupted 90-second native mining window after the requests")


def benchmark(policy, out):
    want = 7001 if policy == "adaptive" else 13365
    run = (RELEASE / "current").resolve(strict=True)
    if json.loads((run/"ready.json").read_text())["mode"] != "non-joined":
        raise RuntimeError("Expected the production non-joined service")
    original = identity(logs(run))
    data = dict(policy=policy, run=run.name, worker_identity=original, rows=[])
    write(out, data)
    with record_clocks(out.with_suffix(".clocks.jsonl")) as history:
        for group, fixtures in enumerate(GROUPS):
            for attempt in range(4):
                while metrics(BASE)["active"]:
                    time.sleep(.5)
                if fixtures[0] >= 540:
                    cool_between_trials(run, 70)
                cold_start(want, history)
                batch = []
                try:
                    for n, notes in enumerate(fixtures):
                        if n:
                            time.sleep(.25)
                        row = request_row(97600+2*group+n, notes, "burst" if n else "cold", history, 13365 if n else want)
                        batch.append(row)
                except ConcurrentDemand:
                    if attempt == 3:
                        raise
                    continue
                data["rows"].extend(batch)
                write(out, data)
                for row in batch:
                    print(policy, row["case"], row["position"], round(row["ttft_ms"], 1), round(row["client_decode_tps"], 2), flush=True)
                break
        mining, end_identity = finish(run, data["rows"], history, want)
        if original != end_identity or (RELEASE / "current").resolve() != run:
            raise RuntimeError("Production workers changed during the trial")
        data.update(mining=mining, complete=True)
        write(out, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=("automatic", "adaptive"))
    parser.add_argument("out", type=Path)
    cfg = parser.parse_args()
    benchmark(cfg.policy, cfg.out)
