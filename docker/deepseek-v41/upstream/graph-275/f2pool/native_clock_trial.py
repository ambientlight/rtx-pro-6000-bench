"""Run a bounded adaptive-clock trial using a completed, matched full-clock baseline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from native_clock_bench import benchmark
from native_clock_compare import compare, write_table
from native_check import rows
from native_report import quantiles
from native_service import RELEASE
from throughput import write


def run(baseline, directory):
    control = json.loads(baseline.read_text())
    if not control.get("complete") or control["policy"] != "automatic":
        raise ValueError("A completed full-clock baseline is required")
    if not Path("/run/dsv41-memory-clock.sock").is_socket():
        raise RuntimeError("Clock socket has not been installed")
    if (RELEASE/"current").resolve().name != control["run"]:
        raise RuntimeError("The production deployment changed after the baseline")
    unit = "dsv41-adaptive-clock.service"
    if subprocess.run(["systemctl","--user","is-active","--quiet",unit]).returncode == 0:
        raise RuntimeError("An adaptive clock controller already owns the production policy")
    result = directory / "adaptive.json"
    if result.exists() or result.with_suffix(".clocks.jsonl").exists():
        raise FileExistsError("Use a fresh trial output directory")
    watcher = Path(__file__).with_name("native") / "clock" / "watch_clock.py"
    with (directory / "adaptive-broker.jsonl").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-I", str(watcher)], stdout=log)
        try:
            benchmark("adaptive", result)
            if proc.poll() is not None:
                raise RuntimeError("Clock controller exited during the trial")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            until = time.monotonic() + 15
            while True:
                observed = subprocess.check_output(
                    ["/usr/bin/nvidia-smi","--query-gpu=clocks.mem","--format=csv,noheader,nounits"], text=True)
                clocks = list(map(int, observed.split()))
                if clocks == [13365]*4 or time.monotonic() > until:
                    break
                time.sleep(.15)
            write(directory/"trial-release.json", dict(final_memory_mhz=clocks, controller_exited=True))
    if clocks != [13365]*4:
        raise RuntimeError("Automatic clocks have not returned on every mining GPU")
    adaptive = json.loads(result.read_text())
    data = compare(control, adaptive)
    broker = rows(RELEASE / "current/broker.jsonl")
    events = rows(directory / "adaptive-broker.jsonl")
    releases = {e["epoch"]:e["monotonic_ns"] for e in events if e["event"]=="automatic_requested"}
    delays = []
    for req in adaptive["rows"]:
        if req["position"] == "cold":
            edges = [e for e in broker if e["event"]=="native_demand" and
                     req["start_ns"] <= e["monotonic_ns"] <= req["start_ns"]+req["ttft_ms"]*1e6]
            if len(edges) != 1 or edges[0]["epoch"] not in releases:
                raise RuntimeError("Missing broker-to-clock handoff on a cold request")
            e = edges[0]
            delays.append((releases[e["epoch"]] - e["monotonic_ns"])/1e6)
    data["cold_broker_demand_to_clock_socket_release_ms"] = quantiles(delays)
    data["post_trial_memory_mhz"] = clocks
    write(directory/"comparison.json", data)
    write_table(directory/"comparison.csv", data)
    print(f"Finished {data['pairs']} request pairs. Idle mining change {data['mining_change_pct']:+.2f}%.",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    run(options.baseline, options.directory)
