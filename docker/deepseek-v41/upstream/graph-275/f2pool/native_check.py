"""Qualify native K780 mining with public Responses fixtures on the private four-rank endpoint."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import fullmodel as host
from workload import graph
from throughput import write

RUN = host.RUN
GATEWAY = RUN/"receipts/gateway.json"
ARM = RUN/"armed"
BASE = "http://127.0.0.1:18002"


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.endswith("\n")]


def logs(directory=RUN):
    return {int(path.name.split("-")[1]): rows(path) for path in sorted((directory/"logs").glob("native-*.jsonl"))}


def wait(fn, seconds, what):
    stop = time.monotonic()+seconds
    while time.monotonic() < stop:
        if (host.STATE/"guard.json").exists() or (host.STATE/"restart.json").exists() or not host.active(host.CANDIDATE):
            raise RuntimeError("Model stopped while waiting for "+what)
        if fn():
            return
        time.sleep(.15)
    raise TimeoutError(what)


def ready():
    for name in host.SMOKES:
        path=host.STATE/name
        if not path.exists() or json.loads(path.read_text()) != {"status":"passed"}:
            return False
    data = logs()
    if any(e["event"] == "native_error" for events in data.values() for e in events):
        raise RuntimeError("A native rank reported an error")
    return (len(data)==4 and all(any(e["event"]=="native_work" and e["graphs"]>=139 for e in events)
                                for events in data.values()) and json.loads(GATEWAY.read_text())["authorized_clients"]==4)


def snapshot():
    ranks = graph.drained_projection(host.STATE/"shapes")
    pool = json.loads(GATEWAY.read_text())
    return dict(phases={r:graph.phase_counters(ranks,r) for r in ranks["shapes"]},
                graphs=graph.snapshot(BASE)["graphs"],
                pool={k:pool[k] for k in ("run_id","armed","submitted","accepted","rejected","pending")},
                monotonic_ns=time.monotonic_ns())


def owners(directory=RUN, *, joined=True):
    data = {}
    for rank, events in logs(directory).items():
        rec = next(e for e in events if e["event"]=="native_rank")
        pid = rec["pid"]
        data[rank] = dict(pid=pid, context_matches=rec["thread_context"] == rec["borrowed_context"],
                          scheduler_receipt=(directory/"serving/shapes"/f"pearl-shapes-{pid}-tp{rank}.json").exists())
        if not joined:
            startup = json.loads((directory/"logs"/f"native-{rank}-{pid}.startup.json").read_text())
            libs = set(startup["pearl_libraries"])
            if ((startup["rank"],startup["pid"],startup["observe"],startup["graph_verify"],startup["operator"])
                    != (rank,pid,False,False,None) or "libpearl_idle.so" not in libs
                    or libs & {"libpearl_wo_b.so", "libpearl_prepare.so", "libpearl_reference.so"}):
                raise RuntimeError(f"Unexpected inference hooks or Pearl libraries in scheduler {rank}")
            data[rank]["loaded_pearl_libraries"] = sorted(libs)
    raw=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,gpu_uuid",
                                 "--format=csv,noheader"],text=True)
    gpu=[dict(pid=int(pid),process=Path(name.strip()).name,gpu_uuid=uuid.strip())
         for pid,name,uuid in (line.split(", ",2) for line in raw.splitlines())]
    for rank,rec in data.items():
        match=[p for p in gpu if p["pid"]==rec["pid"] and "scheduler" in p["process"]]
        if not (rec["context_matches"] and rec["scheduler_receipt"] == joined and len(match)==1):
            raise RuntimeError(f"Could not attribute GPU work to serving rank {rank}")
    if len({p["gpu_uuid"] for p in gpu if p["pid"] in {r["pid"] for r in data.values()}})!=4 or any(p["process"]=="bpm" for p in gpu):
        raise RuntimeError("Unexpected GPU ownership")
    return dict(ranks=data,gpu_processes=gpu)


def bench(file, transitions=0):
    cmd=[sys.executable,str(host.HERE.with_name("native_bench.py")),"--port","18002","--key-file",
         str(host.STATE/"api-key"),"--out",str(RUN/file)]
    if transitions:
        cmd.extend(("--transitions",str(transitions)))
    subprocess.run(cmd,check=True)


def main():
    os.environ["CANARY_API_KEY_FILE"]=str(host.STATE/"api-key")
    wait(ready,650,"four warmed rank-owned miners, inference clients and native smokes")
    write(RUN/"ownership.json",owners())
    write(RUN/"native-profile.json",graph.profile(BASE))
    print("All four GPU contexts matched their SGLang scheduler PIDs",flush=True)
    # A quiet interval after graph warmup supplies steady-state, simultaneously active idle rates.
    time.sleep(8)
    if ARM.exists():
        raise RuntimeError("Expected the inference gateway disarmed before qualification")
    ARM.touch(0o600,exist_ok=False)
    try:
        wait(lambda: json.loads(GATEWAY.read_text())["armed"],10,"inference pool arm")
        write(RUN/"before-public.json",snapshot())
        bench("native.json")
        bench("transitions.json",100)
        time.sleep(.6)
        write(RUN/"after-public.json",snapshot())
    finally:
        ARM.unlink(missing_ok=True)
    wait(lambda: not json.loads(GATEWAY.read_text())["armed"],10,"inference gateway disarm")
    print("Public qualification complete",flush=True)


if __name__ == "__main__":
    main()
