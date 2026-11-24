#!/usr/bin/python3
"""Inspect the original DeepSeek deployment; --apply restores its effective KV pool and the paid miners."""
import argparse
import fcntl
import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import restore_paid_mining as paid

sys.path.insert(0, "/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/scripts")
from dsv41_load import DrainWindow, activity_metrics, is_idle, parse_loads

CGROUP = Path(paid.NAMESPACE)
CID = CGROUP.name.removeprefix("docker-").removesuffix(".scope")
DOCKER = ["/usr/bin/docker", "--context", "shared-operator"]
SYSTEMCTL = "/usr/bin/systemctl"
NVIDIA = "/usr/bin/nvidia-smi"
SUDO = "/usr/bin/sudo"
UNIT = "pearl-f2pool.service"
CAPACITY = 3486976
BASE = "http://127.0.0.1:8000"
SMOKES = [Path("/mnt/hot/dsv41_state/production") / name for name in
          ("smoke.json", "smoke-structured.json", "smoke-tools.json", "smoke-vision.json")]


def read_command(*args, timeout=15):
    return subprocess.check_output(args, timeout=timeout, text=True).strip()


def run(*args, timeout=60):
    subprocess.run(args, check=True, timeout=timeout)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=8) as reply:
        return reply.read()


def limits():
    state = paid.report()
    rows = state["rows"]
    if (state["original_container_state"] != "running" or state["cgroup"] != paid.NAMESPACE or
        not state["namespace_present"] or [r["uuid"] for r in rows] != paid.EXPECTED or
        [r["index"] for r in rows] != list(range(4))):
        raise RuntimeError("The expected original running container, cgroup and GPU UUIDs are required.")
    for row in rows:
        if row["expected_hard_mib"] != paid.LIMIT or row.get("actual_hard_mib") not in ("max", paid.LIMIT):
            raise RuntimeError(f"Unexpected GPU {row['index']} quota; inspect the current deployment.")
    return state


def status():
    get("/health")
    conf = json.loads(get("/server_info"))
    expected = {"tp_size": 4, "ep_size": 4, "context_length": 524288, "mem_fraction_static": 0.87}
    if any(conf.get(k) != v for k, v in expected.items()):
        raise RuntimeError("The original DeepSeek TP4, 524288 context, 0.87 memory profile is required.")
    metrics = get("/metrics").decode()
    capacities = {}
    for labels, value in re.findall(r'^sglang:max_total_num_tokens\{([^}]+)\}\s+(\S+)$', metrics, re.M):
        rank = re.search(r'\btp_rank="(\d+)"', labels)
        if rank is None or rank[1] in capacities:
            raise RuntimeError("Invalid or duplicate TP rank in KV capacity metrics.")
        capacities[rank[1]] = int(float(value))
    if set(capacities) != {"0","1","2","3"} or set(capacities.values()) != {conf["max_total_num_tokens"]}:
        raise RuntimeError("The four rank capacities and /server_info must agree.")
    return conf["max_total_num_tokens"]


def unit_state():
    return read_command(SYSTEMCTL, "show", "-p", "ActiveState", "--value", UNIT)


def argv_for_limit(uuid):
    return [NVIDIA, "memory-limits", "-n", paid.NAMESPACE, "-i", uuid, "--hard-limit", str(paid.LIMIT)]


def preflight(restart):
    # Only these fixed root-owned commands are elevated. Python, Docker and the
    # orchestration/verification helpers continue running as the invoking user.
    run(SUDO, "-v", timeout=180)
    commands = [argv_for_limit(u) for u in paid.EXPECTED]
    commands.append([SYSTEMCTL, "start", UNIT])
    if restart and unit_state() == "active":
        commands.append([SYSTEMCTL, "stop", UNIT])
    for command in commands:
        check = subprocess.run([SUDO, "-n", "-l", "--", *command],
                               stdout=subprocess.DEVNULL, timeout=15)
        if check.returncode:
            raise RuntimeError(f"Missing sudo permission for {command[0]}; production is still running.")


def drain():
    window = DrainWindow(60, [0])
    previous = None
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        active, total = activity_metrics(get("/metrics").decode())
        rows = parse_loads(json.loads(get("/v1/loads")))
        if active or total != previous:
            window = DrainWindow(60, [0])
        if window.observe(rows, time.monotonic()) and not active:
            next_active, next_total = activity_metrics(get("/metrics").decode())
            if not next_active and next_total == total and is_idle(parse_loads(json.loads(get("/v1/loads")))):
                return total
            window = DrainWindow(60, [0])
        previous = total
        time.sleep(5)
    raise TimeoutError("No 60-second idle window was observed; the original is still running.")


def await_weights(start_time, old_inode):
    # In this specific release line 1148 is emitted after reading pre-load free
    # memory. The historical quota was set during target loading, before KV sizing.
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if CGROUP.is_dir() and CGROUP.stat().st_ino != old_inode:
            item = subprocess.run(["/usr/bin/journalctl", "-t", "dsv41", "--since", f"@{start_time:.6f}",
                                   "--no-pager", "-o", "cat", "--grep", "model_runner.py:1148|model_runner.py:1241"],
                                  capture_output=True, text=True, timeout=15)
            if item.returncode not in (0, 1):
                raise RuntimeError("The content-free weight-load journal is unavailable.")
            journal = item.stdout
            if "model_runner.py:1241" in journal:
                raise RuntimeError("The primary weight load finished before all four caps could be applied.")
            pids = set(re.findall(r'worker (\d+); model_runner\.py:1148\]', journal))
            if len(pids) == 4:
                return
        time.sleep(1)
    raise TimeoutError("The new original cgroup and four primary weight-load markers were unavailable.")


def apply_limits():
    for row in limits()["rows"]:
        if row["actual_hard_mib"] == paid.LIMIT:
            continue
        if row["used_mib"] > paid.LIMIT:
            raise RuntimeError(f"GPU {row['index']} usage exceeds the original quota.")
        run(SUDO, "-n", "--", *argv_for_limit(row["uuid"]))
    if not limits()["prior_policy_met"]:
        raise RuntimeError("Four original GPU quotas could not be verified.")


def wait_healthy(since_ns):
    deadline = time.monotonic() + 1500
    while time.monotonic() < deadline:
        if paid.healthy():
            fresh = True
            for path in SMOKES:
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    fresh = False
                    break
                if not stat.S_ISREG(info.st_mode) or info.st_uid != 0:
                    raise RuntimeError(f"Untrusted smoke receipt {path.name}.")
                if info.st_mtime_ns < since_ns:
                    fresh = False
                    break
            if fresh:
                # The frozen original writes each root-owned receipt only after
                # its individual check passes. No privileged receipt read is needed.
                return status()
        time.sleep(3)
    raise TimeoutError("Original DeepSeek bootstrap did not complete. The container has been left running.")


def ensure_running():
    # Docker normally performs an atomic restart. If it failed after stopping,
    # re-start only the recorded original ID, even when the operator interrupted.
    current = read_command(*DOCKER, "inspect", CID, "--format", "{{.Id}} {{.State.Status}}").split()
    if current[0] != CID:
        raise RuntimeError("Original container ID changed.")
    if current[1] not in ("running", "restarting"):
        run(*DOCKER, "start", CID, timeout=120)


def recover():
    before = limits()
    preflight(restart=True)
    initial_boot = CGROUP.stat().st_mtime_ns
    capacity = wait_healthy(initial_boot)
    if capacity < CAPACITY:
        raise RuntimeError(f"Unexpected smaller KV pool {capacity}; automatic changes require manual review.")
    if capacity > CAPACITY:
        print(f"Draining {capacity} slots for restart of the original dsv41 container.", flush=True)
        while True:
            snapshot = drain()
            # Revalidate credentials and inventory before committing to downtime.
            preflight(restart=True)
            limits()
            current = status()
            active_http, total = activity_metrics(get("/metrics").decode())
            if current <= CAPACITY or (not active_http and total == snapshot and is_idle(parse_loads(json.loads(get("/v1/loads"))))):
                break
        if current > CAPACITY:
            active = unit_state()
            if active not in ("inactive", "failed", "active"):
                raise RuntimeError(f"F2Pool state changed to {active}; retry when stable.")
            if active == "active":
                run(SUDO, "-n", "--", SYSTEMCTL, "stop", UNIT)
            old = CGROUP.stat().st_ino
            since = time.time_ns()
            try:
                run(*DOCKER, "restart", CID, timeout=180)
                await_weights(since / 1e9, old)
                apply_limits()
                current = wait_healthy(since)
            finally:
                ensure_running()
        capacity = current
    if capacity != CAPACITY:
        raise RuntimeError(f"The serving original reports {capacity} slots; expected {CAPACITY}. Paid workers stay off.")
    if not paid.healthy():
        raise RuntimeError("The original Docker health check must pass.")
    # Cover an externally recovered cgroup with an already-correct pool, too.
    apply_limits()
    if status() != CAPACITY:
        raise RuntimeError("The original KV capacity changed before paid mining startup.")
    run("/usr/bin/python3", "-I", str(HERE / "restore_paid_mining.py"), "--apply", timeout=240)
    if unit_state() != "active":
        raise RuntimeError("The original F2Pool supervisor has not entered active state.")
    result = {"utc": time.time(), "container": CID, "scope": paid.NAMESPACE, "effective_slots_each": capacity,
              "hard_limit_each_mib": paid.LIMIT, "gpus": paid.EXPECTED, "paid_supervisor": "active",
              "caps_already_at_entry": before["prior_policy_met"]}
    output = HERE / "operator-recovery.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2)+"\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--apply", action="store_true", help="authenticate, drain and restart only when oversized, "
                     "restore historical per-card quotas and KV size, then start the normal F2Pool supervisor")
    args = cli.parse_args()
    if os.geteuid() == 0:
        cli.error("Run this helper as the ordinary operator account. It uses sudo only for the original admin commands.")
    if not args.apply:
        state = limits()
        capacity = status() if paid.healthy() else None
        print(json.dumps({"mode":"inspection", "container":CID, "expected_slots":CAPACITY,
                          "current_slots_per_rank":capacity, "original_quota_on_all_cards":state["prior_policy_met"],
                          "restart_needed":None if capacity is None else capacity > CAPACITY, "paid_supervisor":unit_state(),
                          "apply_action":"authenticate, drain and restart an oversized pool, restore the four original "
                                         "quotas during weight loading, verify serving and start paid mining"}, indent=2))
        return
    lock = HERE / "original-deployment-recovery.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover()


if __name__ == "__main__":
    main()
