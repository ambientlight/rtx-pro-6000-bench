#!/usr/bin/env python3
"""One-shot, fail-closed dsv41 RAM-HiCache rollout after 60 seconds idle.

arm snapshots identities/config; watch is intended for a lingering user-systemd
service. It sends no inference requests, never flushes KV, and never force-kills
requests. Docker's existing graceful stop and restart policy remain unchanged.
"""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from dsv41_load import DrainWindow, is_idle, parse_loads


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docker/deepseek-v41/compose.api.yaml"
BUDGET = 256_000_000_000
# User approved the tighter 256 GB profile; match boot's existing 24 GiB floor.
SPARE_RAM_BYTES = 24 * 1024**3
BASE_URL = "http://127.0.0.1:8000"
INFERENCE = {"/v1/messages", "/v1/responses", "/v1/chat/completions", "/v1/completions", "/generate", "/encode"}
COUNTER = re.compile(r"^sglang:http_requests_(active|total)\{([^}]+)\}\s+([^ ]+)(?:\s+\d+)?$")


def run(*command, timeout=30):
    return subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=timeout)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fetch(path):
    with urllib.request.urlopen(BASE_URL + path, timeout=10) as response:
        return response.read().decode()


def inspect():
    return json.loads(run("docker", "inspect", "dsv41"))[0]


def identity(container):
    return (container["Id"], container["State"]["StartedAt"], container["RestartCount"], container["Image"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roundtrip_profile(profile):
    result = json.loads(json.dumps(profile))
    for service in result["services"].values():
        # Compose config JSON drops zero soft/hard values, but {} is not valid
        # Compose input. Restore the explicitly disabled CPU core-dump limit.
        if service.get("ulimits", {}).get("core") == {}:
            service["ulimits"]["core"] = {"soft": 0, "hard": 0}
    return result


def activity_metrics(text):
    values = {"active": {}, "total": {}}
    for line in text.splitlines():
        match = COUNTER.fullmatch(line)
        if not match:
            continue
        kind, labels, number = match.groups()
        labels = dict(re.findall(r'(\w+)="([^"]*)"', labels))
        endpoint = labels.get("endpoint")
        if endpoint not in INFERENCE:
            continue
        value = float(number)
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError("Invalid HTTP activity counter")
        key = (endpoint, labels.get("method"))
        if key in values[kind]:
            raise ValueError("Duplicate HTTP activity counter")
        values[kind][key] = int(value)
    if not values["active"] or values["active"].keys() != values["total"].keys():
        raise ValueError("Missing or incomplete HTTP inference metrics")
    return sum(values["active"].values()), tuple(sorted(values["total"].items()))


class QuietWindow:
    def __init__(self, seconds=60):
        self.seconds = seconds
        self.reset()

    def reset(self):
        self.window = DrainWindow(self.seconds, {0})
        self.previous = None

    def observe(self, rows, metrics, now):
        active, fingerprint = activity_metrics(metrics)
        changed = self.previous is not None and fingerprint != self.previous
        if active or changed or not is_idle(rows):
            self.reset()
            self.previous = fingerprint
            return False
        self.previous = fingerprint
        return self.window.observe(rows, now)


def mem_available():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) * 1024


def validate_flags(info, enabled):
    flags = info
    required = {
        "enable_hierarchical_cache": enabled,
        "context_length": 524288,
        "mem_fraction_static": 0.87,
        "chunked_prefill_size": 2048,
        "prefill_decode_interval": 4,
        "max_running_requests": 8,
        "tp_size": 4,
        "ep_size": 4,
        "served_model_name": "deepseek-v4-flash",
        "api_key": None,
        "hicache_storage_backend": None,
        "speculative_algorithm": "DSPARK",
        "enable_decoder_swa_bounded_replay": True,
        "enable_encoder_swa_bounded_replay": False,
    }
    if enabled:
        required.update(
            hicache_host_memory_mode="cache",
            hicache_write_policy="write_through",
            hicache_io_backend="kernel",
            hicache_mem_layout="page_first",
        )
    for key, value in required.items():
        if key not in flags or flags[key] != value:
            raise ValueError(f"Unexpected effective {key}; expected {value!r}")


def verify_budget_logs(log, budget_bytes=BUDGET):
    plans, allocations = {}, {}
    for marker, target in (("DSV41_HICACHE_BUDGET ", plans), ("DSV41_HICACHE_ALLOCATED ", allocations)):
        for line in log.splitlines():
            match = re.fullmatch(r"(?:\[[^]\n]+\] )?" + re.escape(marker) + r"(\{.*\})", line)
            if match:
                item = json.loads(match[1])
                if item["rank"] in target:
                    raise ValueError("Repeated cache initialization; restart or unexpected extra pool")
                target[item["rank"]] = item
        if set(target) != set(range(4)):
            raise ValueError("Missing HiCache allocation evidence for one or more ranks")
    for rank, plan in plans.items():
        if plan["host_budget_bytes"] != budget_bytes or plan["planned_bytes_host_total"] > budget_bytes:
            raise ValueError("Invalid HiCache budget plan")
        for key in ("full_host_pages", "swa_host_pages", "tensor_bytes_per_rank"):
            if plan[key] != plans[0][key]:
                raise ValueError("Asymmetric HiCache capacity across ranks")
        if allocations[rank]["host_budget_bytes"] != budget_bytes:
            raise ValueError("Wrong actual HiCache budget")
    actual = sum(item["bytes_per_rank"] for item in allocations.values())
    if not 0 < actual <= budget_bytes:
        raise ValueError("Actual HiCache buffers exceed budget")
    return {"plans": plans, "allocations": allocations, "actual_buffer_bytes_total": actual}


def validate_cache_only_change(before, service):
    """Permit initial enablement or a budget change; preserve all other fields."""
    expected = json.loads(json.dumps(before))
    actual = json.loads(json.dumps(service))
    previous_budget = int(expected["environment"].pop("DSV41_HICACHE_HOST_BUDGET_BYTES", "0"))
    actual["environment"].pop("DSV41_HICACHE_HOST_BUDGET_BYTES")
    if previous_budget == 0:
        before_targets = {mount["target"] for mount in expected["volumes"]}
        added = [mount for mount in actual["volumes"] if mount["target"] not in before_targets]
        if {Path(m["target"]).name for m in added} != {"hybrid_pool_assembler.py", "dsv41_host_budget.py"}:
            raise ValueError("Unexpected additional runtime mounts")
        actual["volumes"] = [mount for mount in actual["volumes"] if mount["target"] in before_targets]
    if actual != expected:
        raise ValueError("Non-HiCache deployment configuration changed")
    if not 0 <= previous_budget <= BUDGET:
        raise ValueError("Unexpected previous cache budget")
    return previous_budget


def projected_spare_ram(available_bytes, previous_cache_bytes, target_budget=BUDGET):
    # Credit only verified, actually allocated old buffers that die with this
    # exact container. Never credit unallocated portions of the old budget.
    if min(available_bytes, previous_cache_bytes) < 0:
        raise ValueError("Invalid RAM accounting inputs")
    return available_bytes + previous_cache_bytes - target_budget


def notify(title, message, priority=3):
    # notify skill: only deployment status is sent; never prompts or credentials.
    try:
        run(
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "10",
            "--max-time",
            "20",
            "--retry",
            "2",
            "-H",
            "Title: " + title,
            "-H",
            f"Priority: {priority}",
            "--data-binary",
            message,
            "https://code.idle.dev/cc_status_ambientlight",
            timeout=75,
        )
        return True
    except Exception:
        return False


def arm(state):
    if (state / "manifest.json").exists():
        raise ValueError("Already armed; use a new private state directory")
    rollback = state / "rollback.compose.json"
    old = roundtrip_profile(json.loads(rollback.read_text()))
    if set(old["services"]) != {"deepseek"}:
        raise ValueError("Invalid rollback profile")
    current = inspect()
    if current["State"]["Status"] != "running":
        raise ValueError("dsv41 is not running")
    info = json.loads(fetch("/server_info"))
    candidate = roundtrip_profile(json.loads(run("docker", "compose", "-f", str(CONFIG), "config", "--format", "json")))
    service = candidate["services"]["deepseek"]
    if set(candidate["services"]) != {"deepseek"} or service["container_name"] != "dsv41":
        raise ValueError("Rollout must target only dsv41")
    if service["environment"].get("DSV41_HICACHE_HOST_BUDGET_BYTES") != str(BUDGET):
        raise ValueError(f"Expected {BUDGET // 10**9} GB host-total HiCache budget")
    if service["restart"] != "unless-stopped":
        raise ValueError("Must preserve restart policy")
    before = old["services"]["deepseek"]
    previous_budget = validate_cache_only_change(before, service)
    live_env = dict(entry.split("=", 1) for entry in current["Config"]["Env"])
    if int(live_env.get("DSV41_HICACHE_HOST_BUDGET_BYTES", "0")) != previous_budget:
        raise ValueError("Rollback profile does not match the live cache budget")
    validate_flags(info, bool(previous_budget))
    old_cache_bytes = 0
    if previous_budget:
        old_allocation = verify_budget_logs(run("docker", "logs", current["Id"]), previous_budget)
        old_cache_bytes = old_allocation["actual_buffer_bytes_total"]
        atomic_json(state / "before.allocation.json", old_allocation)
    expected_image = run("docker", "image", "inspect", "--format", "{{.Id}}", service["image"]).strip()
    if current["Image"] != expected_image:
        raise ValueError("Staged image differs from running image")
    paths = {
        str(CONFIG),
        str(Path(__file__).resolve()),
        str(Path(__file__).with_name("dsv41_load.py").resolve()),
        str(rollback),
    }
    paths.update(m["source"] for m in service["volumes"] if m["type"] == "bind" and Path(m["source"]).is_file())
    atomic_json(state / "candidate.compose.json", candidate)
    atomic_json(rollback, old)
    # Validate serialized INPUT, not only the source configuration. Read-only.
    for path in (state / "candidate.compose.json", rollback):
        run("docker", "compose", "-f", str(path), "config", "--quiet")
    paths.add(str(state / "candidate.compose.json"))
    atomic_json(state / "before.container.private.json", current)
    atomic_json(state / "before.server-info.private.json", info)
    atomic_json(
        state / "manifest.json",
        {
            "armed_at": time.time(),
            "expected_identity": identity(current),
            "image_tag": service["image"],
            "file_hashes": {p: digest(p) for p in sorted(paths)},
            "idle_seconds": 60,
            "poll_seconds": 5,
            "budget_bytes": BUDGET,
            "previous_budget_bytes": previous_budget,
            "previous_cache_bytes": old_cache_bytes,
            "minimum_spare_ram_bytes": SPARE_RAM_BYTES,
            "capture_before": str(Path("/mnt/hot/dsv41_dumps/current").resolve()),
        },
    )
    atomic_json(state / "status.json", {"state": "armed", "updated_at": time.time()})
    print(f"Armed: {state}", flush=True)


def check_identity(manifest):
    current = inspect()
    if list(identity(current)) != manifest["expected_identity"] or current["State"]["Status"] != "running":
        raise RuntimeError("Original container changed or restarted; deferred rollout cancelled")


def check_files(manifest):
    if any(digest(path) != expected for path, expected in manifest["file_hashes"].items()):
        raise RuntimeError("Staged source/config changed; deferred rollout cancelled")
    if (
        run("docker", "image", "inspect", "--format", "{{.Id}}", manifest["image_tag"]).strip()
        != manifest["expected_identity"][3]
    ):
        raise RuntimeError("Image tag changed; deferred rollout cancelled")


def watch(state):
    manifest = json.loads((state / "manifest.json").read_text())
    if (state / "attempt.json").exists():
        raise RuntimeError("A relaunch was already attempted; refusing automatic repetition")
    quiet = QuietWindow(manifest["idle_seconds"])

    def status(stage, **details):
        atomic_json(state / "status.json", {"state": stage, "updated_at": time.time(), **details})

    def sample():
        rows = parse_loads(json.loads(fetch("/v1/loads")))
        metrics = fetch("/metrics")
        ready = quiet.observe(rows, metrics, time.monotonic())
        summary = {
            "running": sum(r["num_running_reqs"] for r in rows),
            "waiting": sum(r["num_waiting_reqs"] for r in rows),
            "active_tokens": sum(r["num_active_tokens"] for r in rows),
            "http_active": activity_metrics(metrics)[0],
            "idle_seconds": 0 if quiet.window.since is None else time.monotonic() - quiet.window.since,
        }
        with (state / "drain.jsonl").open("a") as output:
            output.write(json.dumps({"time": time.time(), "ready": ready, **summary}) + "\n")
        status("waiting_for_idle", **summary)
        return ready

    try:
        check_files(manifest)
        while True:
            check_identity(manifest)
            try:
                drained = sample()
            except Exception as exc:
                quiet.reset()
                status("waiting_for_valid_load", error=type(exc).__name__)
                drained = False
            if drained:
                check_files(manifest)
                check_identity(manifest)
                # Engram is resident; old HiCache buffers are released on stop.
                spare = projected_spare_ram(
                    mem_available(), manifest.get("previous_cache_bytes", 0), manifest["budget_bytes"]
                )
                if spare < manifest.get("minimum_spare_ram_bytes", SPARE_RAM_BYTES):
                    quiet.reset()
                    status("waiting_for_ram", available_bytes=mem_available(), projected_spare_bytes=spare)
                else:
                    others = set(run("docker", "ps", "--format", "{{.Names}}").splitlines())
                    if others & {"dsv4", "dsv41-api", "dsv41-baseline", "qwen3-embed"}:
                        raise RuntimeError("Another scoped GPU model is running; rollout cancelled")
                    # Fresh last-moment observation. New activity resets the timer.
                    time.sleep(1)
                    if sample():
                        break
            time.sleep(manifest["poll_seconds"])
        check_identity(manifest)
        check_files(manifest)
        atomic_json(state / "attempt.json", {"started_at": time.time()})
        status("relaunching")
        # No network notification between the final idle check and recreation.
        with (state / "relaunch.log").open("x") as output:
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(state / "candidate.compose.json"),
                    "up",
                    "-d",
                    "--no-build",
                    "--pull",
                    "never",
                    "--force-recreate",
                    "deepseek",
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=240,
            )
        candidate = inspect()
        candidate_id = candidate["Id"]
        if candidate_id == manifest["expected_identity"][0] or candidate["Image"] != manifest["expected_identity"][3]:
            raise RuntimeError("Unexpected recreated container identity")
        notify(
            "DSV41 HiCache rollout",
            f"60 seconds idle confirmed; dsv41 is starting with the {manifest['budget_bytes'] // 10**9} GB total RAM HiCache budget.",
        )
        deadline = time.monotonic() + 2400
        while time.monotonic() < deadline:
            current = inspect()
            if current["Id"] != candidate_id or current["RestartCount"] != 0 or current["State"]["Status"] != "running":
                raise RuntimeError("Candidate changed, exited or restarted before verification")
            logs = run("docker", "logs", candidate_id, timeout=30)
            if "Ready: deepseek-v4-flash on port 8000; no backend API key" in logs:
                break
            status("startup_checks_pending", container_id=candidate_id)
            time.sleep(10)
        else:
            raise RuntimeError("Candidate did not pass boot checks within 40 minutes")
        (state / "startup.private.log").write_text(logs)
        info = json.loads(fetch("/server_info"))
        validate_flags(info, True)
        allocation = verify_budget_logs(logs, manifest["budget_bytes"])
        if current["HostConfig"]["RestartPolicy"]["Name"] != "unless-stopped":
            raise RuntimeError("Unexpected restart policy after recreation")
        capture = Path("/mnt/hot/dsv41_dumps/current/wire/status.json")
        capture_state = json.loads(capture.read_text())
        if str(capture.parents[1].resolve()) == manifest["capture_before"]:
            raise RuntimeError("Capture has not followed the new container")
        heartbeat = dt.datetime.fromisoformat(capture_state["heartbeat_utc"].replace("Z", "+00:00")).timestamp()
        if capture_state["state"] != "running" or not -5 <= time.time() - heartbeat <= 30:
            raise RuntimeError("Capture heartbeat is stale or recorder is not running")
        if run("systemctl", "--user", "is-active", "dsv41-wire-dump.service").strip() != "active":
            raise RuntimeError("Capture service is not active")
        atomic_json(state / "after.server-info.private.json", info)
        atomic_json(state / "after.container.private.json", current)
        atomic_json(state / "allocation.json", allocation)
        atomic_json(state / "capture.json", capture_state)
        sent = notify(
            "DSV41 HiCache ready",
            f"dsv41 relaunched after 60 seconds idle. RAM-only HiCache enabled within the {manifest['budget_bytes'] // 10**9} GB total budget on all 4 ranks; normal startup checks passed. Context 524288, static fraction 0.87, interval 4 unchanged.",
        )
        status("complete", container_id=candidate_id, allocation=allocation, notification_sent=sent)
    except Exception as exc:
        # Do not stop/roll back a candidate that may already have new user work.
        # Saved rollback config is available to the operator; no retry loop here.
        if (state / "attempt.json").exists():
            try:
                with (state / "failed-startup.private.log").open("w") as output:
                    subprocess.run(
                        ["docker", "logs", "--tail", "15000", "dsv41"],
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=30,
                        check=False,
                    )
                atomic_json(state / "failed.container.private.json", inspect())
            except Exception:
                pass
        status("failed", error=f"{type(exc).__name__}: {exc}", notification_pending=True)
        sent = notify(
            "DSV41 deferred rollout needs attention",
            f"Rollout stopped: {type(exc).__name__}: {exc}. No automatic second relaunch. Evidence: {state}",
            4,
        )
        status("failed", error=f"{type(exc).__name__}: {exc}", notification_sent=sent)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("arm", "watch"))
    parser.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    state = args.state_dir.resolve(strict=True)
    if state.parent != Path("/mnt/hot/dsv41_state") or not state.name.startswith("deferred-hicache-"):
        parser.error("Use a dedicated mktemp-created /mnt/hot/dsv41_state/deferred-hicache-* directory")
    with (state / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"arm": arm, "watch": watch}[args.command](state)


if __name__ == "__main__":
    main()
