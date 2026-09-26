#!/usr/bin/env python3
"""One diagnostic-only rollout after 60 continuous idle seconds; no load tests.

Run arm, then watch in a lingering user-systemd service. Sources and container
identity are frozen. Never retry or automatically roll back a started candidate.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from dsv41_deferred_relaunch import (
    CONFIG, QuietWindow, activity_metrics, atomic_json, check_identity, digest,
    fetch, identity, inspect, mem_available, projected_spare_ram,
    roundtrip_profile, run, validate_flags, verify_budget_logs,
)
from dsv41_load import parse_loads


DIAGNOSTICS = {
    "TORCH_NCCL_EXTRA_DUMP_ON_EXEC": "1",
    "SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS": "60",
    "SGLANG_PYSPY_DUMP_TIMEOUT_SECS": "5",
    "SGLANG_PYSPY_DUMP_TOTAL_TIMEOUT_SECS": "20",
    "SGLANG_PYSPY_DUMP_NATIVE": "1",
    "SGLANG_NCCL_DUMP_BEFORE_CRASH": "1",
    "SGLANG_NCCL_DUMP_BEFORE_CRASH_WAIT_SECS": "10",
    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS": "60",
}
VERSION = "dsv41-v2-diagnostics-v1"
SGLANG = Path("/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay")
CAPTURE = Path("/mnt/hot/dsv41_dumps/current")


def validate_only_diagnostics(old, new):
    old, new = roundtrip_profile(old), roundtrip_profile(new)
    if set(new["services"]) != {"deepseek"}:
        raise ValueError("Only dsv41 may be recreated")
    a, b = old["services"]["deepseek"], new["services"]["deepseek"]
    if b["container_name"] != "dsv41" or b["restart"] != "unless-stopped":
        raise ValueError("Unexpected identity or restart policy")
    if b["labels"]["dev.idle.deployment.version"] != VERSION:
        raise ValueError("Unexpected diagnostic version")
    b["image"] = a["image"]
    b["labels"]["dev.idle.deployment.version"] = a["labels"]["dev.idle.deployment.version"]
    for key, value in DIAGNOSTICS.items():
        if b["environment"].get(key) != value:
            raise ValueError(f"Unexpected diagnostic setting: {key}")
        b["environment"].pop(key)
        a["environment"].pop(key, None)
    if new != old:
        raise ValueError("Non-diagnostic deployment configuration changed")


def logs(container, *, startup_only=False):
    command = ["docker", "logs", "--since", container["State"]["StartedAt"]]
    if startup_only:
        started = dt.datetime.fromisoformat(container["State"]["StartedAt"].replace("Z", "+00:00"))
        command += ["--until", (started + dt.timedelta(minutes=12)).isoformat()]
    return subprocess.check_output(command + [container["Id"]], stderr=subprocess.STDOUT, text=True, timeout=45)


def current_allocation(container):
    # The old Docker log spans multiple launches and is several GB. The recorder
    # already split it by init PID; read just its eight startup budget records.
    captured = json.loads((CAPTURE / "wire/config.json").read_text())["deployment"]
    if (captured["id"], captured["started_at"], captured["image_id"]) != (
        container["Id"], container["State"]["StartedAt"], container["Image"]
    ):
        raise RuntimeError("Capture allocation evidence belongs to another launch")
    lines = run("rg", "-m", "8", r"DSV41_HICACHE_(BUDGET|ALLOCATED) ", str(CAPTURE / "dsv4-native.log"))
    return verify_budget_logs("\n".join(line.partition(" ")[2] for line in lines.splitlines()))


def arm(state, receipt_path):
    if (state / "manifest.json").exists():
        raise ValueError("Already armed; use a new private directory")
    receipt = json.loads(receipt_path.read_text())
    current = inspect()
    if current["State"]["Status"] != "running" or current["Image"] != receipt["parent_image_id"]:
        raise ValueError("Live container is not the tested parent")
    if not receipt["parent_layers_preserved"] or not receipt["built_image_cpu_tests_passed"]:
        raise ValueError("Candidate lacks build/test qualification")
    old_path = Path(current["Config"]["Labels"]["com.docker.compose.project.config_files"])
    old = roundtrip_profile(json.loads(old_path.read_text()))
    new = roundtrip_profile(json.loads(run("docker", "compose", "-f", str(CONFIG), "config", "--format", "json")))
    validate_only_diagnostics(old, new)
    service = new["services"]["deepseek"]
    for tag, expected in ((old["services"]["deepseek"]["image"], current["Image"]),
                          (service["image"], receipt["image_id"])):
        if run("docker", "image", "inspect", "--format", "{{.Id}}", tag).strip() != expected:
            raise ValueError("Image tag changed from tested identity")
    live_env = dict(entry.split("=", 1) for entry in current["Config"]["Env"])
    for key, value in old["services"]["deepseek"]["environment"].items():
        if live_env.get(key) != value:
            raise ValueError(f"Saved previous profile differs from live environment: {key}")
    for rel, expected in receipt["existing_readonly_mount_source_sha256"].items():
        if digest(SGLANG / rel) != expected:
            raise ValueError(f"Tested runtime bind changed: {rel}")
    info = json.loads(fetch("/server_info"))
    validate_flags(info, True)
    allocation = current_allocation(current)
    # Immutable references: no tag lookup race at recreation or later rollback.
    old["services"]["deepseek"]["image"] = current["Image"]
    service["image"] = receipt["image_id"]
    atomic_json(state / "rollback.compose.json", old)
    atomic_json(state / "candidate.compose.json", new)
    for path in (state / "rollback.compose.json", state / "candidate.compose.json"):
        run("docker", "compose", "-f", str(path), "config", "--quiet")
    atomic_json(state / "before.container.private.json", current)
    atomic_json(state / "before.server-info.private.json", info)
    atomic_json(state / "before.allocation.json", allocation)
    paths = {CONFIG, receipt_path, old_path, Path(__file__).resolve(),
             Path(__file__).with_name("dsv41_load.py"), Path(__file__).with_name("dsv41_deferred_relaunch.py"),
             state / "candidate.compose.json", state / "rollback.compose.json"}
    paths.update(Path(m["source"]) for m in service["volumes"]
                 if m["type"] == "bind" and Path(m["source"]).is_file())
    manifest = {
        "armed_at": time.time(), "expected_identity": identity(current),
        "image_id": receipt["image_id"], "image_tag": receipt["image_tag"],
        "receipt": str(receipt_path), "idle_seconds": 60, "poll_seconds": 5,
        "previous_cache_bytes": allocation["actual_buffer_bytes_total"],
        "capture_before": str(CAPTURE.resolve()),
        "file_hashes": {str(p): digest(p) for p in paths},
    }
    atomic_json(state / "manifest.json", manifest)
    atomic_json(state / "status.json", {"state": "armed", "updated_at": time.time()})
    print(f"Armed: {state}", flush=True)


def check_files(manifest):
    if any(digest(path) != expected for path, expected in manifest["file_hashes"].items()):
        raise RuntimeError("Sources changed while waiting; rollout cancelled")
    if run("docker", "image", "inspect", "--format", "{{.Id}}", manifest["image_tag"]).strip() != manifest["image_id"]:
        raise RuntimeError("Candidate tag changed; rollout cancelled")


def verify_started(state, manifest, current, text):
    info = json.loads(fetch("/server_info"))
    validate_flags(info, True)
    allocation = verify_budget_logs(text)
    live_env = dict(entry.split("=", 1) for entry in current["Config"]["Env"])
    if any(live_env.get(key) != value for key, value in DIAGNOSTICS.items()):
        raise RuntimeError("Effective diagnostic environment mismatch")
    if current["HostConfig"]["RestartPolicy"]["Name"] != "unless-stopped":
        raise RuntimeError("Unexpected restart policy")
    receipt = json.loads(Path(manifest["receipt"]).read_text())
    hashes = {"/sgl-workspace/sglang/" + path: receipt["source_sha256"][path] for path in receipt["runtime_files"]}
    actual = json.loads(run("docker", "exec", current["Id"], "python3", "-c",
        "import hashlib,json,pathlib,sys; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in json.loads(sys.argv[1])}))",
        json.dumps(list(hashes))))
    if actual != hashes:
        raise RuntimeError("Running diagnostic files differ from tested snapshot")
    marker = "Persistent CUDA/NCCL diagnostics: "
    paths = [Path(line.split(marker, 1)[1]) for line in text.splitlines() if marker in line]
    if len(paths) != 1 or paths[0].parent != Path("/mnt/hot/dsv41_dumps/diagnostics"):
        raise RuntimeError("Missing or ambiguous current diagnostic directory")
    # Container-root owns these private directories. Inspect through docker exec
    # without weakening permissions or writing to either kind of trigger pipe.
    diagnostic_state = json.loads(run("docker", "exec", current["Id"], "python3", "-c",
        "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); print(json.dumps({'receipt':json.loads((p/'diagnostics.json').read_text()),'nccl_fifos':[r for r in range(4) if (p/f'torch_nccl_pipe_rank_{r}.pipe').is_fifo()]}))",
        str(paths[0])))
    diagnostic = diagnostic_state["receipt"]
    if any(diagnostic["settings"].get(k) != v for k, v in DIAGNOSTICS.items()):
        raise RuntimeError("Boot did not record expected diagnostic settings")
    if diagnostic_state["nccl_fifos"] != list(range(4)):
        raise RuntimeError("Missing NCCL flight-recorder control FIFO")
    capture = json.loads((CAPTURE / "wire/status.json").read_text())
    heartbeat = dt.datetime.fromisoformat(capture["heartbeat_utc"].replace("Z", "+00:00")).timestamp()
    if str(CAPTURE.resolve()) == manifest["capture_before"] or capture["state"] != "running" or not -5 <= time.time() - heartbeat <= 30:
        raise RuntimeError("Capture did not follow new container with a fresh heartbeat")
    if run("systemctl", "--user", "is-active", "dsv41-wire-dump.service").strip() != "active":
        raise RuntimeError("Capture service is not active")
    for name, value in (("after.container.private", current), ("after.server-info.private", info),
                        ("allocation", allocation), ("capture", capture), ("diagnostics", diagnostic)):
        atomic_json(state / f"{name}.json", value)
    return {"container_id": current["Id"], "image_id": current["Image"],
            "started_at": current["State"]["StartedAt"], "restart_count": current["RestartCount"],
            "capture": str(CAPTURE.resolve()), "diagnostics": str(paths[0]),
            "host_cache_actual_bytes": allocation["actual_buffer_bytes_total"], "available_bytes": mem_available()}


def watch(state):
    manifest = json.loads((state / "manifest.json").read_text())
    if (state / "attempt.json").exists():
        raise RuntimeError("Rollout already attempted; no automatic repeat")
    quiet = QuietWindow(manifest["idle_seconds"])

    def status(stage, **details):
        atomic_json(state / "status.json", {"state": stage, "updated_at": time.time(), **details})

    def sample():
        rows = parse_loads(json.loads(fetch("/v1/loads")))
        metrics = fetch("/metrics")
        ready = quiet.observe(rows, metrics, time.monotonic())
        summary = {"running": sum(r["num_running_reqs"] for r in rows),
                   "waiting": sum(r["num_waiting_reqs"] for r in rows),
                   "http_active": activity_metrics(metrics)[0],
                   "idle_seconds": 0 if quiet.window.since is None else time.monotonic() - quiet.window.since}
        with (state / "drain.jsonl").open("a") as stream:
            stream.write(json.dumps({"at": time.time(), "ready": ready, **summary}) + "\n")
        status("waiting_for_idle", **summary)
        return ready

    try:
        while True:
            check_identity(manifest)
            check_files(manifest)
            try:
                drained = sample()
            except Exception as exc:
                quiet.reset()
                status("waiting_for_valid_load", error=type(exc).__name__)
                drained = False
            if drained:
                spare = projected_spare_ram(mem_available(), manifest["previous_cache_bytes"])
                if spare < 24 * 2**30:
                    quiet.reset()
                    status("waiting_for_ram", projected_spare_bytes=spare)
                else:
                    # No slow I/O between final fresh observation and recreate.
                    check_identity(manifest)
                    check_files(manifest)
                    time.sleep(1)
                    try:
                        if sample():
                            break
                    except Exception:
                        quiet.reset()
            time.sleep(manifest["poll_seconds"])
        atomic_json(state / "attempt.json", {"at": time.time(), "image_id": manifest["image_id"], "projected_spare_bytes": spare})
        status("relaunching")
        with (state / "relaunch.log").open("x") as output:
            subprocess.run(["docker", "compose", "-f", str(state / "candidate.compose.json"),
                            "up", "-d", "--no-build", "--pull", "never", "--force-recreate", "deepseek"],
                           stdout=output, stderr=subprocess.STDOUT, check=True, timeout=240)
        after = inspect()
        if after["Id"] == manifest["expected_identity"][0] or after["Image"] != manifest["image_id"]:
            raise RuntimeError("Unexpected recreated identity")
        deadline = time.monotonic() + 2400
        while True:
            current = inspect()
            if current["Id"] != after["Id"] or current["RestartCount"] != 0 or current["State"]["Status"] != "running":
                raise RuntimeError("Candidate changed, exited or restarted during startup")
            text = logs(current)
            (state / "startup.private.log").write_text(text)
            if "Ready: deepseek-v4-flash on port 8000; no backend API key" in text:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Startup checks not complete within 40 minutes")
            status("startup_checks_pending", container_id=current["Id"])
            time.sleep(10)
        result = verify_started(state, manifest, current, text)
        atomic_json(state / "complete.json", {"at": time.time(), **result})
        status("complete", **result)
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}", attempt_exists=(state / "attempt.json").exists())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("arm", "watch"))
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--image-receipt", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    state = args.state_dir.resolve(strict=True)
    if state.parent != Path("/mnt/hot/dsv41_state") or not state.name.startswith("diagnostics-rollout-"):
        parser.error("Use a fresh mktemp-created /mnt/hot/dsv41_state/diagnostics-rollout-* directory")
    with (state / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "arm":
            if not args.image_receipt:
                parser.error("arm needs --image-receipt")
            arm(state, args.image_receipt.resolve(strict=True))
        else:
            watch(state)


if __name__ == "__main__":
    main()
