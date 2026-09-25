#!/usr/bin/env python3
"""One RAM-payload rebalance after 60 idle seconds; no synthetic load test.

prepare freezes the live profile and Python bind sources for rollback, before
source edits. arm freezes the tested candidate. watch never retries a rollout
or automatically rolls back. The model image and inference settings stay fixed.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from dsv41_deferred_relaunch import (
    CONFIG, QuietWindow, activity_metrics, atomic_json, check_identity, digest,
    fetch, identity, inspect, mem_available, projected_spare_ram,
    roundtrip_profile, run, validate_flags, verify_budget_logs,
)
from dsv41_load import parse_loads
from dsv41_relaunch_diagnostics import CAPTURE, current_allocation, logs


FRACTION_KEY = "DSV41_HICACHE_SWA_FRACTION"
FRACTION = "0.5"
VERSION = "dsv41-v2-hicache-split-v2"
PROFILES = {
    "historical-50": (FRACTION, VERSION),
    "l15-67": ("0.67", "dsv41-v2-cache-attribution-v1"),
}
L13_IMAGE = "sha256:fcd6ae8574240c6f0df3c6277b477234328a028007ef7e97acdcb9ed1db1349c"
L13_PROFILE = Path("/mnt/hot/dsv41_state/cache-attribution-rollout-rg2rbn1u/candidate.compose.json")


def profile_settings(profile):
    if profile not in PROFILES:
        raise ValueError("Unapproved cache-split profile")
    return PROFILES[profile]


def freeze_sources(profile, directory):
    result = roundtrip_profile(profile)
    hashes = {}
    for mount in result["services"]["deepseek"]["volumes"]:
        source = Path(mount["source"])
        if mount["type"] != "bind" or source.suffix != ".py":
            continue
        if not mount.get("read_only") or not source.is_file():
            raise ValueError("Expected a read-only Python source bind")
        target = directory / Path(mount["target"]).relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[mount["target"]] = digest(target)
        mount["source"] = str(target)
    if not hashes:
        raise ValueError("Missing runtime Python bind sources")
    return result, hashes


def validate_change(old, new, profile="historical-50"):
    fraction, version = profile_settings(profile)
    old, new = roundtrip_profile(old), roundtrip_profile(new)
    if set(old["services"]) != {"deepseek"} or set(new["services"]) != {"deepseek"}:
        raise ValueError("Only dsv41 may be recreated")
    a, b = old["services"]["deepseek"], new["services"]["deepseek"]
    if b["container_name"] != "dsv41" or b["restart"] != "unless-stopped":
        raise ValueError("Unexpected service identity or restart policy")
    if b["environment"].get(FRACTION_KEY) != fraction:
        raise ValueError(f"Expected exactly {float(fraction) * 100:g}% SWA payload")
    if b["environment"].get("DSV41_HICACHE_HOST_BUDGET_BYTES") != "256000000000":
        raise ValueError("Host total must remain 256 GB")
    if b["labels"].get("dev.idle.deployment.version") != version:
        raise ValueError("Unexpected rollout label")
    if profile == "l15-67" and (
        a["environment"].get(FRACTION_KEY) != "0.6"
        or a["labels"].get("dev.idle.deployment.version") != version
    ):
        raise ValueError("L15 must replace the existing L14 60/40 profile")
    b["environment"].pop(FRACTION_KEY)
    a["environment"].pop(FRACTION_KEY, None)
    b["labels"]["dev.idle.deployment.version"] = a["labels"]["dev.idle.deployment.version"]
    if new != old:
        raise ValueError("Non-cache-split configuration changed")


def verify_fraction(allocation, profile="historical-50"):
    fraction, _ = profile_settings(profile)
    for plan in allocation["plans"].values():
        if plan.get("allocation_mode") != "payload_fraction" or plan.get("requested_swa_payload_fraction") != float(fraction):
            raise ValueError("Requested payload split was not applied")
        full = plan["full_host_pages"] * plan["full_page_bytes"]
        swa = plan["swa_host_pages"] * plan["swa_page_bytes"]
        actual = swa / (full + swa)
        # Whole-page rounding may reduce the share by at most one SWA page.
        if not float(fraction) - plan["swa_page_bytes"] / (full + swa) <= actual <= float(fraction) + 1e-12:
            raise ValueError("Actual payload split does not match requested fraction")
        if plan["full_host_pages"] < plan["device_full_pages"] or plan["swa_host_pages"] < plan["device_swa_pages"]:
            raise ValueError("Host pool cannot hold one device-cache equivalent")
    return actual


def validate_l13_reference(candidate, reference):
    """Require the same image, full profile and Python bytes as archived L13."""
    candidate, reference = roundtrip_profile(candidate), roundtrip_profile(reference)
    for item in (candidate, reference):
        service = item["services"]["deepseek"]
        if service["image"] != L13_IMAGE:
            raise ValueError("L15 requires the immutable L13 image")
        for mount in service["volumes"]:
            if mount["type"] == "bind" and Path(mount["source"]).suffix == ".py":
                # Snapshot locations can differ, their contents cannot.
                mount["source"] = digest(mount["source"])
    if candidate != reference:
        raise ValueError("Candidate differs from the archived L13 configuration or runtime bytes")


def prepare(state):
    if (state / "prepared.json").exists():
        raise ValueError("Already prepared; use a fresh directory")
    current = inspect()
    if current["State"]["Status"] != "running":
        raise ValueError("dsv41 is not running")
    profile = roundtrip_profile(json.loads(run("docker", "compose", "-f", str(CONFIG), "config", "--format", "json")))
    service = profile["services"]["deepseek"]
    if run("docker", "image", "inspect", "--format", "{{.Id}}", service["image"]).strip() != current["Image"]:
        raise ValueError("Source profile no longer describes live image")
    live_env = dict(entry.split("=", 1) for entry in current["Config"]["Env"])
    if any(live_env.get(k) != v for k, v in service["environment"].items()):
        raise ValueError("Source environment differs from live deployment")
    info = json.loads(fetch("/server_info"))
    validate_flags(info, True)
    allocation = current_allocation(current)
    atomic_json(state / "source-profile.json", profile)
    profile["services"]["deepseek"]["image"] = current["Image"]
    rollback, hashes = freeze_sources(profile, state / "rollback-src")
    atomic_json(state / "rollback.compose.json", rollback)
    run("docker", "compose", "-f", str(state / "rollback.compose.json"), "config", "--quiet")
    for name, value in (("before.container.private", current), ("before.server-info.private", info), ("before.allocation", allocation)):
        atomic_json(state / f"{name}.json", value)
    atomic_json(state / "prepared.json", {
        "expected_identity": identity(current), "previous_cache_bytes": allocation["actual_buffer_bytes_total"],
        "capture_before": str(CAPTURE.resolve()), "rollback_source_sha256": hashes,
    })
    print(f"Prepared rollback: {state}", flush=True)


def arm(state, profile="historical-50"):
    if (state / "manifest.json").exists():
        raise ValueError("Already armed")
    prepared = json.loads((state / "prepared.json").read_text())
    check_identity(prepared)
    old = json.loads((state / "source-profile.json").read_text())
    candidate = roundtrip_profile(json.loads(run("docker", "compose", "-f", str(CONFIG), "config", "--format", "json")))
    validate_change(old, candidate, profile)
    tag = candidate["services"]["deepseek"]["image"]
    if run("docker", "image", "inspect", "--format", "{{.Id}}", tag).strip() != prepared["expected_identity"][3]:
        raise ValueError("Image changed")
    sources = [Path(m["source"]) for m in candidate["services"]["deepseek"]["volumes"]
               if m["type"] == "bind" and Path(m["source"]).suffix == ".py"]
    candidate["services"]["deepseek"]["image"] = prepared["expected_identity"][3]
    candidate, hashes = freeze_sources(candidate, state / "candidate-src")
    if profile == "l15-67":
        if hashes != prepared["rollback_source_sha256"]:
            raise ValueError("L15 is configuration-only; runtime source changes are not allowed")
        validate_l13_reference(candidate, json.loads(L13_PROFILE.read_text()))
    atomic_json(state / "candidate.compose.json", candidate)
    run("docker", "compose", "-f", str(state / "candidate.compose.json"), "config", "--quiet")
    paths = {CONFIG, Path(__file__).resolve(), state / "prepared.json", state / "candidate.compose.json", state / "rollback.compose.json", *sources}
    paths.update(Path(__file__).with_name(name) for name in ("dsv41_load.py", "dsv41_deferred_relaunch.py", "dsv41_relaunch_diagnostics.py", "dsv41_cache_monitor.py"))
    for frozen_profile in (candidate, json.loads((state / "rollback.compose.json").read_text())):
        paths.update(Path(m["source"]) for m in frozen_profile["services"]["deepseek"]["volumes"]
                     if m["type"] == "bind" and Path(m["source"]).suffix == ".py")
    atomic_json(state / "manifest.json", {
        **prepared, "armed_at": time.time(), "image_tag": tag, "idle_seconds": 60, "poll_seconds": 5,
        "cache_split_profile": profile, "launch_tag": "L15" if profile == "l15-67" else None,
        "source_sha256": hashes, "file_hashes": {str(p): digest(p) for p in paths},
    })
    atomic_json(state / "status.json", {"state": "armed", "updated_at": time.time()})
    print(f"Armed: {state}", flush=True)


def check_files(manifest):
    if any(digest(path) != expected for path, expected in manifest["file_hashes"].items()):
        raise RuntimeError("Sources changed while waiting; rollout cancelled")
    if run("docker", "image", "inspect", "--format", "{{.Id}}", manifest["image_tag"]).strip() != manifest["expected_identity"][3]:
        raise RuntimeError("Image tag changed")


def verify_started(state, manifest, current, text):
    info = json.loads(fetch("/server_info"))
    validate_flags(info, True)
    allocation = verify_budget_logs(text)
    profile = manifest.get("cache_split_profile", "historical-50")
    share = verify_fraction(allocation, profile)
    live_env = dict(entry.split("=", 1) for entry in current["Config"]["Env"])
    candidate = json.loads((state / "candidate.compose.json").read_text())
    if any(live_env.get(k) != v for k, v in candidate["services"]["deepseek"]["environment"].items()):
        raise RuntimeError("Live environment differs from frozen candidate")
    actual = {}
    for line in run("docker", "exec", current["Id"], "sha256sum", *manifest["source_sha256"]).splitlines():
        sha, path = line.split(maxsplit=1)
        actual[path] = sha
    if actual != manifest["source_sha256"]:
        raise RuntimeError("Running Python binds differ from tested snapshot")
    if current["HostConfig"]["RestartPolicy"]["Name"] != "unless-stopped":
        raise RuntimeError("Restart policy changed")
    capture = json.loads((CAPTURE / "wire/status.json").read_text())
    from datetime import datetime
    heartbeat = datetime.fromisoformat(capture["heartbeat_utc"].replace("Z", "+00:00")).timestamp()
    if str(CAPTURE.resolve()) == manifest["capture_before"] or capture["state"] != "running" or not -5 <= time.time() - heartbeat <= 30:
        raise RuntimeError("Capture has not followed the new container")
    if run("systemctl", "--user", "is-active", "dsv41-wire-dump.service").strip() != "active":
        raise RuntimeError("Capture service not active")
    if profile == "l15-67":
        if current["State"].get("Health", {}).get("Status") != "healthy":
            raise RuntimeError("L15 startup readiness passed but Docker health is not healthy")
        from dsv41_cache_monitor import read_metrics
        metrics = read_metrics(fetch("/metrics"))
        plan = allocation["plans"][0]
        if (metrics["full_capacity"], metrics["swa_capacity"], metrics["diagnostics_enabled"]) != (
            plan["full_host_pages"] * 256, plan["swa_host_pages"] * 256, 1
        ):
            raise RuntimeError("L15 live host capacities or attribution diagnostics differ from startup")
        if run("systemctl", "--user", "is-active", "dsv41-cache-monitor.service").strip() != "active":
            raise RuntimeError("Passive cache monitor is not active")
        atomic_json(state / "metrics.after.json", metrics)
        atomic_json(state / "launch-spec.json", [{
            "tag": manifest.get("launch_tag") or "L15", "launch_utc": current["State"]["StartedAt"],
            "image": "ambientlight/dsv41-sm120 (sha fcd6ae85, SWA_FRACTION=0.67)",
            "dir": CAPTURE.resolve().name,
        }])
    for name, value in (("after.container.private", current), ("after.server-info.private", info), ("allocation", allocation), ("capture", capture)):
        atomic_json(state / f"{name}.json", value)
    return {"launch_tag": manifest.get("launch_tag"), "container_id": current["Id"], "image_id": current["Image"], "started_at": current["State"]["StartedAt"],
            "restart_count": current["RestartCount"], "swa_payload_fraction": share,
            "capture": str(CAPTURE.resolve()), "host_cache_actual_bytes": allocation["actual_buffer_bytes_total"],
            "available_bytes": mem_available()}


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
        details = {"running": sum(r["num_running_reqs"] for r in rows), "waiting": sum(r["num_waiting_reqs"] for r in rows),
                   "http_active": activity_metrics(metrics)[0], "idle_seconds": 0 if quiet.window.since is None else time.monotonic() - quiet.window.since}
        with (state / "drain.jsonl").open("a") as stream:
            stream.write(json.dumps({"at": time.time(), "ready": ready, **details}) + "\n")
        status("waiting_for_idle", **details)
        return ready

    try:
        while True:
            check_identity(manifest)
            check_files(manifest)
            try:
                ready = sample()
            except Exception as exc:
                quiet.reset()
                status("waiting_for_valid_load", error=type(exc).__name__)
                ready = False
            if ready:
                spare = projected_spare_ram(mem_available(), manifest["previous_cache_bytes"])
                if spare < 24 * 2**30:
                    quiet.reset()
                    status("waiting_for_ram", projected_spare_bytes=spare)
                else:
                    check_identity(manifest)
                    check_files(manifest)
                    time.sleep(1)
                    try:
                        if sample():
                            break
                    except Exception:
                        quiet.reset()
            time.sleep(manifest["poll_seconds"])
        atomic_json(state / "attempt.json", {"at": time.time(), "projected_spare_bytes": spare})
        status("relaunching")
        with (state / "relaunch.log").open("x") as output:
            subprocess.run(["docker", "compose", "-f", str(state / "candidate.compose.json"),
                            "up", "-d", "--no-build", "--pull", "never", "--force-recreate", "deepseek"],
                           stdout=output, stderr=subprocess.STDOUT, check=True, timeout=240)
        after = inspect()
        if after["Id"] == manifest["expected_identity"][0] or after["Image"] != manifest["expected_identity"][3]:
            raise RuntimeError("Unexpected recreated identity")
        deadline = time.monotonic() + 2400
        while True:
            current = inspect()
            if current["Id"] != after["Id"] or current["RestartCount"] != 0 or current["State"]["Status"] != "running":
                raise RuntimeError("Candidate changed, exited or restarted during startup")
            text = logs(current)
            (state / "startup.private.log").write_text(text)
            healthy = current["State"].get("Health", {}).get("Status") == "healthy"
            if "Ready: deepseek-v4-flash on port 8000; no backend API key" in text and (
                manifest.get("cache_split_profile") != "l15-67" or healthy
            ):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Startup checks incomplete after 40 minutes")
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
    parser.add_argument("command", choices=("prepare", "arm", "watch"))
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="historical-50",
                        help="Used by arm only; watch follows the frozen manifest")
    args = parser.parse_args()
    os.umask(0o077)
    state = args.state_dir.resolve(strict=True)
    if state.parent != Path("/mnt/hot/dsv41_state") or not state.name.startswith("cache-split-rollout-"):
        parser.error("Use a fresh mktemp-created /mnt/hot/dsv41_state/cache-split-rollout-* directory")
    with (state / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "arm":
            arm(state, args.profile)
        else:
            {"prepare": prepare, "watch": watch}[args.command](state)


if __name__ == "__main__":
    main()
