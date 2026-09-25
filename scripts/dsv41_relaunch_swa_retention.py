#!/usr/bin/env python3
"""One approved SWA image rollout after 60 seconds idle, with private rollback.

Changes only image, deployment label and retention environment. Never flushes
KV, force-kills requests, retries a rollout or automatically rolls back.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from dsv41_build_swa_retention_image import PARENT
from dsv41_deferred_relaunch import (
    CONFIG,
    QuietWindow,
    atomic_json,
    digest,
    fetch,
    identity,
    inspect,
    mem_available,
    projected_spare_ram,
    roundtrip_profile,
    run,
    validate_flags,
    verify_budget_logs,
)
from dsv41_load import parse_loads


def candidate_profile(old, image):
    candidate = json.loads(json.dumps(old))
    if set(candidate["services"]) != {"deepseek"}:
        raise ValueError("Only the dsv41 service may be relaunched")
    service = candidate["services"]["deepseek"]
    if service["container_name"] != "dsv41" or service["restart"] != "unless-stopped":
        raise ValueError("Unexpected service identity or restart policy")
    service["image"] = image
    service["labels"]["dev.idle.deployment.version"] = "dsv41-v2-swa-retention-v1"
    service["environment"].update(
        SGLANG_OPT_HICACHE_SWA_RETAIN_REQUEST_BOUNDARIES="1",
        SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND="python",
    )
    return candidate


def logs(container):
    return subprocess.check_output(["docker", "logs", container], stderr=subprocess.STDOUT, text=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-receipt", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    receipt = json.loads(args.image_receipt.read_text())
    before = inspect()
    if before["Image"] != PARENT or receipt["parent_image_id"] != PARENT:
        raise RuntimeError("Live parent is not the reviewed runtime")
    if before["State"]["Status"] != "running":
        raise RuntimeError("Live container is not running")
    image = receipt["image_tag"]
    if run("docker", "image", "inspect", "--format", "{{.Id}}", image).strip() != receipt["image_id"]:
        raise RuntimeError("Candidate image tag no longer matches its receipt")
    state = Path(tempfile.mkdtemp(prefix="swa-retention-rollout-", dir="/mnt/hot/dsv41_state"))
    print(f"Rollout evidence: {state}", flush=True)
    old = roundtrip_profile(json.loads(run("docker", "compose", "-f", str(CONFIG), "config", "--format", "json")))
    old_image = run("docker", "image", "inspect", "--format", "{{.Id}}", old["services"]["deepseek"]["image"]).strip()
    if old_image != before["Image"]:
        raise RuntimeError("Source Compose no longer describes the live image")
    candidate = candidate_profile(old, image)
    atomic_json(state / "rollback.compose.json", old)
    atomic_json(state / "candidate.compose.json", candidate)
    atomic_json(state / "before.container.private.json", before)
    shutil.copyfile(CONFIG, state / "rollback.compose.yaml")
    info = json.loads(fetch("/server_info"))
    validate_flags(info, True)
    atomic_json(state / "before.server-info.private.json", info)
    previous_allocation = verify_budget_logs(logs(before["Id"]))
    atomic_json(state / "before.allocation.json", previous_allocation)
    capture_before = str(Path("/mnt/hot/dsv41_dumps/current").resolve())
    mounts = candidate["services"]["deepseek"]["volumes"]
    tracked = [CONFIG, args.image_receipt, state / "candidate.compose.json"]
    tracked += [Path(m["source"]) for m in mounts if m["type"] == "bind" and Path(m["source"]).is_file()]
    file_hashes = {str(p): digest(p) for p in tracked}
    for path in (state / "rollback.compose.json", state / "candidate.compose.json"):
        run("docker", "compose", "-f", str(path), "config", "--quiet")

    def stable():
        current = inspect()
        if identity(current) != identity(before) or current["State"]["Status"] != "running":
            raise RuntimeError("Live identity changed while draining")
        if any(digest(p) != h for p, h in file_hashes.items()):
            raise RuntimeError("Deployment sources changed while draining")
        if run("docker", "image", "inspect", "--format", "{{.Id}}", image).strip() != receipt["image_id"]:
            raise RuntimeError("Candidate tag changed while draining")

    quiet = QuietWindow(60)
    while True:
        stable()
        rows = parse_loads(json.loads(fetch("/v1/loads")))
        ready = quiet.observe(rows, fetch("/metrics"), time.monotonic())
        sample = {
            "at": time.time(),
            "ready": ready,
            "running": sum(r["num_running_reqs"] for r in rows),
            "waiting": sum(r["num_waiting_reqs"] for r in rows),
        }
        with (state / "drain.jsonl").open("a") as stream:
            stream.write(json.dumps(sample) + "\n")
        print(json.dumps(sample), flush=True)
        if ready:
            break
        time.sleep(5)
    spare = projected_spare_ram(mem_available(), previous_allocation["actual_buffer_bytes_total"])
    if spare < 24 * 2**30:
        raise RuntimeError("Projected host RAM headroom is below 24 GiB")
    stable()
    atomic_json(
        state / "attempt.json", {"at": time.time(), "image_id": receipt["image_id"], "projected_spare_bytes": spare}
    )
    with (state / "relaunch.log").open("x") as stream:
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
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=240,
        )
    after = inspect()
    if after["Id"] == before["Id"] or after["Image"] != receipt["image_id"]:
        raise RuntimeError("Unexpected recreated container identity")
    atomic_json(state / "after.container.private.json", after)
    print(json.dumps({"state": "starting", "container_id": after["Id"]}), flush=True)
    deadline = time.monotonic() + 2400
    while True:
        current = inspect()
        if current["Id"] != after["Id"] or current["RestartCount"] != 0 or current["State"]["Status"] != "running":
            raise RuntimeError("Candidate exited, restarted or changed during startup")
        log_text = logs(after["Id"])
        (state / "startup.private.log").write_text(log_text)
        if "Ready: deepseek-v4-flash on port 8000; no backend API key" in log_text:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Candidate did not pass startup checks within 40 minutes")
        time.sleep(10)
    info = json.loads(fetch("/server_info"))
    validate_flags(info, True)
    allocation = verify_budget_logs(log_text)
    atomic_json(state / "after.server-info.private.json", info)
    atomic_json(state / "after.allocation.json", allocation)
    capture = Path("/mnt/hot/dsv41_dumps/current")
    if str(capture.resolve()) == capture_before:
        raise RuntimeError("Capture did not follow the new container")
    atomic_json(
        state / "complete.json",
        {
            "at": time.time(),
            "container_id": after["Id"],
            "image_id": receipt["image_id"],
            "capture": str(capture.resolve()),
            "available_bytes": mem_available(),
        },
    )
    print(json.dumps({"state": "startup_passed", "evidence": str(state)}), flush=True)


if __name__ == "__main__":
    main()
