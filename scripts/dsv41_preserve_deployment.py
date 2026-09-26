#!/usr/bin/env python3
"""Save private rollback evidence without stopping or altering any container.

Docker inspection can contain credentials: all artifacts are local, mode 0600,
inside a newly created mode-0700 directory. Never print raw inspect data.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
CONTAINERS = ("dsv4", "qwen3-embed")


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE)


def save(path, value):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(path, flags, 0o600), "w") as handle:
        handle.write(value if isinstance(value, str) else json.dumps(value, indent=2))
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/hot/dsv41_state"))
    args = parser.parse_args()
    os.umask(0o077)
    lock = json.loads((REPO / "docker/deepseek-v41/baseline.lock.json").read_text())
    recipe = Path(lock["recipe_directory"])
    revision = run("git", "-C", str(recipe), "rev-parse", "HEAD").strip()
    if revision != lock["recipe_revision"]:
        raise RuntimeError("0xSero recipe does not match the locked revision")
    if run("git", "-C", str(recipe), "status", "--porcelain").strip():
        raise RuntimeError("Baseline recipe has local changes; preserve/review them first")
    containers = json.loads(run("docker", "inspect", *CONTAINERS))
    by_name = {item["Name"].lstrip("/"): item for item in containers}
    if by_name["dsv4"]["Image"] != lock["v12_image_id"]:
        raise RuntimeError("Current dsv4 is not the reviewed V12 image; reevaluate before stopping")
    now = dt.datetime.now(dt.timezone.utc)
    args.output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.output_root.stat().st_mode & 0o077:
        raise RuntimeError("Output root must already be private (mode 0700)")
    output = args.output_root / ("rollback-" + now.strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(mode=0o700)
    save(output / "baseline.lock.json", lock)
    save(output / "containers.private.json", containers)
    for name in CONTAINERS:
        save(output / (name + ".image.private.json"),
             json.loads(run("docker", "image", "inspect", by_name[name]["Image"])))
    capture = Path("/mnt/hot/dsv4_dumps/current")
    save(output / "capture-position.json", {
        "current": str(capture.resolve()) if capture.exists() else None,
        "native_log_bytes": (capture / "dsv4-native.log").stat().st_size
        if (capture / "dsv4-native.log").exists() else None,
    })
    save(output / "gpu.csv", run("nvidia-smi",
         "--query-gpu=index,name,uuid,memory.total,memory.used,driver_version,power.limit",
         "--format=csv"))
    save(output / "memory.txt", run("free", "-h"))
    save(output / "disk.txt", run("df", "-h", "/mnt/hot"))
    save(output / "capture-unit.txt", run("systemctl", "--user", "cat", "dsv4-wire-dump.service"))
    save(output / "v12-git.txt", run("git", "-C",
         "/mnt/hot/ambientlight/repos/sglang-dsv4-production-overlay", "rev-parse", "HEAD"))
    save(output / "summary.json", {
        "recorded_at": now.isoformat(),
        "containers": [{"name": name, "id": by_name[name]["Id"],
                        "image_id": by_name[name]["Image"],
                        "running": by_name[name]["State"]["Running"],
                        "restart_policy": by_name[name]["HostConfig"]["RestartPolicy"]}
                       for name in CONTAINERS],
        "rollback": "Stop the V4.1 candidate, then docker start dsv4 qwen3-embed; verify health.",
        "thinking_key": "Preserved original mount /mnt/hot/dsv4_state/anthropic-thinking.key",
        "container_mutations_performed": False,
    })
    print(output)


if __name__ == "__main__":
    main()
