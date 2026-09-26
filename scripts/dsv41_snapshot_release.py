#!/usr/bin/env python3
"""Snapshot a running DSV4.1 deployment without stopping it or copying weights.

All output is private. This captures settings, mounted code, the thinking key,
and source provenance; image saving and Git bundles are separate explicit steps.
It never starts inference, changes container settings, or restarts services.
"""

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

from dsv41_build_api_image import API_FILES

REPO = Path(__file__).resolve().parents[1]
SGLANG = Path("/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay")
PREFIX = "/sgl-workspace/sglang/python/sglang/srt/"
DEPLOYMENT_FILES = ("baseline.lock.json", "docs/NO-SWAP.md")


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE).strip()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2))
        stream.write("\n")


def private_copy(source, target):
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with target.open("xb") as stream, Path(source).open("rb") as original:
        shutil.copyfileobj(original, stream)


def source_map():
    sources = {PREFIX + p: SGLANG / "python/sglang/srt" / p for p in API_FILES}
    extra = "mem_cache/unified_cache/host_cache_diagnostics.py"
    sources[PREFIX + extra] = SGLANG / "python/sglang/srt" / extra
    sources["/opt/dsv41/boot.py"] = SGLANG / "deployment/boot.py"
    sources["/opt/dsv41/diagnostic_entrypoint.py"] = REPO / "docker/deepseek-v41/diagnostic_entrypoint.py"
    return sources


def snapshot(output, container):
    if output.is_symlink() or not output.is_dir() or stat.S_IMODE(output.stat().st_mode) != 0o700:
        raise ValueError("Output must be an existing private, real mode-0700 directory")
    if (output / "manifest.json").exists():
        raise ValueError("Refusing to overwrite an existing snapshot")
    before = json.loads(run("docker", "inspect", container))[0]
    if before["Name"] != "/dsv41" or before["State"].get("Health", {}).get("Status") != "healthy":
        raise ValueError("Expected the healthy dsv41 container")
    profile_path = Path(before["Config"]["Labels"]["com.docker.compose.project.config_files"])
    profile = json.loads(profile_path.read_text())
    if set(profile["services"]) != {"deepseek"}:
        raise ValueError("Expected exactly one production inference service")
    service = profile["services"]["deepseek"]
    if run("docker", "image", "inspect", "--format", "{{.Id}}", service["image"]) != before["Image"]:
        raise ValueError("Profile image differs from the live image")
    env = dict(v.split("=", 1) for v in before["Config"]["Env"])
    if any(env.get(k) != str(v) for k, v in service["environment"].items()):
        raise ValueError("Profile environment differs from the live container")
    if before["HostConfig"]["CgroupParent"] != "dsv41.slice" or service.get("cgroup_parent") != "dsv41.slice":
        raise ValueError("Expected the persistent no-swap cgroup")
    swap_max = Path("/sys/fs/cgroup/dsv41.slice/memory.swap.max").read_text().strip()
    if swap_max != "0":
        raise ValueError("No-swap parent limit is not effective")
    local = source_map()
    expected = {path: digest(source) for path, source in local.items()}
    actual = {path: sha for sha, path in (line.split(maxsplit=1) for line in
              run("docker", "exec", before["Id"], "sha256sum", *expected).splitlines())}
    mismatches = [p for p in expected if expected[p] != actual.get(p)]
    if mismatches:
        raise ValueError("Working source differs from deployed source: " + ", ".join(mismatches))
    frozen = copy.deepcopy(profile)
    frozen_service = frozen["services"]["deepseek"]
    frozen_service["image"] = before["Image"]
    frozen_service["pull_policy"] = "never"
    mounts = {m["Destination"]: m for m in before["Mounts"]}
    copied = []
    for mount in frozen_service["volumes"]:
        if mount["type"] != "bind":
            continue
        original = mounts[mount["target"]]
        if mount["source"] != original["Source"] or bool(mount.get("read_only")) == original["RW"]:
            raise ValueError("Profile bind mount differs from the running container")
        source = Path(mount["source"])
        if source.suffix == ".py":
            if mount["target"] not in expected or digest(source) != expected[mount["target"]]:
                raise ValueError("Unreviewed frozen runtime source")
            relative = Path("runtime") / mount["target"].lstrip("/")
        elif mount["target"] == "/run/dsv41/anthropic-thinking.key":
            relative = Path("private/anthropic-thinking.key")
        else:
            continue
        private_copy(source, output / relative)
        mount["source"] = "./" + str(relative)
        mount.setdefault("bind", {})["create_host_path"] = False
        copied.append({"target": mount["target"], "original_source": str(source),
                       "archive_source": str(relative), "sha256": digest(output / relative)})
    private_copy("/etc/systemd/system/dsv41.slice", output / "systemd/dsv41.slice")
    for unit in ("dsv41-wire-dump.service", "dsv41-cache-monitor.service"):
        path = run("systemctl", "--user", "show", unit, "-p", "FragmentPath", "--value")
        private_copy(path, output / "systemd" / unit)
        if digest(path) != digest(REPO / "systemd" / unit):
            raise ValueError("Repository and installed telemetry unit differ: " + unit)
    if digest("/etc/systemd/system/dsv41.slice") != digest(REPO / "systemd/dsv41.slice"):
        raise ValueError("Repository and installed no-swap unit differ")
    save(output / "private/container.inspect.json", before)
    save(output / "private/image.inspect.json", json.loads(run("docker", "image", "inspect", before["Image"])))
    save(output / "private/original.compose.json", profile)
    save(output / "compose.restore.json", frozen)
    # Diff is evidence, not a restorable filesystem snapshot. Mounted content is
    # captured above. Do not assume caches or runtime package changes are images.
    save(output / "private/container.diff.txt", run("docker", "diff", before["Id"]))
    model = Path(mounts["/models/DeepSeek-V4.1-Flash"]["Source"])
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        if (model / name).is_file():
            private_copy(model / name, output / "model-metadata" / name)
    for relative in DEPLOYMENT_FILES:
        # Keep the release archive's flat filenames stable after doc moves.
        private_copy(REPO / "docker/deepseek-v41" / relative, output / "deployment" / Path(relative).name)
    after = json.loads(run("docker", "inspect", container))[0]
    if (before["Id"], before["State"]["StartedAt"], before["RestartCount"]) != (
        after["Id"], after["State"]["StartedAt"], after["RestartCount"]
    ):
        raise ValueError("Deployment changed during snapshot; do not mark release complete")
    manifest = {
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "container_id": before["Id"], "started_at": before["State"]["StartedAt"],
        "image_id": before["Image"], "restart_count": before["RestartCount"],
        "profile_source": str(profile_path), "source_sha256": expected,
        "source_paths": {k: str(v) for k, v in local.items()}, "copied_mounts": copied,
        "git_heads_at_snapshot": {str(repo): run("git", "-C", str(repo), "rev-parse", "HEAD")
                                  for repo in (REPO, SGLANG)},
        "effective_swap_limit_bytes": 0,
        "excluded": ["model weight shards (retain separately at the pinned revision)",
                     "warm RAM/GPU KV cache", "capture and diagnostic history",
                     "regenerable kernel caches and Engram state files"],
        "image_archive": "dsv41-image.tar (created and verified separately)",
        "mutated_or_restarted_inference": False,
    }
    save(output / "manifest.json", manifest)
    print(json.dumps({"release": str(output), "source_files_matching_live": len(actual),
                      "runtime_mounts_copied": len(copied), "container_id": before["Id"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container", default="dsv41")
    args = parser.parse_args()
    os.umask(0o077)
    snapshot(args.output_dir.absolute(), args.container)


if __name__ == "__main__":
    main()
