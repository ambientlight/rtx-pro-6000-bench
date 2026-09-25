#!/usr/bin/env python3
"""Snapshot and verify a narrowly scoped SWA layer; never restart production.

Unlike the release builder, this explicit candidate builder accepts a dirty
worktree. It preserves the five source files and tests, hashes their exact
bytes, verifies their committed baselines against the deployed image, and
never commits or packages unrelated worktree edits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from dsv41_build_api_image import API_FILES, image_files, output


ROOT = Path(__file__).resolve().parents[1]
PARENT = "sha256:fb61d2521beb6188652417734a8b402cabd5209675310cb3d610bfe95f08c5ca"
PARENT_TAG = "ambientlight/dsv41-sm120:dsv41-v2"
PREFIX = "python/sglang/srt/"
FILES = (
    "environ.py",
    "managers/scheduler_components/metrics_reporter.py",
    "mem_cache/unified_cache/components/swa_component.py",
    "mem_cache/unified_cache/unified_tree_core.py",
    "observability/metrics_collector.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay")
    )
    parser.add_argument("--tag", default="ambientlight/dsv41-sm120:swa-retention-v1")
    args = parser.parse_args()
    os.umask(0o077)
    if output("docker", "image", "inspect", "--format", "{{.Id}}", PARENT_TAG) != PARENT:
        raise RuntimeError("Pinned deployment parent unavailable")
    image_paths = {p: "/sgl-workspace/sglang/" + PREFIX + p for p in FILES}
    parent_hashes = image_files(PARENT, list(image_paths.values()))
    for p in FILES:
        committed = subprocess.check_output(["git", "-C", str(args.source), "show", "HEAD:" + PREFIX + p])
        if hashlib.sha256(committed).hexdigest() != parent_hashes[image_paths[p]]:
            raise RuntimeError(f"Committed source baseline differs from pinned runtime: {p}")
    evidence = Path(tempfile.mkdtemp(prefix="swa-retention-image-", dir="/mnt/hot/dsv41_state"))
    context = evidence / "context"
    context.mkdir()
    files = [PREFIX + p for p in FILES] + ["deployment/test_swa_host_retention.py"]
    hashes = {}
    for relative in files:
        target = context / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source / relative, target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    dockerfile = context / "Dockerfile"
    shutil.copyfile(ROOT / "docker/deepseek-v41/Dockerfile.swa-retention", dockerfile)
    manifest = {
        "parent_image_id": PARENT,
        "source_head": output("git", "-C", str(args.source), "rev-parse", "HEAD"),
        "dirty_snapshot": True,
        "source_sha256": hashes,
        "dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        "parent_file_sha256": parent_hashes,
        "runtime_files": [PREFIX + p for p in FILES],
        "notes": "Existing boot/host-budget read-only deployment mounts remain required and unchanged.",
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    (context / "source-manifest.json").write_bytes(manifest_bytes)
    with (evidence / "source.patch").open("wb") as stream:
        subprocess.run(
            ["git", "-C", str(args.source), "diff", "--binary", "--", *[PREFIX + p for p in FILES]],
            stdout=stream,
            check=True,
        )
    subprocess.run(
        [
            "docker",
            "build",
            "--progress=plain",
            "--build-arg",
            "SOURCE_MANIFEST_SHA256=" + hashlib.sha256(manifest_bytes).hexdigest(),
            "--tag",
            args.tag,
            str(context),
        ],
        check=True,
    )
    expected = {image_paths[p]: hashes[PREFIX + p] for p in FILES}
    if output("docker", "image", "inspect", "--format", "{{.Id}}", PARENT_TAG) != PARENT:
        raise RuntimeError("Parent tag changed while building")
    if image_files(args.tag, list(expected)) != expected:
        raise RuntimeError("Candidate runtime differs from the frozen snapshot")
    protected = [
        "/sgl-workspace/sglang/" + PREFIX + p for p in API_FILES if p not in FILES and not p.startswith("mem_cache/")
    ] + [
        "/opt/dsv41/boot.py",
        "/opt/dsv41/adapter/sitecustomize.py",
        "/opt/dsv41/adapter/librow_store.so",
        "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py",
        "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py",
        "/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py",
    ]
    unchanged = image_files(PARENT, protected)
    if image_files(args.tag, protected) != unchanged:
        raise RuntimeError("API, lifecycle, launcher, kernel or Engram baseline changed")
    receipt = {
        **manifest,
        "image_tag": args.tag,
        "image_id": output("docker", "image", "inspect", "--format", "{{.Id}}", args.tag),
        "unchanged_file_sha256": unchanged,
    }
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"image_id": receipt["image_id"], "evidence": str(evidence)}), flush=True)


if __name__ == "__main__":
    main()
