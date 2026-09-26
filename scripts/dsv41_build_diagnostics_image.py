#!/usr/bin/env python3
"""Build a pinned, four-file diagnostic layer; never start/restart inference."""

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
PARENT_TAG = "ambientlight/dsv41-sm120:responses-prefix-v1"
PARENT = "sha256:90d0ef76ed2da4a1bae09a7b9db015bf377952ac77a4b002d3892ce993cc13c3"
SOURCE_BASELINE = "8571611154d4202ec7b571e162c77e9ee52e890f"
PREFIX = "python/sglang/srt/"
FILES = ("environ.py", "managers/tokenizer_manager.py", "utils/watchdog.py", "utils/cudacore_pyspy_dump_utils.py")
TEST = "test/registered/unit/utils/test_crash_diagnostics.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay")
    )
    parser.add_argument("--tag", default="ambientlight/dsv41-sm120:diagnostics-v1")
    args = parser.parse_args()
    os.umask(0o077)
    if output("docker", "image", "inspect", "--format", "{{.Id}}", PARENT_TAG) != PARENT:
        raise RuntimeError("Pinned diagnostic parent changed or is unavailable")
    paths = {p: "/sgl-workspace/sglang/" + PREFIX + p for p in FILES}
    parent_hashes = image_files(PARENT, list(paths.values()))
    # Do not accidentally package drift from an unrelated checkout/image.
    for p in FILES:
        committed = subprocess.check_output(["git", "-C", str(args.source), "show", SOURCE_BASELINE + ":" + PREFIX + p])
        if hashlib.sha256(committed).hexdigest() != parent_hashes[paths[p]]:
            raise RuntimeError(f"Committed diagnostic baseline differs from parent: {p}")
    evidence = Path(tempfile.mkdtemp(prefix="diagnostics-image-", dir="/mnt/hot/dsv41_state"))
    context = evidence / "context"
    context.mkdir()
    hashes = {}
    for relative in [PREFIX + p for p in FILES] + [TEST, "deployment/CRASH_DIAGNOSTICS.md"]:
        target = context / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source / relative, target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    shutil.copyfile(ROOT / "docker/deepseek-v41/Dockerfile.diagnostics", context / "Dockerfile")
    manifest = {
        "parent_image_id": PARENT,
        "source_baseline_revision": SOURCE_BASELINE,
        "source_head": output("git", "-C", str(args.source), "rev-parse", "HEAD"),
        "dirty_snapshot": True,
        "source_sha256": hashes,
        "dockerfile_sha256": hashlib.sha256((context / "Dockerfile").read_bytes()).hexdigest(),
        "parent_file_sha256": parent_hashes,
        "runtime_files": [PREFIX + p for p in FILES],
        "scope": "Fatal diagnostics and cleanup only; no inference/NCCL execution changes.",
    }
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    (context / "source-manifest.json").write_bytes(encoded)
    with (evidence / "source.patch").open("wb") as stream:
        subprocess.run(
            ["git", "-C", str(args.source), "diff", "--binary", SOURCE_BASELINE, "--", *[PREFIX + p for p in FILES]],
            stdout=stream,
            check=True,
        )
    print(f"Build evidence: {evidence}", flush=True)
    subprocess.run(
        [
            "docker",
            "build",
            "--network=none",
            "--progress=plain",
            "--build-arg",
            "SOURCE_MANIFEST_SHA256=" + hashlib.sha256(encoded).hexdigest(),
            "--tag",
            args.tag,
            str(context),
        ],
        check=True,
    )
    if output("docker", "image", "inspect", "--format", "{{.Id}}", PARENT_TAG) != PARENT:
        raise RuntimeError("Parent tag changed during build")
    expected = {paths[p]: hashes[PREFIX + p] for p in FILES}
    if image_files(args.tag, list(expected)) != expected:
        raise RuntimeError("Candidate differs from the frozen source snapshot")
    protected = [
        "/sgl-workspace/sglang/" + PREFIX + p
        for p in API_FILES
        # Supplied by an existing read-only production bind, not in the image.
        if p not in FILES and p != "mem_cache/hybrid_cache/dsv41_host_budget.py"
    ] + [
        "/opt/dsv41/boot.py",
        "/opt/dsv41/adapter/sitecustomize.py",
        "/opt/dsv41/adapter/librow_store.so",
        "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py",
        "/sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py",
        "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py",
        "/sgl-workspace/sglang/python/sglang/srt/distributed/parallel_state.py",
    ]
    unchanged = image_files(PARENT, protected)
    if image_files(args.tag, protected) != unchanged:
        raise RuntimeError("Non-diagnostic serving code changed")
    mounted_sources = {
        relative: hashlib.sha256((args.source / relative).read_bytes()).hexdigest()
        for relative in (
            "deployment/boot.py",
            PREFIX + "mem_cache/hybrid_cache/hybrid_pool_assembler.py",
            PREFIX + "mem_cache/hybrid_cache/dsv41_host_budget.py",
        )
    }
    parent_layers = json.loads(output("docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", PARENT))
    layers = json.loads(output("docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", args.tag))
    if layers[: len(parent_layers)] != parent_layers:
        raise RuntimeError("Candidate does not retain the exact parent layers")
    # Test the bytes actually baked into the image, not the source bind mount.
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--runtime",
            "runc",
            "--network",
            "none",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-v",
            f"{context / TEST}:/sgl-workspace/sglang/{TEST}:ro",
            "--entrypoint",
            "python3",
            args.tag,
            "/sgl-workspace/sglang/" + TEST,
            "-v",
        ],
        check=True,
    )
    receipt = {
        **manifest,
        "image_tag": args.tag,
        "image_id": output("docker", "image", "inspect", "--format", "{{.Id}}", args.tag),
        "unchanged_file_sha256": unchanged,
        "existing_readonly_mount_source_sha256": mounted_sources,
        "parent_layers_preserved": True,
        "built_image_cpu_tests_passed": True,
    }
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"image_id": receipt["image_id"], "receipt": str(evidence / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
