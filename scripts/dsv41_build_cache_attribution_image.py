#!/usr/bin/env python3
"""Freeze/build/test the cache-attribution layer without touching inference."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dsv41_build_api_image import API_FILES, image_files, output

ROOT = Path(__file__).resolve().parents[1]
PARENT_TAG = "ambientlight/dsv41-sm120:diagnostics-v1"
PARENT = "sha256:6fbcd0cc3dee2fd12c492e8c78086f4d01f81b525452ea7d60824086fe8d36de"
PREFIX = "python/sglang/srt/"
NEW_FILE = "mem_cache/unified_cache/host_cache_diagnostics.py"
FILES = (
    NEW_FILE,
    "mem_cache/unified_cache/unified_tree_core.py",
    "managers/scheduler_components/metrics_reporter.py",
    "observability/metrics_collector.py",
)
TESTS = ("test_host_cache_diagnostics", "test_swa_host_retention")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay")
    )
    parser.add_argument("--tag", default="ambientlight/dsv41-sm120:cache-attribution-v1")
    args = parser.parse_args()
    os.umask(0o077)
    if output("docker", "image", "inspect", "--format", "{{.Id}}", PARENT_TAG) != PARENT:
        raise RuntimeError("Pinned diagnostics parent changed or is unavailable")
    paths = {p: "/sgl-workspace/sglang/" + PREFIX + p for p in FILES}
    parent_hashes = image_files(PARENT, [v for k, v in paths.items() if k != NEW_FILE])
    source_head = output("git", "-C", str(args.source), "rev-parse", "HEAD")
    for p in FILES:
        if p == NEW_FILE:
            continue
        committed = subprocess.check_output(["git", "-C", str(args.source), "show", source_head + ":" + PREFIX + p])
        if hashlib.sha256(committed).hexdigest() != parent_hashes[paths[p]]:
            raise RuntimeError(f"Committed baseline differs from the deployed parent: {p}")
    evidence = Path(tempfile.mkdtemp(prefix="cache-attribution-image-", dir="/mnt/hot/dsv41_state"))
    context = evidence / "context"
    context.mkdir()
    hashes = {}
    relative_files = [PREFIX + p for p in FILES] + [f"deployment/{name}.py" for name in TESTS]
    relative_files += ["deployment/HICACHE_CACHE_ATTRIBUTION.md", "deployment/benchmark_host_cache_diagnostics.py"]
    for relative in relative_files:
        target = context / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source / relative, target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    shutil.copyfile(ROOT / "docker/deepseek-v41/Dockerfile.cache-attribution", context / "Dockerfile")
    manifest = {
        "parent_image_id": PARENT,
        "source_head": source_head,
        "dirty_snapshot": True,
        "source_sha256": hashes,
        "parent_file_sha256": parent_hashes,
        "runtime_files": [PREFIX + p for p in FILES],
        "dockerfile_sha256": hashlib.sha256((context / "Dockerfile").read_bytes()).hexdigest(),
        "scope": "Observability only: attribution, bounded residency census and prefix-lookup loss; no eviction-policy, kernel, API, allocator or NCCL changes.",
    }
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    (context / "source-manifest.json").write_bytes(encoded)
    with (evidence / "source.patch").open("wb") as stream:
        subprocess.run(
            ["git", "-C", str(args.source), "diff", "--binary", source_head, "--", *[PREFIX + p for p in FILES]],
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
        raise RuntimeError("Built runtime differs from the frozen snapshot")
    protected = [
        "/sgl-workspace/sglang/" + PREFIX + p
        for p in API_FILES
        if p not in FILES and p != "mem_cache/hybrid_cache/dsv41_host_budget.py"
    ]
    protected += [
        "/opt/dsv41/boot.py",
        "/opt/dsv41/adapter/sitecustomize.py",
        "/opt/dsv41/adapter/librow_store.so",
        "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py",
        "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py",
        "/sgl-workspace/sglang/python/sglang/srt/distributed/parallel_state.py",
    ]
    unchanged = image_files(PARENT, protected)
    if image_files(args.tag, protected) != unchanged:
        raise RuntimeError("Non-observability runtime files changed")
    parent_layers = json.loads(output("docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", PARENT))
    layers = json.loads(output("docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", args.tag))
    if layers[: len(parent_layers)] != parent_layers:
        raise RuntimeError("Parent image layers were not preserved")
    command = [
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
        f"{context / 'deployment'}:/tests:ro",
        "--workdir",
        "/tests",
        "--entrypoint",
        "python3",
        args.tag,
        "-m",
        "unittest",
        "-v",
        *TESTS,
    ]
    with (evidence / "cpu-tests.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    benchmark_command = command[: -len(TESTS) - 3] + ["/tests/benchmark_host_cache_diagnostics.py"]
    with (evidence / "census-cost.log").open("w") as log:
        subprocess.run(benchmark_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = {
        **manifest,
        "image_tag": args.tag,
        "image_id": output("docker", "image", "inspect", "--format", "{{.Id}}", args.tag),
        "unchanged_file_sha256": unchanged,
        "parent_layers_preserved": True,
        "built_image_cpu_tests_passed": True,
        "test_command": command,
        "production_restarted": False,
    }
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"image_id": receipt["image_id"], "receipt": str(evidence / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
