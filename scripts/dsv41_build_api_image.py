#!/usr/bin/env python3
"""Build the scoped deployment overlay, checking source and baseline identities."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
API_FILES = [
    "environ.py",
    "managers/tokenizer_manager.py",
    "utils/watchdog.py",
    "utils/cudacore_pyspy_dump_utils.py",
    "managers/scheduler_components/metrics_reporter.py",
    "mem_cache/unified_cache/components/swa_component.py",
    "mem_cache/unified_cache/unified_tree_core.py",
    "mem_cache/unified_cache/host_cache_diagnostics.py",
    "observability/metrics_collector.py",
    "entrypoints/anthropic/protocol.py",
    "entrypoints/anthropic/serving.py",
    "entrypoints/openai/chat_encoding.py",
    "entrypoints/openai/encoding_dsv4.py",
    "entrypoints/openai/encoding_dsv41.py",
    "entrypoints/openai/protocol.py",
    "entrypoints/openai/serving_chat.py",
    "entrypoints/openai/serving_responses.py",
    "function_call/core_types.py",
    "function_call/deepseekv32_detector.py",
    "function_call/deepseekv41_detector.py",
    "function_call/function_call_parser.py",
    "parser/reasoning_parser.py",
    "mem_cache/hybrid_cache/hybrid_pool_assembler.py",
    "mem_cache/hybrid_cache/dsv41_host_budget.py",
]


def output(*command):
    return subprocess.check_output(command, text=True).strip()


def image_files(image, paths):
    code = (
        "import hashlib,json,pathlib; paths=" + repr(paths) + "; "
        "print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}))"
    )
    return json.loads(
        output(
            "docker",
            "run",
            "--rm",
            "--runtime",
            "runc",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=void",
            "--entrypoint",
            "python3",
            image,
            "-c",
            code,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay")
    )
    parser.add_argument("--tag", default="ambientlight/dsv41-sm120:sero-v12-api-candidate")
    args = parser.parse_args()
    os.umask(0o077)
    lock = json.loads((ROOT / "docker/deepseek-v41/baseline.lock.json").read_text())
    baseline = lock["baseline_image_tag"]
    actual = output("docker", "image", "inspect", "--format", "{{.Id}}", baseline)
    if actual != lock["baseline_image_id"]:
        raise RuntimeError("Baseline image identity changed; do not substitute another runtime")
    if output("git", "-C", str(args.source), "status", "--porcelain"):
        raise RuntimeError("Commit the reviewed overlay and its tests before building")
    revision = output("git", "-C", str(args.source), "rev-parse", "HEAD")
    # Require every runtime source change to be in the explicit Docker COPY set.
    changed = set(
        output("git", "-C", str(args.source), "diff", "--name-only", "96f5a0d", "HEAD", "--", "python/").splitlines()
    )
    allowed = {"python/sglang/srt/" + p for p in API_FILES}
    if changed - allowed:
        raise RuntimeError(f"Unpackaged runtime changes: {sorted(changed - allowed)}")
    subprocess.run(
        [
            "docker",
            "build",
            "--progress=plain",
            "--file",
            str(ROOT / "docker/deepseek-v41/Dockerfile.api"),
            "--build-arg",
            "OVERLAY_REVISION=" + revision,
            "--tag",
            args.tag,
            str(args.source),
        ],
        check=True,
    )
    local_paths = {
        "/sgl-workspace/sglang/python/sglang/srt/" + p: args.source / "python/sglang/srt" / p for p in API_FILES
    }
    local_paths["/opt/dsv41/boot.py"] = args.source / "deployment/boot.py"
    expected = {p: hashlib.sha256(src.read_bytes()).hexdigest() for p, src in local_paths.items()}
    if image_files(args.tag, list(expected)) != expected:
        raise RuntimeError("Built API files differ from reviewed source")
    unchanged_paths = [
        "/opt/dsv41/adapter/sitecustomize.py",
        "/opt/dsv41/adapter/librow_store.so",
        "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py",
        "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py",
    ]
    runtime = image_files(baseline, unchanged_paths)
    if image_files(args.tag, unchanged_paths) != runtime:
        raise RuntimeError("Hardware/lifecycle baseline unexpectedly changed")
    evidence = Path("/mnt/hot/dsv41_state") / (
        "api-image-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    evidence.mkdir(mode=0o700)
    receipt = {
        "image_tag": args.tag,
        "image_id": output("docker", "image", "inspect", "--format", "{{.Id}}", args.tag),
        "overlay_revision": revision,
        "baseline_image_id": actual,
        "api_file_sha256": expected,
        "unchanged_runtime_sha256": runtime,
    }
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"image_id": receipt["image_id"], "receipt": str(evidence / "receipt.json")}))


if __name__ == "__main__":
    main()
