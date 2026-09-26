#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sglang_context=${SGLANG_CONTEXT:-/mnt/hot/ambientlight/repos/sglang-dsv4-production-overlay}
image=${IMAGE:-ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a-pr35480-longctx-containment-v12-code}
base_image=${BASE_IMAGE:-ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a-pr35480-inline-system-v1}
production_base_sha=2c03d62532c880d17ef356c24275d1f7117c1d69
expected_source_sha=71c498b991ce143999132aafd951d16d6b119805

source_sha=$(git -C "${sglang_context}" rev-parse HEAD)
test "${source_sha}" = "${expected_source_sha}"
git -C "${sglang_context}" merge-base --is-ancestor \
  "${production_base_sha}" "${source_sha}"
test -z "$(git -C "${sglang_context}" status --porcelain)"

source_files=(
  python/pyproject.toml
  python/sglang/kernels/jit/csrc/deepseek_v4/hash_topk.cuh
  python/sglang/kernels/jit/csrc/moe/moe_fused_gate.cuh
  python/sglang/kernels/ops/moe/moe_fused_gate.py
  python/sglang/srt/arg_groups/overrides.py
  python/sglang/srt/entrypoints/anthropic/protocol.py
  python/sglang/srt/entrypoints/anthropic/serving.py
  python/sglang/srt/entrypoints/openai/chat_encoding.py
  python/sglang/srt/entrypoints/openai/encoding_dsv4.py
  python/sglang/srt/entrypoints/openai/protocol.py
  python/sglang/srt/entrypoints/openai/serving_chat.py
  python/sglang/srt/entrypoints/openai/serving_responses.py
  python/sglang/srt/function_call/core_types.py
  python/sglang/srt/function_call/deepseekv32_detector.py
  python/sglang/srt/function_call/function_call_parser.py
  python/sglang/srt/layers/moe/hash_topk.py
  python/sglang/srt/managers/scheduler.py
  python/sglang/srt/managers/tokenizer_manager.py
  python/sglang/srt/parser/reasoning_parser.py
)

source_digest=$(
  for source_file in "${source_files[@]}"; do
    sha256sum "${sglang_context}/${source_file}"
  done | sha256sum | cut -d' ' -f1
)

docker image inspect "${base_image}" >/dev/null
docker build \
  --pull=false \
  --file "${here}/Dockerfile.longctx-containment" \
  --build-arg BASE_IMAGE="${base_image}" \
  --build-arg SOURCE_SHA="${source_sha}" \
  --build-arg PRODUCTION_BASE_SHA="${production_base_sha}" \
  --build-arg SOURCE_DIGEST="${source_digest}" \
  --build-arg IMAGE_TAG="${image}" \
  --label ai.sglang.longctx.built-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t "${image}" \
  "${sglang_context}"

docker image inspect "${image}" --format \
  'built {{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}} production-base={{index .Config.Labels "ai.sglang.longctx.production-base-revision"}} source-digest={{index .Config.Labels "ai.sglang.longctx.source-digest"}} policy={{index .Config.Labels "ai.sglang.longctx.protocol-policy"}}'
