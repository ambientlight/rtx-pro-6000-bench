#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_CONTEXT=${SGLANG_CONTEXT:-/mnt/hot/ambientlight/repos/sglang-b03-reasoning-history}
IMAGE=${IMAGE:-ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a-pr35480-inline-system-v1}
BACKPORT_SHA=$(git -C "${SGLANG_CONTEXT}" rev-parse HEAD)
INLINE_SYSTEM_SOURCE_DIGEST=$(
  sha256sum \
    "${SGLANG_CONTEXT}/python/sglang/srt/entrypoints/anthropic/serving.py" \
    "${SGLANG_CONTEXT}/python/sglang/srt/entrypoints/openai/chat_encoding.py" \
    "${SGLANG_CONTEXT}/python/sglang/srt/entrypoints/openai/serving_chat.py" \
  | sha256sum \
  | cut -d' ' -f1
)

test "${BACKPORT_SHA}" = 2c03d62532c880d17ef356c24275d1f7117c1d69
# The first two overlay files intentionally carry the local inline-system
# backport. Keep the unchanged serving pipeline pinned to HEAD.
test -z "$(git -C "${SGLANG_CONTEXT}" status --porcelain -- \
  python/sglang/srt/entrypoints/openai/serving_chat.py)"

docker build \
  --pull=false \
  --file "${HERE}/Dockerfile.reasoning-history-backport" \
  --build-arg BACKPORT_SHA="${BACKPORT_SHA}" \
  --build-arg INLINE_SYSTEM_SOURCE_DIGEST="${INLINE_SYSTEM_SOURCE_DIGEST}" \
  --build-arg IMAGE_TAG="${IMAGE}" \
  --label ai.sglang.backport.built-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t "${IMAGE}" \
  "${SGLANG_CONTEXT}"

docker image inspect "${IMAGE}" \
  --format 'built {{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}} upstream-fix={{index .Config.Labels "ai.sglang.backport.upstream-commit"}} inline-system={{index .Config.Labels "ai.sglang.anthropic.inline-system"}} source-digest={{index .Config.Labels "ai.sglang.anthropic.inline-system.source-digest"}}'
