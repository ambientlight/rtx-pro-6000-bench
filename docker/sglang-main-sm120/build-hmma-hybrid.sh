#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SGLANG_CONTEXT=${SGLANG_CONTEXT:-/mnt/hot/ambientlight/repos/sglang-b03-hybrid}
IMAGE=${IMAGE:-ambientlight/sglang-main-sm120:b03ac355-cu130-hmma-hybrid}

test -f "${SGLANG_CONTEXT}/python/sglang/kernels/ops/attention/flash_mla_sm120.py"
test "$(git -C "${SGLANG_CONTEXT}" rev-parse HEAD)" = \
  b03ac355e795b3a86b26b8732c47c0965fd71bbc

docker build \
  --pull=false \
  --file "${HERE}/Dockerfile.hmma-hybrid" \
  --label ai.sglang.comparison.built-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t "${IMAGE}" \
  "${SGLANG_CONTEXT}"

docker image inspect "${IMAGE}" \
  --format 'built {{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}} hmma={{index .Config.Labels "ai.sglang.hmma.revision"}}'
