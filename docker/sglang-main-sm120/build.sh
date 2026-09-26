#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-ambientlight/sglang-main-sm120:b03ac355-cu130-candidate}

docker build \
  --pull=false \
  --label ai.sglang.comparison.built-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t "${IMAGE}" \
  "${HERE}"

docker image inspect "${IMAGE}" \
  --format 'built {{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
