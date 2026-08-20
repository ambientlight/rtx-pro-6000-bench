#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a}

docker build \
  --pull=false \
  --file "${HERE}/Dockerfile.release" \
  --label ai.sglang.release.built-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t "${IMAGE}" \
  "${HERE}"

docker image inspect "${IMAGE}" \
  --format 'cut {{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}} gate={{index .Config.Labels "ai.sglang.performance-gate"}}'
