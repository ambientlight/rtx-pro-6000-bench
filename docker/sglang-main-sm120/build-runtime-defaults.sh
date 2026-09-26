#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image=${IMAGE:-ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a-pr35480-longctx-containment-v12}
base_image=${BASE_IMAGE:-ambientlight/sglang-sm120-mxfp4:2026.08.0-cu130-sm120a-pr35480-longctx-containment-v12-code}
built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

docker image inspect "${base_image}" >/dev/null
docker build \
  --pull=false \
  --file "${here}/Dockerfile.runtime-defaults" \
  --build-arg BASE_IMAGE="${base_image}" \
  --build-arg IMAGE_TAG="${image}" \
  --build-arg BUILT_AT="${built_at}" \
  --tag "${image}" \
  "${here}"

docker image inspect "${image}" --format \
  'built {{.Id}} parent={{index .Config.Labels "ai.sglang.runtime-defaults.parent"}} effort={{index .Config.Labels "ai.sglang.dsv4.reasoning-effort-default"}} entrypoint={{json .Config.Entrypoint}}'
