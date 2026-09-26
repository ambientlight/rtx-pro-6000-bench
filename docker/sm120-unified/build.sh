#!/usr/bin/env bash
# Build the unified SM120 image (serves DeepSeek-V4-Flash and MiniMax-M3).
#
# The heavy step is compiling sgl-kernel 0.4.3 from source. Its parallelism has two
# knobs that MULTIPLY: SGL_BUILD_JOBS (concurrent nvcc procs) x SGL_NVCC_THREADS
# (nvcc --threads). RAM ~= JOBS * 4-8 GB/TU; docker build has no --memory cap, so JOBS
# is the guardrail. Default 64x4.
#
# Usage:
#   ./build.sh                    # build only (JOBS=64)
#   SGL_BUILD_JOBS=32 ./build.sh  # gentler on a shared box (~130-260 GB peak)
#   PUSH=1 ./build.sh             # build + push all tags (docker login first)
#   CUDA_BASE=nvidia/cuda:13.2.0-devel-ubuntu24.04 ./build.sh   # bump CUDA
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${HERE}/Dockerfile"
CONTEXT="${HERE}"
LOG="${HERE}/build.log"

# Image identity (env-overridable). Repo name encodes engine+arch+quant, not the
# models — they're mounted and picked at runtime via MODEL=.
REGISTRY="${REGISTRY:-ambientlight}"
NAME="${NAME:-sglang-sm120-mxfp4}"
CUDA_BASE="${CUDA_BASE:-nvidia/cuda:13.1.1-devel-ubuntu24.04}"
CU_TAG="cu$(printf '%s' "${CUDA_BASE}" | sed -E 's#.*cuda:([0-9]+)\.([0-9]+).*#\1\2#')"
DATE_TAG="${DATE_TAG:-2026.07.3}"
REPO="${REGISTRY}/${NAME}"
# This legacy builder publishes only its pinned July tag. `latest` belongs to
# the benchmark-qualified image under ../sglang-main-sm120/.
TAGS=( "${REPO}:${DATE_TAG}-${CU_TAG}-sm120a" )
PUSH="${PUSH:-0}"

# Build tunables (env-overridable). RAM ~= JOBS * 8 GB.
SGL_BUILD_JOBS="${SGL_BUILD_JOBS:-64}"
SGL_NVCC_THREADS="${SGL_NVCC_THREADS:-4}"

# --- resource sanity ---
CORES="$(nproc)"
RAM_GB="$(free -g | awk '/Mem/{print $2}')"
FREE_GB="$(free -g | awk '/Mem/{print $7}')"
EFFECTIVE=$(( SGL_BUILD_JOBS * SGL_NVCC_THREADS ))
PEAK_LO=$(( SGL_BUILD_JOBS * 4 ))          # ~4 GB/TU low estimate
PEAK_HI=$(( SGL_BUILD_JOBS * 8 ))          # ~8 GB/TU high estimate

cat <<EOF
========================================================================
 sglang SM120 MXFP4 serving image
------------------------------------------------------------------------
 tags         : ${TAGS[*]}
 dockerfile   : ${DOCKERFILE}
 cuda base    : ${CUDA_BASE}
 host         : ${CORES} cores, ${RAM_GB} GB RAM (${FREE_GB} GB free)
 sgl-kernel   : ${SGL_BUILD_JOBS} jobs x ${SGL_NVCC_THREADS} nvcc-threads
                = ${EFFECTIVE}-way, est. peak ~${PEAK_LO}-${PEAK_HI} GB RAM
 push         : $( [ "${PUSH}" = "1" ] && echo "yes (after build)" || echo "no (set PUSH=1 to push)" )
 log          : ${LOG}
========================================================================
EOF

# assemble one -t per tag (single build, multiple tags)
TAG_ARGS=()
for t in "${TAGS[@]}"; do TAG_ARGS+=( -t "${t}" ); done

echo ">>> starting build at $(date -u '+%Y-%m-%dT%H:%M:%SZ') (tail -f ${LOG} to watch)"
set -x
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --build-arg CUDA_BASE="${CUDA_BASE}" \
  --build-arg SGL_BUILD_JOBS="${SGL_BUILD_JOBS}" \
  --build-arg SGL_NVCC_THREADS="${SGL_NVCC_THREADS}" \
  "${TAG_ARGS[@]}" \
  -f "${DOCKERFILE}" \
  "${CONTEXT}" 2>&1 | tee "${LOG}"
set +x

echo ">>> build finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ">>> images:"
docker images "${REPO}" --format '    {{.Repository}}:{{.Tag}}  {{.Size}}  (created {{.CreatedSince}})'

if [ "${PUSH}" = "1" ]; then
  echo ">>> pushing tags (needs prior 'docker login'):"
  for t in "${TAGS[@]}"; do
    echo "    docker push ${t}"
    docker push "${t}"
  done
else
  echo ">>> to publish (docker login first):"
  for t in "${TAGS[@]}"; do echo "    docker push ${t}"; done
fi

PRIMARY="${TAGS[0]}"
cat <<EOF

Next — serve-test (needs NVIDIA Container Toolkit + free Blackwell SM120 GPUs):
  docker run --rm --gpus all --ipc=host -p 8000:8000 -e MODEL=dsv4 \\
    -v /mnt/hot/ambientlight/models/DeepSeek-V4-Flash:/model:ro ${PRIMARY}
  docker run --rm --gpus all --ipc=host -p 8000:8000 -e MODEL=m3 \\
    -v /mnt/hot/ambientlight/models/minimax-m3-mxfp4:/model:ro ${PRIMARY}
EOF
