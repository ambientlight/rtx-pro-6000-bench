#!/usr/bin/env bash
set -eu
source /tmp/pearl_sgl_gpuenv
export PEARL_OPERATOR_BUILD=/tmp/pearl-gpu-prepare-sm120 PEARL_GPU_PREPARE=1
export PEARL_REPLAY_POOL_HOST=127.0.0.1 PEARL_REPLAY_POOL_PORT=31984
export PEARL_SHAPE_DIR=/tmp/pearl-upstream-native/proof-http-shapes
exec /tmp/pearl-upstream-native/proof-http-oci.sh -m sglang.launch_server \
 --model-path /tmp/pearl-upstream-native/hybrid-dsv41-proof --served-model-name sm120-dsv41-pearl-hybrid \
 --load-format dummy --tp-size 1 --max-running-requests 1 --sampling-backend pytorch \
 --mem-fraction-static .85 --disable-prefill-cuda-graph --disable-decode-cuda-graph \
 --skip-server-warmup --disable-custom-all-reduce --disable-flashinfer-autotune \
 --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 --max-total-tokens 1024 \
 --context-length 512 --chunked-prefill-size 256 --host 127.0.0.1 --port 31983
