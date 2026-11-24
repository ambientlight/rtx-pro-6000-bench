#!/usr/bin/env bash
set -eu
name=$1
args=(--tp-size 1 --max-running-requests 1 --sampling-backend pytorch --mem-fraction-static .85 --disable-prefill-cuda-graph --disable-decode-cuda-graph --skip-server-warmup --disable-custom-all-reduce --host 127.0.0.1 --port 31874)
case $name in
  llama) args+=(--model-path /tmp/pearl-upstream-native/tiny-llama --served-model-name sm120-tiny --max-total-tokens 256 --context-length 128 --chunked-prefill-size 128 --attention-backend triton);;
  dsv) args+=(--model-path /tmp/pearl-upstream-native/tiny-dsv41-recipe --served-model-name sm120-dsv41-canary --load-format dummy --disable-flashinfer-autotune --max-total-tokens 1024 --context-length 512 --chunked-prefill-size 256 --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4);;
esac
exec /tmp/pearl-upstream-native/oci.sh -m sglang.launch_server "${args[@]}"
