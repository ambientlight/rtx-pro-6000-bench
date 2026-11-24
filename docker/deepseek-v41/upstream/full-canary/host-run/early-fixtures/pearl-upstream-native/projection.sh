#!/usr/bin/env bash
set -eu
export PEARL_GPU_PREPARE=1 PEARL_WOB_MODULE=checkpoint_projection PEARL_PUBLIC_REPLAY=1 PEARL_OPERATOR_BUILD=/tmp/pearl-gpu-prepare-sm120
export PEARL_CHECKPOINT_DIR=/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017 MODEL_PATH=/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017
export PEARL_CUDART=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-gpu0-tuning/target/gpu0-study/cuda134-root/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcudart.so.13
exec /tmp/pearl-upstream-native/oci-gpu3.sh "$@"
