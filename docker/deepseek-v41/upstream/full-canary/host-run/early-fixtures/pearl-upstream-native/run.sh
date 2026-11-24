#!/usr/bin/env bash
set -eu
source /tmp/pearl_sgl_gpuenv
export PYTHONPATH=/tmp/pearl-upstream-native/fi:$PWD/../sglang-upstream-sm120-kernels/python:/tmp/pearl-sgl-extra-deps:$PWD/.venv/lib/python3.12/site-packages
export CUDA_VISIBLE_DEVICES=2 SGLANG_SM120_FLASHMLA_BACKEND=flashinfer FLASHINFER_CUDA_ARCH_LIST=12.0f
export CUDA_HOME=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-gpu0-tuning/target/gpu0-study/cuda134-root/usr/local/cuda-13.4
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" PATH="$CUDA_HOME/bin:$PATH" OMP_NUM_THREADS=2
exec "$PEARL_PY" "$@"
