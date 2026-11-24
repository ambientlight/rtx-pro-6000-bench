#!/usr/bin/env bash
set -eu
candidate=/mnt/hot/ambientlight/repos/sglang-upstream-sm120
runtime=/mnt/hot/ambientlight/sglang-sm120-runtime/base-unpacked/rootfs
pkgs="$runtime/opt/sglang/lib/python3.12/site-packages"
fixture_root=/tmp/pearl-upstream-native
export PYTHONPATH="$candidate/python:$fixture_root/oci-site:$pkgs:$fixture_root/fi/flashinfer_jit_cache-0.7.4+cu130.data/purelib:$fixture_root/fi/flashinfer_jit_cache_sm120f-0.7.4+cu130.data/purelib:$fixture_root/fi"
export PEARL_TEST_SITE="$pkgs" CUDA_VISIBLE_DEVICES=GPU-b000b08e-8683-d776-925b-a9704b764ddc CUDA_DEVICE_ORDER=PCI_BUS_ID RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export SGLANG_PLUGINS=no_plugins SGLANG_SM120_FLASHMLA_BACKEND=flashinfer FLASHINFER_CUDA_ARCH_LIST=12.0f MAX_JOBS=4
export CUDA_HOME=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-gpu0-tuning/target/gpu0-study/cuda134-root/usr/local/cuda-13.4
export PATH="$runtime/opt/sglang/bin:$CUDA_HOME/bin:$pkgs/nvidia/cu13/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$pkgs/nvidia/cu13/lib:$pkgs/nvidia/cudnn/lib:$pkgs/nvidia/nccl/lib:$runtime/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
exec "$runtime/opt/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12" "$@"
