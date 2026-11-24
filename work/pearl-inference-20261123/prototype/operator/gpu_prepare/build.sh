#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
out=${1:?Usage: build.sh OUTPUT_DIR}
mkdir -p "$out"
toolkit=${CUDA_HOME:-/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-gpu0-tuning/target/gpu0-study/cuda134-root/usr/local/cuda-13.4}
deps=${PEARL_RUST_DEPS:-/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-v1/target/v1-build/release/deps}
gemm=${PEARL_GEMM_SRC:-/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-v1/third_party/pearl/miner/pearl-gemm}
# Produce a self-contained build directory, including the existing projection/ref checker.
"$src/../build.sh" "$out"
"$toolkit/bin/nvcc" -arch=sm_120 -O3 -std=c++17 --expt-relaxed-constexpr -Xcompiler=-fPIC \
  --shared --cudart=shared -I"$gemm/csrc" -I"$gemm/third_party/cutlass/include" -I"$toolkit/include/cccl" \
  -Xlinker -rpath -Xlinker "$toolkit/targets/x86_64-linux/lib" "$src/prepare.cu" -o "$out/libpearl_prepare.so"
rustc +1.97.0 --edition=2021 --crate-type=cdylib -O -C panic=abort \
  -L "dependency=$deps" --extern "zk_pow=$deps/libzk_pow-fc153e5b815e44e1.rlib" \
  --extern "pearl_blake3=$deps/libpearl_blake3-2e850ac2b7919e97.rlib" \
  --extern "bincode=$deps/libbincode-6e76495728f8471e.rlib" \
  "$src/reference.rs" -o "$out/libpearl_reference.so"
