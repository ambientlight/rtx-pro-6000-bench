# SM120 activation commitment and noise

The eager Pearl projection can keep the quantized input, activation Merkle commitment, v3 seed, and expanded noise on the serving GPU. Set `PEARL_GPU_PREPARE=1` and build the extended operator below. The default reference path remains available.

Each forward uses a keyed BLAKE3 tree over the actual signed activation bytes at K=2,048. A missing sibling at an odd level is carried to the next level. The root and per-job committed weight root feed Pearl's dimension-salted v3 seed derivation. A dense rank-128 basis and a sparse permutation expand to exactly the `M × 2048` signed noise bytes consumed by the original tensor-core projection. The CUDA code reuses the pinned Pearl BLAKE3 compression and commitment-hash headers. The tested miner submodule pins Pearl to `82e6eee7076aafe10123e08db78f100d270c0582` and CUTLASS to `291300ffffa3533a78ee104f08a8490a29ce9ccb`. [CUDA kernels](prepare.cu)

The projection launches preparation and GEMM on the serving stream. A separate stream copies the immutable quantized activation, commitment metadata and transcript to pinned host buffers for proof verification. The proof thread rebuilds the Rust activation tree and checks its root and seed before scanning the transcript or opening a winner. A retained weight handle keeps proofs valid across job replacement. Saturation and missing jobs continue to use the existing calibrated plain projection. [GPU adapter](activation.py) · [Proof ownership](reference.rs) · [Serving hook](../../checkpoint_projection.py)

## Measurements

The following medians use 32 repetitions on a single RTX PRO 6000 Blackwell, GPU 3, with the production F2Pool worker running at the existing 280 W limit. CPU measurements use two Rayon threads. GPU events cover device preparation on already-resident buffers. Tensor-core projection, data transfers, and proof verification are measured separately.

| Proof rows | Cached CPU preparation | GPU preparation | GPU graph replay |
|---|---:|---:|---:|
| 2, single-token encoding | 0.134 ms | 0.029 ms | 0.030 ms |
| 6, DSPARK target verification | 0.181 ms | 0.036 ms | 0.035 ms |
| 2,048, prompt prefill | 9.838 ms | 0.060 ms | 0.059 ms |

The checker compared every root, seed, and activation noise byte against the Rust reference at M=2, 6, 10, 16 and 2,048. The same captured graph produced correct results after input and job changes. A retained, already-retired weight generation produced a verified 38,113-byte witness, and changing one bit of the copied root caused verification to reject it. [Microbenchmark and validation receipt](results-sm120.json)

Twelve matched pairs per mode measured the full warm eager projection through serving-stream output completion with fresh BF16 activations each time. Every pair produced bit-identical outputs and byte-identical verified witnesses. The timed interval includes Torch/Python dispatch, quantization, preparation, projection, scaling and scheduling the proof. CPU proof work drained between samples. The measurements cover one checkpoint shard on a contended GPU. Tensor-parallel collectives, first-token latency and production inference still require separate measurements.

| Serving rows | CPU preparation path | GPU preparation path | Median paired difference |
|---|---:|---:|---:|
| 1 | 7.41 ms | 2.64 ms | −4.77 ms |
| 6 | 6.20 ms | 2.64 ms | −3.57 ms |
| 2,048 | 13.20 ms | 4.58 ms | −8.62 ms |

[Paired eager samples](eager-pairs.json)

The pinned SGLang single-rank Stratum fixture accepted all three real projection proofs after the GPU change. The ordinary prefill, DSPARK verification and singleton proofs each matched the previous CPU-prepared witness SHA-256 under the same public job. Rotation, disconnect and proof saturation preserved calibrated outputs. [Native pool receipt](native-pool.json) · [CPU reference receipt](../../sglang/native-pool.json)

## Reproduce

From the repository root, use the same CPython, CUDA Torch, NumPy and staged SGLang environment as the [native projection fixture](../../sglang/README.md). The build script defaults to the local CUDA 13.4 toolkit, miner source and Rust 1.97.0 artifact directory. Set `CUDA_HOME`, `PEARL_GEMM_SRC`, or `PEARL_RUST_DEPS` to use other locations. Use a fresh build directory because the extended Rust ABI and GPU library must come from the same build.

```sh
proto="$PWD/work/pearl-inference-20261123/prototype"
dir="$proto/operator/gpu_prepare"
export PEARL_OPERATOR_BUILD=$(mktemp -d)
"$dir/build.sh" "$PEARL_OPERATOR_BUILD"
export RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PEARL_CHECKPOINT_DIR=/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017
export CUDA_VISIBLE_DEVICES=3
python "$dir/check.py" --build "$PEARL_OPERATOR_BUILD" --model "$PEARL_CHECKPOINT_DIR" --receipt "$PEARL_OPERATOR_BUILD/preparation.json"
python "$dir/compare_eager.py" --build "$PEARL_OPERATOR_BUILD" --model "$PEARL_CHECKPOINT_DIR" --receipt "$PEARL_OPERATOR_BUILD/pairs.json"

export PEARL_GPU_PREPARE=1 PEARL_PUBLIC_REPLAY=1 PEARL_WOB_MODULE=checkpoint_projection
verifier=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-v1/target/v1-build/release/examples/verify_receipt
python "$proto/sglang/replay_pool_projection.py" --rank 3 --verifier "$verifier" --receipt "$PEARL_OPERATOR_BUILD/native-pool.json"
```

The low-level GPU preparation ABI supports CUDA graph capture with caller-owned scratch and stable launch addresses. The complete serving hook still runs in eager mode. Production CUDA graph integration needs slots for per-replay job state, padded-row eligibility and asynchronous retirement. CPU proof processing still copies each eligible quantized input and transcript to host. This prototype submits at most one winner per local multiply.
