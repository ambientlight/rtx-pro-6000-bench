# Pearl v3 integer projection on SM120

The GPU operator consumes contiguous signed INT7 activations and weights, plus the job-bound signed-byte noise for both operands. Tensor-core instructions compute the cumulative noised dot product and two correction terms. Every 128 K elements the kernel folds the cumulative products into Pearl's ticket transcript. Subtracting the corrections from the same accumulators yields the model's integer projection. At K=2,048 the 16 checkpoints fill the 16 transcript words exactly once.

The prefill layout uses the released 2 × 64 sampled pattern. Decode uses adjacent 2 × 16 tiles. A six-row DSPARK-shaped launch emits 960 real tickets per rank. The kernel excludes inactive rows in its 16-row internal tensor-core tile from both the transcript and the returned outputs. See [device ABI](pearl_wo_b.h) and [CUDA implementation](pearl_wo_b.cu).

## Run the synthetic hardware check

From the repository root on the prototype host, select the intended GPU explicitly. Tests use one card, 76-byte synthetic headers, a finite target and deterministic test matrices.

```bash
operator=work/pearl-inference-20261123/prototype/operator
build=$(mktemp -d)
"$operator/build.sh" "$build"
export CUDA_VISIBLE_DEVICES=GPU-ed6c5013-1e0a-ec80-be70-9aab1629ba5d
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=4
"$build/check" decode
"$build/check" prefill
```

The build script defaults to the local CUDA 13.4 toolkit and pinned Rust 1.97.0 Pearl reference artifacts. Set `CUDA_HOME` and `PEARL_RUST_DEPS` to relocate those dependencies. The host checker derives noise from the operand Merkle roots and immutable header. It validates every output integer and every transcript word against independent CPU arithmetic, tests sampled full Merkle openings with Pearl's reference verifier, and serializes a winning witness. Decode additionally checks that capture and replay reproduce every output and transcript word. The finite fixture difficulty produces both hits and misses.

## Recorded hardware result

| One shard on GPU 0 | Prefill | Single-stream DSPARK decode shape |
|---|---:|---:|
| M × N × K | 2,048 × 5,120 × 2,048 | 6 × 5,120 × 2,048 |
| Exact integer outputs checked | 10,485,760 | 30,720 |
| Exact checkpoint words checked | 1,310,720 | 15,360 |
| Hits / tickets at fixture target | 20,551 / 81,920 | 247 / 960 |
| Official verifier checks, including hit and miss | 10 | 10 |
| Winning bincode proof bytes | 140,609 | 38,113 |
| Warm kernel median / p95 over 32 repeats | 1.591 / 1.598 ms | 0.0450 / 0.0458 ms |

The four-card F2Pool miner was active and DeepSeek was resident. GPU 0 reported 100% utilization, 89 °C and a 280 W cap at the start. CUDA events time one prepared launch at a time on the same buffers. Serving integration requires measurements that also include input quantization, commitments, noise preparation, rescaling, the TP collective and hit handling. The timing above gives neither incremental first-token latency nor a pool credit rate. [Raw samples and hardware snapshot](results-sm120.json)

## Using a caller's buffers

`pearl_operator.Operator(build_dir)` loads both shared libraries. Its `launch` method enqueues on the caller's CUDA stream and leaves lifetime and completion to the caller. It can run inside CUDA graph capture. The Python wrapper uses standard-library ctypes. The `prepare`, `scan` and `proof` calls operate on CPU buffers. Preparation commits the actual signed operands, validates the signal range [-64, 64], and produces deterministic noise from those commitments and the header. Use a new commitment for changed activations, weights, dimensions, sampling configuration or header. The reference interface accepts a separate pool share target on scan and proof. [Python wrapper](pearl_operator.py) · [Host reference](reference.rs)

The proof witness opens quantized input and weight rows. The checker uses public synthetic operands and performs no network access. A serving caller must retain the immutable commitment until its asynchronous result is retired, enforce the traffic eligibility policy, and apply the chosen quantization scales before the model consumes the result. The whole-K integer output supports a single activation scale per row and weight scale per output or a separately calibrated quantizer with that property. Model blockwise K scales would require scale-group outputs or a corresponding kernel epilogue.

## Reuse committed weights across forwards

`op.prepare_weight(header, bt_host, noise_bt_host, n, compact)` commits the supplied shard once and fills the caller's B-noise buffer. Upload that noise once for the corresponding weight handle. Each call to `weight.prepare(a_host, noise_a_host, m)` commits the current activation, fills its A-noise buffer and returns an independent `Reference`. The weight handle can close as soon as a replacement is ready. Outstanding references retain the previous immutable weight tree for proof submission. The original `op.prepare(...)` remains available.

Salted v3 binds the weight root to N. Its noise seed includes the exact job header and sampling configuration. A change to the header, weight bytes, N or sampling pattern requires another weight handle and matching device-side weight noise. Different M values can reuse a handle when the sampling pattern and N are unchanged. Device-side retirement must also honor outstanding CUDA work that reads the old noise and weights.

Measured on the 7985WX with two Rayon threads. Each phase used five warmups and 32 alternating full and cached preparations with changed activation bytes.

| Shard | Old full host prep median | Cached host prep median / p95 | Initial weight prep, one observation |
|---|---:|---:|---:|
| 6 × 5,120 × 2,048 decode | 29.75 ms | 0.246 / 0.327 ms | 41.17 ms |
| 2,048 × 5,120 × 2,048 prefill | 42.27 ms | 10.08 / 12.20 ms | 34.04 ms |

Host preparation includes validation, Merkle commitment and generation of the current activation noise. Copying CPU noise to the GPU, quantization, stream scheduling and proof processing need separate measurement. The activation path still uses the CPU reference implementation. The five regression cases compare noise, transcript decisions and serialized proofs with a full rebuild across activation, batch-size, header and weight changes. Deferred proofs remain valid after weight handles close. The build also passed both full GPU oracle checks through the original ABI. [Full timing distributions and regression results](results-weight-cache.json)

```bash
RAYON_NUM_THREADS=2 python3 "$operator/check_weight_cache.py" "$build"
```

Use `--baseline-build` to compare against an earlier compiled reference library. Without that argument, the comparison rebuilds the entire commitment through the original public preparation entry point.

## Plain projection during job handoff

`Operator.launch_plain(a, bt, out, m, n, stream)` returns the calibrated INT7 matrix product when a mining job or its weight commitment is unavailable. It accepts the same contiguous INT7 input and transposed weight buffers, writes exactly `m × n` INT32 accumulators, and enqueues one tensor-core kernel on the caller's stream. Single-token decode uses one real input row. Column and row tails use predicated loads and stores.

On one SM120, the median of five warm CUDA event timings was 23.7 µs for a single row, 24.5 µs for six verification rows, and 0.955 ms for 2,048 prefill rows at N=5,120 and K=2,048. These kernel timings exclude quantization, output scaling, and the TP collective. All output coordinates matched independent CPU products, including the 10,485,760-output prefill case. Changed-input graph replays and a 17 × 257 tail case also passed. [Plain projection receipt](results-plain-sm120.json)

```sh
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=2 python "$operator/check_plain.py" "$build"
```
