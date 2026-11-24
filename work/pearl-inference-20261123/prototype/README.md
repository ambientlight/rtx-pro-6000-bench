# DeepSeek inference and Pearl mining on SM120

This prototype replaces one DeepSeek target-model projection, `layers.0.attn.wo_b`, on four RTX PRO 6000 Blackwell GPUs. The same integer products produce the model output and the Pearl ticket. SGLang still performs its tensor-parallel reduction. A host broker parks four warm direct miners at request admission and wakes them when all four serving ranks finish their GPU work. The production endpoint continues to run the stock model.

## Measurements

| Test | Result | Scope |
|---|---|---|
| Admission to all four direct miners drained | 3.34 ms median, **4.98 ms p99**, 6.20 ms maximum | 320 warm cycles with 30 arrival/resume races, simulated rank messages, real miners |
| Idle notification to first mining kernel observed on all four cards | 0.77 ms median, **2.83 ms p99**, 3.52 ms maximum | The last GPU signals kernel entry through mapped memory. The host timestamp bounds its start from above. |
| Standalone mining while idle | The 188-tile worker retained **87.9%** of the whole-round rate on GPU 1, and **90.4%** on a repeat. | The first run used four alternating pairs and found the same 1,541 candidates in both arms. In the sweep, 376 tiles retained 90.6% but increased single-worker p99 drain from 4.13 ms to 8.24 ms. We kept 188. |
| Joint serving/mining handoff | 4.84 and 5.17 ms to drain. All four miners stayed parked through three native SGLang forwards, then completed their first resumed slices within 4.42 and 2.43 ms of rank 0 starting its final device fence. | Two supervised TP4 runs with real miners, rank fences, NCCL and twelve verified witnesses each |
| Latest upstream same-GPU HTTP handoff | Across 48 warm requests, cut-in was 4.07 ms median / 7.22 ms p95. Release to mining execution was 2.11 ms median / 2.23 ms p95. | Real miner and two-layer DeepSeek TP1 on the same card. In 32 randomized streaming pairs, time-to-first-token increased 0.34 ms median and 10.96 ms p95. |
| One upstream DeepSeek HTTP request earning prefill and singleton-decode test shares | Two independent v3 acknowledgments and all 163,840 disclosed weight bytes matched the real checkpoint shard. | Sixteen public prompt tokens, two generated tokens, actual layer 0 projection in a two-layer TP1 canary with dummy remaining weights and a local receiving peer. |
| Cold worker restart | A request that arrived during GPU initialization waited **327.63 ms** to drain. | Actual single-card cold counterexample. Both directions of the warm benchmark assume a resident worker and a valid pool job. |
| Arrival to first streamed text on stock DeepSeek | 578 ms median with miners preheld, 533 ms with demand-triggered drain. Paired differences ranged from −1,090 to +145 ms. | 12 randomized matched pairs, same 528-token prompt with 512 cached tokens. The external admission gate took a median of 0.18 and 4.08 ms respectively. The variation exceeds the gate cost. |
| Corrected prefill, DSPARK verification and ordinary decode | All 117,760 distinct reduced BF16 coordinates matched the independent INT7 reference on all four ranks. Twelve full witnesses verified locally. | Actual patched SGLang `RowParallelLinear` and NCCL, checkpoint weights, synthetic activations, M=16, 6 and 1 |
| Full-size prefill and DSPARK projection | 1.591 ms and 0.045 ms median for the respective kernels | M=2,048 and M=6 on one card, warm GPU buffers. Excludes quantization, commitment, copies, scaling, proof work and NCCL. |
| Warm activation commitment and noise preparation | For 2,048 rows the median fell from 9.84 ms on CPU to 0.0603 ms on GPU, and for six rows from 0.181 to 0.0362 ms. | Single checkpoint shard with resident weights. GPU execution time applies only to preparing the commitment and noise. |
| Latest upstream stock versus mining-enabled projection | Medians of 3.085 → 4.170 ms for 2,048 prefill rows, 2.425 → 2.455 ms for verify-six, and 2.653 → 2.474 ms for singleton decode. | Ten alternating pairs per shape on one shared SM120. Excludes TP reduction, HTTP and asynchronous proof acknowledgment. 39 submitted proofs verified on two loopback jobs. |
| Projection error against a software block-FP8 reference | Relative L2 of 3.46% for prefill, 3.45% for six rows and 3.50% for one row | Four real checkpoint shards, seeded BF16 inputs, FP32 reference accumulation. The full-size test sampled 16 prefill rows for this comparison. |
| Inference-origin proof submission | The local Stratum receiver accepted a 140,609-byte prefill proof and a 38,113-byte decode proof under two successive jobs. | Independent deployed-source verifier, local test targets, synthetic activations |
| Native SGLang projection connected to Stratum | One prefill, one DSPARK verification and one ordinary decode submission acknowledged under two successive jobs. The output stayed bit-exact to plain INT7 when the pool disconnected or the proof queue was full. | One native rank, 117,760 BF16 reference coordinates across the three modes, separate CPU verifier, local test pool |

The original four F2Pool workers were active throughout. Stopping their system service required interactive administrator authentication. The timings reflect that contention and the existing 280 W, 7,001 MHz memory-clock configuration. The integer quantizer changes the checkpoint's numerical behavior. Token quality, production throughput, end-to-end latency percentiles and difficulty-weighted F2Pool credit require a full-model canary.

[Latest upstream deployment](../../../docker/deepseek-v41/upstream/README.md) · [Same-GPU HTTP switching](../../../docker/deepseek-v41/upstream/receipts/sm120-http-same-gpu.json) · [Same-request prefill/decode shares](../../../docker/deepseek-v41/upstream/receipts/sm120-http-projection.json) · [Latest upstream projection results](../../../docker/deepseek-v41/upstream/receipts/sm120-projection.json) · [GPU preparation](operator/gpu_prepare/README.md) · [Joint serving/mining run](sglang/joint-tp4.json) · [Native TP4 results](sglang/native-tp4.json) · [SM120 kernel](operator/results-sm120.json) · [Cached host preparation](operator/results-weight-cache.json) · [Checkpoint evaluation](checkpoint-result.json) · [Native SGLang Stratum acceptance](sglang/native-pool.json) · [Full-size wire replay](projection-wire-result.json) · [Four-card handoff and slice sweep](miner/measurements.json) · [Paired streaming latency](ttft-result.json)

## Running the prototype

From this directory, build a private operator directory, then exercise the checkpoint and loopback submission. The build uses the local CUDA 13.4 toolkit and pinned Rust verifier artifacts. The Python replay needs NumPy. `--devices 0 1 2 3` addresses the selected CUDA-visible devices in TP-rank order.

```bash
build=$(mktemp -d)
./operator/build.sh "$build"
export RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=3
export PEARL_CUDART=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-gpu0-tuning/target/gpu0-study/cuda134-root/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcudart.so.13
model=/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017
verifier=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-v1/target/v1-build/release/examples/verify_receipt
python3 checkpoint_replay.py --model "$model" --build "$build" --devices 0 1 2 3
python3 projection_wire.py --model "$model" --build "$build" --verifier "$verifier"
python3 -m unittest discover -s . -p 'test_*.py' -v
```

The [SGLang fixture](sglang/README.md) stages a sparse patch against pinned source and runs the real layer. The [GPU operator](operator/README.md) has an exhaustive integer and transcript checker. The [worker instructions](miner/README.md) build the sliced miner from the deployed source revision and reproduce the handoff tests.

For managed workers, run each sliced worker with `--require-watch` and its own `--control-socket`. Start [the broker](broker.py) with `--watch-workers`, one `--worker` path per GPU and a `--socket` in a private directory. In this mode workers start parked. The broker drains each worker, arms a persistent watchdog connection, and grants idle work only after all ranks have reported idle. Losing the watchdog parks mining. An unexpected disconnect of a holding client also parks mining and requires replacing the faulted worker. Point `PEARL_BROKER_SOCKET` at the broker when running the instrumented serving stack.

## Remaining serving work

- The eager adapter supports on-device activation commitment and noise generation with `PEARL_GPU_PREPARE=1`. Integrate job rotation, padded-row handling and buffer retirement into SGLang’s production DSPARK graph replay. Captured preparation kernels have passed a changing-job test in isolation.
- Extend the exact plain INT7 fallback to graph padding and other currently excluded forwards, then evaluate model logits, task scores and DSPARK acceptance against stock. A mined single-row decode uses two integer planes to preserve the exact calibrated projection, incurring two products for this layer.
- Qualify the external pool's supported job shapes, target handling and difficulty-weighted credit. The eager callback accepts changing jobs from a public loopback session, binds each proof to its originating job and tracks acknowledgments by phase. It currently submits at most one winning ticket per local matrix multiply. Proofs open quantized activation and weight rows, and all inference inputs in these replays are synthetic.
