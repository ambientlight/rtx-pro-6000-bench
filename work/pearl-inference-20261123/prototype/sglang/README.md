# SM120 SGLang prototype

The sparse overlay connects native request admission, each tensor-parallel scheduler, and the target model's first attention output projection to the local Pearl prototype. The reference setting is TP4, DP1, PP1, one scheduler process per GPU, and the Python tokenizer IPC path.

## Source and staging

The production lock identifies image `fa9642cbf037` and preserved source revision `38560c3d`. The checked source for this patch is the deployment overlay at `91051e1883862ac02426e397d44c9e4348439906`. The live endpoint reported `0.0.0.dev0`, TP4, and DSpark with five draft candidates. The image operator exposes image identity and restricts in-container filesystem access. Source hashes pin the six local inputs and the production build receipt independently records the tokenizer hash. [Production lock](../../../../docker/deepseek-v41/baseline.lock.json) · [Source hashes](source-sha256.json)

From the benchmark repository root, create both the sparse overlay and an isolated importable source tree.

```sh
proto="$PWD/work/pearl-inference-20261123/prototype"
source_repo="$PWD/../sglang-dsv41-production-overlay"
scratch=$(mktemp -d)
python3 "$proto/sglang/stage.py" "$source_repo" "$scratch/overlay"
mkdir "$scratch/source"
git -C "$source_repo" archive 91051e1883862ac02426e397d44c9e4348439906 python | tar -x -C "$scratch/source"
cp -r "$scratch/overlay/python/." "$scratch/source/python/"
PEARL_SGL_SOURCE="$source_repo" python3 "$proto/sglang/test_hooks.py" -v
```

The serving source belongs under `/sgl-workspace/sglang/python` in the isolated container. Stage the same source on the tokenizer and all TP ranks so the msgspec wire types agree.

## Request handoff

- Set `PEARL_BROKER_SOCKET` to the private UNIX socket for all tokenizer and scheduler processes. The tokenizer writes `LEASE` immediately after request normalization. CPU text tokenization can overlap the miner's outstanding slice. Multimodal, MossVL, LoRA and encoder-disaggregated inputs wait for the GPU drain before preprocessing.
- Every dispatch waits for the fleet's `DRAINED`, then creates opaque per-item child reservations before sending the ordinary SGLang IPC. Each rank announces `BUSY` and takes every child on its own persistent ordered socket. Streaming, batched requests, and parallel sampling keep the parent only until submission completes. Client cancellation closes its lease. An ambiguous IPC send failure keeps its child parked for inspection. Rank disconnect keeps the fleet held.
- Both normal and overlap scheduler loops reach `on_idle`. SGLang's full-idle predicate accounts for the waiting and grammar queues, pending overlap results, delayed transfers, cache operations and running work. Only after that predicate passes does each rank fence its own device and announce `IDLE`. The last rank and any newly queued admission determine when mining resumes. The device-wide fence covers the forward, communication, sampler and async GPU copies. The broker's watchdog supervises worker process loss.

## Projection and traces

- Set `PEARL_SHAPE_DIR` to an existing private writable directory for per-rank histograms of real and executed row counts. Eager layer 0 and replayed target decode graphs have separate entries. The host-level replay hook recognizes padding and compact ragged layouts with an available real host count. Compiled prefill graph shapes still require a host replay hook. Shape records contain counts only.
- Set `PEARL_WOB_MODULE=checkpoint_projection`, `PEARL_PUBLIC_REPLAY=1`, `PEARL_CHECKPOINT_DIR`, `PEARL_OPERATOR_BUILD`, and a CUDA 13 `PEARL_CUDART` for the eager public-replay layer. Both graph phases must use `--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled` and eager execution. Each eligible call uses the corrected projection result in the existing SGLang all-reduce. Padded shapes, warmup, draft layers, and unsupported modes use the original projection.
- The TP4 layer fixture uses the native SGLang `RowParallelLinear` constructor, its `nn.Module.__call__` path, and a native SGLang `GroupCoordinator` backed by NCCL. A small quant-method fixture carries the unpacked checkpoint BF16 comparison weights, and the standalone layer uses Torch's own CUDA capture predicate. Test inputs are deterministic synthetic activations. The full model executor, tokenizer, and live FP8 quant method require a separate integrated serving run. [GPU receipt](native-tp4.json)

Run the native fixture in a Python environment containing the serving SGLang dependencies, CUDA Torch, NumPy, and the built operator. On this host the successful run used CPython 3.12 with Torch 2.10.0+cu128, its existing venv packages, and an additional scratch installation of orjson.

```sh
export PYTHONPATH="$scratch/source/python:$proto${PYTHONPATH:+:$PYTHONPATH}"
export PEARL_WOB_MODULE=checkpoint_projection PEARL_PUBLIC_REPLAY=1
export PEARL_CHECKPOINT_DIR=/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017
export PEARL_OPERATOR_BUILD=$(mktemp -d)
"$proto/operator/build.sh" "$PEARL_OPERATOR_BUILD"
export PEARL_CUDART=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-gpu0-tuning/target/gpu0-study/cuda134-root/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcudart.so.13
export RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NCCL_BUFFSIZE=1048576
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc-per-node=4 "$proto/sglang/replay_projection.py"
```

To run the same three native projections between a real mining cut-in and resumption, give the joint replay the supervised slicer binary and four GPU IDs. This starts finite offline benchmark jobs in a private temporary directory and uses a local broker. Its rank-0 ingress fixture passes an opaque child to the actual rank hooks via a Gloo broadcast. It checks the four independent direct-mining slice counters before mining projections, after every phase, and after resumption. The joint run omits the optional dense BF16 comparator to reduce GPU memory use. Every GPU still compares the NCCL-reduced output to an independently computed integer result.

```sh
NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 NCCL_BUFFSIZE=262144 \
  python "$proto/sglang/joint_replay.py" /path/to/patched/bpm --gpu 0 1 2 3
```

The supervised joint trial drained all four workers in 4.84 ms. Starting immediately before rank 0's end-of-request GPU fence, all four miners completed their first resumed direct slice within 4.42 ms. Direct slice counts stayed at [3, 2, 8, 6] throughout the three projections and advanced to [4, 3, 9, 7] on resumption. This is a single observation under existing host load. The combined fixture briefly left 61 MiB free per card and each card returned to 1,722 MiB free after the fixture exited. [Joint GPU receipt](joint-tp4.json)

## Native projection with a pool job

The eager callback can obtain its immutable job and target from a public loopback Stratum session. Set `PEARL_REPLAY_POOL_PORT` to the local fixture's port and optionally set `PEARL_REPLAY_POOL_HOST` to its loopback IP. Each rank maintains one session. A background worker prepares the weight commitments for new jobs. While disconnected, awaiting preparation or short of proof capacity, the layer computes the plain INT7 result on the same stream. Submitted proofs retain their job identity through verification, and the client drops obsolete work on a tip change or disconnect. [Pool transport](../replay_pool.py) · [Plain kernel hardware checks](../operator/results-plain-sm120.json)

The single-shard native test starts its own receiver, changes jobs between prefill and decode, and verifies all three submitted witnesses with the separate deployed-source Rust verifier. It also checks exact output equality through job acquisition, proof backpressure and disconnect. Run this in the eager fixture environment above. [Native pool receipt](native-pool.json)

```bash
verifier=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-v1/target/v1-build/release/examples/verify_receipt
CUDA_VISIBLE_DEVICES=3 python "$proto/sglang/replay_pool_projection.py" --rank 3 --verifier "$verifier"
```
