# Public Pearl projection in the DSPARK target graph

This companion captures DeepSeek V4.1 layer 0 `attn.wo_b` in the native SM120 target-verification graph. Each TP4 rank reads its real 5,120 × 2,048 weight shard. The calibrated INT7 activations and weights, plus Pearl v3 noise, feed one projection. The same noisy tensor-core accumulators supply the ticket transcripts and the corrected output consumed by inference. The production prefill backend runs eagerly. Eligible prefill chunks use the existing joined projection. [Adapter](../../../../work/pearl-inference-20261123/prototype/checkpoint_graph.py) · [CUDA projection](../../../../work/pearl-inference-20261123/prototype/operator/pearl_wo_b.cu)

## Restore and build

The source bundle extends the adjacent locked production reconstruction by two commits. The final SGLang commit is `a384e33a52d7bac3e7fac1b0ad24ce9bd4a9f4da`. The operator commit in this companion is `d31d161942d12607663b6626f3b8b2ebabaab5c9`. The build uses CUDA 13.4, the pinned Pearl and CUTLASS sources, and the Rust 1.97.0 release dependency artifacts. [Build identities](build-lock.json) · [Base image and checkpoint](../release.json) · [Operator prerequisites](../../../../work/pearl-inference-20261123/prototype/operator/gpu_prepare/README.md)

Run from the companion checkout. `BASE_OCI_ROOTFS` points to the unpacked candidate OCI. `OUT` is a fresh writable build directory. The local miner toolchain supplies the CUDA and Rust paths used by the original operator build. The script accepts `CUDA_HOME`, `PEARL_GEMM_SRC` and `PEARL_RUST_DEPS` to relocate those inputs. The CUDA compiler needs cuBLAS development headers. The benchmark host provided them via `CPATH=/mnt/hot/ambientlight/sglang-sm120-runtime/full-canary/math-headers`.

```sh
package="$PWD/docker/deepseek-v41/upstream/graph-275"
proto="$PWD/work/pearl-inference-20261123/prototype"
mkdir -p "$OUT"
"$package/restore-source.sh" "$OUT/sglang"
"$package/stage-runtime.sh" "$OUT/sglang" "$BASE_OCI_ROOTFS" "$OUT/overlay"
"$proto/operator/gpu_prepare/build.sh" "$OUT/lib"
python3 "$OUT/sglang/test/manual/test_pearl_graph_hooks.py"
```

For an offline source restore, pass a local Git checkout containing base `481e8e66504cb680ccb1316eaf01ba09b03da5cd` as the second argument to `restore-source.sh`. Staging verifies the two original SGLang modules against the pinned base, copies the complete installed package, replaces those modules and patches a private copy of the checkpoint launcher. The candidate runtime supplies the native SM120 serving dependencies. [Staging script](stage-runtime.sh) · [Launcher change](boot-graph.patch)

## Run the projection fixture

Use the candidate's CUDA-enabled Python and make the CUDA 13 runtime libraries visible as in the base deployment. `PEARL_CHECKPOINT_DIR` points at the pinned 1017 checkpoint. Select the assigned physical SM120 card with `CUDA_VISIBLE_DEVICES` and use its visible ordinal inside the process.

```sh
for rows in 6 48; do
  python "$proto/operator/gpu_prepare/check_graph_projection.py" \
    --model "$PEARL_CHECKPOINT_DIR" --build "$OUT/lib" --device 0 --rank 0 --rows "$rows"
done
```

The six-row run scanned 1,920 tickets across two job-bearing replays. The 48-row run scanned 15,360. Each run independently opened and verified two winners from distinct fixture jobs and matched all six BF16 outputs against a CPU integer dot product. These tests use rank 0's checkpoint shard on a single GPU. Their easy local share target exercises the proof path. [Standalone hardware receipts](standalone.json)

## Serving comparison

Use the same protected loopback launcher, broker, direct workers, checkpoint, capture profile, TP4/EP4 ranks and power setting for both arms. Prepend `$OUT/overlay/pkg` and `$proto` to the existing candidate `PYTHONPATH`. Point its existing private boot wrapper at `$OUT/overlay/boot/boot.py`. Each rank needs its broker socket and an existing writable `PEARL_SHAPE_DIR`. Keep the arms' receipt directories separate. The control uses native checkpoint math and the same worker handoff with `PEARL_WOB_MODULE` and `PEARL_GRAPH_VERIFY` unset.

For the joined arm, use public replay inputs, a loopback v3 job peer, and these additional settings. The launcher resolves full decode graphs plus disabled prefill graphs, DSPARK block size five, six verify rows per request, maximum batch eight, and torch.compile disabled.

```sh
export PEARL_PUBLIC_REPLAY=1 PEARL_GRAPH_VERIFY=1 PEARL_GPU_PREPARE=1
export PEARL_WOB_MODULE=checkpoint_graph PEARL_OPERATOR_BUILD="$OUT/lib"
export PEARL_REPLAY_POOL_HOST=127.0.0.1 PEARL_REPLAY_POOL_PORT="$LOCAL_PEER_PORT"
export PEARL_CHECKPOINT_DIR
```

A device flag selects mining inside the captured graph only when the rank holds admission, all six physical rows per slot correspond to the target batch, an immutable job layout is ready, and a proof-worker slot is available. Other replays use the calibrated plain projection. The proof worker checks the copied activation, commitment and ticket against the originating weight handle. A new pool job invalidates stale submissions. The shape receipt exposes mined and fallback calls by real and executed rows. The counter receipt adds verified and winning tickets, local peer outcomes and eager prefill coverage. Each mined six-row step scans 960 compact tickets per rank, including all candidates evaluated by speculative verification.
