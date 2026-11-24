# Early SM120 verification fixtures

These scripts reproduce the single-GPU kernel, HTTP and serving-contract checks used before the four-GPU joined benchmark. The shell wrappers retain the original harness paths, GPU selection and loopback ports. Their paths assume the corresponding source directories below are available under `/tmp`, alongside the reconstructed SGLang checkout, candidate OCI root filesystem, CUDA libraries and checkpoint. Service launchers and the low-memory worker guard belong on an assigned test GPU.

| Source | Verification |
|---|---|
| `pearl-upstream-native/`, `pearl_sgl_gpuenv` | Synthetic HTTP requests, warm-worker broker, handoff observations and local proof-server startup. The maintained fixture builder generates the small Llama and DeepSeek checkpoint variants. [Checkpoint builder](../../../make_smoke_model.py) · [Proof HTTP client](../../../check_http_projection.py) |
| `sgl-serving-oci-gpu0/` | Runs the attention-page and host Engram fixtures from the reconstructed checkout and records the actual loaded libraries. |
| `sgl-serving-final-qual.py`, `sgl-serving-test-env.sh`, `sgl-native-overt.py` | Runs the offline API contract tests with the checkpoint tokenizer and probes tool-parser behavior. |
| `deployed-anthropic-fixture.patch` | Adapts the previously deployed Anthropic test mocks to the new chat handler. It retains the wire serialization assertions and drops an identity assertion on the internal `strict` object. |
| `pearl-shapes/four_by_eight.rs` | Checks the original four-row/eight-column salted proof pattern against the Rust verifier. The other TP4 projection and singleton checks have their own sources. [TP4 shape probe](../../../../../../work/pearl-inference-20261123/wo_b_v3.rs) · [Singleton probe](../../../../../../work/pearl-inference-20261123/singleton_v3.rs) |

The original CPU serving runner accepts `ROOT` and uses the archived environment settings. Run from the benchmark checkout, with the scripts at their original paths.

```bash
source /tmp/sgl-serving-test-env.sh
"$PYCHECK" /tmp/sgl-serving-final-qual.py
```

To reproduce the baseline mock adaptations, run the following in an empty fixture directory with `OVERLAY` pointing to the previously deployed SGLang repository and `EARLY_FIXTURES` pointing to this archive.

```bash
git -C "$OVERLAY" cat-file blob 63a2307156e5ee5893727a6d2df73a84b3d44e68 > test_dsv41_deployed_compat.py
patch --fuzz=0 -p1 < "$EARLY_FIXTURES/deployed-anthropic-fixture.patch"
"$PYCHECK" -m pytest -q test_dsv41_deployed_compat.py
```

On the dedicated GPU with the matching host layout, the kernel metadata wrapper accepts the fixture script followed by its normal arguments. Both fixtures come from the SGLang source bundle. Supply a fresh receipt path for each run.

```bash
fixture=/tmp/sgl-serving-oci-gpu0
src=/mnt/hot/ambientlight/repos/sglang-upstream-sm120
"$fixture/run.sh" "$fixture/probe.py" "$src/test/manual/kernels/attention/check_dsv41_sm120_pages.py" --receipt "$OUT/attention.json"
"$fixture/run.sh" "$fixture/probe.py" "$src/test/manual/kernels/embeddings/check_dsv41_host_engram.py" "$MODEL" --receipt "$OUT/engram.json"
```
