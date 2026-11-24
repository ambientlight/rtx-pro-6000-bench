# DeepSeek TP4 with Pearl inside SGLang

The four SM120 schedulers own the four CUDA contexts. Each scheduler loads the K780 Rust library and gives its mining thread the scheduler's existing context. Each idle thread replays a short 188-tile CUDA graph on a low-priority stream. A CPU broker closes admission to new mining slices and waits for the GPU completion event on each outstanding slice. After the last inference batch, all four ranks fence their streams and reopen the gate.

In `joined` mode, layer zero's `attn.wo_b` computes both the projection consumed by DeepSeek and Pearl transcripts. This code runs inside the full DSPARK decode graph and eligible prefill calls. Direct and inference work have separate F2Pool connections. An independent local verifier checks a winning inference witness before forwarding it. That witness contains small numerical samples from the request. The owner enabled these submissions for the production endpoint.

In `non-joined` mode, prefill and decode use the checkpoint's original projection. Only the resident idle K780 threads perform mining. Both modes use the same scheduler admission, CUDA contexts, short idle graphs and post-inference release. Select the mode before starting the service so SGLang captures the corresponding inference graphs. On existing installations without a selection, startup uses `joined`.

```sh
runtime=/mnt/hot/ambientlight/sglang-sm120-runtime
code="$runtime/native-production-code/docker/deepseek-v41/upstream/graph-275/f2pool"
python3 "$code/native_service.py" set-mode non-joined
systemctl --user restart dsv41-pearl.service
cat "$runtime/native-production/current/ready.json"
```

Use `set-mode joined` in the same command to restore inference mining. The selection persists across service restarts.

## Non-joined production results on November 24

The canonical Responses endpoint runs in non-joined mode. All four schedulers reported their borrowed CUDA context, the idle K780 library and a disabled Pearl inference operator. Each rank replayed the full DSPARK decode graph. All four startup smokes passed, and F2Pool acknowledged direct work from all four schedulers. [Measurement and activation evidence](non-joined-qualification.json)

| Paid K780 on four GPUs at 275 W each | Sustained TMAC/s | Wall time | F2Pool accepted | Rejected | Peak GPU temperature |
|---|---:|---:|---:|---:|---:|
| Original standalone executable | 1,179.90 | 210.0 s | 64 | 0 | 90°C |
| Native idle threads in SGLang | 1,181.24 | 210.2 s | 62 | 0 | 88°C |

Both miners ran the K780 v1 kernel with K=65,536 and 188 tiles per slice against live jobs. The comparison uses completed MAC counters after a 60-second warmup and seven consecutive 30-second windows. The embedded rate was 100.11% of standalone. Both arms averaged 275 W on each card and their hardware and software thermal slowdown counters stayed flat during the measured mining windows. The first six-second idle samples on three non-joined launches ranged from 1,229.37 to 1,234.60 TMAC/s. The archived six-second standalone calibration measured 1,233.67 TMAC/s. [Matched mining measurements](non-joined-qualification.json) · [Original calibration](../../direct-calibration.json)

| Uncached input tokens | Output | Non-joined first token | Non-joined server prefill tok/s | Non-joined client decode tok/s | Joined client decode tok/s in this run |
|---|---:|---:|---:|---:|---:|
| 1,354–1,359 | 256 | 0.172 s | 8,691 | 115.19 | 116.85 |
| 9,418–9,424 | 256 | 0.963 s | 10,159 | 115.97 | 113.89 |
| 32,779 | 8,192 | 3.309 s | 10,220 | 116.71 | 112.39 |
| 131,387 | 512 | 13.583 s | 9,932 | 112.71 | 111.73 |
| 200,305 | 512 | 21.351 s | 9,643 | 106.10 | Interrupted at cutoff |

The short rows are medians of four requests each. Each long row is one completion with fresh sampling. Prefill uses the server's forward time. Decode uses the client stream after the first token. The ten completed pairs had identical prompt hashes, input lengths and output budgets. The previous separately hosted fast-switch miners recorded 112.78, 112.07 and 117.72 decode tok/s on the same three long prompts. [Serving measurements](non-joined-qualification.json) · [Historical runs](qualification.json)

Over 40 requests the p99 from broker demand to all four miners parked was 1.93 ms. From eligibility to all four GPUs mining it was 0.37 ms. From the final streamed event to the first mining kernel it was 19.41 ms. An additional request immediately after sustained mining parked all four in 1.65 ms and returned to the first mining kernel 19.03 ms after the final event. [Handoff timings](non-joined-qualification.json)

The 92°C supervisor cutoff triggered three times in these long-context tests. Joined serving stopped during the 200K prefill. Non-joined serving completed 200K and stopped during the following 131K prefill. A second non-joined run stopped about a second after the full 32K/8,192 response finished. The successful 131K and 32K samples used benchmark-only cooling intervals to 70°C between requests. All four miners had resumed before each timed request. The final active non-joined session passed the production and post-mining handoff checks. [Thermal readings and test conditions](non-joined-qualification.json)

## Native K780 calibration on November 24

The four schedulers can report the selected miner build in the `slice_profile` field of the `native_rank` startup event. The `nonce-readback` build uses the same CUDA kernels and 188-tile slices as the original deployment. It copies the nonce signal and its Merkle root into pinned host memory once per nonce and reuses those 96 bytes for the remaining slices.

The private miner revision `16d0056d220acc7ce922260baa4813d87274e512` builds this profile with `BPM_SLICE_PROFILE=nonce-readback`. The staged library SHA-256 is `95e160cc8a794ec411818dff0edeecdbb91ac242636504a687be992632bb5299`. On the canonical non-joined deployment, matched 210-second paid intervals measured 1,180.00 TMAC/s before and 1,180.99 after. F2Pool acknowledged shares on every rank, with 60 accepts before and 42 after and zero rejects in either interval. The throughput figures use completed tiles. These short intervals provide limited precision for a payout estimate.

For a comparable paid measurement, collect a native Responses transition log and per-GPU power, clock and thermal samples during one service run. The following command selects a continuous idle interval after the last request, calculates completed mining work on each rank, counts pool acknowledgements and matches each handoff with the GPU start signal. A 210-second window provides seven 30-second rate samples. The headerless `nvidia-smi` CSV columns are timestamp, index, GPU utilization, temperature, power draw, SM clock, memory clock, and the software and hardware thermal slowdown counters in that order. [Measurement helper](../native_calibration_sample.py)

```sh
calibration_dir="$runtime/native-integration/docker/deepseek-v41/upstream/graph-275/f2pool"
python3 "$calibration_dir/native_calibration_sample.py" --run "$run" --requests "$requests" \
    --telemetry "$gpu_csv" --seconds 210 --warm 20 --out "$receipt"
```

## Joined qualification at 275 W per GPU on November 23

| Four-card operating mode | Measured mining rate | Fraction of previous standalone K780 |
|---|---:|---:|
| Previous standalone calibration, 6.0 s | 1,233.67 TMAC/s | 100% |
| Built-in direct mining while idle, 487.4 s | 1,177.64 TMAC/s | 95.46% |
| Joined inference during prefill | 0.38480 TMAC/s | 0.03119% |
| Joined inference during decode | 0.008295 TMAC/s | 0.0006724% |

One TMAC/s is one trillion multiply-accumulates per second. The direct rate uses completed K780 tile counters. The idle row uses the longest common uninterrupted four-GPU idle interval. All four boards averaged about 275 W and 95% GPU utilization. The hottest sample in that interval was 90°C and all four software and hardware thermal counters stayed flat. An earlier eight-second idle sample reached 1,227.85 TMAC/s, or 99.53%. The mining percentages during inference use the independently scanned MAC-equivalent work divided by native prefill-forward or inter-token time across six uncached 256-output Responses requests. Each of those requests also paid the cost of the model's other layers. [Hardware measurements](qualification.json) · [Standalone calibration](../../direct-calibration.json)

| Uncached input | Output | First token | Effective prefill, tok/s | Decode, tok/s | Previous fast-switch decode, tok/s | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 32,779 | 8,192 | 3.62 s | 9,058 | 111.16 | 112.78 | −1.4% |
| 131,387 | 512 | 13.84 s | 9,492 | 103.56 | 112.07 | −7.6% |
| 200,305 | 512 | 21.35 s | 9,383 | 99.28 | 117.72 | −15.7% |

Each long-context row replays the original prompt hash and output budget once. Effective prefill divides the uncached input by time to the first streamed token. The previous fast-switch arm ran the native checkpoint projection and parked its separate miners. Its previous joined counterpart achieved 112.22, 107.47 and 105.08 decode tokens/s on the same three inputs. These samples have independently generated completions. DSPARK throughput depends on which tokens the draft model predicts correctly. [Previous graph runs](../../performance.md) · [Current replay and source hashes](qualification.json)

Three warmed 1.35K-input samples gave median **7,565 native prefill tok/s**, **195 ms** to first token and **120.1 decode tok/s**. Three warmed 9.42K samples gave **9,888 native prefill tok/s**, **992 ms** and **111.7 decode tok/s**. Native prefill uses the server's forward-time counter. On an earlier matched ten-prompt replay, median client decode was **96.3 tok/s on the incumbent** and **114.1 tok/s on the new build**. The first new-build 9.42K request took **6.56 s** to first token. The next took **0.98 s**. In a separate interleaved eight-pair comparison within the new build, permitting the idle miner changed the median per-pair decode rate by **+1.89%**, with individual pairs between **−9.81%** and **+15.64%**. Both arms of that test ran the joined projection. The ten incumbent comparisons and the eight same-build comparisons produced different completion hashes in every pair. Model quality parity remains unmeasured. [Serving and paired results](qualification.json)

| Handoff, 110 requests | Median | p99 |
|---|---:|---:|
| HTTP request start to all four direct miners parked | 8.56 ms | 28.13 ms |
| Broker receives demand to all four parked | 1.60 ms | 1.90 ms |
| Last response event to first direct work on GPU | 18.84 ms | 20.74 ms |
| All ranks release the gate to all four GPUs running direct work | 0.315 ms | 0.393 ms |

During qualification F2Pool acknowledged all **257** direct submissions, with 68, 63, 64 and 62 from the four ranks. At assigned F2Pool difficulty there were zero winning inference submissions. At a separate easier loopback target the independent verifier accepted **39 prefill and 12 decode proofs**, representing every rank in both phases. Pool acknowledgements measure accepted work. Coin payouts are accounted separately by the pool. A client disconnect during a concurrent request also passed. [Qualification receipt](qualification.json)

## Production operation

The enabled user unit `dsv41-pearl.service` runs the existing checkpoint on `127.0.0.1:8000` with the original loopback authentication behavior, four startup smokes and content-free serving diagnostics. Each owner-only session under `native-production/runs` holds its native rank counters. Joined sessions also run the CPU inference gateway and keep its credential and receipts in that session. `native-production/current` identifies the current run. The thermal cutoff is 92°C on any card. The service's failure handler starts the original Docker deployment. A deliberate rollback uses `systemctl --user start dsv41-pearl-fallback.service`; systemd orders model shutdown before the stock instance starts. `systemctl --user start dsv41-pearl.service` reactivates the native deployment.

The serving source starts at official SGLang `c64f2256d583` (2026-11-23 16:00:36 Pacific) plus the SM120 reconstruction. The incremental source bundles in this directory apply to the pinned graph SGLang and K780 parents. Use the [graph restoration instructions](../../README.md) for those bases. [Pinned graph inputs](../../build-lock.json)

| Component | Parent | Native commit |
|---|---|---|
| SGLang | `a384e33a52d7` | `f657eb5ddf4e` |
| K780 Rust/CUDA miner | `c9bc83799ed2` | `cb4bcdd4cf34` |

The following commands rebuild the native library in disposable checkouts on the prepared host. The installed source checkouts supply the bundle parents. The miner checkout also needs its pinned Pearl and CUTLASS submodules.

```sh
runtime=/mnt/hot/ambientlight/sglang-sm120-runtime
f="$runtime/native-integration/docker/deepseek-v41/upstream/graph-275/f2pool"
scratch=$(mktemp -d "$runtime/verify-native.XXXXXX")
for component in sglang miner; do
    bundle=sglang
    if [ "$component" = miner ]; then bundle=k780; fi
    git clone --shared --no-checkout "$runtime/native-$component" "$scratch/$component"
    git -C "$scratch/$component" bundle verify "$f/native/$bundle.bundle"
    git -C "$scratch/$component" fetch "$f/native/$bundle.bundle" refs/heads/sm120/sglang-native-idle
    git -C "$scratch/$component" checkout --detach FETCH_HEAD
done
git -C "$scratch/miner" submodule update --init --recursive
export CUDA_HOME=/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-gpu0-tuning/target/gpu0-study/cuda134-root/usr/local/cuda-13.4
export CPATH="$runtime/full-canary/math-headers"
export BPM_VARIANTS_FILE="$runtime/native-integration/work/pearl-inference-20261123/prototype/miner/v1.txt"
export CARGO_TARGET_DIR="$scratch/target" RUSTC_BOOTSTRAP=1
(cd "$scratch/miner" && RUSTFLAGS='-C panic=abort' cargo +1.97.0 build --offline --locked --release --lib -j 16)
```

The library used in the original joined qualification had SHA-256 `e9ca9d12e105f1e075e45148ab7e84e3b240a20564fd23a00b422a50a611c5fc`. For a fresh production release, use a pinned checkout of the operator repository directly under the runtime directory and run the following. Staging refuses to overwrite a populated production configuration. Existing production installations can be verified directly with `native_live.py`. It sends a small public Responses fixture and waits for a real F2Pool acknowledgement on every GPU.

```sh
code="$runtime/native-production-code/docker/deepseek-v41/upstream/graph-275/f2pool"
PEARL_NATIVE_IDLE_CONFIG="$runtime/native-production/config.json" python3 "$code/native_host.py" prepare \
    --source "$scratch/sglang" --library "$CARGO_TARGET_DIR/release/libpearl_idle.so"
python3 "$code/native_service.py" install
systemctl --user start dsv41-pearl.service
python3 "$code/native_live.py" --out "$runtime/native-production/current/activation.json"
```

When promoting from a live private qualification deployment, use `PEARL_NATIVE_IDLE_CONFIG="$runtime/native-qualification/config.json" python3 "$code/fullmodel.py" promote` after installing the production unit. This serializes the old rollback guard and the new service. The source library and serving overlay are checked against their staged hashes at every production start.

On November 23 the first promotion overlapped the private guard restoring the stock container during shutdown. The stock container was stopped while its API was still starting, and the native deployment started against exclusive GPUs. The production receipt records the four scheduler PIDs and CUDA-context matches, four passing startup smokes, advancing scans in both phases and F2Pool acknowledgements on each GPU. The promotion and fallback regression tests cover the overlapping-start case. This activation did not include a host reboot. [Production activation](production.json)

The superseded system-level `pearl-f2pool.service` was stopped but enabled at activation. This host requires interactive administrator authentication for `sudo systemctl disable pearl-f2pool.service`. Disable that unit before rebooting so only the enabled native service claims the GPUs.

### Workload-based memory clocks on htower

The clock controller observes the live native broker's idle permit. Five continuous idle seconds open a local socket that holds the four GPUs at 7,001 MHz. A new inference claim closes the socket and requests automatic memory clocks while the miner drains. Mining resumes as soon as the broker grants the next idle permit. The controller runs as the serving user. Systemd executes two fixed NVIDIA commands for the four named GPU UUIDs. Closing the connection or losing the controller triggers the reset. [Controller](clock/watch_clock.py) · [Clock service](clock/dsv41-memory-clock@.service)

As `ambientlight` on `htower`, install the root-owned socket once. Opening the listener leaves the current clock policy in effect until the first mining lease.

```sh
f=/mnt/hot/ambientlight/sglang-sm120-runtime/native-integration/docker/deepseek-v41/upstream/graph-275/f2pool
bash "$f/native/clock/install-clock-lease.sh"
```

To qualify the policy on the running non-joined service, use a fresh result directory. The first arm uses automatic clocks. The second runs its own clock controller and releases it at the end. The comparison checks prompt hashes, token counts, CUDA graphs, scheduler and pool identities, and a sustained paid mining window.

```sh
trial=$(mktemp -d)
python3 "$f/native_clock_bench.py" automatic "$trial/automatic.json"
python3 "$f/native_clock_trial.py" "$trial/automatic.json" "$trial"
```

The user unit can run the controller across serving restarts after qualification. Install `watch_clock.py` as `~/.local/libexec/dsv41-watch-clock.py` and the user unit under `~/.config/systemd/user/`. Then reload the user manager and enable `dsv41-adaptive-clock.service`. Stopping the user unit releases its clock lease. [Serving-side systemd unit](clock/dsv41-adaptive-clock.service)
