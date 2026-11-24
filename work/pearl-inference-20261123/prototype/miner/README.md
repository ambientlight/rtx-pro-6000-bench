# Sliced SM120 Pearl worker

The prototype builds the deployed miner revision `157ca8bb` with a second K780 launch entry point for complete, contiguous ranges of globally numbered 160 × 256 output tiles. Each nonce still covers the original 26,112 tiles and full proof K. Nonce advance, commitment, selector construction and A noise run once at the start of a nonce. A pause retains the operands and next tile. A superseding pool job discards the pending cursor. Every completed slice hands its original hit IDs, nonce signal and A root to the existing proof worker.

## Build and run

Set `CUDA_HOME` to a CUDA 13.4 toolkit, with Rust 1.97.0 available. Give `build.sh` the local miner repository and a new checkout directory. It creates a detached worktree, checks out the pinned submodules and applies [the slice and control changes](sliced-worker.patch). The resulting Linux executable is `target/release/bpm`. [Build script](build.sh)

On this rig the built executable is `/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-sglang-prototype/target/cargo/release/bpm` and the CUDA 13.4 toolkit is `/mnt/hot/ambientlight/repos/mining/blackwell_pearl_miner-gpu0-tuning/target/gpu0-study/cuda134-root/usr/local/cuda-13.4`.

Run a bounded offline correctness check on an allocated GPU, substituting the executable path. `--power-limit` checks the existing board cap. It does not set it.

```bash
RAYON_NUM_THREADS=2 "$worker" --device 0 --kernel-variant v1 --k 65536 \
  --bench-job 71 --bench-nonce 1001 --bench-rounds 4 --slice-tiles 188 \
  --slice-check --power-limit 280 --max-temp 92
```

For control, add `--control-socket "$private_dir/gpu0.sock" --slice-tiles 188` to the regular v1 K780 pool worker command or to a finite offline benchmark. Create a fresh private directory with `mktemp -d`. Each worker serves a distinct socket. An offline benchmark exhausts its specified nonce count. Use a separate fresh socket when restarting a worker. Keep workers warm for the interactive handoff. Production F2Pool service definitions and credentials are independent of this build.

## Handoff

The broker opens each worker's Unix stream socket, sends `HOLD\n`, and receives `DRAINED\n` after the last committed operation's CUDA event completes. It sends `RELEASE\n` and receives `RELEASED\n`. Independent holds compose. The last release signals the condition variable immediately. A lost holding connection parks the worker until the supervisor replaces it and reconciles demand. For supervised serving, enable `--require-watch` on each worker and `--watch-workers` on the broker. A fresh supervised worker creates its socket before opening the GPU context. While holding a drained worker the broker sends `WATCH\n` on a second connection and receives `WATCHING\n`. That connection must remain open for mining. If it closes, the worker finishes its committed operation and parks. A replacement broker can reacquire HOLD and arm a new WATCH if the previous HOLD was cleanly released. Malformed new connections cannot release an existing reservation. `STATUS\n` returns one JSON line on a separate connection.

Pool setup participates in the same permit as tile ranges. Its Merkle setup, CUDA initialization and readback can take longer than a warm slice. On pool notification during a hold, the next granted operation rechecks the generation before touching the GPU. Proof and socket work continue on CPU while parked. When the proof queue is full the controlled mode logs dropped tickets so CPU backpressure cannot keep the GPU-control path waiting.

`release_ns`, `first_slice_begin_ns`, `first_gpu_seen_ns` and `first_slice_end_ns` use host CLOCK_MONOTONIC. For the first productive slice following a release, CUDA thread 0 in block 0 publishes a system-scope store to a mapped, aligned host word. The mining thread records when it sees that store, giving an upper bound on first kernel entry in the host clock domain. Host submission and event-completion timestamps bracket this observation. The synthetic fleet harness measures from immediately before it sends the rank idle events. A device hang, replaced worker, unavailable pool job or cold setup has a different latency distribution.

## Replays

- [Watchdog replay](watchdog.py) checks fresh worker parking, startup with an explicit reservation, disconnection while mining, parking at the next boundary and resuming through a replacement watch.
- [Single worker](measure.py) checks overlapping holds and randomized requests against an already running experimental worker. [Four-worker replay](fleet_replay.py) launches four finite offline workers and the [broker](../broker.py). Its four persistent rank clients simulate scheduler messages. It races repeated admissions against idle signals and checks that every worker's completed-slice counter stays frozen during foreground work. An optional stock HTTP smoke sends one short synthetic chat request under an external hold. With `--watch-workers --restart-broker` the replay kills its idle test broker, checks all four cursors stay frozen, starts a replacement and verifies supervised work resumes.
- [Loopback Stratum](loopback.py) exercises the real pool worker, keeps its socket open while mining pauses through an incoming job, and independently verifies the serialized proofs submitted on the local connection. Pass the original tuning `replay-job-changes.py` and `verify_receipt` paths. The fixture acknowledges locally submitted shares. Actual F2Pool credit requires a live pool submission.
- [Tile sweep](tile_sweep.py) compares multiple tile widths on one card, with randomized HOLD timing and alternating matched whole/sliced nonce runs at each width.
- [Paired throughput](speed.py) alternates full and sliced offline workers on one GPU. The original benchmark advances the nonce during eight warmups. The sliced arm offsets its initial nonce by eight so both arms measure identical jobs and hit sets.

## Measured on the four SM120 GPUs

With the existing F2Pool miners running, 320 warm broker admissions measured 3.34 ms median, 4.98 ms p99 and 6.20 ms maximum to drain all four experimental workers. After the synthetic scheduler ranks reported idle, the last of the four K780 kernels signaled actual GPU execution in 0.77 ms median, 2.83 ms p99 and 3.52 ms maximum. The fixture included 30 intentional races between an arrival and the last rank reporting idle. A separate 320-cycle build before the GPU marker measured 6.16 ms p99 to drain. On one fresh supervised GPU1 start, an admission during a pre-first-slice initialization operation took **327.63 ms** to drain. The next two cold holds took 11.58 and 17.90 ms. Warm latency figures apply after initialization and job setup.

On GPU1, with the same production contention, 24 matched K65536 nonces per arm retained 87.93% of full-round idle throughput at 188 tiles per slice. The original depth-two graph completed 2.02 nonces/s and the single-in-flight sliced worker completed 1.78 nonces/s. For each of the four pairs, both modes returned identical hit digests and three sampled CPU-verified proofs. Direct comparison of another four whole nonces matched all 257 hit IDs and their nonce commitments. The local Stratum replay independently verified 21 serialized wire proofs across five headers. A live failure replay killed the idle broker and confirmed all four cursors stayed frozen for 150 ms. A replacement broker reacquired each worker and all four resumed at their next slice.

The single-card width sweep used 300 randomized arrivals and three alternating throughput pairs of six matched nonces per arm at each width. The percentages below use the whole-round control from that width’s own pairs. Every pair matched all candidate digests.

| Tiles per slice | Single-GPU drain p99 | Maximum drain | Sliced rate relative to whole rounds |
|---|---:|---:|---:|
| 188 | 4.13 ms | 4.18 ms | 90.36% |
| 376 | 8.24 ms | 8.35 ms | 90.64% |
| 752 | 17.16 ms | 18.57 ms | 86.88% |

Doubling from 188 to 376 gained 0.28 percentage points in this sweep and doubled the measured p99 wait. The earlier four-pair measurement of 188 varied by more than that gain. The four-GPU reference remains at 188 tiles.

The fork adds Linux-only prototype control and defaults to the deployed v1 kernel in sliced mode. In a production rollout, a supervisor must reconcile serving demand before starting workers or restarting a failed controller. The recorded handoff includes warm local jobs and simulated scheduler messages. Read [the measured results](measurements.json) before assigning latency guarantees.
