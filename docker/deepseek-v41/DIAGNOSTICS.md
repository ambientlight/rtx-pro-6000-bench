# CUDA / NCCL crash capture

The user-requested 0.87 diagnostic profile keeps the existing serving stack,
model, half-native context, RAM offload, TP4/EP4, concurrency 8, automatic KV
sizing and disabled expandable segments. It does not increase SWA capacity.

## September 18 hardening — active

`ambientlight/dsv41-sm120:diagnostics-v1` adds four fatal-path source files on
the exact `responses-prefix-v1` image. It was deployed after **63.6 continuous
idle seconds**, started at **17:44:13 UTC**, and passed built-in arithmetic,
structured JSON, tool round-trip and vision checks by **17:49:36 UTC**.
Container: `8db682f6f4e1`; zero unexpected restarts at qualification.
A plain `docker restart` does not switch images or update environment; this
rollout used Compose recreation against the immutable tested image ID.

The new handler triggers CUDA/NCCL evidence before stack inspection, gives
py-spy 5 seconds per command and 20 seconds overall, and caps the combined
diagnostic phase at 60 seconds. Stack collection and dump waiting share that
window. Python stacks use `--nonblocking` and precede optional native stacks.
Timed-out collectors and descendants are killed without an unbounded wait.
Cleanup runs even when diagnostics raise; a hard scheduler watchdog notifies
the parent immediately instead of first attempting its own stack dump.

The pinned PyTorch 2.13 build additionally needs
`TORCH_NCCL_EXTRA_DUMP_ON_EXEC=1` for its immediate watchdog-exception dump path.
The deployed profile enables it and adds bounded explicit current-run FIFO triggers.
This does not change NCCL algorithms, transports, dependencies or CUDA graphs.

Build receipt: `/mnt/hot/dsv41_state/diagnostics-image-ywucbvdg/receipt.json`.
All 26 CPU-only crash-path tests pass against the built image, as do 93
deployment/helper tests and 1,172 API tests plus 409 subtests. The initial API
run lacked the production signing-key mount expected by the image; the harness
now clears production-only key/recovery defaults (fixtures opt in explicitly),
and the full rerun passed. No live key was mounted or altered. API evidence:
`/mnt/hot/dsv41_state/api-tests-UtwnSLi0/`.
No real CUDA failure was injected; capture on a hardware stall remains unproven.
The fork has a full guide in `deployment/CRASH_DIAGNOSTICS.md`.

The effective settings, four patched runtime hashes and all four NCCL trace
control FIFOs were checked in the running container without triggering dumps.
Capture followed with a fresh heartbeat and zero packet drops. No extra load
test was sent. Evidence and private rollback configuration:
`/mnt/hot/dsv41_state/diagnostics-rollout-c1w07czi/`.

The one-shot `scripts/dsv41_relaunch_diagnostics.py` watcher uses both fresh
`/v1/loads` snapshots and HTTP active/arrival counters: even a short inference
between polls resets the full 60-second timer. Invalid telemetry fails closed;
changed container identity, images or tracked sources cancel the rollout.
There is no automatic second attempt or rollback of a candidate that might
already be serving new work. Its user-systemd job exited successfully.

## Persistent files

Compose mounts `diagnostic_entrypoint.py` read-only. Before executing the
unchanged `/opt/dsv41/boot.py run`, this wrapper creates a private directory:

`/mnt/hot/dsv41_dumps/diagnostics/run-<UTC>-<container-hostname>-<random>/`

A fresh directory is created on **every** start, including Docker automatic
restarts. This matters because NCCL opens its diagnostic log files with
truncation. Older runs are retained; no automatic deletion was added.

| File | Purpose |
|---|---|
| `diagnostics.json` | Allowlisted environment and resolved output paths; no API keys |
| `nccl.<hostname>.<pid>.log` | NCCL INFO logs, all subsystems, one file per process |
| `torch_nccl_trace_rank_<rank>` | PyTorch collective history when dumped on timeout or on demand |
| `torch_nccl_pipe_rank_<rank>.pipe` | PyTorch's manual trace-dump trigger |
| `cuda.<hostname>.<pid>.<epoch>.nvcudmp` | GPU core on a supported CUDA exception or SGLang crash-handler trigger |

Directories and files are private and created by the container's root user.
Inspect them with host administrator privileges or `docker exec dsv41 ...`.
Request/response PCAP and native logs continue independently under the existing
`/mnt/hot/dsv41_dumps/capture-...` directories and `current` symlink.

## Enabled diagnostics

| Setting | Value |
|---|---|
| `CUDA_LOG_FILE` | `stderr`, retained in native container capture |
| `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` | `1` |
| `CUDA_ENABLE_USER_TRIGGERED_COREDUMP` | `1` |
| `CUDA_COREDUMP_SHOW_PROGRESS` | `1` |
| `CUDA_COREDUMP_GENERATION_FLAGS` | `skip_global_memory,skip_abort` |
| `CUDA_COREDUMP_PIPE` | `/tmp/corepipe.cuda.%h.%p` inside the container |
| `NCCL_DEBUG` / `NCCL_DEBUG_SUBSYS` | `INFO` / `ALL` |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` | `8192` bounded entries |
| `TORCH_NCCL_TRACE_CPP_STACK` | `1` |
| `TORCH_NCCL_DUMP_ON_TIMEOUT` | `1` |
| `TORCH_NCCL_EXTRA_DUMP_ON_EXEC` | `1`, verified in the running diagnostic profile |
| `TORCH_NCCL_ENABLE_MONITORING` | `1` (required by the installed PyTorch headers for timeout dumps) |
| `TORCH_NCCL_ENABLE_TIMING` | `0`; no additional collective-start CUDA timing events |
| `SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS` | `60`; total diagnostic window |
| `SGLANG_PYSPY_DUMP_TIMEOUT_SECS` / `SGLANG_PYSPY_DUMP_TOTAL_TIMEOUT_SECS` | `5` / `20` |
| `SGLANG_NCCL_DUMP_BEFORE_CRASH` / `SGLANG_NCCL_DUMP_BEFORE_CRASH_WAIT_SECS` | `1` / `10`; explicit trigger and bounded wait |
| Docker capability | `SYS_PTRACE` added alongside existing `IPC_LOCK` for py-spy |
| CPU core limit | `0`; GPU core files are separate |

Both the installed `TORCH_NCCL_DEBUG_INFO_TEMP_FILE` spelling and the newer
`TORCH_FR_DUMP_TEMP_FILE` spelling point at the same per-run prefix. Filename
substitution `%h/%p/%t` is used only for CUDA/NCCL, not for PyTorch's rank suffix.

CUDA dumps omit bulk global/constant device memory, retaining diagnostic code
and stack/local/shared state. `skip_abort` avoids CUDA's additional automatic
host abort; SGLang/PyTorch retain their own fatal-error handling. CPU core dumps
are disabled to avoid accidentally exporting hundreds of GiB of model/host
memory. GPU dumps remain sensitive and may still be sizable.

The CUDA log can report handled driver probes such as `CUDA_ERROR_INVALID_CONTEXT`
during initialization. These must not be counted as fatal model errors without
the corresponding failure or termination evidence. Enabling diagnostics can
affect performance; this profile has no throughput qualification.

No `CUDA_LAUNCH_BLOCKING`, compute-sanitizer, Nsight session, debug kernel rebuild,
or NCCL per-call TRACE logging is enabled. A crash caused by a hard reset or a
process dying before a dump finishes may still yield incomplete evidence.

Do not write to CUDA core pipes as a routine health check: user-triggered GPU
core generation can interrupt execution. Checking pipe existence is read-only.

References: [NVIDIA CUDA 13.0 GPU core dumps](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-gdb/index.html#gpu-core-dump-support),
[NCCL 2.30.7 logging](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/troubleshooting/logging.html),
[PyTorch flight recorder](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html).

Offline tests: `python3 -m unittest discover -s scripts -p 'test_dsv41*.py'`.
The diagnostics-wrapper tests neither initialize CUDA nor start a server.
