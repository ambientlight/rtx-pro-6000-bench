# CUDA / NCCL crash diagnostics

[compose.api.yaml](../compose.api.yaml) and the
[diagnostic entrypoint](../diagnostic_entrypoint.py) define the capture profile.
Diagnostics change fatal-path evidence collection, not NCCL algorithms,
transports, kernels or cache sizing.

## Persistent artifacts

Every start, including automatic restarts, creates a private directory:

`/mnt/hot/dsv41_dumps/diagnostics/run-<UTC>-<container-hostname>-<random>/`

| File | Purpose |
|---|---|
| `diagnostics.json` | Allowlisted environment and resolved paths; no API keys |
| `nccl.<hostname>.<pid>.log` | Per-process NCCL INFO logs |
| `torch_nccl_trace_rank_<rank>` | Collective history dumped on timeout or explicit trigger |
| `torch_nccl_pipe_rank_<rank>.pipe` | Manual PyTorch trace-dump trigger |
| `cuda.<hostname>.<pid>.<epoch>.nvcudmp` | GPU core on a supported CUDA exception or crash-handler trigger |

Per-start directories prevent restart-time truncation of earlier NCCL evidence.
They are root-owned and private; use host administrator access or
`docker exec dsv41 ...` to inspect them. No automatic deletion is configured.
HTTP PCAP and native logs remain separate under `/mnt/hot/dsv41_dumps/current`.

## Enabled settings and time bounds

| Setting | Value / role |
|---|---|
| `CUDA_LOG_FILE` | `stderr`, retained in native logs |
| `CUDA_ENABLE_COREDUMP_ON_EXCEPTION` / `CUDA_ENABLE_USER_TRIGGERED_COREDUMP` | Both `1` |
| `CUDA_COREDUMP_SHOW_PROGRESS` | `1` |
| `CUDA_COREDUMP_GENERATION_FLAGS` | `skip_global_memory,skip_abort` |
| `CUDA_COREDUMP_PIPE` | `/tmp/corepipe.cuda.%h.%p` inside the container |
| `NCCL_DEBUG` / `NCCL_DEBUG_SUBSYS` | `INFO` / `ALL` |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` / `TORCH_NCCL_TRACE_CPP_STACK` | `8192` / `1` |
| `TORCH_NCCL_DUMP_ON_TIMEOUT` / `TORCH_NCCL_ENABLE_MONITORING` | Both `1` |
| `TORCH_NCCL_EXTRA_DUMP_ON_EXEC` | `1`; required for this PyTorch build's exception-dump path |
| `TORCH_NCCL_ENABLE_TIMING` | `0`; no extra collective-start timing events |
| `SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS` | `60`; combined diagnostic deadline |
| `SGLANG_PYSPY_DUMP_TIMEOUT_SECS` / `SGLANG_PYSPY_DUMP_TOTAL_TIMEOUT_SECS` | `5` per command / `20` overall |
| `SGLANG_NCCL_DUMP_BEFORE_CRASH` / wait seconds | `1` / `10` |
| `SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS` | `60`, within the combined deadline |
| Docker capabilities / CPU core limit | `IPC_LOCK`, `SYS_PTRACE` / `0` |

The handler triggers CUDA/NCCL evidence before stack inspection. Nonblocking
Python stacks precede optional native stacks; timed-out collectors and their
descendants are terminated within bounded cleanup. Hard scheduler watchdogs
notify the parent before attempting diagnostics.

Both PyTorch output-prefix spellings, `TORCH_NCCL_DEBUG_INFO_TEMP_FILE` and
`TORCH_FR_DUMP_TEMP_FILE`, point into the same run directory. CUDA/NCCL use
`%h/%p/%t` substitutions; PyTorch appends its own rank suffix.

## Interpretation and safety

- Handled initialization probes such as `CUDA_ERROR_INVALID_CONTEXT` are not
  proof of a fatal model error. Correlate timestamps with scheduler exceptions,
  termination and restart evidence.
- Check for all rank logs, FIFO existence, trace files and CUDA dumps. Missing
  dumps can mean the process died or the GPU reset before collection completed.
- Do not write to CUDA core pipes as a routine health check; a user-triggered
  GPU dump can interrupt execution.
- CUDA dumps omit bulk device memory but remain sensitive and can be large.
  CPU cores are disabled. Keep all evidence private.
- No `CUDA_LAUNCH_BLOCKING`, compute-sanitizer, Nsight session or NCCL per-call
  TRACE logging is enabled. Diagnostics can still affect performance.

Historical rollout tests, receipts and limitations are in the
[deployment journal](history/STATUS.md#previous-diagnostic-hardening-deployed-after-one-minute-idle).
They did not inject a real GPU fault. The source-side guide is
`sglang-dsv41-production-overlay/deployment/CRASH_DIAGNOSTICS.md`.

References: [CUDA 13.0 GPU core dumps](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-gdb/index.html#gpu-core-dump-support),
[NCCL 2.30.7 logging](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/troubleshooting/logging.html),
[PyTorch flight recorder](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html).
