# Engram and HiCache memory

These are separate consumers of system RAM. Configuration is in
[compose.api.yaml](../compose.api.yaml); [historical experiments](history/README.md)
record how the present split was selected.

## Budgets

| Pool | Configured behavior |
|---|---|
| Engram weights | `OFFLOAD_MODE=ram` preloads and locks the two backing shards, about 189.1 GiB shared physical RAM |
| Engram NVMe row cache | `DSV41_CACHE_GIB=0`; no separate row cache in RAM mode |
| L1 KV on GPU | Automatic sizing at startup under static fraction 0.87; FULL and SWA sized independently |
| L2 HiCache in RAM | `DSV41_HICACHE_HOST_BUDGET_BYTES=256000000000`, across all four ranks |
| L2 tensor payload split | `DSV41_HICACHE_SWA_FRACTION=0.67`: 67% SWA / 33% FULL |
| L3 KV storage | Disabled; no NVMe/storage backend |

The HiCache budget is **256 decimal GB**, not 256 GiB per GPU. It includes
slot-index/metadata allowances; the payload split applies after those allowances.
It is not a cap on process RSS or total system RAM. Engram, scheduler heaps,
temporary allocations and the OS need room in addition to HiCache.

The recorded 67/33 allocation is **12,570,368 FULL tokens and 1,656,064 SWA
slots per rank**, with about 250.78 GB initially allocated buffers across TP4
and a 255.98 GB plan including allowances. Verify a new launch from its
`DSV41_HICACHE_BUDGET` / `DSV41_HICACHE_ALLOCATED` logs and live metrics.
Do not multiply rank-0 logical prefix capacity by four.

Engram offload does not make those weights GPU-resident: selected rows still
transfer from host RAM to the GPUs. Earlier page-residency probes found a small
nonresident fraction, so RAM mode alone is not proof of strictly zero disk I/O.

## How prefixes survive GPU eviction

The deployed HiCache mode is RAM-only `cache`, `write_through`,
`page_first`, with kernel I/O and Python TreeCore.

1. Computed prefixes are backed up from GPU to RAM, including eligible SWA
   windows. Transfer locks protect buffers while asynchronous copies complete.
2. GPU eviction can leave reusable FULL KV and an SWA endpoint in RAM.
3. A later matching request restores missing device components from RAM. If FULL
   remains on GPU, a hit can require only SWA load-back.
4. RAM capacity pressure evicts host entries. A restart loses both warm tiers;
   this is not a persistent disk cache.

A usable prefix needs the FULL chain **and** the required SWA resumption state,
in GPU or RAM. Missing SWA at the longest FULL match can force fallback to an
earlier valid boundary, or a complete miss if none survives. It is not always
an all-or-nothing miss.

## SWA boundaries and eviction

`SGLANG_OPT_HICACHE_SWA_RETAIN_REQUEST_BOUNDARIES=1` prefers completed-request
windows over ordinary intermediate prefill checkpoints. Under host-SWA pressure,
ordinary checkpoints are reclaimed first; retained boundaries remain evictable
when needed. FULL-driven eviction can remove the associated SWA too.

The attention window is 128 tokens, but a retained endpoint normally spans two
256-slot pages (512 slots), including the look-behind needed for exact replay.
A slot is an allocation position, not a whole prompt or an independent request.
SWA slots cost much more than FULL logical-token slots and include draft-state
accounting. Thus slot counts and percentage occupancies are not byte ratios.

Retention does **not** reserve five checkpoints per prompt or pin a fixed number
of conversations. Shared prefixes, branching, coarse nodes and workload shape
determine actual residency. The 67/33 split is an empirical choice for this
workload, not a general optimum or a guarantee of synchronized eviction.

## What to monitor

Use [CACHE-MONITOR.md](CACHE-MONITOR.md) for rank-0 occupancy, eviction causes,
usable FULL coverage and request-level L2 reuse. Boundary evictions are a subset
of total SWA evictions; high ordinary-checkpoint churn alone is not a failure.
Transfer volume is not the same as input tokens saved.

The [L16 traffic report](../../../bench/deepseek-v4.1-flash_TP4_sglang/L16-REAL-TRAFFIC-2026-09-25.md)
records 97.82% combined cache coverage, including 3.62% of input served from RAM
L2. Those are historical workload measurements, not current counters or a
controlled estimate of benefit versus a no-L2 deployment.

[No-swap policy](NO-SWAP.md) protects the container subtree, including anonymous
metadata allocations; it does not reserve RAM or prevent host-OOM termination.
