# Deployment history

Dated records, not launch instructions or live status. “Current”, “staged” and
upstream PR statuses inside these files refer to their original observation
dates. Use the [deployment README](../../README.md) and [operations guide](../OPERATIONS.md)
for the maintained configuration and procedures.

| Record | What it preserves |
|---|---|
| [Deployment journal, Sep 15–25](STATUS.md) | Initial integration, context/memory tuning, launch identities and evidence paths |
| [Expandable-segments failures, Sep 16](EXPANDABLE-STARTUP-FAILURE-2026-09-16.md) | Failed startup checks and allocator-off control |
| [192 GB HiCache, Sep 18](HICACHE-192-2026-09-18.md) | Initial RAM tier and idle-gated rollout |
| [256 GB HiCache, Sep 18](HICACHE-256.md) | Budget expansion and initial replay limitations |
| [SWA retention, Sep 18](SWA-RETENTION-2026-09-18.md) | Root cause, upstream review, patch semantics and qualification |
| [23.6/76.4 split, Sep 18](HICACHE-SPLIT-2026-09-18.md) | First explicit SWA/FULL payload allocation |
| [50/50 split, Sep 19](HICACHE-SPLIT-50-2026-09-19.md) | Intermediate allocation experiment |
| [Attribution and 67/33, Sep 19](CACHE-ATTRIBUTION-2026-09-19.md) | Eviction-cause telemetry, residency census and L13 rollout |
| [60/40 split, Sep 20](HICACHE-SPLIT-60-2026-09-20.md) | L14 comparison |
| [67/33 restored, Sep 20](HICACHE-SPLIT-67-L15-2026-09-20.md) | L15 rollout and guards |
| [Log forwarding, Sep 20](LOG-FORWARDING-2026-09-20.md) | L16 fix for an unread-pipe stall after a UTF-8 decoding error |
| [Preserved release, Sep 25](RELEASE-2026-09-25.md) | Private image/source/settings archive and exact restore procedure, before Responses progress |

Performance reports and the final L16 traffic comparison remain under
[bench/deepseek-v4.1-flash_TP4_sglang](../../../../bench/deepseek-v4.1-flash_TP4_sglang/README.md).
Private receipt/capture paths are retained as evidence references; some captures
have since moved to cold storage. Do not treat historical commands or container
IDs as instructions for the running deployment.

The old migration/rollout helpers, incremental Dockerfiles, one-shot systemd
templates, benchmark probes and local unit tests were retired during cleanup.
Their last retained version is Git revision `dc74842`; for example,
`git show dc74842:scripts/dsv41_cache_pressure.py` reads the original runner.
Historical references below this directory do not imply those tools remain
installed or supported in the current checkout.
