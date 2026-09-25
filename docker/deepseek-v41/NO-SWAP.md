# Persistent DSV4.1 no-swap policy

The September 25, 2026 deployment policy disables swap for `dsv41` and its
descendants, while leaving the host swapfile available to other applications.

## Applied deployment

Recreated `dsv41` at **2026-09-25 12:35:58 UTC** as container `9137db3cfdc7`.
Readiness recorded at **12:43:09 UTC**: healthy, HTTP 200, zero restarts, and
arithmetic, structured-output, tool-round-trip, and native-vision startup checks
passed. The parent swap limit and both parent/container swap usage were **0**.
The slice is enabled for boot; physical RAM remains uncapped. Capture and the
cache monitor followed the new deployment. Available host RAM was about
**36.3 GiB**, with the 128 GiB host swapfile still enabled for other workloads.

The frozen candidate, private rollback evidence, and completion receipt are in
`/mnt/hot/dsv41_state/noswap-rollout-hezHXiLs/`. `complete.json` records the
verification; image, inference environment, and runtime mounts were unchanged.

## Configuration

- Install `systemd/dsv41.slice` from this repository as
  `/etc/systemd/system/dsv41.slice`.
- The system slice sets `MemoryAccounting=yes` and `MemorySwapMax=0`.
- The production Compose service uses `cgroup_parent: dsv41.slice`.
- Enable the slice for boot and load it before recreating the container:

```sh
sudo install -o root -g root -m 0644 systemd/dsv41.slice /etc/systemd/system/dsv41.slice
sudo systemctl daemon-reload
sudo systemctl enable --now dsv41.slice
```

This requires Docker's systemd cgroup driver and cgroup v2. No `MemoryMax`,
Docker `mem_limit`, or cache-budget reduction is introduced. The image, frozen
SGLang overlays, RAM Engram offload, 256 GB L2 budget, and 67% SWA / 33% FULL
split remain unchanged. Global swappiness and the host swapfile are not changed.

The production recreation is derived from the running deployment's frozen
Compose profile, adding only `cgroup_parent`. This preserves its exact image
and source mounts; the repository Compose is also updated for future launches.

## Checking the effective policy

```sh
systemctl show dsv41.slice -p FragmentPath -p MemorySwapMax -p MemoryMax -p ControlGroup
docker inspect dsv41 --format '{{.HostConfig.CgroupParent}}'
cat /sys/fs/cgroup/dsv41.slice/memory.swap.max
cat /sys/fs/cgroup/dsv41.slice/memory.swap.current
```

Expected: parent `dsv41.slice`, parent `memory.swap.max` equal to `0`, and swap
usage `0` after the restart. A Docker child scope can display `max` in its own
`memory.swap.max`; the parent's zero limit still applies to the entire subtree.
Check both the scope's placement and its ancestors, not just Docker's
`MemorySwap` field. The host swapfile remains enabled.

HiCache tensor payloads already use CUDA host registration; the extra policy
also protects scheduler heaps, cache metadata, tokenizer state, and other
anonymous allocations charged to this cgroup. It is not an `mlockall` call and
does not make executable/file-backed pages unevictable. It also does not give
OOM immunity or reserve RAM against other applications.

Existing swapped allocations are not instantly pulled into RAM by changing a
limit. Recreating the container discards the old processes and their swapped
anonymous allocations; the new processes start under the no-swap parent.

## Rollback

Only after user authorization, remove `cgroup_parent` from the production
profile and recreate the container, or explicitly change the slice's swap
policy and reload/restart the slice when it has no active containers. Do not
stop an occupied slice, which can also stop its member containers. Keep the
host swapfile enabled. Historical frozen profiles predating this policy do
not provide no-swap protection.
