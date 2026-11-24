# Host tools used for the full-checkpoint qualification

These are the host-side sources from the original SM120 qualification. The 275 W comparison reused the idle check, OCI Python environment setup, private authentication configuration and GPU namespace inspection. The adjacent graph package holds its own launch and measurement scripts. [275 W benchmark](../../graph-275/README.md)

The original host staged these files under `/mnt/hot/ambientlight/sglang-sm120-runtime/full-canary`, alongside the unpacked candidate image, the deployment checkout and per-run state. The five files in `profile-checks/` were originally at the runtime root. Their absolute paths identify the source checkouts and toolchain used at the time. The protected launcher reads the same pinned checkpoint manifest as the reconstructed image. Its authentication helper writes the key to an owner-only configuration file at runtime. The source archive contains no generated keys or proof payloads. [Pinned checkpoint](../../checkpoint.json) · [Authentication helper](protected_launch.py)

The `math-headers/` directory on the benchmark host linked CUDA math headers from the candidate root filesystem's `usr/local/cuda-13.0/targets/x86_64-linux/include/`. The `bin/python` and `bin/python3` links selected that image's `opt/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12`. The `relocated-site/` hook loads the corresponding Python packages. [Original environment wrapper](env-run)

The launch and restoration scripts control the host services named in their source. For offline verification of the measured results, the comparison client reads the archived traces and miner counters directly. [Offline verification command](../../graph-275/measurements/README.md)
