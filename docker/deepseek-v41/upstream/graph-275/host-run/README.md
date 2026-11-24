# Host harness used for the 275 W comparison

These files preserve the launch, cutover and accounting scripts from the benchmark host. `launch-model` and the measurement drivers use that host’s absolute paths. The control and joined arms shared the same staged Python package, private loopback endpoint, worker broker and direct fleet. [Experiment results](../performance.md) · [Fleet source](../fleet/) · [Trace and handbook client](../../full-canary/graph_compare.py)

## Offline verification

From the benchmark repository root, the checks below run against simulated requests, clocked traces and synthetic proof counters. They use the Python standard library and temporary files.

```sh
export GCOMP="$PWD/docker/deepseek-v41/upstream/full-canary"
checks="$PWD/docker/deepseek-v41/upstream/graph-275/host-run/test-graph-client"
for check in verify.py verify-peer.py verify-scan.py; do
  PYTHONPATH="$GCOMP" python3 "$checks/$check" || exit 1
done
```

The three checks cover graph-positive and eager-fallback classification, request isolation, receipt deduplication, and exact compact and wide ticket accounting. The shipped measurements also support recomputing the equal-target comparison from the two matched traces and their worker cursors. [Offline replay instructions](../measurements/README.md)

## Original runtime layout

On the measurement host the runtime directory was `/mnt/hot/ambientlight/sglang-sm120-runtime`. The files in this directory lived under its `graph-production-275/` subdirectory, alongside `fleet/`, the staged `pkg/` and `boot/` overlay, the private per-arm state and the receipt and log directories. `deployment-source/` provided the benchmark checkout. `graph-canary-code/` contained the upstream SGLang and operator checkouts and native libraries. `candidate-unpacked/rootfs/` supplied the interpreter and installed packages. `full-canary/` supplied the relocated interpreter links, CUDA math header links, site configuration and private state used by `env-run`. The protected launcher creates its per-run authentication config in the private state directory. [Staging recipe](../stage-runtime.sh) · [Expected launcher hashes](../staged-launcher.sha256)

| Source | Role in the measured run |
|---|---|
| `env-run`, `launch-model` | Start a selected arm in a private systemd unit with the staged interpreter, CUDA paths, broker socket and authentication. |
| `switch-to-joined`, `rotate-to-equal-peer` | Verify worker drain before changing serving arms or replacing the qualification proof peer with the direct-difficulty peer. |
| `end-run`, `restore-original`, `verify-restored.py` | Wait for the final measured horizon, park the workers, restart the original container and check the four fresh startup receipts, arithmetic answer, device power caps and original container identity. |
| `control-reference.py`, `capture-transition-counters.py`, `thermal-summary.py` | Sample the original serving reference, monitor historical joined counters and join the power and thermal records to request intervals. |
| `test-graph-client/run-arm.sh`, `test-graph-client/compare-all.py` | Execute the two 2K traces, short handoff burst and long-context replay, then produce paired summaries. The finalized one-second-gap control uses the repeated `switch-fixed-gap-timed.json` trace. |

On this host a separate 30-minute user timer pointed at `restore-original` before exclusive testing began. The held worker status, peer, observer and recovery timer were prepared independently of the orchestrators. Live cutover scripts change the running serving and mining services. Offline verification uses the three fixture scripts above.
