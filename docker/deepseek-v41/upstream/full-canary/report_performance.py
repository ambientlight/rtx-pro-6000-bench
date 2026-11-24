"""Regenerate the full-checkpoint performance report from saved HTTP receipts."""

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {"stock": "Upstream stock", "pearl": "Upstream Pearl", "incumbent": "Restored incumbent"}
COLORS = {"stock": "#2463a6", "pearl": "#b85021", "incumbent": "#24765e"}
LENGTHS = (2048, 8192, 32768)
WIDTHS = (1, 2, 4, 8)
DASH = "—"


def read(path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


def number(value, places=0):
    if value is None:
        return DASH
    return f"{value:,.{places}f}"


def median(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def span(values, places=0):
    values = [float(x) for x in values if x is not None]
    if not values:
        return DASH
    lo, hi = number(min(values), places), number(max(values), places)
    return lo if lo == hi else f"{lo}–{hi}"


def table(headers, rows):
    def line(entries):
        return "| " + " | ".join(map(str, entries)) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def matched_native(native):
    # Compare actual token IDs, including suffix and session nonce, by repeated trial.
    seen = {}
    for key, data in native.items():
        for row in data.get("runs", []):
            fixture = (
                row["input_per_call"],
                row["output_per_call"],
                row["concurrency"],
                tuple(r["input_sha256"] for r in row["requests"]),
            )
            if row["name"] in seen and seen[row["name"]] != fixture:
                raise ValueError(f"Differing token IDs or settings for {key}/{row['name']}")
            seen[row["name"]] = fixture


def summarize_native(native):
    matched_native(native)
    singles, batches = {}, {}
    for arm, receipt in native.items():
        for run in receipt.get("runs", []):
            length, width = run["input_per_call"], run["concurrency"]
            requests = run["requests"]
            if run["name"].startswith("single-"):
                assert width == len(requests) == 1 and run["output_per_call"] == 256
                singles.setdefault((arm, length), []).extend(requests)
            elif run["name"].startswith("parallel-"):
                assert length == 2048 and width == len(requests) and run["output_per_call"] == 128
                batches.setdefault((arm, width), []).append(run)
    return singles, batches


def plot(path, singles, batches):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    specs = [
        (
            "Prefill to first token",
            "Uncached input tokens/s",
            ("2,048", "8,192", "32,768"),
            LENGTHS,
            singles,
            "client_effective_prefill_tps",
            "Input tokens per request",
        ),
        (
            "Concurrent generation",
            "Output tokens/s, including prefill",
            ("1", "2", "4", "8"),
            WIDTHS,
            batches,
            "aggregate_output_tps_including_prefill",
            "Simultaneous requests",
        ),
    ]
    handles = {}
    for ax, (title, ylabel, ticks, xs, data, metric, xlabel) in zip(axes, specs):
        for arm, label in NAMES.items():
            ys = [median(data.get((arm, x), []), metric) for x in xs]
            if all(y is None for y in ys):
                continue
            (line,) = ax.plot(
                range(len(xs)),
                [float("nan") if y is None else y for y in ys],
                marker={"stock": "o", "pearl": "s", "incumbent": "^"}[arm],
                color=COLORS[arm],
                linewidth=2,
                markersize=5,
                label=label,
            )
            handles[arm] = line
        ax.set(title=title, ylabel=ylabel, xlabel=xlabel, xticks=range(len(xs)), xticklabels=ticks)
        ax.set_ylim(bottom=0)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=0.24)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    if handles:
        fig.legend(
            [handles[k] for k in NAMES if k in handles],
            [NAMES[k] for k in NAMES if k in handles],
            loc="outside lower center",
            frameon=False,
            ncol=len(handles),
        )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def historical_window(directory):
    # This row used the exact archived prompt from full_window.py.
    archive = read(directory / "historical-baselines.json")["full_prefill"]
    rows = [dict(zip(archive["columns"], row, strict=True)) for row in archive["rows"]]
    q = next(
        row for row in rows if row["source"].endswith("BENCHMARK-FULL-PREFILL-RESTORED-085-2026-09-16.md")
    )
    return dict(
        payload_sha256="ff7319cb893282458cf77fdcef75ca5aebe0b8a5145708fe616bdab0cdd467df",
        input_tokens=q["input_tokens"],
        cached_tokens=0,
        output_tokens=2,
        answer_check=True,
        first_text_seconds=q["client_ttft_s"],
        wall_seconds=q["client_total_s"],
        server_prefill_seconds=q["native_prefill_s"],
        server_uncached_prefill_tokens_per_second=q["native_prefill_tps"],
        allocator_retry_lines=q["allocator_retry_lines"],
    )


def active_hardware(directory):
    phases = read(directory / "hardware-throughput.json").get("phases", {})
    rows = []
    for arm, name in NAMES.items():
        phase = phases.get(f"{arm}-native", {})
        stats = list(phase.get("within_recorded_requests", {}).values())
        if not stats:
            continue

        def clock(field):
            return span(
                [
                    item
                    for board in stats
                    if board.get(field)
                    for item in [board[field].get("min"), board[field].get("max")]
                ]
            )

        rows.append(
            [
                name,
                sum(b.get("active_samples", 0) for b in stats),
                span([w for b in stats for w in b.get("power_limit_w", [])]),
                clock("memory_clock_mhz_when_active"),
                clock("sm_clock_mhz_when_active"),
                span([b["peak_vram_mib"] / 1024 for b in stats if b.get("peak_vram_mib") is not None], 2),
                number(phase.get("missing_leading_s"), 1),
            ]
        )
    return rows


def pearl_work(directory):
    handoff = read(directory / "candidate-handoff.json")
    observations = handoff.get("requests", [])
    if not observations:
        return []
    n = len(observations)
    passed = sum(
        all(
            q[key]
            for key in (
                "active_mining_before_http",
                "saw_drain",
                "saw_all_resumed",
                "stationary_four_cursors",
                "healthy_all",
            )
        )
        for q in observations
    )
    blocks = [
        "## Mining during generation and between requests",
        f"Across {passed} of {n} traced requests the four direct miners were active before HTTP send, drained during inference, "
        "held their work cursors stationary, and resumed. The table gives the observed range and median. "
        "These eight trials establish no bound on future tail latency. [Handoff observations](candidate-handoff.json)",
    ]
    timings = []
    for label, key in [
        ("HTTP send to observation of all four miners drained", "http_send_to_observed_all_drained_ms"),
        (
            "First miner release to observed GPU re-entry on all four boards",
            "first_release_to_all_observed_gpu_entry_ms",
        ),
        ("HTTP return to last observed GPU re-entry", "http_return_to_last_observed_gpu_entry_ms"),
        ("Individual worker release to its GPU re-entry", "worker_release_to_entry_ms"),
    ]:
        values = [q[key] for q in observations]
        if key == "worker_release_to_entry_ms":
            values = [v for group in values for v in group]
        timings.append(
            [
                label,
                len(values),
                number(min(values), 3),
                number(statistics.median(values), 3),
                number(max(values), 3),
            ]
        )
    blocks.append(table(["Interval", "Observations", "Min (ms)", "Median (ms)", "Max (ms)"], timings))
    polling_gap = handoff["polling_summary"]["maximum_poll_gap_ns"] / 1e6
    blocks.append(
        f"The broker observer polled nominally every millisecond with a maximum gap of {polling_gap:.2f} ms. "
        "A negative HTTP-return interval means the GPU had re-entered mining before the client observed the response. "
        "Worker release-to-entry comes from the worker observing its GPU-mapped start flag."
    )
    traced = read(directory / "candidate-proofs.json")
    if traced.get("distinct_by_rows"):
        by_rows = traced["distinct_by_rows"]
        blocks.append(
            f"The easy-target local peer independently verified **{traced['distinct_verified_sha256']} distinct proofs** during those "
            f"{len(traced['requests'])} full-model calls. Of these, {by_rows[str(traced['prefill_rows'])]} came from prefill and "
            f"{by_rows[str(traced['spec_verify_rows'])]} from speculative target verification during generation, across all "
            f"{traced['tp']} tensor-parallel ranks. [Per-rank proof evidence](candidate-proofs.json)"
        )
    evidence = read(directory / "mining-evidence.json")
    thermal = evidence.get("worker_thermal_guard", {})
    exits = thermal.get("exit_logs", {})
    limit = thermal.get("recovered_fleet", {}).get("verified_worker_thermal_limit_c")
    if exits and limit is not None:
        stopped = " and ".join(k.upper() for k in sorted(exits))
        blocks.append(
            f"After the 524,177-token request completed, {stopped} reached the {limit} °C worker limit on resuming direct mining. "
            "Fleet cleanup removed the broker socket. The next inference admission failed, and cooldown and an operator restart "
            "were required. Existing model ranks held stale broker connections after a fresh fleet started. "
            "[Worker thermal and recovery evidence](mining-evidence.json)"
        )
    proofs = evidence.get("proofs", {})
    if "easy" in proofs and "sustained" in proofs:
        easy, hard = proofs["easy"], proofs["sustained"]
        htotal, hdist = hard["retained_submissions"], hard["distinct_sha256"]
        blocks.append(
            f"Including startup, the easy peer retained {easy['distinct_sha256']} distinct hashes in "
            f"{easy['retained_submissions']} submissions. On the harder job the first {htotal} retained "
            f"verifier-positive submissions contained {hdist} distinct hashes and {htotal-hdist} exact resubmissions. "
            "These local measurements establish neither paid external pool credit nor inference TH/s. "
            "[Peer evidence](mining-evidence.json)"
        )
    projected = read(directory / "projection-work.json")
    if projected.get("rows"):
        blocks.append(
            "Only the target model’s first-layer output projection participates. The table divides its observed real-row "
            "multiply-accumulates across all four ranks by elapsed client time for each phase. The percentage compares that "
            "logical MAC rate with the measured four-worker idle rate. Prefill ends at the first text delta, and target "
            "verification spans first text to terminal status. The counters include calls eligible by shape even when a "
            "pool job or proof queue may be unavailable. [Projection counts and dimensions](projection-work.json)"
        )
        projection_rows = []
        for call in projected["rows"]:
            cells = []
            for tag in ("EXTEND", "TARGET_VERIFY"):
                phase = call["phases"].get(tag) or {}
                cells.extend(
                    [
                        number(phase.get("logical_tmac_s"), 5),
                        number(phase.get("percent_of_measured_idle_mac_rate"), 6),
                    ]
                )
            projection_rows.append([number(call["input_tokens"]), number(call["output_tokens"]), *cells])
        blocks.append(
            table(
                [
                    "Input tokens",
                    "Output tokens",
                    "Prefill TMAC/s",
                    "Prefill % of idle",
                    "Verify TMAC/s",
                    "Verify % of idle",
                ],
                projection_rows,
            )
        )
    idle = evidence.get("idle", {})
    if idle.get("sliced_total_tmac_s") and idle.get("archived_f2pool_total_local_tmac_s"):
        sliced, archived = idle["sliced_total_tmac_s"], idle["archived_f2pool_total_local_tmac_s"]
        nearest = idle.get("immediately_pretest_f2pool", {}).get("local_average_tmac_s_total")
        adjacent_comparison = (
            f"The last F2Pool supervisor snapshot before shutdown reported {nearest/1000:.3f} PMAC/s in summed worker "
            f"lifetime averages, putting the offline workers at **{100*sliced/nearest:.1f}%** of that local rate. "
            if nearest
            else ""
        )
        blocks.append(
            f"Over {idle['elapsed_seconds']:.2f} seconds of sampled intervals between inference requests, the four direct miners sustained "
            f"**{sliced/1000:.3f} PMAC/s**. {adjacent_comparison}An older F2Pool snapshot measured "
            f"{archived/1000:.3f} PMAC/s, giving a ratio of {100*sliced/archived:.1f}%. "
            "The snapshots and experimental workers used different clock settings and time windows. [Idle mining measurements](mining-evidence.json)"
        )
    return blocks


def eager_control(directory):
    job = read(directory / "eager-job.json")
    plain = read(directory / "eager-plain.json")
    if not job.get("complete") or not plain.get("complete"):
        return []
    by_name = {q["name"]: q for q in plain["runs"]}
    rows = []
    matched_output = 0
    for call in job["runs"]:
        ref = by_name[call["name"]]
        if any(call[key] != ref[key] for key in ("input_sha256", "input_tokens", "output_tokens")):
            raise ValueError(f"Different job-on and peer-off fixture for {call['name']}")
        if call["cached_tokens"] != 0 or ref["cached_tokens"] != 0:
            raise ValueError(f"Unexpected prefix hit for {call['name']}")
        match = call["output_ids_sha256"] == ref["output_ids_sha256"]
        matched_output += match
        values = []
        for q in (call, ref):
            values.extend(
                [
                    number(q["ttft_s"], 3),
                    number(q["client_decode_tps"], 1),
                    number(q["spec_accept_length"], 2),
                ]
            )
        rows.append([call["name"].split("-")[-1], *values, "Yes" if match else "No"])
    delta = [
        100 * (q["client_decode_tps"] / by_name[q["name"]]["client_decode_tps"] - 1) for q in job["runs"]
    ]
    blocks = [
        "## Incremental cost of the mining job",
        "Three cold-prefix pairs used the same eager server, calibrated target-layer weights, four direct workers, "
        "2,048-token inputs and 128 forced output tokens. The first series used an active local proof job. "
        "The second used the integer fallback after the peer stopped. The run order was fixed. "
        "[Active-job receipts](eager-job.json) · [Peer-off receipts](eager-plain.json)",
        table(
            [
                "Pair",
                "Job first token (s)",
                "Job client decode (tok/s)",
                "Job accepted length",
                "Peer off first token (s)",
                "Peer off client decode (tok/s)",
                "Peer off accepted length",
                "Same output IDs",
            ],
            rows,
        ),
        f"Across matched pairs, the median change in client decode rate with the job active was {statistics.median(delta):+.1f}% "
        f"(range {min(delta):+.1f}% to {max(delta):+.1f}%). Output token IDs matched in {matched_output} of {len(rows)} pairs.",
    ]
    if job.get("unmeasured_warmup"):
        blocks.insert(
            2,
            "The freshly launched eager process first served a disjoint 2,048-input, 128-output warm-up with the job active.",
        )
    if matched_output < len(rows):
        blocks.append(
            "Different generated sequences can change speculative acceptance and decode throughput. "
            "The paired changes include that effect alongside the cost of projection and proof handling."
        )
    telemetry = read(directory / "hardware-throughput.json").get("phases", {})
    if "eager-job" in telemetry and "eager-plain" in telemetry:
        boards = [
            board
            for name in ("eager-job", "eager-plain")
            for board in telemetry[name]["within_recorded_requests"].values()
        ]
        caps = [value for board in boards for value in board.get("power_limit_w", [])]
        memory = [
            clock
            for board in boards
            for key, clock in (board.get("memory_clock_mhz_when_active") or {}).items()
            if key in ("min", "max")
        ]
        if memory and caps:
            blocks.append(
                f"Active GPU samples during these two series showed {span(memory)} MHz memory clocks under a "
                f"{span(caps)} W per-card cap. [Hardware measurements](hardware-throughput.json)"
            )
    return blocks


def model_checks(directory, revisions):
    rows = []
    for title, filename, reference in [
        ("Upstream Pearl vs upstream stock", "candidate-pearl.json", "compare_to_reconstructed_stock"),
        ("Upstream stock vs original deployment", "candidate-stock.json", "compare_to_incumbent"),
        ("Restored vs pre-outage original deployment", "incumbent-unlocked.json", "compare_to_incumbent"),
    ]:
        receipt = read(directory / filename)
        compare = receipt.get(reference, {})
        trials = compare.get("native") or []
        if trials:
            natural = next(q for q in trials if q["name"] == "natural")
            rows.append(
                [
                    f"[{title}]({filename})",
                    f"{sum(q['exact_tokens'] for q in trials)}/{len(trials)}",
                    "Matched" if natural["exact_text"] else "Differed",
                    "Matched" if compare["arithmetic_equal"] else "Differed",
                ]
            )
    if not rows:
        return []
    blocks = [
        "## Model output checks",
        "Seven short greedy fixtures forced 16 tokens each. The table counts exact output-token matches for one natural prompt "
        "and six synthetic token sequences. The separate arithmetic check expects 42. These fixtures cover deterministic "
        "behavior on the sampled inputs. Broader model quality remains unmeasured.",
        table(["Comparison", "Exact token sequences", "Natural text", "Arithmetic"], rows),
    ]
    if len(rows) == 3:
        blocks.append(
            "The original software also produced different greedy IDs across its two launches. The short trials and "
            "the longer job-on/off comparison show output variation. Attributing a change in model quality would require broader evaluation."
        )
    smoke_groups = [
        revisions.get("profiles", {}).get(k, {}).get("bootstrap_smokes", {}) for k in ("stock", "pearl")
    ]
    if all(
        len(smokes) == 4 and all(v.get("status") == "passed" for v in smokes.values())
        for smokes in smoke_groups
    ):
        blocks.append(
            "Both rebuilt profiles passed the full-model startup checks for text, structured JSON, tool continuation and vision. "
            "[Startup check results](full-run.json)"
        )
    return blocks


def build(directory, output, with_plot):
    native = {k: read(directory / f"{k}-native.json") for k in NAMES}
    handbook = {k: read(directory / f"{k}-handbook.json") for k in NAMES}
    window = {k: read(directory / f"window-{k}.json") for k in NAMES}
    archive = read(directory / "historical-handbook.json")
    old_window = historical_window(directory)
    prefix = {k: read(directory / f"{k}-prefix.json") for k in NAMES}
    singles, batches = summarize_native(native)
    fast = {
        k: number(median(batches.get((k, 8), []), "aggregate_output_tps_including_prefill"), 1) for k in NAMES
    }
    blocks = ["# DeepSeek V4.1 Flash throughput on four RTX PRO 6000s"]
    if all(cell != DASH for cell in fast.values()):
        blocks.append(
            f"At eight simultaneous requests the servers delivered **{fast['stock']} output tokens/s on upstream stock**, "
            f"**{fast['pearl']} with Pearl** and **{fast['incumbent']} on the restored incumbent**, including prompt processing. "
            "Each request consumed 2,048 input tokens and generated 128 output tokens."
        )
    paired = []
    for key, name in (("stock", "upstream stock"), ("incumbent", "the restored incumbent")):
        pearl_rate = median(singles.get(("pearl", 2048), []), "client_decode_tps")
        other_rate = median(singles.get((key, 2048), []), "client_decode_tps")
        if pearl_rate and other_rate:
            paired.append(f"{100*pearl_rate/other_rate:.0f}% of {name}")
    if paired:
        blocks.append(
            "For a single stream with 2,048 input and 256 output tokens, Pearl’s median client decode rate was "
            + " and ".join(paired)
            + "."
        )
    prefill_ratio = {}
    for length in LENGTHS:
        stock = median(singles.get(("stock", length), []), "native_prefill_tps")
        pearl = median(singles.get(("pearl", length), []), "native_prefill_tps")
        if stock and pearl:
            prefill_ratio[length] = f"{100 * pearl / stock:.0f}%"
    if len(prefill_ratio) == len(LENGTHS):
        blocks.append(
            f"Pearl retained {prefill_ratio[2048]}, {prefill_ratio[8192]} and {prefill_ratio[32768]} of stock’s server prefill "
            "throughput at 2,048, 8,192 and 32,768 input tokens, respectively."
        )
    blocks.append(
        "The November runs use the 1017 checkpoint, tensor and expert parallelism of four, and DSPARK block size five. "
        "Upstream stock uses the checkpoint’s native projection and decode graphs. The Pearl profile uses a calibrated INT7 "
        "projection for target layer zero’s `attn.wo_b`, eager execution, a local verifying peer and watched idle miners. "
        "The comparison includes the cost of changing execution modes and proof handling."
    )
    revisions = read(directory / "full-run.json")
    if revisions.get("source_head") and revisions.get("upstream_commit"):
        blocks.append(
            f"Both rebuilt profiles use `{revisions['candidate_branch']}` at `{revisions['source_head'][:12]}`, "
            f"built on `{revisions['upstream_branch']}` at `{revisions['upstream_commit'][:12]}` "
            f"({revisions['upstream_commit_utc']}). "
            f"The deployment branch for this run was `{revisions['deployment_branch']}`. [Build and profile details](full-run.json)"
        )
    if with_plot and (singles or batches):
        plot(output / "throughput.png", singles, batches)
        blocks.append(
            "![Prefill throughput versus input length and aggregate output throughput versus request concurrency](throughput.png)"
        )
    blocks.extend(
        [
            "## Single-request prefill and decode",
            "Each native request forces 256 output tokens at temperature zero. Rows report medians across the counted trials. "
            "Client prefill divides fresh input tokens by time from HTTP send to the first token-bearing event. "
            "The client timer starts after input tokenization. Server prefill uses the scheduler’s forward-entry and prefill-finished timestamps where present. "
            "Server decode uses reported decode throughput. Client decode counts tokens after the first streaming event through the final event, "
            "allowing DSPARK to bundle several tokens. Accepted length is tokens per speculative verification.",
        ]
    )
    rows = []
    for length in LENGTHS:
        for k, name in NAMES.items():
            calls = singles.get((k, length), [])
            if calls:
                rows.append(
                    [
                        name,
                        number(length),
                        len(calls),
                        span([r.get("cached_tokens") for r in calls]),
                        number(median(calls, "ttft_s"), 3),
                        number(median(calls, "client_effective_prefill_tps")),
                        number(median(calls, "native_prefill_tps")),
                        number(median(calls, "native_decode_tps"), 1),
                        number(median(calls, "client_decode_tps"), 1),
                        number(median(calls, "spec_accept_length"), 2),
                    ]
                )
    blocks.append(
        table(
            [
                "Profile",
                "Input tokens",
                "Trials",
                "Cached tokens",
                "First token (s)",
                "Client prefill (tok/s)",
                "Server prefill (tok/s)",
                "Server decode (tok/s)",
                "Client decode (tok/s)",
                "Accepted length",
            ],
            rows,
        )
    )
    slow = []
    for k, name in NAMES.items():
        for length in LENGTHS:
            calls = singles.get((k, length), [])
            if len(calls) < 2:
                continue
            first = calls[0]["ttft_s"]
            rest = median(calls[1:], "ttft_s")
            if first > 2 * rest and first - rest > 1:
                slow.append(
                    f"{name} at {number(length)} input tokens took {first:.2f} s on the first trial "
                    f"and {rest:.2f} s on the median of the remaining {len(calls)-1}"
                )
    if slow:
        blocks.append(
            "First-use latency varied across trials. "
            + ". ".join(slow)
            + ". The table includes these first trials."
        )
    blocks.extend(
        [
            "## Concurrent generation",
            "Each synchronized batch sends 2,048 input tokens per request and forces 128 output tokens per request. "
            "Aggregate throughput divides all generated tokens by elapsed time from the first send to the last response. "
            "First-token latency is the median across requests in that batch. Each profile uses the same token IDs for a given trial.",
        ]
    )
    rows = []
    for width in WIDTHS:
        row = [width]
        for arm in NAMES:
            runs = batches.get((arm, width), [])
            calls = [r for batch in runs for r in batch["requests"]]
            row.extend(
                [
                    number(median(runs, "aggregate_output_tps_including_prefill"), 1),
                    number(median(calls, "ttft_s"), 3),
                ]
            )
        rows.append(row)
    blocks.append(
        table(
            [
                "Requests",
                *[f"{name} {metric}" for name in NAMES.values() for metric in ("tok/s", "first token (s)")],
            ],
            rows,
        )
    )
    prefix_rows = []
    for k, data in prefix.items():
        for q in data.get("runs", []):
            prefix_rows.append(
                [
                    NAMES[k],
                    q["name"],
                    number(q["input_tokens"]),
                    number(q["cached_tokens"]),
                    number(q["ttft_s"], 3),
                    number(q["native_prefill_s"], 3),
                ]
            )
    if prefix_rows:
        blocks.extend(
            [
                "## Prefix reuse",
                "Two consecutive requests share 8,192 leading tokens and have distinct 16-token suffixes. Each generates one token.",
                table(
                    [
                        "Profile",
                        "Request",
                        "Input tokens",
                        "Cached tokens",
                        "First token (s)",
                        "Server prefill (s)",
                    ],
                    prefix_rows,
                ),
            ]
        )
    blocks.extend(
        [
            "## Long handbook generation",
            "The streamed Responses workload uses temperature 1.0 and top-p 0.95. The output cap is listed for each run. "
            "TTFT ends at the first nonempty text delta. Client decode uses (total output tokens − 1) divided by time from first text to the terminal event. "
            "November server decode uses an attributable scheduler time-per-output-token observation, or the equivalent "
            "server first-token and end-to-end latency observations on the original deployment. "
            "Long natural generations have different output lengths. September used an earlier checkpoint at 250 W, NVMe Engram, a 409,600-token window "
            "and a static memory fraction of 0.85.",
        ]
    )
    if all(report.get("complete") for report in handbook.values()):
        signatures = [
            [
                (q["prompt_sha256"], q["input_tokens"], q["output_budget"], q["temperature"], q["top_p"])
                for q in report["runs"]
            ]
            for report in handbook.values()
        ]
        if not all(signature == signatures[0] for signature in signatures[1:]):
            raise ValueError("November long handbook workloads differ across profiles")
        blocks.append(
            "All three November profiles used identical prompt hashes, token counts, sampling settings and output caps. "
            "Their observed output lengths are recorded below."
        )
    rows = []
    for arm, data in [("September 15", archive), *((NAMES[k], handbook[k]) for k in NAMES)]:
        for q in data.get("runs", data.get("rows", [])):
            budget = 8192 if arm == "September 15" else q["output_budget"]
            rows.append(
                [
                    arm,
                    number(q["input_tokens"]),
                    number(q.get("cached_tokens")),
                    number(q["output_tokens"]),
                    number(budget),
                    number(q.get("ttft_s", q.get("ttft_client_s")), 3),
                    number(q.get("e2e_s", q.get("e2e_client_s")), 2),
                    number(q.get("client_effective_prefill_tps", q.get("effective_prefill_tps"))),
                    number(q.get("client_decode_tps"), 1),
                    number(
                        q.get(
                            "native_server_decode_tps",
                            q.get("native_tpot_histogram_decode_tps", q.get("decode_server_tps")),
                        ),
                        1,
                    ),
                    number(q.get("spec_accept_length"), 2),
                ]
            )
    blocks.append(
        table(
            [
                "Profile",
                "Input tokens",
                "Cached",
                "Output tokens",
                "Output cap",
                "First text (s)",
                "Wall (s)",
                "Client prefill (tok/s)",
                "Client decode (tok/s)",
                "Server decode (tok/s)",
                "Accepted length",
            ],
            rows,
        )
    )
    blocks.extend(
        [
            "## Near-full-window replay",
            "The November requests replay the September 16 payload byte for byte. This short answer tests prefill and first-text latency. "
            f"Server prefill uses the reported scheduler interval. Client prefill divides uncached input by time to first text. "
            f"The September baseline logged {old_window['allocator_retry_lines']} recoverable allocator retries with RAM Engram and automatic KV sizing.",
        ]
    )
    rows = []
    window_failures = []
    for arm, q in [("September 16", old_window), *((NAMES[k], window[k]) for k in NAMES)]:
        if not q:
            continue
        if q["payload_sha256"] != old_window["payload_sha256"]:
            raise ValueError(f"Different near-window payload for {arm}")
        if q.get("status") == "failed":
            rows.append([arm, f"{number(q['fixture_input_tokens'])} fixture", *([DASH] * 7), "Failed"])
            window_failures.append(arm)
            continue
        fresh_tps = q.get("effective_client_uncached_tokens_per_second")
        if fresh_tps is None and q.get("cached_tokens") is not None:
            fresh_tps = (q["input_tokens"] - q["cached_tokens"]) / q["first_text_seconds"]
        rows.append(
            [
                arm,
                number(q["input_tokens"]),
                number(q.get("cached_tokens")),
                number(q.get("output_tokens", q.get("usage", {}).get("output_tokens"))),
                number(q.get("first_text_seconds"), 3),
                number(q.get("wall_seconds"), 3),
                number(q.get("server_prefill_seconds"), 3),
                number(q.get("server_uncached_prefill_tokens_per_second")),
                number(fresh_tps),
                "Pass" if q["answer_check"] else "Fail",
            ]
        )
    blocks.append(
        table(
            [
                "Profile",
                "Input tokens",
                "Cached",
                "Output tokens",
                "First text (s)",
                "Wall (s)",
                "Server prefill (s)",
                "Server prefill (tok/s)",
                "Client prefill (tok/s)",
                "Answer 42",
            ],
            rows,
        )
    )
    if NAMES["incumbent"] in window_failures:
        blocks.append(
            "The restored original deployment received the same 524,177-token fixture. All four tensor-parallel ranks "
            "reported an out-of-memory error in the indexer, the stream closed before a terminal response, and Docker restarted "
            "the service. The failed row reports the fixture length only. This prevented a completed throughput or first-token measurement. "
            "[Failed replay evidence](window-incumbent.json)"
        )
        phase = read(directory / "hardware-throughput.json").get("phases", {}).get("window-incumbent", {})
        peaks = [
            board["peak_vram_mib"]
            for board in phase.get("all_phase", {}).values()
            if board.get("peak_vram_mib")
        ]
        if peaks:
            blocks.append(
                f"Physical GPU telemetry reached {number(max(peaks))} MiB ({max(peaks)/1024:.2f} GiB) used on one board "
                "during the approximate outer client-process interval of the failed replay. [Hardware measurements](hardware-throughput.json)"
            )
    blocks.extend(eager_control(directory))
    blocks.extend(model_checks(directory, revisions))
    blocks.extend(pearl_work(directory))
    blocks.extend(
        [
            "## Hardware and coverage",
            "VRAM and clock ranges below are snapshots across all four devices at the start and end of each native series. "
            "Power is the configured per-card limit. The snapshots include other GPU processes. The November performance runs "
            "use 280 W per card. The 250 W September results used a different checkpoint and memory configuration.",
        ]
    )
    hardware = []
    for arm, receipt in native.items():
        gpus = [gpu for key in ("gpu_before", "gpu_after") for gpu in (receipt.get(key) or [])]
        if not gpus:
            continue

        def vals(field):
            return [g.get(field) for g in gpus if g.get(field) not in (None, "[N/A]", "N/A")]

        hardware.append(
            [
                NAMES[arm],
                span(vals("power.limit")),
                span(vals("clocks.current.memory")),
                span(vals("clocks.current.sm")),
                span([float(x) / 1024 for x in vals("memory.used")], 2),
                number(receipt.get("profile", {}).get("max_total_num_tokens")),
            ]
        )
    blocks.append(
        table(
            [
                "Profile",
                "Power limit (W)",
                "Memory clock (MHz)",
                "SM clock (MHz)",
                "VRAM used per GPU (GiB)",
                "KV pool tokens",
            ],
            hardware,
        )
    )
    previous_pool = read(directory / "incumbent.json").get("profile", {}).get("max_total_num_tokens")
    restored_pool = native["incumbent"].get("profile", {}).get("max_total_num_tokens")
    if previous_pool and restored_pool:
        blocks.append(
            f"Before the outage the original service had a {number(previous_pool)}-token KV pool. On restart, with the "
            f"paid miner offline and the original GPU memory cap absent, the same deployment allocated {number(restored_pool)} tokens "
            f"({100*(restored_pool/previous_pool-1):+.1f}%). The restored throughput figures use that larger pool. "
            "[Pre-outage profile](incumbent.json) · [Restored profile](incumbent-native.json)"
        )
    restoration = read(directory / "deployment-restoration.json")
    if restoration and revisions.get("production_after_oom_restart", {}).get("arithmetic_returned_42"):
        blocks.append(
            "After the 524K failure, the original container restarted, passed all four startup checks and returned "
            "42 on a fresh chat request. Its 96,316 MiB per-card limits had reset. The paid F2Pool service "
            "was offline at the end of the benchmark and restoring those limits required administrator access. "
            "The November 21 launch applied them during weight loading, before KV allocation. "
            "[Deployment recovery record](deployment-restoration.json)"
        )
    footprints = {
        key: val.get("host_footprint_gib", {}) for key, val in revisions.get("profiles", {}).items()
    }
    rebuilt = [sample for arm, sample in footprints.items() if arm in ("stock-throughput", "pearl")]
    incumbent_locked = footprints.get("incumbent-unlocked", {}).get("sampled_unevictable_high_water")
    if (
        len(rebuilt) == 2
        and all("sampled_anon_high_water" in sample for sample in rebuilt)
        and incumbent_locked is not None
    ):
        blocks.append(
            f"The rebuilt service cgroups peaked at {span([q['sampled_anon_high_water'] for q in rebuilt], 1)} GiB "
            f"anonymous and {span([q['sampled_unevictable_high_water'] for q in rebuilt], 1)} GiB unevictable, "
            f"reaching {span([q['sampled_cgroup_high_water'] for q in rebuilt], 1)} GiB charged including file cache "
            f"under a 365 GiB cap. The original Docker scope separately recorded {incumbent_locked:.1f} GiB unevictable. "
            "[Host memory accounting](full-run.json)"
        )
    telemetry = active_hardware(directory)
    if telemetry:
        blocks.append(
            "Two-second telemetry sampled active devices during timed native requests. The memory and SM clock ranges cover samples with "
            "nonzero utilization. Peak VRAM is the range of the four board maxima observed during recorded requests. "
            "A positive leading gap means telemetry began after the benchmark. [Hardware measurements](hardware-throughput.json)"
        )
        blocks.append(
            table(
                [
                    "Profile",
                    "Active GPU samples",
                    "Power limit (W)",
                    "Active memory clock (MHz)",
                    "Active SM clock (MHz)",
                    "Peak VRAM per GPU (GiB)",
                    "Leading gap (s)",
                ],
                telemetry,
            )
        )
        phases = read(directory / "hardware-throughput.json").get("phases", {})
        matched = []
        for arm in NAMES:
            for group in ("native", "handbook"):
                samples = phases.get(f"{arm}-{group}", {}).get("within_recorded_requests", {})
                if len(samples) != 4 or any(not board.get("active_samples") for board in samples.values()):
                    break
                matched.extend(samples.values())
        if len(matched) == 24:
            memory = {
                board["memory_clock_mhz_when_active"][bound] for board in matched for bound in ("min", "max")
            }
            caps = {w for board in matched for w in board["power_limit_w"]}
            if len(memory) == len(caps) == 1:
                blocks.append(
                    f"Across all three native and long-handbook series, every active board sample had "
                    f"a {number(next(iter(memory)))} MHz memory clock and a {number(next(iter(caps)))} W cap."
                )
    cover = []
    for k, label in NAMES.items():
        series = []
        for d, filename in [
            (native[k], f"{k}-native.json"),
            (handbook[k], f"{k}-handbook.json"),
            (window[k], f"window-{k}.json"),
            (prefix[k], f"{k}-prefix.json"),
        ]:
            if not d:
                series.append(DASH)
                continue
            state = "complete" if d.get("complete", d.get("answer_check") is not None) else "partial"
            if d.get("status") == "failed":
                state = "failed"
            series.append(f"[{state}]({filename})")
        cover.append([label, *series])
    blocks.append(
        table(["Profile", "Native stream", "Long handbook", "Near-full window", "Prefix reuse"], cover)
    )
    mixed = []
    for key, label in NAMES.items():
        for phase, receipt in (("native", native[key]), ("handbook", handbook[key]), ("prefix", prefix[key])):
            before = (receipt.get("http_before") or {}).get("total")
            after = (receipt.get("http_after") or {}).get("total")
            expected = receipt.get("expected_inference_requests")
            if None not in (before, after, expected) and after - before != expected:
                mixed.append(
                    f"{label} {phase} observed {after-before:g} total inference requests across the run, "
                    f"including {expected} benchmark requests"
                )
    if mixed:
        blocks.append("Session HTTP counters recorded overlapping activity. " + ". ".join(mixed) + ".")
    blocks.append(
        "A dash marks an unavailable metric or receipt. Trial counts are small and include any first-use costs during the series. "
        "Clock, load and accepted length can all affect these rates. "
        "[Native benchmark](throughput.py) · [Full-window replay](full_window.py) · "
        "[September handbook](../../../../bench/deepseek-v4.1-flash_TP4_sglang/BENCHMARK-LONG-TPS-2026-09-15.md) · "
        "[September full window](../../../../bench/deepseek-v4.1-flash_TP4_sglang/BENCHMARK-FULL-PREFILL-RESTORED-085-2026-09-16.md)"
    )
    blocks.append(
        "Regenerate from this directory with `python3 report_performance.py` in a Python environment with Matplotlib. "
        "Use `--no-plot` for a standard-library-only Markdown build."
    )
    (output / "performance.md").write_text("\n\n".join(blocks) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build(args.source_dir, args.output_dir, not args.no_plot)
