"""Plot aggregate serving, direct mining and cooling counters for one recorded F2Pool trial."""
import argparse
import json
from itertools import chain
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

NS = 1_000_000_000
MAX_GAP = 8 * NS  # The trial collector samples every two seconds.
MACS_PER_TILE = 160 * 256 * 65_536
CALIBRATION = Path(__file__).resolve().parents[1] / "direct-calibration.json"


def measurements(row, source):
    if source in row["errors"] or source not in row:
        return None
    if source == "server":
        progress = row["server"]["scheduler_progress"]
        keys = ("realtime_tokens_total.prefill_compute", "realtime_tokens_total.decode")
        if all(key in progress for key in keys):
            return row["began_ns"], tuple(progress[key] for key in keys)
    else:
        idle = row["idle"]
        stamp = idle["source_monotonic_ns"]
        workers, ranks = idle["workers"], idle["ranks"]
        if (0 <= row["end_ns"] - stamp <= MAX_GAP and len(workers) == len(ranks) == 4
                and all(w["watched"] and not w["faulted"] for w in workers)
                and all(isinstance(r.get("cooling_seconds"), (int, float)) for r in ranks)):
            return stamp, (tuple(w["completed_slices"] for w in workers)
                           + tuple(r["cooling_seconds"] for r in ranks))
    return None


def bins(rows, source, start, stop, seconds):
    """Use whole counter deltas between actual sample endpoints. Break at gaps or resets."""
    first = previous = None
    points = (measurements(row, source) for row in rows)
    for point in chain(points, [None]):
        if point and not start <= point[0] <= stop:
            point = None
        continuous = (point is not None and previous is not None
                      and 0 < point[0] - previous[0] <= MAX_GAP
                      and all(new >= old for old, new in zip(previous[1], point[1], strict=True)))
        if not continuous:
            if first and previous and first[0] < previous[0]:
                yield first, previous
            first = point
        elif point[0] - first[0] >= seconds * NS:
            yield first, point
            first = point
        previous = point


def rates(intervals, start, source):
    result = []
    for (t0, old), (t1, new) in intervals:
        dt = (t1 - t0) / NS
        delta = [b-a for a, b in zip(old, new, strict=True)]
        if source == "idle":
            # K780 at K65536 uses 188 tiles per slice, with a 168-tile final slice.
            # At most ceil(n / 139) short slices can occur in n adjacent slices per GPU.
            hi_tiles = 188 * sum(delta[:4])
            lo_tiles = hi_tiles - 20 * sum((n + 138) // 139 for n in delta[:4])
            values = (lo_tiles * MACS_PER_TILE / dt / 1e12, hi_tiles * MACS_PER_TILE / dt / 1e12,
                      sum(delta[4:]) / dt)
        else:
            values = tuple(n / dt for n in delta)
        result.append(((t0-start) / NS / 60, (t1-t0) / NS / 60, values))
    return result


def bars(ax, intervals, value, label, color, end):
    x = [left for left, width, vals in intervals]
    widths = [w for left, w, vals in intervals]
    heights = [vals[value] for left, width, vals in intervals]
    ax.bar(x, heights, width=widths, align="edge", color=color, linewidth=0, zorder=2)
    last = 0
    for left, width, _ in intervals:
        if left > last + 1e-8:
            ax.axvspan(last, left, color="#dde1e3", linewidth=0)
        last = left + width
    if last < end:
        ax.axvspan(last, end, color="#dde1e3", linewidth=0)
    ax.set_title(label, loc="left", fontsize=12, fontweight="bold", pad=5)
    ax.set_ylim(0, max(1, max(heights, default=0)) * 1.14)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def plot(metrics, output, bin_seconds):
    window = json.loads((metrics / "window.json").read_text())
    with (metrics / "series.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.endswith("\n")]
    if not rows:
        raise ValueError("No complete telemetry samples")
    start = window["start_monotonic_ns"]
    stop = min(window["end_monotonic_ns"], max(row["end_ns"] for row in rows))
    elapsed = (stop - start) / NS / 60
    fleet = rates(bins(rows, "idle", start, stop, bin_seconds), start, "idle")
    serving = rates(bins(rows, "server", start, stop, bin_seconds), start, "server")
    if elapsed <= 0 or not (fleet or serving):
        raise ValueError("No consecutive samples inside the trial window")
    pure = json.loads(CALIBRATION.read_text())["calibration"]["total_tmac_s"]
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False, "axes.spines.right":False,
                         "axes.spines.left":False, "axes.edgecolor":"#a5adb2", "xtick.color":"#46545c",
                         "ytick.color":"#46545c", "axes.titlecolor":"#243844", "axes.axisbelow":True})
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1, 1, .7]})
    fig.subplots_adjust(left=.09, right=.977, top=.865, bottom=.12, hspace=.57)
    fig.text(.09, .95, "DeepSeek serving and Pearl mining", fontsize=22, weight="bold", color="#20343f")
    fig.text(.09, .914, f"Four SM120 GPUs at 275 W each  ·  approximately {bin_seconds}-second averages",
             fontsize=11.5, color="#475861")
    bars(axes[0], fleet, 1, "Direct mining  ·  TMAC/s", "#a2c2d3", elapsed)
    axes[0].bar([x for x,w,v in fleet], [v[0] for x,w,v in fleet],
                width=[w for x,w,v in fleet], align="edge", color="#23668c", linewidth=0, zorder=3)
    axes[0].axhline(pure, color="#303f49", ls=(0, (4, 3)), lw=1.2, zorder=4,
                    label=f"Miner-only calibration   {pure:,.0f} TMAC/s")
    axes[0].set_ylim(0, max(axes[0].get_ylim()[1], pure * 1.30))
    axes[0].legend(loc="upper right", frameon=False, fontsize=9, handlelength=2.5)
    bars(axes[1], serving, 0, "Computed prefill  ·  scheduler tokens/s", "#c37632", elapsed)
    bars(axes[2], serving, 1, "Decode  ·  scheduler tokens/s", "#347963", elapsed)
    bars(axes[3], fleet, 2, "GPUs on a mining thermal hold  ·  average count", "#88778a", elapsed)
    axes[3].set_ylim(0, 4.15)
    axes[3].set_yticks([0, 2, 4])
    for ax in axes:
        ax.set_xlim(0, elapsed)
        ax.grid(axis="y", color="#dae0e3", lw=.7)
        ax.tick_params(axis="y", length=0, labelsize=9, pad=7)
    axes[-1].set_xlabel("Minutes since the trial opened", labelpad=8)
    fig.text(.09, .035, "Rates use wall time. Direct mining bounds come from completed slices. "
             "Gray spans mark gaps in telemetry.", color="#516069", fontsize=9)
    fig.savefig(output, dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True, help="Folder with window.json and series.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="PNG, SVG, or PDF file")
    parser.add_argument("--bin-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.bin_seconds < 1:
        parser.error("--bin-seconds must be positive")
    plot(args.metrics_dir, args.output, args.bin_seconds)
