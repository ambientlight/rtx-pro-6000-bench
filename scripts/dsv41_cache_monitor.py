#!/usr/bin/env python3
"""Passive RAM-cache pressure telemetry. Only GET /metrics and /v1/loads.

Pool occupancy cannot prove per-prefix survival or attribute a future miss.
Boundary evictions with spare FULL capacity are warning candidates, not a
correctness verdict. Ordinary SWA checkpoint churn can be healthy.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

from dsv41_load import activity_metrics, atomic_json, fetch, mem_available, parse_loads, run


ROOT = Path("/mnt/hot/dsv41_state/cache-monitor")
CORE = {
    "full_used": "hicache_host_used_tokens",
    "full_capacity": "hicache_host_total_tokens",
    "swa_used": "hicache_swa_host_used_tokens",
    "swa_capacity": "hicache_swa_host_total_tokens",
    "swa_evicted": "hicache_swa_host_evicted_tokens",
    "boundary_evicted": "hicache_swa_host_boundary_evicted_tokens",
    "retention_enabled": "hicache_swa_host_retention_enabled",
}
OPTIONAL = {
    "device_hit_tokens": ("cached_tokens_total", {"cache_source": "device"}),
    "host_hit_tokens": ("cached_tokens_total", {"cache_source": "host"}),
    "backup_full_tokens_all_ranks": ("hicache_backup_tokens_total", {"pool": "kv"}),
    "backup_swa_tokens_all_ranks": ("hicache_backup_tokens_total", {"pool": "swa"}),
    "restore_full_tokens_all_ranks": ("load_back_tokens_total", {"pool": "kv"}),
    "restore_swa_tokens_all_ranks": ("load_back_tokens_total", {"pool": "swa"}),
}
DIAGNOSTIC = {
    "diagnostics_enabled": "hicache_swa_diagnostics_enabled",
    "match_lookups": "hicache_swa_match_lookups",
    "swa_limited_lookups": "hicache_swa_limited_match_lookups",
    "swa_limited_full_tokens": "hicache_swa_limited_match_full_tokens",
    "swa_limited_host_tokens": "hicache_swa_limited_match_host_tokens",
    "census_complete": "hicache_swa_census_complete",
    "census_timestamp_seconds": "hicache_swa_census_timestamp_seconds",
    "census_duration_seconds": "hicache_swa_census_duration_seconds",
    "census_nodes": "hicache_swa_census_nodes",
    "full_tree_tokens": "hicache_full_host_tree_tokens",
    "full_swa_covered_tokens": "hicache_full_host_swa_covered_tokens",
    "full_swa_uncovered_tokens": "hicache_full_host_swa_uncovered_tokens",
    "full_chain_gap_tokens": "hicache_full_host_chain_gap_tokens",
    "swa_marked_slots": "hicache_swa_host_marked_slots",
    "swa_unmarked_slots": "hicache_swa_host_unmarked_slots",
    "swa_marked_nodes": "hicache_swa_marked_nodes",
    "swa_marked_nodes_missing_window": "hicache_swa_marked_nodes_missing_window",
}
CAUSE_METRICS = {
    "swa_evicted": "hicache_swa_host_evicted_tokens_by_cause",
    "boundary_evicted": "hicache_swa_host_boundary_evicted_tokens_by_cause",
}
CAUSES = ("swa_pressure", "full_pressure", "other")
DISPOSITIONS = ("swa_only", "with_full")
CAUSE_KEYS = tuple(f"{kind}:{cause}:{disposition}"
                   for kind in CAUSE_METRICS for cause in CAUSES for disposition in DISPOSITIONS)
LOOKUP_COUNTERS = ("match_lookups", "swa_limited_lookups", "swa_limited_full_tokens", "swa_limited_host_tokens")
COUNTERS = ("swa_evicted", "boundary_evicted", *OPTIONAL, *CAUSE_KEYS, *LOOKUP_COUNTERS)
LINE = re.compile(r"^sglang:([a-zA-Z0-9_:]+)(?:\{(.*)\})?\s+(\S+)(?:\s+\d+)?$")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def read_metrics(raw):
    wanted = set(CORE.values()) | {name for name, _ in OPTIONAL.values()} | set(DIAGNOSTIC.values()) | set(CAUSE_METRICS.values())
    series = {}
    for line in raw.splitlines():
        match = LINE.fullmatch(line)
        if match is None or match[1] not in wanted:
            continue
        name, labels, value = match.groups()
        labels = dict(LABEL.findall(labels or ""))
        if labels.get("model_name", "deepseek-v4-flash") != "deepseek-v4-flash":
            continue
        value = float(value)
        is_time = name in {DIAGNOSTIC["census_timestamp_seconds"], DIAGNOSTIC["census_duration_seconds"]}
        if not math.isfinite(value) or value < 0 or (not is_time and value != int(value)):
            raise ValueError(f"Invalid cache metric {name}")
        series.setdefault(name, []).append((labels, value if is_time else int(value)))
    result = {}
    for key, name in CORE.items():
        values = [value for labels, value in series.get(name, [])
                  if labels.get("tp_rank") == "0" and labels.get("pp_rank", "0") == "0"]
        if len(values) != 1:
            raise ValueError(f"Missing or ambiguous rank-0 metric {name}")
        result[key] = values[0]
    if result["retention_enabled"] not in (0, 1):
        raise ValueError("Invalid retention flag")
    for pool in ("full", "swa"):
        if not 0 <= result[f"{pool}_used"] <= result[f"{pool}_capacity"] or result[f"{pool}_capacity"] <= 0:
            raise ValueError(f"Invalid {pool} pool occupancy")
    if result["boundary_evicted"] > result["swa_evicted"]:
        raise ValueError("Boundary evictions cannot exceed all SWA evictions")
    for key, (name, selector) in OPTIONAL.items():
        values = [value for labels, value in series.get(name, [])
                  if all(labels.get(k) == v for k, v in selector.items())]
        # Lazy-exported series are unknown, not fabricated zeros. The first
        # appearance establishes a baseline; later samples produce deltas.
        result[key] = sum(values) if values else None
    for key, name in DIAGNOSTIC.items():
        values = [value for labels, value in series.get(name, [])
                  if labels.get("tp_rank") == "0" and labels.get("pp_rank", "0") == "0"]
        if len(values) > 1:
            raise ValueError(f"Ambiguous diagnostic metric {name}")
        result[key] = values[0] if values else None
    for kind, name in CAUSE_METRICS.items():
        for cause in CAUSES:
            for disposition in DISPOSITIONS:
                values = [value for labels, value in series.get(name, [])
                          if labels.get("tp_rank") == "0" and labels.get("pp_rank", "0") == "0"
                          and labels.get("trigger") == cause and labels.get("disposition") == disposition]
                if len(values) > 1:
                    raise ValueError(f"Ambiguous eviction attribution {name}")
                result[f"{kind}:{cause}:{disposition}"] = values[0] if values else None
    if result["diagnostics_enabled"] not in (None, 0, 1):
        raise ValueError("Invalid diagnostics flag")
    if result["diagnostics_enabled"] == 1:
        if any(result[k] is None for k in (*DIAGNOSTIC, *CAUSE_KEYS)):
            raise ValueError("Incomplete enabled diagnostic schema")
        if result["census_complete"] not in (0, 1):
            raise ValueError("Invalid census-complete flag")
        for kind in CAUSE_METRICS:
            if sum(result[f"{kind}:{c}:{d}"] for c in CAUSES for d in DISPOSITIONS) != result[kind]:
                raise ValueError("Eviction attribution does not reconcile with legacy total")
        for c in CAUSES:
            for d in DISPOSITIONS:
                if result[f"boundary_evicted:{c}:{d}"] > result[f"swa_evicted:{c}:{d}"]:
                    raise ValueError("Attributed boundaries exceed evictions")
        if result["census_complete"] and (
            result["census_timestamp_seconds"] <= 0 or
            result["full_tree_tokens"] != sum(result[k] for k in ("full_swa_covered_tokens", "full_swa_uncovered_tokens", "full_chain_gap_tokens"))
        ):
            raise ValueError("Invalid completed residency census")
    else:
        # New exporters may emit default zeros on unsupported layouts. They
        # must not be mistaken for a measured absence of eviction or loss.
        for key in (*DIAGNOSTIC, *CAUSE_KEYS):
            if key != "diagnostics_enabled":
                result[key] = None
    return result


def container():
    template = '{"id":{{json .Id}},"started_at":{{json .State.StartedAt}},"restarts":{{json .RestartCount}},"state":{{json .State.Status}}}'
    result = json.loads(run("docker", "inspect", "--format", template, "dsv41"))
    if result["state"] != "running":
        raise RuntimeError("dsv41 is not running")
    return result


def sample():
    before = container()
    loads = parse_loads(json.loads(fetch("/v1/loads")))
    raw = fetch("/metrics")
    metrics = read_metrics(raw)
    active, arrivals = activity_metrics(raw)
    if container() != before:
        raise RuntimeError("Container changed during sample")
    return {"at": time.time(), "container": before, "metrics": metrics,
            "running_requests": sum(row["num_running_reqs"] for row in loads),
            "waiting_requests": sum(row["num_waiting_reqs"] for row in loads),
            "active_tokens": sum(row["num_active_tokens"] for row in loads),
            "http_active": active, "http_arrivals_total": sum(value for _, value in arrivals),
            "available_ram_bytes": mem_available()}


def analyze(current, previous, *, full_headroom=0.80, swa_pressure=0.90):
    metrics = current["metrics"]
    same = previous is not None and previous["container"] == current["container"]
    prior = previous["metrics"] if same else {}
    resets = [key for key in COUNTERS if metrics.get(key) is not None and prior.get(key) is not None
              and metrics[key] < prior[key]]
    # A cache reconstruction can occur without changing container identity.
    cache_reset = any(key in resets for key in ("swa_evicted", "boundary_evicted"))
    deltas = {key: metrics[key] - prior[key] if not cache_reset and metrics.get(key) is not None
              and prior.get(key) is not None and key not in resets else None for key in COUNTERS}
    if all(deltas[key] is not None for key in ("swa_evicted", "boundary_evicted")):
        ordinary = deltas["swa_evicted"] - deltas["boundary_evicted"]
        if ordinary < 0:
            raise ValueError("Inconsistent eviction deltas")
    else:
        ordinary = None
    full = metrics["full_used"] / metrics["full_capacity"]
    swa = metrics["swa_used"] / metrics["swa_capacity"]
    flags = []
    if not metrics["retention_enabled"]:
        flags.append("boundary_retention_disabled")
    if swa >= swa_pressure and full < full_headroom:
        flags.append("swa_pressure_with_full_headroom")
    if deltas["boundary_evicted"] is not None and deltas["boundary_evicted"] > 0:
        flags.append("boundary_eviction_with_full_headroom" if full < full_headroom else "boundary_eviction_observed")
    if resets:
        flags.append("counter_reset")
    census_age = None
    uncovered_fraction = None
    if metrics.get("diagnostics_enabled") == 1:
        census_age = current["at"] - metrics["census_timestamp_seconds"]
        if metrics["census_complete"] and 0 <= census_age <= 150:
            uncovered_fraction = (metrics["full_swa_uncovered_tokens"] / metrics["full_tree_tokens"]
                                  if metrics["full_tree_tokens"] else 0.0)
        else:
            flags.append("census_unavailable_or_stale")
        if (deltas.get("boundary_evicted:swa_pressure:swa_only") or 0) > 0:
            flags.append("boundary_swa_pressure_without_full_removal")
        if sum(deltas.get(f"boundary_evicted:{c}:with_full") or 0 for c in CAUSES) > 0:
            flags.append("boundary_eviction_alongside_full")
    return {"full_used_fraction": full, "swa_used_fraction": swa, "deltas": deltas,
            "ordinary_swa_slots_evicted_delta": ordinary, "flags": flags, "counter_resets": resets,
            "interval_seconds": current["at"] - previous["at"] if same else None,
            "census_age_seconds": census_age,
            "full_swa_uncovered_fraction_of_tree_host_tokens": uncovered_fraction,
            "classification": ("attributed_evictions_and_resumable_prefix_census" if metrics.get("diagnostics_enabled") == 1
                               else "aggregate_candidate_only_not_per_prefix_attribution")}


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, separators=(",", ":")) + "\n")


def record(root, current, previous):
    stamp = datetime.fromisoformat(current["container"]["started_at"].replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%SZ")
    folder = root / f"launch-{stamp}-{current['container']['id'][:12]}"
    folder.mkdir(exist_ok=True)
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        "monitoring_started_at": current["at"], "samples": 0,
        "peak_full_used_fraction": 0, "peak_swa_used_fraction": 0,
        "observed_boundary_slots_evicted": 0, "observed_swa_slots_evicted": 0,
        "counter_resets": 0, "boundary_eviction_with_full_headroom_intervals": 0,
    }
    # The per-launch summary is the atomic checkpoint. A collector crash
    # between updating it and latest.json must not double-count old deltas.
    previous = summary.get("latest_sample", previous)
    analyzed = analyze(current, previous)
    current = {**current, "analysis": analyzed}
    summary.update(last_sample_at=current["at"], container=current["container"], latest_analysis=analyzed,
                   latest_metrics=current["metrics"], latest_sample=current)
    summary["samples"] += 1
    summary["counter_resets"] += bool(analyzed["counter_resets"])
    for pool in ("full", "swa"):
        key = f"peak_{pool}_used_fraction"
        summary[key] = max(summary[key], analyzed[f"{pool}_used_fraction"])
    for kind in ("swa", "boundary"):
        summary[f"observed_{kind}_slots_evicted"] += analyzed["deltas"][f"{kind}_evicted"] or 0
    if "boundary_eviction_with_full_headroom" in analyzed["flags"]:
        summary["boundary_eviction_with_full_headroom_intervals"] += 1
    day = datetime.fromtimestamp(current["at"], timezone.utc).strftime("%Y-%m-%d")
    append(folder / f"samples-{day}.jsonl", current)
    previous_flags = previous.get("analysis", {}).get("flags", []) if previous and previous["container"] == current["container"] else []
    if analyzed["flags"] != previous_flags or (analyzed["deltas"]["boundary_evicted"] or 0) > 0:
        event = {"at": current["at"], "analysis": analyzed, "metrics": current["metrics"]}
        append(folder / f"events-{day}.jsonl", event)
        print(json.dumps({"event": "cache_pressure_observation", "launch": folder.name, **analyzed}), flush=True)
    atomic_json(summary_path, summary)
    atomic_json(root / "latest.json", current)
    atomic_json(root / "status.json", {"state": "monitoring", "updated_at": current["at"],
                                      "launch_directory": str(folder), "flags": analyzed["flags"]})
    temporary = root / "current.tmp"
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(folder.name)
    temporary.replace(root / "current")
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 10:
        parser.error("Require a finite interval of at least ten seconds")
    os.umask(0o077)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest = args.root / "latest.json"
        previous = json.loads(latest.read_text()) if latest.exists() else None
        while True:
            try:
                previous = record(args.root, sample(), previous)
            except Exception as exc:
                failure = {"state": "unavailable", "updated_at": time.time(), "error": type(exc).__name__,
                           "last_valid_sample_at": previous["at"] if previous else None}
                atomic_json(args.root / "status.json", failure)
                print(json.dumps(failure), flush=True)
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
