"""Compare a matched pair of automatic and workload-based memory clock measurements."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import geometric_mean, median


def write_table(path, result):
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as stream:
        table = csv.writer(stream, lineterminator="\n")
        table.writerow(("workload", "context", "start", "samples_per_policy", "metric", "unit",
                        "automatic", "adaptive", "difference", "change_pct", "change_basis"))
        def record(workload, context, start, n, metric, unit, a, b, change, basis):
            table.writerow((workload, context, start, n, metric, unit, round(a, 4), round(b, 4),
                            round(b-a, 4), "" if change is None else round(change, 3), basis))

        for cohort in result["cohorts"]:
            for field, label, unit in (("ttft_ms", "Time to first token", "ms"),
                                       ("server_prefill_tps", "Prefill", "tokens/s"),
                                       ("client_decode_tps", "Streaming decode", "tokens/s")):
                metric = cohort["metrics"][field]
                record("DeepSeek Responses", cohort["case"], cohort["position"], cohort["requests"],
                       label, unit, metric["automatic"], metric["adaptive"], metric["paired_change_pct"],
                       "Cohort medians with percent from paired request ratios")

        a, b = (result["mining"][key] for key in ("automatic", "adaptive"))
        record("Idle K780", "Four GPUs", "idle", 1, "Completed mining work", "TMAC/s",
               a["fleet_tmac_s"], b["fleet_tmac_s"], result["mining_change_pct"], "One 90-second window per policy")
        for i, (x, y) in enumerate(zip(a["rank_tmac_s"], b["rank_tmac_s"], strict=True)):
            record("Idle K780", f"GPU {i}", "idle", 1, "Completed mining work", "TMAC/s", x, y,
                   100*(y/x-1), "One 90-second window per policy")
        for i, name in enumerate(("Accepted shares", "Rejected shares")):
            record("F2Pool", "Four GPUs", "idle", 1, name, "shares", sum(s[i] for s in a["shares"]),
                   sum(s[i] for s in b["shares"]), None, "Observed pool replies in the mining window")
        record("Idle K780", "Hottest GPU", "idle", 1, "Peak core temperature", "degrees C",
               a["peak_core_c"], b["peak_core_c"], None, "Maximum observed during the mining window")
        for key, label in (("demand_to_all_drained_ms", "Demand to all miners drained"),
                           ("client_terminal_to_first_gpu_ms", "Terminal response to first GPU mining")):
            x, y = a["handoff"][key], b["handoff"][key]
            if x["n"] == y["n"] and x["n"]:
                record("GPU handoff", "Four GPUs", "all", x["n"], label, "ms", x["median"], y["median"],
                       None, "Median over matched requests")
    tmp.replace(path)

def compare(left, right):
    if (left["policy"], right["policy"]) != ("automatic", "adaptive") or not all(x["complete"] for x in (left, right)):
        raise ValueError("Expected complete automatic and adaptive arms")
    if left["run"] != right["run"] or left["worker_identity"] != right["worker_identity"]:
        raise ValueError("Cannot compare different serving processes or F2Pool workers")
    groups = {}
    outputs_identical = 0
    for a, b in zip(left["rows"], right["rows"], strict=True):
        if any(a[k] != b[k] for k in ("index", "case", "position", "prompt_sha256", "input_tokens", "output_tokens", "cached_tokens")):
            raise ValueError("Unmatched inference pair")
        initial = 7001 if a["position"] == "cold" else 13365
        if a["before_mhz"] != [13365]*4 or b["before_mhz"] != [initial]*4:
            raise ValueError("Request did not start in the intended clock state")
        groups.setdefault((a["case"], a["position"]), []).append((a,b))
        outputs_identical += a["output_sha256"] == b["output_sha256"]
    cohorts = []
    for (case, position), pairs in groups.items():
        stats = {}
        for field in ("ttft_ms", "server_prefill_tps", "client_decode_tps"):
            values = [(a[field],b[field]) for a,b in pairs]
            stats[field] = dict(automatic=median(x for x,y in values), adaptive=median(y for x,y in values),
                paired_change_pct=100*(geometric_mean(y/x for x,y in values)-1),
                median_paired_difference=median(y-x for x,y in values))
        shifts = [b["all_high_from_http_ms"] for a,b in pairs]
        cohorts.append(dict(case=case, position=position, requests=len(pairs), metrics=stats,
            requests_observed_all_high=sum(x is not None for x in shifts),
            median_observed_ms_until_all_high=median(x for x in shifts if x is not None) if any(x is not None for x in shifts) else None))
    control, candidate = (x["mining"]["fleet_tmac_s"] for x in (left, right))
    within = all(c["metrics"]["client_decode_tps"]["paired_change_pct"] >= -10 for c in cohorts)
    if not cohorts or not all(math.isfinite(c["metrics"]["client_decode_tps"]["paired_change_pct"]) for c in cohorts):
        raise ValueError("Missing decode measurements")
    return dict(service_run=left["run"], pairs=len(left["rows"]), identical_generated_outputs=outputs_identical,
                cohorts=cohorts, observed_decode_within_ten_percent_in_every_cohort=within,
                mining_change_pct=100*(candidate/control-1),
                mining={arm["policy"]:arm["mining"] for arm in (left,right)},
                methodology="Rates are cohort medians. Percentage changes are geometric means of matched request ratios. The 10% comparison describes these samples. Bursts arrive within the five-second clock hold. Cold requests cut in after at least 6.5 seconds of uninterrupted idle mining.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("automatic", "adaptive", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--csv", type=Path, help="Optional separate comparison table")
    args = parser.parse_args()
    if args.csv is not None and args.csv.resolve() == args.out.resolve():
        parser.error("The CSV and JSON outputs need distinct paths")
    result = compare(json.loads(args.automatic.read_text()), json.loads(args.adaptive.read_text()))
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2)+"\n")
    tmp.replace(args.out)
    if args.csv is not None:
        write_table(args.csv, result)
