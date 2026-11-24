"""Compare a matched pair of automatic and workload-based memory clock measurements."""
import argparse
import json
import math
from pathlib import Path
from statistics import geometric_mean, median

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
    args = parser.parse_args()
    result = compare(json.loads(args.automatic.read_text()), json.loads(args.adaptive.read_text()))
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2)+"\n")
    tmp.replace(args.out)
