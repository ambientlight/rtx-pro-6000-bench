from pathlib import Path
import json
import sys

runtime = Path(__file__).resolve().parents[1]
d = runtime / "receipts"
client = runtime.parent / "deployment-source/docker/deepseek-v41/upstream/full-canary"
sys.path.insert(0, str(client))
from graph_compare import compare_traces, compare_handbooks
from throughput import write

modes = {"fixed-gap": ("fixed-gap-timed","fixed-gap","inference-acks"), "fixed-period": ("fixed-period","fixed-period","inference-acks"),
         "handoff-burst": ("handoff-burst","handoff-burst","inference-acks"), "equal-target-fixed-period": ("fixed-period","equal-target-fixed-period","inference-equal-acks")}
for dest, (an,bn,ledger) in modes.items():
    a,b,af,bf,pl = d/f"switch-{an}.json", d/f"joined-{bn}.json", d/f"switch-{an}-fleet.json", d/f"joined-{bn}-fleet.json", d/f"{ledger}.jsonl"
    if not all(p.is_file() for p in (a,b,pl)) or not json.loads(b.read_text()).get("complete"):
        continue
    fleet_ready = all(p.is_file() for p in (af,bf))
    summary = compare_traces(a,b,pl, af if fleet_ready else None, bf if fleet_ready else None)
    write(d/f"compare-{dest}.json", summary)
    print(dest, "matched", len(summary["pairs"]), "delta pct", round(summary["median_paired_decode_change_pct"],3), "fleet", fleet_ready)
a,b=d/"switch-handbook.json",d/"joined-handbook.json"
if a.exists() and b.exists() and all(json.loads(p.read_text())["complete"] for p in (a,b)):
    write(d/"compare-handbook.json", compare_handbooks(a,b))
