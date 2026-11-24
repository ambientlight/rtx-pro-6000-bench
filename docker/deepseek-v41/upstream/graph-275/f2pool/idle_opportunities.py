"""Measure K780 mining work over sample intervals with continuously free GPU workers."""
from bisect import bisect_right
import json

RANKS = range(4)
GPUS = len(RANKS)
TILE_MACS = 160 * 256 * 65536
MAX_GAP_NS = 4_000_000_000
MAX_STATUS_AGE_NS = 2_000_000_000
MIN_INTERVALS = 3
MIN_SECONDS = 10


def interruptions(fleet_dir):
    sessions = [[] for _ in RANKS]
    with (fleet_dir/"events.jsonl").open() as fp:
        for line in fp:
            if line.endswith("\n"):
                event=json.loads(line)
                if event["event"] in ("authorized","connection_end"):
                    sessions[event["rank"]].append(event["monotonic_ns"])
    demand = []
    serving = None
    with (fleet_dir/"broker-events.jsonl").open() as fp:
        for line in fp:
            if not line.endswith("\n"):
                continue
            event=json.loads(line)
            if event["event"]=="state" and event["serving"]!=serving:
                serving=event["serving"]
                if serving:
                    demand.append(event["monotonic_ns"])
    return [sorted(rows) for rows in sessions],sorted(demand)


def contains_event(events,start,end):
    return bisect_right(events,start)!=bisect_right(events,end)


def eligible(endpoint,i):
    worker,rank=endpoint["workers"][i],endpoint["ranks"][i]
    return (worker["watched"] is True and worker["faulted"] is False and worker["holds"]==0
            and worker["release_ns"]>0 and rank["connected"] is True and rank["cooling"] is False)


def tile_bounds(slices):
    # Each 26,112-tile nonce has 138 full 188-tile slices and a 168-tile
    # tail. New pool jobs reset to the first tile. A completed-slice interval
    # starting at any cursor therefore has at most ceil(count / 139) tails.
    upper=slices*188
    return upper - 20*((slices+138)//139),upper


class Measurement:
    def __init__(self):
        self.samples=0
        self.zero_progress=0
        self.elapsed_ns=0
        self.lower_tiles=0
        self.upper_tiles=0

    def add(self,duration,counts):
        self.samples+=1
        self.zero_progress+=int(not any(counts))
        self.elapsed_ns+=duration
        for count in counts:
            lo,hi=tile_bounds(count)
            self.lower_tiles+=lo
            self.upper_tiles+=hi

    def report(self,span_ns,baseline):
        seconds=self.elapsed_ns/1e9
        result=dict(qualifying_intervals=self.samples,included_s=seconds,
                    pct_of_observed_window=100*self.elapsed_ns/span_ns if span_ns>0 else None,
                    intervals_with_zero_completed_slices=self.zero_progress)
        if self.samples>=MIN_INTERVALS and seconds>=MIN_SECONDS:
            bounds=[x*TILE_MACS/seconds/1e12 for x in (self.lower_tiles,self.upper_tiles)]
            result.update(tmac_s_bounds=bounds,historical_pure_tmac_s=baseline,
                          pct_of_historical_pure_bounds=[100*x/baseline for x in bounds])
        return result


def measure(samples,start,end,fleet_dir,calibration):
    sessions,demand=interruptions(fleet_dir)
    cards=[Measurement() for _ in RANKS]
    fleet=Measurement()
    eligible_pairs=0
    for a,b in zip(samples,samples[1:]):
        x,y=a.get("idle"),b.get("idle")
        if x is None or y is None:
            continue
        t0,t1=(s["source_monotonic_ns"] for s in (x,y))
        dt=t1-t0
        if not (start<=t0<t1<=end and dt<=MAX_GAP_NS):
            continue
        if any(not 0<=sample["end_ns"]-endpoint["source_monotonic_ns"]<=MAX_STATUS_AGE_NS
               for sample,endpoint in ((a,x),(b,y))):
            continue
        if x["broker"]["serving"] or y["broker"]["serving"] or contains_event(demand,t0,t1):
            continue
        eligible_pairs+=1
        kept=[]
        for i in RANKS:
            if not (eligible(x,i) and eligible(y,i)) or contains_event(sessions[i],t0,t1):
                continue
            old,new=(s["workers"][i] for s in (x,y))
            count=new["completed_slices"]-old["completed_slices"]
            if (old["release_ns"]!=new["release_ns"] or old["release_ns"]>t0 or count<0
                    or x["ranks"][i]["cooling_seconds"]!=y["ranks"][i]["cooling_seconds"]):
                continue
            cards[i].add(dt,[count])
            kept.append(count)
        if len(kept)==GPUS:
            fleet.add(dt,kept)
    return dict(max_sample_gap_s=MAX_GAP_NS/1e9,max_status_age_s=MAX_STATUS_AGE_NS/1e9,
                rate_reporting_minimum={"seconds":MIN_SECONDS,"intervals":MIN_INTERVALS},
                broker_idle_candidate_pairs=eligible_pairs,
                by_gpu=[dict(gpu=i,**m.report(end-start,calibration["tmac_s_by_gpu"][i])) for i,m in enumerate(cards)],
                all_four_simultaneously_free=fleet.report(end-start,calibration["total_tmac_s"]),
                basis="Duration-weighted completed K780 slice work. Every included interval kept the same native last-release clock, zero holds, continuous pool session, four-rank broker idle, and healthy workers. Zero-progress intervals remain included. Work rates use a separate finite-tail tile bound for each GPU.")
