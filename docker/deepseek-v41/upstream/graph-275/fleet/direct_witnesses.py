"""Attribute verified K780 tickets to GPU cursor intervals, independent of delayed verification."""
import argparse
import bisect
import json
from pathlib import Path


def rows(p):
    with p.open() as f:
        return [json.loads(l) for l in f if l.endswith("\n")]


def within_round(group):
    # kernel780.cu group=((bm*2+wr)*512+bn*4+wc)*32+lane.
    # build.rs compiles pm_k780_slice with the default row-major ordering.
    rowwarp,colwarp = divmod(group//32,512)
    tile=(rowwarp//2)*128+colwarp//4
    if not 0<=tile<26112:
        raise ValueError("invalid direct group")
    return tile//188+1


def gather(run, observers):
    p=json.loads((run/'prewarm.json').read_text())
    states=[p[k] for k in ('fresh_parked','all_mining','fleet_held','checked_stable','parked_with_no_test_clients')]
    states+=rows(run/'handoff.jsonl')
    for path in observers:
        states.extend(e['state'] for e in rows(path) if e['type'] in ('start','workers','stop'))
    # Snapshot intervals can overlap across supervisors. Sorting by timestamp
    # guarantees causality for the individually selected upper/lower bounds.
    states.sort(key=lambda s:s['began_ns'])
    ended=sorted(states,key=lambda s:s['monotonic_ns'])
    starts=[s['began_ns'] for s in states]
    ends=[s['monotonic_ns'] for s in ended]
    captures=[{}, {}, {}, {}]
    losses=[]
    for rank in range(4):
        events=rows(run/f'gpu{rank}'/'events.jsonl')
        if sum(v['event']=='sliced_job' for v in events)!=1 or sum(v['event']=='authorized' for v in events)!=1:
            raise ValueError("fixed job/session continuity failed")
        invalid={'error','round_discarded','proof_queue_full','submit_unknown','share_discarded'}
        cut=next((i for i,v in enumerate(events) if v['event'] in invalid),len(events))
        losses.append(dict(first_non_fifo_event=events[cut]['event'] if cut<len(events) else None,
                           first_non_fifo_unix=events[cut]['time'] if cut<len(events) else None,
                           discarded_after_cut=sum(v['event']=='share_discarded' for v in events[cut:])))
        # Stream termination can discard still-queued tickets. Only the
        # uninterrupted verified prefix is usable for the FIFO cut proof.
        for v in events[:cut]:
            if v['event']=='share_submitted':
                captures[rank][v['proof_sha256']]=(v['nonce'],within_round(v['group']))
    hits=[[] for _ in range(4)]
    for e in rows(run/'peer/ledger.jsonl'):
        rank=e['rank']
        if e["sha256"] not in captures[rank]: continue
        nonce,local=captures[rank][e['sha256']]
        hits[rank].append((e['received_ns'],nonce,local))
    # Origin is the nonce of the first productive round; engine increments
    # by one every 139 slices. First-seen hit is later than or equal to origin.
    # A verified witness's slice must already have retired before peer receipt.
    seeds=[]
    for rank,items in enumerate(hits):
        upper=min(n for _,n,_ in items)
        lower=0
        for received,nonce,local in items:
            idx=bisect.bisect_left(starts,received)
            if idx>=len(states):continue
            cursor=states[idx]['workers'][rank]['completed_slices']
            lower=max(lower,nonce-((cursor-local)//139))
        if lower>upper or upper-lower>=1000:
            raise ValueError(f"invalid nonce-origin width on rank {rank}: {upper-lower}")
        seeds.append((lower,upper))
    def threshold(t):
        lo=bisect.bisect_right(ends,t)-1
        hi=bisect.bisect_left(starts,t)
        if lo<0 or hi>=len(states): raise ValueError("open observation edge")
        return [(x['completed_slices'],y['completed_slices']) for x,y in zip(ended[lo]['workers'],states[hi]['workers'],strict=True)]
    return hits,seeds,threshold,losses


def generated(trace,data):
    hits,seeds,threshold,losses=data
    lo=threshold(trace['requested_horizon_start_monotonic_ns'])
    hi=threshold(trace['requested_horizon_end_monotonic_ns'])
    by_rank=[]
    for rank,items in enumerate(hits):
        seedlo,seedhi=seeds[rank]
        countlo=None
        counthi=0
        for n0 in range(seedlo,seedhi+1):
            positions=[(nonce-n0)*139+local for _,nonce,local in items]
            proven=sum(lo[rank][1]<pos<=hi[rank][0] for pos in positions)
            possible=sum(lo[rank][0]<pos<=hi[rank][1] for pos in positions)
            countlo=proven if countlo is None else min(countlo,proven)
            counthi=max(counthi,possible)
        later=any((nonce-seedhi)*139+local>hi[rank][1] for _,nonce,local in items)
        by_rank.append(dict(verified_min=countlo,verified_max=counthi,origin_width_rounds=seedhi-seedlo,
                            verified_later_slice_seen=later))
    complete=all(x['verified_later_slice_seen'] for x in by_rank)
    return dict(by_rank=by_rank,lower=sum(x['verified_min'] for x in by_rank),
                upper=sum(x['verified_max'] for x in by_rank) if complete else None,
                observed_verified_upper=sum(x['verified_max'] for x in by_rank),
                complete_through_end_by_fifo=complete, end_of_fifo_by_rank=losses, basis="proof's GPU-produced nonce and row-major K780 group, checked CPU receipt, and completed slice cursors; protocol target fixed; only fully classified time windows bound all generated valid local witnesses",
                source="kernel780.cu group mapping; engine.rs and support.cu nonce order; one fixed job and FIFO proof/submit stream per rank")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fleet_run',type=Path);p.add_argument('trace',type=Path);p.add_argument('output',type=Path)
    p.add_argument('observers',nargs='+',type=Path);a=p.parse_args()
    result=generated(json.loads(a.trace.read_text()),gather(a.fleet_run,a.observers))
    with a.output.open('x') as out: json.dump(result,out,indent=2);out.write("\n")
