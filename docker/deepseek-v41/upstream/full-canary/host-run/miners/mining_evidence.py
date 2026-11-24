import collections
import datetime as dt
import itertools
import json
import time
from pathlib import Path
from hardware_summary import parse_samples

runtime=Path(__file__).resolve().parents[1]
base=runtime/'miners'
dst=Path('/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/docker/deepseek-v41/upstream/full-canary/mining-evidence.json')
prod_path=Path('/mnt/hot/ambientlight/repos/mining/pool-trial/five-gpu/f2pool/verified-live.json')
def read(p):
    return json.loads(p.read_text())
def now_utc(t):
    return dt.datetime.fromtimestamp(t,dt.timezone.utc).isoformat(timespec='milliseconds')

def time_source():
    # Both clocks share the same host kernel with the worker observer.
    return time.time()-time.monotonic()

def ownership(event):
    state=event['broker']
    if state['serving'] or not all(state['idle']) or state['claims']:
        return None
    workers=event['workers']
    if any(w['faulted'] or not w['watched'] or w['holds'] or not w['completed_slices'] for w in workers):
        return None
    return tuple(w['release_ns'] for w in workers)

def tiles_after(n):
    # Last slice in every 139-slice round has 168 tiles; cursor starts at tile 0.
    return n//139*26112+min(n%139*188,26112)

def idle():
    record_path=Path((base/'main-proof-run.txt').read_text().strip())/'handoff.jsonl'
    events=[json.loads(line) for line in record_path.read_text().splitlines()]
    skew=time_source()
    physical=parse_samples()
    windows=[]
    for key,interval in itertools.groupby(events,ownership):
        if key is None:continue
        v=list(interval)
        a,b=v[0],v[-1]
        secs=(b['monotonic_ns']-a['monotonic_ns'])/1e9
        if secs<5:continue
        a_time=a['monotonic_ns']/1e9+skew
        b_time=b['monotonic_ns']/1e9+skew
        by_gpu=[tiles_after(end['completed_slices'])-tiles_after(start['completed_slices']) for start,end in zip(a['workers'],b['workers'],strict=True)]
        fresh=[r for r in physical if a_time<=r['time']<=b_time and r['util']>0]
        windows.append(dict(start_utc=now_utc(a_time),end_utc=now_utc(b_time),seconds=secs,tiles_per_gpu=by_gpu,
                            tmac_s_by_gpu=[tiles*160*256*65536/secs/1e12 for tiles in by_gpu],
                            tmac_s_total=sum(by_gpu)*160*256*65536/secs/1e12,
                            physical_active_samples=len(fresh),memory_clock_mhz=sorted(set(r['memory_clock'] for r in fresh)),
                            sampled_power_w=[min(r['watts'] for r in fresh),max(r['watts'] for r in fresh)] if fresh else None))
        if len(windows)==3:break
    elapsed=sum(w['seconds'] for w in windows)
    rates=[sum(w['tiles_per_gpu'][i] for w in windows)*160*256*65536/elapsed/1e12 for i in range(4)]
    prod=read(prod_path)
    old=[prod['GPUs'][f'gpu{i}']['local_TMAC_s'] for i in range(4)]
    return dict(method='First three runs of at least five seconds with the same four generation-specific worker release timestamps at every sample, all broker ranks idle with no claims, and zero worker holds or faults. Completed work is counted after device fence. Full checkpoint was resident. These rates describe offline completed K780 tile work. No live pool connection or payout was part of this measurement.',
        math='One 160 x 256 tile at K=65536 has 2684354560 MACs. The v1 worker covers (M//160)*160 x 32768 rows/columns per nonce, with 26112 tiles and 139 slices at 188 maximum tiles per slice. Cumulative productive slice counters start at zero; conversion accounts exactly for each final 168-tile slice.',
        elapsed_seconds=elapsed,windows=windows,direct_sliced_rate_tmac_s_by_gpu=rates,sliced_total_tmac_s=sum(rates),
        immediately_pretest_f2pool=dict(
            source="journalctl -u pearl-f2pool.service --since '2026-11-23 16:49:20' --until '2026-11-23 16:49:23' (America/Los_Angeles)",
            utc="2026-11-24T00:49:21Z",local_average_tmac_s_by_gpu=[301.5,307.5,309.7,318.5],local_average_tmac_s_total=1237.1,
            total_board_power_w=1120,accepted=15451,rejected=17,stale=73,hottest_c=89,
            observed_relative_pct=100*sum(rates)/1237.1,
            basis="Final production snapshot four minutes before the supervisor exited; per-worker lifetime average reported to one decimal, not pool-side difficulty-weighted hashrate. The preceding five-minute status reported the same 1237.1 local aggregate."),
        archived_f2pool_by_gpu_local_tmac_s=old,archived_f2pool_total_local_tmac_s=sum(old),archived_relative_pct=100*sum(rates)/sum(old),
        archived_f2pool_reference=dict(time_utc=prod['sample_utc'],source=str(prod_path),ack_counts=[prod['GPUs'][f'gpu{i}']['accepted'] for i in range(4)],power_limit_w=280,memory_clock_mhz=[prod['GPUs'][f'gpu{i}']['memory_MHz'] for i in range(4)],configuration='Same physical GPUs, v1 K780 at K65536, pipeline depth 2. Historical mining process reported local MAC/s plus a cumulative accepted share count. Different intervals, slice/clock settings, temperatures and contention limit comparison.'))

def proof_snapshot(path):
    pool=read(path)
    grouped=collections.defaultdict(list)
    for row in pool['accepted']: grouped[row['proof_sha256']].append(row)
    pairs=[]
    for digest, rows in grouped.items():
        if len(rows)>1:
            pairs.append(dict(sha256=digest,tp_rank=rows[0]['tp_rank'],rows=rows[0]['proof_rows'],
                              received_utc=[now_utc(r['received_at']) for r in rows],delta_s=rows[-1]['received_at']-rows[0]['received_at']))
    return dict(acknowledged_total=pool.get('accepted_total',len(pool['accepted'])),retained_submissions=len(pool['accepted']),distinct_sha256=len(grouped),
                duplicate_pairs=pairs,rejected_frames=pool['rejected_frames'],negotiations=pool['negotiations'],target_hex=pool['job']['target'],
                keepalive_frames=pool.get('keepalive_frames'),verified_submissions_by_rank_rows=pool.get('verified_by_rank_rows'),peer_restarts=pool.get('peer_restarts'))

def main():
    hard=proof_snapshot(runtime/'pool-sustained.json')
    easy=proof_snapshot(runtime/'pool.json')
    test_file=base/'keepalive-125s.json'
    out=dict(as_of_utc=now_utc(time.time()),idle=idle(),worker_thermal_guard=read(base/'thermal-guard.json'),final_peer_before_control_off=read(base/'before-peer-stop-summary.json'),proofs=dict(
        easy=easy,sustained=hard,actual_125_second_keepalive_test=read(test_file),
        deduplication_scope='The loopback verifier checks consensus proof validity, target, and opened checkpoint shard. It ACKs valid retransmissions. All easy hashes were retained. The sustained peer records counts for all submissions but retains at most the first 16 hashes; neither its lifetime distinct count nor difficulty-weighted payable credit can be derived. The three repeated digests landed roughly 20.6 seconds apart at 01:21 UTC, before the explicit 01:24:44 UTC restart. No per-connection identifiers were retained for historical submissions. These gaps were below the original 90-second idle read timeout.'))
    temp=dst.with_suffix('.part')
    temp.write_text(json.dumps(out,indent=2)+'\n')
    temp.replace(dst)
if __name__=='__main__': main()
