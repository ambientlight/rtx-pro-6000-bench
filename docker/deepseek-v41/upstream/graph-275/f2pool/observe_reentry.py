"""Watch broker idle edges and sample native worker control clocks without using a GPU API."""
import argparse
import asyncio
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time

WINDOW = ("start_monotonic_ns","end_monotonic_ns")
NATIVE = ("release_ns","first_gpu_seen_ns","first_slice_end_ns","completed_slices","holds","watched","faulted")
EARLY_NS = 150_000_000


def cool_state(fleet):
    raw = json.loads((fleet/"status.json").read_text())
    return dict(published_ns=raw["monotonic_ns"],holds=[r["cooling"] for r in raw["ranks"]])


async def worker(control, rank):
    async with asyncio.timeout(.12):
        reader,writer = await asyncio.open_unix_connection(str(control/f"gpu{rank}.sock"))
        try:
            writer.write(b"STATUS\n")
            await writer.drain()
            row=json.loads(await reader.readline())
            return {k:row[k] for k in NATIVE}
        finally:
            writer.close()
            await writer.wait_closed()


def inotify(path, fn):
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd<0:
        raise OSError(ctypes.get_errno(),"inotify_init1")
    if libc.inotify_add_watch(fd,os.fsencode(path),2|4|1024|2048)<0:
        os.close(fd)
        raise OSError(ctypes.get_errno(),"inotify_add_watch")
    def changed():
        try:
            os.read(fd,65536)
        except BlockingIOError:
            pass
        fn()
    asyncio.get_running_loop().add_reader(fd,changed)
    return fd


class Observer:
    def __init__(self, log, fleet, control):
        self.log,self.fleet,self.control=log,fleet,control
        self.previous=None
        self.held=None
        self.task=None
        self.current=None
        self.done=asyncio.Event()

    def emit(self, kind, **data):
        print(json.dumps(dict(event=kind,observed_ns=time.monotonic_ns(),**data),separators=(",",":")),file=self.log,flush=True)

    def start_edge(self, event):
        if self.task:
            self.task.cancel()
        self.current=event["monotonic_ns"]
        self.task=asyncio.create_task(self.watch(self.current))
        self.emit("idle",edge_ns=self.current,prior_cool=cool_state(self.fleet))

    def update(self, event):
        if event.get("event")!="state":
            return
        serving=event["serving"]
        self.held=event["held"]
        if self.previous is True and not serving:
            self.start_edge(event)
        if not self.previous and serving and self.current is not None:
            self.emit("new_demand",edge_ns=self.current,end_ns=event["monotonic_ns"])
            if self.task:
                self.task.cancel()
            self.current=None
        self.previous=serving

    async def watch(self, edge):
        found=set()
        # Trace all ranks for the first 150ms. The slower tail queries only ranks
        # for which an initial usable GPU timestamp has yet to be observed.
        try:
            while not self.done.is_set() and not self.previous:
                start=time.monotonic_ns()
                elapsed=start-edge
                ranks=list(range(4)) if elapsed<=EARLY_NS else sorted(set(range(4))-found)
                if not ranks:
                    return
                rows=await asyncio.gather(*(worker(self.control,r) for r in ranks),return_exceptions=True)
                if self.previous or self.current!=edge:
                    return
                results={str(r):row for r,row in zip(ranks,rows) if isinstance(row,dict)}
                for key,row in results.items():
                    if row["first_gpu_seen_ns"]>=row["release_ns"]>=edge and row["watched"] and not row["faulted"]:
                        found.add(int(key))
                try:
                    thermal=cool_state(self.fleet)
                except (OSError,ValueError,KeyError):
                    thermal=None
                self.emit("workers",edge_ns=edge,began_ns=start,ended_ns=time.monotonic_ns(),
                          broker_held=self.held,workers=results,thermal=thermal)
                if found==set(range(4)):
                    return
                await asyncio.sleep(.007 if elapsed<EARLY_NS else .15)
        except (OSError,ValueError):
            self.emit("read_failure",edge_ns=edge)


async def observe(args):
    os.umask(0o077)
    window=json.loads((args.run/"serving/window.json").read_text())
    end=window["end_monotonic_ns"]
    fleet=args.run/"idle"
    path=fleet/"broker-events.jsonl"
    logpath=args.run/"metrics"/"reentry-observer.jsonl"
    with path.open() as feed, logpath.open("x",buffering=1) as log:
        obs=Observer(log,fleet,args.control_dir)
        # Establish the state at observer attachment without replaying past edges.
        for line in feed:
            event=json.loads(line)
            if event["event"]=="state":
                obs.previous=event["serving"]
                obs.held=event["held"]
        def consume():
            while True:
                pos=feed.tell()
                line=feed.readline()
                if not line.endswith("\n"):
                    feed.seek(pos)
                    return
                obs.update(json.loads(line))
        fd=inotify(path,consume)
        for signum in (signal.SIGTERM,signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(signum,obs.done.set)
        obs.emit("attached",end_ns=end,initial_serving=obs.previous,initial_held=obs.held)
        try:
            consume()
            while not obs.done.is_set() and time.monotonic_ns()<end and not (args.run/"metrics/stop").exists():
                try:
                    await asyncio.wait_for(obs.done.wait(),.5)
                except TimeoutError:
                    pass
        finally:
            if obs.task:
                obs.task.cancel()
                await asyncio.gather(obs.task,return_exceptions=True)
            asyncio.get_running_loop().remove_reader(fd)
            os.close(fd)
            obs.emit("detached")
    return logpath


def distribution(items):
    observed=sorted(x for x in items if x is not None)
    if not observed:
        return dict(observations=0)
    return dict(observations=len(observed),median_ms=statistics.median(observed),
                p95_ms=observed[math.ceil(.95*len(observed))-1],max_ms=observed[-1])


def analyze_edge(edge, records, demand):
    hits={}
    eligible=set()
    cooling=set()
    first_observed={}
    late_overwrite=set()
    samples=sorted((r for r in records if r["event"]=="workers" and r["began_ns"]<demand),
                   key=lambda r:r["began_ns"])
    for sample in samples:
        for rank,worker in sample["workers"].items():
            if not worker["watched"] or worker["faulted"]:
                continue
            seen,release=worker["first_gpu_seen_ns"],worker["release_ns"]
            first_observed.setdefault(rank,sample["ended_ns"])
            if edge<=release<=seen<demand:
                if rank in hits and hits[rank][0]!=release:
                    late_overwrite.add(rank)
                if rank not in hits or seen<hits[rank][1]:
                    hits[rank]=(release,seen)
                if release-edge<=10_000_000:
                    eligible.add(rank)
            # Once broker owns zero holds, a worker still held ≥5 ms after the
            # edge with its pre-edge release stamp can be the thermal lease.
            # The independently published cooler state corroborates that fact.
            age=sample["began_ns"]-edge
            temp=sample.get("thermal")
            if (sample["broker_held"]==0 and 5_000_000<=age<=EARLY_NS
                    and release<edge and worker["holds"] and temp
                    and temp["holds"][int(rank)] and abs(sample["ended_ns"]-temp["published_ns"]) < 2_000_000_000):
                cooling.add(rank)
    # A thermal lease can clear during the first 10ms. That GPU is available
    # in the early group; the evidence preserves its held rank separately.
    clocks={rank:dict(release_ms=(rel-edge)/1e6,first_gpu_ms=(seen-edge)/1e6,
                      release_to_native_gpu_ms=(seen-rel)/1e6)
            for rank,(rel,seen) in hits.items()}
    no_class=set(map(str,range(4)))-eligible-cooling
    any_hit=min((r["first_gpu_ms"] for r in clocks.values()),default=None)
    first_eligible=min((clocks[r]["first_gpu_ms"] for r in eligible),default=None)
    all_eligible=max((clocks[r]["first_gpu_ms"] for r in eligible),default=None) if not no_class else None
    return dict(edge_ns=edge,ranks=clocks,polls=len(samples),workers_released_within_10ms=sorted(eligible),
                corroborated_cooling_after_broker_release=sorted(cooling),unclassified_at_edge=sorted(no_class),
                superseded_native_release_seen_for_ranks=sorted(late_overwrite),
                first_status_sample_ms={r:(ts-edge)/1e6 for r,ts in first_observed.items()},
                first_observed_gpu_ms=any_hit,first_early_eligible_gpu_ms=first_eligible,
                all_early_eligible_gpus_ms=all_eligible,
                all_four_observed_gpu_ms=max(r["first_gpu_ms"] for r in clocks.values()) if len(clocks)==4 else None)


def summarize(path,window,include_rows=False):
    records={}
    attached=detached=None
    demand={}
    end=window["end_monotonic_ns"]
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                continue
            event=json.loads(line)
            kind=event["event"]
            if kind=="attached":
                attached=event["observed_ns"]
            elif kind=="detached":
                detached=event["observed_ns"]
            elif window["start_monotonic_ns"]<=event["edge_ns"]<end:
                key=event["edge_ns"]
                if kind=="new_demand":
                    demand[key]=event["end_ns"]
                else:
                    records.setdefault(key,[]).append(event)
    rows=[analyze_edge(edge,records[edge],min(demand.get(edge,end),end)) for edge in sorted(records)]
    result=dict(observer_attached_ns=attached,observer_detached_ns=detached,observed_idle_edges=len(rows),
                early_group_definition="Workers with native release within 10 ms of broker idle. All-early latency requires that every other rank has an observed thermal hold.",
                coverage_note="Native first-GPU observations within sampled idle intervals. All-four times include cooling and job preparation. A native release timestamp can be replaced by another cooldown cycle before the observer reads it.",
                edges_with_corroborated_cooling=sum(bool(r["corroborated_cooling_after_broker_release"]) for r in rows),
                incomplete_all_four=sum(r["all_four_observed_gpu_ms"] is None for r in rows),
                early_unclassified_edges=sum(bool(r["unclassified_at_edge"]) for r in rows))
    for key in ("first_observed_gpu_ms","first_early_eligible_gpu_ms","all_early_eligible_gpus_ms","all_four_observed_gpu_ms"):
        result[key]=distribution([r[key] for r in rows])
    if include_rows:
        result["edges"]=rows
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--control-dir",type=Path)
    parser.add_argument("--report",action="store_true")
    parser.add_argument("--edges",action="store_true")
    args=parser.parse_args()
    if args.report:
        window=json.loads((args.run/"serving/window.json").read_text())
        print(json.dumps(summarize(args.run/"metrics/reentry-observer.jsonl",window,args.edges),indent=2))
    elif args.control_dir:
        print(asyncio.run(observe(args)))
    else:
        parser.error("--control-dir required when attaching the observer")


if __name__=="__main__":
    main()
