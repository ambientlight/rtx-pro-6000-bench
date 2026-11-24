"""Check continuously-free worker rates with resets, thermal releases and sampling gaps."""
import json
from pathlib import Path
import tempfile
import unittest
import idle_opportunities as m

SEC=1_000_000_000
CAL={"total_tmac_s":400.,"tmac_s_by_gpu":[100.]*4}


def snapshot(t, counts):
    return dict(end_ns=t*SEC+1,idle=dict(source_monotonic_ns=t*SEC,
        broker={"serving":False},
        workers=[dict(watched=True,faulted=False,holds=0,release_ns=1,completed_slices=c) for c in counts],
        ranks=[dict(cooling=False,cooling_seconds=0.,connected=True) for _ in m.RANKS]))


def measure(samples,pool_events=(),broker_events=()):
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)
        for file,events in (("events.jsonl",pool_events),("broker-events.jsonl",broker_events)):
            (path/file).write_text("".join(json.dumps(x)+"\n" for x in events))
        return m.measure(samples,1*SEC,30*SEC,path,CAL)


class IdleIntervals(unittest.TestCase):
    def test_duration_weighting_zero_progress_and_per_card_tile_bounds(self):
        rows=[snapshot(t,[n]*4) for t,n in ((1,10),(3,149),(7,149),(11,150))]
        out=measure(rows)
        gpu=out["by_gpu"][0]
        self.assertEqual((gpu["qualifying_intervals"],gpu["included_s"],gpu["intervals_with_zero_completed_slices"]),(3,10,1))
        # All 139 slices in one interval allow only one short tail, plus one
        # possible short tail in the final single-slice interval.
        self.assertEqual(gpu["tmac_s_bounds"],[(140*188-40)*m.TILE_MACS/10/1e12,140*188*m.TILE_MACS/10/1e12])
        self.assertEqual(out["all_four_simultaneously_free"]["tmac_s_bounds"],[x*4 for x in gpu["tmac_s_bounds"]])

    def test_excludes_release_cooling_disconnect_reset_stale_and_large_gaps(self):
        for change in ("unwatched","held","faulted","not_connected","cooled","cursor_reset","new_release","stale","gap"):
            with self.subTest(change=change):
                a,b=snapshot(1,[9]*4),snapshot(3,[12]*4)
                worker=b["idle"]["workers"][0];r=b["idle"]["ranks"][0]
                if change=="unwatched": worker["watched"]=False
                if change=="held": worker["holds"]=1
                if change=="faulted": worker["faulted"]=True
                if change=="not_connected": r["connected"]=False
                if change=="cooled": r["cooling_seconds"]=.01
                if change=="cursor_reset": worker["completed_slices"]=2
                if change=="new_release": worker["release_ns"]=2*SEC
                if change=="stale": b["end_ns"]+=3*SEC
                if change=="gap": b=snapshot(9,[12]*4)
                res=measure([a,b])
                self.assertEqual(res["by_gpu"][0]["qualifying_intervals"],0)
                self.assertEqual(res["all_four_simultaneously_free"]["qualifying_intervals"],0)
                self.assertNotIn("tmac_s_bounds",res["by_gpu"][1])

    def test_live_session_and_broker_timeline_reconcile(self):
        rows=[snapshot(t,[t]*4) for t in (1,3,5,7)]
        resets=[dict(rank=1,event="connection_end",monotonic_ns=2*SEC),dict(rank=1,event="authorized",monotonic_ns=2*SEC+10)]
        serving=[dict(event="state",serving=True,monotonic_ns=4*SEC),
                 dict(event="state",serving=False,monotonic_ns=4*SEC+1)]
        result=measure(rows,resets,serving)
        self.assertEqual(result["by_gpu"][0]["qualifying_intervals"],2)
        self.assertEqual(result["by_gpu"][1]["qualifying_intervals"],1)
        self.assertEqual(result["all_four_simultaneously_free"]["qualifying_intervals"],1)
        self.assertNotIn("tmac_s_bounds",result["all_four_simultaneously_free"])


if __name__=="__main__":
    unittest.main()
