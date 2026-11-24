"""Regression checks for independently released or temperature-held mining workers."""
from pathlib import Path
import tempfile
import unittest
import json

from hour_metrics import handoffs
from observe_reentry import analyze_edge


def worker(release, seen, holds=0):
    return dict(watched=True,faulted=False,holds=holds,release_ns=release,first_gpu_seen_ns=seen,completed_slices=5)


class EdgeAccounting(unittest.TestCase):
    def test_native_release_overwrite_and_thermal_attribution(self):
        ms=1_000_000
        e=10_000*ms
        def snap(t,gpus,cool):
            return dict(event="workers",began_ns=e+t*ms,ended_ns=e+t*ms+5000,broker_held=0,
                        workers={str(i):w for i,w in enumerate(gpus)},
                        thermal=dict(published_ns=e+t*ms-25000,holds=cool))
        early=worker(e+100_000,e+600_000)
        hot=worker(e-100*ms,e-99*ms,1)
        slow=worker(e+400*ms,e+401*ms)
        replacement=worker(e+390*ms,e+391*ms)
        rows=[snap(9,[early,hot,early,early],[False,True,False,False]),
              snap(450,[replacement,slow,early,early],[False]*4)]
        result=analyze_edge(e,rows,e+5_000*ms)
        self.assertEqual(result["workers_released_within_10ms"],["0","2","3"])
        self.assertEqual(result["corroborated_cooling_after_broker_release"],["1"])
        self.assertEqual(result["superseded_native_release_seen_for_ranks"],["0"])
        self.assertEqual(result["first_early_eligible_gpu_ms"],.6)
        self.assertEqual(result["all_early_eligible_gpus_ms"],.6)
        self.assertEqual(result["all_four_observed_gpu_ms"],401.)

    def test_ambiguous_rank_does_not_become_known_eligible(self):
        row=dict(event="workers",began_ns=10_000_000,ended_ns=10_001_000,broker_held=0,
                 workers={"1":worker(1_100_000,1_300_000)},thermal=None)
        a=analyze_edge(1_000_000,[row],19_000_000)
        self.assertEqual(a["first_early_eligible_gpu_ms"],.3)
        self.assertIsNone(a["all_early_eligible_gpus_ms"])
        self.assertIsNone(a["all_four_observed_gpu_ms"])

    def test_coarse_stamps_can_come_from_independent_samples(self):
        e=10_000_000
        t=[dict(event="state",monotonic_ns=0,watched=4,held=4,serving=True),
           dict(event="state",monotonic_ns=e,watched=4,held=0,serving=False)]
        blank=worker(0,0,1)
        early=worker(e+100_000,e+500_000)
        late=worker(e+5_000_000,e+6_000_000)
        s=[{"idle":{"workers":[early,blank,early,early]}},{"idle":{"workers":[late,early,early,early]}}]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"brok"
            path.write_text("".join(json.dumps(event)+"\n" for event in t))
            result=handoffs(path,5_000_000,30_000_000,s)
        self.assertEqual(result["idle_to_all_gpu_reentry_ms"]["median_ms"],.5)


if __name__=="__main__":
    unittest.main()
