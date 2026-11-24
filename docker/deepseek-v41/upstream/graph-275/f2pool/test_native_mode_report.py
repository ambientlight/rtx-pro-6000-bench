import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from native_mode_report import embedded_points
from native_report import configured_slice_tiles, idle_rate


class MatchingIdleWindows(unittest.TestCase):
    def test_large_slices_warm_after_nine_graphs(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            self.assertEqual(configured_slice_tiles(config), 188)
            config.write_text('{"tiles": 3008}')
            tiles = configured_slice_tiles(config)
        broker = [dict(event="native_release", epoch=1, monotonic_ns=0)]
        records = {i: [dict(event="native_work", epoch=1, graphs=8 if t < 3 else 9,
                            tiles=t*3008, time=t, monotonic_ns=t*10**9) for t in range(1, 7)]
                   for i in range(4)}
        control = [[dict(elapsed=0), dict(elapsed=2)] for _ in range(4)]
        epoch, points = embedded_points(records, broker, control, 0, 0, slice_tiles=tiles)
        self.assertEqual(epoch, 1)
        self.assertEqual([[p["time"] for p in rank] for rank in points], [[3, 5]] * 4)
        self.assertEqual(idle_rate(records, broker, 6*10**9, slice_tiles=tiles)["common_seconds"], 3)

    def test_ignores_closed_epochs_and_partially_warmed_ranks(self):
        broker = [dict(event="native_release", epoch=1, monotonic_ns=10**9),
                  dict(event="native_demand", epoch=2, monotonic_ns=4*10**9),
                  dict(event="native_release", epoch=3, monotonic_ns=6*10**9)]
        records = {i:[dict(event="native_work", graphs=139 if second>=i else 0,
                           epoch=1 if second<6 else 3, tiles=100*second,
                           time=second, monotonic_ns=second*10**9) for second in range(1,14)]
                   for i in range(4)}
        control = [[dict(elapsed=t) for t in (42,44,46)] for _ in range(4)]
        epoch, samples = embedded_points(records,broker,control,last_request=0,warm=2)
        self.assertEqual(epoch,3)
        for rank in samples:
            self.assertEqual([p["time"] for p in rank],[8,10,12])
        with self.assertRaises(ValueError):
            embedded_points(records,broker,control,last_request=0,warm=6)


if __name__=="__main__":
    unittest.main()
