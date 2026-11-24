import unittest

from native_mode_report import embedded_points


class MatchingIdleWindows(unittest.TestCase):
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
