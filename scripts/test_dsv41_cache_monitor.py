import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dsv41_cache_monitor as monitor


def raw_metrics(**overrides):
    values = dict(full_used=10000, full_capacity=100000, swa_used=9500, swa_capacity=10000,
                  swa_evicted=100, boundary_evicted=10, retention_enabled=1)
    values.update(overrides)
    return "\n".join(f'sglang:{name}{{tp_rank="0",pp_rank="0",model_name="deepseek-v4-flash"}} {values[key]}'
                     for key, name in monitor.CORE.items())


def sample(at=10, **metrics):
    return {"at": at, "container": {"id": "fixture-id", "started_at": "2026-09-18T23:02:46Z", "restarts": 0, "state": "running"},
            "metrics": monitor.read_metrics(raw_metrics(**metrics))}


def diagnostic_raw(**overrides):
    values = {key: 0 for key in monitor.DIAGNOSTIC}
    values.update(diagnostics_enabled=1, census_complete=1, census_timestamp_seconds=9.25,
                  census_duration_seconds=0.0125, full_tree_tokens=10000,
                  full_swa_covered_tokens=9000, full_swa_uncovered_tokens=1000,
                  match_lookups=40, swa_limited_lookups=3,
                  swa_limited_full_tokens=2048, swa_limited_host_tokens=1024)
    values.update({key: 0 for key in monitor.CAUSE_KEYS})
    values.update({"swa_evicted:swa_pressure:swa_only": 100,
                   "boundary_evicted:swa_pressure:swa_only": 10})
    values.update(overrides)
    lines = [raw_metrics()]
    for key, name in monitor.DIAGNOSTIC.items():
        lines.append(f'sglang:{name}{{tp_rank="0",model_name="deepseek-v4-flash"}} {values[key]}')
    for kind, name in monitor.CAUSE_METRICS.items():
        for cause in monitor.CAUSES:
            for disposition in monitor.DISPOSITIONS:
                key = f"{kind}:{cause}:{disposition}"
                lines.append(f'sglang:{name}{{tp_rank="0",model_name="deepseek-v4-flash",trigger="{cause}",disposition="{disposition}"}} {values[key]}')
    return "\n".join(lines)


class CacheMonitorTest(unittest.TestCase):
    def test_core_missing_or_invalid_never_becomes_zero(self):
        for raw in ("", raw_metrics().split("\n", 1)[1], raw_metrics(full_capacity=0),
                    raw_metrics(swa_used=10001), raw_metrics(boundary_evicted=101),
                    raw_metrics(swa_evicted="nan"), raw_metrics(swa_evicted=-1),
                    raw_metrics(swa_evicted=1.5), raw_metrics(retention_enabled=2),
                    raw_metrics() + "\n" + raw_metrics()):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                monitor.read_metrics(raw)

    def test_rank_zero_and_model_selection(self):
        raw = raw_metrics() + "\n" + raw_metrics().replace('tp_rank="0"', 'tp_rank="1"')
        raw += "\n" + raw_metrics().replace('model_name="deepseek-v4-flash"', 'model_name="other"')
        self.assertEqual(monitor.read_metrics(raw)["swa_used"], 9500)

    def test_absent_optional_series_unknown_not_zero(self):
        result = monitor.read_metrics(raw_metrics())
        self.assertIsNone(result["host_hit_tokens"])
        self.assertIsNone(result["restore_full_tokens_all_ranks"])
        self.assertIsNone(result["full_swa_uncovered_tokens"])
        self.assertIsNone(result["boundary_evicted:swa_pressure:swa_only"])

    def test_diagnostic_census_and_disjoint_causes_parse_without_tp_multiplication(self):
        raw = diagnostic_raw()
        result = monitor.read_metrics(raw + "\n" + raw.replace('tp_rank="0"', 'tp_rank="1"'))
        self.assertEqual(result["swa_evicted:swa_pressure:swa_only"], 100)
        self.assertEqual(result["census_duration_seconds"], 0.0125)
        self.assertEqual(result["full_swa_uncovered_tokens"], 1000)

    def test_enabled_diagnostics_require_complete_consistent_schema(self):
        for raw in (
            diagnostic_raw(**{"swa_evicted:swa_pressure:swa_only": 99}),
            diagnostic_raw(**{"boundary_evicted:swa_pressure:with_full": 1}),
            diagnostic_raw(full_swa_covered_tokens=8999),
            diagnostic_raw(census_complete=2),
            diagnostic_raw(census_timestamp_seconds=0),
            diagnostic_raw(diagnostics_enabled=2),
            diagnostic_raw(census_duration_seconds="nan"),
            diagnostic_raw().replace('sglang:hicache_swa_match_lookups{', 'sglang:not_exported{'),
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                monitor.read_metrics(raw)

    def test_disabled_diagnostics_are_not_zero_loss_measurements(self):
        result = monitor.read_metrics(diagnostic_raw(diagnostics_enabled=0))
        self.assertIsNone(result["full_swa_uncovered_tokens"])
        self.assertIsNone(result["swa_evicted:swa_pressure:swa_only"])

    def test_census_fraction_uses_same_snapshot_tree_denominator(self):
        current = sample()
        current["metrics"] = monitor.read_metrics(diagnostic_raw())
        result = monitor.analyze(current, None)
        self.assertEqual(result["full_swa_uncovered_fraction_of_tree_host_tokens"], 0.1)
        self.assertEqual(result["census_age_seconds"], 0.75)
        self.assertEqual(result["classification"], "attributed_evictions_and_resumable_prefix_census")

    def test_stale_or_incomplete_census_is_not_current_zero_loss(self):
        for at, complete in ((161, 1), (10, 0), (8, 1)):
            current = sample(at)
            current["metrics"] = monitor.read_metrics(diagnostic_raw(census_complete=complete))
            result = monitor.analyze(current, None)
            self.assertIsNone(result["full_swa_uncovered_fraction_of_tree_host_tokens"])
            self.assertIn("census_unavailable_or_stale", result["flags"])

    def test_attributed_eviction_deltas_distinguish_trigger_and_full_removal(self):
        previous, current = sample(), sample(40)
        previous["metrics"] = monitor.read_metrics(diagnostic_raw())
        current["metrics"] = monitor.read_metrics(diagnostic_raw())
        # Both events happened in one interval, so both flags must survive.
        current["metrics"]["boundary_evicted:swa_pressure:swa_only"] += 256
        current["metrics"]["boundary_evicted:full_pressure:with_full"] += 512
        result = monitor.analyze(current, previous)
        self.assertIn("boundary_swa_pressure_without_full_removal", result["flags"])
        self.assertIn("boundary_eviction_alongside_full", result["flags"])

    def test_transfer_pool_counters_keep_explicit_all_rank_scope(self):
        raw = raw_metrics() + '\nsglang:load_back_tokens_total{pool="kv"} 1024\nsglang:load_back_tokens_total{pool="swa"} 256'
        result = monitor.read_metrics(raw)
        self.assertEqual(result["restore_full_tokens_all_ranks"], 1024)
        self.assertEqual(result["restore_swa_tokens_all_ranks"], 256)

    def test_boundary_delta_with_spare_full_flagged_as_candidate(self):
        previous = sample(swa_evicted=100, boundary_evicted=10)
        current = sample(40, swa_evicted=612, boundary_evicted=266)
        result = monitor.analyze(current, previous)
        self.assertIn("boundary_eviction_with_full_headroom", result["flags"])
        self.assertEqual(result["ordinary_swa_slots_evicted_delta"], 256)
        self.assertEqual(result["interval_seconds"], 30)

    def test_ordinary_churn_not_claimed_as_boundary_loss(self):
        result = monitor.analyze(sample(40, swa_evicted=612), sample())
        self.assertEqual(result["ordinary_swa_slots_evicted_delta"], 512)
        self.assertEqual(result["flags"], ["swa_pressure_with_full_headroom"])

    def test_full_pressure_does_not_claim_spare_capacity(self):
        result = monitor.analyze(sample(40, full_used=99000, swa_evicted=200, boundary_evicted=100), sample())
        self.assertEqual(result["flags"], ["boundary_eviction_observed"])

    def test_new_container_and_counter_resets_do_not_create_evictions(self):
        previous = sample()
        current = sample(40, swa_evicted=0, boundary_evicted=0)
        result = monitor.analyze(current, previous)
        self.assertIn("counter_reset", result["flags"])
        self.assertIsNone(result["deltas"]["boundary_evicted"])
        current["container"]["restarts"] = 1
        result = monitor.analyze(current, previous)
        self.assertNotIn("counter_reset", result["flags"])
        self.assertIsNone(result["deltas"]["boundary_evicted"])

    def test_fresh_baseline_does_not_attribute_prior_evictions(self):
        result = monitor.analyze(sample(), None)
        self.assertIsNone(result["deltas"]["boundary_evicted"])
        self.assertNotIn("boundary_eviction_with_full_headroom", result["flags"])

    def test_record_resume_uses_atomic_summary_and_rotates_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = monitor.record(root, sample(), None)
            second = monitor.record(root, sample(40, swa_evicted=356, boundary_evicted=266), first)
            # Simulate collector crash before its global latest.json was saved.
            third = monitor.record(root, sample(70, swa_evicted=356, boundary_evicted=266), first)
            summary = json.loads((root / "current/summary.json").read_text())
            self.assertEqual(summary["observed_boundary_slots_evicted"], 256)
            self.assertEqual(summary["samples"], 3)
            self.assertEqual(third["analysis"]["deltas"]["boundary_evicted"], 0)
            changed = copy.deepcopy(sample(100, swa_evicted=0, boundary_evicted=0))
            changed["container"]["started_at"] = "2026-09-19T01:00:00Z"
            monitor.record(root, changed, third)
            summary = json.loads((root / "current/summary.json").read_text())
            self.assertEqual(summary["observed_boundary_slots_evicted"], 0)
            self.assertEqual(len(list(root.glob("launch-*"))), 2)

    def test_sampling_uses_only_read_only_endpoints(self):
        snapshot = sample()["container"]
        loads = [{"num_running_reqs": 0, "num_waiting_reqs": 0, "num_active_tokens": 0}]
        with patch.object(monitor, "container", return_value=snapshot), \
             patch.object(monitor, "parse_loads", return_value=loads), \
             patch.object(monitor, "activity_metrics", return_value=(0, ())), \
             patch.object(monitor, "mem_available", return_value=1234), \
             patch.object(monitor, "fetch", side_effect=["{}", raw_metrics()]) as fetch:
            monitor.sample()
        self.assertEqual([call.args[0] for call in fetch.call_args_list], ["/v1/loads", "/metrics"])

    def test_container_change_mid_sample_rejected(self):
        before = sample()["container"]
        after = dict(before, restarts=1)
        with patch.object(monitor, "container", side_effect=[before, after]), \
             patch.object(monitor, "parse_loads", return_value=[]), \
             patch.object(monitor, "activity_metrics", return_value=(0, ())), \
             patch.object(monitor, "fetch", side_effect=["{}", raw_metrics()]):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                monitor.sample()


if __name__ == "__main__":
    unittest.main()
