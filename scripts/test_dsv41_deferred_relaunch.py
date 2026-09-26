"""Offline rollout safety checks; no Docker mutation or inference traffic."""

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsv41_deferred_relaunch as rollout


def rows(stamp, **overrides):
    return [
        {
            "dp_rank": 0,
            "timestamp": stamp,
            "num_reqs": 0,
            "num_running_reqs": 0,
            "num_waiting_reqs": 0,
            "num_used_tokens": 0,
            "num_total_tokens": 0,
            "num_active_tokens": 0,
            "num_waiting_uncached_tokens": 0,
            "queues": {k: 0 for k in ("waiting", "grammar", "paused", "retracted", "prealloc_ready")},
            **overrides,
        }
    ]


def metrics(total=50, active=0):
    return (
        f'sglang:http_requests_active{{endpoint="/v1/messages",method="POST"}} {active}\n'
        f'sglang:http_requests_total{{endpoint="/v1/messages",method="POST"}} {total}\n'
    )


class IdleWindowTest(unittest.TestCase):
    def test_full_continuous_minute(self):
        window = rollout.QuietWindow()
        for seconds in range(0, 60, 5):
            self.assertFalse(window.observe(rows(seconds), metrics(), seconds))
        self.assertTrue(window.observe(rows(60), metrics(), 60))

    def test_chunked_prefill_without_running_requests_is_busy(self):
        for counter in ("num_active_tokens", "num_total_tokens", "num_used_tokens", "num_waiting_uncached_tokens"):
            window = rollout.QuietWindow()
            window.observe(rows(0), metrics(), 0)
            self.assertFalse(window.observe(rows(60, **{counter: 1000}), metrics(), 60))
            self.assertFalse(window.observe(rows(65), metrics(), 65))
            self.assertTrue(window.observe(rows(125), metrics(), 125))

    def test_brief_request_between_polls_restarts_clock(self):
        window = rollout.QuietWindow()
        window.observe(rows(0), metrics(), 0)
        self.assertFalse(window.observe(rows(60), metrics(total=51), 60))
        self.assertFalse(window.observe(rows(65), metrics(total=51), 65))
        self.assertFalse(window.observe(rows(120), metrics(total=51), 120))
        self.assertTrue(window.observe(rows(125), metrics(total=51), 125))

    def test_http_stream_or_recovery_counts_as_busy(self):
        window = rollout.QuietWindow()
        window.observe(rows(0), metrics(), 0)
        self.assertFalse(window.observe(rows(60), metrics(active=1), 60))

    def test_stale_or_missing_ranks_cannot_trigger(self):
        window = rollout.QuietWindow()
        window.observe(rows(100), metrics(), 0)
        with self.assertRaises(ValueError):
            window.observe(rows(100), metrics(), 60)
        with self.assertRaises(ValueError):
            window.observe(rows(110, dp_rank=1), metrics(), 120)

    def test_bad_metrics_fail_closed(self):
        for text in ("", metrics(active="NaN"), metrics(total=-1), metrics() + metrics()):
            with self.assertRaises(ValueError):
                rollout.activity_metrics(text)

    def test_probe_requests_do_not_count_as_inference(self):
        extra = 'sglang:http_requests_active{endpoint="/metrics",method="GET"} 1\n'
        self.assertEqual(rollout.activity_metrics(metrics() + extra)[0], 0)

    def test_errors_reset_previous_idle_time(self):
        window = rollout.QuietWindow()
        window.observe(rows(0), metrics(), 0)
        window.reset()
        self.assertFalse(window.observe(rows(100), metrics(), 100))
        self.assertTrue(window.observe(rows(160), metrics(), 160))


class DeploymentGuardsTest(unittest.TestCase):
    def test_budget_expansion_preserves_all_other_configuration(self):
        before = {
            "environment": {"DSV41_HICACHE_HOST_BUDGET_BYTES": "192000000000", "MEMORY_FRACTION": "0.87"},
            "volumes": [],
            "restart": "unless-stopped",
        }
        candidate = json.loads(json.dumps(before))
        candidate["environment"]["DSV41_HICACHE_HOST_BUDGET_BYTES"] = "256000000000"
        self.assertEqual(rollout.validate_cache_only_change(before, candidate), 192_000_000_000)
        candidate["environment"]["MEMORY_FRACTION"] = "0.9"
        with self.assertRaises(ValueError):
            rollout.validate_cache_only_change(before, candidate)

    def test_ram_preflight_credits_only_actual_old_buffers(self):
        available, allocated = 103_495_000_000, 183_075_523_072
        spare = rollout.projected_spare_ram(available, allocated)
        self.assertEqual(spare, available + allocated - 256_000_000_000)
        self.assertGreater(spare, rollout.SPARE_RAM_BYTES)
        self.assertLess(rollout.projected_spare_ram(available - 8 * 1024**3, allocated), rollout.SPARE_RAM_BYTES)
        self.assertLess(rollout.projected_spare_ram(available, 0), 0)

    def test_rendered_compose_is_valid_input_with_zero_core_limit(self):
        original = json.loads(rollout.run("docker", "compose", "-f", str(rollout.CONFIG), "config", "--format", "json"))
        fixed = rollout.roundtrip_profile(original)
        self.assertEqual(fixed["services"]["deepseek"]["ulimits"]["core"], {"soft": 0, "hard": 0})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compose.json"
            path.write_text(json.dumps(fixed))
            rollout.run("docker", "compose", "-f", str(path), "config", "--quiet")

    def test_watch_does_not_mutate_on_busy_or_unreadable_load(self):
        for bad_read in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                (state / "manifest.json").write_text(json.dumps({"idle_seconds": 60, "poll_seconds": 5}))
                busy = rows(time.time(), num_active_tokens=500000)[0]

                def fetch(path):
                    if bad_read:
                        raise OSError("unavailable")
                    return json.dumps({"loads": [busy]}) if path == "/v1/loads" else metrics(active=1)

                with (
                    patch.object(rollout, "fetch", side_effect=fetch),
                    patch.object(rollout, "check_files"),
                    patch.object(rollout, "check_identity"),
                    patch.object(rollout, "notify", return_value=True),
                    patch.object(rollout.time, "sleep", side_effect=RuntimeError("end simulation")),
                    patch.object(rollout.subprocess, "run") as mutate,
                ):
                    with self.assertRaisesRegex(RuntimeError, "end simulation"):
                        rollout.watch(state)
                    mutate.assert_not_called()
                    self.assertFalse((state / "attempt.json").exists())

    def test_container_restart_cancels_watcher(self):
        before = {
            "Id": "first",
            "Image": "pinned",
            "State": {"StartedAt": "then", "Status": "running"},
            "RestartCount": 0,
        }
        manifest = {"expected_identity": list(rollout.identity(before))}
        with patch.object(rollout, "inspect", return_value=before):
            rollout.check_identity(manifest)
            before["RestartCount"] = 1
            with self.assertRaises(RuntimeError):
                rollout.check_identity(manifest)

    def test_changed_source_prevents_relaunch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.py"
            source.write_text("original")
            manifest = {
                "file_hashes": {str(source): rollout.digest(source)},
                "image_tag": "tag",
                "expected_identity": [0, 0, 0, "id"],
            }
            with patch.object(rollout, "run", return_value="id"):
                rollout.check_files(manifest)
                source.write_text("changed")
                with self.assertRaises(RuntimeError):
                    rollout.check_files(manifest)

    def log(self, ranks=range(4), size=40_000_000_000, budget=rollout.BUDGET):
        lines = []
        for rank in ranks:
            lines += [
                "DSV41_HICACHE_BUDGET "
                + json.dumps(
                    {
                        "rank": rank,
                        "host_budget_bytes": budget,
                        "planned_bytes_host_total": 190_000_000_000,
                        "full_host_pages": 10,
                        "swa_host_pages": 1,
                        "tensor_bytes_per_rank": size,
                    }
                ),
                "DSV41_HICACHE_ALLOCATED "
                + json.dumps({"rank": rank, "bytes_per_rank": size, "host_budget_bytes": budget}),
            ]
        return "\n".join(lines)

    def test_four_rank_actual_allocation_required(self):
        self.assertEqual(rollout.verify_budget_logs(self.log())["actual_buffer_bytes_total"], 160_000_000_000)
        for log in (self.log(range(3)), self.log(size=70_000_000_000), self.log() + "\n" + self.log([0])):
            with self.assertRaises(ValueError):
                rollout.verify_budget_logs(log)

    def test_previous_192_budget_verification_is_explicit(self):
        old = self.log(budget=192_000_000_000)
        self.assertEqual(rollout.verify_budget_logs(old, 192_000_000_000)["actual_buffer_bytes_total"], 160_000_000_000)
        with self.assertRaises(ValueError):
            rollout.verify_budget_logs(old)

    def test_embedded_request_text_cannot_impersonate_allocation_log(self):
        fake = "[2026-09-18 02:11:00] " + json.dumps({"event": "request.received", "text": self.log()})
        valid = "\n".join("[2026-09-18 02:11:04 TP0 EP0] " + line for line in self.log().splitlines())
        self.assertEqual(rollout.verify_budget_logs(valid + "\n" + fake)["actual_buffer_bytes_total"], 160_000_000_000)
        with self.assertRaises(ValueError):
            rollout.verify_budget_logs(fake)

    def test_attempt_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "manifest.json").write_text("{}")
            (state / "attempt.json").write_text("{}")
            with patch.object(rollout, "run") as command:
                with self.assertRaises(RuntimeError):
                    rollout.watch(state)
                command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
