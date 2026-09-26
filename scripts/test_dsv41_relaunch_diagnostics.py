"""Offline diagnostics rollout guards; no model traffic or Docker mutation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dsv41_relaunch_diagnostics as rollout
from test_dsv41_deferred_relaunch import metrics, rows


class DiagnosticRolloutTest(unittest.TestCase):
    def profiles(self):
        old = {"services": {"deepseek": {
            "container_name": "dsv41", "restart": "unless-stopped", "image": "parent",
            "labels": {"dev.idle.deployment.version": "previous"},
            "environment": {"MEMORY_FRACTION": "0.87"},
        }}}
        new = copy.deepcopy(old)
        service = new["services"]["deepseek"]
        service["image"] = "candidate"
        service["labels"]["dev.idle.deployment.version"] = rollout.VERSION
        service["environment"].update(rollout.DIAGNOSTICS)
        return old, new

    def test_only_diagnostics_allowed_and_inputs_preserved(self):
        old, new = self.profiles()
        original = copy.deepcopy((old, new))
        rollout.validate_only_diagnostics(old, new)
        self.assertEqual((old, new), original)

    def test_serving_or_cache_changes_rejected(self):
        for key in ("MEMORY_FRACTION", "CONTEXT_LENGTH", "DSV41_HICACHE_HOST_BUDGET_BYTES", "NCCL_ALGO"):
            old, new = self.profiles()
            new["services"]["deepseek"]["environment"][key] = "changed"
            with self.assertRaisesRegex(ValueError, "Non-diagnostic"):
                rollout.validate_only_diagnostics(old, new)

    def test_missing_diagnostic_rejected(self):
        old, new = self.profiles()
        del new["services"]["deepseek"]["environment"]["TORCH_NCCL_EXTRA_DUMP_ON_EXEC"]
        with self.assertRaises(ValueError):
            rollout.validate_only_diagnostics(old, new)

    def test_busy_or_invalid_metrics_never_recreate(self):
        for invalid in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                (state / "manifest.json").write_text(json.dumps({"idle_seconds": 60, "poll_seconds": 5}))
                with (
                    patch.object(rollout, "check_identity"), patch.object(rollout, "check_files"),
                    patch.object(rollout, "parse_loads", return_value=rows(1, num_active_tokens=500000)),
                    patch.object(rollout, "fetch", side_effect=["{}", "bad" if invalid else metrics(active=1)]),
                    patch.object(rollout.time, "sleep", side_effect=RuntimeError("end simulation")),
                    patch.object(rollout.subprocess, "run") as mutate,
                ):
                    with self.assertRaisesRegex(RuntimeError, "end simulation"):
                        rollout.watch(state)
                    mutate.assert_not_called()
                    self.assertFalse((state / "attempt.json").exists())

    def test_no_repeat_after_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "manifest.json").write_text("{}")
            (state / "attempt.json").touch()
            with self.assertRaisesRegex(RuntimeError, "no automatic repeat"):
                rollout.watch(state)


if __name__ == "__main__":
    unittest.main()
