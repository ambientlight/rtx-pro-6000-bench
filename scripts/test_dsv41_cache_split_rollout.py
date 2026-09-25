"""Offline guards for the RAM-only payload-split rollout."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dsv41_relaunch_cache_split as rollout
from test_dsv41_deferred_relaunch import metrics, rows


class CacheSplitRolloutTest(unittest.TestCase):
    def profiles(self):
        old = {"services": {"deepseek": {
            "container_name": "dsv41", "restart": "unless-stopped", "image": "same-image",
            "labels": {"dev.idle.deployment.version": "old"},
            "environment": {"MEMORY_FRACTION": "0.87", "DSV41_HICACHE_HOST_BUDGET_BYTES": "256000000000"},
        }}}
        new = copy.deepcopy(old)
        new["services"]["deepseek"]["environment"][rollout.FRACTION_KEY] = rollout.FRACTION
        new["services"]["deepseek"]["labels"]["dev.idle.deployment.version"] = rollout.VERSION
        return old, new

    def test_only_fraction_and_label_change(self):
        old, new = self.profiles()
        original = copy.deepcopy((old, new))
        rollout.validate_change(old, new)
        self.assertEqual((old, new), original)

    def test_rebalance_existing_236_split_to_50(self):
        old, new = self.profiles()
        old["services"]["deepseek"]["environment"][rollout.FRACTION_KEY] = "0.236"
        old["services"]["deepseek"]["labels"]["dev.idle.deployment.version"] = "dsv41-v2-hicache-split-v1"
        rollout.validate_change(old, new)
        self.assertEqual(new["services"]["deepseek"]["environment"][rollout.FRACTION_KEY], "0.5")

    def test_stale_236_candidate_is_rejected(self):
        old, new = self.profiles()
        new["services"]["deepseek"]["environment"][rollout.FRACTION_KEY] = "0.236"
        with self.assertRaisesRegex(ValueError, "50%"):
            rollout.validate_change(old, new)

    def test_rejects_image_inference_diagnostic_and_budget_drift(self):
        for key in ("MEMORY_FRACTION", "CONTEXT_LENGTH", "NCCL_DEBUG", "DSV41_HICACHE_HOST_BUDGET_BYTES", rollout.FRACTION_KEY):
            old, new = self.profiles()
            new["services"]["deepseek"]["environment"][key] = "changed"
            with self.assertRaises(ValueError):
                rollout.validate_change(old, new)
        old, new = self.profiles()
        new["services"]["deepseek"]["image"] = "different-image"
        with self.assertRaises(ValueError):
            rollout.validate_change(old, new)

    def test_source_snapshots_preserve_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("before")
            profile = {"services": {"deepseek": {"volumes": [
                {"type": "bind", "source": str(source), "target": "/opt/source.py", "read_only": True},
                {"type": "volume", "source": "kernel-cache", "target": "/root/.cache"},
            ]}}}
            frozen, hashes = rollout.freeze_sources(profile, root / "snapshot")
            source.write_text("after")
            target = Path(frozen["services"]["deepseek"]["volumes"][0]["source"])
            self.assertEqual(target.read_text(), "before")
            self.assertEqual(rollout.digest(target), hashes["/opt/source.py"])
            self.assertEqual(profile["services"]["deepseek"]["volumes"][0]["source"], str(source))

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

    def test_fraction_verification_rejects_old_mode_and_wrong_split(self):
        plan = dict(allocation_mode="payload_fraction", requested_swa_payload_fraction=0.5,
                    full_host_pages=50000, swa_host_pages=50000, full_page_bytes=100, swa_page_bytes=100,
                    device_full_pages=10, device_swa_pages=10)
        self.assertEqual(rollout.verify_fraction({"plans": {0: plan}}), 0.5)
        for overrides in ({"allocation_mode": "device_ratio"}, {"swa_host_pages": 10000},
                          {"swa_host_pages": 50100}, {"requested_swa_payload_fraction": 0.236},
                          {"device_swa_pages": 60000}):
            with self.assertRaises(ValueError):
                rollout.verify_fraction({"plans": {0: dict(plan, **overrides)}})

    def test_fifty_fifty_page_rounded_production_capacities(self):
        plan = dict(allocation_mode="payload_fraction", requested_swa_payload_fraction=0.5,
                    full_host_pages=73515, swa_host_pages=4770,
                    full_page_bytes=417920, swa_page_bytes=6439680,
                    device_full_pages=12545, device_swa_pages=102)
        share = rollout.verify_fraction({"plans": {rank: plan for rank in range(4)}})
        self.assertAlmostEqual(share, 0.5, delta=0.0001)
        self.assertLessEqual(share, 0.5)
        self.assertEqual(plan["full_host_pages"] * 256, 18819840)
        self.assertEqual(plan["swa_host_pages"] * 256, 1221120)

    def l15_profiles(self):
        old, new = self.profiles()
        for profile in (old, new):
            service = profile["services"]["deepseek"]
            service["image"] = rollout.L13_IMAGE
            service["labels"]["dev.idle.deployment.version"] = rollout.PROFILES["l15-67"][1]
        old["services"]["deepseek"]["environment"][rollout.FRACTION_KEY] = "0.6"
        new["services"]["deepseek"]["environment"][rollout.FRACTION_KEY] = "0.67"
        return old, new

    def test_l15_explicit_profile_accepts_only_fraction_change(self):
        old, new = self.l15_profiles()
        original = copy.deepcopy((old, new))
        rollout.validate_change(old, new, "l15-67")
        self.assertEqual((old, new), original)
        with self.assertRaisesRegex(ValueError, "50%"):
            rollout.validate_change(old, new)

    def test_l15_rejects_other_profiles_or_non_l14_origin(self):
        for value in ("0.5", "0.67", "0.236", None):
            old, new = self.l15_profiles()
            old["services"]["deepseek"]["environment"][rollout.FRACTION_KEY] = value
            with self.assertRaisesRegex(ValueError, "existing L14"):
                rollout.validate_change(old, new, "l15-67")
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            rollout.profile_settings("arbitrary")

    def test_l15_rejects_budget_inference_image_and_diagnostic_drift(self):
        for field in ("MEMORY_FRACTION", "CONTEXT_LENGTH", "NCCL_DEBUG", "DSV41_HICACHE_HOST_BUDGET_BYTES", rollout.FRACTION_KEY):
            old, new = self.l15_profiles()
            new["services"]["deepseek"]["environment"][field] = "changed"
            with self.assertRaises(ValueError):
                rollout.validate_change(old, new, "l15-67")
        old, new = self.l15_profiles()
        new["services"]["deepseek"]["image"] = "other-image"
        with self.assertRaises(ValueError):
            rollout.validate_change(old, new, "l15-67")

    def test_l15_page_rounded_fraction_matches_l13(self):
        plan = dict(allocation_mode="payload_fraction", requested_swa_payload_fraction=0.67,
                    full_host_pages=49103, swa_host_pages=6469,
                    full_page_bytes=417920, swa_page_bytes=6439680,
                    device_full_pages=15870, device_swa_pages=102)
        share = rollout.verify_fraction({"plans": {rank: plan for rank in range(4)}}, "l15-67")
        self.assertAlmostEqual(share, 0.6699691443610555)
        self.assertEqual(plan["full_host_pages"] * 256, 12570368)
        self.assertEqual(plan["swa_host_pages"] * 256, 1656064)
        with self.assertRaises(ValueError):
            rollout.verify_fraction({"plans": {0: plan}})

    def test_l13_reference_checks_image_flags_and_frozen_runtime_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one, two = root / "one.py", root / "two.py"
            one.write_text("same"); two.write_text("same")
            _, candidate = self.l15_profiles()
            candidate["services"]["deepseek"]["volumes"] = [
                {"type": "bind", "source": str(one), "target": "/opt/boot.py", "read_only": True}]
            reference = copy.deepcopy(candidate)
            reference["services"]["deepseek"]["volumes"][0]["source"] = str(two)
            original = copy.deepcopy((candidate, reference))
            rollout.validate_l13_reference(candidate, reference)
            self.assertEqual((candidate, reference), original)
            two.write_text("different")
            with self.assertRaisesRegex(ValueError, "runtime bytes"):
                rollout.validate_l13_reference(candidate, reference)
            two.write_text("same")
            candidate["services"]["deepseek"]["image"] = "changed"
            with self.assertRaisesRegex(ValueError, "immutable L13"):
                rollout.validate_l13_reference(candidate, reference)


if __name__ == "__main__":
    unittest.main()
