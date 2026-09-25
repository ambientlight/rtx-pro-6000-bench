import copy
import unittest

from dsv41_relaunch_swa_retention import candidate_profile


class SWARolloutTest(unittest.TestCase):
    def fixture(self):
        return {
            "name": "dsv41-production",
            "services": {
                "deepseek": {
                    "container_name": "dsv41",
                    "restart": "unless-stopped",
                    "image": "old",
                    "labels": {"dev.idle.deployment.version": "dsv41-v2"},
                    "environment": {
                        "MEMORY_FRACTION": "0.87",
                        "CONTEXT_LENGTH": "524288",
                        "DSV41_HICACHE_HOST_BUDGET_BYTES": "256000000000",
                    },
                    "volumes": [{"source": "/models", "target": "/models"}],
                    "ports": [{"published": "8000", "target": 8000}],
                }
            },
        }

    def test_exact_scope_and_no_mutation(self):
        old = self.fixture()
        original = copy.deepcopy(old)
        new = candidate_profile(old, "candidate")
        self.assertEqual(old, original)
        service = new["services"]["deepseek"]
        self.assertEqual(service.pop("image"), "candidate")
        self.assertEqual(service["labels"]["dev.idle.deployment.version"], "dsv41-v2-swa-retention-v1")
        self.assertEqual(service["environment"].pop("SGLANG_OPT_HICACHE_SWA_RETAIN_REQUEST_BOUNDARIES"), "1")
        self.assertEqual(service["environment"].pop("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"), "python")
        service["image"] = "old"
        service["labels"]["dev.idle.deployment.version"] = "dsv41-v2"
        self.assertEqual(new, old)

    def test_rejects_additional_service(self):
        old = self.fixture()
        old["services"]["embedding"] = {}
        with self.assertRaises(ValueError):
            candidate_profile(old, "candidate")

    def test_rejects_wrong_identity_or_restart(self):
        for key, value in (("container_name", "dsv4"), ("restart", "always")):
            old = self.fixture()
            old["services"]["deepseek"][key] = value
            with self.assertRaises(ValueError):
                candidate_profile(old, "candidate")


if __name__ == "__main__":
    unittest.main()
