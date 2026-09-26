"""Offline launcher-only rollout scope and qualification gates."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import dsv41_relaunch_log_forwarder as rollout


class LogForwarderRolloutTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, text in (('old.py', 'old'), ('new.py', 'new'), ('same.py', 'same')):
            (self.root / name).write_text(text)
        self.old = {'services': {'deepseek': {
            'container_name': 'dsv41', 'restart': 'unless-stopped', 'image': 'sha256:pinned',
            'labels': {'dev.idle.deployment.version': 'old'},
            'environment': {'DSV41_HICACHE_SWA_FRACTION': '0.67', 'DSV41_HICACHE_HOST_BUDGET_BYTES': '256000000000',
                            'MEMORY_FRACTION': '0.87', 'CONTEXT_LENGTH': '524288'},
            'volumes': [
                {'type': 'bind', 'source': str(self.root / 'old.py'), 'target': rollout.BOOT, 'read_only': True},
                {'type': 'bind', 'source': str(self.root / 'same.py'), 'target': '/runtime/cache.py', 'read_only': True},
            ],
        }}}
        self.new = copy.deepcopy(self.old)
        self.new['services']['deepseek']['labels']['dev.idle.deployment.version'] = rollout.VERSION
        self.new['services']['deepseek']['volumes'][0]['source'] = str(self.root / 'new.py')

    def test_only_changed_boot_and_label_accepted_without_mutating_profiles(self):
        original = copy.deepcopy((self.old, self.new))
        rollout.validate_boot_only(self.old, self.new)
        self.assertEqual((self.old, self.new), original)

    def test_inference_budget_and_diagnostics_drift_rejected(self):
        for key in ('MEMORY_FRACTION', 'CONTEXT_LENGTH', 'DSV41_HICACHE_SWA_FRACTION',
                    'DSV41_HICACHE_HOST_BUDGET_BYTES', 'NCCL_DEBUG'):
            candidate = copy.deepcopy(self.new)
            candidate['services']['deepseek']['environment'][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                rollout.validate_boot_only(self.old, candidate)

    def test_image_identity_restart_and_mount_drift_rejected(self):
        for key, value in (('image', 'sha256:other'), ('restart', 'always'), ('container_name', 'other'),
                           ('entrypoint', ['different'])):
            candidate = copy.deepcopy(self.new)
            candidate['services']['deepseek'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                rollout.validate_boot_only(self.old, candidate)
        candidate = copy.deepcopy(self.new)
        candidate['services']['deepseek']['volumes'][0]['read_only'] = False
        with self.assertRaisesRegex(ValueError, 'read-only'):
            rollout.validate_boot_only(self.old, candidate)

    def test_nonboot_source_change_rejected(self):
        self.new['services']['deepseek']['volumes'][1]['source'] = str(self.root / 'new.py')
        with self.assertRaisesRegex(ValueError, 'Non-boot'):
            rollout.validate_boot_only(self.old, self.new)

    def test_noop_or_missing_boot_rejected(self):
        self.new['services']['deepseek']['volumes'][0]['source'] = str(self.root / 'old.py')
        with self.assertRaisesRegex(ValueError, 'changed boot'):
            rollout.validate_boot_only(self.old, self.new)
        self.new['services']['deepseek']['volumes'].pop(0)
        with self.assertRaisesRegex(ValueError, 'changed boot'):
            rollout.validate_boot_only(self.old, self.new)

    def test_tag_and_extra_service_rejected(self):
        self.old['services']['deepseek']['image'] = self.new['services']['deepseek']['image'] = 'mutable-tag'
        with self.assertRaisesRegex(ValueError, 'immutable'):
            rollout.validate_boot_only(self.old, self.new)
        self.new['services']['embedding'] = {}
        with self.assertRaisesRegex(ValueError, 'Only dsv41'):
            rollout.validate_boot_only(self.old, self.new)

    def junit(self, *, failure=False, skipped=False, logging_count=17, api=0):
        root = ET.Element('testsuites'); suite = ET.SubElement(root, 'testsuite')
        for name, count in (('test_boot_logging.LogTests', logging_count), ('test_boot.LaunchPolicyTest', 14), ('api.Test', api)):
            for i in range(count):
                case = ET.SubElement(suite, 'testcase', classname=name, name=f'test_{i}')
                if failure and name.startswith('test_boot_logging') and i == 0: ET.SubElement(case, 'failure')
                if skipped and name.startswith('test_boot_logging') and i == 0: ET.SubElement(case, 'skipped')
        path = self.root / 'tests.xml'; ET.ElementTree(root).write(path)
        return path

    def test_complete_boot_and_api_qualification(self):
        self.assertEqual(rollout.validate_test_results(self.junit()), 31)
        self.assertEqual(rollout.validate_test_results(self.junit(api=1172), full_api=True), 1203)

    def test_failed_skipped_or_missing_launcher_cases_rejected(self):
        for kwargs in ({'failure': True}, {'skipped': True}, {'logging_count': 16}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rollout.validate_test_results(self.junit(**kwargs))
        with self.assertRaisesRegex(ValueError, 'full API'):
            rollout.validate_test_results(self.junit(), full_api=True)

    def test_rearming_is_forbidden(self):
        (self.root / 'manifest.json').write_text('{}')
        with patch.object(rollout, 'inspect') as inspect, self.assertRaisesRegex(ValueError, 'Already armed'):
            rollout.arm(self.root, self.root)
        inspect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
