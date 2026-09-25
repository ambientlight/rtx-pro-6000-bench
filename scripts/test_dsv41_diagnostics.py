"""Offline diagnostics-wrapper tests; no Docker, GPU, model or server actions."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/'docker/deepseek-v41/diagnostic_entrypoint.py'
spec = importlib.util.spec_from_file_location('dsv41_diagnostics',PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DiagnosticEntrypointTest(unittest.TestCase):
    def setUp(self):
        self.old_umask = os.umask(0o077)

    def tearDown(self):
        os.umask(self.old_umask)

    def test_each_restart_has_private_distinct_persistent_paths(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'API_KEY':'never-export-this'}):
            first = module.prepare_diagnostics(tmp)
            second = module.prepare_diagnostics(tmp)
            self.assertNotEqual(first,second)
            for folder in (first,second):
                self.assertEqual(folder.stat().st_mode & 0o777,0o700)
                receipt = folder/'diagnostics.json'
                self.assertEqual(receipt.stat().st_mode & 0o777,0o600)
                self.assertNotIn('never-export-this',receipt.read_text())
                data=json.loads(receipt.read_text())
                self.assertEqual(data['paths']['CUDA_COREDUMP_PIPE'],'/tmp/corepipe.cuda.%h.%p')
                for key,path in data['paths'].items():
                    if key!='CUDA_COREDUMP_PIPE':
                        self.assertTrue(Path(path).is_relative_to(folder))
                self.assertIn('%t',data['paths']['CUDA_COREDUMP_FILE'])
                self.assertNotIn('%',data['paths']['TORCH_NCCL_DEBUG_INFO_TEMP_FILE'])

    def test_relative_root_is_rejected(self):
        with self.assertRaises(ValueError):
            module.prepare_diagnostics('relative-root')

    def test_receipt_records_exception_dump_and_collector_budgets(self):
        values = {
            'TORCH_NCCL_EXTRA_DUMP_ON_EXEC': '1',
            'SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS': '60',
            'SGLANG_PYSPY_DUMP_TIMEOUT_SECS': '5',
            'SGLANG_PYSPY_DUMP_TOTAL_TIMEOUT_SECS': '20',
            'SGLANG_PYSPY_DUMP_NATIVE': '1',
            'SGLANG_NCCL_DUMP_BEFORE_CRASH': '1',
            'SGLANG_NCCL_DUMP_BEFORE_CRASH_WAIT_SECS': '10',
            'SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS': '60',
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, values):
            receipt = module.prepare_diagnostics(tmp) / 'diagnostics.json'
            settings = json.loads(receipt.read_text())['settings']
            self.assertEqual({k: settings[k] for k in values}, values)

    def test_exec_preserves_original_boot_and_arguments(self):
        with patch.object(module,'prepare_diagnostics') as prepare, patch.object(module.os,'execv') as execv, patch.object(module.sys,'argv',['wrapper','run']):
            module.main()
            prepare.assert_called_once()
            execv.assert_called_once_with(module.sys.executable,[module.sys.executable,'-u','/opt/dsv41/boot.py','run'])


if __name__=='__main__':
    unittest.main()
