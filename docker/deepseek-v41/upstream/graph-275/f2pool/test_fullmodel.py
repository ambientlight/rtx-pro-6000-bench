"""Control-plane safety checks with no running service, external pool or accelerator."""
from contextlib import ExitStack
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fullmodel as m


class FullModelTest(unittest.TestCase):
    def test_thermal_cutoff_starts_independent_restore(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'STATE', Path(directory)):
            with patch.object(m,'cards',return_value=[75,92,70,71]), patch.object(m,'run') as run:
                m.guard()
            run.assert_called_once_with('systemctl','--user','start','--no-block',m.RESTORE+'.service')
            receipt=json.loads((Path(directory)/'guard.json').read_text())
            self.assertEqual(receipt['temperatures_c'], [75,92,70,71])
            snapshot = json.loads((Path(directory)/'temperature.json').read_text())
            self.assertEqual(snapshot['temperatures_c'], receipt['temperatures_c'])
            self.assertEqual(snapshot['max_c'], 92)
            self.assertGreaterEqual(receipt['timestamp_ns'], snapshot['timestamp_ns'])

    def test_guard_triggers_restore_if_filesystem_cannot_record_trip(self):
        with patch.object(m, 'cards', side_effect=OSError("telemetry unavailable")), \
             patch.object(m, 'write', side_effect=OSError("storage unavailable")), patch.object(m, 'run') as run:
            with self.assertRaises(OSError):
                m.guard()
            run.assert_called_once_with('systemctl','--user','start','--no-block',m.RESTORE+'.service')

    def test_refuses_nonprivate_gateway_token(self):
        with tempfile.TemporaryDirectory() as directory:
            token=Path(directory)/'token';token.write_text('a'*64);token.chmod(0o644)
            with self.assertRaises(RuntimeError): m.check_private(token)
            token.chmod(0o600);m.check_private(token)

    def test_prepends_live_replay_client_and_preserves_gateway_credential_path(self):
        env = {"PYTHONPATH": "/old/pkg:/old/prototype:/old/site",
               "PEARL_REPLAY_GATEWAY_TOKEN_FILE": "/private/test/token", "API_KEY": "synthetic"}
        with patch.dict(os.environ, env, clear=True), patch.object(os, "execve") as execute:
            m.launch_oci()
        python, argv, staged_env = execute.call_args.args
        self.assertEqual(python, sys.executable)
        self.assertEqual(argv, [python, str(m.OLD / "boot/boot.py"), "run"])
        self.assertEqual(staged_env["PYTHONPATH"], str(m.PROTO) + ":" + env["PYTHONPATH"])
        self.assertEqual(staged_env["PEARL_REPLAY_GATEWAY_TOKEN_FILE"], "/private/test/token")
        self.assertNotIn("API_KEY", staged_env)

    def test_restores_original_after_fencing_runtime(self):
        with tempfile.TemporaryDirectory() as path, ExitStack() as mocks:
            root=Path(path);mocks.enter_context(patch.object(m,'STATE',root))
            (root/'cutover.json').write_text('{"requested_ns": 123}')
            (root/'instance.json').write_text(json.dumps({'cgroup':str(root),'before_inode':root.stat().st_ino,'before_smokes':[1,2,3,4]}))
            calls=[]
            def command(*args,**kwargs):
                calls.append(args)
                return 'original running' if 'inspect' in args else ''
            mocks.enter_context(patch.object(m,'run',side_effect=command))
            mocks.enter_context(patch.object(m,'active',return_value=True))
            verify=mocks.enter_context(patch.object(m,'verify'))
            m.restore();verify.assert_called_once()
            self.assertFalse(json.loads((root/'restart.json').read_text())['fresh'])
            reboot=calls.index((*m.DOCKER,'start','dsv41'))
            for name in (m.SAMPLER,m.CANDIDATE,m.GATEWAY,m.BROKER,m.GUARD):
                self.assertLess(calls.index(('systemctl','--user','stop',name)),reboot)
            self.assertGreater(calls.index(('systemctl','--user','stop',m.RESTORE+'.timer')),reboot)


if __name__=='__main__':
    unittest.main()
