from contextlib import ExitStack
import io
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import native_service as service


class ServingMode(unittest.TestCase):
    def test_supervisor_arms_and_runs_gateway_only_in_joined_mode(self):
        for mode in service.MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, ExitStack() as patches:
                root = Path(tmp)
                run = root / "run"
                (run/"serving").mkdir(parents=True)
                (run/"receipts").mkdir()
                for smoke in service.host.SMOKES:
                    service.save(run/"serving"/smoke, {"status":"passed"})
                started = []
                stop = threading.Event()
                def launch(children, session, name, argv, env=None):
                    started.append(name)
                    if name=="broker":
                        (root/"broker.sock").touch()
                    elif name=="gateway":
                        (run/"receipts/gateway.json").touch()
                    children[name] = Mock()
                    children[name].poll.return_value = None
                def notify(text):
                    if text.startswith("READY=1"):
                        stop.set()
                old_umask = os.umask(0o077)
                patches.callback(os.umask, old_umask)
                patches.enter_context(patch.dict(os.environ, {"RUNTIME_DIRECTORY":str(root)}))
                for target, name, value in ((service,"selected_mode",mode),(service,"fresh_session",run),
                    (service.host,"cards",[60]*4),(service.threading,"Event",stop),
                    (service.urllib.request,"urlopen",io.BytesIO(b'{"data":[{"id":"deepseek-v4-flash"}]}'))):
                    patches.enter_context(patch.object(target,name,return_value=value))
                for target,name in ((service.host,"check_artifacts"),(service,"exclusive_devices"),
                                    (service.signal,"signal"),(service,"stop_children")):
                    patches.enter_context(patch.object(target,name))
                patches.enter_context(patch.object(service,"launch",side_effect=launch))
                patches.enter_context(patch.object(service,"notify",side_effect=notify))
                service.serve()
                self.assertEqual(started,["broker","gateway","model"] if mode=="joined" else ["broker","model"])
                ready=service.read(run/"ready.json")
                self.assertEqual((ready["mode"],ready["inference_proofs_enabled"]),(mode,mode=="joined"))

    def test_mode_persists_and_invalid_updates_preserve_selection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(service, "RELEASE", Path(tmp)):
            self.assertEqual(service.selected_mode(), "joined")
            for mode in ("non-joined", "joined"):
                service.set_mode(mode)
                with self.assertRaises(ValueError):
                    service.set_mode("unknown")
                self.assertEqual(service.selected_mode(), mode)
            (Path(tmp)/"serving-mode").write_text("bad\n")
            with self.assertRaises(ValueError):
                service.selected_mode()

    def test_non_joined_excludes_inherited_projection_and_gateway(self):
        root = Path("/example")
        with patch.dict(service.os.environ, {"PEARL_GRAPH_VERIFY":"1", "PEARL_WOB_MODULE":"checkpoint_graph",
                        "PEARL_SHAPE_DIR":"stale", "PEARL_REPLAY_GATEWAY_TOKEN_FILE":"stale"}):
            idle = service.model_environment(root, root/"broker.sock", "non-joined")
            joined = service.model_environment(root, root/"broker.sock", "joined")
        self.assertEqual({k for k in idle if k.startswith("PEARL_")}, {"PEARL_NATIVE_IDLE_CONFIG","PEARL_BROKER_SOCKET"})
        self.assertEqual(idle["PEARL_NATIVE_IDLE_CONFIG"], joined["PEARL_NATIVE_IDLE_CONFIG"])
        self.assertEqual(joined["PEARL_REPLAY_GATEWAY_TOKEN_FILE"], str(root/"gateway-token"))
        self.assertEqual(joined["PEARL_WOB_MODULE"], "checkpoint_graph")

    def test_new_non_joined_session_requires_only_the_idle_library(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(service, "RELEASE", Path(tmp)):
            root = Path(tmp)
            lib = root / "libpearl_idle.so"
            lib.touch()
            service.save(root/"config.json", {"library":str(lib)})
            service.save(root/"staged.json", {"library_sha256":service.digest(lib), "source":{}})
            with patch.object(service, "VERIFIER", root/"unavailable"):
                run = service.fresh_session(root/"control", "non-joined")
                with self.assertRaises(FileNotFoundError):
                    service.fresh_session(root/"control", "joined")
            self.assertTrue((run/"config.json").is_file())
            self.assertFalse(any((run/p).exists() for p in ("gateway-token","armed","receipts","serving/shapes")))


class StockCutover(unittest.TestCase):
    def test_fresh_stock_boot_during_cutover(self):
        with patch.object(service, "stock_status", return_value="running"), \
             patch.object(service, "stock_available", return_value=False), \
             patch.object(service.host, "ensure_idle") as drain, \
             patch.object(service.host, "run", side_effect=["inactive", "", ""]) as run:
            service.exclusive_devices()
        drain.assert_not_called()
        self.assertIn("stop", run.call_args_list[1].args)

    def test_accepting_stock_service_is_drained(self):
        with patch.object(service, "stock_status", return_value="running"), \
             patch.object(service, "stock_available", return_value=True), \
             patch.object(service.host, "ensure_idle") as drain, \
             patch.object(service.host, "run", side_effect=["failed", "", ""]):
            service.exclusive_devices()
        drain.assert_called_once_with()

    def test_concurrent_fallback_treats_already_running_container_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "stock"
            original.mkdir()
            def race(*args, **kwargs):
                for name in service.host.SMOKES:
                    (original/name).touch()
                raise subprocess.CalledProcessError(1, args)
            with patch.object(service, "RELEASE", root), patch.object(service.host, "ORIGINAL", original), \
                 patch.object(service, "stock_status", side_effect=["exited", "running", "running"]), \
                 patch.object(service, "stock_available", return_value=True), \
                 patch.object(service.time, "time_ns", return_value=1), patch.object(service.host, "run", side_effect=race):
                service.fallback()


if __name__=="__main__":
    unittest.main()
