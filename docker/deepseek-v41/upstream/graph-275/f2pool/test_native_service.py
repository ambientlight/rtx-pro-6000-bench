from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import native_service as service


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
