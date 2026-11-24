import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest


class ClockPolicyTest(unittest.TestCase):
    grace = .18

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.broker = self.listen("broker.sock")
        self.clock = self.listen("clock.sock")
        self.permit = (self.root / "idle.permit").open("w+b", buffering=0)
        self.addCleanup(self.permit.close)
        self.set_epoch(0)
        self.events = (self.root / "events").open("w")
        self.addCleanup(self.events.close)
        self.client = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("watch_clock.py")), "--broker-dir", str(self.root),
             "--lease-socket", str(self.root / "clock.sock"), "--idle-seconds", str(self.grace)],
            stdout=self.events, stderr=subprocess.DEVNULL)
        self.addCleanup(self.stop)
        self.broker.accept()[0].close()
        end = time.monotonic() + 2
        while "broker_state" not in (self.root / "events").read_text():
            if time.monotonic() >= end:
                self.fail("watcher did not read initial state")
            time.sleep(.005)

    def listen(self, name):
        s = socket.socket(socket.AF_UNIX)
        self.addCleanup(s.close)
        s.bind(str(self.root / name))
        s.listen()
        s.settimeout(2)
        return s

    def set_epoch(self, epoch):
        os.pwrite(self.permit.fileno(), struct.pack("<QQ", epoch, os.getpid()), 0)

    def stop(self):
        if self.client.poll() is None:
            self.client.terminate()
        self.client.wait(timeout=2)

    def accept_lease(self):
        self.clock.settimeout(2)
        lease, _ = self.clock.accept()
        self.addCleanup(lease.close)
        lease.settimeout(1)
        return lease

    def test_burst_cancels_timer_and_demand_releases_immediately(self):
        self.set_epoch(1)
        time.sleep(.035)
        self.set_epoch(2)
        self.clock.settimeout(self.grace + .04)
        with self.assertRaises(TimeoutError):
            self.clock.accept()
        start = time.monotonic()
        self.set_epoch(3)
        lease = self.accept_lease()
        self.assertGreaterEqual(time.monotonic() - start, self.grace - .01)
        self.set_epoch(4)
        self.assertEqual(lease.recv(1), b"")

    def test_intervening_demand_and_controller_loss_release_lease(self):
        self.set_epoch(1)
        first = self.accept_lease()
        self.set_epoch(3)  # Both broker edges reached the file before the observer woke.
        self.assertEqual(first.recv(1), b"")
        second = self.accept_lease()
        self.client.kill()
        self.client.wait(timeout=2)
        self.assertEqual(second.recv(1), b"")

    def test_removing_live_permit_restores_automatic(self):
        self.set_epoch(1)
        lease = self.accept_lease()
        (self.root / "idle.permit").unlink()
        self.assertEqual(lease.recv(1), b"")
        self.assertNotEqual(self.client.wait(timeout=2), 0)


if __name__ == "__main__":
    unittest.main()
