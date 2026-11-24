import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest


def run_with_probe():
    from watch_clock import follow
    root = Path(sys.argv[1])
    def read():
        value = (root / "clock-reading").read_text()
        (root / "probe-started").touch()
        if value == "delayed":
            until = time.monotonic() + 2
            while not (root / "finish-probe").exists() and time.monotonic() < until:
                time.sleep(.01)
            return (13365,) * 4
        return (int(value),) * 4
    follow(root, root / "clock.sock", float(sys.argv[2]), verify_after=.06, read_clocks=read)


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

    def start(self, reading=None):
        command = [sys.executable, str(Path(__file__).with_name("watch_clock.py")), "--broker-dir", str(self.root),
                   "--lease-socket", str(self.root / "clock.sock"), "--idle-seconds", str(self.grace)]
        if reading is not None:
            (self.root / "clock-reading").write_text(reading)
            command = [sys.executable, "-c", "from test_watch_clock import run_with_probe; run_with_probe()",
                       str(self.root), str(self.grace)]
        self.client = subprocess.Popen(command, cwd=Path(__file__).parent, stdout=self.events, stderr=subprocess.DEVNULL)
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
        self.start()
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
        self.start()
        self.set_epoch(1)
        first = self.accept_lease()
        self.set_epoch(3)  # Both broker edges reached the file before the observer woke.
        self.assertEqual(first.recv(1), b"")
        second = self.accept_lease()
        self.client.kill()
        self.client.wait(timeout=2)
        self.assertEqual(second.recv(1), b"")

    def test_removing_live_permit_restores_automatic(self):
        self.start()
        self.set_epoch(1)
        lease = self.accept_lease()
        (self.root / "idle.permit").unlink()
        self.assertEqual(lease.recv(1), b"")
        self.assertNotEqual(self.client.wait(timeout=2), 0)

    def test_high_readback_retries_once_in_same_idle_epoch(self):
        self.start("13365")
        self.set_epoch(1)
        first = self.accept_lease()
        self.assertEqual(first.recv(1), b"")
        second = self.accept_lease()
        self.clock.settimeout(.35)
        with self.assertRaises(TimeoutError):
            self.clock.accept()
        self.set_epoch(2)
        self.assertEqual(second.recv(1), b"")
        self.assertEqual((self.root/"events").read_text().count("retry_mining_clock"), 1)

    def test_slow_readback_does_not_block_demand_or_reset_new_lease(self):
        self.start("delayed")
        self.set_epoch(1)
        first = self.accept_lease()
        until = time.monotonic() + 1
        while not (self.root/"probe-started").exists():
            self.assertLess(time.monotonic(), until)
            time.sleep(.01)
        self.set_epoch(2)
        first.settimeout(.25)
        self.assertEqual(first.recv(1), b"")
        (self.root/"clock-reading").write_text("7001")
        self.set_epoch(3)
        second = self.accept_lease()
        (self.root/"finish-probe").touch()
        self.clock.settimeout(.35)
        with self.assertRaises(TimeoutError):
            self.clock.accept()
        self.set_epoch(4)
        self.assertEqual(second.recv(1), b"")
        self.assertNotIn("retry_mining_clock", (self.root/"events").read_text())


if __name__ == "__main__":
    unittest.main()
