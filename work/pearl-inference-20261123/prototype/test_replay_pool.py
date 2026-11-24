"""Exercise the public replay pool transport with local TCP and synthetic witnesses."""
import base64
from contextlib import ExitStack
import gzip
import json
import os
import socket
import struct
import threading
import time
import unittest
from unittest.mock import patch

from replay_pool import MAX_FACTOR, MAX_FRAME, ReplayPool, compact, frame, parse_job, submission_frame


def notice(seed=1):
    target = (0x100000 << (8 * 27)) + 99
    header = struct.pack("<I32s32sII", 1, bytes([seed]) * 32, b"x" * 32, 42, 0x1c00ffff)
    return dict(method="mining.notify", params=dict(job_id="same-id", cert_version=3,
                header=header.hex(), target=f"{target:064x}"))


def until(read, timeout=3):
    limit = time.monotonic() + timeout
    while time.monotonic() < limit:
        result = read()
        if result:
            return result
        time.sleep(.003)
    raise AssertionError("timed out before the expected state")


class PoolTest(unittest.TestCase):
    def test_local_session(self):
        with ExitStack() as stack:
            listener = stack.enter_context(socket.create_server(("127.0.0.1", 0)))
            listener.settimeout(3)
            pool = ReplayPool(*listener.getsockname(), capacity=1)
            stack.callback(pool.close)

            def connection():
                sock, _ = listener.accept()
                stack.enter_context(sock)
                sock.settimeout(3)
                stream = stack.enter_context(sock.makefile("rb"))
                auth = json.loads(stream.readline())
                self.assertEqual((auth["method"], auth["params"]["type"], auth["params"]["wallet"]),
                                 ("mining.authorize", "v2", "fixture"))
                return sock, stream, frame(dict(id=auth["id"], result=True, error=None, type="v2"))

            sock, stream, auth = connection()
            sock.sendall(frame(notice()))
            self.assertIsNone(pool.snapshot())
            sock.sendall(auth)
            job1 = until(pool.snapshot)
            self.assertEqual(job1.share_bits, 0x1e100000)
            self.assertEqual(job1.header.hex(), notice()["params"]["header"])

            def sent(sock_stream, job, phase, payload):
                self.assertTrue(pool.submit(job, phase, payload))
                msg = json.loads(sock_stream.readline())
                self.assertEqual(msg["method"], "mining.submit")
                self.assertEqual(msg["params"]["job_id"], job.id)
                self.assertEqual(gzip.decompress(base64.b64decode(msg["params"]["plain_proof"], validate=True)),
                                 payload)
                return msg["id"]

            first = sent(stream, job1, "decode", b"first synthetic proof")
            # A pending ACK consumes a slot even after the queue has been drained.
            self.assertFalse(pool.submit(job1, "prefill", b"queue full"))
            # Tip changes with reused job ID and an old ACK in the same TCP write.
            sock.sendall(frame(notice(2)) + frame(dict(id=first, result=True)))
            job2 = until(lambda: pool.snapshot() if pool.snapshot() is not job1 else None)
            self.assertEqual(job2.generation, job1.generation)
            self.assertFalse(pool.submit(job1, "decode", b"late old tip"))
            self.assertEqual(pool.counters()["decode"]["stale"], 2)
            self.assertEqual(pool.counters()["decode"]["ack"], 0)

            prefill = sent(stream, job2, "prefill", b"verified prefill")
            sock.sendall(frame(dict(id=prefill, result=True, error=None)))
            until(lambda: pool.counters()["prefill"]["ack"] == 1)
            sock.sendall(frame(notice(2)))  # Identical notification preserves the handle.
            dec = sent(stream, job2, "decode", b"rejected by fixture")
            sock.sendall(frame(dict(id=dec, result=False, error="rejected")))
            until(lambda: pool.counters()["decode"]["reject"] == 1)
            sent(stream, job2, "prefill", b"unacknowledged when disconnected")
            sock.shutdown(socket.SHUT_RDWR)
            until(lambda: pool.snapshot() is None)
            self.assertFalse(pool.submit(job2, "prefill", b"disconnected"))

            sock, stream, auth = connection()
            sock.sendall(auth + frame(notice(2)))
            job3 = until(pool.snapshot)
            self.assertGreater(job3.generation, job2.generation)
            self.assertEqual(job3.header, job2.header)
            self.assertFalse(pool.submit(job2, "decode", b"from previous connection"))
            dec = sent(stream, job3, "decode", b"verified decode")
            sock.sendall(frame(dict(id=dec, result=True, error=None)))
            until(lambda: pool.counters()["decode"]["ack"] == 1)
            self.assertEqual(pool.counters()["prefill"], dict(enqueued=2, ack=1, reject=0, stale=2, drop=1))
            self.assertEqual(pool.counters()["decode"], dict(enqueued=3, ack=1, reject=1, stale=3, drop=0))
            pool.close()
            self.assertIsNone(pool.snapshot())
            self.assertEqual(stream.readline(), b"")
            self.assertFalse(pool._thread.is_alive())

    def test_protocol_validation(self):
        params = notice()["params"]
        for edit in (dict(cert_version=2), dict(cert_version=3.0), dict(header="ab"),
                     dict(target="f" * 64), dict(target="0" * 64), dict(target=" " + "0" * 63),
                     dict(job_id="")):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                parse_job({**params, **edit}, generation=1)
        boundary = ((1 << 256) - 1) // MAX_FACTOR
        for target in (1, 0x7fffff, 0x80ffff, boundary):
            bits = parse_job({**params, "target": f"{target:064x}"}, generation=2).share_bits
            size, mant = bits >> 24, bits & 0x7fffff
            floor = mant << (8 * (size - 3)) if size > 3 else mant >> (8 * (3 - size))
            self.assertLessEqual(floor, target)
            self.assertEqual(compact(floor), bits)
        with self.assertRaises(ValueError):
            parse_job({**params, "target": f"{boundary + 1:064x}"}, generation=2)
        for host in ("example.com", "198.51.100.42", "::"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                ReplayPool(host, 1)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ReplayPool.from_env())

    def test_tip_rotates_while_compressing_and_wire_cap(self):
        started, finish = threading.Event(), threading.Event()

        def compress(serial, job, raw):
            if raw == b"hold compressor":
                started.set()
                assert finish.wait(3)
            return submission_frame(serial, job, raw)

        with patch("replay_pool.submission_frame", compress), socket.create_server(("127.0.0.1", 0)) as listener:
            listener.settimeout(3)
            pool = ReplayPool(*listener.getsockname(), capacity=2)
            try:
                with listener.accept()[0] as sock, sock.makefile("rb") as stream:
                    sock.settimeout(3)
                    auth = json.loads(stream.readline())
                    sock.sendall(frame(dict(id=auth["id"], result=True, type="v2")) + frame(notice()))
                    original = until(pool.snapshot)
                    self.assertTrue(pool.submit(original, "decode", b"hold compressor"))
                    self.assertTrue(started.wait(3))
                    self.assertTrue(pool.submit(original, "prefill", b"queued old job"))
                    sock.sendall(frame(notice(2)))
                    current = until(lambda: pool.snapshot() if pool.snapshot() is not original else None)
                    self.assertEqual(pool.counters()["prefill"]["stale"], 1)
                    self.assertEqual(pool.counters()["decode"]["stale"], 1)
                    finish.set()
                    self.assertTrue(pool.submit(current, "decode", os.urandom(MAX_FRAME)))
                    until(lambda: pool.counters()["decode"]["drop"] == 1)
                    self.assertTrue(pool.submit(current, "decode", b"current"))
                    msg = json.loads(stream.readline())
                    self.assertEqual(gzip.decompress(base64.b64decode(msg["params"]["plain_proof"])), b"current")
                    sock.sendall(frame(dict(id=msg["id"], result=True)))
                    until(lambda: pool.counters()["decode"]["ack"] == 1)
            finally:
                finish.set()
                pool.close()

    def test_bad_auth_does_not_expose_preannounced_job(self):
        with socket.create_server(("127.0.0.1", 0)) as listener:
            listener.settimeout(3)
            pool = ReplayPool(*listener.getsockname())
            try:
                with listener.accept()[0] as sock, sock.makefile("rb") as stream:
                    sock.settimeout(2)
                    auth = json.loads(stream.readline())
                    sock.sendall(frame(notice()) + frame(dict(id=auth["id"], result=True, type="v1")))
                    self.assertEqual(stream.readline(), b"")
                    self.assertIsNone(pool.snapshot())
            finally:
                pool.close()


if __name__ == "__main__":
    unittest.main()
