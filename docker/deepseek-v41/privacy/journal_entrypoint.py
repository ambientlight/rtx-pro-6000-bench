"""Publish validated diagnostics; the application's raw output goes to /dev/null.

Run this stdlib-only supervisor with Python -S. The supervised application keeps
its normal site initialization and privacy hooks. No host socket is mounted.
"""
import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_schema import MAX_DATAGRAM, clean, encode


def publish(raw):
    event = clean(raw)
    encoded = encode(event) if event is not None else None
    if encoded:
        fd, payload = encoded
        try:
            # Bounded records and one publisher avoid mixed output from workers.
            os.write(fd, payload)
        except (BlockingIOError, BrokenPipeError):
            pass  # The private diagnostic files remain the durable original.


def lifecycle(event, **fields):
    publish(json.dumps({'time': time.time(), 'pid': os.getpid(), 'event': event, **fields}))


def main(argv):
    if argv[:1] == ['--']:
        argv = argv[1:]
    if not argv:
        return 64
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # A host pipe-based core handler can bypass RLIMIT_CORE. This process also
    # inherits credentials in its environment, so it must be non-dumpable.
    if ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) != 0:
        lifecycle('log_publisher_failed')
        return 78
    for fd in (1, 2):
        os.set_blocking(fd, False)
    child = None
    pending = []

    def forward(signum, _frame):
        if child is None:
            pending.append(signum)
        else:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward)
    try:
        with tempfile.TemporaryDirectory(prefix='content-free-journal-') as directory:
            address = str(Path(directory) / 'events.sock')
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
                receiver.bind(address)
                receiver.settimeout(0.1)
                env = dict(os.environ, CONTENT_FREE_JOURNAL_SOCKET=address)
                lifecycle('log_publisher_started')
                child = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         start_new_session=True)
                for signum in pending:
                    forward(signum, None)
                ended_at = None
                while True:
                    try:
                        raw, _ancillary, flags, _sender = receiver.recvmsg(MAX_DATAGRAM)
                        if not flags & socket.MSG_TRUNC:
                            publish(raw)
                    except socket.timeout:
                        pass
                    if child.poll() is not None:
                        if ended_at is None:
                            ended_at = time.monotonic()
                        if time.monotonic() - ended_at >= 0.2:
                            break
                code = child.wait()
                lifecycle('log_publisher_exit', exit_code=code)
                return code if code >= 0 else 128 - code
    except BaseException:
        # Never print command arguments, paths supplied by an exception, or a traceback.
        lifecycle('log_publisher_failed')
        return 78
    finally:
        if child is not None and child.poll() is None:
            forward(signal.SIGTERM, None)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                forward(signal.SIGKILL, None)
                child.wait(timeout=5)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
