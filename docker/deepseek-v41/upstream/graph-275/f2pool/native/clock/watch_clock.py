"""Follow the native Pearl broker's GPU permit and hold a clock lease during long idle gaps."""
import argparse
from contextlib import ExitStack
import ctypes
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import stat
import struct
import time


PERMIT = struct.Struct("<QQ")
PEERCRED = struct.Struct("3i")
LEASE = Path("/run/dsv41-memory-clock.sock")


def emit(event, **fields):
    print(json.dumps(dict(event=event, monotonic_ns=time.monotonic_ns(), **fields)), flush=True)


def broker_pid(directory):
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(1)
        client.connect(str(directory / "broker.sock"))
        pid, uid, _ = PEERCRED.unpack(client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, PEERCRED.size))
    if uid != os.getuid():
        raise RuntimeError("Unexpected broker owner")
    return pid


def watch_changes(fd):
    libc = ctypes.CDLL(None, use_errno=True)
    events = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if events < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    # Watch the opened inode. Subscribing before the initial read avoids a missed transition.
    mask = 0x2 | 0x4 | 0x400 | 0x800  # IN_MODIFY, IN_ATTRIB, IN_DELETE_SELF, IN_MOVE_SELF
    if libc.inotify_add_watch(events, os.fsencode(f"/proc/self/fd/{fd}"), mask) < 0:
        error = ctypes.get_errno()
        os.close(events)
        raise OSError(error, "inotify_add_watch")
    return events


def read_epoch(fd, pid):
    st = os.fstat(fd)
    if not st.st_nlink or not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
        raise RuntimeError("Native permit was removed or changed owner")
    value = os.pread(fd, PERMIT.size, 0)
    if len(value) != PERMIT.size:
        raise RuntimeError("Native permit is incomplete")
    epoch, owner = PERMIT.unpack(value)
    if owner != pid:
        raise RuntimeError("Native permit and live broker have different owners")
    return epoch


def follow(directory, lease_path, grace):
    pid = broker_pid(directory)
    with ExitStack() as resources:
        def managed(fd):
            resources.callback(os.close, fd)
            return fd

        permit = managed(os.open(directory / "idle.permit", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC))
        changes = managed(watch_changes(permit))
        owner = managed(os.pidfd_open(pid))
        select = resources.enter_context(selectors.DefaultSelector())
        select.register(owner, selectors.EVENT_READ, "broker")
        select.register(changes, selectors.EVENT_READ, "permit")
        epoch = None
        deadline = None
        lease = None

        def release():
            nonlocal lease
            if lease is not None:
                select.unregister(lease)
                lease.close()
                lease = None
                emit("automatic_requested", epoch=epoch)

        resources.callback(release)
        while True:
            current = read_epoch(permit, pid)
            if current != epoch:
                epoch = current
                # A jump between two odd epochs also contains intervening inference.
                release()
                deadline = time.monotonic() + grace if epoch % 2 else None
                emit("broker_state", epoch=epoch, idle=bool(epoch % 2))
            if deadline is not None and time.monotonic() >= deadline:
                candidate = socket.socket(socket.AF_UNIX)
                candidate.settimeout(.15)
                try:
                    candidate.connect(str(lease_path))
                except OSError as err:
                    candidate.close()
                    deadline = time.monotonic() + grace
                    emit("clock_lease_unavailable", errno=err.errno, epoch=epoch)
                else:
                    lease = candidate
                    select.register(lease, selectors.EVENT_READ, "lease")
                    deadline = None
                    emit("mining_clock_requested", epoch=epoch)
            timeout = None if deadline is None else max(0, deadline - time.monotonic())
            for key, _ in select.select(timeout):
                if key.data == "broker":
                    raise RuntimeError("Native broker exited")
                if key.data == "permit":
                    os.read(changes, 65536)
                elif key.data == "lease":
                    # A failed or stopped privileged service closes its end. Recheck the broker.
                    release()
                    deadline = time.monotonic() + grace if epoch % 2 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    parser.add_argument("--broker-dir", type=Path, default=runtime / "dsv41-pearl")
    parser.add_argument("--lease-socket", type=Path, default=LEASE)
    parser.add_argument("--idle-seconds", type=float, default=5)
    args = parser.parse_args()
    if not 0 < args.idle_seconds <= 3600:
        parser.error("idle-seconds must be between zero and 3600")
    def stop(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    follow(args.broker_dir, args.lease_socket, args.idle_seconds)


if __name__ == "__main__":
    main()
