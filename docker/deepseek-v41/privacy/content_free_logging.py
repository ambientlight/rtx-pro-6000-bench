"""Deployment-only logging policy: operational metadata, never request content.

No arbitrary message formatting, exception rendering, source snippets, locals, request
IDs, headers, or bodies enter the diagnostic sink. Ordinary stdout/stderr must
also be discarded by the launcher; native libraries bypass Python logging.
"""

import functools
import ctypes
import json
import logging
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import sys
import threading
import time

from operational_events import scheduler_metrics

_ROOTS = (
    "/sgl-workspace/sglang/python/", "/opt/sglang/", "/opt/dsv41/",
    "/usr/lib/python3.12/", "/usr/local/lib/python3.13/",
    "/app/litellm/", "/app/.venv/", "/opt/content-free/",
)
_ERRORS = frozenset((
    "AssertionError", "AttributeError", "CancelledError", "ConnectionError",
    "FileNotFoundError", "HTTPException", "IndexError", "JSONDecodeError",
    "KeyError", "MemoryError", "OSError", "OutOfMemoryError", "RuntimeError",
    "TimeoutError", "TypeError", "ValidationError", "ValueError",
))
_LOCK = threading.Lock()
_INSTALLED = False
_MAX_BYTES = 20 * 1024 * 1024


@functools.lru_cache(maxsize=4096)
def source_location(filename):
    # Only installed server/library files. Never retain a synthetic filename
    # (e.g. eval/exec text), source line, function name, or model-supplied path.
    if not isinstance(filename, str) or not re.fullmatch(r"/[A-Za-z0-9_./+\-]+", filename):
        return "omitted"
    path = os.path.realpath(filename)
    for root in _ROOTS:
        if path.startswith(root) and os.path.isfile(path):
            return path
    return "omitted"


def _number(value):
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else 0


def _frame(filename, line):
    source = source_location(filename)
    return {"source": source, "line": _number(line) if source != "omitted" else 0}


def exception_metadata(exc, tb):
    result, seen = [], set()
    while exc is not None and id(exc) not in seen and len(result) < 8:
        seen.add(id(exc))
        name = type(exc).__name__
        frames = []
        current = tb
        while current is not None and len(frames) < 64:
            frames.append(_frame(current.tb_frame.f_code.co_filename, current.tb_lineno))
            current = current.tb_next
        result.append({"type": name if name in _ERRORS else "OtherError", "frames": frames})
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
        tb = exc.__traceback__ if exc is not None else None
    return result


def emit(event):
    """Only call with locally constructed, allowlisted fields."""
    directory = os.environ.get("CONTENT_FREE_LOG_DIR")
    if not directory:
        return  # Initial entrypoint, before its per-start directory exists.
    payload = (json.dumps({"time": time.time(), "pid": os.getpid(), **event},
                          separators=(",", ":")) + "\n").encode("ascii")
    try:
        path = Path(directory) / f"runtime-{os.getpid()}.jsonl"
        with _LOCK:
            if path.exists() and path.stat().st_size + len(payload) > _MAX_BYTES:
                oldest = path.with_suffix(".jsonl.2")
                previous = path.with_suffix(".jsonl.1")
                if previous.exists():
                    previous.replace(oldest)
                path.replace(previous)
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                data = memoryview(payload)
                while data:
                    count = os.write(fd, data)
                    if count <= 0:
                        raise OSError("Diagnostic write failed")
                    data = data[count:]
            finally:
                os.close(fd)
    except BaseException:
        # Do not render errors from this path or continue with a broken sink.
        os._exit(78)
    # Only the locally constructed metadata event reaches the publisher. Raw
    # stdout/stderr remain discarded, including output from native libraries.
    address = os.environ.get('CONTENT_FREE_JOURNAL_SOCKET')
    if address and len(payload) <= 60 * 1024:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
                channel.setblocking(False)
                channel.sendto(payload, address)
        except OSError:
            pass  # Do not block inference; the private file already has this event.


def _after_fork():
    global _LOCK
    _LOCK = threading.Lock()


def install():
    global _INSTALLED
    if _INSTALLED or os.environ.get("CONTENT_FREE_LOGGING") != "1":
        return
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # A piped host core handler can bypass RLIMIT_CORE. Mark every Python
    # process non-dumpable after exec; SYS_PTRACE still permits safe py-spy.
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
            raise OSError("Unable to disable memory dumps")
    _INSTALLED = True
    original = logging.Logger.makeRecord

    def make_record(self, name, level, fn, lno, msg, args, exc_info,
                    func=None, extra=None, sinfo=None):
        # Do not call str(msg), interpolate args, inspect extra, or format an
        # exception. Only two known scheduler functions have a numeric parser;
        # their full messages must match its closed grammar.
        event = {"event": "log", "level": _number(level), **_frame(fn, lno)}
        if level == logging.INFO:
            metrics = scheduler_metrics(event['source'], func, msg, args)
            if metrics is not None:
                event['metrics'] = metrics
        info = exc_info if exc_info and exc_info[1] is not None else sys.exc_info()
        if info[1] is not None:
            event["exceptions"] = exception_metadata(info[1], info[2])
        if name != "uvicorn.access":
            emit(event)
        record = original(self, "runtime", level, event["source"], event["line"],
                          json.dumps(event, separators=(",", ":")), (), None,
                          None, None, None)
        record.threadName = "worker"
        record.processName = "worker"
        return record

    logging.Logger.makeRecord = make_record
    original_handle = logging.Logger.handle

    def handle(self, record):
        # AccessFormatter unpacks a request tuple. No access record should
        # reach it (or any other handler) under this policy.
        if self.name != "uvicorn.access":
            original_handle(self, record)

    logging.Logger.handle = handle

    def unhandled(kind, exc, tb):
        emit({"event": "unhandled_exception", "exceptions": exception_metadata(exc, tb)})

    sys.excepthook = unhandled
    threading.excepthook = lambda args: unhandled(args.exc_type, args.exc_value, args.exc_traceback)
    os.register_at_fork(after_in_child=_after_fork)
    emit({"event": "privacy_started"})


def install_or_exit():
    try:
        install()
    except BaseException:
        os._exit(78)


def discard_stdio():
    # Run in the outer launcher, BEFORE it starts its supervised child pipe.
    # Closing the child's pipe would look like a failed reader to boot.py.
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        if fd > 2:
            os.close(fd)


def collect_pyspy(cmd, timeout, deadline):
    """Keep only numeric thread identity and installed-code frame locations."""
    pid = int(cmd[cmd.index("--pid") + 1])
    native = "--native" in cmd
    command = ["py-spy", "dump", "--json", "--full-filenames", "--pid", str(pid),
               "--native" if native else "--nonblocking"]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=min(timeout, max(0.01, deadline - time.monotonic())))
        event = {"event": "stack_dump", "target_pid": pid, "native": native,
                 "exit_code": process.returncode, "threads": []}
        if process.returncode == 0 and len(output) <= 4 * 1024 * 1024:
            for thread in json.loads(output)[:128]:
                event["threads"].append({
                    "thread_id": _number(thread.get("os_thread_id")),
                    "frames": [_frame(f.get("filename"), f.get("line"))
                               for f in thread.get("frames", [])[:128]],
                })
        emit(event)
    except BaseException as exc:
        emit({"event": "stack_dump_failed", "target_pid": pid,
              "exceptions": exception_metadata(exc, exc.__traceback__)})
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        process.stdout.close()


class PrivacyAdminMiddleware:
    """Do not permit HTTP callers to re-enable payload logging or profiling."""

    BLOCKED = frozenset((
        "/configure_logging", "/start_profile", "/stop_profile", "/set_trace_level",
        "/start_expert_distribution_record", "/stop_expert_distribution_record",
        "/dump_expert_distribution_record",
    ))

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").rstrip("/") in self.BLOCKED:
            body = b'{"error":"Disabled by the deployment logging policy"}'
            await send({"type": "http.response.start", "status": 403,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
