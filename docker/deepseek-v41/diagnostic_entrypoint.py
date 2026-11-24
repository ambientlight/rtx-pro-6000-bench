"""Assign private, content-free diagnostics paths before the model boot.

A new directory is created on every container start, including automatic
restarts, so NCCL's truncating log writer cannot overwrite an earlier crash.
No torch/CUDA imports, model settings, or model startup logic live here.
"""
import datetime as dt
import json
import os
from pathlib import Path
import resource
import socket
import sys
import tempfile


def prepare_diagnostics(root):
    os.umask(0o077)
    root = Path(root)
    if not root.is_absolute():
        raise ValueError('DSV41_DIAGNOSTIC_ROOT must be an absolute persistent path')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(tempfile.mkdtemp(prefix=f'run-{stamp}-{socket.gethostname()}-',dir=root))
    # GPU shared/local memory can also contain prompt-derived values. Keep
    # collective/stack diagnostics, but no CPU/GPU memory or request dumps.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ.update({
        'CONTENT_FREE_LOGGING': '1',
        'CONTENT_FREE_LOG_DIR': str(run),
        'CUDA_ENABLE_COREDUMP_ON_EXCEPTION': '0',
        'CUDA_ENABLE_USER_TRIGGERED_COREDUMP': '0',
        'CUDA_COREDUMP_SHOW_PROGRESS': '0',
        'CUDA_COREDUMP_FILE': os.devnull,
        'SGLANG_CUDA_COREDUMP': '0',
        'SGLANG_CUDA_COREDUMP_BEFORE_CRASH': '0',
        'SGLANG_RESPONSES_TOOL_FAILURE_DUMP_DIR': '',
        'SGLANG_LOG_REQUEST_EXCEEDED_MS': '0',
        'SGLANG_LOG_REQUEST_HEADERS': '',
    })
    os.environ.pop('CUDA_COREDUMP_PIPE', None)
    # Only NCCL filename templates support %h/%p. PyTorch appends rank.
    paths = {
        'NCCL_DEBUG_FILE': str(run/'nccl.%h.%p.log'),
        'TORCH_NCCL_DEBUG_INFO_TEMP_FILE': str(run/'torch_nccl_trace_rank_'),
        'TORCH_FR_DUMP_TEMP_FILE': str(run/'torch_nccl_trace_rank_'),
        'TORCH_NCCL_DEBUG_INFO_PIPE_FILE': str(run/'torch_nccl_pipe_rank_'),
    }
    os.environ.update(paths)
    keys = ('CONTENT_FREE_LOGGING', 'CONTENT_FREE_LOG_DIR',
            'SGLANG_CUDA_COREDUMP', 'SGLANG_CUDA_COREDUMP_BEFORE_CRASH',
            'SGLANG_RESPONSES_TOOL_FAILURE_DUMP_DIR',
            'CUDA_LOG_FILE','CUDA_ENABLE_COREDUMP_ON_EXCEPTION',
            'CUDA_ENABLE_USER_TRIGGERED_COREDUMP','CUDA_COREDUMP_SHOW_PROGRESS',
            'CUDA_COREDUMP_GENERATION_FLAGS','NCCL_DEBUG','NCCL_DEBUG_SUBSYS',
            'TORCH_NCCL_TRACE_BUFFER_SIZE','TORCH_NCCL_TRACE_CPP_STACK',
            'TORCH_NCCL_DUMP_ON_TIMEOUT','TORCH_NCCL_ENABLE_MONITORING',
            'TORCH_NCCL_EXTRA_DUMP_ON_EXEC','TORCH_NCCL_ENABLE_TIMING',
            'SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS',
            'SGLANG_PYSPY_DUMP_TIMEOUT_SECS','SGLANG_PYSPY_DUMP_TOTAL_TIMEOUT_SECS',
            'SGLANG_PYSPY_DUMP_NATIVE','SGLANG_NCCL_DUMP_BEFORE_CRASH',
            'SGLANG_NCCL_DUMP_BEFORE_CRASH_WAIT_SECS',
            'SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS','MEMORY_FRACTION')
    receipt = {'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
               'hostname':socket.gethostname(),'paths':paths,
               'settings':{key:os.environ.get(key) for key in keys}}
    (run/'diagnostics.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return run


def main():
    prepare_diagnostics(os.environ.get('DSV41_DIAGNOSTIC_ROOT',
                                      '/mnt/hot/dsv41_dumps/diagnostics'))
    sys.path.insert(0, '/opt/content-free')
    from content_free_logging import discard_stdio, emit, install_or_exit
    discard_stdio()
    install_or_exit()
    emit({'event': 'server_starting'})
    os.execv(sys.executable,[sys.executable,'-u','/opt/dsv41/boot.py',*sys.argv[1:]])


if __name__=='__main__':
    main()
