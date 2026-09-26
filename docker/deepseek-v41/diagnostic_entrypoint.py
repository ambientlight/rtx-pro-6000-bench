"""Assign persistent, private diagnostics paths before the unchanged model boot.

A new directory is created on every container start, including automatic
restarts, so NCCL's truncating log writer cannot overwrite an earlier crash.
No torch/CUDA imports, model settings, or model startup logic live here.
"""
import datetime as dt
import json
import os
from pathlib import Path
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
    # Only CUDA/NCCL filename templates support %h/%p. PyTorch appends rank.
    paths = {
        'CUDA_COREDUMP_FILE': str(run/'cuda.%h.%p.%t.nvcudmp'),
        'CUDA_COREDUMP_PIPE': '/tmp/corepipe.cuda.%h.%p',
        'NCCL_DEBUG_FILE': str(run/'nccl.%h.%p.log'),
        'TORCH_NCCL_DEBUG_INFO_TEMP_FILE': str(run/'torch_nccl_trace_rank_'),
        'TORCH_FR_DUMP_TEMP_FILE': str(run/'torch_nccl_trace_rank_'),
        'TORCH_NCCL_DEBUG_INFO_PIPE_FILE': str(run/'torch_nccl_pipe_rank_'),
    }
    os.environ.update(paths)
    keys = ('CUDA_LOG_FILE','CUDA_ENABLE_COREDUMP_ON_EXCEPTION',
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
    print(f'Persistent CUDA/NCCL diagnostics: {run}',flush=True)
    return run


def main():
    prepare_diagnostics(os.environ.get('DSV41_DIAGNOSTIC_ROOT',
                                      '/mnt/hot/dsv41_dumps/diagnostics'))
    os.execv(sys.executable,[sys.executable,'-u','/opt/dsv41/boot.py',*sys.argv[1:]])


if __name__=='__main__':
    main()
