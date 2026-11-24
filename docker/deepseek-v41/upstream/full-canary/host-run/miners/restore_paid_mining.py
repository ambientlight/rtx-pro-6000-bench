#!/usr/bin/python3
"""Restore the four previously deployed GPU limits, then start the existing F2Pool service."""
import argparse
import json
import subprocess
from pathlib import Path

NAMESPACE='/sys/fs/cgroup/dsv41.slice/docker-087c42c0a93e58dc2ab902b13c1719a03ba6a2a261a3b685129d7743df5e3130.scope'
EXPECTED=['GPU-ed6c5013-1e0a-ec80-be70-9aab1629ba5d','GPU-401b534a-f12c-63a0-def4-429e3bfa6ef6',
          'GPU-36bcefcc-7d05-78d1-7e86-08f29dd2409b','GPU-b000b08e-8683-d776-925b-a9704b764ddc']
LIMIT=96316
HARDWARE_INSPECTION=Path(__file__).resolve().with_name('check_restored_limits.py')
def report():
    return json.loads(subprocess.check_output(['/usr/bin/python3','-I',str(HARDWARE_INSPECTION)],text=True))
def healthy():
    status=subprocess.check_output(['/usr/bin/docker','--context','shared-operator','ps','--filter','name=^/dsv41$','--format','{{.Status}}'],text=True)
    return '(healthy)' in status
def validate(state):
    if (not healthy() or state['original_container_state']!='running' or state['cgroup']!=NAMESPACE
            or not state['namespace_present'] or [r['uuid'] for r in state['rows']]!=EXPECTED):
        raise SystemExit('Original four-card DeepSeek deployment must be healthy and present.')
    for row in state['rows']:
        if row['expected_hard_mib']!=LIMIT or row.get('actual_hard_mib') not in ('max',LIMIT):
            raise SystemExit(f'GPU {row["index"]}: unexpected configuration, inspect before applying.')
        if row.get('used_mib',LIMIT+1)>LIMIT and row['actual_hard_mib']!=LIMIT:
            raise SystemExit(f'GPU {row["index"]} uses {row["used_mib"]} MiB; its prior {LIMIT} MiB cap cannot be reapplied while this allocation is live.')
def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--apply',action='store_true',help='authenticate with sudo for per-board writes and original systemd startup')
    apply=cli.parse_args().apply
    state=report()
    print(json.dumps(state,indent=2),flush=True)
    validate(state)
    if not apply:
        return
    for row in state['rows']:
        if row['actual_hard_mib'] != LIMIT:
            # Elevate the same two trusted binaries used by the installed service.
            # The helper and inventory lookup continue running as the invoking user.
            subprocess.run(['/usr/bin/sudo','--','/usr/bin/nvidia-smi','memory-limits','-n',NAMESPACE,'-i',row['uuid'],'--hard-limit',str(LIMIT)],check=True)
    final=report()
    validate(final)
    if not final['prior_policy_met']:
        raise SystemExit('Four original caps have not been verified.')
    subprocess.run(['/usr/bin/sudo','--','/usr/bin/systemctl','start','pearl-f2pool.service'],check=True)
    print('Four GPU caps verified. The F2Pool unit is starting and will record all_workers_accepted after all four live ACKs.')
if __name__=='__main__':
    main()
