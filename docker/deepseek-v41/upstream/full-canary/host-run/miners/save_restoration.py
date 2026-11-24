"""Write the final original-container and F2Pool recovery receipt after the 524K test."""
import json
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from check_restored_limits import HERE,inspect

dest=Path('/mnt/hot/ambientlight/repos/rtx-pro-6000-bench/docker/deepseek-v41/upstream/full-canary')
def read(path): return json.loads(path.read_text())
def cmd(*xs): return subprocess.run(xs,capture_output=True,text=True)
def utc(t):return datetime.fromtimestamp(t,timezone.utc).isoformat(timespec='milliseconds')

def main():
    windowlog=HERE.parent/'logs/window-incumbent.log'
    if not windowlog.is_file() or 'unexpected terminal event None' not in windowlog.read_text():
        raise ValueError('expected terminated-incomplete incumbent window receipt')
    cutoff_epoch=windowlog.stat().st_mtime
    prior_healthy=read(HERE/'restored-before-524k.json')
    failure=read(dest/'window-incumbent.json')
    if failure['status']!='failed':raise ValueError('unexpected window result')
    now=inspect()
    startup=read(HERE/'restored-budget.json')
    before=read(HERE/'target-budget.json')
    source=read(HERE/'restore-plan.json')
    start=read(HERE/'restored-first.json')
    old=read(dest/'incumbent.json')['profile']
    health=cmd('docker','--context','shared-operator','ps','--filter','name=^/dsv41$','--format','{{.Status}}').stdout.strip()
    if '(healthy)' not in health or startup['utc']==prior_healthy['utc']:
        raise ValueError('waiting for reloaded original container to become healthy')
    system=cmd('systemctl','show','pearl-f2pool.service','-p','ActiveState','-p','SubState','-p','Result').stdout.strip()
    admin=cmd('sudo','-n','-l','/usr/bin/systemctl','start','pearl-f2pool.service')
    admin2=cmd('sudo','-n','-l','/usr/bin/nvidia-smi','memory-limits','-n',now['cgroup'],'-i',now['rows'][0]['uuid'],'--hard-limit','96316')
    temps=cmd('nvidia-smi','--query-gpu=index,temperature.gpu,power.limit,clocks.current.memory,memory.used','--format=csv,noheader,nounits').stdout.strip().splitlines()
    guard_pass=all(r['used_mib']<=r['expected_hard_mib'] for r in now['rows'])
    result=dict(checked_utc=now['utc'],original_container_id=source['original_container'],docker_health=health,
                health_after_524k='(healthy)' in health,restored_at=(HERE.parent/'incumbent-restore-at').read_text().strip(),
                post_524k=dict(success=False,status=failure['error'],client_failure_utc=utc(cutoff_epoch),server_oom_utc=failure['oom_utc'],server_error=failure['server_error'],server_exit_status=failure['server_exit_code'],workers_reporting_oom=failure['workers_reporting'],attempt_payload_sha256=failure['payload_sha256'],fixture_input_tokens=failure['fixture_input_tokens'],completed_usage=None,answer_check=False,source='window-incumbent.json'),
                before_restart=dict(active_model_kv_tokens=old['max_total_num_tokens'],expected_hard_mib_by_gpu=[r['expected_hard_mib'] for r in before['rows']],limits_source=before['verified_sources']),
                recreated_namespace=dict(path=now['cgroup'],first_observation_utc=start['utc'],first_hard_mib_by_gpu=[r['actual_hard_mib'] for r in start['rows']],first_used_mib_by_gpu=[r['used_mib'] for r in start['rows']]),
                initial_restored_healthy_before_524k=dict(utc=prior_healthy['utc'],runtime=prior_healthy['serving'],used_mib_by_gpu=[r['used_mib'] for r in prior_healthy['rows']]),
                restored_healthy_after_524k=dict(utc=startup['utc'],runtime=startup.get('serving'),used_mib_by_gpu=[r['used_mib'] for r in startup['rows']]),
                post_window_rows=now['rows'],quota_restored=now['prior_policy_met'],all_current_usage_within_former_limit=guard_pass,
                kv_notes='The original container recomputed its automatic KV size on startup while the memory-limits namespace was uncapped and paid miners were absent. After-start limit changes would constrain later allocations; existing KV size would remain.',
                f2pool_unit_state=system.splitlines(),f2pool_failure='The installed supervisor exited when the former Docker cgroup vanished. Restart=no. The unit validates all four preexisting hard limits; it starts workers only when hard limits equal 96316, and does not apply them.',
                admin_authorization_checks={'direct_unprivileged_nvidia_smi_set':dict(rc=4,message='Insufficient Permissions',reported_by='parent canary operator on the new cgroup'),'systemctl_start':dict(rc=admin.returncode,message=admin.stderr.strip()),'nvidia_smi_set_hard_limit':dict(rc=admin2.returncode,message=admin2.stderr.strip())},
                recovery=dict(operator_helper='/mnt/hot/ambientlight/sglang-sm120-runtime/full-canary/miners/restore_paid_mining.py --apply',validated_at_readiness=True,post_window_guard_pass=guard_pass,
                    per_card_command_template='sudo /usr/bin/nvidia-smi memory-limits -n <restored namespace> -i <recorded GPU UUID> --hard-limit 96316',
                    verify_limits=f"/usr/bin/nvidia-smi memory-limits -n {now['cgroup']} --get",start_service=source['admin_start'],
                    if_above_limit="If usage exceeds 96316 on an affected card, the deployed guard prevents lowering the cap below its live allocation. An operator must arrange to release those allocations or coordinate a new container start with the cap applied before the KV pool is sized."),
                final_physical_snapshot=dict(columns=['index','temperature_c','power_limit_w','memory_clock_mhz','total_used_mib'],rows=[list(map(float,(c.strip() for c in line.split(',')))) for line in temps]))
    out=dest/'deployment-restoration.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(filename=out.name,healthy=result['health_after_524k'],used=[r['used_mib'] for r in now['rows']],guard=guard_pass,unit=system,kv=startup['serving']['max_total_num_tokens'])))
if __name__=='__main__': main()
