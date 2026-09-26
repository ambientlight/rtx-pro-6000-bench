#!/usr/bin/env python3
"""Qualify/freeze a launcher-only fix, then reuse the 60-second idle rollout.

No GPU benchmark or synthetic workload. The image, environment, cache allocation
and every non-boot runtime file must match the live L15 deployment. CPU tests run
against the frozen boot with no GPU access. A watcher makes at most one attempt.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from dsv41_deferred_relaunch import (
    CONFIG, atomic_json, digest, fetch, identity, inspect, roundtrip_profile,
    run, validate_flags,
)
from dsv41_relaunch_cache_split import freeze_sources, watch
from dsv41_relaunch_diagnostics import CAPTURE, SGLANG, current_allocation


BOOT = '/opt/dsv41/boot.py'
VERSION = 'dsv41-v2-log-forwarding-v1'


def source_hashes(profile):
    return {mount['target']: digest(mount['source'])
            for mount in profile['services']['deepseek']['volumes']
            if mount['type'] == 'bind' and Path(mount['source']).suffix == '.py'}


def validate_boot_only(old, candidate):
    old, candidate = roundtrip_profile(old), roundtrip_profile(candidate)
    if set(old['services']) != {'deepseek'} or set(candidate['services']) != {'deepseek'}:
        raise ValueError('Only dsv41 may be recreated')
    a, b = old['services']['deepseek'], candidate['services']['deepseek']
    if (b['container_name'], b['restart'], b['labels'].get('dev.idle.deployment.version')) != (
        'dsv41', 'unless-stopped', VERSION
    ):
        raise ValueError('Unexpected identity, restart policy or rollout label')
    if b['environment'].get('DSV41_HICACHE_SWA_FRACTION') != '0.67':
        raise ValueError('The 67/33 allocation must stay unchanged')
    if b['environment'].get('DSV41_HICACHE_HOST_BUDGET_BYTES') != '256000000000':
        raise ValueError('Host cache must stay at 256 GB')
    if not b['image'].startswith('sha256:'):
        raise ValueError('Require an immutable image ID')
    old_hashes, new_hashes = source_hashes(old), source_hashes(candidate)
    if BOOT not in old_hashes or BOOT not in new_hashes or old_hashes[BOOT] == new_hashes[BOOT]:
        raise ValueError('Expected one changed boot launcher')
    if {k: v for k, v in old_hashes.items() if k != BOOT} != {
        k: v for k, v in new_hashes.items() if k != BOOT
    }:
        raise ValueError('Non-boot Python runtime bytes changed')
    b['labels']['dev.idle.deployment.version'] = a['labels']['dev.idle.deployment.version']
    for profile in (old, candidate):
        for mount in profile['services']['deepseek']['volumes']:
            if mount['type'] == 'bind' and Path(mount['source']).suffix == '.py':
                if not mount.get('read_only'):
                    raise ValueError('Runtime Python sources must remain read-only')
                mount['source'] = '<boot>' if mount['target'] == BOOT else digest(mount['source'])
    if old != candidate:
        raise ValueError('Non-launcher deployment configuration changed')


def validate_test_results(path, *, full_api=False):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    if not cases or any(case.find(kind) is not None for case in cases for kind in ('failure', 'error')):
        raise ValueError('Qualification tests did not pass')
    for name, minimum in (('test_boot_logging', 17), ('test_boot.', 14)):
        matched = [case for case in cases if name in case.get('classname', '')]
        if len(matched) < minimum or any(case.find('skipped') is not None for case in matched):
            raise ValueError('Missing unskipped launcher qualification tests')
    if full_api and len(cases) < 1000:
        raise ValueError('Missing full API regression qualification')
    return len(cases)


def qualify_frozen_boot(state, candidate, image_id):
    root = state / 'qualification'
    root.mkdir()
    mount = next(m for m in candidate['services']['deepseek']['volumes'] if m['target'] == BOOT)
    shutil.copyfile(mount['source'], root / 'boot.py')
    for name in ('test_boot.py', 'test_boot_logging.py'):
        shutil.copyfile(SGLANG / 'deployment' / name, root / name)
    name = 'dsv41-log-qualification-' + state.name.rsplit('-', 1)[-1].lower()
    with (state / 'boot-tests.log').open('x') as output:
        try:
            subprocess.run([
                'docker', 'run', '--rm', '--name', name, '--runtime', 'runc',
                '--network', 'none', '--cpus', '2', '--memory', '2g',
                '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--env', 'PYTHONDONTWRITEBYTECODE=1',
                '--mount', f'type=bind,src={root},dst=/qualification,readonly',
                '--mount', f'type=bind,src={state},dst=/evidence',
                '--workdir', '/qualification', '--entrypoint', 'python3', image_id,
                '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--tb=short',
                '--junitxml=/evidence/boot-tests.xml', 'test_boot.py', 'test_boot_logging.py',
            ], stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)
        except subprocess.TimeoutExpired:
            # Only this newly created CPU-test container, never the live model.
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15, check=False)
            raise RuntimeError('Frozen launcher qualification timed out') from None
    if digest(root / 'boot.py') != digest(mount['source']):
        raise RuntimeError('Boot source changed during qualification')
    return validate_test_results(state / 'boot-tests.xml')


def arm(state, api_evidence):
    if (state / 'manifest.json').exists():
        raise ValueError('Already armed; use a fresh private directory')
    current = inspect()
    if current['State']['Status'] != 'running' or current['State'].get('Health', {}).get('Status') != 'healthy':
        raise ValueError('Expected a healthy live deployment')
    image_id = current['Image']
    if (api_evidence / 'image-id.txt').read_text().strip() != image_id:
        raise ValueError('API tests used a different image')
    api_count = validate_test_results(api_evidence / 'results.xml', full_api=True)
    old_path = Path(current['Config']['Labels']['com.docker.compose.project.config_files'])
    old = roundtrip_profile(json.loads(old_path.read_text()))
    candidate = roundtrip_profile(json.loads(run('docker', 'compose', '-f', str(CONFIG), 'config', '--format', 'json')))
    tag = candidate['services']['deepseek']['image']
    if run('docker', 'image', 'inspect', '--format', '{{.Id}}', tag).strip() != image_id:
        raise ValueError('Image tag no longer matches the live image')
    old['services']['deepseek']['image'] = image_id
    candidate['services']['deepseek']['image'] = image_id
    validate_boot_only(old, candidate)
    live_env = dict(value.split('=', 1) for value in current['Config']['Env'])
    if any(live_env.get(k) != v for k, v in old['services']['deepseek']['environment'].items()):
        raise ValueError('Archived profile does not describe the live environment')
    expected = source_hashes(old)
    actual = {path: sha for sha, path in (line.split(maxsplit=1) for line in
              run('docker', 'exec', current['Id'], 'sha256sum', *expected).splitlines())}
    if actual != expected:
        raise ValueError('Archived runtime hashes do not match the live container')
    info = json.loads(fetch('/server_info'))
    validate_flags(info, True)
    allocation = current_allocation(current)
    rollback, old_hashes = freeze_sources(old, state / 'rollback-src')
    candidate, new_hashes = freeze_sources(candidate, state / 'candidate-src')
    validate_boot_only(rollback, candidate)
    for name, obj in (('rollback.compose', rollback), ('candidate.compose', candidate),
                      ('before.container.private', current), ('before.server-info.private', info),
                      ('before.allocation', allocation)):
        atomic_json(state / (name + '.json'), obj)
    for name in ('rollback.compose.json', 'candidate.compose.json'):
        run('docker', 'compose', '-f', str(state / name), 'config', '--quiet')
    boot_count = qualify_frozen_boot(state, candidate, image_id)
    if identity(inspect()) != identity(current):
        raise ValueError('Live deployment changed during qualification')
    paths = {CONFIG, Path(__file__).resolve(), old_path, api_evidence / 'results.xml',
             api_evidence / 'image-id.txt', state / 'boot-tests.xml',
             state / 'candidate.compose.json', state / 'rollback.compose.json'}
    paths.update(Path(__file__).with_name(name) for name in (
        'dsv41_load.py', 'dsv41_deferred_relaunch.py', 'dsv41_relaunch_diagnostics.py',
        'dsv41_relaunch_cache_split.py', 'dsv41_cache_monitor.py'))
    paths.update((state / 'qualification').iterdir())
    for profile in (old, rollback, candidate):
        paths.update(Path(m['source']) for m in profile['services']['deepseek']['volumes']
                     if m['type'] == 'bind' and Path(m['source']).suffix == '.py')
    paths.update(SGLANG / 'deployment' / name for name in ('boot.py', 'test_boot.py', 'test_boot_logging.py'))
    atomic_json(state / 'manifest.json', {
        'armed_at': time.time(), 'expected_identity': identity(current), 'image_tag': tag,
        'idle_seconds': 60, 'poll_seconds': 5, 'previous_cache_bytes': allocation['actual_buffer_bytes_total'],
        'capture_before': str(CAPTURE.resolve()), 'source_sha256': new_hashes,
        'rollback_source_sha256': old_hashes, 'launch_tag': 'L16',
        # Reuse the already qualified unchanged 67/33 budget/health verification.
        'cache_split_profile': 'l15-67', 'change': 'binary-safe-supervised-log-forwarding',
        'api_test_cases': api_count, 'frozen_boot_test_cases': boot_count,
        'file_hashes': {str(p): digest(p) for p in paths},
    })
    atomic_json(state / 'status.json', {'state': 'armed', 'updated_at': time.time()})
    print(f'Armed L16 launcher-only rollout: {state}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('arm', 'watch'))
    parser.add_argument('--state-dir', required=True, type=Path)
    parser.add_argument('--api-evidence', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    state = args.state_dir.resolve(strict=True)
    if state.parent != Path('/mnt/hot/dsv41_state') or not state.name.startswith('log-forwarding-rollout-'):
        parser.error('Use a fresh mktemp-created /mnt/hot/dsv41_state/log-forwarding-rollout-* directory')
    if args.command == 'arm' and args.api_evidence is None:
        parser.error('arm requires --api-evidence from the completed full API regression run')
    with (state / 'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'arm':
            arm(state, args.api_evidence.resolve(strict=True))
        else:
            watch(state)


if __name__ == '__main__':
    main()
