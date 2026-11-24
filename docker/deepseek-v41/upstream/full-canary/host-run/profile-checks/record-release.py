from pathlib import Path
import json,subprocess,hashlib,xml.etree.ElementTree as ET
r=Path('/mnt/hot/ambientlight/sglang-sm120-runtime')
u=Path('docker/deepseek-v41/upstream')
p=u/'receipts/serving-api.json'; report=json.loads(p.read_text()); t=ET.parse(r/'api-oci-deployed-tokenizer.xml').getroot().find('testsuite')
assert int(t.attrib['tests'])==291+185 and all(int(t.attrib[k])==0 for k in ('failures','errors','skipped'))
report['candidate_dependency_image_retest']={
    'integration_ref':'481e8e66504cb680ccb1316eaf01ba09b03da5cd',
    'official_base_manifest':'sha256:b872c66f40dee150709e5106d53f181719b1a4a4e8e950a908cfabff9be1a213',
    'offline_tokenizer_fixture':True, 'torch':'2.14.1+cu130','transformers':'5.19.0',
    'deepseek-recipe':'0.2.0','cryptography':'50.0.2','jsonschema':'4.26.0',
    'tests_passed':291,'subtests_passed':185,'skipped':0,'failed':0,
    'duration_seconds':float(t.attrib['time']),
    'junit_sha256':hashlib.sha256((r/'api-oci-deployed-tokenizer.xml').read_bytes()).hexdigest()}
p.write_text(json.dumps(report,indent=2)+'\n')
oci=r/'dsv41-candidate.oci'
def load(d):return json.loads((oci/'blobs/sha256'/d['digest'].split(':')[1]).read_bytes())
descriptor=json.loads((oci/'index.json').read_text())['manifests'][0]; manifest=load(descriptor); cfg=load(manifest['config'])['config']
def git(*args):return subprocess.check_output(['git','-C','../sglang-upstream-sm120',*args],text=True).strip()
def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()
lock={'upstream_url':'https://github.com/sgl-project/sglang.git','upstream_commit':'c64f2256d58358ff65c53f0e38dc3e6bcfac8484',
      'candidate_branch':'sm120/production-reconstruction','candidate_commit':git('rev-parse','HEAD'),
      'candidate_tree':git('rev-parse','HEAD^{tree}'),'bundle_sha256':sha(u/'sglang-sm120.bundle'),
      'restore_verified':git('rev-parse','HEAD^{tree}')==subprocess.check_output(['git','-C',str(r/'restored-candidate'),'rev-parse','HEAD^{tree}'],text=True).strip(),
      'official_image':{'repository':'docker.io/lmsysorg/sglang','tag':'nightly-dev-20261122-e96e9675','platform':'linux/amd64','manifest_digest':'sha256:b872c66f40dee150709e5106d53f181719b1a4a4e8e950a908cfabff9be1a213','source':'e96e96757018ce842d570781ec8312f8c470493b'},
      'candidate_oci':{'manifest_digest':descriptor['digest'],'config_digest':manifest['config']['digest'],'layer_count':len(manifest['layers']),'application_layer':manifest['layers'][-1]['digest'],'compressed_layer_bytes':sum(i['size'] for i in manifest['layers']),'entrypoint':cfg['Entrypoint'],'command':cfg['Cmd'],'artifact_path':str(oci)},
      'installed_native':{'source_rust_server_sha256':sha('../sglang-upstream-sm120/rust/target/release/libsglang_server.so'),
                          'pearl_sha256':{f.name:sha(f) for f in sorted(Path('/tmp/pearl-gpu-prepare-sm120').glob('*.so'))}},
      'checkpoint':{'repo':'deepseek-ai/DeepSeek-V4.1-Flash-1017','revision':'1bae0e1088993b9b82de08f59de5be5a34e42170','manifest':'checkpoint.json','files':83,'verified_bytes':sum(f['bytes'] for f in json.loads((u/'checkpoint.json').read_text())['files'])}}
(u/'release.json').write_text(json.dumps(lock,indent=2)+'\n')
