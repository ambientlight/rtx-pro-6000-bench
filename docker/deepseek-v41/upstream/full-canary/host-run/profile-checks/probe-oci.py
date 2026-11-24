from pathlib import Path
import os,json,subprocess,sys
r=Path('/mnt/hot/ambientlight/sglang-sm120-runtime')
oci=r/'dsv41-candidate.oci'; root=r/'candidate-unpacked/rootfs'; u=Path.cwd()/'docker/deepseek-v41/upstream'
def load(d):return json.loads((oci/'blobs/sha256'/d['digest'].split(':')[1]).read_bytes())
m=load(json.loads((oci/'index.json').read_text())['manifests'][0]); config=load(m['config'])['config']
env={k:v for k,v in (entry.split('=',1) for entry in config['Env'])}
area=r/'oci-probe-state'; area.mkdir(exist_ok=True)
temp=r/'oci-probe-tmp'; temp.mkdir(exist_ok=True);temp.chmod(0o700)
env.update(HOME='/root',TMPDIR='/tmp')
bindings=['/proc','/dev','/sys',f'{temp}:/tmp',f'{area}:/state',
          f'{u}/check_image.py:/tmp/check_image.py',
          '/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017:/models/DeepSeek-V4.1-Flash']
cmd=[str(r/'image-tools/root/usr/bin/proot'),'-r',str(root),'-w','/']
for bind in bindings:cmd+=['-b',bind]
def run(args):
    return subprocess.run([*cmd,'/usr/bin/env','-i',*[f'{k}={v}' for k,v in env.items()],*args],
        capture_output=True,text=True,timeout=120,env={**os.environ,'PROOT_NO_SECCOMP':'1',
        'LD_LIBRARY_PATH':str(r/'image-tools/root/usr/lib/x86_64-linux-gnu')})
env['CONTENT_FREE_LOGGING']='0'  # proot returns ENOSYS for PR_SET_DUMPABLE.
test=run(['/opt/sglang/bin/python','/tmp/check_image.py']); print(test.returncode,test.stdout,test.stderr,flush=True); test.check_returncode()
data=json.loads(test.stdout)
boot=run(['/opt/sglang/bin/python','-S','/opt/dsv41/boot.py','prepare']);print(boot.returncode,boot.stdout,boot.stderr,flush=True);boot.check_returncode()
args=run(['/opt/sglang/bin/python','-S','/opt/dsv41/boot.py','args']);args.check_returncode()
data.update(manifest=json.loads((oci/'index.json').read_text())['manifests'][0]['digest'],
            rootfs='rootless umoci unpack of all candidate OCI layers; proot against the unpacked root',
            offline_boot_checkpoint_prepare=boot.returncode, rootfs_import_only_privacy_mode='disabled for proot import because its prctl(4) returns ENOSYS',checkpoint_files=83,
            standard_launch_args=json.loads(args.stdout), framework_tests='CPU import and library loading; no GPU model allocation')
(u/'receipts/assembled-image.json').write_text(json.dumps(data,indent=2)+'\n')
