from pathlib import Path
import json,subprocess,tempfile
r=Path('/mnt/hot/ambientlight/sglang-sm120-runtime')
root=r/'candidate-unpacked/rootfs'; state=r/'oci-native-state'; state.mkdir(exist_ok=True)
model='/mnt/hot/ambientlight/models/DeepSeek-V4.1-Flash-1017'
u=Path('docker/deepseek-v41/upstream')
cfg=json.loads((root.parent/'config.json').read_text())['process']
env=dict(s.split('=',1) for s in cfg['env'])
tmp=tempfile.mkdtemp(prefix='sgoci-')
prefix='/opt/python/cpython-3.12.16-linux-x86_64-gnu'
env.update(LD_PRELOAD=str(r/'image-tools/root/usr/lib/x86_64-linux-gnu/fakechroot/libfakechroot.so'),
           FAKECHROOT_EXCLUDE_PATH=f'/proc:/dev:/sys:/mnt/hot:{tmp}',TMPDIR=tmp,PYTHONHOME=prefix,
           PYTHONUSERBASE='/opt/sglang', PATH=f'{prefix}/bin:'+env['PATH'],
           MODEL_PATH=model,STATE_PATH=str(state),DSV41_DIAGNOSTIC_ROOT=str(state/'diagnostics'))
# CPython's base bin and user-site point to the same interpreter and
# site-packages as the OCI venv. The path shim leaves prctl() native.
def run(args):
    result=subprocess.run(['/usr/bin/chroot',str(root),*args],env=env,text=True,
                          capture_output=True,timeout=40)
    print(result.returncode,result.stdout,result.stderr,flush=True)
    result.check_returncode()
    return result
security=run([prefix+'/bin/python3.12','-c','import ctypes,content_free_logging as c,resource; print(c._INSTALLED,ctypes.CDLL(None).prctl(3,0,0,0,0),resource.getrlimit(resource.RLIMIT_CORE))'])
assert security.stdout.strip()=='True 0 (0, 0)'
test=run([*cfg['args'][:-1],'prepare'])
out=u/'receipts/assembled-image.json'
d=json.loads(out.read_text()); d.update(native_privacy_entrypoint={'mode':'unpacked official CPython with user-space path translation and native Linux prctl','installed':True,'dumpable':0,'core_limit':[0,0],'command':'prepare','exit_code':0,'journal_events':[j['event'] for s in test.stdout.splitlines() if s.startswith('{') and (j:=json.loads(s)).get('event')]});out.write_text(json.dumps(d,indent=2)+'\n')
