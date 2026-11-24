import datetime
import hashlib
import json
import os
import runpy
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
candidate=Path('/mnt/hot/ambientlight/repos/sglang-upstream-sm120')
script=Path(sys.argv[1]).resolve()
raw=Path(sys.argv[sys.argv.index('--receipt')+1])
meta=raw.with_suffix('.meta.json')
def out(*args):
    return subprocess.check_output(list(args),text=True).strip()
def commit():
    return out('git','-C',str(candidate),'rev-parse','HEAD')
def sha256(filepath):
    with filepath.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
src_before, start=commit(),now()
sys.argv=sys.argv[1:]
runpy.run_path(str(script),run_name='__main__')
import torch
import flashinfer
versions={}
for dist in ['torch','flashinfer-python','flashinfer-jit-cache','flashinfer-jit-cache-sm120f','triton','sgl-kernel']:
    try:
        versions[dist]=version(dist)
    except PackageNotFoundError:
        versions[dist]=None
maps={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'sparse_mla' in line}
file_hashes={str(Path(p).name):sha256(Path(p)) for p in sorted(maps) if Path(p).is_file()}
assert os.environ['CUDA_VISIBLE_DEVICES']=='0' and torch.cuda.device_count()==1
assert torch.__version__=='2.14.1+cu130' and flashinfer.__version__=='0.7.4'
props=torch.cuda.get_device_properties(0)
result=dict(official_upstream='c64f2256d58358ff65c53f0e38dc3e6bcfac8484',integration_ref_at_launch=src_before,integration_ref_at_finish=commit(),fixture=str(script.relative_to(candidate)),fixture_sha256=sha256(script),started_utc=start,completed_utc=now(),gpu=dict(index=0,uuid=str(props.uuid),compute_capability=list(torch.cuda.get_device_capability()),driver=out('nvidia-smi','-i','0','--query-gpu=driver_version','--format=csv,noheader')),runtime=dict(official_base_manifest='sha256:b872c66f40dee150709e5106d53f181719b1a4a4e8e950a908cfabff9be1a213',mode='Pinned OCI Python and packages on the host NVIDIA driver with matching external FlashInfer wheel cache',python=sys.version.split()[0],packages=versions,torch_cuda=torch.version.cuda,backend=os.environ['SGLANG_SM120_FLASHMLA_BACKEND'],flashinfer_arch_list=os.environ['FLASHINFER_CUDA_ARCH_LIST'],nvcc=out('nvcc','--version').splitlines()[-1],flashinfer_jit_cache_sm120f_cu130_sha256=sha256(Path('/tmp/pearl-upstream-native/sm120f.whl')),loaded_sparse_mla_modules=file_hashes))
meta.write_text(json.dumps(result,indent=2)+'\n')
