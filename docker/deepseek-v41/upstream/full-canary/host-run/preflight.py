import ctypes, importlib.metadata as m, json, mmap, os, resource
from pathlib import Path
import torch,torch.distributed as dist
from sglang.srt.rust_extensions import _server
rank=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(rank)
ctx=dist.init_process_group("nccl",device_id=torch.device("cuda",rank))
host=mmap.mmap(-1,64<<20)
ptr=ctypes.addressof(ctypes.c_byte.from_buffer(host))
cerr=torch.cuda.cudart().cudaHostRegister(ptr,len(host),0)
assert int(cerr)==0, str(cerr)
val=torch.tensor([rank+1],device=f"cuda:{rank}")
dist.all_reduce(val)
assert val.item()==10
assert int(torch.cuda.cudart().cudaHostUnregister(ptr))==0
out={"rank":rank,"gpus":dist.get_world_size(),"allreduce":int(val.item()),"registered_host_bytes":len(host),
     "rlimit_memlock":resource.getrlimit(resource.RLIMIT_MEMLOCK),"torch":torch.__version__,
     "sglang":m.version("sglang"),"flashinfer":m.version("flashinfer-python"),"server_module":_server.__file__}
Path(os.environ["STATE_PATH"],f"preflight-rank{rank}.json").write_text(json.dumps(out,indent=2)+"\n")
dist.destroy_process_group()
