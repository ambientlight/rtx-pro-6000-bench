"""Smoke the installed distribution and native libraries from inside the candidate."""
import ctypes
import importlib.metadata as metadata
import json
import os
from pathlib import Path

import sglang
from sglang.srt.rust_extensions import _server

from checkpoint_projection import Operator
from gpu_prepare.activation import GpuPrepare

rev = os.environ["SGLANG_GIT_COMMIT"]
version = metadata.version("sglang")
assert version == sglang.__version__ == f"0.0.0.dev1+g{rev[:12]}"
assert Path(sglang.__file__).resolve() == Path("/sgl-workspace/sglang/python/sglang/__init__.py")
assert _server.__file__.startswith("/sgl-workspace/sglang/python/sglang/srt/rust_extensions/")
assert Operator is not None and GpuPrepare is not None
libs = sorted(Path(os.environ["PEARL_OPERATOR_BUILD"]).glob("libpearl*.so"))
assert len(libs) == 3
for lib in libs:
    ctypes.CDLL(str(lib))
from content_free_logging import install_or_exit
install_or_exit()

print(json.dumps({
    "revision": rev,
    "sglang": version,
    "module": sglang.__file__,
    "native_server": _server.__file__,
    "pearl_dlopen": [p.name for p in libs],
    "packages": {p: metadata.version(p) for p in ("torch", "flashinfer-python", "flashinfer-jit-cache-sm120f",
                                                 "cryptography", "jsonschema", "deepseek-recipe")},
}, indent=2))
