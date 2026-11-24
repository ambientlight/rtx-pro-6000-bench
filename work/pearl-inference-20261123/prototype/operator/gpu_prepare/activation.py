"""SM120 resident activation preparation and asynchronous Rust proof openings."""
import ctypes as C
from pathlib import Path
from weakref import WeakKeyDictionary

from pearl_operator import Reference, _bind


class GpuPrepare:
    def __init__(self, build, operator):
        root = Path(build).resolve()
        gpu = C.CDLL(str(root / "libpearl_prepare.so"))
        rust = C.CDLL(str(root / "libpearl_reference.so"))
        p, i, z = C.c_void_p, C.c_int, C.c_size_t
        self.operator = operator
        self.scratch_bytes = _bind(gpu, "pearl_activation_scratch", [i], z)
        self._launch = _bind(gpu, "pearl_activation_prepare", [p]*7 + [z, i, i, p], i)
        self._material = _bind(rust, "pearl_weight_material", [p, p], i)
        self._retain = _bind(rust, "pearl_weight_retain", [p], p)
        self._open = _bind(rust, "pearl_activation_open", [p, p, i, p], p)
        self.reference_material = _bind(rust, "pearl_ref_activation_material", [p, p], i)
        self._jobs = WeakKeyDictionary()

    def material(self, weight, device):
        """Stage 64 immutable bytes once per job, pattern and shard."""
        import torch
        data = self._jobs.get(weight)
        if data is None:
            raw = bytearray(64)
            buf = (C.c_uint8 * len(raw)).from_buffer(raw)
            with weight._lock:
                if self._material(weight.ctx, buf):
                    raise ValueError("cannot export retired weight commitment")
            data = torch.frombuffer(raw, dtype=torch.uint8).to(device)
            self._jobs[weight] = data
        return data

    def launch(self, q, material, meta, noise, scratch, n, stream):
        """All args are preallocated contiguous GPU tensors on the serving device.

        material is job_key || root_b, meta is root_a || a_seed. q/noise use
        M x 2048 signed bytes, scratch has at least scratch_bytes(M) bytes.
        """
        rc = self._launch(q.data_ptr(), material.data_ptr(), material.data_ptr()+32,
                          meta.data_ptr(), meta.data_ptr()+32, noise.data_ptr(), scratch.data_ptr(),
                          scratch.numel(), q.shape[0], n, stream.cuda_stream)
        if rc:
            raise RuntimeError(f"Pearl GPU activation preparation launch returned {rc}")

    def prepare(self, q, weight, stream):
        import torch
        material = self.material(weight, q.device)
        meta = torch.empty(64, dtype=torch.uint8, device=q.device)
        noise = torch.empty_like(q)
        scratch = torch.empty(self.scratch_bytes(len(q)), dtype=torch.uint8, device=q.device)
        material.record_stream(stream)
        self.launch(q, material, meta, noise, scratch, weight.n, stream)
        with weight._lock:
            owner = self._retain(weight.ctx)
        if not owner:
            raise ValueError("weight commitment was retired before proof capture")
        count = len(q) // 2 * (weight.n // (16 if weight.compact else 64))
        return noise, DeferredOpening(self, owner, q, meta, count)


class DeferredOpening:
    """Own an immutable job until the copy stream finishes and the proof worker opens A."""
    def __init__(self, backend, owner, q, meta, count):
        self.backend, self.owner, self.ticket_count = backend, owner, count
        self.q, self.meta = q, meta
        self.host_q = self.host_meta = self.ref = None

    def stage(self, stream):
        import torch
        self.host_q = torch.empty(self.q.shape, dtype=torch.int8, pin_memory=True)
        self.host_meta = torch.empty(64, dtype=torch.uint8, pin_memory=True)
        self.host_q.copy_(self.q, non_blocking=True)
        self.host_meta.copy_(self.meta, non_blocking=True)
        self.q.record_stream(stream)
        self.meta.record_stream(stream)
        self.q = self.meta = None

    def scan(self, words_host, bits):
        # Caller synchronized the copy-stream completion event on its proof thread.
        if self.ref is None:
            ctx = self.backend._open(self.owner, self.host_q.data_ptr(), len(self.host_q),
                                     self.host_meta.data_ptr())
            if not ctx:
                raise ValueError("GPU activation commitment failed independent Rust verification")
            self.ref = Reference(self.backend.operator, ctx, self.ticket_count)
            self.host_q = self.host_meta = None
        return self.ref.scan(words_host, bits)

    def proof(self, ticket, bits):
        return self.ref.proof(ticket, bits)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.ref is not None:
            self.ref.close()
        if self.owner:
            self.backend.operator._weight_free(self.owner)
            self.owner = None
