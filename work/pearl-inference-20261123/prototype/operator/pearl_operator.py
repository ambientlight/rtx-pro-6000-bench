"""ctypes access for the SM120 prototype. Pointer arguments are integer addresses.

Caller owns contiguous buffers and the CUDA stream. This module only exposes the
diagnostic host oracle and asynchronous CUDA launch. See pearl_wo_b.h for sizes.
"""
import ctypes as C
from pathlib import Path
from threading import Lock


def _bind(lib, name, args, result):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, result
    return fn


class Operator:
    def __init__(self, build_dir):
        root = Path(build_dir).resolve()
        ref = C.CDLL(str(root / "libpearl_reference.so"))
        gpu = C.CDLL(str(root / "libpearl_wo_b.so"))
        p, i, u, z = C.c_void_p, C.c_int, C.c_uint32, C.c_size_t
        self._new = _bind(ref, "pearl_ref_new", [p]*5 + [i]*3, p)
        self._weight_new = None
        if hasattr(ref, "pearl_weight_new"):
            self._weight_new = _bind(ref, "pearl_weight_new", [p,p,p,i,i], p)
            self._weight_prepare = _bind(ref, "pearl_weight_prepare", [p,p,p,i], p)
            self._weight_free = _bind(ref, "pearl_weight_free", [p], None)
        self._free = _bind(ref, "pearl_ref_free", [p], None)
        self._scan = _bind(ref, "pearl_ref_scan", [p,p,z,u,p], i)
        self._proof = _bind(ref, "pearl_ref_proof", [p,z,u,p,z], C.c_ssize_t)
        self._launch = _bind(gpu, "pearl_wo_b_i7", [p]*6 + [i]*4 + [p], i)
        self._plain = (_bind(gpu, "pearl_wo_b_i7_plain", [p]*3 + [i]*3 + [p], i)
                       if hasattr(gpu, "pearl_wo_b_i7_plain") else None)

    def prepare(self, header, a_host, bt_host, noise_a_host, noise_bt_host, m, n, compact):
        """Generate job-bound noise into the caller's two signed-byte host buffers."""
        if len(header) != 76:
            raise ValueError("expected a 76-byte Pearl header")
        data = C.create_string_buffer(header, len(header))
        ctx = self._new(data, a_host, bt_host, noise_a_host, noise_bt_host, m, n, int(compact))
        if not ctx:
            raise ValueError("invalid operands, header or proof dimensions")
        return Reference(self, ctx, m // 2 * (n // (16 if compact else 64)))

    def prepare_weight(self, header, bt_host, noise_bt_host, n, compact):
        """Commit this exact weight shard and fill its reusable B noise on the host."""
        if self._weight_new is None:
            raise RuntimeError("rebuild libpearl_reference.so for weight caching")
        if len(header) != 76:
            raise ValueError("expected a 76-byte Pearl header")
        packed = C.create_string_buffer(header, len(header))
        ctx = self._weight_new(packed, bt_host, noise_bt_host, n, int(compact))
        if not ctx:
            raise ValueError("invalid weights, header or proof dimensions")
        return Weight(self, ctx, n, compact)

    def launch(self, a, bt, noise_a, noise_bt, output, words, m, n, compact, stream=0):
        """Enqueue on the given CUDA stream. The caller handles stream completion."""
        rc = self._launch(a, bt, noise_a, noise_bt, output, words, m, n, 2048, int(compact), stream)
        if rc:
            raise RuntimeError(f"CUDA launch error {rc}")

    def launch_plain(self, a, bt, out, m, n, stream=0):
        """Enqueue an unmined signed integer projection with exactly m real input rows."""
        if self._plain is None:
            raise RuntimeError("rebuild libpearl_wo_b.so for the plain INT7 projection")
        rc = self._plain(a, bt, out, m, n, 2048, stream)
        if rc:
            raise RuntimeError(f"CUDA plain projection launch error {rc}")


class Weight:
    """Owner of one immutable committed weight. References outlive this handle."""
    def __init__(self, operator, ctx, n, compact):
        self.operator, self.ctx = operator, ctx
        self.n, self.compact = n, compact
        self._lock = Lock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        with self._lock:
            if self.ctx:
                self.operator._weight_free(self.ctx)
                self.ctx = None

    def prepare(self, a_host, noise_a_host, m):
        """Commit the current A and generate its job-, shape- and weight-bound noise."""
        # ctypes releases the GIL. Protect a simultaneous close or cache rotation.
        # The returned Reference holds its own Rust Arc independently of this lock.
        with self._lock:
            if not self.ctx:
                raise ValueError("weight handle is closed")
            ctx = self.operator._weight_prepare(self.ctx, a_host, noise_a_host, m)
        if not ctx:
            raise ValueError("invalid activation or proof dimensions")
        return Reference(self.operator, ctx, (m // 2) * (self.n // (16 if self.compact else 64)))


class Reference:
    def __init__(self, operator, ctx, count):
        self.operator, self.ctx, self.ticket_count = operator, ctx, count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.ctx:
            self.operator._free(self.ctx)
            self.ctx = None

    def scan(self, words_host, share_nbits):
        """Hash all GPU transcripts, check sampled full transcripts and official verdicts."""
        stats = (C.c_uint64 * 4)()
        rc = self.operator._scan(self.ctx, words_host, self.ticket_count, share_nbits, stats)
        if rc:
            raise RuntimeError(f"Pearl reference mismatch ({rc})")
        hits, first_hit, first_miss, checked = stats
        return dict(tickets=self.ticket_count, hits=hits,
                    first_hit=first_hit-1 if first_hit else None,
                    first_miss=first_miss-1 if first_miss else None,
                    oracle_checks=checked)

    def proof(self, ticket, share_nbits):
        """Return a verified winning v3 bincode witness. No pool I/O."""
        size = self.operator._proof(self.ctx, ticket, share_nbits, None, 0)
        if size <= 0:
            raise ValueError(f"invalid Pearl proof or target ({size})")
        buf = C.create_string_buffer(size)
        if self.operator._proof(self.ctx, ticket, share_nbits, buf, size) != size:
            raise RuntimeError("failed to serialize the winning Pearl proof")
        return buf.raw
