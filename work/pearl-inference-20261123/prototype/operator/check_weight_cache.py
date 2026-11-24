#!/usr/bin/env python3
"""Check cache generations and measure current-activation prep on CPU."""
import argparse
import ctypes as C
import hashlib
import json
import random
import statistics
import time
from contextlib import ExitStack
from pearl_operator import Operator


K = 2048
BITS = 0x1E400000
TO_INT7 = bytes(((i % 129) - 64) % 256 for i in range(256))


def signals(rng, length):
    return bytearray(rng.randbytes(length).translate(TO_INT7))


def ptr(buffer):
    return C.addressof(C.c_byte.from_buffer(buffer))


def header(stamp):
    return bytes.fromhex("00" * 4 + "01" * 32 + "02" * 32) + stamp.to_bytes(4, "little") + BITS.to_bytes(4, "little")


def compact_words(a, b, na, nb, m, n):
    """Independent two-by-16 ticket construction for small regression cases."""
    def mixed(values, noise):
        return [int(x)+int(y) for x,y in zip(memoryview(values).cast("b"),memoryview(noise).cast("b"))]
    ma, mb = mixed(a, na), mixed(b, nb)
    words = []
    for row in range(0, m, 2):
        for col in range(0, n, 16):
            transcript = [0] * 16
            for r in range(row,row+2):
                for c in range(col,col+16):
                    av, bv = r*K, c*K
                    dot = 0
                    for cp in range(16):
                        begin = cp*128
                        dot += sum(x*y for x,y in zip(ma[av+begin:av+begin+128],mb[bv+begin:bv+begin+128]))
                        transcript[cp] ^= dot & 0xffffffff
            words.extend(transcript)
    return (C.c_uint32 * len(words))(*words)


def cache_checks(op, baseline):
    # Twenty tickets at M=2 keep non-saturating positive/negative probes easy.
    rng = random.Random(2906)
    n = 320
    bt = signals(rng, n*K)
    nb = bytearray(n*K)
    m = 4
    a = signals(rng,m*K)
    na = bytearray(m*K)
    fresh_nb = bytearray(n*K)
    fresh_na = bytearray(m*K)
    checks = []
    retained = []
    first_b_noise = None
    prior_words = None
    with ExitStack() as scope:
        for step in ("first", "fresh_A", "smaller_M", "new_header", "replacement_weight"):
            if step == "first":
                job = header(3333)
                w = scope.enter_context(op.prepare_weight(job,ptr(bt),ptr(nb),n,True))
                first_b_noise = bytes(nb)
            elif step == "fresh_A":
                a[0] = 1 if a[0] == 0 else 0
            elif step == "smaller_M":
                m = 2
            else:
                w.close()  # previous callbacks still have their own reference
                if step == "new_header":
                    job = header(3334)
                else:
                    bt[n*K-1] = 1 if bt[n*K-1] == 0 else 0
                old_nb = bytes(nb)
                w = scope.enter_context(op.prepare_weight(job,ptr(bt),ptr(nb),n,True))
                assert bytes(nb) != old_nb, "B stayed constant across rotation"
            cached = scope.enter_context(w.prepare(ptr(a),ptr(na),m))
            full = scope.enter_context(baseline.prepare(job,ptr(a),ptr(bt),ptr(fresh_na),ptr(fresh_nb),m,n,True))
            assert na[:m*K] == fresh_na[:m*K] and nb == fresh_nb
            if step in ("fresh_A","smaller_M"):
                assert bytes(nb) == first_b_noise
            words = compact_words(a,bt,na,nb,m,n)
            payload = bytes(words)
            if prior_words is not None:
                assert payload != prior_words, "expected a distinct transcript"
            prior_words = payload
            addr = C.addressof(words)
            stats = cached.scan(addr,BITS)
            assert stats == full.scan(addr,BITS) and 0 < stats["hits"] < stats["tickets"]
            ticket = stats["first_hit"]
            expected = full.proof(ticket,BITS)
            assert cached.proof(ticket,BITS) == expected
            retained.append((cached,ticket,expected))
            checks.append(dict(case=step, m=m, tickets=stats["tickets"], hits=stats["hits"],
                               proof_sha256=hashlib.sha256(expected).hexdigest()))
        w.close()
        for cached,ticket,expected in retained:
            assert cached.proof(ticket,BITS) == expected, "deferred proof changed after rotation"
    return checks


def benchmark(op, baseline, m, n, compact, repeats, warmup):
    rng = random.Random(8273)
    b, a = signals(rng,n*K), signals(rng,m*K)
    na, nb = bytearray(m*K), bytearray(n*K)
    h = header(765)
    def clock_ms():
        return time.perf_counter_ns()/1e6
    start = clock_ms()
    w = op.prepare_weight(h,ptr(b),ptr(nb),n,compact)
    init = clock_ms()-start
    cached_ms, full_ms = [], []
    with w:
        for trial in range(repeats+warmup):
            # Each call sees new activation bytes to prevent hidden activation cache hits.
            a[0] = (trial % 64)
            for mode in (("full","cached") if trial%2 else ("cached","full")):
                start = clock_ms()
                ref = (w.prepare(ptr(a),ptr(na),m) if mode=="cached" else
                       baseline.prepare(h,ptr(a),ptr(b),ptr(na),ptr(nb),m,n,compact))
                elapsed = clock_ms()-start
                ref.close()
                if trial >= warmup:
                    (cached_ms if mode=="cached" else full_ms).append(elapsed)
    def metrics(samples):
        return dict(median_ms=statistics.median(samples),
                    p95_ms=sorted(samples)[(95*len(samples)+99)//100-1],
                    max_ms=max(samples), samples_ms=samples)
    return dict(m=m,n=n,k=K,compact=compact,weight_init_ms=init,warmups=warmup,
                cached=metrics(cached_ms),full=metrics(full_ms))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build")
    parser.add_argument("--baseline-build",help="Optional previous library implementing full preparation")
    parser.add_argument("--repeats",type=int,default=32)
    parser.add_argument("--warmup",type=int,default=5)
    args=parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("expected positive repeats and nonnegative warmup")
    op=Operator(args.build)
    baseline=Operator(args.baseline_build) if args.baseline_build else op
    print(json.dumps(dict(
        scope="Host wall time from entering Python prepare until the reference handle and host noise are ready; excludes weight upload, GPU work, and teardown.",
        regression_cases=cache_checks(op,baseline),
        timings=[benchmark(op,baseline,m,5120,c,args.repeats,args.warmup) for m,c in [(6,True),(2048,False)]]),
        indent=2))

if __name__ == "__main__":
    main()
