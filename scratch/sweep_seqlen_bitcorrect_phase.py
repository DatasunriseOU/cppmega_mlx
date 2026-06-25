"""PHASE sweep: chunked Path-C bwd (path_c_fwd_path_c_bwd) vs MLX fp32 GOLD across
the FULL training seqlen range {128,512,1024,2048,4096} = {2,8,16,32,64} chunks.

Single seqlen per process (argv[1]); the orchestrator launches one fresh process per
seqlen so RSS never compounds. memguard 70 MANDATORY. All 8 grads asserted < 1e-3 vs
the step-by-step pure-MLX fp32 oracle (mamba3_mimo_bwd_metal backend='mlx'), the SAME
GOLD the parity test + b0b1b2 stage tests use. NO fabrication, NO loosened gate.

Prints one machine-parsable line:  SEQLEN_RESULT seq=<S> nchunks=<C> worst=<d> pass=<bool> per=<...>
"""
from __future__ import annotations

import os
import sys
import threading
import time

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK = 0


def _rss_kb() -> int:
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _guard():
    global _PEAK
    while True:
        r = _rss_kb()
        if r > _PEAK:
            _PEAK = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx

from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd,
)
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal

seq = int(sys.argv[1]) if len(sys.argv) > 1 else 128
b, H, P, N, chunk = 1, 128, 64, 64, 64
assert seq % chunk == 0, f"seq {seq} not a multiple of chunk {chunk}"
nchunks = seq // chunk

rng = np.random.RandomState(0)


def f32(*shape, s=0.1):
    return mx.array((rng.randn(*shape) * s).astype(np.float32))


x = f32(b, seq, H, P)
B = f32(b, seq, H, N)
C = f32(b, seq, H, N)
z = f32(b, seq, H, P, s=0.5)
A_head = (-rng.rand(H)).astype(np.float32)
A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
D = mx.array((rng.randn(H)).astype(np.float32))
h0 = f32(b, H, P, N)
cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))

primals = (x, B, C, z, A, dt, D, h0)
names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]


def fwd_y_chunked(x, B, C, z, A, dt, D, h0):
    out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(x, B, C, z, A, dt, D, h0)
    return out[0]


_, grads_chunked = mx.vjp(fwd_y_chunked, primals, (cot_y,))
mx.eval(*grads_chunked)

grads_gold = mamba3_mimo_bwd_metal(cot_y, *primals, backend="mlx")
mx.eval(*grads_gold)


def maxdiff(a, bb):
    a = np.asarray(a.astype(mx.float32), np.float64)
    bb = np.asarray(bb.astype(mx.float32), np.float64)
    return float(np.abs(a - bb).max())


per = {}
worst = 0.0
for nm, gc, gg in zip(names, grads_chunked, grads_gold):
    d = maxdiff(gc, gg)
    per[nm] = d
    worst = max(worst, d)

gate = 1e-3
ok = worst < gate
per_str = ",".join(f"{nm}={per[nm]:.3e}" for nm in names)
print(
    f"SEQLEN_RESULT seq={seq} nchunks={nchunks} worst={worst:.3e} "
    f"pass={ok} per={per_str} peak_rss_gb={_PEAK/1048576:.3f}"
)
sys.exit(0 if ok else 1)
