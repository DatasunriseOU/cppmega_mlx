"""Time the LIVE production VJP (now wired to the B2+B1 fused kernel) end-to-end via
mx.vjp at nam56r, under memguard 70. Compares against the search's measured
3-dispatch baseline (~11043us) and MSL 5002us. NO fabrication.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

_LIM = 70 * 1024 * 1024
_PEAK = 0


def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _g():
    global _PEAK
    while True:
        r = _rss()
        if r > _PEAK:
            _PEAK = r
        if r > _LIM:
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_g, daemon=True).start()
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import mlx.core as mx
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd,
)

b, s, H, P, N = 1, 128, 128, 64, 64
rng = np.random.RandomState(0)


def f32(*sh, sc=0.1):
    return mx.array((rng.randn(*sh) * sc).astype(np.float32))


x = f32(b, s, H, P); B = f32(b, s, H, N); C = f32(b, s, H, N)
z = f32(b, s, H, P, sc=0.5)
A_head = (-rng.rand(H)).astype(np.float32)
A = mx.array(np.broadcast_to(A_head[None, None, :], (b, s, H)).copy())
dt = mx.array((rng.rand(b, s, H) * 0.05).astype(np.float32))
D = mx.array((rng.randn(H)).astype(np.float32))
h0 = f32(b, H, P, N)
cot = mx.array((rng.randn(b, s, H, P) * 0.1).astype(np.float32))
primals = (x, B, C, z, A, dt, D, h0)


def fwd(x_, B_, C_, z_, A_, dt_, D_, h0_):
    return mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(
        x_, B_, C_, z_, A_, dt_, D_, h0_)[0]


# warmup
for _ in range(8):
    _, gr = mx.vjp(fwd, primals, (cot,))
    mx.eval(*gr)

N_IT = 31
times = []
for _ in range(N_IT):
    t0 = time.perf_counter()
    _, gr = mx.vjp(fwd, primals, (cot,))
    mx.eval(*gr)
    times.append(time.perf_counter() - t0)

med = float(np.median(np.asarray(times, np.float64)) * 1e6)
print(f"LIVE VJP (fwd+bwd, B2+B1 fused wired) median = {med:.1f}us "
      f"(min {min(times)*1e6:.1f} max {max(times)*1e6:.1f}) over {N_IT}")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("RC=0")
