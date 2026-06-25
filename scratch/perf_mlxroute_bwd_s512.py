"""MLX tvm_ffi-route chunked bwd-alone vs MSL at the RECEIPT dims (S=512,
8 chunks), to check whether the ~1.0x receipt holds on the PRODUCTION MLX
route at its own dims (the receipt was measured on torch/mps). memguard 70.
"""
import os
import sys
import threading
import time

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK = 0


def _rss_kb():
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
    mamba3_mimo_fwd_path_c,
    _mamba3_chunked_backward_path_c,
    _mamba3_chunked_fwd_intermediates_path_c,
)
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal

# Receipt dims: S=512 (8 chunks of 64). Keep G==H==128 (parity dispatch surface).
b, seq, H, P, N, chunk = 1, 512, 128, 64, 64, 64
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

ITERS = 20
WARMUP = 8


def med_us(times):
    return float(np.median(np.asarray(times, np.float64))) * 1e6


def time_call(fn, label):
    times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        outs = fn()
        mx.eval(*outs)
        t1 = time.perf_counter()
        if i >= WARMUP:
            times.append(t1 - t0)
    m = med_us(times)
    print(f"  {label:42s} {m:9.1f} us")
    return m


cb, dA_cumsum, prev_states = _mamba3_chunked_fwd_intermediates_path_c(x, B, C, A, dt, h0)
mx.eval(cb, dA_cumsum, prev_states)
y_lane, h_last = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
mx.eval(y_lane, h_last)
y_stash = mx.contiguous(y_lane.astype(mx.float16))
mx.eval(y_stash)

print("\n=== MLX route bwd-alone @ receipt dims S=512 ===")
print(f"dims b={b} seq={seq} H={H} P={P} N={N} chunk={chunk} nchunks={seq//chunk}")


def run_chunked_bwd():
    return _mamba3_chunked_backward_path_c(
        cot_y, x, B, C, z, A, dt, D, h0,
        cb=cb, dA_cumsum=dA_cumsum, prev_states=prev_states, y=y_stash,
    )


def run_msl_bwd():
    return mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="metal")


m_chunked = time_call(run_chunked_bwd, "chunked bwd B2->B1->B0 (MLX route)")
m_msl = time_call(run_msl_bwd, "MSL bwd mamba3_mimo_bwd_metal(metal)")
print(f"  -> MLX-route chunked/MSL @ S=512 = {m_chunked/m_msl:.3f}x")
print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("MLX_BWD_S512_DONE")
