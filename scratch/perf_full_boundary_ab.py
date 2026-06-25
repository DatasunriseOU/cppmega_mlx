"""FULL chunked-bwd wrapper boundary A/B, INTERLEAVED (on,off,on,off) to cancel
drift. This captures the boundary's REAL e2e cost: forcing the cmd-buffer
boundary serializes the wrapper's fp16 casts + 3 kernel dispatches + fp32 dD
recompute (no MLX pipeline overlap). Per-isolated-kernel A/B misses this because
each isolated kernel is eval'd alone. memguard 70. base 5d5c878.
TIMING-ONLY probe; production ships boundary=1 (deterministic).
"""
import os, sys, threading, time
_LIM = 70 * 1024 * 1024; _PEAK = 0
def _g():
    global _PEAK
    import resource
    while True:
        r = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
        if r > _PEAK: _PEAK = r
        if r > _LIM: os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
import numpy as np
import mlx.core as mx
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_fwd_path_c, _mamba3_chunked_backward_path_c,
    _mamba3_chunked_fwd_intermediates_path_c)
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal

SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 128
b, seq, H, P, N, chunk = 1, SEQ, 128, 64, 64, 64
nchunks = seq // chunk
rng = np.random.RandomState(0)
def f32(*s, sc=0.1): return mx.array((rng.randn(*s) * sc).astype(np.float32))
x = f32(b, seq, H, P); B = f32(b, seq, H, N); C = f32(b, seq, H, N)
z = f32(b, seq, H, P, sc=0.5)
A_head = (-rng.rand(H)).astype(np.float32)
A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
D = mx.array((rng.randn(H)).astype(np.float32)); h0 = f32(b, H, P, N)
cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))
cb, dA_cumsum, prev_states = _mamba3_chunked_fwd_intermediates_path_c(x, B, C, A, dt, h0)
mx.eval(cb, dA_cumsum, prev_states)
y_lane, _ = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0); mx.eval(y_lane)
y16 = mx.contiguous(y_lane.astype(mx.float16)); mx.eval(y16)
ENV = "TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY"

def full():
    return _mamba3_chunked_backward_path_c(
        cot_y, x, B, C, z, A, dt, D, h0,
        cb=cb, dA_cumsum=dA_cumsum, prev_states=prev_states, y=y16)

def msl():
    return mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="metal")

on, off = [], []
for _ in range(10):
    os.environ[ENV] = "1"; mx.eval(*full())
    os.environ[ENV] = "0"; mx.eval(*full())
for _ in range(30):
    os.environ[ENV] = "1"
    t0 = time.perf_counter(); mx.eval(*full()); on.append(time.perf_counter()-t0)
    os.environ[ENV] = "0"
    t0 = time.perf_counter(); mx.eval(*full()); off.append(time.perf_counter()-t0)
os.environ[ENV] = "1"
mon = float(np.median(on))*1e6; moff = float(np.median(off))*1e6
mmsl = []
for _ in range(8): mx.eval(*msl())
for _ in range(25):
    t0 = time.perf_counter(); mx.eval(*msl()); mmsl.append(time.perf_counter()-t0)
mmsl = float(np.median(mmsl))*1e6
print(f"\n=== FULL wrapper boundary A/B (interleaved) S={SEQ} ===")
print(f"  FULL boundary=1 (prod/det) = {mon:9.1f} us")
print(f"  FULL boundary=0 (probe)    = {moff:9.1f} us")
print(f"  boundary e2e cost (serialization) = {mon-moff:+.1f} us ({100*(mon-moff)/moff:+.1f}%)")
print(f"  MSL target                 = {mmsl:9.1f} us")
print(f"  FULL_det / MSL = {mon/mmsl:.3f}x   FULL_nondet / MSL = {moff/mmsl:.3f}x")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("FULL_BOUNDARY_AB_DONE")
