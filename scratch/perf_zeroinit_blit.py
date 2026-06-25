"""Isolate the zero-init BLIT cost: recompile B0 with zero_init=[] (timing-only,
NOT correct -- it would race; we eval it only for the blit-cost delta) vs the
production zero_init=[0,1,2,3], same inputs, interleaved. memguard 70. base 5d5c878.

RULE#1: this is a TIMING-ONLY probe of the blit cost. The zero_init=[] build is
NEVER shipped (it produces non-deterministic dx/dB). We discard its outputs.
"""
import os
import sys
import threading
import time

_LIM = 70 * 1024 * 1024
_PEAK = 0


def _g():
    global _PEAK
    import resource
    while True:
        r = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
        if r > _PEAK:
            _PEAK = r
        if r > _LIM:
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_g, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx
import tilelang

from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_fwd_path_c,
    _mamba3_chunked_fwd_intermediates_path_c,
    _chunked_bwd_b2_kernel,
    _chunked_bwd_b1_kernel,
)
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    chunk_precompute_bwd_metal_prim,
)

# force boundary on (production) so the delta isolates ONLY the blit
os.environ["TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY"] = "1"

SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 128
b, seq, H, P, N, chunk = 1, SEQ, 128, 64, 64, 64
G = H
nchunks = seq // chunk
rng = np.random.RandomState(0)


def f32(*shape, s=0.1):
    return mx.array((rng.randn(*shape) * s).astype(np.float32))


x = f32(b, seq, H, P); B = f32(b, seq, H, N); C = f32(b, seq, H, N)
z = f32(b, seq, H, P, s=0.5)
A_head = (-rng.rand(H)).astype(np.float32)
A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
D = mx.array((rng.randn(H)).astype(np.float32))
h0 = f32(b, H, P, N)
cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))

cb, dA_cumsum, prev_states = _mamba3_chunked_fwd_intermediates_path_c(x, B, C, A, dt, h0)
mx.eval(cb, dA_cumsum, prev_states)
y_lane, _ = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
mx.eval(y_lane)
y16 = mx.contiguous(y_lane.astype(mx.float16)); mx.eval(y16)


def f16(a):
    return mx.contiguous(a.astype(mx.float16))


x16 = f16(x); B16 = f16(B); C16 = f16(C); z16 = f16(z); D16 = f16(D)
dout16 = f16(cot_y); dt16 = dt.astype(mx.float16)
dt_k = mx.contiguous(mx.transpose(dt16.reshape(b, nchunks, chunk, H), (0, 3, 1, 2)))
cb16 = cb.astype(mx.float16); dA16 = dA_cumsum.astype(mx.float16)
A_head16 = A[:, 0, :][0].astype(mx.float16); prev32 = prev_states.astype(mx.float32)
mx.eval(x16, B16, C16, z16, D16, dout16, dt_k, cb16, dA16, A_head16, prev32)

k_b2 = _chunked_bwd_b2_kernel(b, seq, chunk, G, H, P, N)
_o2 = k_b2(dout16, cb16, x16, z16, dt_k, dA16, C16, B16, prev32, D16, y16); mx.eval(*_o2)
dchunk = _o2[3]; dinp_diag = _o2[4]; dA_y = _o2[5]
k_b1 = _chunked_bwd_b1_kernel(b, seq, chunk, G, H, P, N)
dh_last0 = mx.zeros((b, H, P, N), dtype=mx.float32); mx.eval(dh_last0)
_o1 = k_b1(dchunk, dA16, dh_last0, prev32); mx.eval(*_o1)
dstates, dh0_m, dA_tail = _o1

# Production B0 (zero_init=[0,1,2,3] via default-atomic detection)
k_b0_prod = k_b2  # placeholder; replaced below
from cppmega_mlx.nn._tilelang.mamba3_path_c import _chunked_bwd_b0_kernel
k_b0_prod = _chunked_bwd_b0_kernel(b, seq, chunk, G, H, P, N)

# Timing-only B0 with zero_init forced EMPTY (NOT shipped; races -> nondeterministic).
prim_noinit = chunk_precompute_bwd_metal_prim(b, seq, chunk, G, H, P, N)
prim_noinit = prim_noinit.with_attr("tilelang_metal_zero_init_output_positions", [])
k_b0_noinit = tilelang.compile(
    prim_noinit,
    target="metal",
    execution_backend="tvm_ffi",
    out_idx=[9, 10, 11, 12],
)


def run_prod():
    return k_b0_prod(dstates, dinp_diag, dA_y, dA_tail, dA16, x16, B16, dt_k, A_head16)


def run_noinit():
    return k_b0_noinit(dstates, dinp_diag, dA_y, dA_tail, dA16, x16, B16, dt_k, A_head16)


def interleaved(fa, fb, iters=40, warmup=12):
    a, bb = [], []
    for _ in range(warmup):
        mx.eval(*fa()); mx.eval(*fb())
    for _ in range(iters):
        t0 = time.perf_counter(); mx.eval(*fa()); a.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); mx.eval(*fb()); bb.append(time.perf_counter() - t0)
    return float(np.median(a)) * 1e6, float(np.median(bb)) * 1e6


print(f"\n=== ZERO-INIT BLIT cost (B0, S={SEQ}, boundary forced ON) ===")
prod, noinit = interleaved(run_prod, run_noinit)
print(f"  B0 zero_init=[0,1,2,3] (prod) = {prod:9.1f} us")
print(f"  B0 zero_init=[]   (probe-only)= {noinit:9.1f} us")
print(f"  zero_init BLIT cost (4 bufs, 8.3MiB) = {prod-noinit:+.1f} us "
      f"({100*(prod-noinit)/noinit:+.1f}% of B0)")
print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("ZEROINIT_BLIT_DONE")
