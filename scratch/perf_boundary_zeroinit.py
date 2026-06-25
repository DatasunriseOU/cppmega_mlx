"""Focused A/B: command-buffer boundary cost + zero-init blit cost, per kernel.
Interleaved on/off sampling to cancel drift. memguard 70. base 5d5c878.

(2) cmd-buffer boundary: time each kernel with FORCE_COMMAND_BUFFER_BOUNDARY
    toggled PER-ITER (on,off,on,off,...) so thermal/scheduler drift cancels.
    TIMING-ONLY probe per RULE#1 (does NOT ship the non-deterministic path).
(2b) zero-init blit: B0 zero-inits ALL 4 outputs (no with_attr, atomic_add in
     src -> bridge zeroes range(4)); B2 zero-inits ONLY [6]=dD (1 tiny buf);
     B1 zeroes []. We confirm which positions each kernel zero-inits via the
     adapter, and report the per-output blit byte volume so the zero-init cost
     is attributable.
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

from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_fwd_path_c,
    _mamba3_chunked_fwd_intermediates_path_c,
    _chunked_bwd_b2_kernel,
    _chunked_bwd_b1_kernel,
    _chunked_bwd_b0_kernel,
)

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
y_lane, h_last = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
mx.eval(y_lane, h_last)
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
k_b1 = _chunked_bwd_b1_kernel(b, seq, chunk, G, H, P, N)
k_b0 = _chunked_bwd_b0_kernel(b, seq, chunk, G, H, P, N)


def run_b2():
    return k_b2(dout16, cb16, x16, z16, dt_k, dA16, C16, B16, prev32, D16, y16)


_o2 = run_b2(); mx.eval(*_o2)
dC_m, dx_b2, dz_m, dchunk, dinp_diag, dA_y, _dD = _o2
dh_last0 = mx.zeros((b, H, P, N), dtype=mx.float32); mx.eval(dh_last0)


def run_b1():
    return k_b1(dchunk, dA16, dh_last0, prev32)


_o1 = run_b1(); mx.eval(*_o1)
dstates, dh0_m, dA_tail = _o1


def run_b0():
    return k_b0(dstates, dinp_diag, dA_y, dA_tail, dA16, x16, B16, dt_k, A_head16)


ENV = "TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY"


def interleaved_ab(fn, label, iters=40, warmup=10):
    on, off = [], []
    # warmup both
    for _ in range(warmup):
        os.environ[ENV] = "1"; mx.eval(*fn())
        os.environ[ENV] = "0"; mx.eval(*fn())
    for _ in range(iters):
        os.environ[ENV] = "1"
        t0 = time.perf_counter(); mx.eval(*fn()); on.append(time.perf_counter() - t0)
        os.environ[ENV] = "0"
        t0 = time.perf_counter(); mx.eval(*fn()); off.append(time.perf_counter() - t0)
    on_med = float(np.median(on)) * 1e6
    off_med = float(np.median(off)) * 1e6
    print(f"  {label:30s} on={on_med:8.1f} off={off_med:8.1f} boundary_cost={on_med-off_med:+8.1f} "
          f"({100*(on_med-off_med)/off_med:+.1f}%)")
    os.environ[ENV] = "1"
    return on_med, off_med


# report which positions each kernel zero-inits (from the compiled adapter)
def zero_positions(kernel, name):
    try:
        ad = kernel.adapter if hasattr(kernel, "adapter") else None
    except Exception:
        ad = None
    pos = None
    for obj in (kernel, getattr(kernel, "adapter", None), getattr(kernel, "_adapter", None)):
        if obj is None:
            continue
        m = getattr(obj, "_metal_zero_init_output_positions", None)
        if callable(m):
            try:
                pos = m()
                break
            except Exception:
                pass
    print(f"  {name} zero_init_positions = {pos}")
    return pos


print(f"\n=== BOUNDARY + ZERO-INIT decomp S={SEQ} (nchunks={nchunks}) ===")
print("\n[zero-init owner-output positions per kernel]")
zero_positions(k_b2, "B2")
zero_positions(k_b1, "B1")
zero_positions(k_b0, "B0")

# byte volume of B0's 4 zero-init outputs vs B2's [6]
print("\n[zero-init blit byte volume]")
_ob0 = run_b0(); mx.eval(*_ob0)
b0_bytes = [int(np.prod(o.shape)) * o.dtype.size for o in _ob0]
print(f"  B0 outputs (all zeroed): shapes/bytes = "
      f"{[ (tuple(o.shape), nb) for o,nb in zip(_ob0,b0_bytes) ]}")
print(f"  B0 total zero-init bytes = {sum(b0_bytes)} ({sum(b0_bytes)/1024:.1f} KiB)")
dD_bytes = H * 4
print(f"  B2 [6]=dD zero-init bytes = {dD_bytes} ({dD_bytes/1024:.3f} KiB)")

print("\n[command-buffer boundary A/B (interleaved, TIMING-ONLY probe)]")
interleaved_ab(run_b2, "B2 boundary on-vs-off")
interleaved_ab(run_b1, "B1 boundary on-vs-off")
interleaved_ab(run_b0, "B0 boundary on-vs-off")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("BOUNDARY_ZEROINIT_DONE")
