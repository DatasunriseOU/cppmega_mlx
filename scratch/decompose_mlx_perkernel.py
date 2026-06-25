"""DECISIVE decomposition of the per-kernel MLX-route overhead for the chunked
path-c backward. memguard 70. NO fabrication.

Goal: resolve whether the dominant per-kernel cost is
  (a) PER-COMMAND-BUFFER-COMMIT / boundary (batchable: collapses when N kernels
      share one mx.eval / one MLX command buffer), or
  (b) PER-KERNEL-EXECUTION inflation (lazy alloc, graph materialization, encode,
      device sync) that scales linearly with kernel count and does NOT collapse.

Method (all on the LIVE MLX native-primitive bridge, the path that actually runs
since mx.metal._current_command_buffer is ABSENT on this build):

  E1  Dispatch-count scaling: run the SAME single tvm_ffi kernel N in {1,2,4,8}
      times chained into ONE mx.eval. Fit time = intercept + slope*N.
        slope    = marginal per-dispatch cost (encode+boundary+launch enqueue)
        intercept= fixed per-eval cost (commit + final device sync + py overhead)
      If slope dominates -> per-kernel-intrinsic. If intercept dominates and a
      single eval already batches -> the floor is one commit, already amortized.

  E2  Boundary marginal: same kernel x N, FORCE_COMMAND_BUFFER_BOUNDARY on vs off.
      Measures the cost of the EXTRA determinism boundary (newSharedEvent +
      signal/wait) per dispatch. The producer-ordering boundary is unconditional;
      FORCE adds a 2nd. Delta/​N = per-boundary newSharedEvent+enqueue cost.

  E3  eval/commit floor: time mx.eval() on a trivial 1-op MLX graph and on the
      kernel-output graph WITHOUT vs WITH a trailing device sync, to separate
      commit-enqueue from GPU-wait.
"""
from __future__ import annotations
import os, sys, threading, time
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
        if r > _PEAK: _PEAK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g, daemon=True).start()

FORCE = os.environ.get("FORCE_BND", "1")
os.environ["TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY"] = FORCE
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import mlx.core as mx
from tilelang.jit.adapter import _mlx_tvm_ffi as bridge
from cppmega_mlx.nn._tilelang.mamba3_path_c import _chunked_bwd_b0_kernel

# B0 (chunk_precompute_bwd) is a single, self-contained tvm_ffi kernel: 9 inputs,
# 4 owner outputs. We reuse it as the "unit dispatch" for scaling experiments.
b, s, chunk, G, heads, headdim, state = 1, 128, 64, 1, 128, 64, 64
nch = s // chunk

rng = np.random.RandomState(0)
def f16(*sh, sc=0.1):
    return mx.array((rng.randn(*sh) * sc).astype(np.float16))
def f32(*sh, sc=0.1):
    return mx.array((rng.randn(*sh) * sc).astype(np.float32))

# B0 ABI param shapes (from prim.buffer_map; nch=2 here):
#  0 dstates      [b, nch, heads, headdim, state]  f32
#  1 dinp_diag    [b, nch, heads, headdim, state]  f32
#  2 dA_cumsum_y  [b, heads, nch, chunk]           f32
#  3 dA_cumsum_tl [b, heads, nch, chunk]           f32
#  4 dA_cumsum    [b, heads, nch, chunk]           f16
#  5 x            [b, s, heads, headdim]           f16
#  6 B            [b, s, G, state]                 f16
#  7 dt           [b, heads, nch, chunk]           f16
#  8 A            [heads]                          f16
# EXACT adapter ABI (printed from prim_func.buffer_map):
#  0 dstates      [1, 2, 128, 64, 64]  f32
#  1 dinp_diag    [1, 128, 128, 64, 64] f32
#  2 dA_cumsum_y  [1, 128, 2, 64]      f32
#  3 dA_cumsum_tl [1, 128, 2, 64]      f32
#  4 dA_cumsum    [1, 128, 2, 64]      f16
#  5 x            [1, 128, 128, 64]    f16
#  6 B            [1, 128, 1, 64]      f16
#  7 dt           [1, 128, 2, 64]      f16
#  8 A            [128]                f16
dstates = f32(b, nch, heads, headdim, state)
dinp_diag = f32(b, s, heads, headdim, state)
dA_y = f32(b, s, nch, headdim)
dA_tail = f32(b, s, nch, headdim)
dA16 = f16(b, s, nch, headdim)
x16 = f16(b, s, heads, headdim)
B16 = f16(b, s, G, state)
dt_k = f16(b, s, nch, headdim)
A_head16 = f16(heads)
args = (dstates, dinp_diag, dA_y, dA_tail, dA16, x16, B16, dt_k, A_head16)
for a in args:
    mx.eval(a)

k_b0 = _chunked_bwd_b0_kernel(b, s, chunk, G, heads, headdim, state)

def run_n(n):
    """Chain n independent B0 dispatches into one graph, eval once, sync."""
    outs = []
    for _ in range(n):
        o = k_b0(*args)  # returns list/tuple of 4 owner-output arrays
        outs.extend(o if isinstance(o, (list, tuple)) else [o])
    mx.eval(*outs)
    return outs

def med_us(fn, n_it=21, warm=8):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n_it):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(ts, np.float64)) * 1e6)

print(f"=== memguard70 ON  FORCE_BND={FORCE}  mlx={mx.__version__} ===")
bridge.reset_debug_state()
# sanity: one run, check launch count
_ = run_n(1)
ds = dict(bridge.debug_state())
print(f"[sanity] 1 dispatch -> launches={ds['launches']} "
      f"direct_pipeline={ds['direct_pipeline_launches']} "
      f"force_boundary={ds['force_command_buffer_boundary_enabled']}")

scaling = {}
for n in (1, 2, 4, 8):
    scaling[n] = med_us(lambda n=n: run_n(n))
    print(f"[E1] N={n:2d} dispatches/eval : {scaling[n]:9.1f} us  "
          f"({scaling[n]/n:8.1f} us/dispatch)")

# linear fit time = a + b*N
ns = np.array(sorted(scaling), float)
ys = np.array([scaling[int(n)] for n in ns], float)
A = np.vstack([np.ones_like(ns), ns]).T
(coef, *_rest) = np.linalg.lstsq(A, ys, rcond=None)
intercept, slope = float(coef[0]), float(coef[1])
print(f"[E1-fit] time(us) = {intercept:.1f} (fixed/eval) + {slope:.1f} * N (per-dispatch)")
print(f"[E1-fit] => per-dispatch marginal = {slope:.1f} us ; fixed eval floor = {intercept:.1f} us")

# E3: pure eval/commit floor on a trivial op (no tvm_ffi)
ta = mx.array(np.zeros((1024,), np.float32))
def trivial():
    y = ta + 1.0
    mx.eval(y)
floor = med_us(trivial)
print(f"[E3] trivial 1-op mx.eval floor: {floor:.1f} us")

print("RESULT_JSON " + __import__("json").dumps({
    "force_bnd": FORCE,
    "scaling_us": scaling,
    "per_dispatch_slope_us": slope,
    "fixed_eval_intercept_us": intercept,
    "trivial_eval_floor_us": floor,
    "peak_rss_kb": _PEAK,
}))
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
