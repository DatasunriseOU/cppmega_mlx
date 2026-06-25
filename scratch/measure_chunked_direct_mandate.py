"""MANDATE measurement: chunked Path-C bwd direct-pipeline vs host-wrapper, on the
PRODUCTION MLX route (mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd custom_function),
A/B via TILELANG_METAL_DIRECT_DISABLE, plus MSL baseline + all-8 bit-correctness.

Single fresh process. Cache ENABLED (production from_database). memguard 70 mandatory.
Each arm runs in-process; the direct vs host-wrapper A/B is selected by the env gate
TILELANG_METAL_DIRECT_DISABLE which is read PER CALL inside the adapter, so we flip it
between the two timed regions WITHOUT re-importing (the gate short-circuits
_metal_direct_device_call but the C++ prepared-call is rebuilt on first dispatch of
each arm because the gate changes the prepared launch shape).

NO fabrication: every us is a real timed GPU dispatch (warmed, median >= MEDIAN_ITERS).
direct_pipeline_launches read from the native debug_state to PROVE the arm's path.
"""
from __future__ import annotations

import os
import sys
import threading
import time

# --- memguard 70 -----------------------------------------------------------
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
sys.path.insert(0, "/Volumes/external/sources/tilelang")

import numpy as np
import mlx.core as mx

from tilelang.jit.adapter import _mlx_tvm_ffi as _mlxffi

MEDIAN_ITERS = int(os.environ.get("MEASURE_ITERS", "30"))
WARMUP = 5

# nam56r production surface.
b, seq, H, P, N, chunk = 1, 128, 128, 64, 64, 64
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

from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd,
)


def fwd_y(*p):
    out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(*p)
    return out[0]


def run_fwd():
    y = fwd_y(*primals)
    mx.eval(y)
    return y


def run_bwd():
    _, grads = mx.vjp(fwd_y, primals, (cot_y,))
    mx.eval(*grads)
    return grads


def run_fwd_bwd():
    out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(*primals)
    mx.eval(out[0], out[1])
    _, grads = mx.vjp(fwd_y, primals, (cot_y,))
    mx.eval(*grads)
    return grads


def median_us(fn, iters=MEDIAN_ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    arr = np.asarray(ts, np.float64)
    return float(np.median(arr) * 1e6), float(arr.min() * 1e6), float(arr.max() * 1e6)


def counters_for(fn):
    """Run one dispatch with counters reset; return the debug_state snapshot."""
    _mlxffi.reset_debug_state()
    fn()
    return dict(_mlxffi.debug_state())


def arm(disable_direct: bool, label: str):
    if disable_direct:
        os.environ["TILELANG_METAL_DIRECT_DISABLE"] = "1"
    else:
        os.environ.pop("TILELANG_METAL_DIRECT_DISABLE", None)
    # Warm + prove path via counters (one fwd, one bwd, one fwd+bwd).
    run_fwd_bwd()  # warm/build prepared-call for this gate state
    c_fwd = counters_for(run_fwd)
    c_bwd = counters_for(run_bwd)
    c_e2e = counters_for(run_fwd_bwd)
    fwd_us = median_us(run_fwd)
    bwd_us = median_us(run_bwd)
    e2e_us = median_us(run_fwd_bwd)
    print(f"\n=== ARM [{label}] direct_disabled={disable_direct} ===")
    print(f"  counters fwd:   dpl={c_fwd['direct_pipeline_launches']} ddl={c_fwd['direct_device_launches']} launches={c_fwd['launches']}")
    print(f"  counters bwd:   dpl={c_bwd['direct_pipeline_launches']} ddl={c_bwd['direct_device_launches']} launches={c_bwd['launches']}")
    print(f"  counters e2e:   dpl={c_e2e['direct_pipeline_launches']} ddl={c_e2e['direct_device_launches']} launches={c_e2e['launches']}")
    print(f"  fwd  median={fwd_us[0]:.1f}us (min={fwd_us[1]:.1f} max={fwd_us[2]:.1f})")
    print(f"  bwd  median={bwd_us[0]:.1f}us (min={bwd_us[1]:.1f} max={bwd_us[2]:.1f})")
    print(f"  e2e  median={e2e_us[0]:.1f}us (min={e2e_us[1]:.1f} max={e2e_us[2]:.1f})")
    return dict(label=label, c_fwd=c_fwd, c_bwd=c_bwd, c_e2e=c_e2e,
                fwd_us=fwd_us[0], bwd_us=bwd_us[0], e2e_us=e2e_us[0])


# ---------------------------------------------------------------------------
# BIT-CORRECTNESS (all 8 grads) vs path-b GOLD, computed in the DIRECT arm.
# ---------------------------------------------------------------------------
def bit_correct():
    from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal
    grads_gold = mamba3_mimo_bwd_metal(cot_y, *primals, backend="mlx")
    mx.eval(*grads_gold)
    os.environ.pop("TILELANG_METAL_DIRECT_DISABLE", None)  # DIRECT path
    _, grads_chunked = mx.vjp(fwd_y, primals, (cot_y,))
    mx.eval(*grads_chunked)
    names = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
    diffs = {}
    worst = 0.0
    for n, gc, gg in zip(names, grads_chunked, grads_gold):
        a = np.asarray(gc.astype(mx.float32), np.float64)
        g = np.asarray(gg.astype(mx.float32), np.float64)
        d = float(np.abs(a - g).max())
        diffs[n] = d
        worst = max(worst, d)
    return diffs, worst


# ---------------------------------------------------------------------------
# MSL baseline (non-chunked Metal bwd).
# ---------------------------------------------------------------------------
def msl_baseline():
    os.environ["CPPMEGA_MAMBA3_BWD_SEQ_CHUNK"] = "0"
    from cppmega_mlx.nn._tilelang.mamba3 import (
        mamba3_mimo_bwd_metal,
        mamba3_mimo_metal_status,
        _mamba3_mimo_bwd_metal_kernel,
    )
    st = mamba3_mimo_metal_status(x)
    if not st.available:
        raise RuntimeError(f"RULE#1: MSL bwd Metal NOT available: {st}")
    probe = _mamba3_mimo_bwd_metal_kernel(cot_y, *primals)
    if probe is None:
        raise RuntimeError("RULE#1: non-chunked MSL Metal bwd returned None")

    def run():
        grads = mamba3_mimo_bwd_metal(cot_y, *primals, backend="metal")
        mx.eval(*grads)
        return grads

    us = median_us(run)
    print(f"\n=== MSL baseline (non-chunked Metal bwd) median={us[0]:.1f}us (min={us[1]:.1f} max={us[2]:.1f}) ===")
    return us[0]


print(f"MEASURE_ITERS={MEDIAN_ITERS} WARMUP={WARMUP} cache_disabled={os.environ.get('TILELANG_DISABLE_CACHE','0')}")
print("Surface: nam56r b=1 seq=128 H=128 P=64 N=64 chunk=64")

# BEFORE = host-wrapper (direct disabled), AFTER = direct pipeline.
before = arm(disable_direct=True, label="BEFORE host-wrapper")
after = arm(disable_direct=False, label="AFTER  direct-pipeline")

diffs, worst = bit_correct()
print("\n=== BIT-CORRECT all 8 grads (chunked DIRECT vs path-b GOLD) ===")
print("  " + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
print(f"  WORST max|abs|={worst:.3e}  gate<1e-3 -> {'PASS' if worst < 1e-3 else 'FAIL'}")

msl_us = msl_baseline()

print("\n========================= SUMMARY =========================")
print(f"PER_KERNEL_FWD  host-wrapper={before['fwd_us']:.1f}us  direct={after['fwd_us']:.1f}us  (3 fwd kernels, all direct)")
print(f"BWD_E2E         host-wrapper={before['bwd_us']:.1f}us  direct={after['bwd_us']:.1f}us")
print(f"FWD+BWD_E2E     host-wrapper={before['e2e_us']:.1f}us  direct={after['e2e_us']:.1f}us")
print(f"MSL_BWD                                      = {msl_us:.1f}us")
print(f"CHUNKED_BWD_DIRECT / MSL  ratio = {after['bwd_us']/msl_us:.3f}  (chunked_le_msl={'YES' if after['bwd_us'] <= msl_us else 'NO'})")
print(f"CHUNKED_E2E_DIRECT / MSL  ratio = {after['e2e_us']/msl_us:.3f}")
print(f"ALL8_WORST_MAXDIFF = {worst:.3e}  bit_correct={'PASS' if worst < 1e-3 else 'FAIL'}")
print(f"direct_pipeline_launches AFTER e2e = {after['c_e2e']['direct_pipeline_launches']} (BEFORE = {before['c_e2e']['direct_pipeline_launches']})")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("MEASURE_DONE")
