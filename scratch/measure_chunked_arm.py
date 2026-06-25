"""ONE arm of the chunked direct-pipeline mandate measurement, fresh process.

Path selected by env TILELANG_METAL_DIRECT_DISABLE (set before this runs):
  =1  -> BEFORE (host-wrapper TVMFFIFunctionCall for the chunked kernels)
  unset/0 -> AFTER (direct Metal pipeline for the kernels that qualify)

Cache ENABLED (production from_database). memguard 70. Counters PROVE the path.
Prints machine-readable RESULT_* lines for the orchestrator to parse.

Every us is a real timed GPU dispatch, warmed + median over MEASURE_ITERS.
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
sys.path.insert(0, "/Volumes/external/sources/tilelang")

import numpy as np
import mlx.core as mx

from tilelang.jit.adapter import _mlx_tvm_ffi as _mlxffi

ITERS = int(os.environ.get("MEASURE_ITERS", "30"))
WARMUP = 8
ARM = "BEFORE_hostwrapper" if os.environ.get("TILELANG_METAL_DIRECT_DISABLE") == "1" else "AFTER_direct"

b, seq, H, P, N, chunk = 1, 128, 128, 64, 64, 64
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
primals = (x, B, C, z, A, dt, D, h0)

from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd,
)


def fwd_y(*p):
    return mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(*p)[0]


def run_fwd():
    mx.eval(fwd_y(*primals))


def run_bwd():
    _, g = mx.vjp(fwd_y, primals, (cot_y,))
    mx.eval(*g)


def run_e2e():
    out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(*primals)
    mx.eval(out[0], out[1])
    _, g = mx.vjp(fwd_y, primals, (cot_y,))
    mx.eval(*g)


def median_us(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts, np.float64)
    return float(np.median(a) * 1e6), float(a.min() * 1e6), float(a.max() * 1e6)


def throughput_us(build_graph, n_batch=24, reps=8):
    """Amortize Python/MLX host overhead: enqueue n_batch independent graphs,
    mx.eval ALL at once, divide wall time by n_batch. Repeated `reps` times,
    take the median per-dispatch us. This exposes GPU-side throughput rather than
    the per-call host overhead that dominates a single mx.eval."""
    # warmup
    for _ in range(3):
        outs = [build_graph() for _ in range(n_batch)]
        mx.eval(outs)
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [build_graph() for _ in range(n_batch)]
        mx.eval(outs)
        dt_total = time.perf_counter() - t0
        per.append(dt_total / n_batch)
    a = np.asarray(per, np.float64)
    return float(np.median(a) * 1e6), float(a.min() * 1e6), float(a.max() * 1e6)


def build_bwd_graph():
    _, g = mx.vjp(fwd_y, primals, (cot_y,))
    return list(g)


def build_fwd_graph():
    return [fwd_y(*primals)]


# warm + prove the path
run_e2e()
_mlxffi.reset_debug_state()
run_fwd()
c_fwd = dict(_mlxffi.debug_state())
_mlxffi.reset_debug_state()
run_bwd()
c_bwd = dict(_mlxffi.debug_state())
_mlxffi.reset_debug_state()
run_e2e()
c_e2e = dict(_mlxffi.debug_state())

fwd = median_us(run_fwd)
bwd = median_us(run_bwd)
e2e = median_us(run_e2e)

# Amortized GPU-throughput (host overhead removed by batching N graphs / one eval).
fwd_tp = throughput_us(build_fwd_graph)
bwd_tp = throughput_us(build_bwd_graph)

print(f"ARM={ARM} ITERS={ITERS}")
print(f"COUNTERS_FWD dpl={c_fwd['direct_pipeline_launches']} ddl={c_fwd['direct_device_launches']} launches={c_fwd['launches']}")
print(f"COUNTERS_BWD dpl={c_bwd['direct_pipeline_launches']} ddl={c_bwd['direct_device_launches']} launches={c_bwd['launches']}")
print(f"COUNTERS_E2E dpl={c_e2e['direct_pipeline_launches']} ddl={c_e2e['direct_device_launches']} launches={c_e2e['launches']}")
print(f"RESULT_FWD_US median={fwd[0]:.1f} min={fwd[1]:.1f} max={fwd[2]:.1f}")
print(f"RESULT_BWD_US median={bwd[0]:.1f} min={bwd[1]:.1f} max={bwd[2]:.1f}")
print(f"RESULT_E2E_US median={e2e[0]:.1f} min={e2e[1]:.1f} max={e2e[2]:.1f}")
print(f"RESULT_FWD_TP_US median={fwd_tp[0]:.1f} min={fwd_tp[1]:.1f} max={fwd_tp[2]:.1f}")
print(f"RESULT_BWD_TP_US median={bwd_tp[0]:.1f} min={bwd_tp[1]:.1f} max={bwd_tp[2]:.1f}")
print(f"PEAK_RSS_KB={_PEAK}")
print("ARM_DONE")
