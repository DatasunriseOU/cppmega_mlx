"""Tighter A/B: baseline 3-dispatch vs B2B1_B0 2-dispatch, interleaved, more iters,
to confirm the measured recovery is stable (not noise). memguard 70.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

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
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import mlx.core as mx
from cppmega_mlx.runtime.path_c_backward_fusion_search import (
    make_fused_backward, _build_eval_inputs,
)
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _mamba3_chunked_backward_path_c, _force_chunked_command_buffer_boundary,
)

_force_chunked_command_buffer_boundary()
DIMS = (1, 128, 64, 128, 128, 64, 64)
inp = _build_eval_inputs(DIMS)

bwd_base = make_fused_backward((("B2",), ("B1",), ("B0",)), DIMS)
bwd_fused = make_fused_backward((("B2", "B1"), ("B0",)), DIMS)


def call(bwd):
    g = bwd(inp["dy"], *inp["primals"], cb=inp["cb"], dA_cumsum=inp["dA_cumsum"],
            prev_states=inp["prev_states"], y=inp["y"])
    mx.eval(*g)


# also time the LIVE wrapper as a 3rd reference
def call_live():
    g = _mamba3_chunked_backward_path_c(
        inp["dy"], *inp["primals"], cb=inp["cb"], dA_cumsum=inp["dA_cumsum"],
        prev_states=inp["prev_states"], y=inp["y"])
    mx.eval(*g)


N = 41
WARM = 10
for _ in range(WARM):
    call(bwd_base); call(bwd_fused); call_live()

t_base, t_fused, t_live = [], [], []
for _ in range(N):
    t0 = time.perf_counter(); call(bwd_base); t_base.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); call(bwd_fused); t_fused.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); call_live(); t_live.append(time.perf_counter() - t0)


def med(t):
    return float(np.median(np.asarray(t, np.float64)) * 1e6)


mb, mf, ml = med(t_base), med(t_fused), med(t_live)
print(f"baseline-reimpl 3-dispatch : {mb:.1f}us  (min {min(t_base)*1e6:.1f})")
print(f"LIVE wrapper    3-dispatch : {ml:.1f}us  (min {min(t_live)*1e6:.1f})")
print(f"B2B1_B0 fused   2-dispatch : {mf:.1f}us  (min {min(t_fused)*1e6:.1f})")
print(f"recovered (live - fused)   : {ml - mf:.1f}us")
print(f"recovered (base - fused)   : {mb - mf:.1f}us")
print(f"fused/base = {mf/mb:.3f}  fused/live = {mf/ml:.3f}")
print(f"PEAK_RSS_KB={_PEAK} memguard70=ON")
print("RC=0")
