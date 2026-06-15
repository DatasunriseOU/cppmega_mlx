"""Fast correctness check (small dims): make_fused_backward(baseline) and
make_fused_backward(B2B1_B0) both match the live _mamba3_chunked_backward_path_c
and the path-b GOLD. Validates the wrapper-body reimplementation + the B2+B1 splice.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import mlx.core as mx

from cppmega_mlx.runtime.path_c_backward_fusion_search import (
    make_fused_backward,
    _build_eval_inputs,
    _maxabs,
    _GRAD_NAMES,
)
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _mamba3_chunked_backward_path_c,
    _force_chunked_command_buffer_boundary,
)

_force_chunked_command_buffer_boundary()

# small surface: b=1,s=128,c=64,G=H=2,P=64,N=16 (2 chunks); per-head A.
dims = (1, 128, 64, 2, 2, 64, 16)
inp = _build_eval_inputs(dims)


def run(bwd):
    g = bwd(inp["dy"], *inp["primals"], cb=inp["cb"], dA_cumsum=inp["dA_cumsum"],
            prev_states=inp["prev_states"], y=inp["y"])
    mx.eval(*g)
    return g


print("=== live wrapper vs GOLD ===")
g_live = run(_mamba3_chunked_backward_path_c)
for nm, gc, gg in zip(_GRAD_NAMES, g_live, inp["grads_gold"]):
    print(f"  {nm}: {_maxabs(gc, gg):.2e}")

print("\n=== make_fused_backward(baseline 3-dispatch) vs GOLD ===")
bwd_base = make_fused_backward((("B2",), ("B1",), ("B0",)), dims)
g_base = run(bwd_base)
worst_base = 0.0
for nm, gc, gg in zip(_GRAD_NAMES, g_base, inp["grads_gold"]):
    d = _maxabs(gc, gg); worst_base = max(worst_base, d)
    print(f"  {nm}: {d:.2e}")
print(f"  WORST baseline-reimpl: {worst_base:.2e}  {'PASS' if worst_base < 1e-3 else 'FAIL'}")

print("\n=== make_fused_backward(B2B1_B0 fused) vs GOLD ===")
bwd_fused = make_fused_backward((("B2", "B1"), ("B0",)), dims)
g_fused = run(bwd_fused)
worst_fused = 0.0
for nm, gc, gg in zip(_GRAD_NAMES, g_fused, inp["grads_gold"]):
    d = _maxabs(gc, gg); worst_fused = max(worst_fused, d)
    print(f"  {nm}: {d:.2e}")
print(f"  WORST B2B1-fused: {worst_fused:.2e}  {'PASS' if worst_fused < 1e-3 else 'FAIL'}")

print("\n=== fused vs baseline-reimpl (should be ~identical) ===")
worst_fb = max(_maxabs(a, b) for a, b in zip(g_fused, g_base))
print(f"  WORST fused-vs-baseline: {worst_fb:.2e}")
print("ALL_PASS" if (worst_base < 1e-3 and worst_fused < 1e-3) else "SOME_FAIL")
