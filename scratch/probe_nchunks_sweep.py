"""PROBE (task wh9qyltot phase): per-grad maxdiff-vs-nchunks sweep of the chunked
Path-C backward vs path-b GOLD. memguard 70. NOT a production edit.

seq in {128,512,1024,2048,4096} -> nchunks {2,8,16,32,64} (chunk=64 fixed prod).
Prints the per-grad maxdiff curve + WORST + PASS/FAIL (<1e-3) per seqlen.
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
from cppmega_mlx.runtime.path_c_backward_fusion_search import (
    _build_eval_inputs, _maxabs, _GRAD_NAMES,
)
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _mamba3_chunked_backward_path_c, _force_chunked_command_buffer_boundary,
)

_force_chunked_command_buffer_boundary()

SEQS = [128, 512, 1024, 2048, 4096]
# nam56r config-exact surface: b=1, H=G=128, P=64, N=64, chunk=64 (the FAILING surface).
H = int(os.environ.get("PROBE_H", "128"))
N = int(os.environ.get("PROBE_N", "64"))
print(f"# surface b=1 H=G={H} P=64 N={N} chunk=64")
print("seq nchunks  " + "  ".join(f"{nm:>9}" for nm in _GRAD_NAMES) + "    WORST   gate")
for s in SEQS:
    dims = (1, s, 64, H, H, 64, N)
    inp = _build_eval_inputs(dims)
    g = _mamba3_chunked_backward_path_c(
        inp["dy"], *inp["primals"],
        cb=inp["cb"], dA_cumsum=inp["dA_cumsum"],
        prev_states=inp["prev_states"], y=inp["y"])
    mx.eval(*g)
    diffs = {nm: _maxabs(gc, gg) for nm, gc, gg in zip(_GRAD_NAMES, g, inp["grads_gold"])}
    worst = max(diffs.values())
    nc = s // 64
    row = f"{s:>4} {nc:>5}    " + "  ".join(f"{diffs[nm]:.3e}" for nm in _GRAD_NAMES)
    row += f"    {worst:.3e}  {'PASS' if worst < 1e-3 else 'FAIL'}"
    print(row, flush=True)
    del inp, g

print(f"PEAK_RSS_KB={_PEAK} ({_PEAK/1024/1024:.2f} GB)")
print("DONE")
