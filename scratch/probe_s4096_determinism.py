"""PROBE: single s4096 (64-chunk) run at nam56r surface vs path-b GOLD.
Run N fresh processes to detect intermittent race FAIL. memguard 70.
"""
from __future__ import annotations
import os, sys, threading, time
_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK = 0
def _rss_kb():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _guard():
    global _PEAK
    while True:
        r = _rss_kb()
        if r > _PEAK: _PEAK = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_guard, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
sys.path.insert(0, "/Volumes/external/sources/tilelang")
import numpy as np
import mlx.core as mx
from cppmega_mlx.runtime.path_c_backward_fusion_search import _build_eval_inputs, _maxabs, _GRAD_NAMES
from cppmega_mlx.nn._tilelang.mamba3_path_c import _mamba3_chunked_backward_path_c, _force_chunked_command_buffer_boundary
_force_chunked_command_buffer_boundary()
S = int(os.environ.get("PROBE_SEQ", "4096"))
H = int(os.environ.get("PROBE_H", "128"))
N = int(os.environ.get("PROBE_N", "64"))
dims = (1, S, 64, H, H, 64, N)
inp = _build_eval_inputs(dims)
g = _mamba3_chunked_backward_path_c(inp["dy"], *inp["primals"],
        cb=inp["cb"], dA_cumsum=inp["dA_cumsum"], prev_states=inp["prev_states"], y=inp["y"])
mx.eval(*g)
diffs = {nm: _maxabs(gc, gg) for nm, gc, gg in zip(_GRAD_NAMES, g, inp["grads_gold"])}
worst = max(diffs.values())
print("RESULT s=%d " % S + " ".join(f"{nm}={diffs[nm]:.3e}" for nm in _GRAD_NAMES)
      + f" WORST={worst:.3e} {'PASS' if worst<1e-3 else 'FAIL'}", flush=True)
