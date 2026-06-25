"""CONFIRM (read+probe only, no production edit): at tilelang HEAD c135c5d9 with
the int64 fix NOT applied, does the fp32 2-dispatch snapshot-simd LANE route
(_mamba3_mimo_bwd_path_c_simd_kernel) throw at s4096, and WHERE -- dispatch-1
(snapshot store, _bwd_state_snapshots_kernel_for) or dispatch-2 (lane-grad /
simd-reduce, _bwd_simd_reduce_kernel_for_state_snapshots)?

Strategy: drive the LOWERING of each dispatch's kernel SEPARATELY (build kernel +
lowering, which forces the FlattenBuffer + BindMetalScalarIntrinsics passes that
hit the int32 wall) at s4096. We do NOT need to run e2e -- the throw is a LOWERING
throw (make_const(int32, 2^31) in BindMetalScalarIntrinsics). Under memguard 70.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK_RSS_KB = 0


def _rss_kb() -> int:
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _memguard_thread():
    global _PEAK_RSS_KB
    while True:
        r = _rss_kb()
        if r > _PEAK_RSS_KB:
            _PEAK_RSS_KB = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL self rss_kb={r}\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_memguard_thread, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _bwd_state_snapshots_kernel_for,
    _bwd_simd_reduce_kernel_for_state_snapshots,
    _bwd_simd_p_reduction_supported,
)

HEADS, HEADDIM, STATE, BATCH = 128, 64, 64, 1
F = "float32"


def classify(exc: Exception) -> str:
    s = repr(exc) + "\n" + traceback.format_exc()
    keys = ("2147483648", "exceeds maximum", "int32", "FlattenBuffer",
            "make_const", "IntImm", "BindMetalScalarIntrinsic", "overflow")
    hits = [k for k in keys if k in s]
    return ("INT32-WALL " + ",".join(hits)) if hits else "OTHER"


print(f"_bwd_simd_p_reduction_supported(b=1,h=128,hd=64) = "
      f"{_bwd_simd_p_reduction_supported(batch=BATCH, heads=HEADS, headdim=HEADDIM)}")

for SEQ in [128, 2048, 4096]:
    print(f"\n=== SEQ={SEQ} ===")

    # DISPATCH-1: snapshot store kernel lowering
    try:
        _k1, _l1 = _bwd_state_snapshots_kernel_for(
            BATCH, SEQ, HEADS, HEADDIM, STATE, F, F, F, F, F)
        print(f"  dispatch-1 snapshot-store LOWERS  OK")
    except Exception as e:  # noqa: BLE001
        print(f"  dispatch-1 snapshot-store THROWS  [{classify(e)}]  {type(e).__name__}: {str(e)[:160]}")
        for ln in traceback.format_exc().splitlines():
            if any(s in ln for s in ("2147483", "exceeds", "int32", "make_const", "IntImm", "CoerceIntImm", "BindMetal", "FlattenBuffer")):
                print(f"      >> {ln.strip()[:200]}")

    # DISPATCH-2: simd-reduce (lane-grad + simd P-reduce) kernel lowering
    try:
        _k2, _l2 = _bwd_simd_reduce_kernel_for_state_snapshots(
            BATCH, SEQ, HEADS, HEADDIM, STATE,
            F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F)
        print(f"  dispatch-2 simd-reduce   LOWERS  OK")
    except Exception as e:  # noqa: BLE001
        print(f"  dispatch-2 simd-reduce   THROWS  [{classify(e)}]  {type(e).__name__}: {str(e)[:160]}")
        for ln in traceback.format_exc().splitlines():
            if any(s in ln for s in ("2147483", "exceeds", "int32", "make_const", "IntImm", "CoerceIntImm", "BindMetal", "FlattenBuffer")):
                print(f"      >> {ln.strip()[:200]}")

print(f"\nPEAK_RSS_KB={_PEAK_RSS_KB} (~{_PEAK_RSS_KB/1048576:.3f}GB)  memguard70=ON")
print("RC=0")
