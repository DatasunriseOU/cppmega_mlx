"""DECISIVE verification of the int32 buffer-flattening wall at s4096.

Mandate re-confirmation: the prior phase REVERTED claiming a single-dispatch
full-sequence fp32 LANE backward is STRUCTURALLY IMPOSSIBLE at s4096 because the
forward-store slab / snapshot buffer (BATCH, SEQ+1, HEADS, HEADDIM, STATE) has
2,148,007,936 elements > 2^31, overflowing the int32 element-offset arithmetic in
TileLang/TVM FlattenBuffer.

This script INDEPENDENTLY tests that claim by attempting to lower the production
snapshot kernel (_bwd_state_snapshots_kernel_for, the 2-dispatch fp32 LANE's first
dispatch) at s2048 (must lower) and s4096 (claim: must overflow). Under memguard 70.
"""
from __future__ import annotations

import os
import sys
import threading
import time

# --- memguard 70: self-imposed 70GB RSS killer (mandatory) ------------------
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
            sys.stderr.write(f"[memguard70] KILL self rss_kb={r} (~{r//1048576}GB) > 70GB\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_memguard_thread, daemon=True).start()

os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import tilelang
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _bwd_state_snapshots_kernel_for,
    _bwd_simd_reduce_kernel_for_state_snapshots,
    _CHUNKED_METAL_TARGET,
    _bwd_scan_plan_for,
)

# nam56r production surface: BATCH=1 HEADS=128 HEADDIM=64 STATE=64
BATCH, HEADS, HEADDIM, STATE = 1, 128, 64, 64

for SEQ in [2048, 4094, 4096]:
    plan = _bwd_scan_plan_for(batch=BATCH, seq=SEQ, heads=HEADS, headdim=HEADDIM, state=STATE)
    BLOCK = plan.snapshot_plan.chunk_size
    BLOCKS = (SEQ + BLOCK - 1) // BLOCK
    elems = BATCH * (BLOCKS + 1) * HEADS * HEADDIM * STATE
    print(f"\n=== SEQ={SEQ} BLOCK={BLOCK} BLOCKS={BLOCKS} h_snap_elems={elems:,} (>2^31={elems>2**31}) ===")
    try:
        kernel, lowering = _bwd_state_snapshots_kernel_for(
            BATCH, SEQ, HEADS, HEADDIM, STATE,
            "float32", "float32", "float32", "float32", "float32", "float32",
        )
        print(f"  SNAPSHOT LOWERED OK at SEQ={SEQ} (kernel={type(kernel).__name__})")
    except Exception as e:  # noqa: BLE001
        import traceback
        msg = str(e)
        print(f"  SNAPSHOT FAILED at SEQ={SEQ}: {type(e).__name__}: {msg[:300]}")
        tb = traceback.format_exc()
        # surface the int32 overflow signature if present
        for ln in tb.splitlines():
            if "int32" in ln.lower() or "overflow" in ln.lower() or "2147483" in ln or "FlattenBuffer" in ln or "exceeds" in ln.lower():
                print(f"      >> {ln.strip()}")

print(f"\nPEAK_RSS_KB={_PEAK_RSS_KB} (~{_PEAK_RSS_KB/1048576:.3f}GB)  memguard70=ON")
print("RC=0")
