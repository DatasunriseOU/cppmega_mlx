"""Capture the FULL traceback of the s4096 int32 overflow in the lane-grad kernel
that consumes h_snap (BATCH, SEQ+1, HEADS, HEADDIM, STATE), to pinpoint whether the
2^31 IntImm is a STRIDE literal constructed during FlattenBuffer (before the int64
promotion bound check). Under memguard 70.
"""
from __future__ import annotations
import os, sys, threading, time
_LIM = 70*1024*1024; _PK=0
def _rss():
    import resource; return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
def _g():
    global _PK
    while True:
        r=_rss()
        if r>_PK:_PK=r
        if r>_LIM: sys.stderr.write(f"[memguard70] KILL rss={r}\n"); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g,daemon=True).start()
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY","1")
sys.path.insert(0,"/Volumes/external/sources/cppmega.mlx")

from cppmega_mlx.nn._tilelang.mamba3_path_c import _bwd_lane_grad_kernel_for_state_snapshots
BATCH,HEADS,HEADDIM,STATE=1,128,64,64
SEQ=4096
try:
    out = _bwd_lane_grad_kernel_for_state_snapshots(
        BATCH,SEQ,HEADS,HEADDIM,STATE,
        "float32","float32","float32","float32","float32","float32","float32","float32","float32",
        "float32","float32","float32","float32","float32","float32","float32","float32",
    )
    print("LANE-GRAD KERNEL BUILT OK at s4096 (unexpected)")
except Exception as e:
    import traceback
    print(f"LANE-GRAD KERNEL FAILED: {type(e).__name__}: {str(e)[:160]}")
    print("---FULL TRACEBACK (last 40 lines)---")
    tb = traceback.format_exc().splitlines()
    for ln in tb[-40:]:
        print(ln)
print(f"\nPEAK_RSS_KB={_PK} memguard70=ON\nRC=0")
