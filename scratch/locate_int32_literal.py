"""Pinpoint WHERE the 2147483648 int32 literal arises in the snapshot lowering at
s4096, and whether it is a STRIDE/SIZE constant (TVM signed-int32 literal check) vs
a true addressing overflow. The hand-MSL kernel uses uint32 indexing whose MAX index
is 2^31-1 (fits); TVM FlattenBuffer rejects the 2^31 SIZE literal under signed int32.
Under memguard 70.
"""
from __future__ import annotations
import os, sys, threading, time

_LIM = 70 * 1024 * 1024
_PK = 0
def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _g():
    global _PK
    while True:
        r = _rss()
        if r > _PK: _PK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss={r}\n"); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g, daemon=True).start()

os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import mlx.core as mx
import numpy as np
from cppmega_mlx.nn._tilelang.mamba3_path_c import _bwd_state_snapshots_kernel_for

BATCH, HEADS, HEADDIM, STATE = 1, 128, 64, 64
SEQ = 4096
BLOCKS = SEQ  # BLOCK=1
total = BATCH*(BLOCKS+1)*HEADS*HEADDIM*STATE
print(f"h_snap total elems = {total:,}  (2^31={2**31:,})")
# the per-(batch,block) stride is HEADS*HEADDIM*STATE; the block-major stride that
# the flatten arithmetic multiplies SEQ-many times:
blk_stride = HEADS*HEADDIM*STATE
print(f"block stride (HEADS*HEADDIM*STATE) = {blk_stride:,}")
print(f"(BLOCKS)*block_stride = {(BLOCKS)*blk_stride:,}")
print(f"(BLOCKS+1)*block_stride = {(BLOCKS+1)*blk_stride:,}  <-- the SIZE literal")
print(f"BLOCKS*blk_stride = {BLOCKS*blk_stride:,}  == 2^31? {BLOCKS*blk_stride==2**31}")

# Build the snapshot kernel (lowers) then DISPATCH to surface the runtime int32 check
rng = np.random.RandomState(0)
x = mx.array((rng.randn(BATCH,SEQ,HEADS,HEADDIM)*0.1).astype(np.float32))
B = mx.array((rng.randn(BATCH,SEQ,HEADS,STATE)*0.1).astype(np.float32))
A = mx.array((-rng.rand(BATCH,SEQ,HEADS)).astype(np.float32))
dt = mx.array((rng.rand(BATCH,SEQ,HEADS)*0.05).astype(np.float32))
h0 = mx.array((rng.randn(BATCH,HEADS,HEADDIM,STATE)*0.1).astype(np.float32))
try:
    kernel, lowering = _bwd_state_snapshots_kernel_for(
        BATCH, SEQ, HEADS, HEADDIM, STATE, "float32","float32","float32","float32","float32","float32")
    print("snapshot BUILD ok; dispatching...")
    out = kernel(x, B, A, dt, h0)
    mx.eval(out if isinstance(out, mx.array) else out[0])
    print("snapshot DISPATCH ok (no overflow!)")
except Exception as e:
    import traceback
    print(f"snapshot DISPATCH FAILED: {type(e).__name__}: {str(e)[:200]}")
    for ln in traceback.format_exc().splitlines():
        if any(s in ln for s in ("2147483","exceeds","int32","value <","FlattenBuffer","ElemOffset","MergeMulMod")):
            print(f"   >> {ln.strip()}")

print(f"\nPEAK_RSS_KB={_PK} (~{_PK/1048576:.3f}GB) memguard70=ON")
print("RC=0")
