"""LOCAL Apple-Metal A/B for the batched B2 GEMM prim (§TB1).

Builds+times the byte-identical SERIAL prod Metal prim vs the NEW batched-large-
tile Metal GEMM prim on SMALL bounded shapes (<<80GB local RSS guard), and checks
the GEMM-able grad outputs (dchunk_states + the shared dC/dz/dx/dD/dinp/dA_y) max
|abs| vs the serial prod. RULE #1: build/run errors propagate; no fallback.

The Metal batched prim GEMM-ifies DYX + dchunk_states (Apple 32KB pool); the other
terms stay serial — SAME serial math as the byte-identical prim, so parity must be
tight everywhere.
"""
import os
import time

import numpy as np
import mlx.core as mx  # noqa: F401  (forces Metal availability)
import tilelang
import tilelang.language as T  # noqa: F401
import torch

from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    chunk_scan_combine_bwd_metal_prim,
    chunk_scan_combine_bwd_metal_gemm_prim_batched,
    _b2_batched_heads_per_cta,
)

DEV = "mps"  # tilelang Metal kernels require MPS-resident torch tensors.

# SMALL bounded prod-shaped tile: chunk=64 (the prod L), P=N=64 (prod), but tiny
# S/H so total RAM is a few hundred MB, well under the 80GB local guard.
# IN-BUDGET Apple config: L=P=N=32 (NOT prod 64) keeps the threadgroup staging
# under Apple's HARD 32KB pool so the batched Metal prim actually BUILDS+RUNS. The
# prod L=P=N=64 batched Metal prim is the MEASURED SMEM-NO-GO (49664 B > 32768 B,
# the dispatcher gate RAISES) — reported separately, honest (RULE #1).
import os as _os2
if _os2.environ.get("CPPMEGA_METAL_PROD64") == "1":
    B, S, CH, G, H, P, N = 1, 128, 64, 2, 4, 64, 64  # prod dims -> expect SMEM RAISE
else:
    B, S, CH, G, H, P, N = 1, 64, 32, 2, 4, 32, 32   # in-budget, real measurement
nchunks = S // CH
HPC = _b2_batched_heads_per_cta(H, H // G, int(os.environ.get("CPPMEGA_PATH_C_METAL_HEADS_PER_CTA", "2")))

rng = np.random.RandomState(0)
def f16(shape):
    return torch.tensor((rng.randn(*shape) * 0.1).astype(np.float32), dtype=torch.float16, device=DEV)
def f32(shape):
    return torch.tensor((rng.randn(*shape) * 0.1).astype(np.float32), dtype=torch.float32, device=DEV)

dout = f16((B, S, H, P)); cb = f16((B, nchunks, G, CH, CH)); x = f16((B, S, H, P))
z = f16((B, S, H, P)); dt = f16((B, H, nchunks, CH)); dA = f16((B, H, nchunks, CH))
C = f16((B, S, G, N)); Bm = f16((B, S, G, N)); prev = f32((B, nchunks, H, P, N))
D = f16((H,)); y = f16((B, S, H, P))

def zeros():
    return (torch.zeros(B, S, H, N, dtype=torch.float32, device=DEV),
            torch.zeros(B, S, H, P, dtype=torch.float32, device=DEV),
            torch.zeros(B, S, H, P, dtype=torch.float32, device=DEV),
            torch.zeros(B, nchunks, H, P, N, dtype=torch.float32, device=DEV),
            torch.zeros(B, S, H, P, N, dtype=torch.float32, device=DEV),
            torch.zeros(B, H, nchunks, CH, dtype=torch.float32, device=DEV),
            torch.zeros(H, dtype=torch.float32, device=DEV))

OUT = [11, 12, 13, 14, 15, 16, 17]

def compile_prim(prim):
    return tilelang.compile(prim, out_idx=OUT, target=None)

def run(kern):
    o = zeros()
    kern(dout, cb, x, z, dt, dA, C, Bm, prev, D, y, *o)
    torch.mps.synchronize()
    return o

def timeit(kern, iters=30):
    run(kern)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        o = zeros()
        kern(dout, cb, x, z, dt, dA, C, Bm, prev, D, y, *o)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

print(f"[METAL-B2-BATCHED] dims B={B} S={S} chunk={CH} G={G} H={H} P={P} N={N} HPC={HPC}")

serial_prim = chunk_scan_combine_bwd_metal_prim(B, S, CH, G, H, P, N)
batched_prim = chunk_scan_combine_bwd_metal_gemm_prim_batched(B, S, CH, G, H, P, N, heads_per_cta=HPC)

k_serial = compile_prim(serial_prim)
k_batched = compile_prim(batched_prim)

o_s = run(k_serial)
o_b = run(k_batched)
names = ["dC", "dx", "dz", "dchunk", "dinp", "dA_y", "dD"]
maxabs = {nm: float((a - b).abs().max().cpu()) for nm, a, b in zip(names, o_s, o_b)}

t_s = timeit(k_serial)
t_b = timeit(k_batched)
speedup = t_s / t_b if t_b > 0 else float("nan")
verdict = "GO" if t_b < t_s else "NO-GO"
worst = max(maxabs.values())
parity = "PASS" if worst < 1e-3 else "FAIL"

print(f"[METAL-B2-BATCHED] serial={t_s:.3f}ms  batched={t_b:.3f}ms  "
      f"speedup={speedup:.3f}x  {verdict}  HPC={HPC}")
print(f"[METAL-B2-BATCHED] parity {parity} worst={worst:.2e}  "
      + "  ".join(f"{k}={v:.2e}" for k, v in maxabs.items()))
print("METAL_RESULT_JSON " + str({
    "serial_ms": round(t_s, 4), "batched_ms": round(t_b, 4),
    "speedup": round(speedup, 4), "verdict": verdict, "HPC": HPC,
    "parity": parity, "maxabs": {k: f"{v:.2e}" for k, v in maxabs.items()},
}))
