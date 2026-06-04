"""Local Apple-GPU parity + timing: F0/B2/B0 Metal-GEMM prims vs the serial Metal
prims (the byte-identical parity reference).

For each prim we compile BOTH the serial Metal prim (flag OFF) and the Metal-GEMM
twin (CPPMEGA_PATH_C_METAL_GEMM=1), drive them with the SAME fp16 inputs on the
Apple GPU, and compare EVERY output element (fp16 gate 5e-4) + time both.

RULE #1: a parity miss FAILS (no subset, no fallback). Run:
  .venv/bin/python scratch/probe_f0b2b0_metal_gemm_parity.py
"""
import os
import sys
import time

import tilelang  # noqa: F401  (load dev-root tilelang first)
import torch

from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
    build_chunk_precompute_metal,
)
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    build_chunk_scan_combine_bwd_metal,
    build_chunk_precompute_bwd_metal,
)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
GATE = 5e-4

# prod-ish small dims (all multiples of 16/8 so the GEMM divisibility holds):
B, S, CH, G, H, P, N = 1, 128, 64, 1, 2, 64, 64
NCH = S // CH


def _sync():
    if DEV == "mps":
        torch.mps.synchronize()


def _time(kernel, args, outs, iters=50):
    for _ in range(5):
        kernel(*args, *outs)
    _sync()
    t0 = time.time()
    for _ in range(iters):
        kernel(*args, *outs)
    _sync()
    return (time.time() - t0) / iters * 1e3


def _build(builder, flag):
    if flag:
        os.environ["CPPMEGA_PATH_C_METAL_GEMM"] = "1"
    else:
        os.environ.pop("CPPMEGA_PATH_C_METAL_GEMM", None)
    return builder(B, S, CH, G, H, P, N, target=None)


def f0_inputs():
    torch.manual_seed(0)
    f16 = torch.float16
    x = (torch.randn(B, S, H, P, device=DEV, dtype=f16) * 0.1).contiguous()
    Bt = (torch.randn(B, S, G, N, device=DEV, dtype=f16) * 0.1).contiguous()
    C = (torch.randn(B, S, G, N, device=DEV, dtype=f16) * 0.1).contiguous()
    A = (-torch.rand(H, device=DEV, dtype=f16)).contiguous()
    dt = (torch.rand(B, S, H, device=DEV, dtype=f16) * 0.05).contiguous()
    return [x, Bt, C, A, dt]


def f0_outs():
    f16, f32 = torch.float16, torch.float32
    cb = torch.zeros(B, NCH, G, CH, CH, device=DEV, dtype=f16)
    dac = torch.zeros(B, H, NCH, CH, device=DEV, dtype=f16)
    ss = torch.zeros(B, NCH, H, P, N, device=DEV, dtype=f32)
    return [cb, dac, ss]


def run_f0():
    args = f0_inputs()
    k_ser = _build(build_chunk_precompute_metal, flag=False)
    o_ser = f0_outs()
    k_ser(*args, *o_ser)
    _sync()
    k_gem = _build(build_chunk_precompute_metal, flag=True)
    o_gem = f0_outs()
    k_gem(*args, *o_gem)
    _sync()
    names = ["cb", "dA_cumsum", "summary_states"]
    worst = 0.0
    for nm, a, b in zip(names, o_ser, o_gem):
        d = float((a.float() - b.float()).abs().max())
        nan = bool(torch.isnan(b).any())
        worst = max(worst, d)
        print(f"  F0 {nm:16s} max|abs diff|={d:.3e} nan={nan}")
    ms_ser = _time(k_ser, args, o_ser)
    ms_gem = _time(k_gem, args, o_gem)
    print(f"  F0 serial={ms_ser:.4f}ms  gemm={ms_gem:.4f}ms  "
          f"speedup={ms_ser/ms_gem:.2f}x  worst={worst:.3e} gate={GATE}")
    return worst <= GATE


def run_gate(name, builder):
    """B2/B0 GEMM flag must RAISE (honest scope, RULE #1) — and the serial prim
    (flag OFF) must still compile+run."""
    os.environ["CPPMEGA_PATH_C_METAL_GEMM"] = "1"
    raised = False
    try:
        builder(B, S, CH, G, H, P, N, target=None)
    except NotImplementedError as e:
        raised = True
        print(f"  {name} GEMM-flag RAISED (expected): {str(e).splitlines()[0][:60]}")
    os.environ.pop("CPPMEGA_PATH_C_METAL_GEMM", None)
    k = builder(B, S, CH, G, H, P, N, target=None)
    print(f"  {name} serial prim compiles OK (flag off): {type(k).__name__}")
    return raised


if __name__ == "__main__":
    print(f"device={DEV} dims B={B} S={S} chunk={CH} G={G} H={H} P={P} N={N}")
    ok = True
    print("== F0 chunk_precompute (GEMM-ified) ==")
    ok &= run_f0()
    print("== B2 scan_combine_bwd (GEMM flag -> honest RAISE; serial works) ==")
    ok &= run_gate("B2", build_chunk_scan_combine_bwd_metal)
    print("== B0 precompute_bwd (GEMM flag -> honest RAISE; serial works) ==")
    ok &= run_gate("B0", build_chunk_precompute_bwd_metal)
    print("ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
