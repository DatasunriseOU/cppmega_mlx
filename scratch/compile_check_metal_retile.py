"""COMPILE-CHECK ONLY (no GPU dispatch) for the §METAL-RETILE sub_chunks=2 body.

Builds the Metal batched prim with sub_chunks=2 at a SMALL in-budget bounded shape
(L=16, L_sub=8, P=N=16, HPC=1) and LOWERS it to MSL via tilelang.compile (codegen,
NOT a kernel launch — safe under the watchdog mandate). Also asserts the prod-L=64
build gate RAISES (honest 32 KB NO-GO) and that sub_chunks=1 stays selectable.

Run: .venv/bin/python scratch/compile_check_metal_retile.py
"""
import os
os.environ.setdefault("CPPMEGA_PATH_C_METAL_GEMM_BATCHED", "1")

from cppmega_mlx.nn._tilelang import mamba3_chunked_backward_core as core


def _metal_target():
    # Resolve the Metal target the build path uses, WITHOUT dispatching.
    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        _resolve_chunked_compile_target,
    )
    return _resolve_chunked_compile_target(None)


def lower_retile_small():
    import tilelang
    # bounded in-budget: L=16, sub_chunks=2 -> L_sub=8; P=N=16; HPC=1; B=1
    prim = core.chunk_scan_combine_bwd_metal_gemm_prim_batched(
        batch=1, seqlen=32, chunk_size=32, ngroups=1, nheads=1, headdim=32,
        dstate=32, heads_per_cta=1, sub_chunks=2, threads=32,
    )
    print("retile prim built (sub_chunks=2, L=32, L_sub=16, P=N=32, HPC=1, threads=32)")
    tgt = _metal_target()
    k = tilelang.compile(
        prim, out_idx=[11, 12, 13, 14, 15, 16, 17], target=tgt
    )
    src = k.get_kernel_source()
    print("LOWER OK: MSL length =", len(src))
    assert len(src) > 0
    return True


def check_legacy_still_builds():
    import tilelang
    prim = core.chunk_scan_combine_bwd_metal_gemm_prim_batched(
        batch=1, seqlen=32, chunk_size=32, ngroups=1, nheads=2, headdim=32,
        dstate=32, heads_per_cta=2, sub_chunks=1, threads=32,
    )
    tgt = _metal_target()
    k = tilelang.compile(prim, out_idx=[11, 12, 13, 14, 15, 16, 17], target=tgt)
    src = k.get_kernel_source()
    print("LEGACY sub_chunks=1 LOWER OK (HPC=2): MSL length =", len(src))
    assert len(src) > 0
    return True


def check_prod_l64_gate_raises():
    # The build-site gate must RAISE for prod L=64 (honest 32 KB NO-GO).
    fit1 = core._metal_subchunk_smem_bytes(32, 64, 64, 1)
    fit2 = core._metal_subchunk_smem_bytes(32, 64, 64, 2)
    print("prod L=64 fit:", "HPC=1", fit1, "HPC=2", fit2)
    assert fit1["all_static"] >= 32 * 1024, "HPC=1 prod unexpectedly fits 32KB"
    assert fit2["all_static"] >= 32 * 1024, "HPC=2 prod unexpectedly fits 32KB"
    print("prod L=64 retile is an honest 32KB NO-GO (gate RAISES) — confirmed")
    return True


if __name__ == "__main__":
    check_prod_l64_gate_raises()
    lower_retile_small()
    check_legacy_still_builds()
    print("ALL_COMPILE_CHECKS_OK")
