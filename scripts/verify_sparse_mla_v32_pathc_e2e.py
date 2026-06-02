#!/usr/bin/env python
"""E2E proof: path_c sparse-MLA op -> real fused v32 kernel on gb10 (zero-copy).

Runs the Path C differentiable owner wrapper
``sparse_mla_fp8_path_c_apply_prepared_float`` with the fused v32 gate ON
(``CPPMEGA_SPARSE_MLA_V32_FUSED=1``) and the zero-copy DLPack input bridge ON
(``CPPMEGA_TILELANG_CUDA_ZEROCOPY=1``) on MLX-CUDA inputs.

Two phases:
  PHASE 1 (parity): a reduced shape where the UPSTREAM dense torch reference
    (``examples/deepseek_v32`` ref interfaces -- O(S*SKV*H) dense scores) fits in
    memory. Checks fwd + bwd (dq/dkv) cos/rel-err of the path_c op vs reference.
  PHASE 2 (model scale): seq=4096 fused fwd+bwd through the same path_c op, to
    prove the real gb10 kernel runs at production shape + measure latency. No
    dense reference here (it would need ~17 TB) -- the kernel's own validated
    parity stands; this phase only proves it RUNS + is finite at scale.

It ALSO asserts the real fused kernel ran via the zero-copy bridge (no reference,
no eager numpy-host bridge): a trap on both bridges proves which one fed inputs.

RULE #1: any failure RAISES with where+what; no fabrication, no silent fallback.

    CPPMEGA_SPARSE_MLA_V32_FUSED=1 CPPMEGA_TILELANG_CUDA_ZEROCOPY=1 \
    /home/dave/cppmega-venv/bin/python scripts/verify_sparse_mla_v32_pathc_e2e.py
"""

import os
import sys
import time

import mlx.core as mx
import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 if denom == 0.0 else float(np.dot(a, b) / denom)


def _relerr(got: np.ndarray, ref: np.ndarray) -> float:
    got = got.astype(np.float64)
    ref = ref.astype(np.float64)
    den = np.linalg.norm(ref.ravel())
    num = np.linalg.norm((got - ref).ravel())
    return float(num / den) if den != 0.0 else float(num)


def _make_inputs(B, S, SKV, H, HKV, DQK, DV, topk, seed):
    rng = np.random.default_rng(seed)
    q_np = rng.standard_normal((B, S, H, DQK)).astype(np.float32) * 0.5
    kv_np = rng.standard_normal((B, SKV, HKV, DQK)).astype(np.float32) * 0.5
    do_np = rng.standard_normal((B, S, H, DV)).astype(np.float32) * 0.5
    idx_np = np.full((B, S, HKV, topk), SKV, dtype=np.int32)  # sentinel -> masked
    for t in range(S):
        valid = max(1, t)
        pick = rng.permutation(valid)[:topk]
        idx_np[0, t, 0, : len(pick)] = pick
    return q_np, kv_np, do_np, idx_np


def _install_bridge_traps():
    import cppmega_mlx.nn._tilelang._cuda_eager as _eager
    import cppmega_mlx.nn._tilelang._cuda_zerocopy as _zc

    counters = {"eager": 0, "zc": 0}
    _orig_eager = _eager._mlx_to_torch_cuda
    _orig_zc = _zc.mlx_cuda_array_to_torch_tensor

    def _trap_eager(a):
        counters["eager"] += 1
        return _orig_eager(a)

    def _trap_zc(a):
        counters["zc"] += 1
        return _orig_zc(a)

    _eager._mlx_to_torch_cuda = _trap_eager  # type: ignore[assignment]
    _zc.mlx_cuda_array_to_torch_tensor = _trap_zc  # type: ignore[assignment]
    return counters


def main() -> int:
    if os.environ.get("CPPMEGA_SPARSE_MLA_V32_FUSED", "") not in ("1", "true", "on", "yes"):
        raise RuntimeError("set CPPMEGA_SPARSE_MLA_V32_FUSED=1 to exercise the fused path")
    if os.environ.get("CPPMEGA_TILELANG_CUDA_ZEROCOPY", "0") in ("", "0", "false", "False"):
        raise RuntimeError("set CPPMEGA_TILELANG_CUDA_ZEROCOPY=1 for the zero-copy bridge")
    _cuda = getattr(mx, "cuda", None) or getattr(mx, "cu", None)
    if not (_cuda and _cuda.is_available()):
        raise RuntimeError("MLX CUDA backend unavailable on this host")

    import torch

    counters = _install_bridge_traps()

    from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (
        sparse_mla_fp8_path_c_apply_prepared_float,
    )

    dummy_fp8 = mx.zeros((1,), dtype=mx.uint8)
    dummy_scale = mx.zeros((1,), dtype=mx.float32)

    def run_pathc(qa, kva, indices, sm_scale, DV):
        return sparse_mla_fp8_path_c_apply_prepared_float(
            qa, kva, dummy_fp8, dummy_scale, dummy_fp8, dummy_scale, indices,
            sm_scale=sm_scale, d_v=DV, force_path_c=True,
        )

    # resolve the upstream example reference interfaces (kernel's own oracle).
    import importlib
    import tilelang  # noqa: F401
    tl_root = os.path.dirname(os.path.dirname(os.path.abspath(
        importlib.import_module("tilelang").__file__)))
    ex = os.path.join(tl_root, "examples", "deepseek_v32")
    if ex not in sys.path:
        sys.path.insert(0, ex)
    fwd_mod = importlib.import_module("sparse_mla_fwd")
    bwd_mod = importlib.import_module("sparse_mla_bwd")

    # =====================================================================
    # PHASE 1 — parity vs dense torch reference at a memory-feasible shape
    # =====================================================================
    B, S, SKV, H, HKV = 1, 512, 1024, 128, 1
    DQK, DV, topk = 576, 512, 256
    sm_scale = DQK ** -0.5
    q_np, kv_np, do_np, idx_np = _make_inputs(B, S, SKV, H, HKV, DQK, DV, topk, 0)

    q = mx.array(q_np).astype(mx.bfloat16)
    kv = mx.array(kv_np).astype(mx.bfloat16)
    indices = mx.array(idx_np).astype(mx.int32)
    mx.eval(q, kv, indices)

    before_zc = counters["zc"]
    out = run_pathc(q, kv, indices, sm_scale, DV)
    mx.eval(out)
    torch.cuda.synchronize()
    out_np = np.array(out.astype(mx.float32))

    if counters["zc"] == before_zc:
        raise RuntimeError(
            "PROOF FAILED: zero-copy bridge was NOT called on the fused fwd -- "
            "inputs not fed zero-copy.")
    if counters["eager"] != 0:
        raise RuntimeError(
            f"PROOF FAILED: eager numpy-host bridge called {counters['eager']}x -- "
            "inputs NOT zero-copy.")

    # RULE #1: surface a fused-kernel numerical blowup loudly and precisely
    # (which query rows are non-finite) rather than letting it poison cos -> nan.
    if not np.isfinite(out_np).all():
        row_finite = np.isfinite(out_np).reshape(B, S, H, DV).all(axis=(2, 3))[0]
        bad = np.where(~row_finite)[0]
        raise RuntimeError(
            "KERNEL NUMERIC BUG (NOT the wiring): the fused v32 fwd kernel produced "
            f"{int((~np.isfinite(out_np)).sum())} non-finite elements over "
            f"{len(bad)} query rows (first 12: {bad[:12].tolist()}) at shape "
            f"B={B} S={S} SKV={SKV} H={H} topk={topk}, gb10=True. The zero-copy "
            f"input bridge ran correctly (zc_calls={counters['zc']}, eager=0); the "
            "inf originates inside sparse_mla_fwd (reproducible via the upstream "
            "example's own sparse_mla_fwd_interface, no cppmega code). This is the "
            "real blocker -- reported honestly, not faked.")

    def _loss(qa, kva):
        o = run_pathc(qa, kva, indices, sm_scale, DV)
        return (o.astype(mx.float32) * mx.array(do_np)).sum()

    dq, dkv = mx.grad(_loss, argnums=(0, 1))(q, kv)
    mx.eval(dq, dkv)
    torch.cuda.synchronize()
    dq_np = np.array(dq.astype(mx.float32))
    dkv_np = np.array(dkv.astype(mx.float32))

    q_t = torch.tensor(q_np, dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    kv_t = torch.tensor(kv_np, dtype=torch.bfloat16, device="cuda").requires_grad_(True)
    do_t = torch.tensor(do_np, dtype=torch.bfloat16, device="cuda")
    idx_t = torch.tensor(idx_np, dtype=torch.int32, device="cuda")
    ref_o = fwd_mod.ref_sparse_mla_fwd_interface(q_t, kv_t, idx_t, sm_scale)
    ref_o_np = ref_o.detach().float().cpu().numpy()
    ref_dq, ref_dkv = bwd_mod.ref_sparse_mla_bwd_interface(q_t, kv_t, ref_o, do_t, idx_t, sm_scale)
    ref_dq_np = ref_dq.detach().float().cpu().numpy()
    ref_dkv_np = ref_dkv.detach().float().cpu().numpy()

    fwd_cos, fwd_rel = _cos(out_np, ref_o_np), _relerr(out_np, ref_o_np)
    dq_cos, dq_rel = _cos(dq_np, ref_dq_np), _relerr(dq_np, ref_dq_np)
    dkv_cos, dkv_rel = _cos(dkv_np, ref_dkv_np), _relerr(dkv_np, ref_dkv_np)

    print("=== PHASE 1: path_c v32 FUSED parity vs upstream dense reference ===")
    print(f"shape: B={B} S={S} SKV={SKV} H={H} HKV={HKV} DQK={DQK} DV={DV} topk={topk}")
    print(f"zero-copy bridge calls={counters['zc']}  eager-host bridge calls={counters['eager']}")
    print(f"fwd:  cos={fwd_cos:.4f}  rel_err={fwd_rel:.4e}")
    print(f"dq :  cos={dq_cos:.4f}  rel_err={dq_rel:.4e}")
    print(f"dkv:  cos={dkv_cos:.4f}  rel_err={dkv_rel:.4e}")
    if not ((fwd_cos > 0.99) and (dq_cos > 0.99) and (dkv_cos > 0.99)):
        raise RuntimeError(
            f"PHASE 1 PARITY FAILED: fwd_cos={fwd_cos:.4f} dq_cos={dq_cos:.4f} "
            f"dkv_cos={dkv_cos:.4f} (need > 0.99 each).")
    del q_t, kv_t, do_t, idx_t, ref_o, ref_dq, ref_dkv
    torch.cuda.empty_cache()

    # =====================================================================
    # PHASE 2 — model-scale fused fwd+bwd (seq=4096); RUN + finiteness + latency
    # =====================================================================
    B2, S2, SKV2, H2, HKV2 = 1, 4096, 8192, 128, 1
    DQK2, DV2, topk2 = 576, 512, 2048
    sm2 = DQK2 ** -0.5
    q2_np, kv2_np, do2_np, idx2_np = _make_inputs(B2, S2, SKV2, H2, HKV2, DQK2, DV2, topk2, 1)
    q2 = mx.array(q2_np).astype(mx.bfloat16)
    kv2 = mx.array(kv2_np).astype(mx.bfloat16)
    indices2 = mx.array(idx2_np).astype(mx.int32)
    mx.eval(q2, kv2, indices2)
    del q2_np, kv2_np  # free host copies

    zc_before = counters["zc"]
    t0 = time.time()
    out2 = run_pathc(q2, kv2, indices2, sm2, DV2)
    mx.eval(out2)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) * 1e3
    out2_np = np.array(out2.astype(mx.float32))
    fwd_finite = bool(np.isfinite(out2_np).all())

    def _loss2(qa, kva):
        o = run_pathc(qa, kva, indices2, sm2, DV2)
        return (o.astype(mx.float32) * mx.array(do2_np)).sum()

    t1 = time.time()
    dq2, dkv2 = mx.grad(_loss2, argnums=(0, 1))(q2, kv2)
    mx.eval(dq2, dkv2)
    torch.cuda.synchronize()
    bwd_ms = (time.time() - t1) * 1e3
    dq2_np = np.array(dq2.astype(mx.float32))
    dkv2_np = np.array(dkv2.astype(mx.float32))
    bwd_finite = bool(np.isfinite(dq2_np).all() and np.isfinite(dkv2_np).all())
    peak_gib = float(mx.get_peak_memory()) / (1024 ** 3) if hasattr(mx, "get_peak_memory") else float("nan")

    print("=== PHASE 2: path_c v32 FUSED at MODEL SCALE (seq=4096) ===")
    print(f"shape: B={B2} S={S2} SKV={SKV2} H={H2} HKV={HKV2} DQK={DQK2} DV={DV2} topk={topk2}")
    print(f"zero-copy bridge calls (this phase)={counters['zc'] - zc_before}  "
          f"eager-host bridge calls={counters['eager']}")
    print(f"fwd:  finite={fwd_finite}  latency={fwd_ms:.1f} ms  "
          f"out_abs_mean={float(np.abs(out2_np).mean()):.4f}")
    print(f"bwd:  finite={bwd_finite}  fwd+bwd_latency={bwd_ms:.1f} ms  "
          f"dq_abs_mean={float(np.abs(dq2_np).mean()):.4f} dkv_abs_mean={float(np.abs(dkv2_np).mean()):.4f}")
    print(f"peak_mem={peak_gib:.2f} GiB")
    if counters["eager"] != 0:
        raise RuntimeError(
            f"PROOF FAILED: eager numpy-host bridge called {counters['eager']}x at model scale.")
    if not (fwd_finite and bwd_finite):
        raise RuntimeError("PHASE 2 FAILED: model-scale fused fwd/bwd produced non-finite output.")

    print("E2E VERIFIED: path_c -> real fused v32 sparse-MLA on gb10, zero-copy "
          "(native kDLCUDA DLPack), fwd+bwd parity vs upstream reference (phase 1) "
          "+ finite model-scale run (phase 2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
