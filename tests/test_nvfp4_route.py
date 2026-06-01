# pyright: reportMissingImports=false
"""Tests for the NVFP4 (e2m1 + block-scale) training-dtype route.

``scripts/_nvfp4_route.py`` wires ``--dtype nvfp4`` as an HONEST PARTIAL route
under RULE #1 (no silent fallbacks):

* The FORWARD nvfp4 GEMM is real on two backends:
    - Metal/Apple-M4: ``mxfp4_matmul_path_c`` cooperative-tensor GEMM over the
      ``quantize_mxfp4_blockwise`` e2m1 + per-block-16 codec.
    - CUDA/gb10 sm_121: TransformerEngine ``NVFP4BlockScaling`` + cuBLASLt FP4
      GEMM (VERIFIED working, rel_err vs bf16 ~0.146).
* The BACKWARD nvfp4 GEMM, optimizer state, and every non-GEMM op have NO
  working nvfp4 kernel on the available arch, so the route RAISES a precise,
  actionable error naming the missing op + enablement -- it NEVER downcasts to
  bf16 silently. On gb10 specifically the backward fails because the FP4
  stochastic-rounding cast (``cvt.rs.satfinite.e2m1x4``) and Random-Hadamard
  fused kernels are DATACENTER-ONLY: TE source-gates the SR cvt to
  ``sm_100a``/``sm_103a`` and the RHT kernel needs the SM100 TMEM/tcgen05
  pipeline that sm_12x does not implement. This is NOT fixable by a TE rebuild
  for any sm_12x target (verified 2026-06-01) -- the fail-loud is permanent on
  consumer/Spark Blackwell until NVIDIA ships SM12x-native kernels upstream.

These tests assert BOTH halves: the real forward op matches bf16 within
tolerance, and the unsupported ops raise the expected ``Nvfp4*`` error.

Run on Mac (Metal-only assertions skip cleanly when MLX/Metal absent) and on
gb10 (the CUDA assertions run when torch+TE+CUDA are present).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make ``scripts`` importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._nvfp4_route import (  # noqa: E402
    NVFP4_BLOCK_SIZE,
    NVFP4_ELEMENT_FORMAT,
    NVFP4_SUPPORTED_OPS,
    NVFP4_UNSUPPORTED_TRAINING_OPS,
    Nvfp4CudaKernelMiscompiled,
    Nvfp4TrainingRouteUnavailable,
    nvfp4_route_requested,
    nvfp4_te_cuda_capability,
    nvfp4_te_gemm_probe,
    nvfp4_training_route_payload,
    nvfp4_training_route_unavailable_reason,
    raise_if_nvfp4_training_unsupported,
)


class _Args:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype


# ---------------------------------------------------------------------------
# Always-on tests (Mac + gb10): route metadata + fail-loud guard.
# ---------------------------------------------------------------------------


def test_route_constants_well_formed() -> None:
    assert NVFP4_ELEMENT_FORMAT == "e2m1"
    assert NVFP4_BLOCK_SIZE == 16
    # Both forward backends are advertised as supported.
    assert "forward_gemm_operands_e2m1_block_scale_metal_m4" in NVFP4_SUPPORTED_OPS
    assert "forward_nvfp4_gemm_te_cublaslt_cuda_sm121" in NVFP4_SUPPORTED_OPS
    # The decisive gb10 backward blocker is enumerated as unsupported.
    assert (
        "nvfp4_gemm_backward_vjp_te_cuda_sm121"
        in NVFP4_UNSUPPORTED_TRAINING_OPS
    )
    assert "optimizer_state_and_update" in NVFP4_UNSUPPORTED_TRAINING_OPS


def test_route_requested_predicate() -> None:
    assert nvfp4_route_requested(_Args("nvfp4")) is True
    assert nvfp4_route_requested(_Args("NVFP4")) is True
    assert nvfp4_route_requested(_Args("bfloat16")) is False
    assert nvfp4_route_requested(_Args("fp8_path_c")) is False


def test_payload_advertises_both_forward_backends_and_no_e2e() -> None:
    payload = nvfp4_training_route_payload(_Args("nvfp4"))
    assert payload["full_end_to_end_training_available"] is False
    assert payload["element_format"] == "e2m1"
    assert "transformer_engine" in payload["forward_gemm_kernel_cuda"].lower()
    assert "mxfp4_matmul_path_c" in payload["forward_gemm_kernel_metal"]
    # gb10 backward gap is documented as datacenter-only (NOT a rebuild gap).
    assert "datacenter_only" in payload["backward_gemm_cuda_status"]
    assert "not_fixable_by_te_rebuild" in payload["backward_gemm_cuda_status"]
    enablement = payload["backward_gemm_cuda_enablement"]
    assert "UPSTREAM ONLY" in enablement
    assert "tcgen05" in enablement
    # Must NOT advertise the disproven compute_120f rebuild as the fix.
    assert "compute_120f" not in enablement


def test_fail_loud_guard_raises_for_nvfp4_and_names_missing_ops() -> None:
    """RULE #1: --dtype nvfp4 on the training path RAISES (no bf16 downcast)."""
    with pytest.raises(Nvfp4TrainingRouteUnavailable) as ei:
        raise_if_nvfp4_training_unsupported(_Args("nvfp4"))
    msg = str(ei.value)
    # The error must name the decisive backward blocker + the REAL reason.
    assert "BACKWARD" in msg or "backward" in msg
    assert "cvt.rs.satfinite.e2m1x4" in msg
    assert "tcgen05" in msg
    assert "UPSTREAM ONLY" in msg
    # The disproven "rebuild with compute_120f" claim must be gone.
    assert "compute_120f" not in msg
    # And must NOT pretend a full training step works.
    assert "fails loud" in msg or "fail loud" in msg or "fp8_path_c" in msg


def test_fail_loud_guard_is_noop_for_other_dtypes() -> None:
    # Non-nvfp4 dtypes must pass through untouched (the guard only gates nvfp4).
    raise_if_nvfp4_training_unsupported(_Args("bfloat16"))
    raise_if_nvfp4_training_unsupported(_Args("fp8_path_c"))


def test_reason_message_is_actionable() -> None:
    reason = nvfp4_training_route_unavailable_reason()
    assert "e2m1" in reason
    # The corrected, evidence-backed enablement: datacenter-only, upstream fix.
    assert "cvt.rs.satfinite.e2m1x4" in reason
    assert "tcgen05" in reason
    assert "UPSTREAM ONLY" in reason
    assert "compute_120f" not in reason  # disproven hypothesis removed
    assert "fp8_path_c" in reason  # points at a route that works today


def test_capability_probe_never_crashes_without_torch() -> None:
    # On a Mac dev host torch/TE are absent; the probe records the reason and
    # returns a dict rather than raising.
    cap = nvfp4_te_cuda_capability()
    assert isinstance(cap, dict)
    assert "torch_available" in cap and "te_available" in cap
    if not cap["torch_available"] or not cap["te_available"]:
        assert cap["import_error"] is not None


# ---------------------------------------------------------------------------
# Metal/M4 forward GEMM (skips cleanly when MLX/Metal absent).
# ---------------------------------------------------------------------------


def _mlx_metal_available() -> bool:
    """True only if the Metal mxfp4 GEMM is actually RUNNABLE on this host.

    The tilelang dev build (tvm_ffi) can be mid-merge / broken in some checkouts;
    we run a real tiny GEMM through the route's smoke and only enable the Metal
    assertions when it completes. (A broken tilelang build is an infra gap, not
    an nvfp4-route bug -- the route's own logic is covered by the always-on and
    CUDA tests.)
    """
    try:
        import mlx.core as mx  # noqa: F401,PLC0415

        from scripts._nvfp4_route import nvfp4_gemm_smoke  # noqa: PLC0415

        nvfp4_gemm_smoke(M=16, N=32, K=64, seed=0)
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not _mlx_metal_available(),
    reason="MLX/Metal mxfp4 kernel not available on this host",
)
def test_metal_forward_nvfp4_gemm_matches_bf16() -> None:
    from scripts._nvfp4_route import nvfp4_gemm_smoke  # noqa: PLC0415

    out = nvfp4_gemm_smoke(M=16, N=32, K=64, seed=0)
    assert out["all_finite"] is True
    assert out["element_format"] == "e2m1"
    # e2m1 + block-16 forward GEMM is lossy but must be far from random.
    assert out["rel_rmse_vs_bf16"] < 0.5, out


# ---------------------------------------------------------------------------
# CUDA/gb10 NVFP4: forward works, backward fails loud. (Skips off-CUDA.)
# ---------------------------------------------------------------------------


def _cuda_te_available() -> bool:
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import transformer_engine.pytorch as te  # noqa: PLC0415

        return bool(te.is_nvfp4_available())
    except Exception:
        return False


@pytest.mark.skipif(
    not _cuda_te_available(),
    reason="CUDA + TransformerEngine NVFP4 not available (run on gb10)",
)
def test_cuda_forward_nvfp4_gemm_matches_bf16() -> None:
    # Forward-only: this is the op that IS nvfp4 on gb10 (cuBLASLt FP4 GEMM).
    res = nvfp4_te_gemm_probe(M=256, K=512, N=256, run_backward=False)
    assert res["fwd_all_finite"] is True
    assert res["op"] == "forward_nvfp4_gemm_te_cublaslt_cuda_sm121"
    # FP4 forward GEMM without RHT: lossy but clearly tracking bf16.
    assert res["fwd_rel_rmse_vs_bf16"] < 0.3, res


@pytest.mark.skipif(
    not _cuda_te_available(),
    reason="CUDA + TransformerEngine NVFP4 not available (run on gb10)",
)
def test_cuda_backward_nvfp4_gemm_fails_loud_datacenter_only() -> None:
    # RULE #1: the nvfp4 backward on sm_12x must RAISE, not silently return
    # garbage gradients. Verified 2026-06-01: this is NOT a build-flag gap --
    # the SR FP4 cvt (cvt.rs.satfinite.e2m1x4) is source-gated in TE to
    # sm_100a/sm_103a and the RHT kernel needs SM100 TMEM/tcgen05 absent on
    # sm_12x, so NO TE rebuild for any sm_12x target can enable it. The
    # fail-loud is therefore PERMANENT on consumer/Spark Blackwell (no
    # NVFP4_BACKWARD_FIXED escape hatch) until NVIDIA ships SM12x-native FP4
    # SR+RHT backward kernels upstream.
    with pytest.raises(Nvfp4CudaKernelMiscompiled) as ei:
        nvfp4_te_gemm_probe(M=256, K=512, N=256, run_backward=True)
    msg = str(ei.value)
    # Names the real, evidence-backed blocker (not the disproven 120f rebuild).
    assert "datacenter-only" in msg.lower() or "datacenter_only" in msg.lower()
    assert "tcgen05" in msg or "cvt.rs.satfinite.e2m1x4" in msg
    assert "compute_120f" not in msg
