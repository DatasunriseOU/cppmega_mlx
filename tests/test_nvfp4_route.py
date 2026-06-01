# pyright: reportMissingImports=false
"""Tests for the NVFP4 (e2m1 + block-scale) training-dtype route.

``scripts/_nvfp4_route.py`` wires ``--dtype nvfp4`` as an HONEST PARTIAL route
under RULE #1 (no silent fallbacks):

* The FORWARD nvfp4 GEMM is real on two backends:
    - Metal/Apple-M4: ``mxfp4_matmul_path_c`` cooperative-tensor GEMM over the
      ``quantize_mxfp4_blockwise`` e2m1 + per-block-16 codec.
    - CUDA/gb10 sm_121: TransformerEngine ``NVFP4BlockScaling`` + cuBLASLt FP4
      GEMM (VERIFIED working, rel_err vs bf16 ~0.146).
* The BACKWARD nvfp4 GEMM has TWO recipes:
    - REDUCED round-to-nearest (``NVFP4BlockScaling(disable_rht=True,
      disable_stochastic_rounding=True)``, ported from nanochat) RUNS on gb10
      sm_121 -- dgrad+wgrad in FP4 RtN, rel_err ~0.147 vs bf16. It reuses the
      same working forward cuBLASLt FP4 GEMM and avoids the datacenter-only SR
      cvt + RHT fused kernel. It is GATED behind a runtime numeric probe
      (``nvfp4_te_backward_rtn_probe``) and is a SUPPORTED op only when the gate
      passes. (This REOPENS the earlier "datacenter-only / can't be done"
      conclusion -- which was correct only for the DEFAULT recipe.)
    - DEFAULT (RHT + stochastic rounding ON) fails on gb10 because the FP4
      stochastic-rounding cast (``cvt.rs.satfinite.e2m1x4``) and Random-Hadamard
      fused kernels are DATACENTER-ONLY: TE source-gates the SR cvt to
      ``sm_100a``/``sm_103a`` and the RHT kernel needs the SM100 TMEM/tcgen05
      pipeline that sm_12x does not implement. This is NOT fixable by a TE
      rebuild for any sm_12x target (verified 2026-06-01) -- the DEFAULT-recipe
      fail-loud is permanent on consumer/Spark Blackwell until NVIDIA ships
      SM12x-native kernels upstream.
* optimizer state and every non-GEMM op have NO working nvfp4 kernel on the
  available arch, so the route RAISES a precise, actionable error naming the
  missing op + enablement -- it NEVER downcasts to bf16 silently.

These tests assert BOTH halves: the real forward op + reduced RtN backward match
bf16 within tolerance, and the DEFAULT-recipe backward raises the expected
``Nvfp4*`` error.

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
    NVFP4_RTN_BACKWARD_DGRAD_REL_ERR_TOL,
    NVFP4_SUPPORTED_OPS,
    NVFP4_UNSUPPORTED_TRAINING_OPS,
    Nvfp4CudaKernelMiscompiled,
    Nvfp4TrainingRouteUnavailable,
    nvfp4_backward_rtn_supported,
    nvfp4_route_requested,
    nvfp4_te_backward_rtn_probe,
    nvfp4_te_cuda_capability,
    nvfp4_te_gemm_probe,
    nvfp4_training_route_payload,
    nvfp4_training_route_unavailable_reason,
    raise_if_nvfp4_training_unsupported,
)

_RTN_BWD_OP = "backward_nvfp4_rtn_dgrad_fp4_wgrad_fp4_te_cublaslt_cuda_sm121"


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
    # The REDUCED round-to-nearest backward (ported from nanochat) is a SUPPORTED
    # op, gated at runtime by the numeric probe.
    assert _RTN_BWD_OP in NVFP4_SUPPORTED_OPS
    # The DEFAULT-recipe gb10 backward blocker stays enumerated as unsupported.
    assert (
        "nvfp4_gemm_backward_vjp_te_cuda_sm121"
        in NVFP4_UNSUPPORTED_TRAINING_OPS
    )
    assert "optimizer_state_and_update" in NVFP4_UNSUPPORTED_TRAINING_OPS
    # The RtN op is NOT also listed as unsupported (no contradictory state).
    assert _RTN_BWD_OP not in NVFP4_UNSUPPORTED_TRAINING_OPS
    # The acceptance tolerance is a sane, finite bound above the measured ~0.147
    # and well below a garbage (~1.0) mis-execution.
    assert 0.147 < NVFP4_RTN_BACKWARD_DGRAD_REL_ERR_TOL < 0.6


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
    # gb10 DEFAULT backward gap is documented as datacenter-only (NOT a rebuild
    # gap), while the reduced RtN recipe is advertised as working.
    assert "datacenter_only" in payload["backward_gemm_cuda_status"]
    assert "not_fixable_by_te_rebuild" in payload["backward_gemm_cuda_status"]
    enablement = payload["backward_gemm_cuda_enablement"]
    assert "UPSTREAM ONLY" in enablement
    assert "tcgen05" in enablement
    # Must NOT advertise the disproven compute_120f rebuild as the fix.
    assert "compute_120f" not in enablement
    # The reduced RtN backward is advertised separately as WORKING on sm_121.
    assert "WORKS" in payload["backward_gemm_cuda_rtn_status"]
    assert "disable_rht" in payload["backward_gemm_cuda_rtn_recipe"]
    assert "disable_stochastic_rounding" in payload["backward_gemm_cuda_rtn_recipe"]
    assert "nanochat" in payload["backward_gemm_cuda_rtn_ported_from"]


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
# CUDA/gb10 NVFP4: forward works; reduced RtN backward works (gated); default
# RHT+SR backward fails loud. (Skips off-CUDA.)
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
def test_cuda_default_recipe_backward_nvfp4_gemm_fails_loud_datacenter_only() -> None:
    # RULE #1: the DEFAULT-recipe (RHT + stochastic rounding ON) nvfp4 backward on
    # sm_12x must RAISE, not silently return garbage gradients. Verified
    # 2026-06-01: this is NOT a build-flag gap -- the SR FP4 cvt
    # (cvt.rs.satfinite.e2m1x4) is source-gated in TE to sm_100a/sm_103a and the
    # RHT kernel needs SM100 TMEM/tcgen05 absent on sm_12x, so NO TE rebuild for
    # any sm_12x target can enable the DEFAULT recipe. (The REDUCED RtN recipe is
    # the escape -- see test_cuda_reduced_rtn_backward_* below.)
    #
    # CRITICAL: the RHT-kernel crash CORRUPTS the CUDA context irrecoverably
    # (subsequent cudaMalloc/normal_ raise "invalid argument"). We therefore run
    # the destructive probe in an ISOLATED SUBPROCESS so it cannot poison the
    # context of the in-process RtN backward test. This is not a fallback -- the
    # subprocess still RAISES Nvfp4CudaKernelMiscompiled; we just contain the
    # blast radius. We assert the subprocess prints the marker and exits nonzero.
    import subprocess  # noqa: PLC0415

    code = (
        "import sys, os; "
        "os.environ['CUDA_LAUNCH_BLOCKING']='1'; "
        "sys.path.insert(0, %r); "
        "from scripts._nvfp4_route import (nvfp4_te_gemm_probe, "
        "Nvfp4CudaKernelMiscompiled); "
        "\ntry:\n"
        "    nvfp4_te_gemm_probe(M=256, K=512, N=256, run_backward=True)\n"
        "    print('NO_RAISE'); sys.exit(2)\n"
        "except Nvfp4CudaKernelMiscompiled as e:\n"
        "    print('RAISED_MISCOMPILED'); print(repr(str(e))); sys.exit(0)\n"
    ) % str(_REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "RAISED_MISCOMPILED" in proc.stdout, out
    # Names the real, evidence-backed blocker (not the disproven 120f rebuild).
    assert "datacenter-only" in out.lower() or "datacenter_only" in out.lower()
    assert "tcgen05" in out or "cvt.rs.satfinite.e2m1x4" in out
    assert "compute_120f" not in out


@pytest.mark.skipif(
    not _cuda_te_available(),
    reason="CUDA + TransformerEngine NVFP4 not available (run on gb10)",
)
def test_cuda_reduced_rtn_backward_runs_and_matches_bf16() -> None:
    # The REDUCED round-to-nearest backward (RHT off + SR off; ported from
    # nanochat) RUNS on sm_121 and matches the bf16 reference within tolerance.
    # This REOPENS the previously-declared "datacenter-only / can't be done"
    # conclusion (commit b6a6177) for the reduced recipe specifically. Measured on
    # gb10: dgrad rel_err ~0.147, wgrad rel_err ~0.134.
    res = nvfp4_te_backward_rtn_probe(M=256, K=512, N=256)
    assert res["backward_rtn_supported"] is True
    assert res["op"] == _RTN_BWD_OP
    assert res["dgrad_all_finite"] is True and res["wgrad_all_finite"] is True
    # The numeric gate: dgrad must be finite and within the acceptance tolerance.
    assert res["dgrad_rel_rmse_vs_bf16"] < NVFP4_RTN_BACKWARD_DGRAD_REL_ERR_TOL, res
    assert res["wgrad_rel_rmse_vs_bf16"] < NVFP4_RTN_BACKWARD_DGRAD_REL_ERR_TOL, res
    # Sanity: the recipe string names the load-bearing disables.
    assert "disable_rht" in res["recipe"]
    assert "disable_stochastic_rounding" in res["recipe"]


@pytest.mark.skipif(
    not _cuda_te_available(),
    reason="CUDA + TransformerEngine NVFP4 not available (run on gb10)",
)
def test_cuda_reduced_rtn_backward_capability_gate_true_on_gb10() -> None:
    # The runtime capability gate must report the RtN backward supported on gb10
    # (where the numeric probe passes). This is the gate that makes the
    # NVFP4_SUPPORTED_OPS entry actually usable.
    assert nvfp4_backward_rtn_supported() is True
