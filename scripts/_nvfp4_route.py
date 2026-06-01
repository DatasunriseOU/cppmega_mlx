"""NVFP4 (e2m1 + block-scale) training-dtype route for m04_train_step.

NVFP4 is the native Blackwell (sm_120/sm_121, e.g. gb10) 4-bit *training* format:
e2m1 elements with a 2D block scaling (NVIDIA's recipe uses E4M3 block scales over
16-element blocks; the OCP MX e2m1 codec this repo already ships uses fp32 block
scales over 16-element blocks -- the same block geometry). The goal of NVFP4
training is 2x operand memory + throughput over MXFP8.

This module wires the *route* (mirroring fp8_path_c): a carrier dtype, a route
predicate, a precision-route payload, and -- crucially under RULE #1 -- a
FAIL-LOUD gate that RAISES with exactly which training ops lack an NVFP4 kernel
instead of silently downcasting to bf16.

What IS nvfp4 today:
  * (Metal/M4) forward GEMM operands: quantize bf16 -> e2m1 + per-block-16 scale
    via the ``cppmega_mlx.quant.mxfp4_metal`` codec, multiply via the proven
    ``cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c`` cooperative-tensor GEMM
    (commit 29c112d). :func:`nvfp4_gemm_smoke` exercises exactly this.
  * (CUDA/gb10 sm_121) forward NVFP4 GEMM via NVIDIA TransformerEngine
    ``NVFP4BlockScaling`` recipe + ``te.fp8_autocast`` over a ``te.Linear``:
    cuBLASLt e2m1 + E4M3-block-scale matmul. VERIFIED WORKING on gb10
    (rel_err vs bf16 ~0.146, finite). :func:`nvfp4_te_gemm_probe` exercises this.

What is NOT nvfp4 yet -- the route RAISES, it does NOT bf16-fallback (RULE #1):
  * (CUDA/gb10 sm_121) backward / VJP NVFP4 GEMM (dgrad + wgrad). The backward
    stochastic-rounding (SR) FP4 cast and the Random-Hadamard-Transform (RHT)
    fused quant are GENUINELY DATACENTER-ONLY, NOT a build-flag gap. A TE
    rebuild for any sm_12x target CANNOT unblock this. Hard evidence (verified
    2026-06-01 against TE 2.16.0.dev0 @ git 8d1d79bf, source-resident on gb10):
      1. SOURCE GATE. ``common/util/ptx.cuh`` defines
         ``ARCH_HAS_STOCHASTIC_ROUNDING`` as
         ``NVTE_CUDA_ARCH_MATCHES(ptx::ArchSpecific<100>, ptx::ArchSpecific<103>)``.
         ``ArchSpecific<N>`` matches ONLY when ``__CUDA_ARCH__ == N*10`` AND the
         build is the arch-specific ``a`` variant (``__CUDA_ARCH_SPECIFIC__``).
         So SR is enabled ONLY for ``sm_100a`` / ``sm_103a`` (datacenter B200/
         B300). It is NOT in the Blackwell *family* set
         (``FamilySpecific<100>,<110>,<120>``) -- so even ``sm_120f``/``sm_121f``
         leaves ``has_rs == false`` and every SR FP4 cast hits the device-error
         path ``NVTE_DEVICE_ERROR("FP4 cvt PTX instructions are architecture-
         specific. Try recompiling with sm_XXXa instead of sm_XXX.")``. The
         underlying instr is ``cvt.rs.satfinite.e2m1x4.f32`` (round-with-select
         SR FP4 cvt), used by both ``mul_cvt_bf16_to_fp4_8x_stochastic_rounding``
         (ptx.cuh) and ``StochasticNumericConverterBase`` (the RHT kernel).
      2. NO 12x a/f TARGET. nvcc 13.0/13.1/13.2/13.3 ``--list-gpu-code`` emit
         ONLY plain ``sm_120`` / ``sm_121`` for the 12x family -- there is no
         ``sm_120a``/``sm_120f``/``sm_121a``/``sm_121f`` code target to compile
         for. (The ``120f`` family target that helped CUTLASS NVFP4 *MoE forward*
         tactics, cf. cutlass#3096, does not exist as an nvcc -gencode here and
         would not flip the SR source gate anyway.)
      3. NO HARDWARE PATH. The RHT fused kernel
         (``row_cast_col_hadamard_transform_cast_fusion.cu``) is built on the
         CUTLASS SM100 blockscaled pipeline (TMEM load/store, ``SM100_MMA``,
         ``sm100_blockscaled_layout``, cluster launch) = tcgen05/TMEM. NVIDIA
         staff (johnny_nv, NVIDIA Developer Forums, "tcgen05 FP4 ... SM121")
         state SM12x (GB10 / DGX Spark / RTX 50) does NOT implement ``tcgen05``
         and therefore does NOT support the associated FP4 Tensor-Core / TMEM
         blockscaled path. Hence the RHT kernel's "CUDA Error: invalid argument"
         on gb10 -- TMEM ops simply do not exist on this SM.
    VERIFIED BROKEN on gb10, and verified UNFIXABLE by rebuild. Enablement is
    NOT in our hands: it needs NVIDIA to ship SM12x-native (non-tcgen05) FP4 SR
    + RHT backward kernels in a future TE/cuBLASLt release. Until then the
    nvfp4 backward stays fail-loud on consumer/Spark Blackwell. For a full
    training step today use ``--dtype fp8_path_c`` or ``--dtype bfloat16``.
  * optimizer state / update in nvfp4 (AdamW moments stay fp32 only),
  * every non-GEMM op in the HybridTinyLM graph (RMSNorm, SwiGLU, softmax,
    Sparse-MLA attention scores, Mamba3 selective scan, M2RNN recurrence,
    embedding / LM-head, residual adds) -- none have an nvfp4 kernel.

Because a real training step needs all of the above, this route delivers the
honest partial: ``--dtype nvfp4`` is accepted (no argparse rejection), the
nvfp4 forward-GEMM operand path is real and smoke-tested, and the training step
RAISES :class:`Nvfp4TrainingRouteUnavailable` naming the precise missing
kernels. No silent bf16 downcast anywhere.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import mlx.core as mx

NVFP4_DTYPE = "nvfp4"

# NVFP4 carries parameters/activations in bf16 exactly like the fp8_path_* routes
# until the model graph owns native nvfp4 producers; the 4-bit e2m1 payloads are
# materialized per-op at the GEMM boundary, not as the carrier dtype.
NVFP4_CARRIER_DTYPE = "bfloat16"

# OCP-MX / NVFP4 block geometry: 16 elements per block, one scale per block.
NVFP4_BLOCK_SIZE = 16
NVFP4_ELEMENT_FORMAT = "e2m1"
NVFP4_BLOCK_SCALE_FORMAT = "fp32_block_scale_v1"  # repo codec; NVIDIA uses E4M3

NVFP4_ROUTE_KIND = "nvfp4"
NVFP4_E2E_TRAINING_BLOCKER_TYPE = "nvfp4_training_route_unavailable"

# The forward GEMM operand path that IS nvfp4 today. NOTE: the GEMM kernel
# (mxfp4_matmul_path_c) is a Metal/Apple-M4 cooperative-tensor kernel -- it
# RAISES "MLX Metal unavailable" on CUDA/Blackwell (gb10) rather than falling
# back, so on gb10 even this op needs a CUDA/Blackwell nvfp4 GEMM kernel (listed
# in NVFP4_UNSUPPORTED_TRAINING_OPS as ``cuda_blackwell_nvfp4_gemm``). The
# operand codec (e2m1 + block scale) is backend-agnostic and runs anywhere.
NVFP4_SUPPORTED_OPS = (
    # Metal/Apple-M4: e2m1 + block-scale operands through the cooperative-tensor
    # mxfp4 GEMM. Backend-agnostic codec; GEMM kernel is Metal-only.
    "forward_gemm_operands_e2m1_block_scale_metal_m4",
    # CUDA/gb10 sm_121: forward NVFP4 GEMM via TransformerEngine NVFP4BlockScaling
    # (cuBLASLt e2m1 + E4M3 block scale). VERIFIED WORKING on gb10.
    "forward_nvfp4_gemm_te_cublaslt_cuda_sm121",
)

# Ops that have NO working nvfp4 kernel on the target arch. The training step
# RAISES naming these rather than silently running them in bf16 (RULE #1: no
# silent downcast).
NVFP4_UNSUPPORTED_TRAINING_OPS = (
    "cuda_blackwell_nvfp4_gemm_metal_only",  # Metal fwd GEMM does not run on CUDA
    # CUDA/gb10: backward NVFP4 GEMM (dgrad+wgrad). The SR FP4 cast
    # (cvt.rs.satfinite.e2m1x4) + RHT fused quant are DATACENTER-ONLY, source-
    # gated in TE to ArchSpecific<100>/<103> (sm_100a/sm_103a) and built on the
    # CUTLASS SM100 TMEM/tcgen05 pipeline, which sm_12x does NOT implement. A TE
    # rebuild for any sm_12x target CANNOT fix this (nvcc 13.x has no 12x a/f
    # code target; the SR gate is C++ source, not gencode). VERIFIED BROKEN +
    # UNFIXABLE-BY-REBUILD on gb10. Enablement is upstream: NVIDIA must ship
    # SM12x-native FP4 SR+RHT backward kernels in a future TE/cuBLASLt release.
    "nvfp4_gemm_backward_vjp_te_cuda_sm121",
    "gemm_backward_vjp",            # no nvfp4 grad kernel for the Metal e2m1 GEMM
    "optimizer_state_and_update",  # AdamW moments are fp32-only; no nvfp4 state
    "rmsnorm",
    "swiglu_ffn",
    "softmax",
    "sparse_mla_attention",        # nvfp4 attention scores/PV: no kernel
    "mamba3_selective_scan",
    "m2rnn_recurrence",
    "embedding_and_lm_head",
    "residual_add",
)


class Nvfp4TrainingRouteUnavailable(RuntimeError):
    """Raised when --dtype nvfp4 reaches a training op that has no nvfp4 kernel.

    Carries the precise op list so the m04 receipt reports an actionable next
    blocker instead of an argparse rejection or a silent bf16 downcast.
    """


def nvfp4_dtype_requested(
    args: "argparse.Namespace | Any",
) -> bool:
    """Return whether the CLI/config dtype names the NVFP4 route."""

    return str(getattr(args, "dtype", "")).strip().lower() == NVFP4_DTYPE


# The NVFP4 route has no AUTO long-seq gate of its own (unlike fp8_path_c, whose
# Path C arenas blow up at long seq). nvfp4_route_requested == dtype_requested.
nvfp4_route_requested = nvfp4_dtype_requested


def nvfp4_gemm_smoke(
    *,
    M: int = 16,
    N: int = 32,
    K: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Exercise the one op that IS nvfp4: forward e2m1 + block-scale GEMM.

    Quantizes two bf16 operands to e2m1 + per-block-16 scale with the committed
    mxfp4 codec, multiplies them through the M4-verified cooperative-tensor
    ``mxfp4_matmul_path_c`` GEMM, and returns finite-ness + the relative error
    vs the bf16 reference. Shapes default to the minimal cooperative tile
    (M%16==0, N%32==0, K%16==0). RAISES (never falls back) if Metal/kernel is
    unavailable or a shape admits no cooperative tile.
    """

    from cppmega_mlx.quant.mxfp4_metal import quantize_mxfp4_blockwise
    from cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c import (
        MXFP4_BLOCK_SIZE,
        mxfp4_matmul_path_c,
    )

    if K % NVFP4_BLOCK_SIZE != 0:
        raise Nvfp4TrainingRouteUnavailable(
            f"nvfp4_gemm_smoke: K={K} must be a multiple of the block size "
            f"{NVFP4_BLOCK_SIZE}"
        )

    mx.random.seed(seed)
    a = mx.random.normal((M, K)).astype(mx.bfloat16)
    b = mx.random.normal((N, K)).astype(mx.bfloat16)

    def _quant_rows(mat: mx.array, rows: int, cols: int) -> tuple[mx.array, mx.array]:
        # Per-row e2m1 + block-scale quantization, packed to (rows, cols//2)
        # uint8 nibbles and (rows, cols//16) fp32 scales -- the layout the
        # mxfp4_matmul_path_c kernel consumes.
        q_rows = []
        s_rows = []
        bytes_per_row = (cols + 1) // 2
        n_blocks = cols // NVFP4_BLOCK_SIZE
        for r in range(rows):
            qd, sc = quantize_mxfp4_blockwise(
                mat[r], block_size=MXFP4_BLOCK_SIZE
            )
            if int(qd.shape[0]) != bytes_per_row:
                raise Nvfp4TrainingRouteUnavailable(
                    f"nvfp4_gemm_smoke: packed row len {int(qd.shape[0])} != "
                    f"{bytes_per_row} for cols={cols}"
                )
            if int(sc.shape[0]) != n_blocks:
                raise Nvfp4TrainingRouteUnavailable(
                    f"nvfp4_gemm_smoke: scale row len {int(sc.shape[0])} != "
                    f"{n_blocks} blocks"
                )
            q_rows.append(qd.reshape(1, bytes_per_row))
            s_rows.append(sc.reshape(1, n_blocks))
        return mx.concatenate(q_rows, axis=0), mx.concatenate(s_rows, axis=0)

    a_q, a_s = _quant_rows(a, M, K)
    b_q, b_s = _quant_rows(b, N, K)
    out = mx.zeros((M, N), dtype=mx.float32)

    out = mxfp4_matmul_path_c(
        a_q, a_s, b_q, b_s, M=M, N=N, K=K, out=out,
        block_size=MXFP4_BLOCK_SIZE,
    )
    mx.eval(out)

    ref = (a.astype(mx.float32) @ b.astype(mx.float32).T)
    mx.eval(ref)
    all_finite = bool(mx.all(mx.isfinite(out)))
    diff = (out - ref).reshape(-1)
    denom = float(mx.sqrt(mx.mean(ref.reshape(-1) * ref.reshape(-1)))) + 1e-12
    rel_rmse = float(mx.sqrt(mx.mean(diff * diff))) / denom

    return {
        "op": NVFP4_SUPPORTED_OPS[0],
        "M": M,
        "N": N,
        "K": K,
        "block_size": NVFP4_BLOCK_SIZE,
        "element_format": NVFP4_ELEMENT_FORMAT,
        "block_scale_format": NVFP4_BLOCK_SCALE_FORMAT,
        "all_finite": all_finite,
        "rel_rmse_vs_bf16": rel_rmse,
        "kernel": "cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c.mxfp4_matmul_path_c",
        "codec": "cppmega_mlx.quant.mxfp4_metal.quantize_mxfp4_blockwise",
    }


# Exact TransformerEngine assertion substrings that prove the FP4 SR-cvt / RHT
# kernels are unavailable on this arch. The SR FP4 cast (cvt.rs.satfinite.
# e2m1x4) is source-gated in TE to ArchSpecific<100>/<103> (sm_100a/sm_103a),
# and the RHT fused kernel needs the SM100 TMEM/tcgen05 pipeline -- neither of
# which exists on sm_12x. We match on these to fail loud with WHERE+WHAT rather
# than let TE return garbage gradients (which it does silently for the SR path).
NVFP4_TE_ARCH_PTX_MARKERS = (
    "FP4 cvt PTX instructions are architecture-specific",
    "Try recompiling with sm_XXXa",
    "row_col_rht_gemm",
    "row_cast_col_hadamard_transform_cast_fusion",
)


class Nvfp4CudaKernelMiscompiled(Nvfp4TrainingRouteUnavailable):
    """Raised when TE reports nvfp4 'available' but its FP4 stochastic-rounding
    cast / RHT backward kernels are not executable on this (consumer/Spark
    Blackwell, sm_12x) arch, so the backward mis-executes.

    This is NOT a build-flag gap: the SR FP4 cvt (``cvt.rs.satfinite.e2m1x4``)
    is source-gated in TE to ``ArchSpecific<100>``/``<103>`` (sm_100a/sm_103a
    datacenter) and the RHT kernel is built on the SM100 TMEM/tcgen05 pipeline
    that sm_12x does not implement -- so a TE rebuild for any sm_12x target
    cannot enable it. This is the RULE #1 guard against TE's misleading
    ``check_nvfp4_support()`` returning True while the SR FP4 cast silently
    produces garbage gradients on gb10 (sm_121).
    """


def nvfp4_te_cuda_capability() -> dict[str, Any]:
    """Probe the real CUDA/TransformerEngine NVFP4 capability on this host.

    Returns a dict describing whether TE reports nvfp4 'available', the GPU
    compute capability, and the TE-compiled SASS arch list (best effort). Does
    NOT silently swallow import errors -- if TE/torch is absent it records the
    precise reason. This is a *probe*, not a gate; the gate is
    :func:`nvfp4_te_gemm_probe` which actually runs the kernels.
    """

    info: dict[str, Any] = {
        "torch_available": False,
        "te_available": False,
        "cuda_device": None,
        "compute_capability": None,
        "te_version": None,
        "te_check_nvfp4_support": None,
        "te_is_nvfp4_available": None,
        "import_error": None,
    }
    try:
        import torch  # noqa: PLC0415

        info["torch_available"] = True
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"sm_{cc[0]}{cc[1]}"
    except Exception as exc:  # torch genuinely absent -> record, do not hide
        info["import_error"] = f"torch: {exc!r}"
        return info
    try:
        import transformer_engine  # noqa: PLC0415
        import transformer_engine.pytorch as te  # noqa: PLC0415
        from transformer_engine.pytorch.fp8 import (  # noqa: PLC0415
            check_nvfp4_support,
        )

        info["te_available"] = True
        info["te_version"] = getattr(transformer_engine, "__version__", None)
        info["te_check_nvfp4_support"] = check_nvfp4_support()
        info["te_is_nvfp4_available"] = bool(te.is_nvfp4_available())
    except Exception as exc:
        info["import_error"] = f"transformer_engine: {exc!r}"
    return info


def nvfp4_te_gemm_probe(
    *,
    M: int = 256,
    K: int = 512,
    N: int = 256,
    seed: int = 0,
    run_backward: bool = True,
) -> dict[str, Any]:
    """Exercise the REAL CUDA NVFP4 GEMM (TransformerEngine) and report honestly.

    Forward NVFP4 GEMM (cuBLASLt e2m1 + E4M3 block scale via TE
    ``NVFP4BlockScaling`` + ``te.fp8_autocast``) is VERIFIED WORKING on gb10
    (sm_121). The backward (dgrad/wgrad with stochastic-rounding FP4 cast and
    RHT) is VERIFIED BROKEN there, and verified UNFIXABLE by any sm_12x rebuild:
    the SR cvt (``cvt.rs.satfinite.e2m1x4``) is TE-source-gated to sm_100a/
    sm_103a and the RHT kernel needs the SM100 TMEM/tcgen05 pipeline that sm_12x
    does not implement (datacenter-only).

    This function:
      * runs the forward NVFP4 GEMM and returns its rel_err vs bf16 (real work);
      * if ``run_backward`` and the FP4 SR-cvt / RHT kernels are unavailable on
        this arch, RAISES :class:`Nvfp4CudaKernelMiscompiled` (RULE #1 fail-loud)
        instead of returning the garbage gradients TE would otherwise hand back
        silently.

    RAISES if torch/TE/CUDA is unavailable (no Metal/bf16 fallback).
    """

    import torch  # noqa: PLC0415
    import transformer_engine.pytorch as te  # noqa: PLC0415
    from transformer_engine.common.recipe import NVFP4BlockScaling  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise Nvfp4TrainingRouteUnavailable(
            "nvfp4_te_gemm_probe: CUDA is not available; the TE NVFP4 GEMM is a "
            "CUDA/Blackwell (sm_120/sm_121) kernel. Run on gb10."
        )
    cc = torch.cuda.get_device_capability(0)
    arch = f"sm_{cc[0]}{cc[1]}"
    torch.manual_seed(seed)
    dev = "cuda"

    lin = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(dev)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, requires_grad=True)

    with torch.no_grad():
        y_bf = lin(x)
        g = torch.randn_like(y_bf)

    def _rel(a: "torch.Tensor", b: "torch.Tensor") -> float:
        return (a.float() - b.float()).norm().item() / (
            b.float().norm().item() + 1e-12
        )

    # Forward NVFP4 GEMM. RHT + SR are forward-irrelevant for the fwd-inp/weight
    # casts on gb10; disable them so the forward path uses only the working
    # cuBLASLt FP4 GEMM (the broken bits are the SR/RHT *backward* casts).
    fwd_recipe = NVFP4BlockScaling(
        disable_rht=True, disable_stochastic_rounding=True
    )
    with torch.no_grad(), te.fp8_autocast(enabled=True, fp8_recipe=fwd_recipe):
        y_fp4 = lin(x)
    torch.cuda.synchronize()
    fwd_rel = _rel(y_fp4, y_bf)
    fwd_finite = bool(torch.isfinite(y_fp4).all())

    result: dict[str, Any] = {
        "op": "forward_nvfp4_gemm_te_cublaslt_cuda_sm121",
        "arch": arch,
        "M": M,
        "N": N,
        "K": K,
        "element_format": NVFP4_ELEMENT_FORMAT,
        "block_size": NVFP4_BLOCK_SIZE,
        "fwd_rel_rmse_vs_bf16": fwd_rel,
        "fwd_all_finite": fwd_finite,
        "kernel": "transformer_engine.pytorch (NVFP4BlockScaling, cuBLASLt FP4)",
        "backward_attempted": run_backward,
    }

    if not run_backward:
        return result

    # Backward NVFP4 GEMM under the PRODUCTION-DEFAULT recipe (RHT on fwd_inp +
    # bwd, stochastic rounding on bwd). CRITICAL: use a FRESH te.Linear so no
    # quantizer/workspace state cached from the disable_rht forward above can
    # mask the RHT failure -- that masking would be a silent fallback (RULE #1).
    #
    # IMPORTANT: this kernel error is ASYNC. Set CUDA_LAUNCH_BLOCKING=1 so the
    # RHT fused-kernel "CUDA error: invalid argument" surfaces here as a Python
    # exception rather than corrupting the context silently downstream.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    lin_bwd = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(dev)
    with torch.no_grad():
        lin_bwd.weight.copy_(lin.weight)

    # Clean bf16 reference grads on the fresh module (plain autograd, no autocast).
    x_ref = x.detach().clone().requires_grad_(True)
    lin_bwd.weight.grad = None
    y_bf_ref = lin_bwd(x_ref)
    y_bf_ref.backward(g)
    torch.cuda.synchronize()
    xg_bf = x_ref.grad.detach().clone()
    wg_bf = lin_bwd.weight.grad.detach().clone()

    x_fp4 = x.detach().clone().requires_grad_(True)
    lin_bwd.weight.grad = None
    bwd_recipe = NVFP4BlockScaling()  # full default: fwd RHT + bwd RHT + SR
    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=bwd_recipe):
            y = lin_bwd(x_fp4)
        y.backward(g)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if any(m in msg for m in NVFP4_TE_ARCH_PTX_MARKERS):
            raise Nvfp4CudaKernelMiscompiled(
                f"nvfp4 BACKWARD GEMM is unavailable on {arch}: TransformerEngine "
                f"reported nvfp4 'available' but its FP4 stochastic-rounding cast "
                f"(cvt.rs.satfinite.e2m1x4) and Random-Hadamard fused kernels are "
                f"DATACENTER-ONLY and NOT executable on consumer/Spark Blackwell. "
                f"Underlying TE error: {msg.strip()[:300]}. This is NOT a build-"
                f"flag gap: TE source-gates the SR cvt to ArchSpecific<100>/<103> "
                f"(sm_100a/sm_103a) -- it is excluded from the sm_120 FAMILY set "
                f"-- and the RHT kernel needs the SM100 TMEM/tcgen05 pipeline that "
                f"sm_12x does not implement (NVIDIA confirms SM12x has no tcgen05). "
                f"nvcc 13.x exposes no sm_12x a/f code target, so a TE rebuild "
                f"CANNOT enable this. Enablement is upstream: NVIDIA must ship "
                f"SM12x-native FP4 SR+RHT backward kernels in a future TE/cuBLASLt "
                f"release. For a full training step today use --dtype fp8_path_c "
                f"or --dtype bfloat16."
            ) from exc
        raise

    # If TE did NOT raise, the SR FP4 cast path may have silently mis-executed
    # and the gradients are garbage (rel_err ~1.0 vs bf16). Detect + FAIL LOUD.
    xg_fp4 = x_fp4.grad
    wg_fp4 = lin_bwd.weight.grad
    if xg_fp4 is None or wg_fp4 is None:
        raise Nvfp4CudaKernelMiscompiled(
            f"nvfp4 BACKWARD GEMM produced no gradients on {arch}: the TE FP4 "
            f"backward did not populate x.grad / weight.grad. The FP4 SR cvt + "
            f"RHT kernels are datacenter-only (SR source-gated to sm_100a/sm_103a; "
            f"RHT needs SM100 TMEM/tcgen05 absent on sm_12x). NOT fixable by a TE "
            f"rebuild for any sm_12x target; enablement is upstream (NVIDIA must "
            f"ship SM12x-native FP4 SR+RHT backward kernels)."
        )
    dgrad_rel = _rel(xg_fp4, xg_bf)
    wgrad_rel = _rel(wg_fp4, wg_bf)
    if dgrad_rel > 0.6 or wgrad_rel > 0.6:
        raise Nvfp4CudaKernelMiscompiled(
            f"nvfp4 BACKWARD GEMM produced garbage gradients on {arch} "
            f"(dgrad rel_err={dgrad_rel:.3f}, wgrad rel_err={wgrad_rel:.3f}; "
            f"a working FP4 bwd is well under 0.5). The TE stochastic-rounding "
            f"FP4 cast did not surface a hard error but mis-executed: the SR cvt "
            f"(cvt.rs.satfinite.e2m1x4) is datacenter-only (source-gated to "
            f"sm_100a/sm_103a) and unavailable on sm_12x. NOT fixable by a TE "
            f"rebuild for any sm_12x target; enablement is upstream."
        )
    result["backward_ok"] = True
    result["dgrad_rel_rmse_vs_bf16"] = dgrad_rel
    result["wgrad_rel_rmse_vs_bf16"] = wgrad_rel
    return result


def nvfp4_training_route_payload(
    args: "argparse.Namespace | Any",
) -> dict[str, Any]:
    """precision_route payload for the NVFP4 route (mirrors fp8_path_c shape)."""

    requested = nvfp4_route_requested(args)
    return {
        "requested": NVFP4_DTYPE,
        "kind": NVFP4_ROUTE_KIND,
        "status": NVFP4_E2E_TRAINING_BLOCKER_TYPE,
        "blocker_type": NVFP4_E2E_TRAINING_BLOCKER_TYPE if requested else None,
        "carrier_dtype": NVFP4_CARRIER_DTYPE,
        "element_format": NVFP4_ELEMENT_FORMAT,
        "block_size": NVFP4_BLOCK_SIZE,
        "block_scale_format": NVFP4_BLOCK_SCALE_FORMAT,
        "nvfp4_supported_ops": list(NVFP4_SUPPORTED_OPS),
        "nvfp4_unsupported_training_ops": list(NVFP4_UNSUPPORTED_TRAINING_OPS),
        "forward_gemm_kernel_metal": (
            "cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c.mxfp4_matmul_path_c"
        ),
        "forward_gemm_kernel_cuda": (
            "transformer_engine.pytorch NVFP4BlockScaling + cuBLASLt (sm_121)"
        ),
        "backward_gemm_cuda_status": (
            "broken_on_gb10_sm121_fp4_sr_cvt_and_rht_are_datacenter_only_"
            "not_fixable_by_te_rebuild"
        ),
        "backward_gemm_cuda_enablement": (
            "UPSTREAM ONLY: the SR FP4 cvt (cvt.rs.satfinite.e2m1x4) is "
            "source-gated in TE to sm_100a/sm_103a and the RHT kernel needs "
            "SM100 TMEM/tcgen05 absent on sm_12x; nvcc 13.x exposes no sm_12x "
            "a/f target, so a TE rebuild cannot enable it. Needs NVIDIA to ship "
            "SM12x-native FP4 SR+RHT backward kernels in a future TE/cuBLASLt."
        ),
        "operand_codec": (
            "cppmega_mlx.quant.mxfp4_metal.quantize_mxfp4_blockwise"
        ),
        "full_end_to_end_training_available": False,
        "large_tensor_staging_allowed": False,
        "hidden_wrapper_quantization_allowed": False,
        "kernel_boundary_quantization_allowed": False,
    }


def nvfp4_training_route_unavailable_reason() -> str:
    """The precise, actionable RAISE message for the NVFP4 training step."""

    ops = ", ".join(NVFP4_UNSUPPORTED_TRAINING_OPS)
    return (
        "nvfp4 is wired as a training dtype route (carrier bf16, e2m1 + "
        f"per-block-{NVFP4_BLOCK_SIZE} scale operands, E4M3 block scales). The "
        "FORWARD nvfp4 GEMM is real and verified on both backends: Metal/M4 "
        "(cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c) and CUDA/gb10 sm_121 "
        "(TransformerEngine NVFP4BlockScaling + cuBLASLt, rel_err vs bf16 "
        "~0.146). A full training step additionally needs nvfp4 kernels that "
        "are MISSING or BROKEN on the available arch: "
        f"{ops}. The decisive blocker on gb10 is the nvfp4 BACKWARD GEMM, and it "
        "is GENUINELY DATACENTER-ONLY -- NOT a build-flag gap fixable by a TE "
        "rebuild. The backward stochastic-rounding FP4 cast uses "
        "'cvt.rs.satfinite.e2m1x4', which TransformerEngine source-gates to "
        "ArchSpecific<100>/<103> (sm_100a/sm_103a) -- it is excluded from the "
        "sm_120 family set, so any sm_12x build leaves has_rs=false and the cast "
        "asserts 'FP4 cvt PTX instructions are architecture-specific'. The "
        "Random-Hadamard fused kernel is built on the CUTLASS SM100 TMEM/tcgen05 "
        "pipeline, which sm_12x (GB10/DGX Spark/RTX50) does NOT implement (per "
        "NVIDIA), so it raises 'CUDA Error: invalid argument'. nvcc 13.x exposes "
        "no sm_12x a/f code target. Rather than silently downcast these ops to "
        "bf16 (forbidden), the nvfp4 route fails loud here. Enablement is "
        "UPSTREAM ONLY: NVIDIA must ship SM12x-native FP4 SR + RHT backward "
        "kernels in a future TransformerEngine/cuBLASLt release. For a full "
        "training step TODAY run --dtype fp8_path_c or --dtype bfloat16."
    )


def raise_if_nvfp4_training_unsupported(
    args: "argparse.Namespace | Any",
) -> None:
    """RAISE the precise nvfp4 training blocker if --dtype nvfp4 is requested.

    Call this on the training critical path BEFORE any op would otherwise run
    in bf16. This is the RULE #1 fail-loud guard: it never returns a degraded
    bf16 path for nvfp4; it raises with WHERE (this guard) + WHAT (the missing
    nvfp4 training kernels).
    """

    if not nvfp4_route_requested(args):
        return
    raise Nvfp4TrainingRouteUnavailable(nvfp4_training_route_unavailable_reason())
