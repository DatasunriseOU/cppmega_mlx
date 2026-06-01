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

What IS nvfp4 backward today (verified 2026-06-01, ported from nanochat):
  * (CUDA/gb10 sm_121) backward / VJP NVFP4 GEMM under the REDUCED round-to-
    nearest (RtN) recipe ``NVFP4BlockScaling(disable_rht=True,
    disable_stochastic_rounding=True)``. With RHT and stochastic-rounding OFF,
    the backward grad-quant degrades to plain RtN E2M1 + E4M3 16-element block
    scale -- the SAME cuBLASLt CUDA_R_4F_E2M1 x UE4M3 path the forward already
    proves works. This avoids BOTH the datacenter-only SR cvt
    (``cvt.rs.satfinite.e2m1x4``) AND the RHT fused kernel
    (``row_col_rht_gemm_ntt_w_sfc``) that crash the default recipe. MEASURED on
    gb10 (sm_121, TE 2.16.0.dev0+a46079cb): fwd rel_err 0.147, dgrad rel_err
    0.1465, wgrad rel_err 0.1343 vs a pure-bf16 te.Linear reference -- finite,
    deterministic (SR off), reproducible. This was previously declared
    "datacenter-only / impossible" (commit b6a6177); that conclusion was correct
    ONLY for the DEFAULT RHT+SR recipe and was too broad. The reduced RtN
    backward is gated behind a runtime NUMERIC probe (:func:`nvfp4_te_backward_rtn_probe`)
    and only marked supported when dgrad rel_err is finite and within tolerance.
    NOTE: nanochat's TE used ``override_linear_precision=(False,False,True)`` to
    force wgrad->BF16; THIS TE build (2.16.0.dev0+a46079cb) has NO
    ``override_linear_precision`` kwarg -- the equivalent levers are
    ``backward_override in {None,'high_precision','dequantized'}``. With the bare
    ``disable_rht+disable_stochastic_rounding`` recipe, BOTH dgrad and wgrad run
    in FP4 (RtN) and pass the gate, so we do NOT need the wgrad-BF16 split on this
    build (it is available via ``backward_override='high_precision'`` if a future
    TE regresses wgrad). Accuracy caveat (RULE #1 honesty): the probe proves
    "runs + matches bf16 within ~0.15 rel_err"; it does NOT prove full-convergence
    accuracy parity over a long training run. nanochat's receipt (loss 11.09->6.58
    over 22 steps) is the integration evidence it trains.

What is STILL NOT nvfp4 -- the route RAISES, it does NOT bf16-fallback (RULE #1):
  * (CUDA/gb10 sm_121) backward / VJP NVFP4 GEMM under the DEFAULT recipe
    (``NVFP4BlockScaling()`` with RHT + stochastic rounding ON). The backward
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
    VERIFIED BROKEN on gb10, and verified UNFIXABLE by rebuild for the DEFAULT
    RHT+SR recipe. Enablement of the DEFAULT recipe is NOT in our hands: it needs
    NVIDIA to ship SM12x-native (non-tcgen05) FP4 SR + RHT backward kernels in a
    future TE/cuBLASLt release. The DEFAULT-recipe backward stays fail-loud on
    consumer/Spark Blackwell. (The REDUCED RtN recipe above sidesteps all of this
    and DOES run -- see "What IS nvfp4 backward today".) For a full untouched
    nvfp4 training step today use ``--dtype fp8_path_c`` or ``--dtype bfloat16``.
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
    # CUDA/gb10 sm_121: REDUCED round-to-nearest backward NVFP4 GEMM (dgrad+wgrad
    # in FP4 RtN) via NVFP4BlockScaling(disable_rht=True,
    # disable_stochastic_rounding=True). Avoids the datacenter-only SR cvt + RHT
    # fused kernel; reuses the same working forward cuBLASLt FP4 GEMM. Ported from
    # nanochat (CHANGELOG_GB10.md:240-268). VERIFIED WORKING on gb10
    # (dgrad rel_err ~0.147, wgrad rel_err ~0.134 vs bf16). This op is GATED at
    # runtime by nvfp4_te_backward_rtn_probe() -- it is "supported" only when the
    # numeric gate passes on the actual device. On any host where the gate does
    # NOT pass (e.g. a future arch that breaks the RtN path), the route falls back
    # to the DEFAULT-recipe fail-loud raise; it never silently degrades to bf16.
    "backward_nvfp4_rtn_dgrad_fp4_wgrad_fp4_te_cublaslt_cuda_sm121",
)

# The reduced-recipe RtN backward acceptance threshold. The forward FP4 GEMM
# measures rel_err ~0.147 on gb10; the backward dgrad/wgrad measure ~0.147/0.134.
# We accept the RtN backward as "supported" only when dgrad rel_err is finite and
# below this bound. 0.35 leaves headroom over the measured 0.147 for shape/seed
# variation while still rejecting a garbage (~1.0) mis-execution. (RULE #1: this
# is a numeric GATE, not a rubber stamp -- if the kernel ever mis-executes the
# gate fails and the route stays fail-loud.)
NVFP4_RTN_BACKWARD_DGRAD_REL_ERR_TOL = 0.35

# Ops that have NO working nvfp4 kernel on the target arch. The training step
# RAISES naming these rather than silently running them in bf16 (RULE #1: no
# silent downcast).
NVFP4_UNSUPPORTED_TRAINING_OPS = (
    "cuda_blackwell_nvfp4_gemm_metal_only",  # Metal fwd GEMM does not run on CUDA
    # CUDA/gb10: backward NVFP4 GEMM under the DEFAULT recipe (RHT + stochastic
    # rounding ON). The SR FP4 cast (cvt.rs.satfinite.e2m1x4) + RHT fused quant
    # are DATACENTER-ONLY, source-gated in TE to ArchSpecific<100>/<103>
    # (sm_100a/sm_103a) and built on the CUTLASS SM100 TMEM/tcgen05 pipeline,
    # which sm_12x does NOT implement. A TE rebuild for any sm_12x target CANNOT
    # fix this (nvcc 13.x has no 12x a/f code target; the SR gate is C++ source,
    # not gencode). VERIFIED BROKEN + UNFIXABLE-BY-REBUILD on gb10. Enablement of
    # the DEFAULT recipe is upstream: NVIDIA must ship SM12x-native FP4 SR+RHT
    # backward kernels. NOTE: the REDUCED RtN recipe (disable_rht=True,
    # disable_stochastic_rounding=True) DOES run -- see
    # "backward_nvfp4_rtn_dgrad_fp4_wgrad_fp4_te_cublaslt_cuda_sm121" in
    # NVFP4_SUPPORTED_OPS. Only the DEFAULT recipe remains blocked.
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


def build_nvfp4_rtn_recipe() -> "Any":
    """Build the REDUCED round-to-nearest NVFP4 recipe ported from nanochat.

    Returns a ``NVFP4BlockScaling`` configured to AVOID the datacenter-only
    backward kernels: ``disable_rht=True`` (no Random-Hadamard fused quant) +
    ``disable_stochastic_rounding=True`` (no ``cvt.rs.satfinite.e2m1x4`` SR cvt).
    With both off, TE's ``fp4_quant_bwd_grad`` degrades to plain RtN E2M1 + E4M3
    16-element block scale -- the same cuBLASLt CUDA_R_4F_E2M1 x UE4M3 path the
    forward already proves works on sm_121.

    API NOTE (delta from nanochat): nanochat's TE took
    ``override_linear_precision=(False,False,True)`` to force wgrad->BF16 and a
    ``disable_sr`` alias. THIS TE build (2.16.0.dev0+a46079cb) has NEITHER kwarg
    -- it exposes ``disable_rht`` / ``disable_stochastic_rounding`` /
    ``disable_2d_quantization`` / ``backward_override in {None,'high_precision',
    'dequantized'}``. The bare ``disable_rht+disable_stochastic_rounding`` recipe
    runs BOTH dgrad and wgrad in FP4 RtN on this build (measured dgrad rel_err
    ~0.147, wgrad ~0.134), so no wgrad-BF16 split is needed here. We build the
    recipe with graceful per-kwarg fallback (mirroring nanochat
    ``gpt.py:_try_make_recipe``) so the same code path survives a TE version that
    renamed/added/removed a kwarg, rather than hard-failing on an unknown name.

    RAISES if ``NVFP4BlockScaling`` cannot be imported (no silent fallback).
    """

    from transformer_engine.common.recipe import (  # noqa: PLC0415
        Format,
        NVFP4BlockScaling,
    )

    # Preferred (this-build) kwargs, most-specific first; we try the full set,
    # then drop kwargs one at a time if TE rejects an unknown name. We NEVER drop
    # the two load-bearing disables silently -- if neither RHT nor SR can be
    # disabled, the recipe is the broken default and we RAISE.
    candidate_kwarg_sets = (
        dict(
            disable_rht=True,
            disable_stochastic_rounding=True,
            fp4_format=Format.E2M1,
        ),
        # Older/alt TE that used `disable_sr` alias + override_linear_precision
        # (nanochat's build). (False,False,True) => fprop FP4 + dgrad FP4 + wgrad
        # BF16. Kept for cross-version robustness; not used on this gb10 build.
        dict(
            disable_rht=True,
            disable_sr=True,
            disable_stochastic_rounding=True,
            override_linear_precision=(False, False, True),
            fp4_format=Format.E2M1,
        ),
        dict(disable_rht=True, disable_stochastic_rounding=True),
    )
    last_exc: Exception | None = None
    for kwargs in candidate_kwarg_sets:
        try:
            recipe = NVFP4BlockScaling(**kwargs)
        except Exception as exc:  # noqa: BLE001 -- unknown kwarg on this TE build
            last_exc = exc
            continue
        # Sanity: the backward grad-quant MUST have RHT off AND SR off, else this
        # is not actually the reduced recipe and the datacenter path would run.
        bwd = getattr(recipe, "fp4_quant_bwd_grad", None)
        if bwd is not None:
            rht_on = bool(getattr(bwd, "random_hadamard_transform", True))
            sr_on = bool(getattr(bwd, "stochastic_rounding", True))
            if rht_on or sr_on:
                last_exc = RuntimeError(
                    "build_nvfp4_rtn_recipe: TE accepted the kwargs but the "
                    "backward grad-quant still has "
                    f"RHT={rht_on}/SR={sr_on} -- this is NOT the reduced RtN "
                    "recipe; refusing to return a recipe that would hit the "
                    "datacenter-only SR/RHT backward path."
                )
                continue
        return recipe
    raise Nvfp4TrainingRouteUnavailable(
        "build_nvfp4_rtn_recipe: could not construct a reduced RtN "
        "NVFP4BlockScaling recipe with RHT+SR disabled on this TE build "
        f"(last error: {last_exc!r}). This TE version's recipe API differs from "
        "both this gb10 build and nanochat's -- adjust the kwarg sets. NOT "
        "falling back to the default (RHT+SR) recipe, which is broken on sm_12x."
    )


def _te_autocast(te: "Any", recipe: "Any") -> "Any":
    """Return a TE autocast context for ``recipe``, preferring the newer API.

    Newer TE exposes ``te.autocast(enabled=True, recipe=...)``; older TE uses
    ``te.fp8_autocast(enabled=True, fp8_recipe=...)``. We try the new name first
    (matching nanochat ``gpt.py:1092``) and fall back to the legacy name. This is
    NOT a precision fallback -- both wrap the SAME recipe; it only handles the API
    rename. RAISES if neither entrypoint exists.
    """

    if hasattr(te, "autocast"):
        try:
            return te.autocast(enabled=True, recipe=recipe)
        except TypeError:
            pass
    if hasattr(te, "fp8_autocast"):
        return te.fp8_autocast(enabled=True, fp8_recipe=recipe)
    raise Nvfp4TrainingRouteUnavailable(
        "_te_autocast: TransformerEngine exposes neither te.autocast nor "
        "te.fp8_autocast -- cannot enter an NVFP4 recipe context."
    )


def nvfp4_te_backward_rtn_probe(
    *,
    M: int = 256,
    K: int = 512,
    N: int = 256,
    seed: int = 0,
    tol: float | None = None,
) -> dict[str, Any]:
    """Exercise the REDUCED round-to-nearest NVFP4 BACKWARD GEMM (the nanochat
    port) and GATE it on a numeric dgrad/wgrad rel-err check.

    Builds a ``te.Linear`` under ``te.autocast(recipe=build_nvfp4_rtn_recipe())``
    (RHT off + SR off), runs ``fwd`` then ``loss.backward()`` on a small GEMM, and
    measures dgrad (input-grad) and wgrad rel-err vs a pure-BF16 reference
    ``te.Linear`` (plain autograd, no autocast). Uses a FRESH module +
    ``CUDA_LAUNCH_BLOCKING=1`` to defeat TE async-error masking (same discipline
    as the forward probe).

    GATE (RULE #1): returns ``backward_rtn_supported=True`` ONLY when the FP4
    backward runs, produces finite dgrad/wgrad, AND dgrad rel-err is below ``tol``
    (default :data:`NVFP4_RTN_BACKWARD_DGRAD_REL_ERR_TOL`). If the kernel raises an
    arch-specific PTX/RHT error, or produces NaN/garbage, or exceeds the
    tolerance, this RAISES :class:`Nvfp4CudaKernelMiscompiled` -- it never reports
    a degraded path as supported and never silently downcasts to bf16.

    RAISES if torch/TE/CUDA is unavailable (no Metal/bf16 fallback).
    """

    import torch  # noqa: PLC0415
    import transformer_engine.pytorch as te  # noqa: PLC0415

    if tol is None:
        tol = NVFP4_RTN_BACKWARD_DGRAD_REL_ERR_TOL

    if not torch.cuda.is_available():
        raise Nvfp4TrainingRouteUnavailable(
            "nvfp4_te_backward_rtn_probe: CUDA is not available; the reduced RtN "
            "NVFP4 backward is a CUDA/Blackwell (sm_120/sm_121) path. Run on gb10."
        )
    # ASYNC kernel errors must surface HERE, not corrupt the context downstream.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    cc = torch.cuda.get_device_capability(0)
    arch = f"sm_{cc[0]}{cc[1]}"
    torch.manual_seed(seed)
    dev = "cuda"

    def _rel(a: "torch.Tensor", b: "torch.Tensor") -> float:
        return (a.float() - b.float()).norm().item() / (
            b.float().norm().item() + 1e-12
        )

    # Pure-BF16 reference grads (plain autograd, no autocast) on a fresh module.
    lin = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(dev)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    g = torch.randn(M, N, device=dev, dtype=torch.bfloat16)

    x_ref = x.detach().clone().requires_grad_(True)
    lin.weight.grad = None
    y_ref = lin(x_ref)
    y_ref.backward(g)
    torch.cuda.synchronize()
    xg_bf = x_ref.grad.detach().clone()
    wg_bf = lin.weight.grad.detach().clone()
    y_bf = y_ref.detach().clone()

    # Reduced RtN FP4 fwd+bwd on the SAME module weights.
    recipe = build_nvfp4_rtn_recipe()
    x_fp4 = x.detach().clone().requires_grad_(True)
    lin.weight.grad = None
    try:
        with _te_autocast(te, recipe):
            y = lin(x_fp4)
            y.backward(g)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if any(m in msg for m in NVFP4_TE_ARCH_PTX_MARKERS):
            raise Nvfp4CudaKernelMiscompiled(
                f"reduced RtN nvfp4 BACKWARD GEMM unexpectedly hit a datacenter-"
                f"only SR/RHT kernel on {arch} despite disable_rht+"
                f"disable_stochastic_rounding. Underlying TE error: "
                f"{msg.strip()[:300]}. The recipe did not actually disable the "
                f"RHT/SR backward path on this TE build -- keeping fail-loud."
            ) from exc
        raise

    xg_fp4 = x_fp4.grad
    wg_fp4 = lin.weight.grad
    if xg_fp4 is None or wg_fp4 is None:
        raise Nvfp4CudaKernelMiscompiled(
            f"reduced RtN nvfp4 BACKWARD GEMM produced no gradients on {arch}: "
            f"the FP4 backward did not populate x.grad / weight.grad."
        )
    dgrad_finite = bool(torch.isfinite(xg_fp4).all())
    wgrad_finite = bool(torch.isfinite(wg_fp4).all())
    dgrad_rel = _rel(xg_fp4, xg_bf)
    wgrad_rel = _rel(wg_fp4, wg_bf)
    fwd_rel = _rel(y.detach(), y_bf)

    if not (dgrad_finite and wgrad_finite):
        raise Nvfp4CudaKernelMiscompiled(
            f"reduced RtN nvfp4 BACKWARD GEMM produced non-finite gradients on "
            f"{arch} (dgrad_finite={dgrad_finite}, wgrad_finite={wgrad_finite}). "
            f"NOT marking the RtN backward supported."
        )
    if not (dgrad_rel < tol):  # also catches NaN via the finite check above
        raise Nvfp4CudaKernelMiscompiled(
            f"reduced RtN nvfp4 BACKWARD GEMM dgrad rel_err={dgrad_rel:.4f} "
            f"EXCEEDS the acceptance tolerance {tol} on {arch} "
            f"(wgrad rel_err={wgrad_rel:.4f}). The reduced recipe ran but does "
            f"not match the bf16 reference closely enough to be trusted; keeping "
            f"fail-loud rather than shipping a degraded backward (RULE #1)."
        )

    return {
        "op": "backward_nvfp4_rtn_dgrad_fp4_wgrad_fp4_te_cublaslt_cuda_sm121",
        "arch": arch,
        "M": M,
        "N": N,
        "K": K,
        "element_format": NVFP4_ELEMENT_FORMAT,
        "block_size": NVFP4_BLOCK_SIZE,
        "recipe": "NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)",
        "fwd_rel_rmse_vs_bf16": fwd_rel,
        "dgrad_rel_rmse_vs_bf16": dgrad_rel,
        "wgrad_rel_rmse_vs_bf16": wgrad_rel,
        "dgrad_all_finite": dgrad_finite,
        "wgrad_all_finite": wgrad_finite,
        "tol": tol,
        "backward_rtn_supported": True,
        "kernel": "transformer_engine.pytorch (NVFP4BlockScaling RtN, cuBLASLt FP4)",
        "ported_from": "nanochat CHANGELOG_GB10.md:240-268, gpt.py:951-961",
    }


def nvfp4_backward_rtn_supported() -> bool:
    """Runtime capability gate for the reduced RtN NVFP4 backward op.

    Returns ``True`` ONLY when :func:`nvfp4_te_backward_rtn_probe` runs the real
    FP4 backward on THIS device and its numeric gate passes. Returns ``False`` if
    torch/TE/CUDA is absent or the probe raises (e.g. the gate fails or the
    kernel mis-executes). This is the runtime guard that gates the
    ``backward_nvfp4_rtn_*`` supported-op: an op listed in NVFP4_SUPPORTED_OPS is
    only ACTUALLY usable when this returns True for the live host.

    Does NOT raise -- it is a boolean capability query. The fail-loud RAISE lives
    in :func:`nvfp4_te_backward_rtn_probe` (called when you actually want the
    error detail) and in :func:`raise_if_nvfp4_training_unsupported`.
    """

    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return False
        import transformer_engine.pytorch as te  # noqa: PLC0415

        if not bool(te.is_nvfp4_available()):
            return False
    except Exception:
        return False
    try:
        res = nvfp4_te_backward_rtn_probe()
    except Exception:
        return False
    return bool(res.get("backward_rtn_supported", False))


def nvfp4_te_gemm_probe(
    *,
    M: int = 256,
    K: int = 512,
    N: int = 256,
    seed: int = 0,
    run_backward: bool = True,
) -> dict[str, Any]:
    """Exercise the REAL CUDA NVFP4 GEMM (TransformerEngine) and report honestly.

    This probes the FORWARD FP4 GEMM and the DEFAULT-recipe backward (RHT + SR
    ON). The DEFAULT backward is VERIFIED BROKEN on gb10 and UNFIXABLE by any
    sm_12x rebuild: the SR cvt (``cvt.rs.satfinite.e2m1x4``) is TE-source-gated to
    sm_100a/sm_103a and the RHT kernel needs the SM100 TMEM/tcgen05 pipeline that
    sm_12x does not implement (datacenter-only).

    NOTE: for the WORKING reduced round-to-nearest backward (RHT off + SR off,
    dgrad+wgrad in FP4 RtN), use :func:`nvfp4_te_backward_rtn_probe` instead --
    that path DOES run on sm_121 (dgrad rel_err ~0.147). This function deliberately
    exercises the DEFAULT recipe to keep the fail-loud guard honest about what
    actually breaks.

    This function:
      * runs the forward NVFP4 GEMM and returns its rel_err vs bf16 (real work);
      * if ``run_backward``, runs the DEFAULT-recipe backward and RAISES
        :class:`Nvfp4CudaKernelMiscompiled` (RULE #1 fail-loud) when the FP4
        SR-cvt / RHT kernels are unavailable on this arch, instead of returning
        the garbage gradients TE would otherwise hand back silently.

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
            "default_rht_sr_recipe_broken_on_gb10_sm121_fp4_sr_cvt_and_rht_are_"
            "datacenter_only_not_fixable_by_te_rebuild__but_reduced_rtn_recipe_"
            "works"
        ),
        "backward_gemm_cuda_enablement": (
            "UPSTREAM ONLY (for the DEFAULT recipe): the SR FP4 cvt "
            "(cvt.rs.satfinite.e2m1x4) is source-gated in TE to sm_100a/sm_103a "
            "and the RHT kernel needs SM100 TMEM/tcgen05 absent on sm_12x; nvcc "
            "13.x exposes no sm_12x a/f target, so a TE rebuild cannot enable it. "
            "Needs NVIDIA to ship SM12x-native FP4 SR+RHT backward kernels in a "
            "future TE/cuBLASLt."
        ),
        # The REDUCED round-to-nearest backward (RHT off + SR off) DOES run on
        # sm_121; gated at runtime by nvfp4_te_backward_rtn_probe().
        "backward_gemm_cuda_rtn_status": (
            "reduced_rtn_recipe_disable_rht_disable_stochastic_rounding_WORKS_on_"
            "gb10_sm121_dgrad_fp4_wgrad_fp4_rel_err_~0.147_gated_by_numeric_probe"
        ),
        "backward_gemm_cuda_rtn_recipe": (
            "NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)"
        ),
        "backward_gemm_cuda_rtn_ported_from": (
            "nanochat CHANGELOG_GB10.md:240-268, gpt.py:951-961"
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
        "~0.146). The BACKWARD GEMM now has TWO recipes with different status: "
        "(1) the REDUCED round-to-nearest recipe "
        "NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True) "
        "RUNS on gb10/sm_121 (dgrad+wgrad FP4 RtN, rel_err ~0.147; ported from "
        "nanochat) and is exposed via nvfp4_te_backward_rtn_probe() gated by a "
        "numeric check; (2) the DEFAULT recipe (RHT + stochastic rounding ON) is "
        "GENUINELY DATACENTER-ONLY -- NOT a build-flag gap fixable by a TE "
        "rebuild. The default backward stochastic-rounding FP4 cast uses "
        "'cvt.rs.satfinite.e2m1x4', which TransformerEngine source-gates to "
        "ArchSpecific<100>/<103> (sm_100a/sm_103a) -- it is excluded from the "
        "sm_120 family set, so any sm_12x build leaves has_rs=false and the cast "
        "asserts 'FP4 cvt PTX instructions are architecture-specific'. The "
        "Random-Hadamard fused kernel is built on the CUTLASS SM100 TMEM/tcgen05 "
        "pipeline, which sm_12x (GB10/DGX Spark/RTX50) does NOT implement (per "
        "NVIDIA), so it raises 'CUDA Error: invalid argument'. nvcc 13.x exposes "
        "no sm_12x a/f code target. Enabling the DEFAULT recipe is therefore "
        "UPSTREAM ONLY: NVIDIA must ship SM12x-native FP4 SR + RHT backward "
        "kernels in a future TransformerEngine/cuBLASLt release. SEPARATELY, a "
        "full end-to-end training step ALSO needs nvfp4 kernels that are still "
        "MISSING on the available arch: "
        f"{ops}. Rather than silently downcast those ops to bf16 (forbidden), the "
        "nvfp4 route fails loud here. For a full training step TODAY run "
        "--dtype fp8_path_c or --dtype bfloat16."
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
