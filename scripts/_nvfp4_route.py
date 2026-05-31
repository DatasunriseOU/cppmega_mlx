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

What IS nvfp4 today (M4-verified, committed):
  * forward GEMM operands: quantize bf16 -> e2m1 + per-block-16 scale via the
    ``cppmega_mlx.quant.mxfp4_metal`` codec, multiply via the proven
    ``cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c`` cooperative-tensor GEMM
    (commit 29c112d). :func:`nvfp4_gemm_smoke` exercises exactly this.

What is NOT nvfp4 yet (no kernel -> the route RAISES, it does not bf16-fallback):
  * backward / VJP for the e2m1 GEMM (no nvfp4 grad kernel),
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
NVFP4_SUPPORTED_OPS = ("forward_gemm_operands_e2m1_block_scale_metal_m4",)

# Ops that have NO nvfp4 kernel. The training step RAISES naming these rather
# than silently running them in bf16 (RULE #1: no silent downcast).
NVFP4_UNSUPPORTED_TRAINING_OPS = (
    "cuda_blackwell_nvfp4_gemm",   # forward GEMM kernel exists for Metal/M4 only
    "gemm_backward_vjp",            # no nvfp4 grad kernel for the e2m1 GEMM
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
        "forward_gemm_kernel": (
            "cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c.mxfp4_matmul_path_c"
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
        f"per-block-{NVFP4_BLOCK_SIZE} scale operands) and the forward-GEMM "
        "operand path is real and M4-verified "
        "(cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c). A full training step "
        "additionally needs nvfp4 kernels that DO NOT EXIST yet: "
        f"{ops}. Rather than silently downcast these ops to bf16 (forbidden), "
        "the nvfp4 route fails loud here. Next blocker: implement the nvfp4 "
        "GEMM backward/VJP, then non-GEMM ops, then optimizer-state support; "
        "run --dtype fp8_path_c or --dtype bfloat16 for a full training step "
        "today."
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
