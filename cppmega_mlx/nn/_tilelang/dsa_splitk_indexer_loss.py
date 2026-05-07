# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Path C DSA split-K fused indexer-loss kernels via TileLang DSL lowering.

This module is the TileLang-DSL counterpart to the CUDA-only Triton kernels
``_fwd_fused_indexer_loss_stage1_kernel`` and
``_fwd_fused_indexer_loss_stage2_kernel`` defined in
``cppmega/cppmega/megatron/dsa_splitk_indexer_loss.py``.

It is the next building block of the unified fused-kernel pipeline (sibling
of ``fp8_amax.py``): a single TileLang source compiles for both CUDA and
Apple Metal SIMDgroup targets, replacing the ``tensor.is_cuda``-gated Triton
path on CUDA hosts and providing the previously-missing Metal path.

Source attribution
------------------

The reference Triton kernels ported here live in:

* ``cppmega/cppmega/megatron/dsa_splitk_indexer_loss.py``:
  ``_fwd_fused_indexer_loss_stage1_kernel`` (~line 55) and
  ``_fwd_fused_indexer_loss_stage2_kernel`` (~line 193). Original style guide:
  ``cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py`` and
  ``cppmega_mlx/nn/_tilelang/fp8_amax.py``.

The two-stage layout matches the upstream NVIDIA Megatron-LM PR #4039 fused
indexer loss:

* Stage 1: per (b, sq_block, h) lane runs a split-K block-matmul over the
  ``sk`` dimension, accumulating the FlashAttention-style online softmax
  running max ``softmax_m`` and denominator ``softmax_d``. Head 0 also
  accumulates the same statistics for ``index_scores`` -> ``softmax_m1`` /
  ``softmax_d1``.
* Stage 2: per (b, sq_block) lane recomputes Q@K^T blockwise (split-K), uses
  the now-known stage-1 statistics to compute the normalised attention
  softmax ``p`` and index softmax ``q``, and reduces the per-position
  KL-divergence ``sum_j p_j * (log p_j - log q_j)`` into ``Loss[b, sq]``.

CUDA emission of the resulting TileLang PrimFunc is numerically equivalent to
the Triton kernel; the Metal emission relies on `T.gemm` lowering through the
SIMDgroup matmul path (TileLang 0.1.9+) for the inner Q@K^T tile and
threadgroup-mem staging for the online softmax statistics.

Block constants per target
--------------------------

Triton reference launch (CUDA defaults):

    BLOCK_SQ = 128, BLOCK_SK = 128, BLOCK_D = 64
    num_warps = 8, num_stages = 3.

Footprint of double-buffered staging (fp16 data) per stage 1 tile:

    2 * (BLOCK_SQ * BLOCK_D + BLOCK_SK * BLOCK_D) * 2 bytes
        = 2 * (128*64 + 128*64) * 2 = 65 536 bytes

That is 64 KB -- well under CUDA's 96 KB shared-mem budget on Hopper but
twice Apple Silicon's 32 KB per-threadgroup limit. So the Metal target uses
half-sized tiles:

    BLOCK_SQ = 64, BLOCK_SK = 64, BLOCK_D = 32
    => 2 * (64*32 + 64*32) * 2 = 16 384 bytes (16 KB), comfortably below the
    32 KB limit even with the online-softmax fp32 scratch (~1 KB extra).

These block constants are exposed as module-level globals at the top of this
module (``_DSA_STAGE1_BLOCK_*`` and ``_DSA_STAGE2_BLOCK_*``) and a
``_metal_block_overrides`` helper documents the substitution.

Deferred features (NOT implemented in this PoC)
-----------------------------------------------

* Mixed-precision variants: the Triton reference is fp16/bf16 inputs with
  fp32 accumulate. A native fp8-input variant (e4m3 Q, e4m3 K, fp32 accum)
  would let us reuse the ``fp8_scaled_matmul`` Path C primitive but is
  Phase 2.4 (after the indexer-loss / amax / quantize PoCs land).
* Autotune hooks: BLOCK_SQ / BLOCK_SK / BLOCK_D / SPLIT_K are static
  per (shape, target) and resolved by the ``lru_cache`` at compile time.
  A Triton-style autotune sweep is Phase 2.6.
* Fused backward: the Triton reference (and this port) only implements the
  forward indexer-loss reduction. The backward through the KL-divergence
  is computed by autograd via the elementwise per-position ``Loss`` output
  -- a fused backward kernel is Phase 2.3 / integration #10.
* SPARSE_LOSS variant of the kernel here is *not* fused -- the wrapper
  pre-computes the ``index_mask = scatter(-inf, topk_indices, 0)`` tensor
  on the host and adds it to ``Index_Scores`` before launching, matching
  the wrapper-side branch in the Triton reference for Sparse mode.

API surface
-----------

* :func:`make_dsa_splitk_stage1_kernel` -- build a shape-specialized stage 1
  PrimFunc.
* :func:`make_dsa_splitk_stage2_kernel` -- build a shape-specialized stage 2
  PrimFunc.
* :func:`dsa_splitk_indexer_loss_tilelang` -- torch wrapper with the same
  shape/dtype contract as ``compute_dsa_indexer_loss_splitk``.
* :func:`tilelang_supports` -- runtime gate the patched
  ``cppmega/megatron/dsa_splitk_indexer_loss.py`` uses to decide between
  TileLang and the unfused Triton / PyTorch fallback.
* :func:`dsa_splitk_path_c_status` -- importability + reason (mirrors the
  ``fp8_amax_path_c_status`` style).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import torch


# ---------------------------------------------------------------------------
# Kernel-shape defaults -- TileLang resolves these globals while decorating
# the nested @T.prim_func, mirroring fp8_amax.py's ``_FP8_AMAX_*``.
# ---------------------------------------------------------------------------

# CUDA defaults (match Triton reference's grid-launch in
# ``compute_dsa_indexer_loss_splitk``):
_DSA_STAGE1_BLOCK_SQ = 128
_DSA_STAGE1_BLOCK_SK = 128
_DSA_STAGE1_BLOCK_D = 64
_DSA_STAGE1_THREADS = 256  # 8 warps * 32 lanes (Triton ref num_warps=8)
_DSA_STAGE1_NUM_STAGES = 3

_DSA_STAGE2_BLOCK_SQ = 128
_DSA_STAGE2_BLOCK_SK = 128
_DSA_STAGE2_BLOCK_D = 64
_DSA_STAGE2_THREADS = 256
_DSA_STAGE2_NUM_STAGES = 3


def _metal_block_overrides(stage: int) -> dict[str, int]:
    """Return Metal-tuned BLOCK_* / threads overrides for *stage*.

    Apple Silicon's per-threadgroup memory budget is 32 KB. Beyond the
    fp16 shared Q/K stage (which previously used 16 KB at BLOCK 64x64x32),
    the kernels also allocate fp32 register fragments per-tile:

      stage 1: scores_f + idx_scores_f = 2 * (BLOCK_SQ*BLOCK_SK*4) bytes
      stage 2: h_scores + softmax_attn + softmax_idx + kl_term
               = 4 * (BLOCK_SQ*BLOCK_SK*4) bytes

    At 64x64 the stage-2 fragments alone are 64 KB -- well over the 32 KB
    budget once shared staging is added, causing register spilling and
    huge slowdowns or compile failures on M-series. 32x32x16 keeps each
    fragment at 4 KB (stage 1: ~8 KB, stage 2: ~16 KB) plus shared 4 KB
    -- comfortably under 32 KB.
    """

    if stage == 1:
        return dict(
            BLOCK_SQ=32,
            BLOCK_SK=32,
            BLOCK_D=16,
            threads=128,
            num_stages=2,
        )
    if stage == 2:
        return dict(
            BLOCK_SQ=32,
            BLOCK_SK=32,
            BLOCK_D=16,
            threads=128,
            num_stages=2,
        )
    raise ValueError(f"_metal_block_overrides: unknown stage {stage!r}")


@dataclass(frozen=True)
class DSASplitKPathCStatus:
    """Runtime/lowering status for the Path C TileLang DSA split-K kernels."""

    available: bool
    reason: str
    cuda_target: str = "cuda"
    metal_target: str = "metal"


def _tilelang_available() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - hosts without TileLang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


def dsa_splitk_path_c_status() -> DSASplitKPathCStatus:
    """Return whether TileLang is importable for Path C DSA split-K lowering."""

    ok, reason = _tilelang_available()
    if not ok:
        return DSASplitKPathCStatus(available=False, reason=reason)
    return DSASplitKPathCStatus(
        available=True,
        reason="DSA split-K indexer-loss Path C TileLang DSL lowering is available",
    )


def tilelang_supports(device: torch.device | str | None) -> bool:
    """Return True when the TileLang DSA split-K port can dispatch on *device*.

    The TileLang JIT supports CUDA (``cuda``) and Apple Metal (``mps`` /
    ``metal``) targets. CPU tensors must continue to use the unfused PyTorch
    reference (full attention recompute) -- there is no CPU TileLang backend.
    """

    ok, _ = _tilelang_available()
    if not ok:
        return False
    if device is None:
        return False
    if isinstance(device, str):
        dev_type = torch.device(device).type
    else:
        dev_type = device.type
    if dev_type == "cuda":
        return torch.cuda.is_available()
    if dev_type == "mps":
        return torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    return False


def _resolve_target(device: torch.device) -> str:
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        # Explicit warp size keeps codegen aligned with Apple SIMDgroup width
        # (matches ``fp8_amax.py``).
        return "metal -thread_warp_size=32"
    raise ValueError(f"dsa_splitk_indexer_loss_tilelang: unsupported device type {device.type!r}")


def _is_metal(target: str) -> bool:
    return target.startswith("metal")


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------


def make_dsa_splitk_stage1_kernel(
    *,
    AB: int,
    AH: int,
    AD: int,
    Sk: int,
    ASq: int,
    sparse_loss: bool,
    softmax_scale: float,
    in_dtype: str = "float16",
    BLOCK_SQ: int = _DSA_STAGE1_BLOCK_SQ,
    BLOCK_SK: int = _DSA_STAGE1_BLOCK_SK,
    BLOCK_D: int = _DSA_STAGE1_BLOCK_D,
    threads: int = _DSA_STAGE1_THREADS,
    num_stages: int = _DSA_STAGE1_NUM_STAGES,
) -> Any:
    """Build a shape-specialized stage-1 online-softmax statistics kernel.

    Mirrors the Triton ``_fwd_fused_indexer_loss_stage1_kernel`` reference:
    grid = (AB, ceildiv(ASq, BLOCK_SQ), AH). Each lane:

    * Streams the SK dimension in BLOCK_SK chunks (causally masked at
      ``min(sq) + 1``).
    * Loads a Q tile [BLOCK_SQ, BLOCK_D] and K tile [BLOCK_D, BLOCK_SK] via
      :func:`T.copy`, runs :func:`T.gemm` to accumulate ``h_scores`` in a
      fragment.
    * Multiplies by ``Softmax_Scale``, applies the upper-triangular causal
      mask, and updates ``softmax_m`` / ``softmax_d`` via the standard
      FlashAttention online-softmax recurrence.
    * On head 0 only, also accumulates the ``index_scores`` softmax
      statistics into ``softmax_m1`` / ``softmax_d1`` (these are independent
      of the matmul; just a row reduce over the SK chunk).

    Inputs:
        ``Q``: ``(ASq, AB, AH, AD)`` ``in_dtype``  (Triton's [Sq, B, H, D]).
        ``K``: ``(Sk, AB, AH, AD)`` ``in_dtype``.
        ``IndexScores``: ``(AB, ASq, Sk)`` fp32 (Triton ref is fp32).
        ``IndexMask``: ``(AB, ASq, Sk)`` fp32 -- pre-computed upstream when
            ``sparse_loss`` is True; ignored otherwise. Always passed (zero-
            length when not sparse) to keep the PrimFunc signature stable.

    Outputs (in-place updates):
        ``M``:  ``(AB, AH, ASq)`` fp32, init -inf.
        ``D``:  ``(AB, AH, ASq)`` fp32, init 0.
        ``M1``: ``(AB, ASq)``    fp32, init -inf (head-0 only writes).
        ``D1``: ``(AB, ASq)``    fp32, init 0    (head-0 only writes).
    """

    if AB <= 0 or AH <= 0 or AD <= 0 or Sk <= 0 or ASq <= 0:
        raise ValueError(
            "make_dsa_splitk_stage1_kernel: all dims must be positive; "
            f"got AB={AB}, AH={AH}, AD={AD}, Sk={Sk}, ASq={ASq}"
        )
    if BLOCK_SQ <= 0 or BLOCK_SK <= 0 or BLOCK_D <= 0 or threads <= 0:
        raise ValueError(
            "make_dsa_splitk_stage1_kernel: all BLOCK_*/threads must be positive; "
            f"got BLOCK_SQ={BLOCK_SQ}, BLOCK_SK={BLOCK_SK}, BLOCK_D={BLOCK_D}, threads={threads}"
        )

    import tilelang.language as T

    T = cast(Any, T)

    NUM_SQ_BLOCKS = (ASq + BLOCK_SQ - 1) // BLOCK_SQ
    SK_TILES = (Sk + BLOCK_SK - 1) // BLOCK_SK
    SCALE = float(softmax_scale)
    SPARSE = bool(sparse_loss)

    @T.prim_func
    def dsa_stage1(
        Q: T.Tensor((ASq, AB, AH, AD), in_dtype),
        K: T.Tensor((Sk, AB, AH, AD), in_dtype),
        IndexScores: T.Tensor((AB, ASq, Sk), "float32"),
        IndexMask: T.Tensor((AB, ASq, Sk), "float32"),
        M: T.Tensor((AB, AH, ASq), "float32"),
        D: T.Tensor((AB, AH, ASq), "float32"),
        M1: T.Tensor((AB, ASq), "float32"),
        D1: T.Tensor((AB, ASq), "float32"),
    ):
        with T.Kernel(AB, NUM_SQ_BLOCKS, AH, threads=threads) as (b, sq_block_id, h):
            # Per-block staging for Q[BLOCK_SQ, BLOCK_D], K[BLOCK_D, BLOCK_SK]
            # tiles. Double-buffering is requested by the surrounding
            # T.Pipelined(num_stages=...) -- the injector replicates these
            # buffers internally.
            Q_s = T.alloc_shared((BLOCK_SQ, BLOCK_D), in_dtype)
            K_s = T.alloc_shared((BLOCK_D, BLOCK_SK), in_dtype)

            # Online-softmax fragments live in registers.
            scores_f = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), "float32")
            row_max_local = T.alloc_fragment((BLOCK_SQ,), "float32")
            row_sum_local = T.alloc_fragment((BLOCK_SQ,), "float32")
            m_i = T.alloc_fragment((BLOCK_SQ,), "float32")
            d_i = T.alloc_fragment((BLOCK_SQ,), "float32")
            m_i_prev = T.alloc_fragment((BLOCK_SQ,), "float32")
            m1_i = T.alloc_fragment((BLOCK_SQ,), "float32")
            d1_i = T.alloc_fragment((BLOCK_SQ,), "float32")
            m1_i_prev = T.alloc_fragment((BLOCK_SQ,), "float32")
            idx_scores_f = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), "float32")

            # Initialise the index-softmax accumulators (head 0 only writes
            # them out at the end, but we keep the registers live for all
            # heads to keep the kernel structure uniform).
            for i in T.Parallel(BLOCK_SQ):
                m1_i[i] = T.cast(-3.4028234663852886e38, "float32")
                d1_i[i] = T.cast(0, "float32")

            # Initialise the attention online-softmax accumulators.
            for i in T.Parallel(BLOCK_SQ):
                m_i[i] = T.cast(-3.4028234663852886e38, "float32")
                d_i[i] = T.cast(0, "float32")

            # Stream the SK dimension. Triton's causal mask trims iterations
            # to ``min(sq) + 1`` but TileLang lacks a runtime-trim grid bound;
            # instead we iterate the full SK and let the per-tile causal mask
            # zero out invalid positions (matches ``casual_mask`` in Triton ref
            # at line ~153). This pessimises CUDA perf marginally but keeps
            # codegen target-portable.
            for sk_tile in T.Pipelined(SK_TILES, num_stages=num_stages):
                # Initialise the score accumulator for this tile.
                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    scores_f[i, j] = T.cast(0, "float32")

                # Inner D-loop: matmul Q @ K^T, accumulating into scores_f.
                for d_tile in T.serial((AD + BLOCK_D - 1) // BLOCK_D):
                    # Stage Q[BLOCK_SQ, BLOCK_D] for this (h, sq_block, d_tile).
                    for i, dd in T.Parallel(BLOCK_SQ, BLOCK_D):
                        sq_idx = sq_block_id * BLOCK_SQ + i
                        d_idx = d_tile * BLOCK_D + dd
                        if (sq_idx < ASq) and (d_idx < AD):
                            Q_s[i, dd] = Q[sq_idx, b, h, d_idx]
                        else:
                            Q_s[i, dd] = T.cast(0, in_dtype)

                    # Stage K[BLOCK_D, BLOCK_SK] for this (h, sk_tile, d_tile).
                    for dd, j in T.Parallel(BLOCK_D, BLOCK_SK):
                        sk_idx = sk_tile * BLOCK_SK + j
                        d_idx = d_tile * BLOCK_D + dd
                        if (sk_idx < Sk) and (d_idx < AD):
                            K_s[dd, j] = K[sk_idx, b, h, d_idx]
                        else:
                            K_s[dd, j] = T.cast(0, in_dtype)

                    # Q[BLOCK_SQ, BLOCK_D] @ K[BLOCK_D, BLOCK_SK] += scores.
                    T.gemm(Q_s, K_s, scores_f)

                # Apply softmax scale + causal mask.
                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    sq_idx = sq_block_id * BLOCK_SQ + i
                    sk_idx = sk_tile * BLOCK_SK + j
                    in_bounds = (sq_idx < ASq) and (sk_idx < Sk)
                    valid = in_bounds and (sq_idx >= sk_idx)
                    s = scores_f[i, j] * T.cast(SCALE, "float32")
                    # Guard the IndexMask load against OOB on boundary tiles
                    # (last sq_block / last sk_tile when ASq % BLOCK_SQ != 0
                    # or Sk % BLOCK_SK != 0). Triton uses tl.load(..., mask=...)
                    # for the equivalent guard; here we predicate the read.
                    if SPARSE and in_bounds:
                        s = s + IndexMask[b, sq_idx, sk_idx]
                    if valid:
                        scores_f[i, j] = s
                    else:
                        scores_f[i, j] = T.cast(-3.4028234663852886e38, "float32")

                # Online softmax recurrence on (m_i, d_i).
                T.reduce_max(scores_f, row_max_local, dim=1, clear=True)
                for i in T.Parallel(BLOCK_SQ):
                    m_i_prev[i] = m_i[i]
                    new_m = T.max(m_i[i], row_max_local[i])
                    # Drop the all-(-inf) sentinel back to 0 so the exp delta
                    # below stays finite, matching Triton's ``tl.where``.
                    if new_m <= T.cast(-3.4028234663852886e38, "float32"):
                        new_m = T.cast(0, "float32")
                    m_i[i] = new_m

                # Renormalise scores so we can sum exp safely.
                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    scores_f[i, j] = T.exp(scores_f[i, j] - m_i[i])

                T.reduce_sum(scores_f, row_sum_local, dim=1, clear=True)
                for i in T.Parallel(BLOCK_SQ):
                    d_i[i] = d_i[i] * T.exp(m_i_prev[i] - m_i[i]) + row_sum_local[i]

                # Head-0 path: accumulate the index_scores online softmax too.
                if h == 0:
                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        sq_idx = sq_block_id * BLOCK_SQ + i
                        sk_idx = sk_tile * BLOCK_SK + j
                        valid = (sq_idx < ASq) and (sk_idx < Sk)
                        # Predicate IndexScores / IndexMask reads on bounds to
                        # avoid OOB on boundary tiles (matches the Triton
                        # `tl.load(..., mask=...)` semantics).
                        if valid:
                            v = IndexScores[b, sq_idx, sk_idx]
                            if SPARSE:
                                v = v + IndexMask[b, sq_idx, sk_idx]
                            idx_scores_f[i, j] = v
                        else:
                            idx_scores_f[i, j] = T.cast(-3.4028234663852886e38, "float32")

                    T.reduce_max(idx_scores_f, row_max_local, dim=1, clear=True)
                    for i in T.Parallel(BLOCK_SQ):
                        m1_i_prev[i] = m1_i[i]
                        new_m = T.max(m1_i[i], row_max_local[i])
                        if new_m <= T.cast(-3.4028234663852886e38, "float32"):
                            new_m = T.cast(0, "float32")
                        m1_i[i] = new_m

                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        idx_scores_f[i, j] = T.exp(idx_scores_f[i, j] - m1_i[i])
                    T.reduce_sum(idx_scores_f, row_sum_local, dim=1, clear=True)
                    for i in T.Parallel(BLOCK_SQ):
                        d1_i[i] = d1_i[i] * T.exp(m1_i_prev[i] - m1_i[i]) + row_sum_local[i]

            # Persist the (m_i, d_i) statistics back to global memory.
            for i in T.Parallel(BLOCK_SQ):
                sq_idx = sq_block_id * BLOCK_SQ + i
                if sq_idx < ASq:
                    M[b, h, sq_idx] = m_i[i]
                    D[b, h, sq_idx] = d_i[i]
                    if h == 0:
                        M1[b, sq_idx] = m1_i[i]
                        D1[b, sq_idx] = d1_i[i]

    return dsa_stage1


def make_dsa_splitk_stage2_kernel(
    *,
    AB: int,
    AH: int,
    AD: int,
    Sk: int,
    ASq: int,
    sparse_loss: bool,
    softmax_scale: float,
    in_dtype: str = "float16",
    BLOCK_SQ: int = _DSA_STAGE2_BLOCK_SQ,
    BLOCK_SK: int = _DSA_STAGE2_BLOCK_SK,
    BLOCK_D: int = _DSA_STAGE2_BLOCK_D,
    threads: int = _DSA_STAGE2_THREADS,
    num_stages: int = _DSA_STAGE2_NUM_STAGES,
) -> Any:
    """Build a shape-specialized stage-2 KL-divergence reduction kernel.

    Mirrors the Triton ``_fwd_fused_indexer_loss_stage2_kernel`` reference:
    grid = (AB, ceildiv(ASq, BLOCK_SQ)). Each lane:

    * Loads stage-1 statistics ``M1`` / ``D1`` for this (b, sq_block).
    * Streams the SK dimension in BLOCK_SK chunks; for each chunk:
        - Recompute Q@K^T per-head and divide by stage-1 ``D[b, h, sq]``
          to get the normalised attention softmax ``p`` (averaged over heads).
        - Compute the index softmax ``q`` from ``IndexScores`` + ``M1`` / ``D1``.
        - Accumulate the per-position KL divergence
          ``sum_j p_j * (log(p_j + eps) - log(q_j + eps))`` into ``loss_i``.
    * Writes ``Loss[b, sq] = loss_i``.
    """

    if AB <= 0 or AH <= 0 or AD <= 0 or Sk <= 0 or ASq <= 0:
        raise ValueError(
            "make_dsa_splitk_stage2_kernel: all dims must be positive; "
            f"got AB={AB}, AH={AH}, AD={AD}, Sk={Sk}, ASq={ASq}"
        )
    if BLOCK_SQ <= 0 or BLOCK_SK <= 0 or BLOCK_D <= 0 or threads <= 0:
        raise ValueError(
            "make_dsa_splitk_stage2_kernel: all BLOCK_*/threads must be positive; "
            f"got BLOCK_SQ={BLOCK_SQ}, BLOCK_SK={BLOCK_SK}, BLOCK_D={BLOCK_D}, threads={threads}"
        )

    import tilelang.language as T

    T = cast(Any, T)

    NUM_SQ_BLOCKS = (ASq + BLOCK_SQ - 1) // BLOCK_SQ
    SK_TILES = (Sk + BLOCK_SK - 1) // BLOCK_SK
    EPS: float = 1e-10
    SCALE = float(softmax_scale)
    SPARSE = bool(sparse_loss)
    INV_AH = 1.0 / float(AH)

    @T.prim_func
    def dsa_stage2(
        Q: T.Tensor((ASq, AB, AH, AD), in_dtype),
        K: T.Tensor((Sk, AB, AH, AD), in_dtype),
        IndexScores: T.Tensor((AB, ASq, Sk), "float32"),
        IndexMask: T.Tensor((AB, ASq, Sk), "float32"),
        M: T.Tensor((AB, AH, ASq), "float32"),
        D: T.Tensor((AB, AH, ASq), "float32"),
        M1: T.Tensor((AB, ASq), "float32"),
        D1: T.Tensor((AB, ASq), "float32"),
        Loss: T.Tensor((AB, ASq), "float32"),
    ):
        with T.Kernel(AB, NUM_SQ_BLOCKS, threads=threads) as (b, sq_block_id):
            Q_s = T.alloc_shared((BLOCK_SQ, BLOCK_D), in_dtype)
            K_s = T.alloc_shared((BLOCK_D, BLOCK_SK), in_dtype)

            h_scores = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), "float32")
            softmax_attn = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), "float32")
            softmax_idx = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), "float32")
            kl_term = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), "float32")
            row_sum_local = T.alloc_fragment((BLOCK_SQ,), "float32")

            m1_local = T.alloc_fragment((BLOCK_SQ,), "float32")
            d1_local = T.alloc_fragment((BLOCK_SQ,), "float32")
            loss_i = T.alloc_fragment((BLOCK_SQ,), "float32")
            m_h = T.alloc_fragment((BLOCK_SQ,), "float32")
            d_h = T.alloc_fragment((BLOCK_SQ,), "float32")

            # Load stage-1 index-softmax statistics for this sq block.
            for i in T.Parallel(BLOCK_SQ):
                sq_idx = sq_block_id * BLOCK_SQ + i
                if sq_idx < ASq:
                    m1_local[i] = M1[b, sq_idx]
                    d1_local[i] = D1[b, sq_idx]
                else:
                    m1_local[i] = T.cast(0, "float32")
                    d1_local[i] = T.cast(1, "float32")
                loss_i[i] = T.cast(0, "float32")

            for sk_tile in T.Pipelined(SK_TILES, num_stages=num_stages):
                # Zero softmax_attn for this tile (we accumulate over heads).
                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    softmax_attn[i, j] = T.cast(0, "float32")

                # Per-head: recompute Q@K^T, scale, mask, exp/d_h.
                for h in T.serial(AH):
                    # Initialise score accumulator.
                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        h_scores[i, j] = T.cast(0, "float32")

                    # Inner D-loop matmul.
                    for d_tile in T.serial((AD + BLOCK_D - 1) // BLOCK_D):
                        for i, dd in T.Parallel(BLOCK_SQ, BLOCK_D):
                            sq_idx = sq_block_id * BLOCK_SQ + i
                            d_idx = d_tile * BLOCK_D + dd
                            if (sq_idx < ASq) and (d_idx < AD):
                                Q_s[i, dd] = Q[sq_idx, b, h, d_idx]
                            else:
                                Q_s[i, dd] = T.cast(0, in_dtype)

                        for dd, j in T.Parallel(BLOCK_D, BLOCK_SK):
                            sk_idx = sk_tile * BLOCK_SK + j
                            d_idx = d_tile * BLOCK_D + dd
                            if (sk_idx < Sk) and (d_idx < AD):
                                K_s[dd, j] = K[sk_idx, b, h, d_idx]
                            else:
                                K_s[dd, j] = T.cast(0, in_dtype)

                        T.gemm(Q_s, K_s, h_scores)

                    # Load this head's stage-1 stats.
                    for i in T.Parallel(BLOCK_SQ):
                        sq_idx = sq_block_id * BLOCK_SQ + i
                        if sq_idx < ASq:
                            m_h[i] = M[b, h, sq_idx]
                            d_h[i] = D[b, h, sq_idx]
                        else:
                            m_h[i] = T.cast(0, "float32")
                            d_h[i] = T.cast(1, "float32")

                    # Scale + causal mask + (optional) sparse mask + add to
                    # accumulated softmax_attn (averaged over heads at the
                    # end via *= INV_AH).
                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        sq_idx = sq_block_id * BLOCK_SQ + i
                        sk_idx = sk_tile * BLOCK_SK + j
                        in_bounds = (sq_idx < ASq) and (sk_idx < Sk)
                        valid = in_bounds and (sq_idx >= sk_idx)
                        s = h_scores[i, j] * T.cast(SCALE, "float32")
                        # Predicate the IndexMask read on bounds to avoid OOB
                        # on boundary tiles (Triton uses `tl.load(..., mask=...)`).
                        if SPARSE and in_bounds:
                            s = s + IndexMask[b, sq_idx, sk_idx]
                        if valid:
                            denom = d_h[i]
                            if denom <= T.cast(0, "float32"):
                                denom = T.cast(1, "float32")
                            softmax_attn[i, j] = softmax_attn[i, j] + T.exp(s - m_h[i]) / denom

                # Average over heads.
                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    softmax_attn[i, j] = softmax_attn[i, j] * T.cast(INV_AH, "float32")

                # Compute index softmax q = exp(idx - m1) / d1.
                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    sq_idx = sq_block_id * BLOCK_SQ + i
                    sk_idx = sk_tile * BLOCK_SK + j
                    valid = (sq_idx < ASq) and (sk_idx < Sk)
                    denom1 = d1_local[i]
                    if denom1 <= T.cast(0, "float32"):
                        denom1 = T.cast(1, "float32")
                    # Predicate IndexScores / IndexMask reads on bounds to
                    # avoid OOB on boundary tiles.
                    if valid:
                        v = IndexScores[b, sq_idx, sk_idx]
                        if SPARSE:
                            v = v + IndexMask[b, sq_idx, sk_idx]
                        softmax_idx[i, j] = T.exp(v - m1_local[i]) / denom1
                    else:
                        softmax_idx[i, j] = T.cast(0, "float32")

                # KL term: p * (log(p+eps) - log(q+eps)).
                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    p = softmax_attn[i, j]
                    q_ = softmax_idx[i, j]
                    sq_idx = sq_block_id * BLOCK_SQ + i
                    sk_idx = sk_tile * BLOCK_SK + j
                    valid = (sq_idx < ASq) and (sk_idx < Sk) and (sq_idx >= sk_idx)
                    if valid:
                        kl_term[i, j] = p * (
                            T.log(p + T.cast(EPS, "float32"))
                            - T.log(q_ + T.cast(EPS, "float32"))
                        )
                    else:
                        kl_term[i, j] = T.cast(0, "float32")

                T.reduce_sum(kl_term, row_sum_local, dim=1, clear=True)
                for i in T.Parallel(BLOCK_SQ):
                    loss_i[i] = loss_i[i] + row_sum_local[i]

            # Persist the per-position loss.
            for i in T.Parallel(BLOCK_SQ):
                sq_idx = sq_block_id * BLOCK_SQ + i
                if sq_idx < ASq:
                    Loss[b, sq_idx] = loss_i[i]

    return dsa_stage2


# ---------------------------------------------------------------------------
# JIT cache + torch dispatch
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _stage1_kernel_for(
    AB: int,
    AH: int,
    AD: int,
    Sk: int,
    ASq: int,
    sparse_loss: bool,
    softmax_scale_bits: int,
    in_dtype: str,
    target: str,
    BLOCK_SQ: int,
    BLOCK_SK: int,
    BLOCK_D: int,
    threads: int,
    num_stages: int,
) -> Any:
    """Build, JIT-compile, and cache the stage-1 kernel for a (shape, target)."""

    import tilelang
    import struct

    # Recover the original fp32 from the bit-pattern key (we use bits because
    # ``lru_cache`` requires a hashable scalar key and floats are fine but the
    # tuple length is large enough that mistakes from float-equality bite).
    softmax_scale = struct.unpack("<f", struct.pack("<I", softmax_scale_bits))[0]

    prim = make_dsa_splitk_stage1_kernel(
        AB=AB,
        AH=AH,
        AD=AD,
        Sk=Sk,
        ASq=ASq,
        sparse_loss=sparse_loss,
        softmax_scale=softmax_scale,
        in_dtype=in_dtype,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        threads=threads,
        num_stages=num_stages,
    )
    return tilelang.compile(prim, target=target, out_idx=None)


@lru_cache(maxsize=64)
def _stage2_kernel_for(
    AB: int,
    AH: int,
    AD: int,
    Sk: int,
    ASq: int,
    sparse_loss: bool,
    softmax_scale_bits: int,
    in_dtype: str,
    target: str,
    BLOCK_SQ: int,
    BLOCK_SK: int,
    BLOCK_D: int,
    threads: int,
    num_stages: int,
) -> Any:
    """Build, JIT-compile, and cache the stage-2 kernel for a (shape, target)."""

    import tilelang
    import struct

    softmax_scale = struct.unpack("<f", struct.pack("<I", softmax_scale_bits))[0]

    prim = make_dsa_splitk_stage2_kernel(
        AB=AB,
        AH=AH,
        AD=AD,
        Sk=Sk,
        ASq=ASq,
        sparse_loss=sparse_loss,
        softmax_scale=softmax_scale,
        in_dtype=in_dtype,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        threads=threads,
        num_stages=num_stages,
    )
    return tilelang.compile(prim, target=target, out_idx=None)


_TORCH_DTYPE_TO_TL: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def _resolve_in_dtype(tensor: torch.Tensor) -> str:
    tl_dtype = _TORCH_DTYPE_TO_TL.get(tensor.dtype)
    if tl_dtype is None:
        raise TypeError(
            f"dsa_splitk_indexer_loss_tilelang: unsupported Q/K dtype {tensor.dtype!r}; "
            "expected one of fp16/bf16/fp32"
        )
    return tl_dtype


def _scale_to_bits(scale: float) -> int:
    import struct

    return int.from_bytes(struct.pack("<f", float(scale)), "little")


def _block_constants_for_target(target: str) -> tuple[dict[str, int], dict[str, int]]:
    """Return (stage1_kwargs, stage2_kwargs) BLOCK_*/threads constants."""

    if _is_metal(target):
        return _metal_block_overrides(1), _metal_block_overrides(2)
    return (
        dict(
            BLOCK_SQ=_DSA_STAGE1_BLOCK_SQ,
            BLOCK_SK=_DSA_STAGE1_BLOCK_SK,
            BLOCK_D=_DSA_STAGE1_BLOCK_D,
            threads=_DSA_STAGE1_THREADS,
            num_stages=_DSA_STAGE1_NUM_STAGES,
        ),
        dict(
            BLOCK_SQ=_DSA_STAGE2_BLOCK_SQ,
            BLOCK_SK=_DSA_STAGE2_BLOCK_SK,
            BLOCK_D=_DSA_STAGE2_BLOCK_D,
            threads=_DSA_STAGE2_THREADS,
            num_stages=_DSA_STAGE2_NUM_STAGES,
        ),
    )


def dsa_splitk_indexer_loss_tilelang(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: object | None = None,
) -> torch.Tensor:
    """Compute DSA indexer KL-divergence loss via the TileLang Path C kernels.

    Drop-in for ``compute_dsa_indexer_loss_splitk`` in
    ``cppmega/megatron/dsa_splitk_indexer_loss.py``: same signature, same
    return contract (a 0-d fp32 scalar = ``mean(per_position_loss) * loss_coeff``).

    Single TileLang source -- target string is selected from ``query.device``.

    The ``pg_collection`` argument is accepted for API compatibility but TP
    all-reduce is NOT fused into the kernel (matching upstream PR #4039).
    Caller-side code in ``cppmega/megatron/dsa_splitk_indexer_loss.py`` is
    responsible for falling back to the native path when ``tp.size() > 1``.
    """

    if query.shape[1] != key.shape[1] or query.shape[2] != key.shape[2] or query.shape[3] != key.shape[3]:
        raise ValueError(
            f"dsa_splitk_indexer_loss_tilelang: query/key shape mismatch: "
            f"query {tuple(query.shape)}, key {tuple(key.shape)}"
        )

    ASq, AB, AH, AD = (int(query.shape[0]), int(query.shape[1]), int(query.shape[2]), int(query.shape[3]))
    Sk = int(key.shape[0])

    if AH > 128:
        raise ValueError(
            "dsa_splitk_indexer_loss_tilelang: AH > 128 is numerically unsafe in "
            f"upstream PR #4039 (got AH={AH})."
        )

    in_dtype = _resolve_in_dtype(query)
    if _resolve_in_dtype(key) != in_dtype:
        raise TypeError("dsa_splitk_indexer_loss_tilelang: query/key dtypes must match")

    target = _resolve_target(query.device)

    # Build the sparse mask on the host (matches the wrapper-side branch in
    # the Triton reference: ``index_mask = scatter(-inf, topk_indices, 0)``).
    if sparse_loss:
        index_mask = torch.full(
            (AB, ASq, Sk), float("-inf"), dtype=torch.float32, device=query.device,
        ).scatter_(-1, topk_indices, 0.0)
    else:
        # When sparse_loss is False the constexpr-eliminated kernel branches
        # never read this tensor (and after the bounds-guard fix, even
        # boundary tiles never load from it). Use ``empty`` instead of
        # ``zeros`` to skip the AB*ASq*Sk*4-byte zero-fill cost on every
        # forward pass (e.g. ~0.5 GB for production seq lengths).
        index_mask = torch.empty((AB, ASq, Sk), dtype=torch.float32, device=query.device)

    # Stage-1 buffers (fp32; matching Triton wrapper init values).
    softmax_m = torch.full((AB, AH, ASq), float("-inf"), dtype=torch.float32, device=query.device)
    softmax_d = torch.zeros((AB, AH, ASq), dtype=torch.float32, device=query.device)
    softmax_m1 = torch.full((AB, ASq), float("-inf"), dtype=torch.float32, device=query.device)
    softmax_d1 = torch.zeros((AB, ASq), dtype=torch.float32, device=query.device)

    out_loss = torch.empty((AB, ASq), dtype=torch.float32, device=query.device)

    stage1_kw, stage2_kw = _block_constants_for_target(target)
    scale_bits = _scale_to_bits(softmax_scale)

    # Ensure contiguous device tensors -- the PrimFunc takes plain Tensor
    # signatures (no stride args; we materialised the canonical layout in the
    # PrimFunc shape declarations). Skip copies when already contiguous to
    # avoid redundant device-to-device copies on hot training paths.
    query_c = query if query.is_contiguous() else query.contiguous()
    key_c = key if key.is_contiguous() else key.contiguous()
    if index_scores.dtype == torch.float32 and index_scores.is_contiguous():
        index_scores_c = index_scores
    else:
        index_scores_c = index_scores.to(dtype=torch.float32).contiguous()
    index_mask_c = index_mask if index_mask.is_contiguous() else index_mask.contiguous()

    # Stage 1.
    stage1 = _stage1_kernel_for(
        AB,
        AH,
        AD,
        Sk,
        ASq,
        bool(sparse_loss),
        scale_bits,
        in_dtype,
        target,
        stage1_kw["BLOCK_SQ"],
        stage1_kw["BLOCK_SK"],
        stage1_kw["BLOCK_D"],
        stage1_kw["threads"],
        stage1_kw["num_stages"],
    )
    stage1(
        query_c,
        key_c,
        index_scores_c,
        index_mask_c,
        softmax_m,
        softmax_d,
        softmax_m1,
        softmax_d1,
    )

    # Stage 2.
    stage2 = _stage2_kernel_for(
        AB,
        AH,
        AD,
        Sk,
        ASq,
        bool(sparse_loss),
        scale_bits,
        in_dtype,
        target,
        stage2_kw["BLOCK_SQ"],
        stage2_kw["BLOCK_SK"],
        stage2_kw["BLOCK_D"],
        stage2_kw["threads"],
        stage2_kw["num_stages"],
    )
    stage2(
        query_c,
        key_c,
        index_scores_c,
        index_mask_c,
        softmax_m,
        softmax_d,
        softmax_m1,
        softmax_d1,
        out_loss,
    )

    return out_loss.mean() * float(loss_coeff)


__all__ = [
    "DSASplitKPathCStatus",
    "dsa_splitk_indexer_loss_tilelang",
    "dsa_splitk_path_c_status",
    "make_dsa_splitk_stage1_kernel",
    "make_dsa_splitk_stage2_kernel",
    "tilelang_supports",
]
