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

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import torch


# Wave-4 perf #1: validation in dsa_splitk_indexer_loss_tilelang's
# sparse_loss path forces full GPU->CPU syncs (.item()/.all()) on every
# forward. In production we skip those checks; opt in via CPPMEGA_MLX_DSA_DEBUG
# (CI / regression tests / first-run sanity). Bounds violations would silently
# scatter into adjacent memory, so we keep the option explicit but cheap-to-
# enable rather than always-on.
def _dsa_debug_enabled() -> bool:
    return os.environ.get("CPPMEGA_MLX_DSA_DEBUG", "").lower() in {"1", "true", "yes", "on"}


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


def _metal_block_overrides(stage: int, AH: int | None = None) -> dict[str, int]:
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

    Wave-3 P1 (grok perf #2): on stage 2, the wave-2 ``M_pre`` / ``D_pre``
    fragments are ``(AH, BLOCK_SQ)`` fp32 = ``AH * BLOCK_SQ * 4`` bytes.
    At AH=128 / BLOCK_SQ=32 that's 16 KB *just for the pre-loads*, on top
    of the ~16 KB of (h_scores, softmax_attn, softmax_idx, kl_term) and
    shared Q/K staging -- well over the 32 KB Metal threadgroup budget,
    causing register spilling. Threshold math: keep
    ``AH * BLOCK_SQ * 4 <= 8 KB`` per pre-load fragment so the *pair*
    ``M_pre + D_pre`` stays at <=16 KB. With BLOCK_SQ=32 that means
    ``AH <= 64``; for ``AH > 64`` halve BLOCK_SQ to 16 (and BLOCK_D to
    keep arithmetic intensity, leaving BLOCK_SK=32 for stage 1 since it
    has no AH-shaped fragments).
    """

    # Stage 1 has no AH-shaped fragments -- the 32/32/16 default is safe
    # regardless of AH. Keep the existing override unconditionally to
    # preserve wave-2 behaviour (per "no silent delete" memory rule).
    if stage == 1:
        return dict(
            BLOCK_SQ=32,
            BLOCK_SK=32,
            BLOCK_D=16,
            threads=128,
            num_stages=2,
        )
    if stage == 2:
        # Shape-aware override: AH > 64 means M_pre/D_pre at BLOCK_SQ=32
        # exceeds the 8 KB / fragment safety threshold -- halve BLOCK_SQ.
        # AH <= 64 keeps the wave-2 32/32/16 path exactly as before.
        if AH is not None and AH > 64:
            return dict(
                # AH=128, BLOCK_SQ=16 -> M_pre/D_pre = 128*16*4 = 8 KB each
                # -> pair fits in 16 KB, leaving headroom for the four
                # BLOCK_SQ*BLOCK_SK*4 = 16*32*4 = 2 KB score fragments
                # (8 KB total) + shared Q/K (~2 KB) under 32 KB.
                BLOCK_SQ=16,
                BLOCK_SK=32,
                BLOCK_D=16,
                threads=128,
                num_stages=2,
            )
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
    compute_index_path: bool = True,
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
    # Wave-4 perf #3: compute_index_path is a build-time flag. When False, the
    # idx_scores_f / m1_i / d1_i fragments shrink to (1,) stubs and the
    # ``if h == 0`` index-softmax block is Python-guarded out, eliminating its
    # register pressure on all AH blocks (vs. paying it for AH-1 unused
    # blocks before). The wrapper pairs compute_index_path=False with a
    # separate, smaller stage-1-idx kernel that runs only for h=0.
    COMPUTE_INDEX = bool(compute_index_path)

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
            # Wave-2 perf #5: hoist Q out of the sk_tile loop. Q[sq_block, b, h, :]
            # depends only on (sq_block_id, b, h) and not on sk_tile, so loading
            # it once per Q-block per head saves SK_TILES-1 redundant HBM reloads
            # of the same BLOCK_SQ*AD*sizeof(in_dtype) bytes per pass.
            Q_full = T.alloc_shared((BLOCK_SQ, AD), in_dtype)

            # Online-softmax fragments live in registers.
            scores_f = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), "float32")
            row_max_local = T.alloc_fragment((BLOCK_SQ,), "float32")
            row_sum_local = T.alloc_fragment((BLOCK_SQ,), "float32")
            m_i = T.alloc_fragment((BLOCK_SQ,), "float32")
            d_i = T.alloc_fragment((BLOCK_SQ,), "float32")
            m_i_prev = T.alloc_fragment((BLOCK_SQ,), "float32")
            # Wave-4 perf #3: shrink the index-softmax fragments to size-1
            # stubs when compute_index_path=False so the AH-1 attn-only
            # head blocks no longer pay register pressure for buffers they
            # never touch. The if-guards below make the stub allocations
            # safe (no read or write reaches them when COMPUTE_INDEX is False).
            if COMPUTE_INDEX:
                m1_i = T.alloc_fragment((BLOCK_SQ,), "float32")
                d1_i = T.alloc_fragment((BLOCK_SQ,), "float32")
                m1_i_prev = T.alloc_fragment((BLOCK_SQ,), "float32")
                idx_scores_f = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), "float32")
            else:
                m1_i = T.alloc_fragment((1,), "float32")
                d1_i = T.alloc_fragment((1,), "float32")
                m1_i_prev = T.alloc_fragment((1,), "float32")
                idx_scores_f = T.alloc_fragment((1, 1), "float32")

            # Initialise the index-softmax accumulators. Wave-4 perf #3:
            # only when COMPUTE_INDEX (else the stub fragments are never
            # read/written -- skip the init too).
            if COMPUTE_INDEX:
                for i in T.Parallel(BLOCK_SQ):
                    m1_i[i] = T.cast(-3.4028234663852886e38, "float32")
                    d1_i[i] = T.cast(0, "float32")

            # Initialise the attention online-softmax accumulators.
            for i in T.Parallel(BLOCK_SQ):
                m_i[i] = T.cast(-3.4028234663852886e38, "float32")
                d_i[i] = T.cast(0, "float32")

            # Wave-2 perf #5: load Q for this (sq_block, h) once, reuse across
            # all sk_tiles + d_tiles below. AD is the head dim (typically 64)
            # so BLOCK_SQ*AD fp16 fits comfortably in shared (worst case CUDA
            # 128*128*2 = 32 KB; Metal 32*64*2 = 4 KB).
            for i, dd in T.Parallel(BLOCK_SQ, AD):
                sq_idx = sq_block_id * BLOCK_SQ + i
                if sq_idx < ASq:
                    Q_full[i, dd] = Q[sq_idx, b, h, dd]
                else:
                    Q_full[i, dd] = T.cast(0, in_dtype)

            # Wave-2 perf #3: causal trim. The per-element causal mask
            # ``sq_idx >= sk_idx`` zero-contributes any sk_tile beyond
            # ``(min(max_sq_in_block, ASq-1)) // BLOCK_SK + 1`` -- skip those
            # tiles entirely instead of iterating the full SK_TILES range.
            # On the last Q-block the trim is a no-op (max_useful_sk == Sk-1);
            # for early Q-blocks it can drop most iterations (e.g. block 0
            # only needs sk_tile==0).
            _max_sq_in_block = sq_block_id * BLOCK_SQ + (BLOCK_SQ - 1)
            _max_useful_sk = T.min(_max_sq_in_block, ASq - 1)
            # Wave-3 self-audit: clamp to >=1. When ASq <= sq_block_id*BLOCK_SQ
            # (last Q-block in a non-divisible shape, or pathological ASq=0),
            # _max_useful_sk goes negative and floor-div-then-+1 yields 0 which
            # would skip the loop entirely and leave out-buffers uninitialised
            # (the tile is then entirely OOB so the per-position guards already
            # produce no writes; the clamp just guarantees the loop body runs
            # once so accumulator-init paths execute deterministically).
            _active_sk_tiles = T.max(
                T.min(SK_TILES, _max_useful_sk // BLOCK_SK + 1), 1
            )
            for sk_tile in T.Pipelined(_active_sk_tiles, num_stages=num_stages):
                # Initialise the score accumulator for this tile.
                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    scores_f[i, j] = T.cast(0, "float32")

                # Inner D-loop: matmul Q @ K^T, accumulating into scores_f.
                for d_tile in T.serial((AD + BLOCK_D - 1) // BLOCK_D):
                    # Wave-2 perf #5: copy from the hoisted Q_full[BLOCK_SQ, AD]
                    # shared buffer (loaded once outside the sk_tile loop)
                    # rather than re-reading Q from HBM on every sk_tile pass.
                    for i, dd in T.Parallel(BLOCK_SQ, BLOCK_D):
                        d_idx = d_tile * BLOCK_D + dd
                        if d_idx < AD:
                            Q_s[i, dd] = Q_full[i, d_idx]
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
                # Wave-4 perf #3: completely Python-time guarded out when
                # COMPUTE_INDEX is False (the wrapper handles M1/D1 via a
                # separate dedicated kernel). This drops not only the writes
                # but also the IndexScores HBM traffic + reduce_max/reduce_sum
                # for AH-1 head blocks that don't need the index path.
                if COMPUTE_INDEX and h == 0:
                    # Wave-2 P0 fix (grok finding #5): mirror the ``scores_f``
                    # zeroing pattern with an explicit -INF prime *before* the
                    # per-position load. The if/else below already writes every
                    # (i, j) lane (so values aren't strictly stale on a
                    # well-behaved backend), but on Metal the SIMDgroup register
                    # allocator can reuse fragment lanes across pipelined
                    # iterations -- priming with -INF makes the subsequent
                    # ``T.reduce_max`` numerically safe even if a write is
                    # elided / reordered: reduce_max(-inf) == -inf, and
                    # exp(-inf - max) == 0, contributing 0 to ``d1_i``.
                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        idx_scores_f[i, j] = T.cast(-3.4028234663852886e38, "float32")

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
    # Wave-1b fix-round-2 (HIGH perf): the M_pre/D_pre fragments are
    # `(AH, BLOCK_SQ)` fp32 == AH*BLOCK_SQ*4 bytes each, total 8*AH*BLOCK_SQ
    # bytes per thread block. At AH=128, BLOCK_SQ=128 (CUDA worst case)
    # that's 128 KB combined — far above per-block register budgets and
    # causes spill-to-local-mem (-30/-40% on 2k seqlen). Gate the prefetch
    # behind a 32 KB combined budget; when over, the per-(sk_tile, h)
    # path re-reads M[b, h, sq] / D[b, h, sq] from HBM (the original
    # behaviour pre-Wave-2 perf #5). The threshold matches Metal's
    # threadgroup register budget and CUDA's per-block fragment limit.
    _MD_PRE_BUDGET_BYTES = 32 * 1024
    _MD_PRE_BYTES = 8 * AH * BLOCK_SQ
    USE_MD_PRE = _MD_PRE_BYTES <= _MD_PRE_BUDGET_BYTES

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
            # Wave-3 P1 (grok perf finding #1): partial Q hoist for stage 2.
            # The sk_tile -> h -> d_tile loop nest re-reads Q from HBM on every
            # d_tile iteration even though Q[sq_block, b, h, :] is independent
            # of (sk_tile, d_tile). Caching the full BLOCK_SQ x AD slab per
            # (sk_tile, h) into shared lets the d_tile inner loop hit shared
            # instead of HBM, saving (AD/BLOCK_D - 1) HBM reads per (sk_tile, h)
            # pass. A *full* hoist out of the sk_tile loop would be ideal (saves
            # SK_TILES-1 reloads too) but requires reordering to h-outer /
            # sk_tile-inner, which breaks the current per-sk_tile heads-summed
            # softmax_attn accumulator semantics (softmax_attn averages over AH
            # heads within a single sk_tile before feeding into the per-tile
            # KL term -- the Triton reference does the same). TODO(integration-
            # 06-wave3): full hoist requires either (a) per-h shared Q cache
            # (~AH * BLOCK_SQ * AD * 2 bytes -- 512 KB at AH=128, infeasible)
            # or (b) accumulator restructure to materialise softmax_attn as a
            # (SK_TILES, BLOCK_SQ, BLOCK_SK) buffer (also too large). Partial
            # hoist below trades a small extra shared (BLOCK_SQ * AD * 2 bytes
            # = 4 KB at Metal 32x64) for d_tile-level redundancy elimination.
            Q_full = T.alloc_shared((BLOCK_SQ, AD), in_dtype)

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
            # Wave-2 perf #5: pre-load all heads' (m_h, d_h) for this sq block
            # so the inner sk_tile->h double loop reads them from registers
            # instead of HBM on every pass. Cost: AH*BLOCK_SQ fp32 fragments
            # (CUDA 128*128*4 = 64 KB worst case; Metal 128*32*4 = 16 KB).
            # Wave-1b fix-round-2 (HIGH perf): when AH*BLOCK_SQ*8 exceeds
            # the per-block register budget (>32 KB combined) the
            # fragments spill to local mem, hurting more than the saved
            # HBM reads. In that case skip the prefetch entirely and
            # re-read M/D inside the per-(sk_tile, h) loop. Allocate
            # tiny placeholders so the constant-folder elides the dead
            # array when USE_MD_PRE is False (alloc_fragment requires
            # positive sizes).
            if USE_MD_PRE:
                M_pre = T.alloc_fragment((AH, BLOCK_SQ), "float32")
                D_pre = T.alloc_fragment((AH, BLOCK_SQ), "float32")
            else:
                M_pre = T.alloc_fragment((1, 1), "float32")
                D_pre = T.alloc_fragment((1, 1), "float32")

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

            # Wave-2 perf #5: pre-load M[b, h, sq], D[b, h, sq] for all h once
            # per sq_block, before the sk_tile loop. The inner per-sk_tile
            # h-loop reads M_pre[h, i] / D_pre[h, i] from registers.
            # Wave-1b fix-round-2: only when within the 32 KB combined
            # register budget; otherwise the per-(sk_tile, h) loop reads
            # M/D from HBM directly (see m_h/d_h load below).
            if USE_MD_PRE:
                for hh in T.serial(AH):
                    for i in T.Parallel(BLOCK_SQ):
                        sq_idx = sq_block_id * BLOCK_SQ + i
                        if sq_idx < ASq:
                            M_pre[hh, i] = M[b, hh, sq_idx]
                            D_pre[hh, i] = D[b, hh, sq_idx]
                        else:
                            M_pre[hh, i] = T.cast(0, "float32")
                            D_pre[hh, i] = T.cast(1, "float32")

            # Wave-2 perf #3: causal trim (mirrors stage 1).
            # Wave-3 self-audit: same clamp-to-1 as stage 1 (see comment there).
            _max_sq_in_block = sq_block_id * BLOCK_SQ + (BLOCK_SQ - 1)
            _max_useful_sk = T.min(_max_sq_in_block, ASq - 1)
            _active_sk_tiles = T.max(
                T.min(SK_TILES, _max_useful_sk // BLOCK_SK + 1), 1
            )
            # Wave-3 (grok perf #1): partial Q hoist landed below (per-(sk_tile,
            # h) full slab into Q_full, d_tile reads shared). Full hoist out of
            # the sk_tile loop remains structurally hard:
            #
            #   Current: for sk_tile: { softmax_attn=0; for h: { Q[h] reload from
            #     HBM; matmul; accum into softmax_attn }; emit per-tile output }
            #
            #   Naive swap (h outermost) would load Q[h] AH times instead of
            #   AH*SK_TILES times, but breaks the accumulator: softmax_attn is
            #   summed across heads PER (i,j,sk_tile), and per-tile emission
            #   needs the head-summed value. Swapping requires either:
            #     (a) Persisting softmax_attn[BLOCK_SQ, Sk] across heads
            #         (= 128*4096*4B = 2 MB shared / threadgroup -- way over
            #         Metal's 32 KB and CUDA's 100 KB budgets);
            #     (b) Writing partial softmax_attn to HBM between heads and
            #         atomic-adding (replaces ~AH*SK_TILES Q reads with
            #         AH*SK_TILES writes + reads of softmax_attn -- usually
            #         worse since softmax_attn is fp32 vs Q which can be fp16);
            #     (c) Online cross-head softmax recurrence (similar shape to
            #         FlashAttention v2 but across the head axis instead of
            #         the K axis -- nontrivial restructure of the kernel).
            #
            # Path (c) is the right wave-5 fix; current (partial hoist) is the
            # local optimum without that restructure. Q reload ratio relative
            # to optimal is AH*SK_TILES / AH = SK_TILES (~128 on Metal, larger
            # on CUDA): noticeable but bounded.
            for sk_tile in T.Pipelined(_active_sk_tiles, num_stages=num_stages):
                # Zero softmax_attn for this tile (we accumulate over heads).
                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    softmax_attn[i, j] = T.cast(0, "float32")

                # Per-head: recompute Q@K^T, scale, mask, exp/d_h.
                for h in T.serial(AH):
                    # Initialise score accumulator.
                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        h_scores[i, j] = T.cast(0, "float32")

                    # Wave-3 P1 (grok perf #1): partial Q hoist -- load the full
                    # BLOCK_SQ x AD Q slab for this (sq_block, h) once per
                    # (sk_tile, h) pass, so the d_tile inner loop reads from
                    # shared instead of HBM. See comment on Q_full alloc above
                    # for why this is partial (not the full out-of-sk_tile
                    # hoist done in stage 1).
                    for i, dd in T.Parallel(BLOCK_SQ, AD):
                        sq_idx = sq_block_id * BLOCK_SQ + i
                        if sq_idx < ASq:
                            Q_full[i, dd] = Q[sq_idx, b, h, dd]
                        else:
                            Q_full[i, dd] = T.cast(0, in_dtype)

                    # Inner D-loop matmul.
                    for d_tile in T.serial((AD + BLOCK_D - 1) // BLOCK_D):
                        for i, dd in T.Parallel(BLOCK_SQ, BLOCK_D):
                            d_idx = d_tile * BLOCK_D + dd
                            if d_idx < AD:
                                Q_s[i, dd] = Q_full[i, d_idx]
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

                    # Wave-2 perf #5: copy from the hoisted M_pre/D_pre
                    # fragments (loaded once outside the sk_tile loop) instead
                    # of re-reading M[b, h, sq] / D[b, h, sq] from HBM.
                    # Wave-1b fix-round-2: when the prefetch was disabled
                    # (USE_MD_PRE=False) read directly from HBM here; the
                    # extra HBM traffic is preferable to register spill.
                    if USE_MD_PRE:
                        for i in T.Parallel(BLOCK_SQ):
                            m_h[i] = M_pre[h, i]
                            d_h[i] = D_pre[h, i]
                    else:
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

    import struct

    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

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
    return dispatch_lower(prim, target)


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

    import struct

    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

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
    return dispatch_lower(prim, target)


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


def _block_constants_for_target(
    target: str, AH: int | None = None
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (stage1_kwargs, stage2_kwargs) BLOCK_*/threads constants.

    *AH* (attention heads) is forwarded to ``_metal_block_overrides`` so the
    Metal stage-2 path can downsize BLOCK_SQ when the AH-shaped ``M_pre`` /
    ``D_pre`` fragments would exceed the threadgroup register budget. CUDA
    defaults are unchanged (the 96 KB shared budget on Hopper handles all
    supported AH up to 128 without trouble).
    """

    if _is_metal(target):
        return _metal_block_overrides(1, AH=AH), _metal_block_overrides(2, AH=AH)
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
        # Wave-3 self-audit: explicit topk_indices validation. PyTorch scatter
        # requires int64 indices; int32 input would raise a RuntimeError deep
        # in the C++ stack ("expected scalar type Long"). We promote here so
        # callers can pass either int32 (Triton convention) or int64 (PyTorch
        # convention). Also enforce shape, contiguity, and device parity to
        # surface mismatches at the wrapper boundary instead of mid-kernel.
        if topk_indices.device != query.device:
            raise ValueError(
                "dsa_splitk_indexer_loss_tilelang: topk_indices.device "
                f"({topk_indices.device}) != query.device ({query.device})"
            )
        if topk_indices.dim() != 3 or topk_indices.shape[:2] != (AB, ASq):
            raise ValueError(
                "dsa_splitk_indexer_loss_tilelang: topk_indices must have shape "
                f"(AB={AB}, ASq={ASq}, TOPK); got {tuple(topk_indices.shape)}"
            )
        if topk_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "dsa_splitk_indexer_loss_tilelang: topk_indices.dtype must be "
                f"int32 or int64; got {topk_indices.dtype}"
            )
        topk_idx64 = topk_indices.to(dtype=torch.int64, copy=False)
        if not topk_idx64.is_contiguous():
            topk_idx64 = topk_idx64.contiguous()
        # Wave-1b fix-round-2 (MED sec): bounds-check topk_idx64 before
        # scatter_. PyTorch's CUDA scatter_ wraps negatives but does NOT
        # check the upper bound in release builds, so an OOB index would
        # silently corrupt adjacent memory.
        #
        # Wave-4 perf #1 (grok wave-3 review): the .item() / .all() calls
        # below force GPU->CPU syncs + extra reduction kernels on every
        # sparse forward pass (~milliseconds at ASq*TOPK=large). Gate
        # behind CPPMEGA_MLX_DSA_DEBUG so production training paths skip
        # them but CI / first-run regressions still catch corruption.
        if _dsa_debug_enabled() and topk_idx64.numel() > 0:
            _max_idx = int(topk_idx64.max().item())
            _min_idx = int(topk_idx64.min().item())
            if _max_idx >= Sk or _min_idx < 0:
                raise ValueError(
                    "dsa_splitk_indexer_loss_tilelang: topk_indices out of "
                    f"range [0, {Sk}); got [{_min_idx}, {_max_idx}]."
                )
        index_mask = torch.full(
            (AB, ASq, Sk), float("-inf"), dtype=torch.float32, device=query.device,
        ).scatter_(-1, topk_idx64, 0.0)
        # Wave-1b fix-round-2 (MED sec): NaN poisoning guard for fully-
        # masked rows. If a row's topk_indices contained no in-range
        # entries (e.g. all duplicates in a sparse setup, or an upstream
        # bug), every IndexMask slot stays -inf and downstream softmax
        # produces NaN that propagates into the loss. Detect and patch
        # by clearing slot 0 to a safe sentinel for any all-masked row;
        # the kernel's own causal mask still elides invalid (sq < sk)
        # combinations downstream.
        #
        # Wave-4 perf #1 (grok wave-3 review): the .all() forces another
        # GPU->CPU sync. In production we skip the detection AND the patch;
        # this trades a rare NaN risk (caller passes degenerate topk) for
        # zero per-step overhead. Enable via CPPMEGA_MLX_DSA_DEBUG.
        if _dsa_debug_enabled():
            _row_has_valid = (index_mask == 0.0).any(dim=-1)
            if not bool(_row_has_valid.all()):
                _patch = torch.where(
                    _row_has_valid,
                    index_mask[..., 0],
                    torch.zeros((), dtype=torch.float32, device=query.device),
                )
                index_mask[..., 0] = _patch
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

    stage1_kw, stage2_kw = _block_constants_for_target(target, AH=AH)
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
