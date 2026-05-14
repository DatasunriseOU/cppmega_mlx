"""Path B/Path C ports of cppmega's TileLang topk-selector kernel.

Source attribution
------------------

Forward source on gb10:
    cppmega/megatron/tilelang_sparse_mla/topk_selector.py
License:
    cppmega upstream tracks NVIDIA Megatron-LM PR #3674 ("DSA thd" branch),
    which carries the same Apache 2.0 / BSD-3-Clause headers as Megatron-LM.
    The vendored fragments below are limited to the kernel structure and
    radix-select strategy; they do not duplicate proprietary content.

What the source kernel does
---------------------------

``topk_selector(input, starts, ends, topk)`` returns, per batch row, the
``topk`` indices into ``input[bx, starts[bx]:ends[bx]]`` of the largest values.
The CUDA implementation runs a two-stage radix-select inside one threadgroup:

  * Stage 1 builds a 256-bin histogram from the high byte of a sign-flipped
    fp16/fp32 representation, then prefix-sums it via a Hillis-Steele scan
    over 256 threads, finds the threshold bucket, emits "definitely above"
    indices into the output and tail candidates into shared memory.
  * Stage 2 (up to 4 rounds) re-runs the radix scan one byte deeper to
    refine the threshold and finalize tail emission.

It uses:
  * ``T.alloc_shared([257], int32)`` for the histogram
  * ``T.alloc_shared([2, 4096], int32)`` for the tail-candidate buffer
  * ``T.atomic_add(..., return_prev=True)`` to assign output positions
  * ``T.sync_threads(3, RADIX)`` partial barriers covering only the first
    256 threads
  * BLOCK_SIZE=1024 threadgroup size with RADIX=256

Apple Metal status (TileLang 0.1.9, MLX 0.31.x)
-----------------------------------------------

The original CUDA-style TileLang schedule through
``tilelang.engine.lower(.., target=Target('metal'))`` remained blocked even
after multiple probe sweeps:

  * ``T.alloc_shared`` on every layout we tried lowers to storage scope
    ``shared.dyn`` which TVM's Metal codegen rejects with
    ``Fatal: Unknown storage scope shared.dyn``.
  * Histogram fills sized ``RADIX+1`` over BLOCK_SIZE threads fail
    LowerTileOp's injective-layout check.

Direct-MSL bypass and TileLang Path C (this module)
---------------------------------------------------

Path B skips TileLang entirely and emits MSL by hand through
``mx.fast.metal_kernel``. Path C keeps a real TileLang DSL ``@T.prim_func``,
but uses the same Metal-friendly one-threadgroup-per-row algorithm instead of
the upstream radix/histogram schedule. The production Path C surface is the
explicit tvm-ffi owner-output route
``topk_selector_tilelang_direct(..., out=...)``; the older no-``out``
``mx.fast.metal_kernel`` wrapper is disabled by default and reserved for debug
probes. Both variants cooperate via *static* (compile-time-sized)
``threadgroup`` arrays - no ``shared.dyn`` scope, no off-by-one fills.

Performance vs MLX's built-in ``argpartition``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MLX's ``mx.argpartition`` is itself an optimised Metal kernel; bench results
in ``bench/tilelang_ports/topk_selector.json`` show that for the
top-k-only contract our hand-written MSL kernel runs ~0.2-1x argpartition.
We keep the direct-MSL kernel because:

1. It demonstrates the bypass-TileLang pattern with the same correctness
   contract as gb10's CUDA topk_selector.
2. The ``[starts, ends)`` per-row mask is fused into a single kernel
   (argpartition needs an extra ``mx.where`` masking pass).
3. It is the building block for fused selection inside the sparse-MLA
   kernels in this same package (where MSL wins decisively because the
   selection happens inline with the attention reduction).

If you want raw top-k speed, ``backend="mlx"`` (or the ``mx.argpartition``
reference) is preferable. ``backend="metal"`` is the right choice when the
``[starts, ends)`` mask is set or when integrating into a downstream MSL
pipeline.

Algorithm sketch::

  for each (b) in parallel:
    # Each thread sweeps a strided slice and keeps a private sorted list of
    # its local K largest valid entries.
    for j = tid; j < T; j += threads:
      if starts[b] <= j < ends[b]:
        insert (score, j) into local top-K list

    # All threads write their local top-K lists to a bounded threadgroup
    # buffer. A tree merge combines pairs of lists until thread 0 owns the
    # final top-K list for the row.
    for stride in powers_of_two:
      merge top-K(pair_a, pair_b) -> pair_a

The kernel avoids an O(T_PAD) shared buffer. Threadgroup memory is bounded by
``BLOCK_SIZE * K * (sizeof(float) + sizeof(int))`` and the host caps
``BLOCK_SIZE`` for the requested K so the pair buffer stays within the M-series
threadgroup budget.

The mini-heap/list approach is O(T / threads * K + BLOCK_SIZE * K * log threads)
with small compile-time K. It is not faster than MLX ``argpartition`` as a
standalone top-k, but it is a useful direct-MSL smoke and a future fused
sparse-MLA building block.

Return contract:

  * Output indices are int32, shape (B, k).
  * Output ordering matches the source kernel contract: the indices come back
    in *value-descending* order (the radix kernel produces them in arbitrary
    fill order, but this MSL kernel writes them out greedy by-rank).
    Tests assert *set* equality with the reference, not sequence equality,
    so this divergence is invisible at the test boundary.
  * Sentinel handling: if ``starts``/``ends`` produce fewer than ``k`` valid
    columns, the valid indices are emitted first and remaining output slots are
    filled with ``-1``.
"""

from __future__ import annotations

# pyright: reportInvalidTypeForm=false

import importlib
import os
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._engine_dispatch import (
    dispatch_lower,
)


# CPPMEGA Z3 wiring (beads cppmega-mlx-cuz):
#
# Two surfaces live in this module:
#   * The Path B direct-MSL kernel (``mx.fast.metal_kernel`` with hand-written
#     MSL strings) -- TileLang lowering does NOT run for these so PassConfigs
#     do not apply. Z3 idea #11 (intra-warp barrier elision) would require
#     either porting the Path B kernel onto the TileLang DSL or adding a
#     post-MSL textual barrier-elision pass; neither is in scope for the
#     ``cppmega-mlx-cuz`` wiring task.
#   * The Path C TileLang DSL kernel built in ``_path_c_tvm_ffi_kernel_for`` --
#     this compiles through native tvm-ffi for owner-output production calls.
#     The debug-only ``_path_c_kernel_for`` wrapper still requests extracted
#     MSL for the legacy no-out probe, and both routes opt into the Z3
#     PassConfigs registered in the active libtilelang build.
#
# Idea #4 (``tl.drop_provable_bound_checks``) is the PassConfig we expect to
# materially affect codegen on the Path C topk merge kernel: the
# ``lane + stride < THREADS`` guard inside the merge-tree is the kind of
# bound check the analyzer-or-Z3 prover discharges. The central barrier proof
# hook also enables TileLang's Metal merge-round cleanup pass, replacing the
# old cppmega-side MSL regex rewrite. Idea #9 (``tl.simd_lift_reductions``)
# is declared in the TileLang source tree but may not yet be registered in
# every in-tree libtilelang build; we filter unsupported keys at runtime.
_TOPK_PATH_C_CANDIDATE_PASS_CONFIGS: dict[str, Any] = {
    "tl.drop_provable_bound_checks": True,
    "tl.z3_proof.barrier_minimization": True,
    "tl.simd_lift_reductions": True,  # filtered out at runtime if not registered
}

_TOPK_PATH_C_FILTERED_KEYS_LOGGED: set[str] = set()
_TOPK_PATH_C_PASS_CONFIGS_CACHE: dict[str, Any] | None = None
# Guards first-time populate of ``_TOPK_PATH_C_PASS_CONFIGS_CACHE`` so two
# MLX threads lowering this kernel concurrently don't race the probe loop.
_topk_path_c_pass_configs_cache_lock = threading.Lock()
_TOPK_LEGACY_NO_OUT_ENV = "CPPMEGA_TOPK_PATH_C_LEGACY_NO_OUT"


def _topk_filter_supported_pass_configs(candidates: dict[str, Any]) -> dict[str, Any]:
    """Drop PassConfig keys not registered in the active libtilelang build."""

    try:
        from tilelang import tvm  # type: ignore
    except Exception:
        return {}

    supported: dict[str, Any] = {}
    for key, value in candidates.items():
        try:
            with tvm.transform.PassContext(opt_level=3, config={key: value}):
                pass
        except Exception:
            if key not in _TOPK_PATH_C_FILTERED_KEYS_LOGGED:
                _TOPK_PATH_C_FILTERED_KEYS_LOGGED.add(key)
                print(
                    f"[cppmega-mlx-cuz] dropping unsupported PassConfig "
                    f"key {key!r} from topk_selector path-c lowering "
                    f"(not registered in active libtilelang).",
                    file=sys.stderr,
                )
            continue
        supported[key] = value
    return supported


def _topk_path_c_pass_configs() -> dict[str, Any]:
    """Return the PassConfig dict to thread through this kernel's lowering.

    The env var ``CPPMEGA_TOPK_PATH_C_NO_Z3`` forces the legacy lowering
    (no PassContext) for parity tests / debug.
    """

    if os.environ.get("CPPMEGA_TOPK_PATH_C_NO_Z3", "0") not in (
        "0",
        "",
        "false",
        "False",
    ):
        return {}
    global _TOPK_PATH_C_PASS_CONFIGS_CACHE
    with _topk_path_c_pass_configs_cache_lock:
        if _TOPK_PATH_C_PASS_CONFIGS_CACHE is None:
            _TOPK_PATH_C_PASS_CONFIGS_CACHE = _topk_filter_supported_pass_configs(
                _TOPK_PATH_C_CANDIDATE_PASS_CONFIGS
            )
        return dict(_TOPK_PATH_C_PASS_CONFIGS_CACHE)


def _topk_legacy_no_out_enabled() -> bool:
    """Return whether the debug-only no-owner-output Path C wrapper is enabled."""

    return os.environ.get(_TOPK_LEGACY_NO_OUT_ENV, "0") not in (
        "0",
        "",
        "false",
        "False",
    )


# TileLang's eager builder reads these module globals after _path_c_kernel_for
# overwrites them for the current shape. Static placeholders keep pyright from
# treating the macro globals inside the nested @T.prim_func as missing.
_TOPK_C_B = 1
_TOPK_C_N = 1
_TOPK_C_K = 1
_TOPK_C_THREADS = 1
_TOPK_C_LOG_THREADS = 0
_TOPK_C_SCORE_DTYPE = "float32"


# ---------------------------------------------------------------------------
# Public reference implementation.
# ---------------------------------------------------------------------------

def topk_selector_reference(
    scores: mx.array,
    k: int,
    *,
    starts: mx.array | None = None,
    ends: mx.array | None = None,
) -> mx.array:
    """Pure-MLX reference for cppmega's topk_selector.

    Returns the column indices (into the last axis of ``scores``) of the
    ``k`` largest values per row. Output dtype is ``mx.int32`` to match the
    cppmega source contract.

    Parameters
    ----------
    scores:
        ``(B, T)`` array of values to rank. Must have rank 2.
    k:
        Number of indices to return per row. Must satisfy ``1 <= k <= T``.
    starts, ends:
        Optional ``(B,)`` int32 arrays giving an inclusive lower bound and
        exclusive upper bound on valid columns per row. Indices outside
        ``[starts[b], ends[b])`` are masked to ``-inf`` before selection.

    The order of the returned indices matches the order produced by
    ``mx.argpartition(-scores, k, axis=-1)[..., :k]``: it is *not* sorted by
    score. Tests must therefore compare set membership, not sequence
    equality, to the source kernel's output.
    """

    if scores.ndim != 2:
        raise ValueError(
            f"topk_selector expects a (B, T) array; got shape {scores.shape}"
        )
    if k < 1:
        raise ValueError(f"topk must be >= 1; got {k}")
    seq_len = int(scores.shape[1])
    if k > seq_len:
        raise ValueError(
            f"topk={k} exceeds sequence length {seq_len}"
        )

    # Apply the [starts, ends) mask if present.
    masked = scores
    has_interval_mask = starts is not None or ends is not None
    if has_interval_mask:
        batch = int(scores.shape[0])
        start_arr = (
            mx.zeros((batch,), dtype=mx.int32)
            if starts is None
            else starts
        )
        end_arr = (
            mx.full((batch,), seq_len, dtype=mx.int32)
            if ends is None
            else ends
        )
        if start_arr.shape != (batch,) or end_arr.shape != (batch,):
            raise ValueError(
                "starts/ends must have shape (B,); "
                f"got starts={start_arr.shape}, ends={end_arr.shape}"
            )
        start_i = mx.minimum(
            mx.maximum(start_arr.astype(mx.int32), mx.array(0, dtype=mx.int32)),
            mx.array(seq_len, dtype=mx.int32),
        )
        end_i = mx.minimum(
            mx.maximum(end_arr.astype(mx.int32), mx.array(0, dtype=mx.int32)),
            mx.array(seq_len, dtype=mx.int32),
        )
        valid_counts = mx.maximum(end_i - start_i, mx.array(0, dtype=mx.int32))
        col = mx.arange(seq_len, dtype=mx.int32)[None, :]
        valid = (col >= start_i[:, None]) & (col < end_i[:, None])
        # Use the smallest representable value of `scores.dtype` for masking.
        if scores.dtype == mx.float32 or scores.dtype == mx.float16 or scores.dtype == mx.bfloat16:
            mask_value = mx.array(float("-inf"), dtype=scores.dtype)
        else:
            mask_value = mx.array(-2**31, dtype=scores.dtype)
        masked = mx.where(valid, scores, mask_value)
    else:
        valid_counts = mx.full(
            (int(scores.shape[0]),),
            seq_len,
            dtype=mx.int32,
        )

    # argpartition picks the indices of the `k` largest values (when sorting
    # the *negated* scores ascending, the first k elements are those with
    # the largest original scores). MLX guarantees the partition contract;
    # ordering within the slice is implementation-defined.
    if k == seq_len and not has_interval_mask:
        # argpartition with kth == size is undefined for some backends; just
        # take an arange.
        order = mx.broadcast_to(
            mx.arange(seq_len, dtype=mx.int32)[None, :],
            scores.shape,
        )
        part = order
    elif k == seq_len:
        part = mx.argsort(-masked, axis=-1).astype(mx.int32)
    else:
        part = mx.argpartition(-masked, kth=k, axis=-1)[..., :k]
    rank = mx.arange(k, dtype=mx.int32)[None, :]
    return mx.where(rank < valid_counts[:, None], part.astype(mx.int32), -1)


# ---------------------------------------------------------------------------
# Direct-MSL Path B kernel (bypasses TileLang entirely).
# ---------------------------------------------------------------------------


_TOPK_SOURCE = """
    // One threadgroup per row, BLOCK_SIZE threads cooperate.
    //
    // Inputs:
    //   scores  [B, T]   float
    //   starts  [B]      int32   (ignored when HAS_STARTS == 0)
    //   ends    [B]      int32   (ignored when HAS_ENDS == 0)
    //
    // Output:
    //   indices [B, K]   int32   (k largest indices, value-descending)
    //
    // Algorithm: each thread builds a private heap of size K over its
    // strided slice, then we merge BLOCK_SIZE local heaps via shared memory
    // tree reduction. This is O(T/B * log K + B * K * log K) which beats
    // the K-passes approach when K is large. For small K (<=64) the constant
    // factor dominates either way; for K up to 256 this scales well.
    //
    // We avoid an O(T_PAD) shared buffer entirely (which was the binding
    // budget on M-series at T_PAD=4096) and use only:
    //   - thread-local arrays of size K
    //   - shared merge buffer of size 2*K floats + 2*K ints
    //   - one int32 reduction scratch
    // total threadgroup memory: 2*K*8 bytes + 8 bytes ~= 16*K + 8.

    // Per-thread mini-heap (min-heap; we keep the K largest seen so far,
    // so root is the smallest of those K and any new value > root replaces
    // it). Implemented as a sorted insertion list of size K (ascending).
    //
    // Local arrays use compile-time-sized K so the compiler can keep
    // them in registers / private memory.
    float local_vals[K];
    int   local_idx[K];

    threadgroup float merge_vals[K_DOUBLE];
    threadgroup int   merge_idx[K_DOUBLE];

    uint b = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint threads = BLOCK_SIZE;

    if (b >= uint(BATCH)) {
        return;
    }

    int s_start = 0;
    int s_end = int(SEQ_LEN);
    if (HAS_STARTS != 0) {
        int sv = starts[b];
        if (sv >= 0) s_start = sv;
    }
    if (HAS_ENDS != 0) {
        int ev = ends[b];
        s_end = ev;
    }
    if (s_start < 0) s_start = 0;
    if (s_start > int(SEQ_LEN)) s_start = int(SEQ_LEN);
    if (s_end < 0) s_end = 0;
    if (s_end > int(SEQ_LEN)) s_end = int(SEQ_LEN);

    uint base_in = b * uint(SEQ_LEN);
    uint base_out = b * uint(K);

    // Initialise local heap with -INF, idx -1.
    for (uint j = 0; j < uint(K); ++j) {
        local_vals[j] = -INFINITY;
        local_idx[j]  = -1;
    }

    // Phase 1: each thread sweeps its strided slice and inserts into local heap.
    // local_vals is kept sorted ascending so position 0 is the smallest of
    // the top-K so far.
    for (uint j = tid; j < uint(SEQ_LEN); j += threads) {
        int jj = int(j);
        bool valid = (jj >= s_start) && (jj < s_end);
        if (!valid) continue;
        float v = float(scores[base_in + j]);
        // If v <= smallest-of-top-K, skip.
        if (v <= local_vals[0]) continue;
        // Find insertion point so the array stays sorted ascending.
        // Walk from index 0 upward; shift left smaller-than-v entries.
        // Slot 0 falls off (it is the previous minimum).
        uint pos = 0;
        // Use a fixed unrolled bound K. Compiler unrolls.
        for (uint p = 1; p < uint(K); ++p) {
            if (v > local_vals[p]) {
                local_vals[p - 1] = local_vals[p];
                local_idx[p - 1]  = local_idx[p];
                pos = p;
            } else {
                break;
            }
        }
        local_vals[pos] = v;
        local_idx[pos]  = jj;
    }

    // Phase 2: merge BLOCK_SIZE local heaps via tree reduction.
    // At each step pairs of threads merge their lists into a single list of
    // size K (taking the K largest of the 2K combined). The active half
    // halves on each step.
    //
    // We write our local list into shared memory in canonical descending
    // order so merging is "two-pointer pick-larger" linear scan.

    // Use one half of merge buffer per thread of the pair.
    // Layout: thread 'a' writes [tid * 0 .. K] (left list),
    //         thread 'b' would write [K .. 2K]; we use a slot per active
    //         thread but with stride 2*K, so we need len BLOCK_SIZE * 2 * K.
    // That's too big. Instead we do an in-place pairwise merge in halves.

    // Single-merge-buffer scheme:
    //   The active threads are tid in [0, count). Pair (a=2t, b=2t+1).
    //   Both write to merge_vals[2 * t * K .. 2 * t * K + 2K] but we only have
    //   K_DOUBLE = 2K slots total; we therefore process pairs serially.
    // To avoid a serialization bottleneck we split the merge into log2(threads)
    // rounds where round r merges threads 2t into 2t+1's slot. Each round
    // halves the active count. Each round needs only (K_DOUBLE) shared slots,
    // but is naturally serialized across rounds; within a round all surviving
    // pairs work in parallel by partitioning the merge buffer.
    //
    // Simpler and correct: emit into a large per-thread shared block. Apple's
    // 32KiB threadgroup budget allows BLOCK_SIZE=64, K=256 -> 64*256*8 = 128 KiB
    // overflow. So we fall back to round-by-round merges using a single shared
    // buffer of 2K slots, with BLOCK_SIZE-1 sequential merge steps.
    //
    // Simpler alternative (chosen): each thread writes its sorted-descending
    // top-K to a global staging buffer in shared memory of size BLOCK_SIZE * K.
    // For BLOCK_SIZE=32, K=256 -> 32*256*8 = 64KiB > 32KiB budget. So we cap
    // BLOCK_SIZE * K * 8 <= 32KiB at dispatch time.

    // Step 1: each thread writes its descending-order top-K to merge buffer
    // at offset tid * 0 ... no, we use a different scheme. To keep the
    // shared budget low we do log2(BLOCK_SIZE) rounds with thread 0 taking
    // over more slots each round.
    //
    // Round r (r = 0..log2(BLOCK_SIZE)-1):
    //   Active threads: tid such that tid % (1 << (r+1)) == 0.
    //   For each active thread t = tid, partner p = tid + (1 << r).
    //   Their lists are in registers; t's list is local_vals/idx (length K),
    //   p's list is in shared memory at offsets [p*K .. p*K + K].
    //   Merge via two-pointer sweep into local_vals (taking K largest).
    //
    // This requires that *every* thread first writes its local_vals/idx into
    // shared memory under offset [tid * K .. tid * K + K]. That uses
    // BLOCK_SIZE * K * 8 bytes; we cap that at dispatch time.
    //
    // Note: K_DOUBLE = 2 * K is provided by the host as a template constant
    // because MSL compile-time arithmetic on template constants is limited.

    // (no-op marker so we can keep K_DOUBLE referenced as a template arg)
    (void)merge_vals[0];
    (void)merge_idx[0];

    // Tree reduction: each round we treat two lists of size K, merge into
    // list of size K (top-K of the 2K combined).
    // We use a single shared scratch sized BLOCK_SIZE * K floats; it lives
    // as `pair_vals`, `pair_idx`. Allocated below. We also need the scratch
    // to outlive this scope, so it's threadgroup-allocated at the top.

    // (Implementation moved to extra threadgroup arrays declared at top.)

    // The full implementation requires more than 32KiB of threadgroup memory
    // at K=256, BLOCK_SIZE=64. We therefore cap BLOCK_SIZE at the host based
    // on K so the merge buffer fits.

    // Write our local sorted-ascending list to merge buffer at slot `tid`.
    // We reuse the merge_vals/merge_idx buffers across rounds; the first
    // round needs BLOCK_SIZE * K slots, the next BLOCK_SIZE/2 * K, etc.
    // The buffer is sized BLOCK_SIZE * K (allocated as a separate pair_*
    // buffer below). To respect the threadgroup budget, the host sets
    // BLOCK_SIZE = min(256, max(1, 32768 / (K * 8) - safety)).

    // PHASE 2 (tree merge): write local list to per-thread slot in pair_*
    // arrays.
    // Note these arrays are statically sized as PAIR_BUF = BLOCK_SIZE * K.

    threadgroup float pair_vals[PAIR_BUF];
    threadgroup int   pair_idx[PAIR_BUF];

    uint slot_base = tid * uint(K);
    for (uint i = 0; i < uint(K); ++i) {
        pair_vals[slot_base + i] = local_vals[i];
        pair_idx[slot_base + i]  = local_idx[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree merge.
    for (uint stride = 1; stride < threads; stride <<= 1) {
        if ((tid % (stride << 1)) == 0u) {
            uint other = tid + stride;
            if (other < threads) {
                // Local list is stored at slot_base, other's at other*K.
                // Both ascending sorted (length K). Merge -> take top K.
                // Output: write top K (in ascending order) back into slot_base.
                // Use local arrays for the merge accumulator.
                float merged_v[K];
                int   merged_i[K];
                // Two-pointer scan from the *largest* end (descending walk).
                // a-end = K-1, b-end = K-1; produce K largest into merged_v
                // in *descending* order, then reverse.
                int ap = int(K) - 1;
                int bp = int(K) - 1;
                for (uint t = 0; t < uint(K); ++t) {
                    float a_v = (ap >= 0) ? pair_vals[slot_base + uint(ap)] : -INFINITY;
                    float b_v = (bp >= 0) ? pair_vals[other * uint(K) + uint(bp)] : -INFINITY;
                    if (a_v >= b_v) {
                        merged_v[uint(K) - 1 - t] = a_v;
                        merged_i[uint(K) - 1 - t] = (ap >= 0) ? pair_idx[slot_base + uint(ap)] : -1;
                        ap -= 1;
                    } else {
                        merged_v[uint(K) - 1 - t] = b_v;
                        merged_i[uint(K) - 1 - t] = (bp >= 0) ? pair_idx[other * uint(K) + uint(bp)] : -1;
                        bp -= 1;
                    }
                }
                for (uint i = 0; i < uint(K); ++i) {
                    pair_vals[slot_base + i] = merged_v[i];
                    pair_idx[slot_base + i]  = merged_i[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 holds the merged top-K in ascending order at slot 0..K.
    // The contract emits them in *descending* order (largest first).
    if (tid == 0) {
        for (uint i = 0; i < uint(K); ++i) {
            indices[base_out + i] = pair_idx[uint(K) - 1 - i];
        }
    }
"""


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


_TOPK_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_topk_selector",
    input_names=["scores", "starts", "ends"],
    output_names=["indices"],
    source=_TOPK_SOURCE,
)


def topk_selector_metal(
    scores: mx.array,
    k: int,
    *,
    starts: mx.array | None = None,
    ends: mx.array | None = None,
) -> mx.array | None:
    """Direct-MSL Path B forward.

    Returns ``None`` if Metal is not eligible (no Metal device, dtype
    unsupported, etc.). The MSL kernel uses static ``BLOCK_SIZE * K`` pair
    buffers, and the host picks a power-of-two ``BLOCK_SIZE`` that keeps the
    pair merge scratch within the M-series threadgroup-memory budget.
    """

    if scores.ndim != 2:
        raise ValueError(
            f"topk_selector expects a (B, T) array; got shape {scores.shape}"
        )
    batch = int(scores.shape[0])
    seq_len = int(scores.shape[1])
    if k < 1 or k > seq_len:
        raise ValueError(f"topk must be in [1, {seq_len}]; got {k}")

    if _TOPK_KERNEL is None:
        return None
    status = _msl_transform.msl_dispatch_status(scores)
    if not status.available:
        return None

    # Threadgroup memory budget on M-series Apple GPUs is 32 KiB per
    # threadgroup. Each pair_vals/pair_idx slot is 4 bytes; total slot cost
    # is 8 bytes per (block_size, k) combo. We reserve ~24 KiB for the
    # pair buffers so the rest stays available for the kernel runtime / MSL
    # spill (Apple's compiler also stages stack frames in threadgroup mem
    # when register pressure is high).
    max_pair_slots = 24 * 1024 // 8  # = 3072
    cap_for_k = max(1, max_pair_slots // max(int(k), 1))
    # Prefer a power of 2 (tree reduction halves each round).
    block_size = min(64, cap_for_k)
    p = 1
    while (p << 1) <= block_size:
        p <<= 1
    block_size = p
    if block_size < 1:
        block_size = 1

    # The MSL kernel does an fp32 internal compare; we promote at the boundary.
    work_scores = scores
    if scores.dtype not in (mx.float32, mx.float16):
        work_scores = scores.astype(mx.float32)

    has_starts = 1 if starts is not None else 0
    has_ends = 1 if ends is not None else 0
    if starts is None:
        starts_buf = mx.full((batch,), -1, dtype=mx.int32)
    else:
        if starts.shape != (batch,):
            raise ValueError(
                f"starts must have shape ({batch},); got {starts.shape}"
            )
        starts_buf = starts.astype(mx.int32)
    if ends is None:
        ends_buf = mx.full((batch,), -1, dtype=mx.int32)
    else:
        if ends.shape != (batch,):
            raise ValueError(
                f"ends must have shape ({batch},); got {ends.shape}"
            )
        ends_buf = ends.astype(mx.int32)

    pair_buf = block_size * int(k)
    k_double = 2 * int(k)
    template = [
        ("BATCH", batch),
        ("SEQ_LEN", seq_len),
        ("K", int(k)),
        ("K_DOUBLE", int(k_double)),
        ("PAIR_BUF", int(pair_buf)),
        ("BLOCK_SIZE", int(block_size)),
        ("HAS_STARTS", int(has_starts)),
        ("HAS_ENDS", int(has_ends)),
    ]

    try:
        outputs = _msl_transform.dispatch(
            _TOPK_KERNEL,
            inputs=[work_scores, starts_buf, ends_buf],
            output_shapes=[(batch, int(k))],
            output_dtypes=[mx.int32],
            grid=(batch * block_size, 1, 1),
            threadgroup=(block_size, 1, 1),
            template=template,
        )
    except Exception:
        return None
    return outputs[0]


# ---------------------------------------------------------------------------
# TileLang DSL Path C kernel.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathCStatus:
    """Why Path C (TileLang DSL -> Metal -> MLX) is or isn't available."""

    available: bool
    reason: str


class TopKPathCDirectError(RuntimeError):
    """Raised when the owner-output tvm-ffi topk route cannot run safely."""


_PATH_C_OK_REASON = (
    "topk_selector TileLang DSL Path C is available; the shape-specialized "
    "threadgroup merge kernel lowers to static Metal threadgroup buffers."
)


@lru_cache(maxsize=1)
def _tilelang_available() -> tuple[bool, str]:
    def _try_import_tilelang() -> tuple[bool, str]:
        try:
            import tilelang  # type: ignore[reportMissingImports]  # noqa: F401
            from tilelang import tvm as _tvm  # type: ignore[reportMissingImports]  # noqa: F401
            import tilelang.language as _T  # type: ignore[reportMissingImports]  # noqa: F401
        except Exception as exc:  # pragma: no cover - host without TileLang build
            return False, f"tilelang import failed: {exc}"
        return True, "tilelang importable"

    ok, reason = _try_import_tilelang()
    if ok:
        return ok, reason

    # The cppmega test/bench environment keeps TileLang in a sibling dev tree,
    # and tests deliberately scrub TILELANG_ROOT/PYTHONPATH between cases.
    # Reuse the bench harness bootstrap once before treating Path C as absent.
    try:
        from scripts.bench_tilelang_fp8_path_c import (
            _prepare_tilelang_import_environment,
        )

        _prepare_tilelang_import_environment()
        importlib.invalidate_caches()
    except Exception as prep_exc:  # pragma: no cover - host without dev tree
        return False, f"{reason}; TileLang dev import bootstrap failed: {prep_exc}"

    return _try_import_tilelang()


@lru_cache(maxsize=1)
def topk_selector_path_c_status() -> PathCStatus:
    """Return whether the TileLang DSL Path C topk selector can dispatch."""

    if not _msl_transform.can_run_metal():
        return PathCStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    ok, reason = _tilelang_available()
    if not ok:
        return PathCStatus(available=False, reason=reason)
    return PathCStatus(available=True, reason=_PATH_C_OK_REASON)


def _path_c_threads_for(k: int) -> int:
    """Pick a power-of-two threadgroup that fits Apple's 32 KiB shared budget."""

    max_shared_bytes = 32 * 1024
    max_threads_for_k = max(1, max_shared_bytes // (max(k, 1) * 8))
    # The Path C schedule is scalar top-K insertion plus a shared-memory tree
    # merge. k<=32 is latency-bound on M4 Max and the paired local sweep showed
    # 32 lanes beating narrower groups; K>=64 benefits from the wider scan.
    preferred_threads = 32 if k <= 32 else 64
    threads = min(preferred_threads, max_threads_for_k)
    pow2 = 1
    while (pow2 << 1) <= threads:
        pow2 <<= 1
    return max(1, pow2)


def _path_c_score_dtype(scores: mx.array) -> tuple[str, mx.array] | None:
    if scores.dtype == mx.float32:
        return "float32", scores
    if scores.dtype == mx.float16:
        return "float16", scores
    if scores.dtype == mx.bfloat16:
        return "float32", scores.astype(mx.float32)
    return None


def _path_c_score_dtype_direct(scores: mx.array) -> str | None:
    if scores.dtype == mx.float32:
        return "float32"
    if scores.dtype == mx.float16:
        return "float16"
    return None


@lru_cache(maxsize=128)
def _path_c_default_interval_buffers(batch: int, seq_len: int) -> tuple[mx.array, mx.array]:
    return (
        mx.zeros((batch,), dtype=mx.int32),
        mx.full((batch,), seq_len, dtype=mx.int32),
    )


def _path_c_interval_buffers(
    *,
    batch: int,
    seq_len: int,
    starts: mx.array | None,
    ends: mx.array | None,
) -> tuple[mx.array, mx.array]:
    if starts is None and ends is None:
        return _path_c_default_interval_buffers(batch, seq_len)

    zero = mx.array(0, dtype=mx.int32)
    seq = mx.array(seq_len, dtype=mx.int32)
    if starts is None:
        starts_i = mx.zeros((batch,), dtype=mx.int32)
    else:
        if starts.shape != (batch,):
            raise ValueError(f"starts must have shape ({batch},); got {starts.shape}")
        starts_i = starts.astype(mx.int32)
    if ends is None:
        ends_i = mx.full((batch,), seq_len, dtype=mx.int32)
    else:
        if ends.shape != (batch,):
            raise ValueError(f"ends must have shape ({batch},); got {ends.shape}")
        ends_i = ends.astype(mx.int32)
    return mx.minimum(mx.maximum(starts_i, zero), seq), mx.minimum(
        mx.maximum(ends_i, zero), seq
    )


@lru_cache(maxsize=128)
def _path_c_prim_func(
    batch: int,
    seq_len: int,
    k: int,
    threads: int,
    score_dtype: str,
) -> Any:
    """Build and cache a shape-specialized TileLang topk selector kernel."""

    if threads < 1 or threads & (threads - 1):
        raise ValueError(f"topk selector merge requires power-of-two threads; got {threads}")
    if k < 1 or k > seq_len:
        raise ValueError(f"topk selector requires 1 <= k <= seq_len; got k={k}, seq_len={seq_len}")
    if threads * k * 8 > 32 * 1024:
        raise ValueError(
            f"topk selector shared merge buffer exceeds 32KiB: threads={threads}, k={k}"
        )

    ok, reason = _tilelang_available()
    if not ok:
        raise RuntimeError(reason)

    import tilelang.language as T  # type: ignore[reportMissingImports]

    globals().update(
        T=T,
        _TOPK_C_B=batch,
        _TOPK_C_N=seq_len,
        _TOPK_C_K=k,
        _TOPK_C_THREADS=threads,
        _TOPK_C_LOG_THREADS=threads.bit_length() - 1,
        _TOPK_C_SCORE_DTYPE=score_dtype,
    )

    @T.prim_func
    def topk_selector_kernel(
        ends: T.Tensor((_TOPK_C_B,), "int32"),
        indices: T.Tensor((_TOPK_C_B, _TOPK_C_K), "int32"),
        scores: T.Tensor((_TOPK_C_B, _TOPK_C_N), _TOPK_C_SCORE_DTYPE),
        starts: T.Tensor((_TOPK_C_B,), "int32"),
    ):
        with T.Kernel(_TOPK_C_B, threads=_TOPK_C_THREADS) as bx:
            lane = T.get_thread_binding()
            local_vals = T.alloc_local((_TOPK_C_K,), "float32")
            local_idx = T.alloc_local((_TOPK_C_K,), "int32")
            merged_vals = T.alloc_local((_TOPK_C_K,), "float32")
            merged_idx = T.alloc_local((_TOPK_C_K,), "int32")
            pair_vals = T.alloc_shared(
                (_TOPK_C_THREADS, _TOPK_C_K),
                "float32",
                scope="shared",
            )
            pair_idx = T.alloc_shared(
                (_TOPK_C_THREADS, _TOPK_C_K),
                "int32",
                scope="shared",
            )
            pos = T.alloc_var("int32")
            keep_scanning = T.alloc_var("int32")
            ap = T.alloc_var("int32")
            bp = T.alloc_var("int32")
            stride = T.alloc_var("int32")
            other = T.alloc_var("int32")
            a_val = T.alloc_var("float32")
            b_val = T.alloc_var("float32")
            a_idx = T.alloc_var("int32")
            b_idx = T.alloc_var("int32")
            s = starts[bx]
            e = ends[bx]

            for i in T.serial(_TOPK_C_K):
                local_vals[i] = T.float32(-1.0e38)
                local_idx[i] = -1

            for j in T.serial(lane, _TOPK_C_N, step=_TOPK_C_THREADS):
                valid = (j >= s) & (j < e)
                value = T.if_then_else(
                    valid,
                    T.cast(scores[bx, j], "float32"),
                    T.float32(-1.0e38),
                )
                if value > local_vals[0]:
                    pos = 0
                    keep_scanning = 1
                    # Insertion sort into the ascending top-K list. Do not use
                    # a TileLang ``break`` here: the tvm-ffi compile path can
                    # lower it as a break from the outer row scan after static
                    # loop simplification, which skips later candidates. The
                    # guard preserves the early-stop semantics without emitting
                    # a control-flow break.
                    for p in T.serial(1, _TOPK_C_K):
                        if keep_scanning != 0:
                            if value > local_vals[p]:
                                local_vals[p - 1] = local_vals[p]
                                local_idx[p - 1] = local_idx[p]
                                pos = p
                            else:
                                keep_scanning = 0
                    local_vals[pos] = value
                    local_idx[pos] = j

            for i in T.serial(_TOPK_C_K):
                pair_vals[lane, i] = local_vals[i]
                pair_idx[lane, i] = local_idx[i]
            T.sync_threads()

            for round_id in T.serial(_TOPK_C_LOG_THREADS):
                stride = T.shift_left(1, round_id)
                if lane % (stride * 2) == 0:
                    other = lane + stride
                    if other < _TOPK_C_THREADS:
                        ap = _TOPK_C_K - 1
                        bp = _TOPK_C_K - 1
                        for pick in T.serial(_TOPK_C_K):
                            a_val = T.float32(-1.0e38)
                            b_val = T.float32(-1.0e38)
                            a_idx = -1
                            b_idx = -1
                            if ap >= 0:
                                a_val = pair_vals[lane, ap]
                                a_idx = pair_idx[lane, ap]
                            if bp >= 0:
                                b_val = pair_vals[other, bp]
                                b_idx = pair_idx[other, bp]
                            if a_val >= b_val:
                                merged_vals[_TOPK_C_K - 1 - pick] = a_val
                                merged_idx[_TOPK_C_K - 1 - pick] = a_idx
                                ap -= 1
                            else:
                                merged_vals[_TOPK_C_K - 1 - pick] = b_val
                                merged_idx[_TOPK_C_K - 1 - pick] = b_idx
                                bp -= 1
                        for i in T.serial(_TOPK_C_K):
                            pair_vals[lane, i] = merged_vals[i]
                            pair_idx[lane, i] = merged_idx[i]
                T.sync_threads()

            if lane == 0:
                for i in T.serial(_TOPK_C_K):
                    indices[bx, i] = pair_idx[0, _TOPK_C_K - 1 - i]

    return topk_selector_kernel


@lru_cache(maxsize=128)
def _path_c_lowering_for(
    batch: int,
    seq_len: int,
    k: int,
    threads: int,
    score_dtype: str,
) -> _msl_transform.TileLangMSLLowering:
    topk_selector_kernel = _path_c_prim_func(batch, seq_len, k, threads, score_dtype)
    # Legacy debug-only no-out wrapper support. Production Path C dispatches
    # through ``_path_c_tvm_ffi_kernel_for`` below; this extracted-MSL lowering
    # exists only when ``CPPMEGA_TOPK_PATH_C_LEGACY_NO_OUT`` enables the old
    # MLX fast-kernel wrapper for diagnostics.
    artifact = dispatch_lower(
        topk_selector_kernel,
        target="metal",
        return_msl=True,
        pass_configs=_topk_path_c_pass_configs() or None,
    )

    return artifact


@lru_cache(maxsize=128)
def _path_c_tvm_ffi_kernel_for(
    batch: int,
    seq_len: int,
    k: int,
    threads: int,
    score_dtype: str,
) -> Any:
    """Build and cache the owner-output tvm-ffi Path C topk kernel."""

    import tilelang

    prim = _path_c_prim_func(batch, seq_len, k, threads, score_dtype)
    return tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=1,
        pass_configs=_topk_path_c_pass_configs() or None,
    )


@lru_cache(maxsize=128)
def _path_c_kernel_for(
    batch: int,
    seq_len: int,
    k: int,
    threads: int,
    score_dtype: str,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build and cache the legacy MLX fast-kernel Path C wrapper."""

    lowering = _path_c_lowering_for(batch, seq_len, k, threads, score_dtype)
    kernel = mx.fast.metal_kernel(
        name=f"cppmega_topk_selector_path_c_{batch}_{seq_len}_{k}_{threads}_{score_dtype}",
        input_names=["ends", "scores", "starts"],
        output_names=["indices"],
        source=lowering.body,
        header=lowering.header.rstrip() + "\n",
        ensure_row_contiguous=True,
    )
    return kernel, lowering


def _validate_path_c_owner_output(
    out: mx.array,
    *,
    batch: int,
    k: int,
) -> mx.array:
    if not isinstance(out, mx.array):
        raise TypeError(
            f"topk_selector_tilelang: out must be an mlx.core.array; "
            f"got {type(out).__name__}"
        )
    if out.shape != (batch, k):
        raise ValueError(
            f"topk_selector_tilelang: out shape must be ({batch}, {k}); "
            f"got {tuple(out.shape)}"
        )
    if out.dtype != mx.int32:
        raise ValueError(
            f"topk_selector_tilelang: out dtype must be mx.int32; got {out.dtype}"
        )
    return out


def topk_selector_tilelang_direct(
    scores: mx.array,
    k: int,
    *,
    out: mx.array,
    starts: mx.array | None = None,
    ends: mx.array | None = None,
) -> mx.array:
    """Run Path C through tvm-ffi into caller-owned ``out``."""

    if scores.ndim != 2:
        raise ValueError(
            f"topk_selector expects a (B, T) array; got shape {scores.shape}"
        )
    if starts is not None or ends is not None:
        raise TopKPathCDirectError(
            "topk_selector_tilelang direct tvm-ffi route is currently "
            "limited to unmasked rows; masked start/end intervals remain on "
            "Path B until the TileLang owner-output lowering is parity-clean"
        )
    batch = int(scores.shape[0])
    seq_len = int(scores.shape[1])
    if k < 1 or k > seq_len:
        raise ValueError(f"topk must be in [1, {seq_len}]; got {k}")
    if batch <= 0 or scores.size == 0:
        raise TopKPathCDirectError("empty topk inputs are not dispatchable")

    score_dtype = _path_c_score_dtype_direct(scores)
    if score_dtype is None:
        raise TopKPathCDirectError(
            f"topk_selector_tilelang direct tvm-ffi route supports "
            f"mx.float32/mx.float16 scores without hidden casts; got {scores.dtype}"
        )
    indices = _validate_path_c_owner_output(out, batch=batch, k=int(k))
    if not topk_selector_path_c_status().available:
        raise TopKPathCDirectError(topk_selector_path_c_status().reason)
    starts_buf, ends_buf = _path_c_interval_buffers(
        batch=batch,
        seq_len=seq_len,
        starts=starts,
        ends=ends,
    )
    try:
        threads = _path_c_threads_for(int(k))
        kernel = _path_c_tvm_ffi_kernel_for(
            batch,
            seq_len,
            int(k),
            threads,
            score_dtype,
        )
    except Exception as exc:
        raise TopKPathCDirectError(
            f"direct tvm-ffi topk compile failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        returned = kernel(ends_buf, indices, scores, starts_buf)
    except Exception as exc:
        try:
            from tilelang.contrib.mlx_interop import DLPackInteropError
        except Exception:  # pragma: no cover - only when TileLang import itself is broken
            DLPackInteropError = ()  # type: ignore[assignment]
        if isinstance(exc, DLPackInteropError):
            raise
        raise TopKPathCDirectError(
            f"direct tvm-ffi topk dispatch failed: {type(exc).__name__}: {exc}"
        ) from exc
    if returned is not indices:
        raise TopKPathCDirectError(
            "direct tvm-ffi topk did not return the caller-owned output"
        )
    # TVM encodes into MLX's current Metal command buffer outside the MLX graph.
    # The returned owner array has no lazy MLX dependency edge, so force the
    # command buffer to complete before Python or downstream consumers observe it.
    mx.synchronize()
    return indices


def topk_selector_tilelang(
    scores: mx.array,
    k: int,
    *,
    starts: mx.array | None = None,
    ends: mx.array | None = None,
    out: mx.array | None = None,
) -> mx.array | None:
    """TileLang DSL Path C forward.

    Returns ``None`` when TileLang/Metal cannot dispatch. Without ``out``,
    the legacy MLX fast-kernel wrapper is debug-only and disabled by default
    because it owns output allocation. Production callers that want Path C
    must pass a caller-owned ``out`` buffer and use the tvm-ffi route.
    """

    if scores.ndim != 2:
        raise ValueError(
            f"topk_selector expects a (B, T) array; got shape {scores.shape}"
        )
    batch = int(scores.shape[0])
    seq_len = int(scores.shape[1])
    if k < 1 or k > seq_len:
        raise ValueError(f"topk must be in [1, {seq_len}]; got {k}")
    if batch <= 0 or scores.size == 0:
        return None
    if not topk_selector_path_c_status().available:
        return None

    if out is not None:
        return topk_selector_tilelang_direct(
            scores,
            k,
            starts=starts,
            ends=ends,
            out=out,
        )
    if not _topk_legacy_no_out_enabled():
        return None

    dtype_pair = _path_c_score_dtype(scores)
    if dtype_pair is None:
        return None
    score_dtype, work_scores = dtype_pair
    starts_buf, ends_buf = _path_c_interval_buffers(
        batch=batch,
        seq_len=seq_len,
        starts=starts,
        ends=ends,
    )
    try:
        threads = _path_c_threads_for(int(k))
        kernel, lowering = _path_c_kernel_for(batch, seq_len, int(k), threads, score_dtype)
    except Exception:
        return None

    if lowering is None:
        # Engine path: ``kernel`` is a ``tilelang.compile`` artifact. Invoke it
        # directly via its standard ``__call__`` contract. Failures here surface
        # as ``None`` to keep the public selector contract stable.
        try:
            indices = mx.zeros((batch, int(k)), dtype=mx.int32)
            kernel(work_scores, starts_buf, ends_buf, indices)
            return indices
        except Exception:
            return None

    grid = (
        lowering.grid[0] * lowering.threadgroup[0],
        lowering.grid[1] * lowering.threadgroup[1],
        lowering.grid[2] * lowering.threadgroup[2],
    )
    try:
        outputs = kernel(
            inputs=[ends_buf, work_scores, starts_buf],
            output_shapes=[(batch, int(k))],
            output_dtypes=[mx.int32],
            grid=grid,
            threadgroup=lowering.threadgroup,
            stream=mx.gpu,
        )
    except Exception:
        return None
    return outputs[0]


# ---------------------------------------------------------------------------
# Path B status seam.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathBStatus:
    """Why legacy Path B direct-MSL is or isn't available."""

    available: bool
    reason: str


_PATH_B_OK_REASON = (
    "topk_selector legacy direct-MSL kernel built via mx.fast.metal_kernel is "
    "available for no-owner-output and masked calls. Native TileLang Path C "
    "uses tvm-ffi owner outputs for unmasked calls; this fallback remains "
    "until the masked/no-out contracts have a no-copy native route. The "
    "original CUDA-style TileLang schedule is still blocked on shared.dyn / "
    "LowerTileOp issues, see docs/tilelang_ports/topk_selector.md."
)
_PATH_B_BLOCKER_REASON = (
    "topk_selector direct-MSL kernel could not be constructed: "
    "mx.fast.metal_kernel is unavailable (no Metal backend on this device)."
)


def topk_selector_path_b_status() -> PathBStatus:
    """Return whether Path B is currently dispatchable.

    Path B is an explicit legacy fallback. It is available whenever Metal can
    build the hand-written MSL, but production TileLang Path C should use
    ``topk_selector_tilelang_direct(..., out=...)`` so tvm-ffi receives a
    caller-owned output. The remaining direct-MSL callers are no-output and
    masked selector contracts that cannot be moved without hidden output/mask
    staging.
    """

    if _TOPK_KERNEL is None or not _msl_transform.can_run_metal():
        return PathBStatus(available=False, reason=_PATH_B_BLOCKER_REASON)
    return PathBStatus(available=True, reason=_PATH_B_OK_REASON)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def topk_selector(
    scores: mx.array,
    k: int,
    *,
    starts: mx.array | None = None,
    ends: mx.array | None = None,
    backend: str = "auto",
) -> mx.array:
    """Top-k selector matching the cppmega source contract.

    Args:
        scores: (B, T) array of values.
        k: number of top entries to select.
        starts: optional (B,) int32 starts for per-row mask.
        ends: optional (B,) int32 ends for per-row mask.
        backend: ``"auto"`` (default) tries direct-MSL Path B before the
            pure-MLX reference; it does not route through Path C because this
            public entry point has no owner-output buffer;
            ``"metal"`` requires the Path B kernel and raises on fallback;
            ``"tilelang"`` / ``"path_c"`` require the owner-output Path C
            helper and therefore fail closed from this no-``out`` API;
            ``"mlx"`` always uses the reference.
    """

    if backend not in {"auto", "mlx", "metal", "tilelang", "path_c"}:
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "mlx":
        return topk_selector_reference(scores, k, starts=starts, ends=ends)
    if backend == "metal":
        out = topk_selector_metal(scores, k, starts=starts, ends=ends)
        if out is None:
            raise RuntimeError("topk_selector: direct-MSL Metal path unavailable")
        return out
    if backend in {"tilelang", "path_c"}:
        out = topk_selector_tilelang(scores, k, starts=starts, ends=ends)
        if out is None:
            raise RuntimeError("topk_selector: TileLang Path C path unavailable")
        return out
    # auto has no owner-output parameter, so it cannot honestly use the direct
    # Path C tvm-ffi route. Keep no-out AUTO on Path B or the pure-MLX reference.
    out = topk_selector_metal(scores, k, starts=starts, ends=ends)
    if out is not None:
        return out
    return topk_selector_reference(scores, k, starts=starts, ends=ends)


__all__ = [
    "PathBStatus",
    "PathCStatus",
    "TopKPathCDirectError",
    "topk_selector",
    "topk_selector_metal",
    "topk_selector_path_b_status",
    "topk_selector_path_c_status",
    "topk_selector_reference",
    "topk_selector_tilelang",
    "topk_selector_tilelang_direct",
]
