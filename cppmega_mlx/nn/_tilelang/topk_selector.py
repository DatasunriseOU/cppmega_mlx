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

The old Path B hand-written MSL bypass is retired. Path C keeps a real TileLang
DSL ``@T.prim_func`` and uses the same Metal-friendly one-threadgroup-per-row
algorithm instead of the upstream radix/histogram schedule. The production Path
C surface is the explicit tvm-ffi owner-output route
``topk_selector_tilelang_direct(..., out=...)``. It uses *static*
(compile-time-sized) ``threadgroup`` arrays - no ``shared.dyn`` scope, no
off-by-one fills.

Performance vs MLX's built-in ``argpartition``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MLX's ``mx.argpartition`` is itself an optimised Metal kernel. Unmasked AUTO
routes now use the TileLang/tvm-ffi owner-output path when it is available;
masked intervals fall back to the pure-MLX reference until the owner-output
Path C route grows masked-row parity. Explicit ``backend="metal"`` is retained
only as a fail-closed compatibility spelling for callers that still probe the
retired Path B surface.

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
# Retired direct-MSL Path B surface.
# ---------------------------------------------------------------------------


def topk_selector_metal(
    scores: mx.array,
    k: int,
    *,
    starts: mx.array | None = None,
    ends: mx.array | None = None,
) -> mx.array | None:
    """Retired direct-MSL Path B compatibility surface."""

    if scores.ndim != 2:
        raise ValueError(
            f"topk_selector expects a (B, T) array; got shape {scores.shape}"
        )
    seq_len = int(scores.shape[1])
    if k < 1 or k > seq_len:
        raise ValueError(f"topk must be in [1, {seq_len}]; got {k}")
    del starts, ends
    return None


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
) -> tuple[None, _msl_transform.TileLangMSLLowering]:
    """Return the debug lowering for the retired no-output Path C wrapper."""

    lowering = _path_c_lowering_for(batch, seq_len, k, threads, score_dtype)
    return None, lowering


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
    return None


# ---------------------------------------------------------------------------
# Path B status seam.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathBStatus:
    """Why legacy Path B direct-MSL is or isn't available."""

    available: bool
    reason: str


_PATH_B_RETIRED_REASON = (
    "topk_selector direct-MSL Path B is retired. Unmasked AUTO calls should use "
    "the TileLang/tvm-ffi owner-output route; masked or unsupported no-output "
    "calls fall back to the pure-MLX reference until masked owner-output Path C "
    "parity lands."
)


def topk_selector_path_b_status() -> PathBStatus:
    """Return whether Path B is currently dispatchable.

    Path B previously built hand-written MSL through ``mx.fast.metal_kernel``.
    That surface is retired so production code cannot silently re-enter the
    direct-MSL boundary.
    """

    return PathBStatus(available=False, reason=_PATH_B_RETIRED_REASON)


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
        backend: ``"auto"`` (default) tries TileLang Path C with an owner-output
            buffer for unmasked calls, then falls back to the pure-MLX reference;
            ``"metal"`` fails closed because Path B is retired;
            ``"tilelang"`` / ``"path_c"`` require the owner-output Path C
            helper and allocate the public result buffer for unmasked calls;
            ``"mlx"`` always uses the reference.
    """

    if backend not in {"auto", "mlx", "metal", "tilelang", "path_c"}:
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "mlx":
        return topk_selector_reference(scores, k, starts=starts, ends=ends)
    if backend == "metal":
        raise RuntimeError("topk_selector: direct-MSL Path B is retired")
    if backend in {"tilelang", "path_c"}:
        try:
            owner_out = mx.zeros((int(scores.shape[0]), int(k)), dtype=mx.int32)
            return topk_selector_tilelang_direct(scores, k, out=owner_out)
        except TopKPathCDirectError as exc:
            raise RuntimeError("topk_selector: TileLang Path C path unavailable") from exc
    if starts is None and ends is None:
        try:
            owner_out = mx.zeros((int(scores.shape[0]), int(k)), dtype=mx.int32)
            return topk_selector_tilelang_direct(scores, k, out=owner_out)
        except TopKPathCDirectError:
            pass
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
