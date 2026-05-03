"""Path B port of cppmega's TileLang topk-selector kernel.

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

The Path B pipeline lowers a TileLang ``PrimFunc`` to MSL via
``tilelang.engine.lower.lower(prim, target=Target("metal"))``. The
``topk_selector`` kernel **does not lower** on this stack.

Probed concretely with ``BLOCK_SIZE in {128, 256}`` and
``RADIX in {64, 128, 256}``, ``SMEM_INPUT_SIZE=512``:

  * Whenever the histogram size (RADIX+1) is not a multiple of BLOCK_SIZE,
    TileLang's ``LowerTileOp`` raises::

        InternalError: Loop layout is not injective:
        Fragment([257] -> [2], replicate: 1, thread: 256, ...,
                 forward_thread: _i % 256, forward_index: [_i // 256], ...)
            errors: ["The iterations do not traverse full iter space",
                     "Index mapping does not form a bijective transform."]
            loop AST: for i in T.parallel(257):
                s_histogram[i] = 0

    The auto-vectorizer cannot tile the off-by-one fill, even though the
    underlying ``T.fill`` is trivial.

  * With matching sizes (e.g. BLOCK_SIZE=256, RADIX=128), TileLang gets past
    the layout check but the metal codegen rejects the storage scope::

        Fatal: Unknown storage scope `shared.dyn`

    TileLang allocates threadgroup memory under ``shared.dyn`` (dynamic
    shared scope), and TVM's metal backend in 0.1.9 does not implement it.
    Any ``T.alloc_shared`` is therefore unreachable on Apple. We confirmed
    this with a tiny shared-memory copy as well.

A minimal probe is::

    @T.prim_func
    def use_shared(A: T.Tensor((256,), 'float32'),
                   B: T.Tensor((256,), 'float32')):
        with T.Kernel(1, threads=256) as bx:
            tx = T.get_thread_binding()
            s = T.alloc_shared([256], 'float32')
            s[tx] = A[tx]; T.sync_threads(); B[tx] = s[tx] * 2.0

    target = tvm.target.Target('metal')
    with target: lower(use_shared, target=target)
    # -> Fatal: Unknown storage scope `shared.dyn`

This is a hard codegen blocker for *every* TileLang kernel that uses
threadgroup memory or partial barriers, which includes the entire DSA
sparse-MLA family the topk-selector belongs to. There is no Path B
forward path until either (a) tilelang adds ``shared.dyn`` support to its
metal backend, or (b) we hand-write the MSL ourselves bypassing TileLang.

What this module provides today
-------------------------------

* :func:`topk_selector` -- a pure-MLX forward that matches the source
  contract: returns the indices of the ``topk`` largest values per batch
  row, optionally restricted to ``[starts, ends)``. This is the runtime
  path on Apple Silicon.
* :func:`topk_selector_reference` -- the explicit reference, used by tests.
* :func:`topk_selector_path_b_status` -- introspection of why Path B is
  unavailable, mirroring the same pattern used by the other ports in this
  package.

The function is non-differentiable (top-k indices), so we do not need a
:class:`mx.custom_function` wrapper. Top-k is a discrete selector; tests
treat parity as the indices' set-membership for the largest k values.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


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
    if starts is not None or ends is not None:
        batch = int(scores.shape[0])
        if starts is None:
            starts = mx.zeros((batch,), dtype=mx.int32)
        if ends is None:
            ends = mx.full((batch,), seq_len, dtype=mx.int32)
        if starts.shape != (batch,) or ends.shape != (batch,):
            raise ValueError(
                "starts/ends must have shape (B,); "
                f"got starts={starts.shape}, ends={ends.shape}"
            )
        col = mx.arange(seq_len, dtype=mx.int32)[None, :]
        valid = (col >= starts[:, None]) & (col < ends[:, None])
        # Use the smallest representable value of `scores.dtype` for masking.
        if scores.dtype == mx.float32 or scores.dtype == mx.float16 or scores.dtype == mx.bfloat16:
            mask_value = mx.array(float("-inf"), dtype=scores.dtype)
        else:
            mask_value = mx.array(-2**31, dtype=scores.dtype)
        masked = mx.where(valid, scores, mask_value)

    # argpartition picks the indices of the `k` largest values (when sorting
    # the *negated* scores ascending, the first k elements are those with
    # the largest original scores). MLX guarantees the partition contract;
    # ordering within the slice is implementation-defined.
    if k == seq_len:
        # argpartition with kth == size is undefined for some backends; just
        # take an arange.
        order = mx.broadcast_to(
            mx.arange(seq_len, dtype=mx.int32)[None, :],
            scores.shape,
        )
        return order
    part = mx.argpartition(-masked, kth=k, axis=-1)[..., :k]
    return part.astype(mx.int32)


# ---------------------------------------------------------------------------
# Path B status seam.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathBStatus:
    """Why Path B (TileLang->MSL->MLX) is or isn't available."""

    available: bool
    reason: str


_PATH_B_BLOCKER_REASON = (
    "topk_selector cannot be lowered through TileLang 0.1.9's metal target: "
    "(a) the histogram fill `T.parallel(257)` over 256 threads fails "
    "LowerTileOp's injective-layout check, and (b) any T.alloc_shared lowers "
    "to storage scope `shared.dyn`, which TVM's metal codegen rejects with "
    "`Fatal: Unknown storage scope shared.dyn`. See "
    "docs/tilelang_ports/topk_selector.md for the probe transcript."
)


def topk_selector_path_b_status() -> PathBStatus:
    """Return the persistent reason Path B is unavailable for this kernel."""

    return PathBStatus(available=False, reason=_PATH_B_BLOCKER_REASON)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def topk_selector(
    scores: mx.array,
    k: int,
    *,
    starts: mx.array | None = None,
    ends: mx.array | None = None,
) -> mx.array:
    """Top-k selector matching the cppmega source contract.

    On Apple Silicon, Path B (TileLang->MSL->MLX) is unavailable for this
    kernel; see :func:`topk_selector_path_b_status` for the reason. We
    therefore always run the pure-MLX reference. The signature is identical
    to what a future Metal-native implementation would expose, so callers can
    swap implementations transparently.
    """

    return topk_selector_reference(scores, k, starts=starts, ends=ends)


__all__ = [
    "PathBStatus",
    "topk_selector",
    "topk_selector_path_b_status",
    "topk_selector_reference",
]
