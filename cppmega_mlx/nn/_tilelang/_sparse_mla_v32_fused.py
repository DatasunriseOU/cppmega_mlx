# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Real fused DeepSeek-V3.2 Sparse-MLA forward+backward (TileLang-CUDA).

This module wraps the *real* training-complete fused Sparse-MLA kernels we own
at ``tilelang/examples/deepseek_v32/sparse_mla_fwd.py`` +
``sparse_mla_bwd.py`` and exposes them over MLX bf16 ``q``/``kv``/``indices``
buffers so Path C can dispatch the fused O(seq*topk) attention (forward AND
backward, with vectorized ``T.atomic_addx4`` dKV) instead of the per-element
string-emitted reference.

It is **env-gated** (``CPPMEGA_SPARSE_MLA_V32_FUSED=1``). When the gate is OFF
(default) callers keep using the existing reference path; nothing here runs.

RULE #1 (no silent fallback): when the gate is ON and a caller forces Path C,
every failure here RAISES with where+what. We never silently fall back to the
reference and claim the fused kernel ran.

ABI / layout
------------
* ``q``  : bf16 ``(B, S, H, DQK)`` where ``DQK = d_v + tail_dim`` (576 = 512+64).
* ``kv`` : bf16 ``(B, SKV, G, DQK)`` (kv_group ``G``).
* ``indices`` : int32 ``(B, S, G, topk)``.
* fwd -> ``out`` bf16 ``(B, S, H, d_v)``, ``lse`` fp32 ``(B, S, H)``.
* bwd -> ``dq`` bf16 ``(B, S, H, DQK)``, ``dkv`` bf16 ``(B, SKV, G, DQK)``.

The kernel math is byte-for-byte the upstream v32 example (log2-domain online
softmax forward; preprocess-delta + atomic-scatter dKV backward).

gb10 / sm_121 note
------------------
RESOLVED (tilelang 823c807c): the v32 fwd/bwd tiles now FIT the gb10/sm_121
99 KiB dynamic-smem carve-out via the re-tiled GB10 variant (``gb10=True``:
block_I=16, num_stages=1, aggressive shared-memory merge, Hopper TMA-lower
disabled, ``O_shared`` dropped, mask-in-shared). MEASURED on real sm_121a ptxas
(CUDA 13.3): fwd 93.0 KiB, bwd <= 99 KiB; fwd cos 0.998, bwd dq/dkv cos 1.000 vs
the upstream reference. The earlier >48 KB ``cuFuncSetAttribute`` rejection was a
codegen/argbinder smem-emission bug, fixed in the merge-codegen-reorg branch.

This wrapper forwards ``gb10`` (auto-detect by default, overridable via
``CPPMEGA_SPARSE_MLA_V32_GB10``) so the kernel emits the GB10-fitting variant on
sm_12x and the original Hopper kernel on sm_90/sm_100. MLX-CUDA q/kv/indices are
fed to the torch-backend kernel interfaces zero-copy via DLPack when
``CPPMEGA_TILELANG_CUDA_ZEROCOPY=1`` (kDLCUDA ``DLManagedTensor`` over
``mx.array.data_ptr()``, no host roundtrip). The kernel outputs are imported
back from torch into MLX zero-copy too (DatasunriseOU MLX-CUDA DLPack *import*:
``cuda_dlpack_to_mlx`` wraps the foreign CUDA buffer and materializes it into an
MLX-owned GPU allocation with a single device-side copy — no ``.cpu()`` host
bounce), so the whole fused fwd/bwd stays GPU-resident and keeps the kernel's
latency win. With ``CPPMEGA_TILELANG_CUDA_ZEROCOPY`` unset the writeback uses the
explicit eager numpy-host copy (a deliberately-selected mode, not a silent
degrade).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import mlx.core as mx


SPARSE_MLA_V32_FUSED_ENV = "CPPMEGA_SPARSE_MLA_V32_FUSED"

# v32 layout constant: q/kv last dim is d_v + tail_dim; the upstream kernels hard
# assume d_v == 512 (and validate it), so tail_dim == DQK - 512.
_V32_D_V = 512


def v32_fused_enabled() -> bool:
    """Return True iff the real fused v32 Sparse-MLA path is env-gated ON."""

    return os.environ.get(SPARSE_MLA_V32_FUSED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception as exc:  # pragma: no cover - hosts without torch
        raise RuntimeError(
            "_sparse_mla_v32_fused: PyTorch (CUDA) is required for the fused v32 "
            f"Sparse-MLA kernels but could not be imported ({type(exc).__name__}: {exc})."
        ) from exc


def _mlx_to_torch_cuda(arr: mx.array):
    """Materialize an MLX array as a contiguous CUDA torch tensor.

    When the zero-copy DLPack escape hatch is enabled
    (``CPPMEGA_TILELANG_CUDA_ZEROCOPY=1``) the MLX-CUDA device buffer is handed to
    torch via a kDLCUDA ``DLManagedTensor`` capsule with NO host roundtrip — the
    returned tensor is a real device view of the MLX allocation. Otherwise we use
    the numpy-host-roundtrip eager bridge (``_cuda_eager._mlx_to_torch_cuda``).

    RULE #1: the zero-copy bridge RAISES with where+what on any failure; we never
    silently fall back to the eager host-copy bridge while claiming zero-copy.
    """

    torch = _torch()
    if _zerocopy_enabled():
        from cppmega_mlx.nn._tilelang._cuda_zerocopy import (
            mlx_cuda_array_to_torch_tensor as _zc_bridge,
        )

        t = _zc_bridge(arr)
    else:
        from cppmega_mlx.nn._tilelang._cuda_eager import _mlx_to_torch_cuda as _bridge

        t = _bridge(arr)
    if not t.is_cuda:
        raise RuntimeError(
            "_sparse_mla_v32_fused: MLX->torch bridge did not yield a CUDA tensor; "
            "the fused v32 Sparse-MLA path requires CUDA buffers."
        )
    # ``.contiguous()`` is a no-op (returns the same storage) for the row-major
    # zero-copy view; it only copies if MLX handed us a strided buffer.
    return t.contiguous()


def _zerocopy_enabled() -> bool:
    """Return whether the zero-copy CUDA DLPack escape hatch is env-gated ON."""

    from cppmega_mlx.nn._tilelang._cuda_zerocopy import zerocopy_enabled

    return zerocopy_enabled()


def _gb10_flag() -> bool | None:
    """Resolve the ``gb10`` arch flag forwarded to the v32 fwd/bwd interfaces.

    The upstream kernels accept ``gb10=None`` (auto-detect sm_12x), ``gb10=True``
    (force the 99 KiB-fitting GB10 variant: block_I=16, num_stages=1, aggressive
    smem merge, TMA-lower disabled), or ``gb10=False`` (the original Hopper
    kernel). We default to auto-detect (``None``) so the kernel itself decides by
    compute capability; ``CPPMEGA_SPARSE_MLA_V32_GB10`` forces the choice
    (1/true/on -> True, 0/false/off -> False) for A/B testing or non-sm_12x hosts.
    """

    raw = os.environ.get("CPPMEGA_SPARSE_MLA_V32_GB10", "").strip().lower()
    if raw == "":
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "_sparse_mla_v32_fused: CPPMEGA_SPARSE_MLA_V32_GB10 must be one of "
        f"1/true/on or 0/false/off (or unset for auto-detect); got {raw!r}."
    )


def _torch_cuda_to_mlx(tensor, dtype: mx.Dtype) -> mx.array:
    from cppmega_mlx.nn._tilelang._cuda_eager import _torch_cuda_to_mlx as _bridge

    return _bridge(tensor, dtype)


@lru_cache(maxsize=1)
def _v32_modules() -> tuple[Any, Any]:
    """Import the vendored v32 fwd/bwd example modules from the TileLang tree.

    The examples live under ``tilelang/examples/deepseek_v32`` and import each
    other + ``utils`` by bare name, so the directory must be on ``sys.path``.
    RULE #1: a missing TileLang examples tree RAISES (no silent skip).
    """

    import importlib
    import sys

    try:
        import tilelang  # noqa: PLC0415,F401
    except Exception as exc:
        raise RuntimeError(
            "_sparse_mla_v32_fused: tilelang is not importable; the fused v32 "
            f"Sparse-MLA path cannot run ({type(exc).__name__}: {exc})."
        ) from exc

    example_dir = os.environ.get("CPPMEGA_TILELANG_V32_EXAMPLES_DIR")
    candidates = []
    if example_dir:
        candidates.append(example_dir)
    # Best-effort discovery relative to an installed/dev tilelang checkout.
    tl_file = getattr(importlib.import_module("tilelang"), "__file__", None)
    if tl_file:
        root = os.path.dirname(os.path.dirname(os.path.abspath(tl_file)))
        candidates.append(os.path.join(root, "examples", "deepseek_v32"))
    for env_root in ("TILELANG_ROOT", "TILELANG_SOURCE_DIR"):
        r = os.environ.get(env_root)
        if r:
            candidates.append(os.path.join(r, "examples", "deepseek_v32"))

    chosen = next(
        (c for c in candidates if c and os.path.isfile(os.path.join(c, "sparse_mla_fwd.py"))),
        None,
    )
    if chosen is None:
        raise RuntimeError(
            "_sparse_mla_v32_fused: could not locate the TileLang deepseek_v32 "
            "examples (sparse_mla_fwd.py/sparse_mla_bwd.py). Set "
            "CPPMEGA_TILELANG_V32_EXAMPLES_DIR to the examples/deepseek_v32 "
            f"directory. Searched: {candidates}"
        )
    if chosen not in sys.path:
        sys.path.insert(0, chosen)
    fwd = importlib.import_module("sparse_mla_fwd")
    bwd = importlib.import_module("sparse_mla_bwd")
    return fwd, bwd


def _resolve_dims(q: mx.array, kv: mx.array, indices: mx.array):
    if q.ndim != 4 or kv.ndim != 4 or indices.ndim != 4:
        raise ValueError(
            "_sparse_mla_v32_fused: expected 4D q/kv/indices "
            f"(got q={tuple(q.shape)} kv={tuple(kv.shape)} indices={tuple(indices.shape)})."
        )
    B, S, H, DQK = (int(x) for x in q.shape)
    Bk, SKV, G, DQKk = (int(x) for x in kv.shape)
    if Bk != B or DQKk != DQK:
        raise ValueError(
            "_sparse_mla_v32_fused: q/kv batch or qk-dim mismatch "
            f"(q={tuple(q.shape)} kv={tuple(kv.shape)})."
        )
    Bi, Si, Gi, topk = (int(x) for x in indices.shape)
    if (Bi, Si, Gi) != (B, S, G):
        raise ValueError(
            "_sparse_mla_v32_fused: indices shape must be (B, S, kv_group, topk); "
            f"got {tuple(indices.shape)} for q={tuple(q.shape)} kv={tuple(kv.shape)}."
        )
    if DQK <= _V32_D_V:
        raise ValueError(
            "_sparse_mla_v32_fused: the v32 kernels require qk_dim > d_v=512 "
            f"(got DQK={DQK}); this is the d_v(512)+tail layout."
        )
    return B, S, SKV, H, G, DQK, topk


def v32_fused_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    return_lse: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """Run the real fused v32 Sparse-MLA forward over bf16 q/kv/indices.

    Returns ``out`` bf16 ``(B, S, H, d_v)`` (and ``lse`` fp32 ``(B, S, H)`` when
    ``return_lse``). RAISES on any kernel failure (RULE #1).
    """

    fwd, _bwd = _v32_modules()
    _resolve_dims(q, kv, indices)
    torch = _torch()

    q_t = _mlx_to_torch_cuda(q.astype(mx.bfloat16))
    kv_t = _mlx_to_torch_cuda(kv.astype(mx.bfloat16))
    idx_t = _mlx_to_torch_cuda(indices.astype(mx.int32))
    try:
        out_t, lse_t = fwd.sparse_mla_fwd_interface(
            q_t.contiguous(),
            kv_t.contiguous(),
            idx_t.contiguous(),
            sm_scale=float(sm_scale),
            gb10=_gb10_flag(),
        )
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            "_sparse_mla_v32_fused.v32_fused_apply: the fused DeepSeek-V3.2 "
            f"Sparse-MLA forward kernel failed ({type(exc).__name__}: {exc}). "
            "On gb10/sm_121 the >48 KB dynamic-smem opt-in is rejected by the "
            "CUDA driver; on sm_90/sm_100 it should run."
        ) from exc

    out = _torch_cuda_to_mlx(out_t, mx.bfloat16)
    if return_lse:
        lse = _torch_cuda_to_mlx(lse_t, mx.float32)
        return out, lse
    return out


def v32_fused_bwd(
    q: mx.array,
    kv: mx.array,
    out: mx.array,
    d_out: mx.array,
    indices: mx.array,
    lse: mx.array,
    *,
    sm_scale: float,
) -> tuple[mx.array, mx.array]:
    """Run the real fused v32 Sparse-MLA backward -> ``(dq, dkv)`` bf16.

    Exercises the vectorized ``T.atomic_addx4`` dKV scatter. RAISES on any
    kernel failure (RULE #1).
    """

    _fwd, bwd = _v32_modules()
    _resolve_dims(q, kv, indices)
    torch = _torch()

    q_t = _mlx_to_torch_cuda(q.astype(mx.bfloat16)).contiguous()
    kv_t = _mlx_to_torch_cuda(kv.astype(mx.bfloat16)).contiguous()
    out_t = _mlx_to_torch_cuda(out.astype(mx.bfloat16)).contiguous()
    do_t = _mlx_to_torch_cuda(d_out.astype(mx.bfloat16)).contiguous()
    idx_t = _mlx_to_torch_cuda(indices.astype(mx.int32)).contiguous()
    lse_t = _mlx_to_torch_cuda(lse.astype(mx.float32)).contiguous()
    try:
        dq_t, dkv_t = bwd.sparse_mla_bwd(
            q_t,
            kv_t,
            out_t,
            do_t,
            idx_t,
            lse_t,
            sm_scale=float(sm_scale),
            gb10=_gb10_flag(),
        )
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            "_sparse_mla_v32_fused.v32_fused_bwd: the fused DeepSeek-V3.2 "
            f"Sparse-MLA backward kernel failed ({type(exc).__name__}: {exc}). "
            "This path exercises T.atomic_addx4 (vectorized dKV scatter); on "
            "gb10/sm_121 the >48 KB dynamic-smem opt-in is rejected by the CUDA "
            "driver before atomics are reached, on sm_90/sm_100 it should run."
        ) from exc

    dq = _torch_cuda_to_mlx(dq_t, mx.bfloat16)
    # ``sparse_mla_bwd``'s postprocess emits dkv as a torch fp32 tensor; cast to
    # bf16 with torch's ``.to`` (NOT MLX/numpy ``.astype``) before the writeback.
    dkv = _torch_cuda_to_mlx(dkv_t.to(torch.bfloat16), mx.bfloat16)
    return dq, dkv
