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
The v32 fwd/bwd tiles request >48 KB dynamic shared memory. On gb10/sm_121 the
TileLang/TVM runtime's ``cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)``
opt-in is rejected by the driver above ~48 KB even though the device reports a
99 KB opt-in carve-out, so the fused kernel does not yet run there (it lowers +
compiles for sm_121a; only the runtime smem opt-in is blocked). On H100 (sm_90)
/ B200 (sm_100) the carve-out fits and the fused path runs. This is surfaced
loudly when forced, never silently degraded.
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
    """Materialize an MLX array as a contiguous CUDA torch tensor (no host copy
    when a CUDA->CUDA DLPack bridge is available; otherwise via DLPack)."""

    torch = _torch()
    from cppmega_mlx.nn._tilelang._cuda_eager import _mlx_to_torch_cuda as _bridge

    t = _bridge(arr)
    if not t.is_cuda:
        raise RuntimeError(
            "_sparse_mla_v32_fused: MLX->torch bridge did not yield a CUDA tensor; "
            "the fused v32 Sparse-MLA path requires CUDA buffers."
        )
    return t.contiguous()


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
    dkv = _torch_cuda_to_mlx(dkv_t.astype(torch.bfloat16), mx.bfloat16)
    return dq, dkv
