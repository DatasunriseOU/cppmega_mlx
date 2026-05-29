"""CUDA per-op EAGER kernel path for cppmega's TileLang Path-C ops.

The historical per-op EAGER kernels (Path C) lower their TileLang
``@T.prim_func`` to **Apple MSL** via ``mx.fast.metal_kernel`` / the tvm-ffi
Metal boundary, and every ``*_path_c_status()`` gate fails closed when
``mx.metal.is_available()`` is False. On a CUDA host (e.g. gb10 / sm_121) MLX
reports ``mx.default_device() == mx.gpu`` but ``mx.metal.is_available()`` is
False, so the Metal-only EAGER path raises "MLX Metal backend is not
available" and training cannot use a fused per-op kernel.

This module adds a **CUDA branch** for that EAGER path. It reuses the existing
in-tree static-shape TileLang prim_funcs and compiles them with
``target="cuda"`` instead of metal:

* **sparse_mla** — reuses :func:`...sparse_mla_path_c._make_sparse_mla_fwd_prim`
  verbatim (it is a scalar / shared-memory softmax kernel with no ``T.gemm``,
  no ``T.dynamic`` and no Metal intrinsics, so it lowers cleanly to CUDA).
* **mamba3 (fwd)** — the production Metal prim_func uses the Metal-only
  ``tir.metal.thread_position_in_grid_x`` intrinsic *and* ``T.alloc_var(init=)``
  accumulators (both reject CUDA codegen here), so we vendor a CUDA-safe copy
  of the *same recurrence math* using ``T.get_thread_binding()`` and
  ``T.alloc_local`` accumulators. Backward is left to the pure-MLX reference
  VJP (documented TODO to port the Metal SIMD lane-grad bwd).
* **m2rnn** — not yet ported; see :func:`m2rnn_supported_cuda_eager` (TODO).

MLX on this host has no CUDA-array DLPack export
(``a.__dlpack__`` raises "CUDA DLPack export is not supported"), so the
MLX<->kernel boundary uses a numpy host roundtrip into torch CUDA tensors.
That is correct (if not zero-copy) for an EAGER fallback whose job is to run a
real TileLang-CUDA kernel rather than raise. The Metal path is untouched: every
public entry below is only invoked from a ``cuda_eager_available()`` branch the
callers add *after* the existing ``can_run_metal()`` checks.
"""

# pyright: reportMissingImports=false, reportInvalidTypeForm=false
#
# NOTE: deliberately *no* ``from __future__ import annotations`` here. TileLang's
# ``@T.prim_func`` eager builder re-evaluates each parameter annotation
# (``T.Tensor((BATCH, ...), ...)``) via ``get_type_hints``. With PEP 563
# stringized annotations the shape names (BATCH/SEQ/...) would only be looked up
# in module globals and fail (they are closure locals). Keeping annotations
# eager lets the builder evaluate them with the enclosing scope intact, matching
# the in-tree prim builders in ``sparse_mla_path_c.py`` / ``mamba3_path_c.py``.

import functools
from typing import Any

import mlx.core as mx


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def cuda_eager_available() -> tuple[bool, str]:
    """Return ``(ok, reason)`` for the TileLang-CUDA EAGER path on this host.

    Available when Metal is unavailable (so we are *not* on Apple), torch with
    CUDA is importable, and TileLang imports. We intentionally do **not** gate
    on ``mx.metal`` being present — the whole point is the non-Metal host.
    """

    metal = getattr(mx, "metal", None)
    if metal is not None and metal.is_available():
        # Apple host with a working Metal backend: stay on the MSL path.
        return False, "Metal backend is available; use the MSL EAGER path"
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - torch always present on gb10
        return False, f"torch import failed: {exc}"
    if not torch.cuda.is_available():
        return False, "torch.cuda is not available"
    try:
        import tilelang  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return False, f"tilelang import failed: {exc}"
    return True, "TileLang-CUDA EAGER path ready"


# ---------------------------------------------------------------------------
# MLX <-> torch.cuda bridge (numpy host roundtrip)
# ---------------------------------------------------------------------------


_MLX_TO_TORCH_DTYPE: dict[Any, Any] = {}
_NP_DTYPE_FOR_MLX: dict[Any, str] = {}


def _init_dtype_maps() -> None:
    import numpy as np
    import torch

    if _MLX_TO_TORCH_DTYPE:
        return
    _MLX_TO_TORCH_DTYPE.update(
        {
            mx.float32: torch.float32,
            mx.float16: torch.float16,
            mx.bfloat16: torch.bfloat16,
            mx.int32: torch.int32,
            mx.int64: torch.int64,
        }
    )
    _NP_DTYPE_FOR_MLX.update(
        {
            mx.float32: "float32",
            mx.float16: "float16",
            mx.bfloat16: "float32",  # numpy has no bf16; carry through fp32
            mx.int32: "int32",
            mx.int64: "int64",
        }
    )
    del np


def _mlx_to_torch_cuda(a: mx.array) -> Any:
    """Copy an ``mx.array`` to a contiguous CUDA torch tensor (host roundtrip)."""

    import numpy as np
    import torch

    _init_dtype_maps()
    torch_dtype = _MLX_TO_TORCH_DTYPE.get(a.dtype)
    if torch_dtype is None:
        raise TypeError(f"_cuda_eager: unsupported mlx dtype {a.dtype}")
    np_dtype = _NP_DTYPE_FOR_MLX[a.dtype]
    if a.dtype == mx.bfloat16:
        host = np.array(a.astype(mx.float32))
    else:
        host = np.array(a)
    host = np.ascontiguousarray(host.astype(np_dtype))
    t = torch.from_numpy(host).to(device="cuda")
    if t.dtype != torch_dtype:
        t = t.to(torch_dtype)
    return t.contiguous()


def _torch_cuda_to_mlx(t: Any, out_dtype: Any) -> mx.array:
    """Copy a CUDA torch tensor back into an ``mx.array`` of ``out_dtype``."""

    import numpy as np  # noqa: F401
    import torch

    cpu = t.detach().to(device="cpu")
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.to(torch.float32)
    arr = mx.array(cpu.numpy())
    if arr.dtype != out_dtype:
        arr = arr.astype(out_dtype)
    return arr


# ---------------------------------------------------------------------------
# sparse_mla — reuse the in-tree static-shape fwd prim_func, target="cuda"
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def _sparse_mla_fwd_cuda_kernel(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
) -> Any:
    """Compile the reused Sparse-MLA fwd prim_func for CUDA (caller-owned out)."""

    import tilelang

    # Reuse the exact static-shape prim_func used by the Metal owner-output
    # route. It is target-agnostic TileLang DSL (no T.gemm / no T.dynamic / no
    # Metal intrinsics), so the only change is target="cuda".
    from cppmega_mlx.nn._tilelang.sparse_mla_path_c import _make_sparse_mla_fwd_prim

    prim = _make_sparse_mla_fwd_prim(
        BATCH,
        SEQ_LEN,
        HEADS,
        QK_DIM,
        KV_GROUP,
        HEAD_KV,
        TOPK,
        SEQ_LEN_KV,
        D_V,
        THREADS,
    )
    # out_idx=None -> all six buffers (q, kv, indices, sm_scale_buf, out, lse)
    # are positional; caller owns ``out`` and ``lse``.
    return tilelang.compile(prim, target="cuda", out_idx=None)


def sparse_mla_fwd_cuda_eager(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> tuple[mx.array, mx.array] | None:
    """TileLang-CUDA EAGER Sparse-MLA forward → ``(out, lse)`` or ``None``.

    Mirrors :func:`sparse_mla_fwd_path_c`'s contract: fp16 carrier q/kv,
    int32 indices (sentinel -1), fp16 ``out[B,S,H,d_v]`` and fp32
    ``lse[B,S,H]``. Returns ``None`` on any failure so the caller falls back to
    the pure-MLX reference.
    """

    ok, _reason = cuda_eager_available()
    if not ok:
        return None

    from cppmega_mlx.nn.sparse_mla import _resolve_shapes
    from cppmega_mlx.nn._tilelang.sparse_mla_path_c import _threadgroup_size

    try:
        import torch
    except Exception:
        return None

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    sm_scale_value = shapes.qk_dim ** -0.5 if sm_scale is None else float(sm_scale)

    q16 = q.astype(mx.float16)
    kv16 = kv.astype(mx.float16)
    idx32 = indices.astype(mx.int32)

    try:
        kernel = _sparse_mla_fwd_cuda_kernel(
            shapes.batch,
            shapes.seq_len,
            shapes.heads,
            shapes.qk_dim,
            shapes.kv_group,
            shapes.head_kv,
            shapes.topk,
            shapes.seq_len_kv,
            shapes.d_v,
            _threadgroup_size(shapes.topk),
        )
        # The prim_func declares 1-D flat buffers; pass flattened CUDA tensors.
        q_t = _mlx_to_torch_cuda(q16).reshape(-1)
        kv_t = _mlx_to_torch_cuda(kv16).reshape(-1)
        idx_t = _mlx_to_torch_cuda(idx32).reshape(-1)
        sc_t = torch.tensor([sm_scale_value], dtype=torch.float32, device="cuda")
        out_t = torch.zeros(
            shapes.batch * shapes.seq_len * shapes.heads * shapes.d_v,
            dtype=torch.float16,
            device="cuda",
        )
        lse_t = torch.zeros(
            shapes.batch * shapes.seq_len * shapes.heads,
            dtype=torch.float32,
            device="cuda",
        )
        kernel(q_t, kv_t, idx_t, sc_t, out_t, lse_t)
        torch.cuda.synchronize()
    except Exception:
        return None

    out = _torch_cuda_to_mlx(
        out_t.reshape(shapes.batch, shapes.seq_len, shapes.heads, shapes.d_v),
        mx.float16,
    )
    lse = _torch_cuda_to_mlx(
        lse_t.reshape(shapes.batch, shapes.seq_len, shapes.heads),
        mx.float32,
    )
    return out, lse


# ---------------------------------------------------------------------------
# mamba3 (fwd) — vendored CUDA-safe copy of the production recurrence
# ---------------------------------------------------------------------------


def _make_mamba3_fwd_cuda_prim(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
) -> Any:
    """Build a CUDA-safe Mamba3 MIMO forward (y, h_last) prim_func.

    Math is identical to ``mamba3_path_c._make_fwd_prim_func`` (per (b,h,p) lane
    sequential state scan: ``h = exp(A*dt)*h + x*B``; ``y = z*sigmoid(z)*(C·h +
    D*x)``), but adapted for CUDA codegen:

    * lane index from ``T.get_thread_binding()`` + block id, not the Metal
      ``tir.metal.thread_position_in_grid_x`` intrinsic;
    * ``T.alloc_local`` accumulators instead of ``T.alloc_var(init=)`` (the
      latter emits a non-lvalue ``float ya = 0x0p+0f`` that nvcc rejects here).

    Defined at module scope (not nested) so TileLang's ``@T.prim_func``
    ``get_type_hints`` re-evaluation of the ``T.Tensor((BATCH, ...))``
    annotations resolves the shape names from this module's globals.
    All carriers fp32 (the production fp32 EAGER carrier path).
    """

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = min(256, max(1, LANES))
    ad = "float32"

    @T.prim_func
    def fwd(
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), "float32"),
        C: T.Tensor((BATCH, SEQ, HEADS, STATE), "float32"),
        z: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        A: T.Tensor((BATCH, SEQ, HEADS), "float32"),
        dt: T.Tensor((BATCH, SEQ, HEADS), "float32"),
        D: T.Tensor((HEADS,), "float32"),
        h0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), "float32"),
        y: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        h_last: T.Tensor((BATCH, HEADS, HEADDIM, STATE), "float32"),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as bx:
            lane = T.get_thread_binding()
            gl = bx * THREADS + lane
            h_state = T.alloc_local((STATE,), ad)
            y_acc = T.alloc_local((1,), ad)
            decay = T.alloc_local((1,), ad)
            if gl < LANES:
                p = gl % HEADDIM
                h = (gl // HEADDIM) % HEADS
                b = gl // (HEADDIM * HEADS)
                D_h = D[h]
                for n in T.serial(STATE):
                    h_state[n] = h0[b, h, p, n]
                for t in T.serial(SEQ):
                    decay[0] = T.exp(A[b, t, h] * dt[b, t, h])
                    x_val = x[b, t, h, p]
                    z_val = z[b, t, h, p]
                    y_acc[0] = 0.0
                    for n in T.serial(STATE):
                        new_h = decay[0] * h_state[n] + x_val * B[b, t, h, n]
                        h_state[n] = new_h
                        y_acc[0] = y_acc[0] + new_h * C[b, t, h, n]
                    y_skipped = y_acc[0] + D_h * x_val
                    sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    y[b, t, h, p] = z_val * sig_z * y_skipped
                for n in T.serial(STATE):
                    h_last[b, h, p, n] = h_state[n]

    return fwd


@functools.lru_cache(maxsize=128)
def _mamba3_fwd_cuda_kernel(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
) -> Any:
    """Compile the CUDA-safe Mamba3 MIMO forward kernel (caller-owned y/h_last)."""

    import tilelang

    prim = _make_mamba3_fwd_cuda_prim(BATCH, SEQ, HEADS, HEADDIM, STATE)
    return tilelang.compile(prim, target="cuda", out_idx=None)


def mamba3_mimo_fwd_cuda_eager(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array] | None:
    """TileLang-CUDA EAGER Mamba3 MIMO forward → ``(y, h_last)`` or ``None``.

    fp32 carriers only (matches the production fp32 EAGER path). Returns
    ``None`` on any failure so the caller can fall back to the pure-MLX scan.
    """

    ok, _reason = cuda_eager_available()
    if not ok:
        return None

    from cppmega_mlx.nn._tilelang.mamba3 import _validate_inputs

    try:
        import torch
    except Exception:
        return None

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    if seq == 0:
        return mx.zeros((batch, 0, heads, headdim), dtype=x.dtype), h0

    xf = x.astype(mx.float32)
    Bf = B.astype(mx.float32)
    Cf = C.astype(mx.float32)
    zf = z.astype(mx.float32)
    Af = A.astype(mx.float32)
    dtf = dt.astype(mx.float32)
    Df = D.astype(mx.float32)
    h0f = h0.astype(mx.float32)

    try:
        kernel = _mamba3_fwd_cuda_kernel(batch, seq, heads, headdim, state)
        x_t = _mlx_to_torch_cuda(xf)
        B_t = _mlx_to_torch_cuda(Bf)
        C_t = _mlx_to_torch_cuda(Cf)
        z_t = _mlx_to_torch_cuda(zf)
        A_t = _mlx_to_torch_cuda(Af)
        dt_t = _mlx_to_torch_cuda(dtf)
        D_t = _mlx_to_torch_cuda(Df)
        h0_t = _mlx_to_torch_cuda(h0f)
        y_t = torch.zeros(
            batch, seq, heads, headdim, dtype=torch.float32, device="cuda"
        )
        hl_t = torch.zeros(
            batch, heads, headdim, state, dtype=torch.float32, device="cuda"
        )
        kernel(x_t, B_t, C_t, z_t, A_t, dt_t, D_t, h0_t, y_t, hl_t)
        torch.cuda.synchronize()
    except Exception:
        return None

    y = _torch_cuda_to_mlx(y_t, x.dtype)
    h_last = _torch_cuda_to_mlx(hl_t, h0.dtype)
    return y, h_last


# ---------------------------------------------------------------------------
# m2rnn — TODO
# ---------------------------------------------------------------------------


def m2rnn_supported_cuda_eager() -> tuple[bool, str]:
    """m2rnn CUDA EAGER kernel is not yet ported.

    TODO: the m2rnn Path C kernels in ``m2rnn_path_c.py`` are Metal-MSL
    lowered (and several use the Metal ``tir.metal.*`` thread intrinsics like
    mamba3); porting requires a CUDA-safe affine-scan prim_func analogous to
    :func:`_mamba3_fwd_cuda_kernel`. Until then m2rnn stays on the pure-MLX
    reference on CUDA.
    """

    return False, "m2rnn CUDA EAGER kernel not yet ported (uses Metal MSL path)"


__all__ = [
    "cuda_eager_available",
    "sparse_mla_fwd_cuda_eager",
    "mamba3_mimo_fwd_cuda_eager",
    "m2rnn_supported_cuda_eager",
]
