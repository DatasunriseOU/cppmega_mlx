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
            mx.uint8: torch.uint8,
        }
    )
    _NP_DTYPE_FOR_MLX.update(
        {
            mx.float32: "float32",
            mx.float16: "float16",
            mx.bfloat16: "float32",  # numpy has no bf16; carry through fp32
            mx.int32: "int32",
            mx.int64: "int64",
            mx.uint8: "uint8",
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
# FP8 Sparse-MLA — per-token FP8 producer (CUDA-safe) + prepared-buffer apply
# ---------------------------------------------------------------------------
#
# Path C's FP8 Sparse-MLA refuses a full-size scaled-tensor fallback: it must
# consume prepared ``q_fp8 / q_scale / kv_fp8 / kv_scale`` (per-token amax ->
# scale -> e4m3 cast) buffers. On Metal those are produced by a single-pass
# tvm-ffi quant kernel; here we run the equivalent TileLang kernels with
# ``target="cuda"``.
#
# The in-tree producer prim (``_make_fp8_per_token_quant_kernel``) uses
# ``T.alloc_var("float32")`` for the per-element ``normalized`` temporary, which
# nvcc rejects (it emits ``float* normalized`` and then assigns a float to the
# pointer). We therefore vendor a CUDA-safe copy that uses a one-element
# ``T.alloc_local`` accumulator instead — identical amax/scale/e4m3 math, same
# uint8 + fp32 output layout that the apply consumer expects. The prepared-buffer
# *apply* kernel (``_make_fp8_sparse_mla_apply_kernel``) is already CUDA-safe
# (``T.alloc_local`` / ``T.alloc_shared`` / portable e4m3 codec), so we compile
# it verbatim with ``target="cuda"``.


def _make_fp8_per_token_quant_cuda_prim(rows: int, K: int, in_dtype: str) -> Any:
    """CUDA-safe per-token FP8 (e4m3 -> uint8) quant + per-row fp32 scale.

    Mirrors ``sparse_mla_fp8_path_c._make_fp8_per_token_quant_kernel`` exactly
    (per-row amax, ``scale = max(amax/448, 1e-12)``, clamp to +/-448, e4m3
    encode), but replaces the ``T.alloc_var("float32")`` per-element temporary
    with a one-slot ``T.alloc_local`` so CUDA codegen accepts it.

    Defined at module scope (no ``from __future__ import annotations`` here) so
    TileLang's ``get_type_hints`` re-evaluation resolves the shape names.
    """

    import tilelang.language as T
    from tilelang.tileop.metal_quant import float_to_fp8_e4m3fn_bits

    g = globals()
    g.update(
        _CUDA_FP8_PTQ_ROWS=int(rows),
        _CUDA_FP8_PTQ_K=int(K),
        _CUDA_FP8_PTQ_IN=str(in_dtype),
    )

    @T.prim_func
    def fp8_per_token_quant_cuda(
        x: T.Tensor((_CUDA_FP8_PTQ_ROWS * _CUDA_FP8_PTQ_K,), _CUDA_FP8_PTQ_IN),
        fp8: T.Tensor((_CUDA_FP8_PTQ_ROWS * _CUDA_FP8_PTQ_K,), "uint8"),
        scale: T.Tensor((_CUDA_FP8_PTQ_ROWS,), "float32"),
    ):
        with T.Kernel(_CUDA_FP8_PTQ_ROWS, threads=256) as row:
            x_abs = T.alloc_fragment((_CUDA_FP8_PTQ_K,), "float32")
            row_amax = T.alloc_fragment((1,), "float32")
            nrm = T.alloc_local((1,), "float32")
            base = row * _CUDA_FP8_PTQ_K
            for k in T.Parallel(_CUDA_FP8_PTQ_K):
                x_abs[k] = T.abs(T.cast(x[base + k], "float32"))
            T.reduce_max(x_abs, row_amax, dim=0, clear=True)
            row_scale = T.max(
                row_amax[0] * T.cast(1.0 / 448.0, "float32"),
                T.cast(1.0e-12, "float32"),
            )
            if T.get_thread_binding(0) == 0:
                scale[row] = row_scale
            for k in T.Parallel(_CUDA_FP8_PTQ_K):
                nrm[0] = T.cast(x[base + k], "float32") / row_scale
                nrm[0] = T.max(nrm[0], T.cast(-448.0, "float32"))
                nrm[0] = T.min(nrm[0], T.cast(448.0, "float32"))
                fp8[base + k] = float_to_fp8_e4m3fn_bits(nrm[0])

    return fp8_per_token_quant_cuda


@functools.lru_cache(maxsize=128)
def _fp8_per_token_quant_cuda_kernel(rows: int, K: int, in_dtype: str) -> Any:
    import tilelang

    prim = _make_fp8_per_token_quant_cuda_prim(rows, K, in_dtype)
    return tilelang.compile(prim, target="cuda", out_idx=None)


def fp8_per_token_quant_cuda_eager(x: mx.array) -> tuple[mx.array, mx.array] | None:
    """Quantize ``x`` to e4m3 (uint8) + per-final-dim-row fp32 scale on CUDA.

    Returns ``(fp8, scale)`` where ``fp8`` has the same shape as ``x`` (uint8
    storage) and ``scale`` has shape ``x.shape[:-1]`` (float32), matching
    :func:`sparse_mla_fp8_path_c._to_fp8_with_per_token_scale_metal`. Returns
    ``None`` on any failure so the caller can re-raise the documented guard.
    """

    ok, _reason = cuda_eager_available()
    if not ok:
        return None
    if x.ndim < 2 or x.size == 0:
        return None
    if x.dtype not in {mx.float32, mx.float16, mx.bfloat16}:
        return None

    try:
        import torch
    except Exception:
        return None

    K = int(x.shape[-1])
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    in_dtype = {
        mx.float32: "float32",
        mx.float16: "float16",
        mx.bfloat16: "bfloat16",
    }[x.dtype]
    # bf16 has no numpy carrier in the bridge; quantize from a fp32 carrier so
    # the producer math is bit-identical to the Metal path (which also widens
    # to fp32 before amax/encode).
    if x.dtype == mx.bfloat16:
        x_in = x.astype(mx.float32)
        in_dtype = "float32"
    else:
        x_in = x

    try:
        kernel = _fp8_per_token_quant_cuda_kernel(rows, K, in_dtype)
        x_t = _mlx_to_torch_cuda(x_in).reshape(-1)
        fp8_t = torch.zeros(rows * K, dtype=torch.uint8, device="cuda")
        scale_t = torch.zeros(rows, dtype=torch.float32, device="cuda")
        kernel(x_t, fp8_t, scale_t)
        torch.cuda.synchronize()
    except Exception:
        return None

    fp8 = _torch_cuda_to_mlx(fp8_t.reshape(x.shape), mx.uint8)
    scale = _torch_cuda_to_mlx(scale_t.reshape(x.shape[:-1]), mx.float32)
    return fp8, scale


@functools.lru_cache(maxsize=128)
def _fp8_sparse_mla_apply_cuda_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    K: int,
    d_v: int,
    threads: int,
    out_dtype: str,
    index_dtype: str,
) -> Any:
    """Compile the in-tree prepared-buffer FP8 Sparse-MLA apply prim for CUDA."""

    import tilelang

    from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (
        _make_fp8_sparse_mla_apply_kernel,
    )

    prim = _make_fp8_sparse_mla_apply_kernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        head_kv=head_kv,
        topk=topk,
        K=K,
        d_v=d_v,
        threads=threads,
        out_dtype=out_dtype,
        index_dtype=index_dtype,
    )
    return tilelang.compile(prim, target="cuda", out_idx=None)


_TORCH_DTYPE_FOR_OUT: dict[str, Any] = {}


def _torch_out_dtype(name: str) -> Any:
    import torch

    if not _TORCH_DTYPE_FOR_OUT:
        _TORCH_DTYPE_FOR_OUT.update(
            {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
        )
    return _TORCH_DTYPE_FOR_OUT[name]


def fp8_sparse_mla_apply_cuda_eager(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    sinks: mx.array | None,
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    K: int,
    d_v: int,
    threads: int,
    out_dtype: str,
    index_dtype: str,
) -> tuple[mx.array, mx.array] | None:
    """TileLang-CUDA EAGER prepared-buffer FP8 Sparse-MLA apply -> ``(out, lse)``.

    Consumes the prepared ``q_fp8/q_scale/kv_fp8/kv_scale`` buffers exactly like
    the Metal apply kernel; returns flat ``out`` reshaped to
    ``(B,S,H,d_v)`` and ``lse`` reshaped to ``(B,S,H)``. ``None`` on failure.
    """

    ok, _reason = cuda_eager_available()
    if not ok:
        return None
    try:
        import torch
    except Exception:
        return None

    lanes = batch * seq_len * heads
    mx_out_dtype = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[out_dtype]

    try:
        kernel = _fp8_sparse_mla_apply_cuda_kernel(
            batch,
            seq_len,
            heads,
            seq_len_kv,
            kv_group,
            head_kv,
            topk,
            K,
            d_v,
            threads,
            out_dtype,
            index_dtype,
        )
        q_fp8_t = _mlx_to_torch_cuda(q_fp8.astype(mx.uint8)).reshape(-1)
        q_scale_t = _mlx_to_torch_cuda(q_scale.astype(mx.float32)).reshape(-1)
        kv_fp8_t = _mlx_to_torch_cuda(kv_fp8.astype(mx.uint8)).reshape(-1)
        kv_scale_t = _mlx_to_torch_cuda(kv_scale.astype(mx.float32)).reshape(-1)
        idx_t = _mlx_to_torch_cuda(indices).reshape(-1)
        sm_t = torch.tensor([float(sm_scale)], dtype=torch.float32, device="cuda")
        if sinks is None:
            sinks_t = torch.zeros(heads, dtype=torch.float32, device="cuda")
            has_sinks_t = torch.zeros(1, dtype=torch.int32, device="cuda")
        else:
            sinks_t = _mlx_to_torch_cuda(sinks.astype(mx.float32)).reshape(-1)
            has_sinks_t = torch.ones(1, dtype=torch.int32, device="cuda")
        out_t = torch.zeros(
            lanes * d_v, dtype=_torch_out_dtype(out_dtype), device="cuda"
        )
        lse_t = torch.zeros(lanes, dtype=torch.float32, device="cuda")
        kernel(
            q_fp8_t,
            q_scale_t,
            kv_fp8_t,
            kv_scale_t,
            idx_t,
            sm_t,
            sinks_t,
            has_sinks_t,
            out_t,
            lse_t,
        )
        torch.cuda.synchronize()
    except Exception:
        return None

    out = _torch_cuda_to_mlx(
        out_t.reshape(batch, seq_len, heads, d_v), mx_out_dtype
    )
    lse = _torch_cuda_to_mlx(
        lse_t.reshape(batch, seq_len, heads), mx.float32
    )
    return out, lse


# ---------------------------------------------------------------------------
# m2rnn (fwd) — vendored CUDA-safe mapped-packed-post affine scan
# ---------------------------------------------------------------------------
#
# The production m2rnn Path C forward (``m2rnn_path_c._make_mapped_packed_post_
# fwd_prim_func``) is a per-(b, head, v) lane sequential affine scan
# (``h = f*h + (1-f)*tanh(h@W + k*v)``; ``post = (q.h + v*D) * g * sigmoid(g)``).
# It is already CUDA-shaped (block-id + ``T.get_thread_binding`` lane,
# ``T.alloc_local`` state), with ONE CUDA blocker: the gate temporary
# ``sig_g = T.alloc_var(T.float32, init=0.0)`` (the same non-lvalue
# ``float sig_g = 0`` nvcc reject as mamba3's accumulators). We vendor a
# CUDA-safe copy that uses a one-slot ``T.alloc_local`` for the gate instead.
# Math, buffer order, and output layout are byte-for-byte the same as the Metal
# prim, so the consumer (``m2rnn`` nn module) sees identical ``(post, h_last)``.
#
# Defined at module scope (no ``from __future__ import annotations``) so
# TileLang's ``get_type_hints`` re-evaluation resolves the shape names.


def _make_m2rnn_mapped_packed_post_fwd_cuda_prim(
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    w_heads: int,
    f_heads: int,
    k_dim: int,
    v_dim: int,
    projected_dim: int,
    carrier_dtype: str,
) -> Any:
    """CUDA-safe mapped-packed-post m2rnn forward (post, h_last, tanh_cache)."""

    import tilelang.language as T

    conv_dim = q_heads * k_dim + k_heads * k_dim + v_heads * v_dim
    k_offset = q_heads * k_dim
    v_offset = k_offset + k_heads * k_dim
    q_group = total_heads // q_heads
    k_group = total_heads // k_heads
    v_group = total_heads // v_heads
    g_repeat = total_heads // g_heads
    w_group = total_heads // w_heads
    f_group = total_heads // f_heads
    g_dim = g_heads * v_dim
    g_offset = projected_dim - g_dim
    features = total_heads * v_dim
    lanes = batch * features
    threads = min(256, max(1, lanes))
    accum_dtype = "float32"

    @T.prim_func
    def fwd(
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        W: T.Tensor((w_heads, v_dim, v_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, f_heads), carrier_dtype),
        h0: T.Tensor((batch, total_heads, k_dim, v_dim), carrier_dtype),
        D: T.Tensor((total_heads, v_dim), carrier_dtype),
        projected: T.Tensor((batch, seq, projected_dim), carrier_dtype),
        h_last: T.Tensor((batch, total_heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, total_heads, k_dim, v_dim), carrier_dtype),
        post: T.Tensor((batch, seq, features), carrier_dtype),
    ):
        with T.Kernel(T.ceildiv(lanes, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = bx * threads + tid
            h_state = T.alloc_local((k_dim, v_dim), accum_dtype)
            h_next = T.alloc_local((k_dim, v_dim), accum_dtype)
            y_acc = T.alloc_local((1,), accum_dtype)
            acc = T.alloc_local((1,), accum_dtype)
            sig_g = T.alloc_local((1,), accum_dtype)
            if lane < lanes:
                feature = lane % features
                vv_out = feature % v_dim
                h = feature // v_dim
                b = lane // features
                q_src = h // q_group
                k_src = h // k_group
                v_src = h // v_group
                w_src = h // w_group
                f_src = h // f_group
                g_flat = feature // g_repeat
                q_head_offset = q_src * k_dim
                k_head_offset = k_offset + k_src * k_dim
                v_head_offset = v_offset + v_src * v_dim
                v_index = v_head_offset + vv_out
                g_index = g_offset + g_flat
                d_val = T.cast(D[h, vv_out], accum_dtype)

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        h_state[kk, vv] = T.cast(h0[b, h, kk, vv], accum_dtype)

                for t in T.serial(seq):
                    f_val = T.cast(xf[b, t, f_src], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    y_acc[0] = 0.0

                    for kk in T.serial(k_dim):
                        k_val = T.cast(
                            conv_input[b, t, k_head_offset + kk],
                            accum_dtype,
                        )
                        q_val = T.cast(
                            conv_input[b, t, q_head_offset + kk],
                            accum_dtype,
                        )
                        for vv in T.serial(v_dim):
                            acc[0] = 0.0
                            for v0 in T.serial(v_dim):
                                acc[0] = acc[0] + h_state[kk, v0] * T.cast(
                                    W[w_src, v0, vv],
                                    accum_dtype,
                                )
                            z = acc[0] + k_val * T.cast(
                                conv_input[b, t, v_head_offset + vv],
                                accum_dtype,
                            )
                            tz = T.tanh(z)
                            if vv_out == 0:
                                tanh_cache[b, t, h, kk, vv] = T.cast(
                                    tz,
                                    carrier_dtype,
                                )
                            h_next[kk, vv] = f_val * h_state[kk, vv] + one_minus_f * tz
                        y_acc[0] = y_acc[0] + q_val * h_next[kk, vv_out]

                    g_val = T.cast(projected[b, t, g_index], accum_dtype)
                    if g_val >= 0.0:
                        sig_g[0] = 1.0 / (1.0 + T.exp(-g_val))
                    else:
                        sig_g[0] = T.exp(g_val)
                        sig_g[0] = sig_g[0] / (1.0 + sig_g[0])
                    v_val = T.cast(conv_input[b, t, v_index], accum_dtype)
                    post[b, t, feature] = T.cast(
                        (y_acc[0] + v_val * d_val) * g_val * sig_g[0],
                        carrier_dtype,
                    )

                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_state[kk, vv] = h_next[kk, vv]
                if vv_out == 0:
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_last[b, h, kk, vv] = T.cast(
                                h_state[kk, vv],
                                carrier_dtype,
                            )

    return fwd


@functools.lru_cache(maxsize=128)
def _m2rnn_mapped_packed_post_fwd_cuda_kernel(
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    w_heads: int,
    f_heads: int,
    k_dim: int,
    v_dim: int,
    projected_dim: int,
    carrier_dtype: str,
) -> Any:
    import tilelang

    prim = _make_m2rnn_mapped_packed_post_fwd_cuda_prim(
        batch,
        seq,
        total_heads,
        q_heads,
        k_heads,
        v_heads,
        g_heads,
        w_heads,
        f_heads,
        k_dim,
        v_dim,
        projected_dim,
        carrier_dtype,
    )
    return tilelang.compile(prim, target="cuda", out_idx=None)


def m2rnn_mapped_packed_post_fwd_cuda_eager(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    w_heads: int,
    f_heads: int,
    k_dim: int,
    v_dim: int,
    projected_dim: int,
    carrier_dtype: str,
) -> tuple[mx.array, mx.array, mx.array] | None:
    """TileLang-CUDA EAGER mapped-packed-post m2rnn fwd -> (post, h_last, tanh_cache).

    Returns ``None`` on any failure so the caller can fall back to the pure-MLX
    reference scan. ``post`` has shape ``(B, S, total_heads*v_dim)``, ``h_last``
    ``(B, total_heads, k_dim, v_dim)``, ``tanh_cache``
    ``(B, S, total_heads, k_dim, v_dim)`` (all in ``conv_input.dtype``).
    """

    ok, _reason = cuda_eager_available()
    if not ok:
        return None
    try:
        import torch
    except Exception:
        return None

    batch, seq, _ = (int(x) for x in conv_input.shape)
    features = total_heads * v_dim
    out_dtype = conv_input.dtype

    # The vendored prim takes fp32/fp16 carriers; widen bf16 to fp32 (the bridge
    # has no native bf16 numpy carrier) — matches the Metal carrier widening.
    if out_dtype == mx.bfloat16:
        carrier_mx = mx.float32
        carrier_name = "float32"
    else:
        carrier_mx = out_dtype
        carrier_name = carrier_dtype

    try:
        kernel = _m2rnn_mapped_packed_post_fwd_cuda_kernel(
            batch,
            seq,
            total_heads,
            q_heads,
            k_heads,
            v_heads,
            g_heads,
            w_heads,
            f_heads,
            k_dim,
            v_dim,
            projected_dim,
            carrier_name,
        )
        ci_t = _mlx_to_torch_cuda(conv_input.astype(carrier_mx))
        W_t = _mlx_to_torch_cuda(W.astype(carrier_mx))
        xf_t = _mlx_to_torch_cuda(xf.astype(carrier_mx))
        h0_t = _mlx_to_torch_cuda(h0.astype(carrier_mx))
        D_t = _mlx_to_torch_cuda(D.astype(carrier_mx))
        proj_t = _mlx_to_torch_cuda(projected.astype(carrier_mx))
        torch_carrier = _MLX_TO_TORCH_DTYPE[carrier_mx]
        h_last_t = torch.zeros(
            batch, total_heads, k_dim, v_dim, dtype=torch_carrier, device="cuda"
        )
        tanh_t = torch.zeros(
            batch, seq, total_heads, k_dim, v_dim, dtype=torch_carrier, device="cuda"
        )
        post_t = torch.zeros(
            batch, seq, features, dtype=torch_carrier, device="cuda"
        )
        kernel(ci_t, W_t, xf_t, h0_t, D_t, proj_t, h_last_t, tanh_t, post_t)
        torch.cuda.synchronize()
    except Exception:
        return None

    post = _torch_cuda_to_mlx(post_t, out_dtype)
    h_last = _torch_cuda_to_mlx(h_last_t, out_dtype)
    tanh_cache = _torch_cuda_to_mlx(tanh_t, out_dtype)
    return post, h_last, tanh_cache


def m2rnn_supported_cuda_eager() -> tuple[bool, str]:
    """Report whether the m2rnn TileLang-CUDA EAGER forward can dispatch."""

    return cuda_eager_available()


__all__ = [
    "cuda_eager_available",
    "sparse_mla_fwd_cuda_eager",
    "mamba3_mimo_fwd_cuda_eager",
    "fp8_per_token_quant_cuda_eager",
    "fp8_sparse_mla_apply_cuda_eager",
    "m2rnn_mapped_packed_post_fwd_cuda_eager",
    "m2rnn_supported_cuda_eager",
]
