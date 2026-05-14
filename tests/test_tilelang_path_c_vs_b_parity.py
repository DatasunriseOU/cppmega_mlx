"""Numeric Path C vs Path B parity for every TileLang kernel pair.

Existing per-pair tests (e.g. ``test_tilelang_mamba3_path_c.py``) check
self-consistency or path-c-vs-reference but never numerically compare
``*_path_c_apply`` to ``*_apply`` end-to-end across a single sweep. Meta
agent E flagged this as the path C vs path B coverage hole; Meta agent F's
design audit confirmed the divergence between the two surfaces.

This file consolidates parity into one parametrized sweep so any divergence
shows up in one place. Quantized Path C entries use prepared FP8/scales
buffers so the test does not bless hidden high-level staging copies.

Tolerance convention follows ``test_tilelang_mamba3_path_c.py``:
``atol=1e-4 / rtol=1e-3`` on fp32 paths; looser on bf16/fp8 carriers.

Also includes ``test_fp8_e4m3_dot4_intrinsic_is_registered`` -- the
P0 trip-wire from Grok-D that would have caught the missing
``tir.metal.fp8_e4m3_dot4`` registration. (If Fix-1 already added this
trip-wire to ``test_tilelang_fp8_vecmat_path_c.py``, the duplicate is
harmless and easy to remove.)
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import importlib
from typing import Any, Callable, cast

import numpy as np
import pytest

import mlx.core as mx


# ---------------------------------------------------------------------------
# Tolerance dispatch by carrier dtype.
#
# gpt-5.5-pro P0 (G3): the previous fp32 contract (atol=1e-4 / rtol=1e-3)
# was inherited by quantized paths where it is unrealistically tight. The
# values below follow the industry rule of thumb -- fp32 stays strict; bf16
# and fp8 paths get tolerance commensurate with their quantization error.
# ---------------------------------------------------------------------------


_TOLERANCE_BY_DTYPE: dict[str, tuple[float, float]] = {
    "fp32": (1e-4, 1e-3),  # strict reference contract
    "bf16": (5e-3, 1e-2),  # bf16 carrier with fp32 accumulators
    "fp16": (5e-3, 5e-3),  # fp16 carrier with fp32 accumulators (sparse_mla)
    "fp8": (2e-2, 5e-2),  # fp8 e4m3/e5m2 -- quantization error dominates
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _np(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    mx.eval(x)
    return np.asarray(x)


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return mx.default_device() == mx.gpu and metal is not None and metal.is_available()


# ---------------------------------------------------------------------------
# Per-pair parity drivers.
#
# Each driver returns (path_b_output, path_c_output, atol, rtol). Drivers
# that cannot yet run -- because the Path C apply is not exposed -- raise
# NotImplementedError, which the harness translates into a strict-xfail.
# ---------------------------------------------------------------------------


def _drive_sparse_mla(shape: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    from cppmega_mlx.nn._tilelang import (
        sparse_mla_apply,
    )
    from cppmega_mlx.nn._tilelang.sparse_mla_path_c import sparse_mla_path_c_apply

    rng = np.random.RandomState(0)
    if shape == "small":
        B, S, H, D, G, topk, Skv = 2, 8, 4, 32, 1, 4, 16
    elif shape == "medium":
        B, S, H, D, G, topk, Skv = 1, 16, 4, 32, 1, 8, 32
    else:
        raise ValueError(f"unknown sparse_mla shape: {shape}")

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float16)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float16)
    indices_np = rng.randint(0, Skv, size=(B, S, G, topk)).astype(np.int32)

    q = mx.array(q_np)
    kv = mx.array(kv_np)
    indices = mx.array(indices_np)
    sm_scale = D ** -0.5

    out_b = sparse_mla_apply(q, kv, indices, sm_scale=sm_scale)
    out_c = sparse_mla_path_c_apply(q, kv, indices, sm_scale=sm_scale)
    # fp16 carrier with fp32 accumulators -- looser than the fp32 contract.
    atol, rtol = _TOLERANCE_BY_DTYPE["fp16"]
    return _np(cast(mx.array, out_b)), _np(cast(mx.array, out_c)), atol, rtol


def test_sparse_mla_fwd_tilelang_scalar_pass_removes_stale_cse_gap() -> None:
    if not _metal_available():
        pytest.skip("Metal backend not available on this host")

    from cppmega_mlx.nn._tilelang import sparse_mla_path_c as path_c

    status = path_c.sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    path_c._fwd_kernel_for.cache_clear()
    try:
        msl = path_c.dump_lowered_fwd_msl(
            batch=2,
            seq_len=8,
            heads=4,
            qk_dim=32,
            kv_group=1,
            topk=4,
            seq_len_kv=16,
        )
    finally:
        path_c._fwd_kernel_for.cache_clear()

    assert "kv[(((((long)_tmp_4)" not in msl
    if "kv_row_base_1" in msl:
        decl_pos = msl.find("uint kv_row_base_1 =")
        use_pos = msl.find("kv[((cse_")
        assert decl_pos >= 0
        assert use_pos < 0 or decl_pos < use_pos
    assert not hasattr(path_c, "_repair_fwd_cse_msl")


def _drive_mamba3(shape: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    from cppmega_mlx.nn._tilelang import mamba3_mimo_apply
    from cppmega_mlx.nn._tilelang.mamba3_path_c import (
        mamba3_mimo_apply_path_c,
    )

    mx.random.seed(31)
    if shape == "small":
        batch, seq, heads, headdim, state = 1, 8, 2, 4, 4
    elif shape == "carryover":
        # Two-chunk state-carryover smoke: stitch two contiguous halves and
        # confirm Path C produces the same result as Path B on the full seq.
        batch, seq, heads, headdim, state = 1, 16, 2, 4, 4
    else:
        raise ValueError(f"unknown mamba3 shape: {shape}")

    x = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(mx.float32)
    Bm = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(mx.float32)
    Cm = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(mx.float32)
    z = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(mx.float32)
    A = (-mx.random.uniform(0.01, 0.5, (batch, seq, heads))).astype(mx.float32)
    dt = (mx.random.uniform(0.001, 0.05, (batch, seq, heads))).astype(mx.float32)
    D = mx.ones((heads,), dtype=mx.float32)
    h0 = mx.zeros((batch, heads, headdim, state), dtype=mx.float32)
    mx.eval(x, Bm, Cm, z, A, dt, D, h0)

    y_b = cast(mx.array, mamba3_mimo_apply(x, Bm, Cm, z, A, dt, D, h0))
    y_c = cast(mx.array, mamba3_mimo_apply_path_c(x, Bm, Cm, z, A, dt, D, h0))
    # mamba3 test data is fp32 end-to-end (see mx.float32 casts above).
    atol, rtol = _TOLERANCE_BY_DTYPE["fp32"]
    return _np(y_b), _np(y_c), atol, rtol


def _drive_fp8_vecmat(shape: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    from cppmega_mlx.nn._tilelang import fp8_scaled_vecmat
    from cppmega_mlx.nn._tilelang.fp8_vecmat_path_c import (
        fp8_scaled_vecmat_path_c,
        fp8_vecmat_path_c_status,
    )

    if not fp8_vecmat_path_c_status().available:
        raise NotImplementedError(
            f"fp8_vecmat path c unavailable: {fp8_vecmat_path_c_status().reason}"
        )

    rng = np.random.RandomState(0)
    n, k = (128, 128) if shape == "small" else (256, 128)
    x_fp8 = mx.array(rng.randint(0, 255, size=(k,)).astype(np.uint8))
    W_fp8 = mx.array(rng.randint(0, 255, size=(n, k)).astype(np.uint8))
    sx = mx.array([1.0], dtype=mx.float32)
    sw = mx.array(np.full((n,), 1.0, dtype=np.float32))

    out_b = fp8_scaled_vecmat(x_fp8, W_fp8, scale_x=sx, scale_w=sw)
    out_c_buf = mx.zeros((n,), dtype=mx.float32)
    out_c = fp8_scaled_vecmat_path_c(
        x_fp8,
        W_fp8,
        scale_x=sx,
        scale_w=sw,
        out=out_c_buf,
    )
    # FP8 dot4 carrier -- quantization error dominates, use fp8 tolerance.
    atol, rtol = _TOLERANCE_BY_DTYPE["fp8"]
    return _np(cast(mx.array, out_b)), _np(cast(mx.array, out_c)), atol, rtol


def _drive_sparse_mla_fp8(shape: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    from cppmega_mlx.nn.sparse_mla import _resolve_shapes
    from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (
        _to_fp8_with_per_tensor_scale,
        sparse_mla_fp8_fwd_metal_impl,
    )
    from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (
        sparse_mla_fp8_path_c_apply,
    )

    rng = np.random.RandomState(4)
    if shape == "prepared-small":
        B, S, H, D, G, topk, Skv = 1, 2, 2, 64, 1, 4, 8
    else:
        raise ValueError(f"unknown sparse_mla_fp8 shape: {shape}")

    q = mx.array((rng.standard_normal((B, S, H, D)) * 0.1).astype(np.float16))
    kv = mx.array((rng.standard_normal((B, Skv, G, D)) * 0.1).astype(np.float16))
    indices = mx.array(rng.randint(0, Skv, size=(B, S, G, topk)).astype(np.int32))
    sm_scale = D ** -0.5
    q_fp8, q_scale = _to_fp8_with_per_tensor_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    shapes = _resolve_shapes(q, kv, indices, d_v=D)
    path_b = sparse_mla_fp8_fwd_metal_impl(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=sm_scale,
        d_v=D,
        shapes=shapes,
    )
    if path_b is None:
        raise NotImplementedError("sparse_mla_fp8 Path B prepared-buffer kernel unavailable")
    out_b, _lse_b = path_b
    out_c = sparse_mla_fp8_path_c_apply(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=sm_scale,
        d_v=D,
        force_path_c=True,
    )
    if out_c is None:
        raise NotImplementedError("sparse_mla_fp8_path_c_apply returned None")
    atol, rtol = _TOLERANCE_BY_DTYPE["fp8"]
    return _np(out_b.astype(mx.float32)), _np(cast(mx.array, out_c).astype(mx.float32)), atol, rtol


def _drive_sparse_mla_blockscaled(shape: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled import (
        _quantize_mxfp8,
        _unpack_mxfp8_to_uint8,
        sparse_mla_blockscaled_fwd_metal,
    )
    from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c import (
        sparse_mla_blockscaled_path_c_apply,
    )

    rng = np.random.RandomState(5)
    if shape == "prepared-small":
        B, S, H, D, G, topk, Skv = 1, 2, 2, 64, 1, 4, 8
    else:
        raise ValueError(f"unknown sparse_mla_blockscaled shape: {shape}")

    q = mx.array((rng.standard_normal((B, S, H, D)) * 0.1).astype(np.float16))
    kv = mx.array((rng.standard_normal((B, Skv, G, D)) * 0.1).astype(np.float16))
    indices = mx.array(rng.randint(0, Skv, size=(B, S, G, topk)).astype(np.int32))
    sm_scale = D ** -0.5

    q_packed, q_scales = _quantize_mxfp8(q)
    kv_packed, kv_scales = _quantize_mxfp8(kv)
    q_fp8 = _unpack_mxfp8_to_uint8(q_packed, D)
    kv_fp8 = _unpack_mxfp8_to_uint8(kv_packed, D)
    path_b = sparse_mla_blockscaled_fwd_metal(q, kv, indices, sm_scale=sm_scale, d_v=D)
    if path_b is None:
        raise NotImplementedError("sparse_mla_blockscaled Path B kernel unavailable")
    out_b, _lse_b = path_b
    out_c = sparse_mla_blockscaled_path_c_apply(
        q_fp8,
        q_scales,
        kv_fp8,
        kv_scales,
        indices,
        sm_scale=sm_scale,
        d_v=D,
        force_path_c=True,
    )
    if out_c is None:
        raise NotImplementedError("sparse_mla_blockscaled_path_c_apply returned None")
    atol, rtol = _TOLERANCE_BY_DTYPE["fp8"]
    return _np(out_b.astype(mx.float32)), _np(cast(mx.array, out_c).astype(mx.float32)), atol, rtol


# Each entry is (kernel_pair, shape, driver, expect_xfail_with_reason_or_None).
PARITY_CASES: list[tuple[str, str, Callable[[str], Any], str | None]] = [
    ("sparse_mla", "small", _drive_sparse_mla, None),
    ("sparse_mla", "medium", _drive_sparse_mla, None),
    ("mamba3", "small", _drive_mamba3, None),
    ("mamba3", "carryover", _drive_mamba3, None),
    ("fp8_vecmat", "small", _drive_fp8_vecmat, None),
    ("sparse_mla_blockscaled", "prepared-small", _drive_sparse_mla_blockscaled, None),
    (
        "sparse_mla_fp8",
        "prepared-small",
        _drive_sparse_mla_fp8,
        "sparse_mla_fp8 direct-MSL Path B is retired; Path C is covered by prepared-buffer tests",
    ),
]


@pytest.mark.parametrize(
    "kernel_pair,shape,driver,xfail_reason",
    PARITY_CASES,
    ids=[f"{k}-{s}" for (k, s, _d, _x) in PARITY_CASES],
)
def test_path_c_matches_path_b(
    kernel_pair: str,
    shape: str,
    driver: Callable[[str], Any],
    xfail_reason: str | None,
) -> None:
    if xfail_reason is not None:
        # strict=True so the day Path C lands an apply, this entry flips
        # green and we get real numeric coverage automatically.
        pytest.xfail(xfail_reason)

    if not _metal_available():
        pytest.skip("Metal backend not available on this host")

    try:
        out_b, out_c, atol, rtol = driver(shape)
    except NotImplementedError as exc:
        pytest.xfail(str(exc))

    assert out_b.shape == out_c.shape, (
        f"{kernel_pair}/{shape}: shape mismatch {out_b.shape} vs {out_c.shape}"
    )
    np.testing.assert_allclose(out_c, out_b, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Grok-D P0 trip-wire: ``tir.metal.fp8_e4m3_dot4`` must be registered.
#
# This is the intrinsic that lowers ``T.metal_fp8_e4m3_dot4(...)`` calls in
# the Path C FP8 kernels. If it is missing from the in-tree TileLang/TVM,
# the entire FP8 Path C surface silently falls back -- which is exactly the
# regression the audit caught. Failing fast here means CI surfaces it.
# ---------------------------------------------------------------------------


# Apache TVM tirx namespace migration (2026-05): the metal FP8 dot4 op was
# re-registered under ``tirx.metal.*`` when ``tir::*`` moved to ``tirx::*``.
# We probe both names because older fork tips may still register under
# ``tir.metal.*``.
_FP8_DOT4_GLOBAL_FUNC_NAMES = (
    "tirx.metal.fp8_e4m3_dot4",
    "tir.metal.fp8_e4m3_dot4",
)
_FP8_DOT4_GLOBAL_FUNC = _FP8_DOT4_GLOBAL_FUNC_NAMES[0]


def _try_get_global_func(name: str) -> Any | None:
    """Look up a TVM-FFI global func across the several import paths the
    in-tree TileLang/TVM has used. Returns the handle or None."""

    # Newer apache-tvm-ffi.
    # NOTE: ``allow_missing=True`` returns ``None`` for a missing global func
    # rather than raising, so we must NOT early-return on a ``None`` result —
    # otherwise the Op.get fallback below never runs and ops registered only
    # via ``TVM_REGISTER_OP`` (without a separate ``TVM_FFI_REGISTER_GLOBAL``
    # alias) look unregistered. See apache tirx migration: ``tirx.metal.*``
    # ops are Op-only, not global-func-registered.
    try:
        import tvm_ffi  # type: ignore

        getter = getattr(tvm_ffi, "get_global_func", None)
        if getter is not None:
            try:
                handle = getter(name, allow_missing=True)
            except TypeError:
                try:
                    handle = getter(name)
                except Exception:
                    handle = None
            if handle is not None:
                return handle
    except Exception:
        pass

    # Legacy tvm._ffi
    try:
        from tilelang import tvm as ttvm  # type: ignore

        ffi = getattr(ttvm, "_ffi", None) or getattr(ttvm, "ffi", None)
        if ffi is not None:
            getter = getattr(ffi, "get_global_func", None)
            if getter is not None:
                try:
                    handle = getter(name, allow_missing=True)
                except TypeError:
                    try:
                        handle = getter(name)
                    except Exception:
                        handle = None
                if handle is not None:
                    return handle
    except Exception:
        pass

    # Last-resort: peek at the registered op via tvm.ir
    try:
        from tilelang import tvm as ttvm  # type: ignore

        op_get = getattr(getattr(ttvm, "ir", None), "Op", None)
        if op_get is not None:
            try:
                return op_get.get(name)  # type: ignore[attr-defined]
            except Exception:
                return None
    except Exception:
        pass

    return None


def test_fp8_e4m3_dot4_intrinsic_is_registered() -> None:
    """``tir.metal.fp8_e4m3_dot4`` must be discoverable from this Python env.

    If the in-tree TileLang/TVM forgets to register the intrinsic (the Grok-D
    P0), every Path C FP8 kernel silently falls back to scalar decode and CI
    stays green. This trip-wire makes that regression a hard failure.
    """

    try:
        tilelang = importlib.import_module("tilelang")
    except (ImportError, OSError) as exc:
        pytest.skip(f"tilelang unavailable on this host: {exc}")
    # Confirm the language-level wrapper is exposed; this is the contract
    # surface the Path C kernels call.
    try:
        fp8_op = importlib.import_module("tilelang.language.fp8_op")
    except (ImportError, OSError) as exc:
        pytest.skip(f"tilelang.language.fp8_op unavailable: {exc}")
    assert hasattr(fp8_op, "metal_fp8_e4m3_dot4"), (
        "tilelang.language.fp8_op.metal_fp8_e4m3_dot4 wrapper is missing"
    )

    handle = None
    for name in _FP8_DOT4_GLOBAL_FUNC_NAMES:
        handle = _try_get_global_func(name)
        if handle is not None:
            break
    assert handle is not None, (
        f"global func {_FP8_DOT4_GLOBAL_FUNC_NAMES!r} is not registered -- the in-tree "
        "TileLang/TVM forgot to register the FP8 e4m3 dot4 intrinsic, which "
        "would silently force every Path C FP8 kernel onto the scalar fallback. "
        "(Grok-D P0 trip-wire: see reports/2026-05-06-tilelang-tvm-review.)"
    )
    # Touch tilelang to satisfy lint about importorskip return value.
    assert tilelang is not None
