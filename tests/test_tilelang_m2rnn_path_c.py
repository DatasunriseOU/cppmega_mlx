"""Coverage for the Path C TileLang DSL m2rnn forward surface."""

# pyright: reportMissingImports=false

from __future__ import annotations

import importlib

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang.m2rnn import m2rnn_apply, m2rnn_metal_status


def _np(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    mx.eval(x)
    return np.asarray(x)


def _make_m2rnn_inputs(
    *,
    batch: int = 1,
    seq: int = 4,
    heads: int = 2,
    k_dim: int = 4,
    v_dim: int = 4,
    dtype: mx.Dtype = mx.float32,
    seed: int = 7,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = (mx.random.normal((batch, seq, heads, k_dim)) * 0.1).astype(dtype)
    k = (mx.random.normal((batch, seq, heads, k_dim)) * 0.1).astype(dtype)
    v = (mx.random.normal((batch, seq, heads, v_dim)) * 0.1).astype(dtype)
    W = (mx.random.normal((heads, v_dim, v_dim)) * 0.1).astype(dtype)
    xf = (mx.random.uniform(0.001, 0.05, (batch, seq, heads))).astype(dtype)
    h0 = mx.zeros((batch, heads, k_dim, v_dim), dtype=dtype)
    mx.eval(q, k, v, W, xf, h0)
    return q, k, v, W, xf, h0


def test_m2rnn_path_c_module_imports() -> None:
    module = importlib.import_module("cppmega_mlx.nn._tilelang.m2rnn_path_c")
    assert hasattr(module, "m2rnn_apply_path_c")


def _try_import_m2rnn_path_c():  # type: ignore[no-untyped-def]
    try:
        return importlib.import_module("cppmega_mlx.nn._tilelang.m2rnn_path_c")
    except Exception:
        return None


def test_m2rnn_path_c_forward_matches_path_b_when_available() -> None:
    """When ``m2rnn_path_c.m2rnn_apply_path_c`` exists, it must match Path B
    within fp32 tolerance on a small canonical shape."""

    module = _try_import_m2rnn_path_c()
    if module is None:
        pytest.xfail("m2rnn_path_c module not implemented yet")

    apply_path_c = getattr(module, "m2rnn_apply_path_c", None)
    if apply_path_c is None:
        pytest.xfail(
            "m2rnn_path_c module exists but does not expose m2rnn_apply_path_c yet"
        )

    if not m2rnn_metal_status().available:
        pytest.skip("m2rnn Metal Path B is not available on this host")

    inputs = _make_m2rnn_inputs(dtype=mx.float32)
    y_pc = apply_path_c(*inputs, force_path_c=True)
    y_pb = m2rnn_apply(*inputs)
    np.testing.assert_allclose(_np(y_pc), _np(y_pb), rtol=1e-3, atol=1e-4)
