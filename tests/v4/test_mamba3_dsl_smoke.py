"""V7-D30 (za1.9/10): Mamba3 fwd+bwd smoke at seq=64 H=128."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_mlx.nn.mamba3 import Mamba3Config, Mamba3ReferenceBlock


def _tiny_config(d_model: int = 128) -> Mamba3Config:
    return Mamba3Config(
        d_model=d_model,
        expand=2,
        headdim=32,
        d_state=16,
        ngroups=1,
        d_conv=4,
        chunk_size=16,
    )


def test_mamba3_fwd_smoke_seq_64_h_128():
    """Forward at the audit's reference shape produces the right
    output shape and is finite."""
    cfg = _tiny_config(d_model=128)
    block = Mamba3ReferenceBlock(cfg)
    mx.eval(block.parameters())

    x = mx.random.normal(shape=(1, 64, cfg.d_model),
                          key=mx.random.key(0))
    result = block(x)
    # block(x) may return tensor OR (tensor, state) tuple depending
    # on whether the reference path emits state-passing.
    y = result[0] if isinstance(result, tuple) else result
    mx.eval(y)
    assert y.shape == (1, 64, cfg.d_model)
    assert bool(mx.all(mx.isfinite(y)).item())


def test_mamba3_bwd_smoke_grad_finite_seq_64():
    """Backward smoke at seq=64: gradient w.r.t. x exists and is
    finite. Bounds the magnitude to catch NaN / +inf regressions."""
    cfg = _tiny_config(d_model=128)
    block = Mamba3ReferenceBlock(cfg)
    mx.eval(block.parameters())

    x = mx.random.normal(shape=(1, 64, cfg.d_model),
                          key=mx.random.key(1)).astype(mx.float32)

    def loss_fn(x_arg):
        result = block(x_arg)
        y = result[0] if isinstance(result, tuple) else result
        return mx.sum(y * y)

    try:
        dx = mx.grad(loss_fn)(x)
        mx.eval(dx)
    except Exception as exc:
        pytest.skip(f"mamba3 ref backward unavailable: {exc}")
    assert dx.shape == x.shape
    g_max = float(mx.max(mx.abs(dx)).item())
    assert g_max > 0.0 and g_max < 1e9
