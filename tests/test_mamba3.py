from __future__ import annotations

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.nn.mamba3 import (
    DEFAULT_CHUNK_SIZE,
    Mamba3CacheState,
    Mamba3Config,
    Mamba3ReferenceBlock,
    _chunked_mamba3_diagonal_scan,
    _compute_trapezoidal_scale,
    causal_depthwise_conv1d,
    compute_mamba3_in_proj_dims,
    compute_num_rope_angles,
)


def _rand(shape: tuple[int, ...], seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape, dtype=np.float32))


def _use_mlx_gpu() -> None:
    assert mx.metal.is_available()
    mx.set_default_device(mx.Device(mx.gpu, 0))
    assert mx.default_device().type == mx.gpu


def _flat_params(model: nn.Module) -> dict[str, np.ndarray]:
    mx.eval(model.parameters())
    return {name: np.array(value) for name, value in tree_flatten(model.parameters())}


def _assert_close(actual: mx.array, expected: mx.array, *, atol: float = 1e-5) -> None:
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=atol, rtol=atol)


def _is_finite(x: mx.array) -> bool:
    return bool(np.isfinite(np.array(x.astype(mx.float32))).all())


def _tiny_config() -> Mamba3Config:
    return Mamba3Config(
        d_model=12,
        expand=2,
        headdim=4,
        d_state=6,
        ngroups=3,
        mimo_rank=2,
        is_mimo=True,
        d_conv=3,
        chunk_size=5,
        rope_fraction=0.5,
    )


def _larger_mimo_config() -> Mamba3Config:
    return Mamba3Config(
        d_model=16,
        expand=2,
        headdim=4,
        d_state=8,
        ngroups=4,
        mimo_rank=3,
        is_mimo=True,
        d_conv=4,
        chunk_size=9,
        rope_fraction=1.0,
    )


def _stress_mimo_config() -> Mamba3Config:
    return Mamba3Config(
        d_model=24,
        expand=2,
        headdim=8,
        d_state=8,
        ngroups=3,
        mimo_rank=2,
        is_mimo=True,
        d_conv=4,
        chunk_size=11,
        rope_fraction=1.0,
    )


def _loss_fn(model: Mamba3ReferenceBlock, x: mx.array, y: mx.array) -> mx.array:
    pred, _ = model(x)
    return mx.mean(mx.square(pred - y))


def _sequential_mamba3_diagonal_scan(
    log_decay: mx.array,
    inp: mx.array,
    C: mx.array,
    x: mx.array,
    z: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    h = h0
    outputs: list[mx.array] = []
    seq = inp.shape[1]
    for s in range(seq):
        h = mx.exp(log_decay[:, s]) * h + inp[:, s]
        y = mx.sum(h * C[:, s, :, None, :], axis=-1)
        y = y + D.astype(y.dtype) * x[:, s]
        outputs.append(nn.silu(z[:, s]) * y)
    if not outputs:
        return mx.zeros((inp.shape[0], 0, inp.shape[2], inp.shape[3]), dtype=inp.dtype), h
    return mx.stack(outputs, axis=1), h


def test_projection_dims_match_cppmega_layout() -> None:
    cfg = _tiny_config()
    dims = compute_mamba3_in_proj_dims(cfg)
    block = Mamba3ReferenceBlock(cfg)

    assert DEFAULT_CHUNK_SIZE == 128
    assert cfg.d_inner == 24
    assert cfg.nheads == 6
    assert dims.d_bc == 36
    assert dims.num_rope_angles == compute_num_rope_angles(cfg.d_state, cfg.rope_fraction)
    assert dims.split_sizes == [24, 24, 36, 36, 6, 6, 6, 1]
    assert dims.total == sum(dims.split_sizes)
    assert block.conv_weight.shape == (cfg.d_inner + 2 * dims.d_bc, cfg.d_conv, 1)
    assert block.conv_bias.shape == (cfg.d_inner + 2 * dims.d_bc,)
    assert block.D.shape == (cfg.nheads,)


def test_config_validation_fails_before_runtime_reshape_errors() -> None:
    with pytest.raises(ValueError, match="d_model"):
        Mamba3Config(d_model=0)
    with pytest.raises(ValueError, match="headdim"):
        Mamba3Config(d_model=10, expand=1, headdim=4)
    with pytest.raises(ValueError, match="ngroups"):
        Mamba3Config(d_model=12, expand=2, headdim=4, ngroups=5)
    with pytest.raises(ValueError, match="rope_fraction"):
        Mamba3Config(d_model=12, expand=2, headdim=4, d_state=6, rope_fraction=0.25)
    with pytest.raises(ValueError, match="dt_min"):
        Mamba3Config(d_model=12, expand=2, headdim=4, dt_min=0.2, dt_max=0.1)
    with pytest.raises(ValueError, match="dt_init_floor"):
        Mamba3Config(d_model=12, expand=2, headdim=4, dt_init_floor=0.2, dt_max=0.1)
    with pytest.raises(ValueError, match="A_floor"):
        Mamba3Config(d_model=12, expand=2, headdim=4, A_floor=0.0)


def test_reference_block_state_shapes_match_source_cache_contract() -> None:
    siso_cfg = Mamba3Config(
        d_model=32,
        expand=1,
        headdim=4,
        d_state=8,
        ngroups=2,
        mimo_rank=4,
        is_mimo=False,
        rope_fraction=1.0,
    )
    siso = Mamba3ReferenceBlock(siso_cfg)

    assert siso.mamba_state_shapes_per_request() == (
        (8, 4),
        (8, 4, 8),
        (1, 8, 8),
        (8, 4),
    )

    mimo_cfg = Mamba3Config(
        d_model=16,
        expand=2,
        headdim=4,
        d_state=8,
        ngroups=4,
        mimo_rank=3,
        is_mimo=True,
        rope_fraction=0.5,
    )
    mimo = Mamba3ReferenceBlock(mimo_cfg)

    assert mimo.mamba_state_shapes_per_request() == (
        (8, 2),
        (8, 4, 8),
        (3, 8, 8),
        (8, 4),
    )


def test_reference_block_zero_cache_and_return_cache_match_source_contract() -> None:
    _use_mlx_gpu()
    mx.random.seed(115)
    cfg = _tiny_config()
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 6, cfg.d_model), seed=94)
    cache = block.zero_cache_state(hidden.shape[0], dtype=hidden.dtype)

    out, returned = block(hidden, cache=cache, return_cache=True)
    mx.eval(out, returned.angle_dt, returned.ssm, returned.k, returned.v)

    assert out.shape == hidden.shape
    assert returned.angle_dt.shape == (2, cfg.nheads, block.dims.num_rope_angles)
    assert returned.ssm.shape == (2, cfg.nheads, cfg.headdim, cfg.d_state)
    assert returned.k.shape == (2, cfg.effective_mimo_rank, cfg.nheads, cfg.d_state)
    assert returned.v.shape == (2, cfg.nheads, cfg.headdim)
    assert returned.angle_dt.dtype == hidden.dtype
    assert returned.ssm.dtype == hidden.dtype
    assert returned.k.dtype == hidden.dtype
    assert returned.v.dtype == hidden.dtype
    assert np.isfinite(np.array(returned.angle_dt)).all()
    assert np.isfinite(np.array(returned.ssm)).all()
    assert np.isfinite(np.array(returned.k)).all()
    assert np.isfinite(np.array(returned.v)).all()
    assert np.max(np.abs(np.array(returned.angle_dt))) > 0
    assert np.max(np.abs(np.array(returned.ssm))) > 0
    assert np.max(np.abs(np.array(returned.k))) > 0
    assert np.max(np.abs(np.array(returned.v))) > 0


def test_reference_block_rejects_mismatched_initial_state_shape() -> None:
    _use_mlx_gpu()
    mx.random.seed(111)
    cfg = _tiny_config()
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 5, 12), seed=91)
    wrong_h0 = mx.zeros((2, cfg.nheads, cfg.headdim + 1, cfg.d_state))

    with pytest.raises(ValueError, match="h0 must have shape"):
        block(hidden, h0=wrong_h0)


def test_reference_block_rejects_mismatched_initial_state_dtype() -> None:
    _use_mlx_gpu()
    mx.random.seed(119)
    cfg = _tiny_config()
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 5, cfg.d_model), seed=101)
    wrong_h0 = mx.zeros(
        (2, cfg.nheads, cfg.headdim, cfg.d_state),
        dtype=mx.float16,
    )

    with pytest.raises(TypeError, match="h0 dtype"):
        block(hidden, h0=wrong_h0)


def test_reference_block_rejects_mismatched_cache_shape_dtype_or_duplicate_state() -> None:
    _use_mlx_gpu()
    mx.random.seed(116)
    cfg = _tiny_config()
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 5, cfg.d_model), seed=95)
    cache = block.zero_cache_state(hidden.shape[0], dtype=hidden.dtype)

    with pytest.raises(ValueError, match="cache.angle_dt must have shape"):
        block(
            hidden,
            cache=Mamba3CacheState(
                angle_dt=mx.zeros((2, cfg.nheads, block.dims.num_rope_angles + 1)),
                ssm=cache.ssm,
                k=cache.k,
                v=cache.v,
            ),
        )
    with pytest.raises(ValueError, match="cache.ssm must have dtype"):
        block(
            hidden,
            cache=Mamba3CacheState(
                angle_dt=cache.angle_dt,
                ssm=cache.ssm.astype(mx.bfloat16),
                k=cache.k,
                v=cache.v,
            ),
        )
    with pytest.raises(ValueError, match="either h0 or cache"):
        block(hidden, h0=cache.ssm, cache=cache)


def test_reference_block_initial_state_is_observable_and_differentiable() -> None:
    _use_mlx_gpu()
    mx.random.seed(114)
    cfg = _tiny_config()
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 6, cfg.d_model), seed=92)
    state_shape = (2, cfg.nheads, cfg.headdim, cfg.d_state)
    zero_h0 = mx.zeros(state_shape, dtype=hidden.dtype)
    seeded_h0 = 0.05 * _rand(state_shape, seed=93)

    implicit_out, implicit_state = block(hidden)
    zero_out, zero_state = block(hidden, h0=zero_h0)
    seeded_out, seeded_state = block(hidden, h0=seeded_h0)
    mx.eval(implicit_out, implicit_state, zero_out, zero_state, seeded_out, seeded_state)

    _assert_close(zero_out, implicit_out, atol=2e-5)
    _assert_close(zero_state, implicit_state, atol=2e-5)
    assert np.max(np.abs(np.array(seeded_out - zero_out))) > 0
    assert np.max(np.abs(np.array(seeded_state - zero_state))) > 0
    assert np.isfinite(np.array(seeded_out)).all()
    assert np.isfinite(np.array(seeded_state)).all()

    def loss_from_h0(h0: mx.array) -> mx.array:
        out, final_state = block(hidden, h0=h0)
        return mx.mean(mx.square(out)) + 0.01 * mx.mean(mx.square(final_state))

    loss, h0_grad = mx.value_and_grad(loss_from_h0)(seeded_h0)
    mx.eval(loss, h0_grad)
    grad_np = np.array(h0_grad)

    assert math.isfinite(float(loss.item()))
    assert h0_grad.shape == state_shape
    assert np.isfinite(grad_np).all()
    assert np.max(np.abs(grad_np)) > 0


def test_reference_block_cache_ssm_continuation_matches_h0_with_zero_angle_offset() -> None:
    _use_mlx_gpu()
    mx.random.seed(117)
    cfg = _tiny_config()
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 6, cfg.d_model), seed=96)
    state_shape = (2, cfg.nheads, cfg.headdim, cfg.d_state)
    seeded_h0 = 0.03 * _rand(state_shape, seed=97)
    cache = Mamba3CacheState(
        angle_dt=mx.zeros((2, cfg.nheads, block.dims.num_rope_angles), dtype=hidden.dtype),
        ssm=seeded_h0,
        k=0.02
        * _rand((2, cfg.effective_mimo_rank, cfg.nheads, cfg.d_state), seed=98),
        v=0.02 * _rand((2, cfg.nheads, cfg.headdim), seed=99),
    )

    out_h0, state_h0 = block(hidden, h0=seeded_h0)
    out_cache, state_cache = block(hidden, cache=cache)
    out_cache_returned, returned = block(hidden, cache=cache, return_cache=True)
    mx.eval(
        out_h0,
        state_h0,
        out_cache,
        state_cache,
        out_cache_returned,
        returned.angle_dt,
        returned.ssm,
        returned.k,
        returned.v,
    )

    _assert_close(out_cache, out_h0, atol=2e-5)
    _assert_close(state_cache, state_h0, atol=2e-5)
    _assert_close(out_cache_returned, out_h0, atol=2e-5)
    _assert_close(returned.ssm, state_h0, atol=2e-5)
    assert returned.k.shape == cache.k.shape
    assert returned.v.shape == cache.v.shape
    assert np.max(np.abs(np.array(returned.k - cache.k))) > 0
    assert np.max(np.abs(np.array(returned.v - cache.v))) > 0


def test_reference_block_preserves_hidden_shape() -> None:
    _use_mlx_gpu()
    mx.random.seed(101)
    block = Mamba3ReferenceBlock(_tiny_config())
    hidden = _rand((2, 7, 12), seed=7)

    out, state = block(hidden)
    mx.eval(out, state)

    assert out.shape == hidden.shape
    assert state.shape == (2, 6, 4, 6)
    assert np.isfinite(np.array(out)).all()
    assert np.isfinite(np.array(state)).all()


def test_reference_block_gpu_shape_and_dtype_contract() -> None:
    _use_mlx_gpu()
    mx.random.seed(102)
    cfg = _tiny_config()
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 7, 12), seed=21).astype(mx.bfloat16)

    out, state = block(hidden)
    mx.eval(out, state)

    assert out.shape == hidden.shape
    assert state.shape == (2, cfg.nheads, cfg.headdim, cfg.d_state)
    assert out.dtype == mx.float32
    assert state.dtype == mx.float32
    assert np.isfinite(np.array(out)).all()
    assert np.isfinite(np.array(state)).all()

    block.set_dtype(mx.bfloat16)
    out_bf16, state_bf16 = block(hidden)
    mx.eval(out_bf16, state_bf16)

    assert out_bf16.dtype == mx.bfloat16
    assert state_bf16.dtype == mx.bfloat16
    assert _is_finite(out_bf16)
    assert _is_finite(state_bf16)


def test_causal_depthwise_conv_does_not_look_right() -> None:
    x = _rand((1, 6, 4), seed=6)
    changed_suffix = mx.concatenate([x[:, :3], x[:, 3:] + 100.0], axis=1)
    weight = mx.ones((4, 3, 1))
    bias = mx.zeros((4,))

    base = causal_depthwise_conv1d(x, weight, bias)
    changed = causal_depthwise_conv1d(changed_suffix, weight, bias)
    mx.eval(base, changed)

    _assert_close(changed[:, :3], base[:, :3])


def test_trapezoidal_scale_uses_current_and_next_dt_without_full_like() -> None:
    dt = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    trap = mx.zeros_like(dt)

    scale = _compute_trapezoidal_scale(dt, trap)
    mx.eval(scale)

    expected = mx.array([[[2.0, 3.0], [4.0, 5.0], [2.5, 3.0]]])
    _assert_close(scale, expected)


def test_chunked_diagonal_scan_matches_sequential_reference_with_h0() -> None:
    _use_mlx_gpu()
    batch, seq, nheads, headdim, d_state = 2, 11, 3, 4, 5
    log_decay = -0.01 * mx.abs(_rand((batch, seq, nheads, 1, 1), seed=201))
    inp = 0.05 * _rand((batch, seq, nheads, headdim, d_state), seed=202)
    C = _rand((batch, seq, nheads, d_state), seed=203)
    x = _rand((batch, seq, nheads, headdim), seed=204)
    z = _rand((batch, seq, nheads, headdim), seed=205)
    D = _rand((nheads, headdim), seed=206)
    h0 = 0.1 * _rand((batch, nheads, headdim, d_state), seed=207)

    out_seq, h_seq = _sequential_mamba3_diagonal_scan(log_decay, inp, C, x, z, D, h0)
    out_chunk, h_chunk = _chunked_mamba3_diagonal_scan(
        log_decay,
        inp,
        C,
        x,
        z,
        D,
        h0,
        chunk_size=4,
    )
    mx.eval(out_seq, h_seq, out_chunk, h_chunk)

    _assert_close(out_chunk, out_seq, atol=2e-5)
    _assert_close(h_chunk, h_seq, atol=2e-5)


def test_chunked_diagonal_scan_accepts_source_style_per_head_d_skip() -> None:
    _use_mlx_gpu()
    batch, seq, nheads, headdim, d_state = 1, 7, 3, 4, 5
    log_decay = -0.01 * mx.abs(_rand((batch, seq, nheads, 1, 1), seed=241))
    inp = 0.04 * _rand((batch, seq, nheads, headdim, d_state), seed=242)
    C = _rand((batch, seq, nheads, d_state), seed=243)
    x = _rand((batch, seq, nheads, headdim), seed=244)
    z = _rand((batch, seq, nheads, headdim), seed=245)
    D_per_head = _rand((nheads,), seed=246)
    D_expanded = mx.broadcast_to(D_per_head[:, None], (nheads, headdim))
    h0 = 0.1 * _rand((batch, nheads, headdim, d_state), seed=247)

    out_source, h_source = _chunked_mamba3_diagonal_scan(
        log_decay, inp, C, x, z, D_per_head, h0, chunk_size=3
    )
    out_expanded, h_expanded = _chunked_mamba3_diagonal_scan(
        log_decay, inp, C, x, z, D_expanded, h0, chunk_size=3
    )
    mx.eval(out_source, h_source, out_expanded, h_expanded)

    _assert_close(out_source, out_expanded, atol=2e-5)
    _assert_close(h_source, h_expanded, atol=2e-5)


def test_chunked_diagonal_scan_large_chunk_matches_sequential_reference() -> None:
    _use_mlx_gpu()
    batch, seq, nheads, headdim, d_state = 1, 37, 2, 2, 3
    log_decay = -0.005 * mx.abs(_rand((batch, seq, nheads, 1, 1), seed=221))
    inp = 0.02 * _rand((batch, seq, nheads, headdim, d_state), seed=222)
    C = _rand((batch, seq, nheads, d_state), seed=223)
    x = _rand((batch, seq, nheads, headdim), seed=224)
    z = _rand((batch, seq, nheads, headdim), seed=225)
    D = _rand((nheads, headdim), seed=226)
    h0 = 0.1 * _rand((batch, nheads, headdim, d_state), seed=227)

    out_seq, h_seq = _sequential_mamba3_diagonal_scan(log_decay, inp, C, x, z, D, h0)
    out_chunk, h_chunk = _chunked_mamba3_diagonal_scan(
        log_decay,
        inp,
        C,
        x,
        z,
        D,
        h0,
        chunk_size=128,
    )
    mx.eval(out_seq, h_seq, out_chunk, h_chunk)

    _assert_close(out_chunk, out_seq, atol=3e-5)
    _assert_close(h_chunk, h_seq, atol=3e-5)


def test_chunked_diagonal_scan_is_chunk_size_invariant() -> None:
    _use_mlx_gpu()
    batch, seq, nheads, headdim, d_state = 1, 41, 2, 3, 4
    log_decay = -0.004 * mx.abs(_rand((batch, seq, nheads, 1, 1), seed=231))
    inp = 0.02 * _rand((batch, seq, nheads, headdim, d_state), seed=232)
    C = _rand((batch, seq, nheads, d_state), seed=233)
    x = _rand((batch, seq, nheads, headdim), seed=234)
    z = _rand((batch, seq, nheads, headdim), seed=235)
    D = _rand((nheads, headdim), seed=236)
    h0 = 0.1 * _rand((batch, nheads, headdim, d_state), seed=237)

    ref_out, ref_h = _chunked_mamba3_diagonal_scan(
        log_decay, inp, C, x, z, D, h0, chunk_size=1
    )
    for chunk_size in (2, 7, 31, 32, 128):
        out, h = _chunked_mamba3_diagonal_scan(
            log_decay, inp, C, x, z, D, h0, chunk_size=chunk_size
        )
        mx.eval(ref_out, ref_h, out, h)
        _assert_close(out, ref_out, atol=3e-5)
        _assert_close(h, ref_h, atol=3e-5)


def test_chunked_diagonal_scan_stitches_prefix_suffix() -> None:
    _use_mlx_gpu()
    batch, seq, nheads, headdim, d_state = 1, 9, 2, 3, 4
    log_decay = -0.02 * mx.abs(_rand((batch, seq, nheads, 1, 1), seed=211))
    inp = 0.03 * _rand((batch, seq, nheads, headdim, d_state), seed=212)
    C = _rand((batch, seq, nheads, d_state), seed=213)
    x = _rand((batch, seq, nheads, headdim), seed=214)
    z = _rand((batch, seq, nheads, headdim), seed=215)
    D = _rand((nheads, headdim), seed=216)
    h0 = 0.1 * _rand((batch, nheads, headdim, d_state), seed=217)
    split = 5

    full_out, full_h = _chunked_mamba3_diagonal_scan(
        log_decay, inp, C, x, z, D, h0, chunk_size=3
    )
    prefix_out, prefix_h = _chunked_mamba3_diagonal_scan(
        log_decay[:, :split],
        inp[:, :split],
        C[:, :split],
        x[:, :split],
        z[:, :split],
        D,
        h0,
        chunk_size=2,
    )
    suffix_out, suffix_h = _chunked_mamba3_diagonal_scan(
        log_decay[:, split:],
        inp[:, split:],
        C[:, split:],
        x[:, split:],
        z[:, split:],
        D,
        prefix_h,
        chunk_size=4,
    )
    stitched = mx.concatenate([prefix_out, suffix_out], axis=1)
    mx.eval(full_out, full_h, stitched, suffix_h)

    _assert_close(stitched, full_out, atol=2e-5)
    _assert_close(suffix_h, full_h, atol=2e-5)


def test_reference_block_prefix_matches_full_run_until_trapezoidal_lookahead() -> None:
    _use_mlx_gpu()
    mx.random.seed(103)
    block = Mamba3ReferenceBlock(_tiny_config())
    hidden = _rand((2, 8, 12), seed=8)
    prefix_len = 5

    full, _ = block(hidden)
    prefix, _ = block(hidden[:, :prefix_len])
    mx.eval(full, prefix)

    # Author Mamba3's trapezoidal K/B scale intentionally uses dt[t + 1].
    # The final prefix token has a different next token than the full run.
    _assert_close(full[:, : prefix_len - 1], prefix[:, : prefix_len - 1], atol=2e-5)
    assert np.max(np.abs(np.array(full[:, prefix_len - 1] - prefix[:, prefix_len - 1]))) > 0


def test_reference_block_lookahead_changes_only_prefix_boundary_token() -> None:
    _use_mlx_gpu()
    mx.random.seed(110)
    block = Mamba3ReferenceBlock(_tiny_config())
    hidden = _rand((2, 8, 12), seed=81)
    prefix_len = 5
    changed_next = hidden.at[:, prefix_len:].add(2.5)

    base, _ = block(hidden)
    changed, _ = block(changed_next)
    mx.eval(base, changed)

    # Causal conv plus trapezoidal dt[t + 1] should expose exactly one
    # lookahead-sensitive token before the changed suffix starts.
    _assert_close(base[:, : prefix_len - 1], changed[:, : prefix_len - 1], atol=2e-5)
    assert np.max(np.abs(np.array(base[:, prefix_len - 1] - changed[:, prefix_len - 1]))) > 0


def test_reference_block_cache_continuation_is_explicitly_not_full_split_equivalence() -> None:
    _use_mlx_gpu()
    mx.random.seed(118)
    cfg = _tiny_config()
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 8, cfg.d_model), seed=100)
    split = 5

    full_out, _ = block(hidden)
    prefix_out, prefix_cache = block(hidden[:, :split], return_cache=True)
    suffix_out, suffix_cache = block(hidden[:, split:], cache=prefix_cache, return_cache=True)
    h0_suffix_out, h0_suffix_state = block(hidden[:, split:], h0=prefix_cache.ssm)
    stitched = mx.concatenate([prefix_out, suffix_out], axis=1)
    mx.eval(
        full_out,
        prefix_out,
        prefix_cache.angle_dt,
        suffix_out,
        suffix_cache.angle_dt,
        suffix_cache.ssm,
        h0_suffix_out,
        h0_suffix_state,
        stitched,
    )

    # The local cache continues source-shaped angle + SSM state. It deliberately
    # does not carry the causal-conv tail or trapezoidal boundary lookahead, so
    # arbitrary split equality against a full prompt remains unsupported.
    assert suffix_cache.ssm.shape == h0_suffix_state.shape
    assert np.isfinite(np.array(suffix_cache.ssm)).all()
    assert np.isfinite(np.array(h0_suffix_state)).all()
    assert np.max(np.abs(np.array(suffix_cache.ssm - h0_suffix_state))) > 0
    assert np.max(np.abs(np.array(suffix_out - h0_suffix_out))) > 0
    assert np.max(np.abs(np.array(suffix_cache.angle_dt - prefix_cache.angle_dt))) > 0
    assert np.max(np.abs(np.array(stitched - full_out))) > 0


def test_reference_block_bc_channels_are_conv_sensitive() -> None:
    _use_mlx_gpu()
    mx.random.seed(108)
    cfg = _tiny_config()
    dims = compute_mamba3_in_proj_dims(cfg)
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 6, 12), seed=61)

    base_out, base_state = block(hidden)
    x_end = cfg.d_inner
    b_end = x_end + dims.d_bc
    c_end = b_end + dims.d_bc
    conv_bias = mx.array(block.conv_bias)
    conv_bias = conv_bias.at[x_end:c_end].add(0.25)
    block.conv_bias = conv_bias
    changed_out, changed_state = block(hidden)
    mx.eval(base_out, base_state, changed_out, changed_state)

    assert base_out.shape == changed_out.shape == hidden.shape
    assert base_state.shape == changed_state.shape
    assert np.max(np.abs(np.array(changed_state - base_state))) > 0
    assert np.max(np.abs(np.array(changed_out - base_out))) > 0


def test_reference_block_larger_mimo_runtime_and_projection_slice_gradients() -> None:
    _use_mlx_gpu()
    mx.random.seed(112)
    cfg = _larger_mimo_config()
    dims = compute_mamba3_in_proj_dims(cfg)
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 17, cfg.d_model), seed=121)
    target = _rand((2, 17, cfg.d_model), seed=122)

    out, state = block(hidden)
    loss, grads = nn.value_and_grad(block, _loss_fn)(block, hidden, target)
    mx.eval(out, state, loss, grads)
    flat_grads = dict(tree_flatten(grads))

    assert out.shape == hidden.shape
    assert state.shape == (2, cfg.nheads, cfg.headdim, cfg.d_state)
    assert state.dtype == mx.float32
    assert math.isfinite(float(loss.item()))
    assert _is_finite(out)
    assert _is_finite(state)
    assert flat_grads["D"].shape == (cfg.nheads,)

    start = 0
    in_proj_grad = np.array(flat_grads["in_proj.weight"])
    for name, size in zip(
        ("z", "x", "B", "C", "dd_dt", "dd_A", "trap", "angles"),
        dims.split_sizes,
        strict=True,
    ):
        grad_slice = in_proj_grad[start : start + size]
        assert np.isfinite(grad_slice).all(), name
        assert np.max(np.abs(grad_slice)) > 0, name
        start += size
    assert start == dims.total

    for name in (
        "conv_weight",
        "conv_bias",
        "dt_bias",
        "B_norm_weight",
        "C_norm_weight",
        "B_bias",
        "C_bias",
        "D",
        "out_proj.weight",
    ):
        grad = np.array(flat_grads[name])
        assert np.isfinite(grad).all(), name
        assert np.max(np.abs(grad)) > 0, name


def test_reference_block_larger_compiled_train_step_matches_eager_state_and_gradients() -> None:
    _use_mlx_gpu()
    cfg = _stress_mimo_config()
    dims = compute_mamba3_in_proj_dims(cfg)
    hidden = _rand((2, 25, cfg.d_model), seed=151)
    target = _rand((2, 25, cfg.d_model), seed=152)
    state_target = _rand((2, cfg.nheads, cfg.headdim, cfg.d_state), seed=153)

    def stress_loss(
        model: Mamba3ReferenceBlock,
        x: mx.array,
        y: mx.array,
        h_target: mx.array,
    ) -> mx.array:
        pred, final_state = model(x)
        return mx.mean(mx.square(pred - y)) + 0.02 * mx.mean(mx.square(final_state - h_target))

    mx.random.seed(113)
    eager_block = Mamba3ReferenceBlock(cfg)
    mx.eval(eager_block.parameters())
    mx.random.seed(113)
    compiled_block = Mamba3ReferenceBlock(cfg)
    optimizer = optim.Adam(learning_rate=5e-3)
    before_params = _flat_params(compiled_block)

    eager_out, eager_state = eager_block(hidden)
    eager_loss, eager_grads = nn.value_and_grad(eager_block, stress_loss)(
        eager_block,
        hidden,
        target,
        state_target,
    )
    mx.eval(eager_out, eager_state, eager_loss, eager_grads)

    loss_and_grad = nn.value_and_grad(compiled_block, stress_loss)
    captured_state = [compiled_block.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=captured_state, outputs=captured_state)
    def step(x: mx.array, y: mx.array, h_target: mx.array):
        pred, final_state = compiled_block(x)
        loss, grads = loss_and_grad(compiled_block, x, y, h_target)
        optimizer.update(compiled_block, grads)
        return loss, grads, pred, final_state

    compiled_loss, compiled_grads, compiled_out, compiled_state = step(hidden, target, state_target)
    mx.eval(captured_state, compiled_loss, compiled_grads, compiled_out, compiled_state)
    after_params = _flat_params(compiled_block)

    assert compiled_out.shape == hidden.shape
    assert compiled_state.shape == state_target.shape
    assert math.isfinite(float(compiled_loss.item()))
    _assert_close(compiled_loss, eager_loss, atol=5e-4)
    _assert_close(compiled_out, eager_out, atol=5e-4)
    _assert_close(compiled_state, eager_state, atol=5e-4)

    eager_flat_grads = dict(tree_flatten(eager_grads))
    compiled_flat_grads = dict(tree_flatten(compiled_grads))
    for name in ("conv_weight", "dt_bias", "D", "out_proj.weight"):
        compiled_grad = compiled_flat_grads[name]
        grad_np = np.array(compiled_grad)
        assert np.isfinite(grad_np).all(), name
        assert np.max(np.abs(grad_np)) > 0, name
        _assert_close(compiled_grad, eager_flat_grads[name], atol=8e-4)
        assert np.max(np.abs(after_params[name] - before_params[name])) > 0, name

    start = 0
    in_proj_grad = np.array(compiled_flat_grads["in_proj.weight"])
    for name, size in zip(
        ("z", "x", "B", "C", "dd_dt", "dd_A", "trap", "angles"),
        dims.split_sizes,
        strict=True,
    ):
        grad_slice = in_proj_grad[start : start + size]
        assert np.isfinite(grad_slice).all(), name
        assert np.max(np.abs(grad_slice)) > 0, name
        start += size
    assert start == dims.total


def test_reference_block_trains_with_finite_gradients() -> None:
    _use_mlx_gpu()
    mx.random.seed(104)
    block = Mamba3ReferenceBlock(_tiny_config())
    optimizer = optim.Adam(learning_rate=1e-2)
    hidden = _rand((2, 6, 12), seed=11)
    target = _rand((2, 6, 12), seed=12)
    before = _flat_params(block)

    loss_and_grad = nn.value_and_grad(block, _loss_fn)
    loss, grads = loss_and_grad(block, hidden, target)
    for _, grad in tree_flatten(grads):
        assert np.isfinite(np.array(grad)).all()
    optimizer.update(block, grads)
    mx.eval(block.parameters(), optimizer.state, loss)
    after = _flat_params(block)

    assert math.isfinite(float(loss.item()))
    assert any(np.max(np.abs(after[name] - before[name])) > 0 for name in before)


def test_reference_block_compiled_train_step_updates_key_parameters() -> None:
    _use_mlx_gpu()
    mx.random.seed(109)
    block = Mamba3ReferenceBlock(_tiny_config())
    optimizer = optim.Adam(learning_rate=1e-2)
    hidden = _rand((2, 6, 12), seed=71)
    target = _rand((2, 6, 12), seed=72)
    before_params = _flat_params(block)
    loss_and_grad = nn.value_and_grad(block, _loss_fn)
    captured_state = [block.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=captured_state, outputs=captured_state)
    def step(x: mx.array, y: mx.array):
        loss, grads = loss_and_grad(block, x, y)
        optimizer.update(block, grads)
        return loss, grads

    loss, grads = step(hidden, target)
    mx.eval(captured_state, loss, grads)
    after_params = _flat_params(block)
    flat_grads = dict(tree_flatten(grads))

    assert math.isfinite(float(loss.item()))
    for name in (
        "in_proj.weight",
        "out_proj.weight",
        "conv_weight",
        "dt_bias",
        "B_norm_weight",
        "C_norm_weight",
        "B_bias",
        "C_bias",
        "D",
    ):
        grad = np.array(flat_grads[name])
        assert np.isfinite(grad).all(), name
        assert np.max(np.abs(grad)) > 0, name
        assert np.max(np.abs(after_params[name] - before_params[name])) > 0, name


def test_reference_block_key_parameters_receive_nonzero_gradients() -> None:
    _use_mlx_gpu()
    mx.random.seed(105)
    block = Mamba3ReferenceBlock(_tiny_config())
    hidden = _rand((2, 6, 12), seed=31)
    target = _rand((2, 6, 12), seed=32)

    loss, grads = nn.value_and_grad(block, _loss_fn)(block, hidden, target)
    mx.eval(loss, grads)
    flat_grads = dict(tree_flatten(grads))

    assert math.isfinite(float(loss.item()))
    for name in (
        "in_proj.weight",
        "out_proj.weight",
        "conv_weight",
        "conv_bias",
        "dt_bias",
        "B_norm_weight",
        "C_norm_weight",
        "B_bias",
        "C_bias",
        "D",
    ):
        grad = np.array(flat_grads[name])
        assert np.isfinite(grad).all(), name
        assert np.max(np.abs(grad)) > 0, name


def test_reference_block_short_train_step_changes_loss_and_representative_params() -> None:
    _use_mlx_gpu()
    mx.random.seed(106)
    block = Mamba3ReferenceBlock(_tiny_config())
    optimizer = optim.Adam(learning_rate=1e-2)
    hidden = _rand((2, 6, 12), seed=41)
    target = _rand((2, 6, 12), seed=42)
    before_params = _flat_params(block)

    loss_and_grad = nn.value_and_grad(block, _loss_fn)
    before_loss, grads = loss_and_grad(block, hidden, target)
    optimizer.update(block, grads)
    after_loss = _loss_fn(block, hidden, target)
    mx.eval(before_loss, after_loss, block.parameters(), optimizer.state)
    after_params = _flat_params(block)

    before_loss_value = float(before_loss.item())
    after_loss_value = float(after_loss.item())
    assert math.isfinite(before_loss_value)
    assert math.isfinite(after_loss_value)
    assert after_loss_value < before_loss_value
    for name in ("in_proj.weight", "out_proj.weight", "conv_weight", "dt_bias", "D"):
        assert np.max(np.abs(after_params[name] - before_params[name])) > 0, name


def test_reference_block_handles_sequence_tail_shorter_than_chunk_size() -> None:
    _use_mlx_gpu()
    mx.random.seed(107)
    cfg = Mamba3Config(
        d_model=12,
        expand=2,
        headdim=4,
        d_state=6,
        ngroups=3,
        mimo_rank=2,
        is_mimo=True,
        d_conv=3,
        chunk_size=5,
        rope_fraction=0.5,
    )
    assert 7 % cfg.chunk_size != 0
    block = Mamba3ReferenceBlock(cfg)
    hidden = _rand((2, 7, 12), seed=51)
    target = _rand((2, 7, 12), seed=52)

    loss, grads = nn.value_and_grad(block, _loss_fn)(block, hidden, target)
    out, state = block(hidden)
    mx.eval(loss, grads, out, state)

    assert out.shape == hidden.shape
    assert state.shape == (2, cfg.nheads, cfg.headdim, cfg.d_state)
    assert math.isfinite(float(loss.item()))
    assert np.isfinite(np.array(out)).all()
    assert np.isfinite(np.array(state)).all()
    assert np.max(np.abs(np.array(dict(tree_flatten(grads))["in_proj.weight"]))) > 0
