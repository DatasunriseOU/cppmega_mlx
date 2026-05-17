"""End-to-end tests for the M2RNN production dispatcher.

These tests verify that :class:`cppmega_mlx.nn.m2rnn.M2RNNMixer` honors the
:class:`cppmega_mlx.runtime.kernel_policy.KernelPath` selection and records
the actual kernel used into the dispatch log. M2RNN direct-MSL Path B is
retired; AUTO uses the correctness-first reference route while explicit
PATH_C exercises the TileLang route.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn.m2rnn import M2RNNConfig, M2RNNMixer, M2RNNMixerState
from cppmega_mlx.runtime.kernel_policy import (
    KernelPath,
    clear_dispatch_log,
    get_dispatch_log,
    selected_path,
)


_METAL_AVAILABLE = mx.metal.is_available()


def _make_block(seed: int = 0) -> tuple[M2RNNMixer, mx.array]:
    mx.random.seed(seed)
    cfg = M2RNNConfig(
        d_model=16,
        k_head_dim=4,
        v_head_dim=3,
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=2,
        num_f_heads=2,
        num_g_heads=2,
        num_weight_heads=1,
        conv_kernel=4,
        chunk_size=8,
    )
    block = M2RNNMixer(cfg)
    hidden = mx.random.normal((1, 8, cfg.d_model), dtype=mx.float32) * 0.1
    return block, hidden


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch):
    clear_dispatch_log()
    monkeypatch.delenv("CPPMEGA_KERNEL_PATH", raising=False)
    monkeypatch.delenv("CPPMEGA_KERNEL_PATH__M2RNN", raising=False)
    yield
    clear_dispatch_log()


def test_default_auto_dispatches_without_path_b() -> None:
    block, hidden = _make_block()
    assert selected_path("m2rnn") is KernelPath.AUTO
    out, _ = block(hidden)
    mx.eval(out)
    log = get_dispatch_log()
    matches = [e for e in log if e["op_name"] == "m2rnn"]
    assert matches, f"no m2rnn dispatch recorded: {log}"
    assert matches[-1]["path"] == "auto"
    assert matches[-1]["kernel_used"] in {
        "path_c_tilelang_dsl_packed_post",
        "reference_pure_mlx",
    }


def test_reference_policy_forces_pure_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "ref")
    block, hidden = _make_block()
    out, _ = block(hidden)
    mx.eval(out)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert matches[-1]["path"] == "ref"
    assert matches[-1]["kernel_used"] == "reference_pure_mlx"


def test_path_b_policy_fails_closed_after_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_b")
    block, hidden = _make_block()
    with pytest.raises(RuntimeError, match="Path B direct-MSL kernel is retired"):
        block(hidden)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert not matches


def test_path_c_dispatches_grouped_heads_without_path_b_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit Path C lowers grouped production heads inside TileLang."""

    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    pytest.importorskip("tilelang")
    from cppmega_mlx.nn._tilelang import m2rnn as m2rnn_path_b
    from cppmega_mlx.nn._tilelang.m2rnn_path_c import m2rnn_path_c_status

    status = m2rnn_path_c_status()
    if not status.available:
        pytest.skip(f"m2rnn Path C unavailable on this host: {status.reason}")

    def fail_path_b(*_args: object, **_kwargs: object) -> tuple[mx.array, mx.array]:
        raise AssertionError("Path C dispatch fell back to Path B")

    monkeypatch.setattr(m2rnn_path_b, "m2rnn_apply_with_state", fail_path_b)
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    block, hidden = _make_block()
    h0 = block.initial_h0(hidden.shape[0], hidden.dtype)
    out, h = block(hidden, h0=h0)
    mx.eval(out, h)
    assert out.shape == hidden.shape
    assert h.shape == (
        1,
        block.config.num_heads,
        block.config.k_head_dim,
        block.config.v_head_dim,
    )
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert matches[-1]["path"] == "path_c"
    assert matches[-1]["kernel_used"] == "path_c_tilelang_dsl_packed_post"


def test_path_c_dispatch_requires_existing_h0_without_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega_mlx.nn._tilelang import m2rnn_path_c

    def fail_path_c_status(*_args: object, **_kwargs: object) -> m2rnn_path_c.M2RNNPathCStatus:
        raise AssertionError("Path C status should not run before explicit h0 exists")

    def fail_path_c_kernel(*_args: object, **_kwargs: object) -> tuple[mx.array, mx.array]:
        raise AssertionError("Path C should fail before allocating h0 or launching")

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_mapped_packed_path_c_status",
        fail_path_c_status,
    )
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_apply_mapped_packed_with_state_path_c",
        fail_path_c_kernel,
    )
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    cfg = M2RNNConfig(
        d_model=16,
        k_head_dim=4,
        v_head_dim=4,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=2,
        num_f_heads=2,
        num_g_heads=2,
        num_weight_heads=2,
        conv_kernel=1,
        chunk_size=8,
    )
    block = M2RNNMixer(cfg)
    hidden = mx.random.normal((1, 4, cfg.d_model), dtype=mx.float32) * 0.1

    with pytest.raises(RuntimeError, match="existing h0 tensor"):
        block(hidden)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert not matches, f"failed Path C dispatch should not log success: {matches}"


def test_path_c_dispatch_accepts_explicit_initial_h0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega_mlx.nn._tilelang import m2rnn_path_c

    seen: dict[str, tuple[int, ...]] = {}

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_mapped_packed_path_c_status",
        lambda *_args, **_kwargs: m2rnn_path_c.M2RNNPathCStatus(
            True,
            "forced packed recurrence available",
        ),
    )

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_post_residual_gate_path_c_status",
        lambda *_args, **_kwargs: m2rnn_path_c.M2RNNPathCStatus(
            True,
            "forced post available",
        ),
    )

    def fake_recurrence_kernel(
        conv_input: mx.array,
        W: mx.array,
        xf: mx.array,
        h0: mx.array,
        **_kwargs: object,
    ) -> tuple[mx.array, mx.array]:
        batch, seq, _conv_dim = conv_input.shape
        heads = h0.shape[1]
        v_dim = W.shape[-1]
        seen["conv_input"] = tuple(conv_input.shape)
        seen["xf"] = tuple(xf.shape)
        seen["h0"] = tuple(h0.shape)
        return mx.zeros((batch, seq, heads * v_dim), dtype=conv_input.dtype), h0

    def fake_post_kernel(
        y: mx.array,
        conv_input: mx.array,
        D: mx.array,
        projected: mx.array,
        **_kwargs: object,
    ) -> mx.array:
        seen["D"] = tuple(D.shape)
        seen["projected"] = tuple(projected.shape)
        return y

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_apply_mapped_packed_with_state_path_c",
        fake_recurrence_kernel,
    )
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_apply_post_residual_gate_path_c",
        fake_post_kernel,
    )
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    cfg = M2RNNConfig(
        d_model=16,
        k_head_dim=4,
        v_head_dim=4,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=2,
        num_f_heads=2,
        num_g_heads=2,
        num_weight_heads=2,
        conv_kernel=1,
        chunk_size=8,
    )
    mx.random.seed(17)
    block = M2RNNMixer(cfg)
    hidden = mx.random.normal((1, 4, cfg.d_model), dtype=mx.float32) * 0.1
    h0 = block.initial_h0(hidden.shape[0], hidden.dtype)
    out, h = block(hidden, h0=h0)
    mx.eval(out, h)
    assert out.shape == hidden.shape
    assert seen["conv_input"] == (1, 4, 24)
    assert seen["xf"] == (1, 4, 2)
    assert seen["h0"] == (1, 2, 4, 4)
    assert seen["D"] == (2, 4)
    assert seen["projected"] == (1, 4, 34)
    assert h.shape == seen["h0"]
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert matches[-1]["path"] == "path_c"
    assert matches[-1]["kernel_used"] == "path_c_tilelang_dsl_packed_post"


def test_path_c_dispatch_uses_tilelang_post_without_v_g_broadcast_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega_mlx.nn import m2rnn as m2rnn_mod
    from cppmega_mlx.nn._tilelang import m2rnn_path_c

    seen: dict[str, tuple[int, ...] | int] = {}

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_mapped_packed_path_c_status",
        lambda *_args, **_kwargs: m2rnn_path_c.M2RNNPathCStatus(
            True,
            "forced packed recurrence available",
        ),
    )
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_post_residual_gate_path_c_status",
        lambda *_args, **_kwargs: m2rnn_path_c.M2RNNPathCStatus(
            True,
            "forced post available",
        ),
    )

    def fake_recurrence(
        conv_input: mx.array,
        W: mx.array,
        xf: mx.array,
        h0: mx.array,
        **kwargs: object,
    ) -> tuple[mx.array, mx.array]:
        batch, seq, _conv_dim = conv_input.shape
        seen["conv_input"] = tuple(conv_input.shape)
        seen["W"] = tuple(W.shape)
        seen["xf"] = tuple(xf.shape)
        seen["h0"] = tuple(h0.shape)
        seen["recurrence_v_heads"] = int(kwargs["v_heads"])
        return mx.zeros((batch, seq, h0.shape[1], h0.shape[-1]), dtype=conv_input.dtype), h0

    def fake_post(
        y: mx.array,
        conv_input: mx.array,
        D: mx.array,
        projected: mx.array,
        **kwargs: object,
    ) -> mx.array:
        seen["D"] = tuple(D.shape)
        seen["projected"] = tuple(projected.shape)
        seen["post_v_heads"] = int(kwargs["v_heads"])
        seen["post_g_heads"] = int(kwargs["g_heads"])
        return y.reshape(y.shape[0], y.shape[1], -1)

    def fail_broadcast(*_args: object, **_kwargs: object) -> mx.array:
        raise AssertionError("Path C post route must not materialize v head broadcast in MLX")

    def fail_repeat(*_args: object, **_kwargs: object) -> mx.array:
        raise AssertionError("Path C post route must not materialize g repeat in MLX")

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_apply_mapped_packed_with_state_path_c",
        fake_recurrence,
    )
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_apply_post_residual_gate_path_c",
        fake_post,
    )
    monkeypatch.setattr(m2rnn_mod, "_broadcast_heads", fail_broadcast)
    monkeypatch.setattr(m2rnn_mod.mx, "repeat", fail_repeat)
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")

    cfg = M2RNNConfig(
        d_model=16,
        k_head_dim=4,
        v_head_dim=3,
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=2,
        num_f_heads=4,
        num_g_heads=2,
        num_weight_heads=1,
        conv_kernel=1,
        chunk_size=8,
    )
    mx.random.seed(31)
    block = M2RNNMixer(cfg)
    hidden = mx.random.normal((1, 4, cfg.d_model), dtype=mx.float32) * 0.1
    h0 = block.initial_h0(hidden.shape[0], hidden.dtype)

    out, h = block(hidden, h0=h0)
    mx.eval(out, h)

    assert out.shape == hidden.shape
    assert h.shape == h0.shape
    assert seen["conv_input"] == (1, 4, 14)
    assert seen["W"] == (1, 3, 3)
    assert seen["xf"] == (1, 4, 4)
    assert seen["h0"] == (1, 4, 4, 3)
    assert seen["D"] == (4, 3)
    assert seen["projected"] == (1, 4, 24)
    assert seen["recurrence_v_heads"] == 2
    assert seen["post_v_heads"] == 2
    assert seen["post_g_heads"] == 2
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert matches[-1]["path"] == "path_c"
    assert matches[-1]["kernel_used"] == "path_c_tilelang_dsl_packed_post"


def test_path_c_transform_dlpack_boundary_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega_mlx.nn._tilelang import m2rnn_path_c

    def fail_inside_mlx_transform(*_args: object, **_kwargs: object) -> tuple[mx.array, mx.array]:
        raise RuntimeError(
            "MLX array import failed: [eval] Attempting to eval an array during "
            "function transformations like compile or vmap is not allowed."
        )

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_mapped_packed_path_c_status",
        lambda *_args, **_kwargs: m2rnn_path_c.M2RNNPathCStatus(
            True,
            "forced packed recurrence available",
        ),
    )
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_apply_mapped_packed_with_state_path_c",
        fail_inside_mlx_transform,
    )
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")

    cfg = M2RNNConfig(
        d_model=16,
        k_head_dim=4,
        v_head_dim=4,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=2,
        num_f_heads=2,
        num_g_heads=2,
        num_weight_heads=2,
        conv_kernel=1,
        chunk_size=8,
    )
    mx.random.seed(19)
    block = M2RNNMixer(cfg)
    hidden = mx.random.normal((1, 4, cfg.d_model), dtype=mx.float32) * 0.1
    h0 = block.initial_h0(hidden.shape[0], hidden.dtype)

    with pytest.raises(RuntimeError, match="MLX array import failed"):
        block(hidden, h0=h0)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert not matches, f"failed Path C dispatch should not log success: {matches}"


def test_path_c_dispatch_checks_packed_status_before_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega_mlx.nn._tilelang import m2rnn_path_c

    seen: dict[str, object] = {}

    def forced_unavailable(
        conv_input: mx.array,
        *_args: object,
        **_kwargs: object,
    ) -> m2rnn_path_c.M2RNNPathCStatus:
        seen["shape"] = tuple(conv_input.shape)
        seen["dtype"] = conv_input.dtype
        return m2rnn_path_c.M2RNNPathCStatus(False, "forced K=16 bf16 unavailable")

    def fail_path_c_kernel(*_args: object, **_kwargs: object) -> tuple[mx.array, mx.array]:
        raise AssertionError("Path C callable must not run after unavailable status")

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_mapped_packed_path_c_status",
        forced_unavailable,
    )
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_apply_mapped_packed_with_state_path_c",
        fail_path_c_kernel,
    )
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    cfg = M2RNNConfig(
        d_model=16,
        k_head_dim=16,
        v_head_dim=4,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=2,
        num_f_heads=2,
        num_g_heads=2,
        num_weight_heads=2,
        conv_kernel=1,
        chunk_size=8,
    )
    block = M2RNNMixer(cfg)
    block.set_dtype(mx.bfloat16)
    hidden = (mx.random.normal((1, 4, cfg.d_model)) * 0.1).astype(mx.bfloat16)
    h0 = block.initial_h0(hidden.shape[0], hidden.dtype)

    with pytest.raises(RuntimeError, match="forced K=16 bf16 unavailable"):
        block(hidden, h0=h0)
    assert seen["shape"] == (1, 4, cfg.num_heads * (2 * cfg.k_head_dim + cfg.v_head_dim))
    assert seen["dtype"] == mx.bfloat16
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert not matches, f"failed Path C dispatch should not log success: {matches}"


def test_path_c_dispatch_uses_packed_state_without_qkv_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    pytest.importorskip("tilelang")
    from cppmega_mlx.nn._tilelang.m2rnn_path_c import m2rnn_path_c_status

    status = m2rnn_path_c_status()
    if not status.available:
        pytest.skip(f"m2rnn Path C unavailable on this host: {status.reason}")

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    cfg = M2RNNConfig(
        d_model=16,
        k_head_dim=4,
        v_head_dim=4,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=2,
        num_f_heads=2,
        num_g_heads=2,
        num_weight_heads=2,
        conv_kernel=1,
        chunk_size=8,
    )
    mx.random.seed(13)
    block = M2RNNMixer(cfg)
    hidden = mx.random.normal((1, 4, cfg.d_model), dtype=mx.float32) * 0.1
    mixer_state = M2RNNMixerState(
        h=mx.zeros((1, cfg.num_heads, cfg.k_head_dim, cfg.v_head_dim), dtype=mx.float32),
        conv_state=mx.zeros((1, cfg.conv_kernel - 1, block.conv_dim), dtype=mx.float32),
    )

    out, next_state = block(hidden, mixer_state=mixer_state, return_state=True)
    mx.eval(out, next_state.h, next_state.conv_state)
    assert out.shape == hidden.shape
    assert next_state.h.shape == mixer_state.h.shape
    assert next_state.conv_state.shape == mixer_state.conv_state.shape
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert matches[-1]["path"] == "path_c"
    assert matches[-1]["kernel_used"] == "path_c_tilelang_dsl_packed_post"


def test_block_grad_flows_through_path_c_with_explicit_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    pytest.importorskip("tilelang")
    from mlx.utils import tree_flatten
    from cppmega_mlx.nn._tilelang.m2rnn_path_c import m2rnn_path_c_status

    status = m2rnn_path_c_status()
    if not status.available:
        pytest.skip(f"m2rnn Path C unavailable on this host: {status.reason}")

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    cfg = M2RNNConfig(
        d_model=16,
        k_head_dim=4,
        v_head_dim=4,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=2,
        num_f_heads=2,
        num_g_heads=2,
        num_weight_heads=2,
        conv_kernel=1,
        chunk_size=8,
    )
    mx.random.seed(14)
    block = M2RNNMixer(cfg)
    hidden = mx.random.normal((1, 4, cfg.d_model), dtype=mx.float32) * 0.1
    mixer_state = M2RNNMixerState(
        h=mx.zeros((1, cfg.num_heads, cfg.k_head_dim, cfg.v_head_dim), dtype=mx.float32),
        conv_state=mx.zeros((1, cfg.conv_kernel - 1, block.conv_dim), dtype=mx.float32),
    )

    def loss_fn(params, hidden_):
        block.update(params)
        out, _state = block(hidden_, mixer_state=mixer_state, return_state=True)
        return mx.mean(out * out)

    loss, grads = mx.value_and_grad(loss_fn)(block.trainable_parameters(), hidden)
    mx.eval(loss, grads)
    flat_grads = tree_flatten(grads)
    assert flat_grads, "expected at least one trainable parameter"
    for _, grad in flat_grads:
        assert np.isfinite(np.array(grad)).all()


def test_per_op_override_selects_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "auto")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__M2RNN", "ref")
    block, hidden = _make_block()
    out, _ = block(hidden)
    mx.eval(out)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert matches[-1]["kernel_used"] == "reference_pure_mlx"


def test_block_grad_flows_through_auto_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlx.utils import tree_flatten

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "auto")
    block, hidden = _make_block(seed=2)

    def loss_fn(params, hidden_):
        block.update(params)
        out, _ = block(hidden_)
        return mx.mean(out * out)

    params = block.trainable_parameters()
    grad_fn = mx.value_and_grad(loss_fn, argnums=0)
    loss, grads = grad_fn(params, hidden)
    mx.eval(loss, grads)
    flat_grads = tree_flatten(grads)
    assert flat_grads, "expected at least one trainable parameter"
    for _, g in flat_grads:
        arr = np.array(g)
        assert np.all(np.isfinite(arr))


def test_block_auto_matches_reference_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward equality between AUTO and the pure-MLX reference."""

    block, hidden = _make_block(seed=4)

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "ref")
    out_ref, h_ref = block(hidden)
    mx.eval(out_ref, h_ref)

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "auto")
    out_auto, h_auto = block(hidden)
    mx.eval(out_auto, h_auto)

    np.testing.assert_allclose(
        np.array(out_auto).astype(np.float32),
        np.array(out_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(h_auto).astype(np.float32),
        np.array(h_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_return_state_routes_through_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache assembly path also picks up h_last from the selected route."""

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "auto")
    block, hidden = _make_block(seed=5)
    out, mixer_state = block(hidden, return_state=True)
    mx.eval(out, mixer_state.h, mixer_state.conv_state)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert matches[-1]["kernel_used"] in {
        "path_c_tilelang_dsl_packed_post",
        "reference_pure_mlx",
    }
    cfg = block.config
    assert mixer_state.h.shape == (
        1, cfg.num_heads, cfg.k_head_dim, cfg.v_head_dim,
    )


def test_hybrid_lm_e2e_with_r_block_trains_loss_decreases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny HybridLM with an R block trains for 5 steps; loss should decrease.

    Verifies the dispatch log shows m2rnn avoided the retired Path B route.
    """

    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")

    try:
        from cppmega_mlx.models.hybrid_lm import HybridTinyLM, HybridTinyConfig
    except ValueError as exc:
        if "Unable to compare versions" in str(exc):
            pytest.skip(f"broken package version metadata in this venv: {exc}")
        raise
    import mlx.optimizers as optim

    monkeypatch.delenv("CPPMEGA_KERNEL_PATH", raising=False)
    mx.random.seed(99)
    cfg = HybridTinyConfig(
        vocab_size=64,
        hidden_size=16,
        pattern="R",  # single R block per layer
        depth=2,
    )
    model = HybridTinyLM(cfg)

    batch_size, seq_len = 2, 8
    input_ids = mx.random.randint(0, cfg.vocab_size, (batch_size, seq_len))
    targets = mx.random.randint(0, cfg.vocab_size, (batch_size, seq_len))

    import mlx.nn as nn

    def loss_fn(params, input_ids_, targets_):
        model.update(params)
        logits = model(input_ids_)
        return mx.mean(nn.losses.cross_entropy(logits, targets_, reduction="none"))

    optimizer = optim.SGD(learning_rate=0.05)
    grad_fn = mx.value_and_grad(loss_fn, argnums=0)
    losses: list[float] = []
    params = model.trainable_parameters()
    for _ in range(5):
        loss, grads = grad_fn(params, input_ids, targets)
        params = optimizer.apply_gradients(grads, params)
        mx.eval(params, loss)
        losses.append(float(loss))

    assert all(np.isfinite(losses)), f"non-finite loss: {losses}"
    # Final loss should be lower than initial (or equal — we use a very small lr).
    assert losses[-1] <= losses[0] + 1e-3, f"loss did not decrease: {losses}"

    # Dispatch should have recorded an M2RNN run without direct-MSL Path B.
    log = get_dispatch_log()
    matches = [e for e in log if e["op_name"] == "m2rnn"]
    assert matches, f"no m2rnn dispatch in log: {log}"
    assert all(m["kernel_used"] != "metal_kernel_fwd_v1" for m in matches), (
        f"unexpected retired path_b dispatch, log: {matches}"
    )
