"""End-to-end tests for the M2RNN production dispatcher.

These tests verify that :class:`cppmega_mlx.nn.m2rnn.M2RNNMixer` honors the
:class:`cppmega_mlx.runtime.kernel_policy.KernelPath` selection and records
the actual kernel used into the dispatch log. They run on Metal when
available; on hosts without Metal the AUTO/PATH_B paths gracefully fall
back to the reference (PATH_B raises in that case).
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn.m2rnn import M2RNNConfig, M2RNNMixer
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


def test_default_auto_dispatches_path_b_when_metal_available() -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    block, hidden = _make_block()
    assert selected_path("m2rnn") is KernelPath.AUTO
    out, _ = block(hidden)
    mx.eval(out)
    log = get_dispatch_log()
    matches = [e for e in log if e["op_name"] == "m2rnn"]
    assert matches, f"no m2rnn dispatch recorded: {log}"
    assert matches[-1]["path"] == "auto"
    assert matches[-1]["kernel_used"] == "metal_kernel_fwd_v1"


def test_reference_policy_forces_pure_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "ref")
    block, hidden = _make_block()
    out, _ = block(hidden)
    mx.eval(out)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert matches[-1]["path"] == "ref"
    assert matches[-1]["kernel_used"] == "reference_pure_mlx"


def test_path_b_policy_forces_metal(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_b")
    block, hidden = _make_block()
    out, _ = block(hidden)
    mx.eval(out)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert matches[-1]["path"] == "path_b"
    assert matches[-1]["kernel_used"] == "metal_kernel_fwd_v1"


def test_path_c_dispatch_fails_closed_before_broadcast_or_state_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit Path C refuses mixer broadcast/state allocation instead of copying."""

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
    with pytest.raises(RuntimeError, match="pre-aligned.*head counts"):
        block(hidden)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert not matches, f"failed Path C dispatch should not log success: {matches}"


def test_path_c_dispatch_requires_existing_h0_without_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega_mlx.nn._tilelang import m2rnn_path_c

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_path_c_status",
        lambda: m2rnn_path_c.M2RNNPathCStatus(True, "forced available"),
    )

    def fail_path_c_kernel(*_args: object, **_kwargs: object) -> tuple[mx.array, mx.array]:
        raise AssertionError("Path C should fail before allocating h0 or launching")

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_apply_with_state_path_c",
        fail_path_c_kernel,
    )
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    cfg = M2RNNConfig(
        d_model=16,
        k_head_dim=4,
        v_head_dim=3,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=2,
        num_f_heads=2,
        num_g_heads=2,
        num_weight_heads=2,
        conv_kernel=4,
        chunk_size=8,
    )
    block = M2RNNMixer(cfg)
    hidden = mx.random.normal((1, 8, cfg.d_model), dtype=mx.float32) * 0.1
    with pytest.raises(RuntimeError, match="existing h0 tensor"):
        block(hidden)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert not matches, f"failed Path C dispatch should not log success: {matches}"


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


def test_block_grad_flows_through_path_b(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
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


def test_block_path_b_matches_reference_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward equality between Path B and the pure-MLX reference."""

    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    block, hidden = _make_block(seed=4)

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "ref")
    out_ref, h_ref = block(hidden)
    mx.eval(out_ref, h_ref)

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_b")
    out_b, h_b = block(hidden)
    mx.eval(out_b, h_b)

    np.testing.assert_allclose(
        np.array(out_b).astype(np.float32),
        np.array(out_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(h_b).astype(np.float32),
        np.array(h_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_return_state_routes_through_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache assembly path also picks up h_last from Path B."""

    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_b")
    block, hidden = _make_block(seed=5)
    out, mixer_state = block(hidden, return_state=True)
    mx.eval(out, mixer_state.h, mixer_state.conv_state)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "m2rnn"]
    assert matches[-1]["kernel_used"] == "metal_kernel_fwd_v1"
    cfg = block.config
    assert mixer_state.h.shape == (
        1, cfg.num_heads, cfg.k_head_dim, cfg.v_head_dim,
    )


def test_hybrid_lm_e2e_with_r_block_trains_loss_decreases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny HybridLM with an R block trains for 5 steps; loss should decrease.

    Verifies the dispatch log shows m2rnn ran on Path B by default.
    """

    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")

    from cppmega_mlx.models.hybrid_lm import HybridTinyLM, HybridTinyConfig
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

    # Dispatch should have recorded a Path B m2rnn run.
    log = get_dispatch_log()
    matches = [e for e in log if e["op_name"] == "m2rnn"]
    assert matches, f"no m2rnn dispatch in log: {log}"
    assert any(m["kernel_used"] == "metal_kernel_fwd_v1" for m in matches), (
        f"expected at least one path_b dispatch, log: {matches}"
    )
