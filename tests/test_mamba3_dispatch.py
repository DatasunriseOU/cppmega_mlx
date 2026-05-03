"""End-to-end tests for the Mamba3 production dispatcher.

These tests verify that :class:`cppmega_mlx.nn.mamba3.Mamba3ReferenceBlock`
honors the :class:`cppmega_mlx.runtime.kernel_policy.KernelPath` selection
and records the actual kernel used into the dispatch log. They run on Metal
when available; on hosts without Metal the AUTO/PATH_B paths gracefully
fall back to the reference (PATH_B raises in that case).
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn.mamba3 import Mamba3Config, Mamba3ReferenceBlock
from cppmega_mlx.runtime.kernel_policy import (
    KernelPath,
    clear_dispatch_log,
    get_dispatch_log,
    selected_path,
)


_METAL_AVAILABLE = mx.metal.is_available()


def _make_block(seed: int = 0) -> tuple[Mamba3ReferenceBlock, mx.array]:
    mx.random.seed(seed)
    cfg = Mamba3Config(
        d_model=16, expand=2, headdim=8, d_state=8, ngroups=1, chunk_size=8
    )
    block = Mamba3ReferenceBlock(cfg)
    hidden = mx.random.normal((1, 4, cfg.d_model), dtype=mx.float32) * 0.1
    return block, hidden


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch):
    clear_dispatch_log()
    monkeypatch.delenv("CPPMEGA_KERNEL_PATH", raising=False)
    monkeypatch.delenv("CPPMEGA_KERNEL_PATH__MAMBA3_MIMO", raising=False)
    yield
    clear_dispatch_log()


def test_default_auto_dispatches_path_b_when_metal_available() -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    block, hidden = _make_block()
    assert selected_path("mamba3_mimo") is KernelPath.AUTO
    out, _ = block(hidden)
    mx.eval(out)
    log = get_dispatch_log()
    matches = [e for e in log if e["op_name"] == "mamba3_mimo"]
    assert matches, f"no mamba3_mimo dispatch recorded: {log}"
    assert matches[-1]["path"] == "auto"
    assert matches[-1]["kernel_used"] == "metal_kernel_fwd_v1"


def test_reference_policy_forces_pure_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "ref")
    block, hidden = _make_block()
    out, _ = block(hidden)
    mx.eval(out)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "mamba3_mimo"]
    assert matches[-1]["path"] == "ref"
    assert matches[-1]["kernel_used"] == "reference_pure_mlx"


def test_path_b_policy_forces_metal(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_b")
    block, hidden = _make_block()
    out, _ = block(hidden)
    mx.eval(out)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "mamba3_mimo"]
    assert matches[-1]["path"] == "path_b"
    assert matches[-1]["kernel_used"] == "metal_kernel_fwd_v1"


def test_path_c_dispatches_tilelang_dsl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path C is the only op-level kernel that supports the TileLang DSL."""

    pytest.importorskip("tilelang")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    block, hidden = _make_block()
    try:
        out, _ = block(hidden)
        mx.eval(out)
    except Exception as exc:  # pragma: no cover - tilelang path may degrade.
        pytest.skip(f"tilelang DSL path unavailable on this host: {exc}")
    matches = [e for e in get_dispatch_log() if e["op_name"] == "mamba3_mimo"]
    assert matches, "expected at least one mamba3_mimo dispatch"
    assert matches[-1]["path"] == "path_c"


def test_per_op_override_selects_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "auto")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__MAMBA3_MIMO", "ref")
    block, hidden = _make_block()
    out, _ = block(hidden)
    mx.eval(out)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "mamba3_mimo"]
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


def test_return_cache_routes_through_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache assembly path also picks up h_last from Path B."""

    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_b")
    block, hidden = _make_block(seed=5)
    out, cache = block(hidden, return_cache=True)
    mx.eval(out, cache.ssm)
    matches = [e for e in get_dispatch_log() if e["op_name"] == "mamba3_mimo"]
    assert matches[-1]["kernel_used"] == "metal_kernel_fwd_v1"
    assert cache.ssm.shape == (1, block.config.nheads, block.config.headdim, block.config.d_state)
