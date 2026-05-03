"""End-to-end tests for the sparse-MLA production dispatcher.

These tests verify that :func:`cppmega_mlx.nn.sparse_mla.sparse_mla_attention`
honors the :class:`cppmega_mlx.runtime.kernel_policy.KernelPath` selection
and records the actual kernel used into the dispatch log. They run on Metal
when available; on hosts without Metal the AUTO and PATH_B paths gracefully
fall back to the reference (PATH_B raises in that case).
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn.sparse_mla import (
    sparse_mla_attention,
    sparse_mla_attention_reference,
)
from cppmega_mlx.runtime.kernel_policy import (
    KernelPath,
    clear_dispatch_log,
    get_dispatch_log,
    selected_path,
)


_METAL_AVAILABLE = mx.metal.is_available()


def _make_inputs(*, batch=1, seq=4, heads=2, qk_dim=16, kv_group=1, topk=4, kv_seq=8, dtype=np.float16):
    rng = np.random.default_rng(0)
    q = mx.array(rng.standard_normal((batch, seq, heads, qk_dim)).astype(dtype))
    kv = mx.array(rng.standard_normal((batch, kv_seq, kv_group, qk_dim)).astype(dtype))
    indices = mx.array(rng.integers(0, kv_seq, size=(batch, seq, kv_group, topk)).astype(np.int32))
    return q, kv, indices


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch):
    clear_dispatch_log()
    monkeypatch.delenv("CPPMEGA_KERNEL_PATH", raising=False)
    monkeypatch.delenv("CPPMEGA_KERNEL_PATH__SPARSE_MLA", raising=False)
    yield
    clear_dispatch_log()


def test_default_auto_routes_to_path_b_when_metal_available() -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    q, kv, indices = _make_inputs()
    assert selected_path("sparse_mla") is KernelPath.AUTO
    out = sparse_mla_attention(q, kv, indices)
    mx.eval(out)
    log = get_dispatch_log()
    assert len(log) == 1
    assert log[0]["op_name"] == "sparse_mla"
    assert log[0]["path"] == "auto"
    assert log[0]["kernel_used"] == "metal_kernel_fwd_v1"


def test_reference_policy_forces_pure_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "ref")
    q, kv, indices = _make_inputs()
    out = sparse_mla_attention(q, kv, indices)
    mx.eval(out)
    log = get_dispatch_log()
    assert log[-1]["path"] == "ref"
    assert log[-1]["kernel_used"] == "reference_pure_mlx"


def test_path_b_policy_forces_metal_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_b")
    q, kv, indices = _make_inputs()
    out = sparse_mla_attention(q, kv, indices)
    mx.eval(out)
    log = get_dispatch_log()
    assert log[-1]["path"] == "path_b"
    assert log[-1]["kernel_used"] == "metal_kernel_fwd_v1"


def test_path_c_raises_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    q, kv, indices = _make_inputs()
    with pytest.raises(NotImplementedError, match="sparse-MLA Path C"):
        sparse_mla_attention(q, kv, indices)


def test_per_op_override_selects_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "auto")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__SPARSE_MLA", "ref")
    q, kv, indices = _make_inputs()
    out = sparse_mla_attention(q, kv, indices)
    mx.eval(out)
    log = get_dispatch_log()
    assert log[-1]["kernel_used"] == "reference_pure_mlx"


def test_dispatch_grad_flows_through_path_b(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "auto")
    rng = np.random.default_rng(2)
    B, S, H, D = 1, 4, 2, 16
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32))
    kv = mx.array(rng.standard_normal((B, 8, 1, D)).astype(np.float32))
    indices = mx.array(rng.integers(0, 8, size=(B, S, 1, 4)).astype(np.int32))

    def loss(q_, kv_):
        return mx.sum(sparse_mla_attention(q_, kv_, indices) ** 2)

    dq, dkv = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq, dkv)
    assert np.all(np.isfinite(np.array(dq)))
    assert np.all(np.isfinite(np.array(dkv)))
    assert np.linalg.norm(np.array(dq)) > 0
    assert np.linalg.norm(np.array(dkv)) > 0


def test_return_lse_path_routes_through_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "ref")
    q, kv, indices = _make_inputs()
    out, lse = sparse_mla_attention(q, kv, indices, return_lse=True)
    mx.eval(out, lse)
    assert out.shape[:-1] == lse.shape


def test_path_b_and_reference_are_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spot check Path B vs reference parity through the dispatcher."""

    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    rng = np.random.default_rng(3)
    B, S, H, D = 1, 4, 2, 32
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float16))
    kv = mx.array(rng.standard_normal((B, 8, 1, D)).astype(np.float16))
    indices = mx.array(rng.integers(0, 8, size=(B, S, 1, 4)).astype(np.int32))

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_b")
    out_b = sparse_mla_attention(q, kv, indices)
    mx.eval(out_b)
    out_ref = sparse_mla_attention_reference(q, kv, indices)
    mx.eval(out_ref)
    np.testing.assert_allclose(
        np.array(out_b).astype(np.float32),
        np.array(out_ref).astype(np.float32),
        rtol=1e-3,
        atol=2e-3,
    )
