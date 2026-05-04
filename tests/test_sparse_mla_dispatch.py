"""End-to-end tests for the sparse-MLA production dispatcher.

These tests verify that :func:`cppmega_mlx.nn.sparse_mla.sparse_mla_attention`
honors the :class:`cppmega_mlx.runtime.kernel_policy.KernelPath` selection
and records the actual kernel used into the dispatch log. They run on Metal
when available; on hosts without Metal the AUTO and PATH_B paths gracefully
fall back to the reference (PATH_B raises in that case).
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import mlx.core as mx
from mlx.core import array as MLXArray

from cppmega_mlx.nn.sparse_mla import (
    _sparse_mla_path_c_receipt_allows_auto_promotion,
    sparse_mla_attention,
    sparse_mla_attention_reference,
)
from cppmega_mlx.nn._tilelang.sparse_mla_path_c import sparse_mla_path_c_status
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


def test_default_auto_routes_to_path_b_when_shape_is_not_receipted() -> None:
    """AUTO stays Path B for shapes that are not covered by the Path C receipt."""

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


def test_sparse_mla_path_c_receipt_blocks_auto_promotion() -> None:
    q, kv, indices = _make_inputs()
    assert _sparse_mla_path_c_receipt_allows_auto_promotion(q=q, kv=kv, indices=indices) is False


def test_sparse_mla_path_c_receipt_accepts_checked_in_strict_green_small_shape() -> None:
    q, kv, indices = _make_inputs(batch=2, seq=128, heads=8, qk_dim=64, topk=16, kv_seq=128)
    assert _sparse_mla_path_c_receipt_allows_auto_promotion(q=q, kv=kv, indices=indices) is True


def test_sparse_mla_path_c_receipt_accepts_checked_in_mid_shape() -> None:
    q, kv, indices = _make_inputs(batch=4, seq=512, heads=8, qk_dim=64, topk=32, kv_seq=512)
    assert _sparse_mla_path_c_receipt_allows_auto_promotion(q=q, kv=kv, indices=indices) is True


def test_sparse_mla_path_c_receipt_accepts_checked_in_strict_green_shape() -> None:
    q, kv, indices = _make_inputs(batch=4, seq=1024, heads=8, qk_dim=64, topk=64, kv_seq=1024)
    assert _sparse_mla_path_c_receipt_allows_auto_promotion(q=q, kv=kv, indices=indices) is True


def test_sparse_mla_auto_promotes_checked_in_small_green_shape_to_path_c() -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    q, kv, indices = _make_inputs(batch=2, seq=128, heads=8, qk_dim=64, topk=16, kv_seq=128)
    out = sparse_mla_attention(q, kv, indices)
    mx.eval(out)
    log = get_dispatch_log()
    assert log[-1]["op_name"] == "sparse_mla"
    assert log[-1]["path"] == "auto"
    assert log[-1]["kernel_used"] == "tilelang_path_c_fwd_bwd_v1"


def test_sparse_mla_auto_promotes_receipted_green_shape_to_path_c() -> None:
    if not _METAL_AVAILABLE:
        pytest.skip("Metal not available")
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    q, kv, indices = _make_inputs(batch=4, seq=1024, heads=8, qk_dim=64, topk=64, kv_seq=1024)
    out = sparse_mla_attention(q, kv, indices)
    mx.eval(out)
    log = get_dispatch_log()
    assert log[-1]["op_name"] == "sparse_mla"
    assert log[-1]["path"] == "auto"
    assert log[-1]["kernel_used"] == "tilelang_path_c_fwd_bwd_v1"


def test_sparse_mla_path_c_receipt_gate_accepts_synthetic_strict_green_receipt(
    tmp_path: Path,
) -> None:
    q, kv, indices = _make_inputs(batch=2, seq=128, heads=8, qk_dim=64, topk=16, kv_seq=128)
    receipt_path = tmp_path / "sparse_mla_strict_green.json"
    receipt_path.write_text(
        json.dumps(
            {
                "kernel": "sparse_mla",
                "fwd_only": False,
                "strict_policy": {
                    "phase": "all",
                    "fwd_only": False,
                    "requires_path_b_and_path_c": True,
                    "path_c_over_path_b_max_ratio": 1.0,
                },
                "path_b_status": {"available": True, "reason": "ok"},
                "path_c_status": {"available": True, "reason": "ok"},
                "rows": [
                    {
                        "shape": {
                            "B": 2,
                            "S": 128,
                            "H": 8,
                            "D": 64,
                            "G": 1,
                            "topk": 16,
                            "Skv": 128,
                        },
                        "fwd_path_c_no_worse_than_path_b": True,
                        "fwd_path_c_no_worse_than_path_b_paired": True,
                        "bwd_path_c_no_worse_than_path_b": True,
                        "bwd_path_c_no_worse_than_path_b_paired": True,
                    }
                ],
            }
        )
    )

    assert (
        _sparse_mla_path_c_receipt_allows_auto_promotion(
            receipt_path,
            q=q,
            kv=kv,
            indices=indices,
        )
        is True
    )


def test_sparse_mla_path_c_receipt_gate_honors_fp16_carrier_receipt(
    tmp_path: Path,
) -> None:
    q, kv, indices = _make_inputs(batch=2, seq=128, heads=8, qk_dim=64, topk=16, kv_seq=128)
    q_fp32, kv_fp32, _ = _make_inputs(
        batch=2,
        seq=128,
        heads=8,
        qk_dim=64,
        topk=16,
        kv_seq=128,
        dtype=np.float32,
    )
    receipt_path = tmp_path / "sparse_mla_strict_green_fp16.json"
    receipt_path.write_text(
        json.dumps(
            {
                "kernel": "sparse_mla",
                "fwd_only": False,
                "fp16_carrier": True,
                "strict_policy": {
                    "phase": "all",
                    "fwd_only": False,
                    "requires_path_b_and_path_c": True,
                    "path_c_over_path_b_max_ratio": 1.0,
                },
                "path_b_status": {"available": True, "reason": "ok"},
                "path_c_status": {"available": True, "reason": "ok"},
                "rows": [
                    {
                        "shape": {
                            "B": 2,
                            "S": 128,
                            "H": 8,
                            "D": 64,
                            "G": 1,
                            "topk": 16,
                            "Skv": 128,
                        },
                        "fwd_path_c_no_worse_than_path_b": True,
                        "fwd_path_c_no_worse_than_path_b_paired": True,
                        "bwd_path_c_no_worse_than_path_b": True,
                        "bwd_path_c_no_worse_than_path_b_paired": True,
                    }
                ],
            }
        )
    )

    assert _sparse_mla_path_c_receipt_allows_auto_promotion(
        receipt_path,
        q=q,
        kv=kv,
        indices=indices,
    )
    assert (
        _sparse_mla_path_c_receipt_allows_auto_promotion(
            receipt_path,
            q=q_fp32,
            kv=kv_fp32,
            indices=indices,
        )
        is False
    )


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


def test_path_c_policy_forces_tilelang_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    q, kv, indices = _make_inputs(qk_dim=32)
    out = sparse_mla_attention(q, kv, indices)
    ref = sparse_mla_attention_reference(q, kv, indices)
    mx.eval(out, ref)
    log = get_dispatch_log()
    assert log[-1]["path"] == "path_c"
    assert log[-1]["kernel_used"] == "tilelang_path_c_fwd_bwd_v1"
    np.testing.assert_allclose(
        np.array(out).astype(np.float32),
        np.array(ref).astype(np.float32),
        rtol=5e-3,
        atol=8e-3,
    )


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
        out = sparse_mla_attention(q_, kv_, indices)
        assert isinstance(out, MLXArray)
        out_arr = cast(mx.array, out)
        return mx.sum(out_arr**2)

    dq, dkv = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq, dkv)
    assert np.all(np.isfinite(np.array(dq)))
    assert np.all(np.isfinite(np.array(dkv)))
    assert np.linalg.norm(np.array(dq)) > 0
    assert np.linalg.norm(np.array(dkv)) > 0


def test_dispatch_grad_flows_through_path_c(monkeypatch: pytest.MonkeyPatch) -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    rng = np.random.default_rng(4)
    B, S, H, D = 1, 4, 2, 16
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32))
    kv = mx.array(rng.standard_normal((B, 8, 1, D)).astype(np.float32))
    indices = mx.array(rng.integers(0, 8, size=(B, S, 1, 4)).astype(np.int32))

    def loss(q_, kv_):
        out = sparse_mla_attention(q_, kv_, indices)
        assert isinstance(out, MLXArray)
        out_arr = cast(mx.array, out)
        return mx.sum(out_arr**2)

    dq, dkv = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq, dkv)
    log = get_dispatch_log()
    assert log[-1]["path"] == "path_c"
    assert log[-1]["kernel_used"] == "tilelang_path_c_fwd_bwd_v1"
    assert np.all(np.isfinite(np.array(dq)))
    assert np.all(np.isfinite(np.array(dkv)))
    assert np.linalg.norm(np.array(dq)) > 0
    assert np.linalg.norm(np.array(dkv)) > 0


def test_dispatch_path_c_grad_supports_packed_value_tail_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    rng = np.random.default_rng(5)
    B, S, H, D = 1, 4, 4, 16
    G = 2
    d_v = 8
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32))
    kv = mx.array(rng.standard_normal((B, 8, G, D)).astype(np.float32))
    indices_np = rng.integers(0, 8, size=(B, S, G, 4)).astype(np.int32)
    indices_np[0, 0, 0, :] = -1
    indices_np[0, 1, :, ::2] = -1
    indices = mx.array(indices_np)

    def path_c_loss(q_, kv_):
        out = sparse_mla_attention(q_, kv_, indices, d_v=d_v)
        assert isinstance(out, MLXArray)
        return mx.sum(cast(mx.array, out) ** 2)

    def reference_loss(q_, kv_):
        out = sparse_mla_attention_reference(q_, kv_, indices, d_v=d_v)
        return mx.sum(cast(mx.array, out) ** 2)

    dq, dkv = mx.grad(path_c_loss, argnums=(0, 1))(q, kv)
    dq_ref, dkv_ref = mx.grad(reference_loss, argnums=(0, 1))(q, kv)
    mx.eval(dq, dkv, dq_ref, dkv_ref)

    log = get_dispatch_log()
    assert log[-1]["path"] == "path_c"
    assert log[-1]["kernel_used"] == "tilelang_path_c_fwd_bwd_v1"
    np.testing.assert_allclose(
        np.array(dq).astype(np.float32),
        np.array(dq_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(dkv).astype(np.float32),
        np.array(dkv_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    assert np.linalg.norm(np.array(dkv[..., d_v:]).astype(np.float32)) > 0


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
