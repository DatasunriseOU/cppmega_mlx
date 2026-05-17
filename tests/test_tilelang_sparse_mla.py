"""Tests for sparse-MLA Path C + pure-MLX reference parity.

The regular Path B direct-MSL forward surface is retired; the compatibility
module now reports an explicit unavailable status and falls back to the
pure-MLX reference unless the production dispatcher selects Path C.

The tests verify:

1. The pure-MLX reference matches a hand-rolled NumPy reference (forward
   parity oracle).
2. The reference is differentiable via mx.value_and_grad and gradient norms
   are finite.
3. The TileLang/tvm-ffi Path C forward and backward match the pure-MLX
   reference.
4. The retired Path B compatibility surface fails closed instead of
   constructing raw direct-MSL kernels.

Tolerances: rtol=1e-3, atol=1e-3 for fp16 (plus generous fp32 hand checks).
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import mlx.core as mx

from typing import cast

import cppmega_mlx.nn._tilelang.sparse_mla_path_c as sparse_mla_path_c  # noqa: E402
from cppmega_mlx.nn._tilelang.sparse_mla import (  # noqa: E402
    SparseMLAMetalStatus,
    sparse_mla_apply,
    sparse_mla_bwd_metal,
    sparse_mla_fwd_metal,
    sparse_mla_metal_status,
)
from cppmega_mlx.nn._tilelang.sparse_mla_path_c import (  # noqa: E402
    _bwd_direct_lowering_for,
    _fwd_kernel_for,
    _mlx_total_thread_grid,
    _threadgroup_size,
    dump_lowered_bwd_msl,
    dump_lowered_fwd_msl,
    SparseMLAPathCDirectError,
    SparseMLAPathCStatus,
    sparse_mla_bwd_path_c,
    sparse_mla_fwd_path_c,
    sparse_mla_path_c_status,
)
from cppmega_mlx.nn.sparse_mla import (  # noqa: E402
    sparse_mla_attention,
    sparse_mla_attention_reference,
)


def _assert_bwd_direct_owner_output_msl(msl: str) -> None:
    """Guard debug MSL against leaking the removed public partial-output ABI."""

    assert "device float* dkv" in msl
    assert "device half* dkv_partial" not in msl
    assert "dkv_partial" not in msl
    assert "tl::AtomicAdd" in msl
    assert "device half* dq" in msl


def _assert_bwd_direct_owner_output_lane_loops(
    msl: str,
    *,
    qk_dim: int,
    topk: int,
    threads: int,
    d_v: int | None = None,
) -> None:
    """Check the owner-output bwd debug MSL keeps the optimized lane loops."""

    if d_v is None:
        d_v = qk_dim
    _assert_bwd_direct_owner_output_msl(msl)
    assert "uint tid = thread_position_in_threadgroup.x;" in msl
    assert "uint gid = threadgroup_position_in_grid.x;" in msl
    assert f"uint threads = {threads};" in msl
    assert "round_id" not in msl
    assert "half condval" not in msl
    assert "((int)threadIdx.x)" not in msl
    assert "int stride;" not in msl
    assert f"for (uint d = tid; d < {qk_dim}; d += threads)" in msl
    assert f"for (uint kd = tid; kd < {topk * qk_dim}; kd += threads)" in msl
    assert "uint q_row_base =" in msl
    assert "uint d_out_row =" in msl
    assert "uint kv_b_base =" in msl
    assert "uint idx_base =" in msl
    assert "uint dkv_pb =" in msl
    assert "kv[kv_row_base + d]" in msl
    if f"uint k = kd / {qk_dim};" in msl:
        assert f"uint d = kd % {qk_dim};" in msl
        assert "ds[k]" in msl
    else:
        assert f"ds[(kd / {qk_dim})]" in msl
    assert "int gather_idx = indices[idx_base + k];" in msl
    assert "float qv = float(q[q_row_base + d]);" in msl
    assert "if (0 <= gather_idx)" in msl
    assert "sm_scale * ds[" in msl
    if d_v < qk_dim:
        assert f"for (uint d = 0; d < {d_v}; ++d)" in msl
    assert "float dod = float(d_out[d_out_row + d]);" in msl
    assert "sumexp <= 0.000000e+00f" in msl
    assert msl.count("for (uint stride = threads / 2; stride > 0; stride >>= 1)") == 3


# ---------------------------------------------------------------------------
# Hand-rolled NumPy reference (correctness oracle)
# ---------------------------------------------------------------------------


def _np_sparse_mla(
    q: np.ndarray,
    kv: np.ndarray,
    indices: np.ndarray,
    *,
    sm_scale: float,
    d_v: int,
) -> np.ndarray:
    """Per-token loop reference for sparse-MLA in float32."""

    B, S, H, qk_dim = q.shape
    _, Skv, G, _ = kv.shape
    head_kv = H // G
    out = np.zeros((B, S, H, d_v), dtype=np.float32)
    q32 = q.astype(np.float32)
    kv32 = kv.astype(np.float32)
    for b in range(B):
        for s in range(S):
            for g in range(G):
                k_indices = indices[b, s, g, :]
                valid = k_indices != -1
                gathered = kv32[b, np.maximum(k_indices, 0), g]
                for h_off in range(head_kv):
                    h = g * head_kv + h_off
                    qrow = q32[b, s, h, :]
                    scores = (qrow @ gathered.T) * sm_scale
                    scores = np.where(valid, scores, -np.inf)
                    if not valid.any():
                        out[b, s, h, :] = 0
                        continue
                    m = scores.max()
                    exp = np.exp(scores - m)
                    exp = np.where(valid, exp, 0.0)
                    probs = exp / exp.sum()
                    out[b, s, h, :] = probs @ gathered[:, :d_v]
    return out


# ---------------------------------------------------------------------------
# Shape grid for the parity oracle
# ---------------------------------------------------------------------------


SMOKE_SHAPES = [
    pytest.param(
        dict(B=2, S=128, H=8, D=64, G=1, topk=16, Skv=128),
        id="B2_S128_H8_D64",
    ),
    pytest.param(
        dict(B=4, S=512, H=8, D=64, G=1, topk=32, Skv=512),
        id="B4_S512_H8_D64",
    ),
    pytest.param(
        dict(B=1, S=64, H=8, D=64, G=2, topk=16, Skv=128),
        id="B1_S64_H8_D64_G2",
    ),
    pytest.param(
        dict(B=2, S=64, H=4, D=48, G=1, topk=16, Skv=96, d_v=32),
        id="tail_dim16",
    ),
]


# ---------------------------------------------------------------------------
# Forward parity (reference vs hand NumPy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", SMOKE_SHAPES)
def test_reference_matches_numpy_oracle(cfg) -> None:
    rng = np.random.default_rng(0)
    B, S, H, D = cfg["B"], cfg["S"], cfg["H"], cfg["D"]
    G = cfg["G"]
    topk = cfg["topk"]
    Skv = cfg["Skv"]
    d_v = cfg.get("d_v", D)

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float16)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)

    sm_scale = D ** -0.5

    out_mlx = sparse_mla_attention_reference(
        mx.array(q_np), mx.array(kv_np), mx.array(indices_np), sm_scale=sm_scale, d_v=d_v
    )
    mx.eval(out_mlx)
    out_np = np.array(out_mlx).astype(np.float32)

    ref = _np_sparse_mla(q_np, kv_np, indices_np, sm_scale=sm_scale, d_v=d_v)
    np.testing.assert_allclose(out_np, ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Mask handling: -1 sentinel zeros that token's output, isn't NaN
# ---------------------------------------------------------------------------


def test_invalid_indices_zero_output() -> None:
    rng = np.random.default_rng(1)
    B, S, H, D = 2, 8, 4, 32
    G = 1
    topk = 4
    Skv = 16

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float16)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    indices_np[0, 0, 0, :] = -1  # all invalid for first token

    out = sparse_mla_attention_reference(
        mx.array(q_np), mx.array(kv_np), mx.array(indices_np)
    )
    mx.eval(out)
    out_np = np.array(out)
    assert not np.isnan(out_np).any()
    np.testing.assert_array_equal(out_np[0, 0, 0], np.zeros(D, dtype=out_np.dtype))


def test_partial_invalid_indices_match_oracle() -> None:
    rng = np.random.default_rng(2)
    B, S, H, D = 1, 4, 4, 16
    G = 1
    topk = 6
    Skv = 8
    sm_scale = D ** -0.5

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float16)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    # Mask half the indices for token (0,1)
    indices_np[0, 1, 0, ::2] = -1

    out_mlx = sparse_mla_attention_reference(
        mx.array(q_np), mx.array(kv_np), mx.array(indices_np), sm_scale=sm_scale
    )
    mx.eval(out_mlx)
    ref = _np_sparse_mla(q_np, kv_np, indices_np, sm_scale=sm_scale, d_v=D)
    np.testing.assert_allclose(np.array(out_mlx).astype(np.float32), ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Backward parity: gradient norms should be finite and match between two
# autograd traces of the reference (anchors backward correctness through MLX).
# ---------------------------------------------------------------------------


def test_reference_backward_finite() -> None:
    rng = np.random.default_rng(3)
    B, S, H, D = 2, 16, 4, 32
    G = 1
    topk = 8
    Skv = 32

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float32)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float32)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)

    q = mx.array(q_np)
    kv = mx.array(kv_np)
    indices = mx.array(indices_np)

    def loss(q_in: mx.array, kv_in: mx.array) -> mx.array:
        out = cast(mx.array, sparse_mla_attention(q_in, kv_in, indices, sm_scale=D ** -0.5))
        return mx.mean(out * out)

    grads = mx.grad(loss, argnums=(0, 1))(q, kv)
    dq, dkv = grads
    mx.eval(dq, dkv)
    assert dq.shape == q.shape
    assert dkv.shape == kv.shape
    dq_np = np.array(dq)
    dkv_np = np.array(dkv)
    assert np.isfinite(dq_np).all()
    assert np.isfinite(dkv_np).all()
    # Gradients should be non-zero somewhere
    assert np.linalg.norm(dq_np) > 0
    assert np.linalg.norm(dkv_np) > 0


def test_reference_backward_against_finite_difference() -> None:
    """Spot-check a single q entry's gradient via central finite differences.

    Targets the pure-MLX reference explicitly (independent of the production
    dispatcher) so the FD comparison stays at fp32 precision.
    """

    rng = np.random.default_rng(4)
    B, S, H, D = 1, 4, 2, 8
    G = 1
    topk = 3
    Skv = 6
    sm_scale = D ** -0.5

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float32)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float32)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)

    q = mx.array(q_np)
    kv = mx.array(kv_np)
    indices = mx.array(indices_np)

    def scalar_loss(q_in: mx.array) -> mx.array:
        out = sparse_mla_attention_reference(q_in, kv, indices, sm_scale=sm_scale)
        return mx.sum(out)

    grad_q = mx.grad(scalar_loss)(q)
    mx.eval(grad_q)
    grad_q_np = np.array(grad_q)

    eps = 1e-3
    # Probe a handful of entries in q
    probes = [(0, 0, 0, 0), (0, 1, 1, 3), (0, 2, 0, 5)]
    for idx in probes:
        q_plus = q_np.copy()
        q_plus[idx] += eps
        q_minus = q_np.copy()
        q_minus[idx] -= eps
        loss_plus = float(np.array(scalar_loss(mx.array(q_plus))))
        loss_minus = float(np.array(scalar_loss(mx.array(q_minus))))
        fd = (loss_plus - loss_minus) / (2 * eps)
        analytic = float(grad_q_np[idx])
        np.testing.assert_allclose(analytic, fd, atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------------
# Path B status surface
# ---------------------------------------------------------------------------


def test_metal_status_reports_available() -> None:
    """The retired direct-MSL bypass should report unavailable on Metal."""

    status = sparse_mla_metal_status()
    assert isinstance(status, SparseMLAMetalStatus)
    if mx.metal.is_available():
        assert status.available is False
        assert status.fp16_carrier is True
        assert "direct-MSL Path B is retired" in status.reason
    else:
        assert status.available is False


def test_apply_matches_reference_within_fp16_tolerance() -> None:
    rng = np.random.default_rng(5)
    B, S, H, D = 2, 8, 4, 32
    G = 1
    topk = 4
    Skv = 16

    q_np = rng.standard_normal((B, S, H, D)).astype(np.float16)
    kv_np = rng.standard_normal((B, Skv, G, D)).astype(np.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)

    out_apply = sparse_mla_apply(
        mx.array(q_np), mx.array(kv_np), mx.array(indices_np), sm_scale=D ** -0.5
    )
    out_ref = sparse_mla_attention_reference(
        mx.array(q_np), mx.array(kv_np), mx.array(indices_np), sm_scale=D ** -0.5
    )
    mx.eval(out_apply, out_ref)
    # fp16 tolerance: the MSL kernel uses fp16 carrier with fp32 accumulators.
    np.testing.assert_allclose(
        np.array(out_apply).astype(np.float32),
        np.array(out_ref).astype(np.float32),
        rtol=1e-3,
        atol=2e-3,
    )


def test_apply_force_metal_raises_for_retired_path_b() -> None:
    """force_metal=True preserves Path B semantics and raises after retirement."""

    rng = np.random.default_rng(6)
    q = mx.array(rng.standard_normal((1, 4, 2, 32)).astype(np.float16))
    kv = mx.array(rng.standard_normal((1, 8, 1, 32)).astype(np.float16))
    indices = mx.array(rng.integers(0, 8, size=(1, 4, 1, 4)).astype(np.int32))
    with pytest.raises(RuntimeError, match="direct-MSL Path B is retired"):
        sparse_mla_apply(q, kv, indices, force_metal=True)


# ---------------------------------------------------------------------------
# Retired Path B compatibility surface.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        dict(B=1, S=4, H=2, D=16, G=1, topk=4, Skv=8),
        dict(B=2, S=16, H=4, D=32, G=1, topk=8, Skv=32),
        dict(B=1, S=8, H=4, D=32, G=2, topk=8, Skv=16),
        dict(B=2, S=8, H=4, D=48, G=1, topk=8, Skv=16, d_v=32),
    ],
    ids=["small", "medium", "multigroup", "tail_dim"],
)
def test_path_b_forward_surface_is_retired(cfg) -> None:
    rng = np.random.default_rng(13)
    B, S, H, D = cfg["B"], cfg["S"], cfg["H"], cfg["D"]
    G = cfg["G"]
    topk = cfg["topk"]
    Skv = cfg["Skv"]
    d_v = cfg.get("d_v")

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float16))
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float16))
    indices = mx.array(rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32))

    result = sparse_mla_fwd_metal(q, kv, indices, d_v=d_v)
    assert result is None


def test_path_b_forward_retirement_preserves_shape_validation() -> None:
    """The retired surface still validates inputs before failing closed."""

    rng = np.random.default_rng(17)
    B, S, H, D = 2, 4, 2, 32
    G = 2
    topk = 4
    Skv = 8
    q = mx.array(rng.standard_normal((B, S, H + 1, D)).astype(np.float16))
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float16))
    ind_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    ind_np[0, 0, 0, :] = -1  # all invalid for first token
    indices = mx.array(ind_np)

    with pytest.raises(ValueError, match="heads .* not divisible"):
        sparse_mla_fwd_metal(q, kv, indices)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(B=1, S=4, H=2, D=16, G=1, topk=4, Skv=8),
        dict(B=1, S=8, H=4, D=16, G=2, topk=8, Skv=16, d_v=8),
        dict(B=1, S=32, H=4, D=64, G=1, topk=32, Skv=64),
    ],
    ids=["fp16_small_16x16", "fp16_multigroup_tail_16x16", "fp16_topk32_32x32"],
)
def test_path_c_forward_fp16_parity(cfg) -> None:
    """TileLang DSL Path C fp16 forward matches the pure-MLX reference."""

    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)
    assert status.fp16_carrier is True

    rng = np.random.default_rng(19)
    B, S, H, D = cfg["B"], cfg["S"], cfg["H"], cfg["D"]
    G = cfg["G"]
    topk = cfg["topk"]
    Skv = cfg["Skv"]
    d_v = cfg.get("d_v", D)

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    indices_np[0, 0, 0, ::2] = -1
    indices = mx.array(indices_np)

    result = sparse_mla_fwd_path_c(q, kv, indices, d_v=d_v)
    assert result is not None, "TileLang DSL Path C fp16 forward kernel must dispatch"
    out_path_c, lse_path_c = result
    mx.eval(out_path_c, lse_path_c)

    out_ref, lse_ref = sparse_mla_attention_reference(q, kv, indices, d_v=d_v, return_lse=True)
    mx.eval(out_ref, lse_ref)

    assert out_path_c.dtype == mx.float16
    assert lse_path_c.dtype == mx.float32
    np.testing.assert_allclose(
        np.array(out_path_c.astype(mx.float32)),
        np.array(out_ref.astype(mx.float32)),
        rtol=5e-3,
        atol=8e-3,
    )
    np.testing.assert_allclose(
        np.array(lse_path_c).astype(np.float32),
        np.array(lse_ref).astype(np.float32),
        rtol=5e-3,
        atol=8e-3,
    )


def test_path_c_forward_int64_indices_parity() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    rng = np.random.default_rng(191)
    B, S, H, D, G, topk, Skv = 1, 2, 2, 16, 1, 4, 8
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int64)
    indices_np[0, 0, 0, 0] = -1
    indices = mx.array(indices_np)

    result = sparse_mla_fwd_path_c(q, kv, indices)
    assert result is not None
    out_path_c, _lse_path_c = result
    out_ref = sparse_mla_attention_reference(q, kv, indices)
    mx.eval(out_path_c, out_ref)
    np.testing.assert_allclose(
        np.array(out_path_c.astype(mx.float32)),
        np.array(out_ref.astype(mx.float32)),
        rtol=5e-3,
        atol=5e-3,
    )


def test_path_c_forward_fp16_invalid_indices_zero_output() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    rng = np.random.default_rng(21)
    B, S, H, D = 1, 4, 2, 16
    G = 1
    topk = 4
    Skv = 8
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    indices_np[0, 0, 0, :] = -1
    indices = mx.array(indices_np)

    result = sparse_mla_fwd_path_c(q, kv, indices)
    assert result is not None
    out, lse = result
    mx.eval(out, lse)
    out_np = np.array(out.astype(mx.float32))
    lse_np = np.array(lse)
    assert out.dtype == mx.float16
    assert not np.isnan(out_np).any()
    assert not np.isnan(lse_np).any()
    np.testing.assert_array_equal(out_np[0, 0, 0], np.zeros(D, dtype=np.float32))
    assert lse_np[0, 0, 0] == 0.0


def test_path_c_forward_direct_owner_output_reuses_buffers_without_mlx_fast_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_legacy_kernel(*_: object, **__: object) -> object:
        raise AssertionError("owner-output Sparse-MLA fwd must not build mx.fast fallback")

    monkeypatch.setattr(
        sparse_mla_path_c,
        "sparse_mla_path_c_status",
        lambda: SparseMLAPathCStatus(True, "test"),
    )
    monkeypatch.setattr(sparse_mla_path_c, "_fwd_kernel_for", fail_legacy_kernel)
    monkeypatch.setattr(sparse_mla_path_c.mx.fast, "metal_kernel", fail_legacy_kernel)

    calls: list[tuple[tuple[object, ...], tuple[mx.array, mx.array]]] = []

    def fake_kernel_for(*_: object, **__: object):
        def fake_kernel(*kernel_args: object, out: tuple[mx.array, mx.array]):
            calls.append((kernel_args, out))
            return list(out)

        return fake_kernel

    monkeypatch.setattr(
        sparse_mla_path_c,
        "_fwd_direct_tvm_ffi_kernel_for",
        fake_kernel_for,
    )

    rng = np.random.default_rng(67)
    B, S, H, D, G, topk, Skv = 1, 2, 2, 16, 1, 4, 8
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float16))
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float16))
    indices = mx.array(rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32))
    sm_scale_buf = mx.array([0.25], dtype=mx.float32)
    out = mx.zeros((B, S, H, D), dtype=mx.float16)
    lse = mx.zeros((B, S, H), dtype=mx.float32)

    returned = sparse_mla_fwd_path_c(
        q,
        kv,
        indices,
        sm_scale_buf=sm_scale_buf,
        out=out,
        lse=lse,
    )

    assert returned is not None
    returned_out, returned_lse = returned
    assert returned_out is out
    assert returned_lse is lse
    assert len(calls) == 1
    kernel_args, owner_outputs = calls[0]
    assert kernel_args[0] is indices
    assert kernel_args[1] is kv
    assert kernel_args[2] is q
    assert kernel_args[3] is sm_scale_buf
    assert owner_outputs[0] is lse
    assert owner_outputs[1] is out


def test_path_c_forward_direct_bf16_owner_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_legacy_kernel(*_: object, **__: object) -> object:
        raise AssertionError("BF16 owner-output gate must not fall back to mx.fast")

    monkeypatch.setattr(
        sparse_mla_path_c,
        "sparse_mla_path_c_status",
        lambda: SparseMLAPathCStatus(True, "test"),
    )
    monkeypatch.setattr(sparse_mla_path_c, "_fwd_kernel_for", fail_legacy_kernel)

    B, S, H, D, G, topk, Skv = 1, 2, 2, 16, 1, 4, 8
    q = mx.zeros((B, S, H, D), dtype=mx.bfloat16)
    kv = mx.zeros((B, Skv, G, D), dtype=mx.bfloat16)
    indices = mx.zeros((B, S, G, topk), dtype=mx.int32)
    sm_scale_buf = mx.array([0.25], dtype=mx.float32)
    out = mx.zeros((B, S, H, D), dtype=mx.float16)
    lse = mx.zeros((B, S, H), dtype=mx.float32)

    with pytest.raises(SparseMLAPathCDirectError, match="BF16 owner-output"):
        sparse_mla_fwd_path_c(
            q,
            kv,
            indices,
            sm_scale_buf=sm_scale_buf,
            out=out,
            lse=lse,
        )


def test_path_c_forward_accepts_int64_indices_without_hidden_cast() -> None:
    B, S, H, D, G, topk, Skv = 1, 2, 2, 16, 1, 4, 8
    _q = mx.zeros((B, S, H, D), dtype=mx.float16)
    _kv = mx.zeros((B, Skv, G, D), dtype=mx.float16)
    indices = mx.array(np.zeros((B, S, G, topk), dtype=np.int64))

    assert sparse_mla_path_c._index_dtype_name(indices, op_name="test") == "int64"
    assert sparse_mla_path_c._require_supported_indices_no_hidden_cast(
        indices,
        op_name="test",
    ) is indices


def test_backward_compat_shim_matches_reference() -> None:
    rng = np.random.default_rng(23)
    B, S, H, D = 2, 8, 4, 32
    G = 1
    topk = 8
    Skv = 16

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices = mx.array(rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32))
    d_out = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32))

    grads = sparse_mla_bwd_metal(q, kv, d_out, indices)
    assert grads is not None
    dq_msl, dkv_msl = grads
    mx.eval(dq_msl, dkv_msl)

    # Reference: autograd of pure-MLX path.
    def loss(q_, kv_):
        out = sparse_mla_attention_reference(q_, kv_, indices)
        return mx.sum(out * d_out)

    dq_ref, dkv_ref = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq_ref, dkv_ref)

    # fp16 carrier means slightly looser tolerance than fp32.
    np.testing.assert_allclose(
        np.array(dq_msl).astype(np.float32),
        np.array(dq_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(dkv_msl).astype(np.float32),
        np.array(dkv_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.parametrize(
    "cfg",
    [
        dict(B=1, S=4, H=2, D=16, G=1, topk=4, Skv=8),
        dict(B=1, S=4, H=4, D=16, G=2, topk=4, Skv=8, d_v=8),
        dict(B=1, S=8, H=4, D=32, G=1, topk=16, Skv=32),
        dict(B=1, S=32, H=4, D=64, G=1, topk=32, Skv=64),
    ],
    ids=["small", "multigroup_tail_dim", "topk16_threadgroup", "topk32_threadgroup"],
)
def test_path_c_backward_parity(cfg) -> None:
    """TileLang DSL Path C sparse-MLA backward matches the pure-MLX VJP."""

    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    rng = np.random.default_rng(31)
    B, S, H, D = cfg["B"], cfg["S"], cfg["H"], cfg["D"]
    G = cfg["G"]
    topk = cfg["topk"]
    Skv = cfg["Skv"]
    d_v = cfg.get("d_v", D)

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    indices_np[0, 0, 0, :] = -1
    indices_np[0, 1, 0, ::2] = -1
    indices = mx.array(indices_np)
    d_out = mx.array(rng.standard_normal((B, S, H, d_v)).astype(np.float32)).astype(mx.float16)

    grads = sparse_mla_bwd_path_c(q, kv, d_out, indices, d_v=d_v)
    assert grads is not None, "TileLang DSL Path C backward kernel must dispatch"
    dq_path_c, dkv_path_c = grads
    mx.eval(dq_path_c, dkv_path_c)

    def loss(q_, kv_):
        out = sparse_mla_attention_reference(q_, kv_, indices, d_v=d_v)
        return mx.sum(out * d_out)

    dq_ref, dkv_ref = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq_ref, dkv_ref)

    # Path C backward returns final owner-output gradients from the TileLang route.
    np.testing.assert_allclose(
        np.array(dq_path_c).astype(np.float32),
        np.array(dq_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(dkv_path_c).astype(np.float32),
        np.array(dkv_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_sparse_mla_bwd_metal_compat_shim_matches_path_c_and_reference() -> None:
    """The legacy bwd name delegates to Path C final owner-output gradients."""

    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    rng = np.random.default_rng(57)
    B, S, H, D = 1, 4, 4, 16
    G = 2
    topk = 4
    Skv = 8
    d_v = D

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    indices_np[0, 0, 0, :] = -1
    indices_np[0, 1, :, ::2] = -1
    indices = mx.array(indices_np)
    d_out = mx.array(rng.standard_normal((B, S, H, d_v)).astype(np.float32)).astype(mx.float16)

    compat = sparse_mla_bwd_metal(q, kv, d_out, indices, d_v=d_v)
    path_c = sparse_mla_bwd_path_c(q, kv, d_out, indices, d_v=d_v)
    assert compat is not None, "compat backward shim must dispatch through Path C"
    assert path_c is not None, "Path C backward must dispatch for direct parity"
    dq_compat, dkv_compat = compat
    dq_c, dkv_c = path_c
    mx.eval(dq_compat, dkv_compat, dq_c, dkv_c)

    def loss(q_, kv_):
        out = sparse_mla_attention_reference(q_, kv_, indices, d_v=d_v)
        return mx.sum(out * d_out)

    dq_ref, dkv_ref = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq_ref, dkv_ref)

    np.testing.assert_allclose(
        np.array(dq_c).astype(np.float32),
        np.array(dq_compat).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    assert dkv_compat.dtype == mx.float32
    assert dkv_c.dtype == mx.float32
    np.testing.assert_allclose(
        np.array(dkv_compat).astype(np.float32),
        np.array(dkv_c).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(dq_c).astype(np.float32),
        np.array(dq_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(dkv_c).astype(np.float32),
        np.array(dkv_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_path_c_backward_accumulates_duplicate_kv_indices() -> None:
    """Repeated topk hits must scatter-add into final Path C dKV rows."""

    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    rng = np.random.default_rng(58)
    B, S, H, D = 1, 4, 4, 16
    G = 2
    Skv = 8
    d_v = D

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices_np = np.array(
        [
            [
                [[2, 2, 2, 5], [3, 3, 6, 3]],
                [[2, -1, 2, 2], [3, 7, 3, -1]],
                [[1, 2, 2, 4], [3, 3, 3, 0]],
                [[-1, 2, 5, 2], [6, 3, 3, 3]],
            ]
        ],
        dtype=np.int32,
    )
    indices = mx.array(indices_np)
    d_out = mx.array(rng.standard_normal((B, S, H, d_v)).astype(np.float32)).astype(mx.float16)

    grads = sparse_mla_bwd_path_c(q, kv, d_out, indices, d_v=d_v)
    assert grads is not None, "Path C duplicate-index backward must dispatch"
    dq_path_c, dkv_path_c = grads
    mx.eval(dq_path_c, dkv_path_c)

    def loss(q_, kv_):
        out = sparse_mla_attention_reference(q_, kv_, indices, d_v=d_v)
        return mx.sum(out * d_out)

    dq_ref, dkv_ref = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq_ref, dkv_ref)

    dq_path_c_np = np.array(dq_path_c).astype(np.float32)
    dkv_path_c_np = np.array(dkv_path_c).astype(np.float32)
    dq_ref_np = np.array(dq_ref).astype(np.float32)
    dkv_ref_np = np.array(dkv_ref).astype(np.float32)

    assert np.linalg.norm(dkv_ref_np[0, 2, 0]) > 0
    assert np.linalg.norm(dkv_ref_np[0, 3, 1]) > 0
    np.testing.assert_allclose(dq_path_c_np, dq_ref_np, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(dkv_path_c_np, dkv_ref_np, rtol=5e-3, atol=5e-3)


def test_path_c_backward_int64_indices_tail_dim_parity() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    rng = np.random.default_rng(591)
    B, S, H, D, G, topk, Skv, d_v = 1, 2, 2, 16, 1, 4, 8, 8
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int64)
    indices_np[0, 0, 0, 0] = -1
    indices = mx.array(indices_np)
    d_out = mx.array(rng.standard_normal((B, S, H, d_v)).astype(np.float32)).astype(mx.float16)

    grads = sparse_mla_bwd_path_c(q, kv, d_out, indices, d_v=d_v)
    assert grads is not None
    dq_path_c, dkv_path_c = grads

    def loss(q_, kv_):
        out = sparse_mla_attention_reference(q_, kv_, indices, d_v=d_v)
        return mx.sum(out * d_out)

    dq_ref, dkv_ref = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq_path_c, dkv_path_c, dq_ref, dkv_ref)
    np.testing.assert_allclose(
        np.array(dq_path_c).astype(np.float32),
        np.array(dq_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(dkv_path_c).astype(np.float32),
        np.array(dkv_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_path_c_backward_reuses_int32_indices_for_owner_output_route() -> None:
    """Avoid an extra MLX cast/copy on the bwd hot path when indices are int32."""

    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    rng = np.random.default_rng(59)
    B, S, H, D = 1, 4, 2, 16
    G = 1
    topk = 4
    Skv = 8

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices = mx.array(rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32))
    d_out = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)

    assert sparse_mla_path_c._require_supported_indices_no_hidden_cast(
        indices,
        op_name="test",
    ) is indices
    grads = sparse_mla_bwd_path_c(q, kv, d_out, indices)
    assert grads is not None, "Path C backward owner-output kernel must dispatch"
    dq, dkv = grads
    mx.eval(dq, dkv)
    assert dq.shape == q.shape
    assert dkv.shape == kv.shape
    assert dkv.dtype == mx.float32


def test_path_c_backward_accepts_int64_indices_without_hidden_cast() -> None:
    B, S, H, D, G, topk, Skv = 1, 2, 2, 16, 1, 4, 8
    _q = mx.zeros((B, S, H, D), dtype=mx.float16)
    _kv = mx.zeros((B, Skv, G, D), dtype=mx.float16)
    _d_out = mx.zeros((B, S, H, D), dtype=mx.float16)
    indices = mx.array(np.zeros((B, S, G, topk), dtype=np.int64))

    assert sparse_mla_path_c._index_dtype_name(indices, op_name="test") == "int64"
    assert sparse_mla_path_c._require_supported_indices_no_hidden_cast(
        indices,
        op_name="test",
    ) is indices


def test_path_c_topk32_matches_reference_and_compat_bwd() -> None:
    """Path C topk32 keeps fwd/bwd parity with the reference and compat shim."""

    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    rng = np.random.default_rng(123)
    B, S, H, D = 1, 32, 4, 64
    G = 1
    topk = 32
    Skv = 64

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    indices_np[0, 0, 0, :] = -1
    indices_np[0, 1, 0, ::3] = -1
    indices = mx.array(indices_np)
    d_out = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(
        mx.float16
    )

    fwd_c = sparse_mla_fwd_path_c(q, kv, indices)
    assert fwd_c is not None, "Path C topk32 forward must dispatch"
    out_c, lse_c = fwd_c
    out_ref, lse_ref = sparse_mla_attention_reference(q, kv, indices, return_lse=True)
    mx.eval(out_ref, lse_ref, out_c, lse_c)

    np.testing.assert_allclose(
        np.array(out_c).astype(np.float32),
        np.array(out_ref).astype(np.float32),
        rtol=5e-3,
        atol=8e-3,
    )
    np.testing.assert_allclose(
        np.array(lse_c).astype(np.float32),
        np.array(lse_ref).astype(np.float32),
        rtol=5e-3,
        atol=8e-3,
    )

    bwd_compat = sparse_mla_bwd_metal(q, kv, d_out, indices)
    bwd_c = sparse_mla_bwd_path_c(q, kv, d_out, indices)
    assert bwd_compat is not None, "compat topk32 backward must dispatch"
    assert bwd_c is not None, "Path C topk32 backward must dispatch"
    dq_compat, dkv_compat = bwd_compat
    dq_c, dkv_c = bwd_c
    mx.eval(dq_compat, dkv_compat, dq_c, dkv_c)

    np.testing.assert_allclose(
        np.array(dq_c).astype(np.float32),
        np.array(dq_compat).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    assert dkv_compat.dtype == mx.float32
    assert dkv_c.dtype == mx.float32

    def loss(q_, kv_):
        out = sparse_mla_attention_reference(q_, kv_, indices)
        return mx.sum(out * d_out)

    dq_ref, dkv_ref = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq_ref, dkv_ref)
    np.testing.assert_allclose(
        np.array(dq_c).astype(np.float32),
        np.array(dq_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(dkv_c).astype(np.float32),
        np.array(dkv_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_path_c_topk64_matches_reference_and_compat_bwd() -> None:
    """Path C topk64 keeps fwd/bwd parity with the reference and compat shim."""

    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    rng = np.random.default_rng(127)
    B, S, H, D = 1, 32, 4, 64
    G = 1
    topk = 64
    Skv = 128

    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    indices_np[0, 0, 0, :] = -1
    indices_np[0, 1, 0, ::4] = -1
    indices = mx.array(indices_np)
    d_out = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(
        mx.float16
    )

    fwd_c = sparse_mla_fwd_path_c(q, kv, indices)
    assert fwd_c is not None, "Path C topk64 forward must dispatch"
    out_c, lse_c = fwd_c
    out_ref, lse_ref = sparse_mla_attention_reference(q, kv, indices, return_lse=True)
    mx.eval(out_ref, lse_ref, out_c, lse_c)

    np.testing.assert_allclose(
        np.array(out_c).astype(np.float32),
        np.array(out_ref).astype(np.float32),
        rtol=5e-3,
        atol=8e-3,
    )
    np.testing.assert_allclose(
        np.array(lse_c).astype(np.float32),
        np.array(lse_ref).astype(np.float32),
        rtol=5e-3,
        atol=8e-3,
    )

    bwd_compat = sparse_mla_bwd_metal(q, kv, d_out, indices)
    bwd_c = sparse_mla_bwd_path_c(q, kv, d_out, indices)
    assert bwd_compat is not None, "compat topk64 backward must dispatch"
    assert bwd_c is not None, "Path C topk64 backward must dispatch"
    dq_compat, dkv_compat = bwd_compat
    dq_c, dkv_c = bwd_c
    mx.eval(dq_compat, dkv_compat, dq_c, dkv_c)

    np.testing.assert_allclose(
        np.array(dq_c).astype(np.float32),
        np.array(dq_compat).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    assert dkv_compat.dtype == mx.float32
    assert dkv_c.dtype == mx.float32

    def loss(q_, kv_):
        out = sparse_mla_attention_reference(q_, kv_, indices)
        return mx.sum(out * d_out)

    dq_ref, dkv_ref = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq_ref, dkv_ref)
    np.testing.assert_allclose(
        np.array(dq_c).astype(np.float32),
        np.array(dq_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.array(dkv_c).astype(np.float32),
        np.array(dkv_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_bench_strict_failure_still_writes_receipt(monkeypatch, tmp_path) -> None:
    """A red strict gate must leave a JSON receipt for diagnosis."""

    import scripts.bench_tilelang_sparse_mla as bench_sparse_mla

    class _PathCStatus:
        available = True
        reason = "test status"

    class _PathBStatus:
        available = False
        reason = "retired"

    fake_shape = {
        "name": "fake_sparse_mla",
        "B": 1,
        "S": 1,
        "H": 1,
        "D": 16,
        "G": 1,
        "topk": 4,
        "Skv": 4,
    }

    def fake_bench_shape(*_args, **_kwargs):
        return {
            "shape": fake_shape,
            "path_b": {"available": False, "reason": "retired"},
            "path_c": {"available": True, "reason": "ok"},
            "fwd_path_c_ms": {"ok": False},
        }

    out_path = tmp_path / "strict_fail.json"
    monkeypatch.setattr(bench_sparse_mla, "DEFAULT_SHAPES", [fake_shape])
    monkeypatch.setattr(bench_sparse_mla, "_bench_shape", fake_bench_shape)
    monkeypatch.setattr(bench_sparse_mla, "sparse_mla_metal_status", lambda: _PathBStatus())
    monkeypatch.setattr(bench_sparse_mla, "sparse_mla_path_c_status", lambda: _PathCStatus())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_tilelang_sparse_mla.py",
            "--strict",
            "--fwd-only",
            "--shape",
            "fake_sparse_mla",
            "--max-ratio",
            "1.0",
            "--out",
            str(out_path),
        ],
    )

    assert bench_sparse_mla.main() == 2
    payload = json.loads(out_path.read_text())
    assert payload["strict"]["enabled"] is True
    assert payload["strict"]["passed"] is False
    assert payload["strict"]["failures"] == [
        "fake_sparse_mla: forward strict gate failed path_c_ok=False"
    ]
    assert payload["rows"][0]["shape"]["name"] == "fake_sparse_mla"


def test_path_c_backward_lowered_msl_uses_threadgroup_reductions() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_bwd_msl(
        batch=1,
        seq_len=4,
        heads=2,
        qk_dim=16,
        kv_group=1,
        topk=8,
        seq_len_kv=16,
    )
    lowered = msl.lower()
    _assert_bwd_direct_owner_output_msl(msl)
    assert "kernel void" in msl
    assert "thread_position_in_threadgroup" in msl
    assert "device half* q" in msl or "device const half* q" in msl
    assert "device half* kv" in msl or "device const half* kv" in msl
    assert "threadgroup float" in lowered
    assert "threadgroup_barrier" in lowered
    assert "if (((0 <=" not in lowered
    assert "uint tid = thread_position_in_threadgroup.x;" in msl
    assert "uint gid = threadgroup_position_in_grid.x;" in msl
    assert "uint3 threadIdx =" not in msl
    assert "uint3 blockIdx =" not in msl
    assert "uint threads =" in msl
    assert "int tid = int(threadIdx.x);" not in msl
    assert "int gid = int(blockIdx.x);" not in msl
    assert "((int)threadIdx.x)" not in msl
    assert "((int)blockIdx.x)" not in msl
    assert "for (int _tmp" not in msl
    assert "int stride;" not in msl
    assert "half condval" not in msl
    assert "round_id" not in msl
    assert "sumexp <= 0.000000e+00f" in msl
    assert "return;" in msl
    assert "uint q_row_base =" in msl
    assert "uint d_out_row =" in msl
    assert "uint kv_b_base =" in msl
    assert "uint idx_base =" in msl
    assert "uint dkv_pb =" in msl
    assert "indices[idx_base + k]" in msl
    assert "q[q_row_base + d]" in msl
    assert "d_out[d_out_row + d]" in msl
    assert "dq[q_row_base + d]" in msl
    assert "q[((gid * 16) + d)]" not in msl
    assert "d_out[((gid * 16) + d_1)]" not in msl
    assert msl.count("for (uint stride = threads / 2; stride > 0; stride >>= 1)") == 3
    assert "float local_max = -INFINITY;" in msl
    assert "float local_sum = 0.0f;" in msl
    assert "float local_rs = 0.0f;" in msl
    assert "float inv_sum = 1.0f / sumexp;" in msl


def test_path_c_backward_topk4_msl_declares_or_rewrites_stride() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_bwd_msl(
        batch=1,
        seq_len=4,
        heads=2,
        qk_dim=16,
        kv_group=1,
        topk=4,
        seq_len_kv=8,
    )
    _assert_bwd_direct_owner_output_msl(msl)
    assert "round_id" not in msl
    assert "    stride = (" not in msl
    assert msl.count("for (uint stride = threads / 2; stride > 0; stride >>= 1)") == 3


def test_path_c_backward_bench_shape_msl_uses_owner_output_lane_loops() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_bwd_msl(
        batch=2,
        seq_len=128,
        heads=8,
        qk_dim=64,
        kv_group=1,
        topk=16,
        seq_len_kv=128,
    )
    _assert_bwd_direct_owner_output_lane_loops(
        msl,
        qk_dim=64,
        topk=16,
        threads=16,
    )
    assert "indices[((gid >> 3) * 16) + k]" not in msl
    assert "q[((gid * 64) + d)]" not in msl
    assert "d_out[((gid * 64) + d_1)]" not in msl


def test_path_c_backward_topk32_bench_shape_msl_uses_owner_output_lane_loops() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_bwd_msl(
        batch=4,
        seq_len=512,
        heads=8,
        qk_dim=64,
        kv_group=1,
        topk=32,
        seq_len_kv=512,
    )
    _assert_bwd_direct_owner_output_lane_loops(
        msl,
        qk_dim=64,
        topk=32,
        threads=32,
    )
    assert "indices[((gid >> 3) * 32) + k]" not in msl
    assert "q[((gid * 64) + d)]" not in msl
    assert "d_out[((gid * 64) + d_1)]" not in msl


def test_path_c_backward_topk64_bench_shape_msl_uses_owner_output_lane_loops() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_bwd_msl(
        batch=4,
        seq_len=1024,
        heads=8,
        qk_dim=64,
        kv_group=1,
        topk=64,
        seq_len_kv=1024,
    )
    _assert_bwd_direct_owner_output_lane_loops(
        msl,
        qk_dim=64,
        topk=64,
        threads=64,
    )
    assert "indices[((gid >> 3) * 64) + k]" not in msl
    assert "q[((gid * 64) + d)]" not in msl
    assert "d_out[((gid * 64) + d_1)]" not in msl


def test_path_c_backward_tail_dim_msl_uses_kd_element_offsets() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_bwd_msl(
        batch=2,
        seq_len=64,
        heads=4,
        qk_dim=48,
        kv_group=1,
        topk=16,
        seq_len_kv=96,
        d_v=32,
    )
    _assert_bwd_direct_owner_output_lane_loops(
        msl,
        qk_dim=48,
        topk=16,
        threads=16,
        d_v=32,
    )
    assert "indices[((gid >> 2) * 16) + k]" not in msl
    assert "q[((gid * 48) + d)]" not in msl
    assert "d_out[((gid * 32) + d_1)]" not in msl


def test_path_c_forward_lowered_msl_uses_threadgroup_reductions() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_fwd_msl(
        batch=1,
        seq_len=4,
        heads=2,
        qk_dim=16,
        kv_group=1,
        topk=8,
        seq_len_kv=16,
    )
    lowered = msl.lower()
    assert "kernel void" in msl
    assert "thread_position_in_threadgroup" in msl
    assert "device half* q" in msl or "device const half* q" in msl
    assert "device half* kv" in msl or "device const half* kv" in msl
    assert "device half* out" in msl
    assert "device float* lse" in msl
    assert "threadgroup float" in lowered
    assert "threadgroup_barrier" in lowered
    assert "gather_idx[0] < 16" not in lowered
    assert "if (((0 <=" not in lowered
    assert "uint tid = thread_position_in_threadgroup.x;" in msl
    assert "uint gid = threadgroup_position_in_grid.x;" in msl
    assert "uint3 threadIdx =" not in msl
    assert "uint3 blockIdx =" not in msl
    assert "uint threads =" in msl
    assert "int tid = int(threadIdx.x);" not in msl
    assert "int gid = int(blockIdx.x);" not in msl
    assert "((int)threadIdx.x)" not in msl
    assert "((int)blockIdx.x)" not in msl
    assert "row_max == -INFINITY" in msl
    assert "return;" in msl
    assert "kv_row_base" in msl
    assert "int stride;" not in msl
    assert "if (0 <= gather_idx)" not in msl
    assert "half condval" not in msl
    assert msl.count("if (gather_idx < 0) {") == 2
    assert msl.count("continue;") == 2
    assert "round_id" not in msl
    assert msl.count("for (uint stride = threads / 2; stride > 0; stride >>= 1)") == 2
    assert (
        "inv_sum = (sumexp > 0.000000e+00f) ? "
        "(1.000000e+00f / sumexp) : 0.000000e+00f;"
    ) in msl
    assert "lse[gid] = (row_max + log(sumexp));" in msl
    assert "if (0.000000e+00f < sumexp)" not in msl
    assert "t.tvm_mma_sync" not in lowered


def test_path_c_forward_topk4_msl_declares_or_rewrites_stride() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_fwd_msl(
        batch=1,
        seq_len=4,
        heads=2,
        qk_dim=16,
        kv_group=1,
        topk=4,
        seq_len_kv=8,
    )
    assert "round_id" not in msl
    assert "    stride = (" not in msl
    assert msl.count("for (uint stride = threads / 2; stride > 0; stride >>= 1)") == 2


def test_path_c_forward_bench_shape_msl_uses_path_b_lane_loops() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_fwd_msl(
        batch=2,
        seq_len=128,
        heads=8,
        qk_dim=64,
        kv_group=1,
        topk=16,
        seq_len_kv=128,
    )
    assert "uint tid = thread_position_in_threadgroup.x;" in msl
    assert "uint gid = threadgroup_position_in_grid.x;" in msl
    assert "uint3 threadIdx =" not in msl
    assert "uint3 blockIdx =" not in msl
    assert "uint threads = 16;" in msl
    assert "int tid = int(threadIdx.x);" not in msl
    assert "int gid = int(blockIdx.x);" not in msl
    assert "((int)threadIdx.x)" not in msl
    assert "((int)blockIdx.x)" not in msl
    assert "for (int _tmp" not in msl
    assert "for (uint k = tid; k < 16; k += threads)" in msl
    assert "{\n    uint k = tid;" not in msl
    assert "for (int k = tid;" not in msl
    assert "for (uint d = tid; d < 64; d += threads)" in msl
    assert "scores[k]" in msl
    assert "uint q_row_base =" in msl
    assert "uint kv_b_base =" in msl
    assert "uint idx_base =" in msl
    assert "uint out_row =" in msl
    assert "indices[idx_base + k]" in msl
    assert "kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim" in msl
    assert "q[q_row_base + d]" in msl
    assert "kv[kv_row_base + d]" in msl
    assert "kv_row_base_1" not in msl
    assert "out[out_row + d]" in msl
    assert "indices[((gid >> 3) * 16) + k]" not in msl
    assert "out[(gid * 64) + d]" not in msl
    assert "out[(((gid * 64) + (_tmp_4 * 16)) + tid)]" not in msl
    assert "int stride;" not in msl
    assert "if (0 <= gather_idx)" not in msl
    assert "half condval" not in msl
    assert msl.count("if (gather_idx < 0) {") == 2
    assert msl.count("continue;") == 2
    assert "round_id" not in msl
    assert msl.count("for (uint stride = threads / 2; stride > 0; stride >>= 1)") == 2
    assert (
        "inv_sum = (sumexp > 0.000000e+00f) ? "
        "(1.000000e+00f / sumexp) : 0.000000e+00f;"
    ) in msl
    assert "lse[gid] = (row_max + log(sumexp));" in msl
    assert "if (0.000000e+00f < sumexp)" not in msl


def test_path_c_forward_topk32_bench_shape_msl_uses_path_b_lane_loops() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_fwd_msl(
        batch=4,
        seq_len=512,
        heads=8,
        qk_dim=64,
        kv_group=1,
        topk=32,
        seq_len_kv=512,
    )
    assert "uint tid = thread_position_in_threadgroup.x;" in msl
    assert "uint gid = threadgroup_position_in_grid.x;" in msl
    assert "uint3 threadIdx =" not in msl
    assert "uint3 blockIdx =" not in msl
    assert "uint threads = 32;" in msl
    assert "int tid = int(threadIdx.x);" not in msl
    assert "int gid = int(blockIdx.x);" not in msl
    assert "((int)threadIdx.x)" not in msl
    assert "((int)blockIdx.x)" not in msl
    assert "_tmp" not in msl
    assert "round_id" not in msl
    assert "half condval" not in msl
    assert "int stride;" not in msl
    assert "if (0 <= gather_idx)" not in msl
    assert "for (uint k = tid; k < 32; k += threads)" in msl
    assert "for (uint d = tid; d < 64; d += threads)" in msl
    assert "uint q_row_base =" in msl
    assert "uint kv_b_base =" in msl
    assert "uint idx_base =" in msl
    assert "uint out_row =" in msl
    assert "indices[idx_base + k]" in msl
    assert "kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim" in msl
    assert "q[q_row_base + d]" in msl
    assert "kv[kv_row_base + d]" in msl
    assert "kv_row_base_1" not in msl
    assert "out[out_row + d]" in msl
    assert "indices[((gid >> 3) * 32) + k]" not in msl
    assert "out[(gid * 64) + d]" not in msl
    assert "out[(((gid * 64) + (_tmp_4 * 32)) + tid)]" not in msl
    assert msl.count("if (gather_idx < 0) {") == 2
    assert msl.count("continue;") == 2
    assert msl.count("for (uint stride = threads / 2; stride > 0; stride >>= 1)") == 2
    assert (
        "inv_sum = (sumexp > 0.000000e+00f) ? "
        "(1.000000e+00f / sumexp) : 0.000000e+00f;"
    ) in msl
    assert "lse[gid] = (row_max + log(sumexp));" in msl
    assert "if (0.000000e+00f < sumexp)" not in msl


def test_path_c_forward_topk64_bench_shape_msl_uses_path_b_lane_loops() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    msl = dump_lowered_fwd_msl(
        batch=4,
        seq_len=1024,
        heads=8,
        qk_dim=64,
        kv_group=1,
        topk=64,
        seq_len_kv=1024,
    )
    assert "uint tid = thread_position_in_threadgroup.x;" in msl
    assert "uint gid = threadgroup_position_in_grid.x;" in msl
    assert "uint3 threadIdx =" not in msl
    assert "uint3 blockIdx =" not in msl
    assert "uint threads = 64;" in msl
    assert "int tid = int(threadIdx.x);" not in msl
    assert "int gid = int(blockIdx.x);" not in msl
    assert "((int)threadIdx.x)" not in msl
    assert "((int)blockIdx.x)" not in msl
    assert "_tmp" not in msl
    assert "round_id" not in msl
    assert "half condval" not in msl
    assert "int stride;" not in msl
    assert "if (0 <= gather_idx)" not in msl
    assert "for (uint k = tid; k < 64; k += threads)" in msl
    assert "for (uint d = tid; d < 64; d += threads)" in msl
    assert "uint q_row_base =" in msl
    assert "uint kv_b_base =" in msl
    assert "uint idx_base =" in msl
    assert "uint out_row =" in msl
    assert "indices[idx_base + k]" in msl
    assert "kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim" in msl
    assert "q[q_row_base + d]" in msl
    assert "kv[kv_row_base + d]" in msl
    assert "kv_row_base_1" not in msl
    assert "out[out_row + d]" in msl
    assert "indices[((gid >> 3) * 64) + k]" not in msl
    assert "out[(gid * 64) + d]" not in msl
    assert "out[(((gid * 64) + (_tmp_4 * 64)) + tid)]" not in msl
    assert msl.count("if (gather_idx < 0) {") == 2
    assert msl.count("continue;") == 2
    assert msl.count("for (uint stride = threads / 2; stride > 0; stride >>= 1)") == 2
    assert (
        "inv_sum = (sumexp > 0.000000e+00f) ? "
        "(1.000000e+00f / sumexp) : 0.000000e+00f;"
    ) in msl
    assert "lse[gid] = (row_max + log(sumexp));" in msl
    assert "if (0.000000e+00f < sumexp)" not in msl


def test_path_c_forward_bench_shape_dispatch_smoke() -> None:
    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    rng = np.random.default_rng(43)
    B, S, H, D = 2, 128, 8, 64
    G = 1
    topk = 16
    Skv = 128
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32)).astype(mx.float16)
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32)).astype(mx.float16)
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    indices_np[0, 0, 0, :] = -1
    indices = mx.array(indices_np)

    result = sparse_mla_fwd_path_c(q, kv, indices)
    assert result is not None, "bench-shape Path C forward must compile and dispatch"
    out, lse = result
    mx.eval(out, lse)
    assert out.shape == (B, S, H, D)
    assert lse.shape == (B, S, H)


def test_path_c_forward_lowering_does_not_build_mlx_fast_alias() -> None:
    """Path C lowering helpers must not build an mx.fast Path B alias."""

    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    kernel, lowering = _fwd_kernel_for(1, 8, 2, 16, 1, 2, 4, 8, 16, 4)

    assert kernel is None
    assert "kernel void" not in lowering.body
    assert "threadgroup_position_in_grid.x" in lowering.body
    assert "thread_position_in_threadgroup.x" in lowering.body


@pytest.mark.parametrize("topk", [16, 32, 64])
def test_path_c_lowering_dispatch_grid_matches_mlx_total_thread_contract(topk: int) -> None:
    """MLX launches total threads, so TileLang's block grid must be scaled once."""

    status = sparse_mla_path_c_status()
    if not status.available:
        pytest.skip(status.reason)

    batch, seq_len, heads, qk_dim = 2, 128, 8, 64
    kv_group, seq_len_kv, d_v = 1, 128, 64
    head_kv = heads // kv_group
    threads = _threadgroup_size(topk)
    lanes = batch * seq_len * heads
    args = (
        batch,
        seq_len,
        heads,
        qk_dim,
        kv_group,
        head_kv,
        topk,
        seq_len_kv,
        d_v,
        threads,
    )

    _kernel, fwd_lowering = _fwd_kernel_for(*args)
    _kernel, bwd_lowering = _bwd_direct_lowering_for(*args)

    for lowering in (fwd_lowering, bwd_lowering):
        assert lowering.grid == (lanes, 1, 1)
        assert lowering.threadgroup == (threads, 1, 1)
        assert _mlx_total_thread_grid(lowering) == (lanes * threads, 1, 1)


def test_apply_backward_through_reference_fallback() -> None:
    """``sparse_mla_apply`` still propagates gradients after Path B retirement."""

    rng = np.random.default_rng(29)
    B, S, H, D = 1, 4, 2, 16
    G = 1
    topk = 4
    Skv = 8
    q = mx.array(rng.standard_normal((B, S, H, D)).astype(np.float32))
    kv = mx.array(rng.standard_normal((B, Skv, G, D)).astype(np.float32))
    indices = mx.array(rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32))

    def loss(q_, kv_):
        out = cast(mx.array, sparse_mla_apply(q_, kv_, indices))
        return mx.sum(out * out)

    dq, dkv = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(dq, dkv)
    assert np.all(np.isfinite(np.array(dq)))
    assert np.all(np.isfinite(np.array(dkv)))
    assert np.linalg.norm(np.array(dq)) > 0
    assert np.linalg.norm(np.array(dkv)) > 0
