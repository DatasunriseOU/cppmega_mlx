"""Tests for the Path B topk_selector port.

The Path B Metal kernel is now available via direct-MSL bypass (see
``cppmega_mlx/nn/_tilelang/topk_selector.py``). The previous TileLang
``shared.dyn`` / ``LowerTileOp`` blockers are bypassed by emitting MSL
through ``mx.fast.metal_kernel`` directly with static threadgroup arrays.

The tests verify:

1. The pure-MLX reference returns the correct top-k indices (set-equality
   to a NumPy oracle) for a sweep of (B, T, k) shapes.
2. The direct-MSL Path B kernel produces the same set of indices as the
   reference (set-equality, since both partition contracts are
   order-unspecified).
3. Output shape and dtype match the cppmega source contract.
4. Edge cases (k=1, k=seq_len, and start/end masking) are exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang.topk_selector import (  # noqa: E402
    PathBStatus,
    topk_selector,
    topk_selector_metal,
    topk_selector_path_b_status,
    topk_selector_reference,
)


# ---------------------------------------------------------------------------
# NumPy oracle.
# ---------------------------------------------------------------------------

def _np_topk_indices(
    scores: np.ndarray,
    k: int,
    *,
    starts: np.ndarray | None = None,
    ends: np.ndarray | None = None,
) -> list[set[int]]:
    """Return per-row sets of indices of the k largest values."""

    out: list[set[int]] = []
    B, T = scores.shape
    for b in range(B):
        row = scores[b].copy()
        if starts is not None or ends is not None:
            s = 0 if starts is None else int(starts[b])
            e = T if ends is None else int(ends[b])
            mask = np.ones(T, dtype=bool)
            mask[:s] = False
            mask[e:] = False
            row = np.where(mask, row, np.float32("-inf"))
        order = np.argsort(-row, kind="stable")[:k]
        out.append(set(int(x) for x in order))
    return out


def _to_index_sets(indices: mx.array) -> list[set[int]]:
    mx.eval(indices)
    arr = np.asarray(indices)
    return [set(int(x) for x in row) for row in arr]


# ---------------------------------------------------------------------------
# Pure-MLX reference parity.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("seq_len", [64, 512, 2048])
@pytest.mark.parametrize("k", [1, 8, 32])
def test_reference_matches_numpy_oracle(batch: int, seq_len: int, k: int) -> None:
    if k > seq_len:
        pytest.skip("k must be <= seq_len")
    rng = np.random.default_rng(seed=batch * 1000 + seq_len + k)
    scores_np = rng.standard_normal((batch, seq_len)).astype(np.float32)
    # Make values unique so set-equality is well defined (no ties to resolve).
    scores_np = scores_np + 1e-3 * np.arange(scores_np.size).reshape(scores_np.shape).astype(np.float32) / scores_np.size

    out = topk_selector_reference(mx.array(scores_np), k)
    actual = _to_index_sets(out)
    expected = _np_topk_indices(scores_np, k)

    assert actual == expected
    assert out.dtype == mx.int32
    assert tuple(out.shape) == (batch, k)


def test_reference_handles_full_seq_topk() -> None:
    rng = np.random.default_rng(0)
    seq_len = 32
    scores = mx.array(rng.standard_normal((2, seq_len)).astype(np.float32))
    out = topk_selector_reference(scores, k=seq_len)
    assert tuple(out.shape) == (2, seq_len)
    # All indices [0, seq_len) must appear in each row.
    sets = _to_index_sets(out)
    for s in sets:
        assert s == set(range(seq_len))


def test_reference_k_one_returns_argmax() -> None:
    scores = mx.array(np.array([
        [0.5, 1.5, -2.0, 3.0],
        [-1.0, -2.0, 0.0, -0.5],
    ], dtype=np.float32))
    out = topk_selector_reference(scores, k=1)
    mx.eval(out)
    arr = np.asarray(out).reshape(-1)
    # Row 0 max at idx 3, row 1 max at idx 2.
    assert arr[0] == 3
    assert arr[1] == 2


def test_reference_starts_ends_mask_excludes_outside_range() -> None:
    rng = np.random.default_rng(7)
    batch, seq_len, k = 3, 32, 4
    scores_np = rng.standard_normal((batch, seq_len)).astype(np.float32)
    starts_np = np.array([4, 0, 8], dtype=np.int32)
    ends_np = np.array([16, 8, 24], dtype=np.int32)

    out = topk_selector_reference(
        mx.array(scores_np),
        k,
        starts=mx.array(starts_np),
        ends=mx.array(ends_np),
    )
    actual = _to_index_sets(out)
    expected = _np_topk_indices(scores_np, k, starts=starts_np, ends=ends_np)
    assert actual == expected
    for b in range(batch):
        for idx in actual[b]:
            assert starts_np[b] <= idx < ends_np[b], (
                f"idx {idx} not in [{starts_np[b]}, {ends_np[b]})"
            )


# ---------------------------------------------------------------------------
# Shape / dtype invariants.
# ---------------------------------------------------------------------------

def test_reference_output_dtype_is_int32_for_float16_input() -> None:
    scores = mx.array(np.zeros((2, 8), dtype=np.float16))
    out = topk_selector_reference(scores, k=2)
    assert out.dtype == mx.int32


def test_reference_output_dtype_is_int32_for_bfloat16_input() -> None:
    scores = (mx.zeros((2, 8), dtype=mx.bfloat16) + mx.array(0.1))
    out = topk_selector_reference(scores, k=2)
    assert out.dtype == mx.int32


def test_reference_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        topk_selector_reference(mx.zeros((3,)), k=1)


def test_reference_rejects_invalid_k() -> None:
    scores = mx.zeros((2, 8))
    with pytest.raises(ValueError):
        topk_selector_reference(scores, k=0)
    with pytest.raises(ValueError):
        topk_selector_reference(scores, k=9)


# ---------------------------------------------------------------------------
# Public entry point + status seam.
# ---------------------------------------------------------------------------

def test_public_topk_selector_matches_reference() -> None:
    rng = np.random.default_rng(11)
    scores_np = rng.standard_normal((2, 64)).astype(np.float32)
    out_pub = topk_selector(mx.array(scores_np), k=8)
    out_ref = topk_selector_reference(mx.array(scores_np), k=8)
    assert _to_index_sets(out_pub) == _to_index_sets(out_ref)


def test_path_b_status_reports_available_on_metal() -> None:
    status = topk_selector_path_b_status()
    assert isinstance(status, PathBStatus)
    if mx.metal.is_available():
        assert status.available is True
        assert "direct-MSL" in status.reason or "available" in status.reason
    else:
        assert status.available is False


def test_path_b_status_reason_is_stable() -> None:
    s1 = topk_selector_path_b_status()
    s2 = topk_selector_path_b_status()
    assert s1.reason == s2.reason


# ---------------------------------------------------------------------------
# Direct-MSL Path B kernel parity (replaces the previous "blocked" placeholder).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("seq_len", [64, 512, 2048])
@pytest.mark.parametrize("k", [1, 8, 32])
def test_path_b_forward_parity_set_equality(batch: int, seq_len: int, k: int) -> None:
    if k > seq_len:
        pytest.skip("k must be <= seq_len")
    rng = np.random.default_rng(seed=batch * 7919 + seq_len * 31 + k)
    scores_np = rng.standard_normal((batch, seq_len)).astype(np.float32)
    # Make values unique so set-equality is well defined (no ties to resolve).
    scores_np = (
        scores_np
        + 1e-3
        * np.arange(scores_np.size).reshape(scores_np.shape).astype(np.float32)
        / scores_np.size
    )
    scores = mx.array(scores_np)
    out_msl = topk_selector_metal(scores, k)
    assert out_msl is not None, "direct-MSL Path B kernel must dispatch"
    out_ref = topk_selector_reference(scores, k)
    actual = _to_index_sets(out_msl)
    expected = _to_index_sets(out_ref)
    assert actual == expected
    assert out_msl.dtype == mx.int32
    assert tuple(out_msl.shape) == (batch, k)


def test_path_b_forward_parity_with_starts_ends() -> None:
    rng = np.random.default_rng(7)
    batch, seq_len, k = 3, 32, 4
    scores_np = rng.standard_normal((batch, seq_len)).astype(np.float32)
    starts_np = np.array([4, 0, 8], dtype=np.int32)
    ends_np = np.array([16, 8, 24], dtype=np.int32)
    scores = mx.array(scores_np)
    starts = mx.array(starts_np)
    ends = mx.array(ends_np)

    out_msl = topk_selector_metal(scores, k, starts=starts, ends=ends)
    assert out_msl is not None
    out_ref = topk_selector_reference(scores, k, starts=starts, ends=ends)
    actual = _to_index_sets(out_msl)
    expected = _to_index_sets(out_ref)
    assert actual == expected


def test_public_entry_point_metal_backend_dispatches_kernel() -> None:
    rng = np.random.default_rng(11)
    scores = mx.array(rng.standard_normal((2, 64)).astype(np.float32))
    # backend='metal' should not raise now.
    out_metal = topk_selector(scores, k=8, backend="metal")
    out_ref = topk_selector(scores, k=8, backend="mlx")
    assert _to_index_sets(out_metal) == _to_index_sets(out_ref)
