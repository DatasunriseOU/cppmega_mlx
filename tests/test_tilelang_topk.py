"""Tests for the Path B/Path C topk_selector ports.

The Path B Metal kernel is now available via direct-MSL bypass (see
``cppmega_mlx/nn/_tilelang/topk_selector.py``). The previous TileLang
``shared.dyn`` / ``LowerTileOp`` blockers are bypassed by emitting MSL
through ``mx.fast.metal_kernel`` directly with static threadgroup arrays.
Path C uses a TileLang DSL PrimFunc lowered to Metal, then launched through
``mx.fast.metal_kernel`` with the TileLang-generated kernel body.

The tests verify:

1. The pure-MLX reference returns the correct top-k indices (set-equality
   to a NumPy oracle) for a sweep of (B, T, k) shapes.
2. The direct-MSL Path B and TileLang DSL Path C kernels produce the same set
   of indices as the reference (set-equality, since all partition contracts are
   order-unspecified).
3. Output shape and dtype match the cppmega source contract.
4. Edge cases (k=1, k=seq_len, and start/end masking) are exercised.
"""

from __future__ import annotations

import numpy as np  # type: ignore[reportMissingImports]
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang.topk_selector import (  # noqa: E402
    PathBStatus,
    PathCStatus,
    _path_c_kernel_for,
    _path_c_threads_for,
    topk_selector,
    topk_selector_metal,
    topk_selector_path_b_status,
    topk_selector_path_c_status,
    topk_selector_reference,
    topk_selector_tilelang,
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
        valid_count = T
        if starts is not None or ends is not None:
            s = 0 if starts is None else int(np.clip(starts[b], 0, T))
            e = T if ends is None else int(np.clip(ends[b], 0, T))
            valid_count = max(0, e - s)
            mask = np.ones(T, dtype=bool)
            mask[:s] = False
            mask[e:] = False
            row = np.where(mask, row, np.float32("-inf"))
        order = np.argsort(-row, kind="stable")[:k].astype(np.int32)
        if valid_count < k:
            order[valid_count:] = -1
        out.append(set(int(x) for x in order))
    return out


def _to_index_sets(indices: mx.array) -> list[set[int]]:
    mx.eval(indices)
    arr = np.asarray(indices)
    return [set(int(x) for x in row) for row in arr]


def _acceptance_scores(
    *,
    batch: int,
    seq_len: int,
    k: int,
    dtype: mx.Dtype,
    seed: int,
) -> mx.array:
    """Build top-k fixtures with a hard gap at the k boundary.

    bfloat16 has few mantissa bits, so random normal inputs can create ties at
    the top-k boundary after dtype conversion. The selector contract does not
    define tie-breaking; these acceptance fixtures verify dispatch/parity for
    the production shapes without depending on implementation-specific ties.
    """

    rng = np.random.default_rng(seed)
    scores_np = np.full((batch, seq_len), -1.0, dtype=np.float32)
    for row in range(batch):
        top = rng.choice(seq_len, size=k, replace=False)
        # Exactly k entries are above the rest; ties within the top-k set are
        # harmless because set membership is the asserted contract.
        scores_np[row, top] = 1.0
    return mx.array(scores_np).astype(dtype)


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


def test_reference_short_and_empty_intervals_use_negative_one_sentinel() -> None:
    scores = mx.array(np.array([
        [9.0, 8.0, 7.0, 6.0, 5.0],
        [4.0, 3.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
    ], dtype=np.float32))
    starts = mx.array(np.array([1, 2, 4], dtype=np.int32))
    ends = mx.array(np.array([3, 2, 99], dtype=np.int32))

    out = topk_selector_reference(scores, k=4, starts=starts, ends=ends)
    mx.eval(out)
    arr = np.asarray(out)

    assert set(int(x) for x in arr[0]) == {1, 2, -1}
    assert arr[1].tolist() == [-1, -1, -1, -1]
    assert set(int(x) for x in arr[2]) == {4, -1}


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


def test_path_c_status_reports_tilelang_metal_state() -> None:
    status = topk_selector_path_c_status()
    assert isinstance(status, PathCStatus)
    if mx.metal.is_available():
        assert isinstance(status.available, bool)
        assert status.reason
    else:
        assert status.available is False


def test_path_c_status_reason_is_stable() -> None:
    s1 = topk_selector_path_c_status()
    s2 = topk_selector_path_c_status()
    assert s1.reason == s2.reason


def test_path_c_lowering_uses_single_merge_active_branch() -> None:
    threads = _path_c_threads_for(8)
    _, lowering = _path_c_kernel_for(1, 64, 8, threads, "float32")
    assert lowering.body.count("threadgroup_barrier") == 2
    assert lowering.body.count("stride * 2") == 1


def test_path_c_lowering_keeps_k32_insertion_scan_unbroken_for_perf() -> None:
    threads = _path_c_threads_for(32)
    _, lowering = _path_c_kernel_for(1, 512, 32, threads, "float32")
    assert "break;" not in lowering.body


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


def test_path_b_forward_parity_with_short_and_empty_intervals() -> None:
    rng = np.random.default_rng(19)
    batch, seq_len, k = 4, 64, 8
    scores = mx.array(rng.standard_normal((batch, seq_len)).astype(np.float32))
    starts = mx.array(np.array([0, 7, 32, 63], dtype=np.int32))
    ends = mx.array(np.array([3, 7, 65, 64], dtype=np.int32))

    out_msl = topk_selector_metal(scores, k, starts=starts, ends=ends)
    assert out_msl is not None
    out_ref = topk_selector_reference(scores, k, starts=starts, ends=ends)
    assert _to_index_sets(out_msl) == _to_index_sets(out_ref)


def test_path_b_forward_parity_with_negative_user_ends() -> None:
    scores = mx.array(np.array([[5.0, 4.0, 3.0, 2.0]], dtype=np.float32))
    starts = mx.array(np.array([0], dtype=np.int32))
    ends = mx.array(np.array([-1], dtype=np.int32))

    out_msl = topk_selector_metal(scores, k=2, starts=starts, ends=ends)
    assert out_msl is not None
    out_ref = topk_selector_reference(scores, k=2, starts=starts, ends=ends)
    mx.eval(out_msl, out_ref)
    assert np.asarray(out_msl).tolist() == [[-1, -1]]
    assert np.asarray(out_msl).tolist() == np.asarray(out_ref).tolist()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_path_b_acceptance_shape_512x64_dispatches(dtype: mx.Dtype) -> None:
    scores = _acceptance_scores(batch=4, seq_len=512, k=64, dtype=dtype, seed=512)
    out_msl = topk_selector_metal(scores, k=64)
    assert out_msl is not None
    out_ref = topk_selector_reference(scores, k=64)
    assert _to_index_sets(out_msl) == _to_index_sets(out_ref)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_path_b_acceptance_shape_4096x256_dispatches(dtype: mx.Dtype) -> None:
    scores = _acceptance_scores(batch=1, seq_len=4096, k=256, dtype=dtype, seed=4096)
    out_msl = topk_selector_metal(scores, k=256)
    assert out_msl is not None
    out_ref = topk_selector_reference(scores, k=256)
    assert _to_index_sets(out_msl) == _to_index_sets(out_ref)


def test_path_b_full_seq_topk_dispatches() -> None:
    rng = np.random.default_rng(32)
    scores = mx.array(rng.standard_normal((2, 32)).astype(np.float32))
    out_msl = topk_selector_metal(scores, k=32)
    assert out_msl is not None
    out_ref = topk_selector_reference(scores, k=32)
    assert _to_index_sets(out_msl) == _to_index_sets(out_ref)


def test_public_entry_point_metal_backend_dispatches_kernel() -> None:
    rng = np.random.default_rng(11)
    scores = mx.array(rng.standard_normal((2, 64)).astype(np.float32))
    # backend='metal' should not raise now.
    out_metal = topk_selector(scores, k=8, backend="metal")
    out_ref = topk_selector(scores, k=8, backend="mlx")
    assert _to_index_sets(out_metal) == _to_index_sets(out_ref)


# ---------------------------------------------------------------------------
# TileLang DSL Path C kernel parity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch,seq_len,k",
    [(2, 64, 4), (1, 64, 8), (1, 512, 32), (4, 512, 64)],
)
def test_path_c_forward_parity_set_equality(batch: int, seq_len: int, k: int) -> None:
    rng = np.random.default_rng(seed=batch * 3571 + seq_len * 17 + k)
    scores_np = rng.standard_normal((batch, seq_len)).astype(np.float32)
    scores_np = (
        scores_np
        + 1e-3
        * np.arange(scores_np.size).reshape(scores_np.shape).astype(np.float32)
        / scores_np.size
    )
    scores = mx.array(scores_np)
    out_c = topk_selector_tilelang(scores, k)
    assert out_c is not None, topk_selector_path_c_status().reason
    out_ref = topk_selector_reference(scores, k)
    out_b = topk_selector_metal(scores, k)
    assert out_b is not None
    assert _to_index_sets(out_c) == _to_index_sets(out_ref)
    assert _to_index_sets(out_c) == _to_index_sets(out_b)
    assert out_c.dtype == mx.int32
    assert tuple(out_c.shape) == (batch, k)


def test_path_c_forward_parity_with_starts_ends() -> None:
    rng = np.random.default_rng(20260504)
    batch, seq_len, k = 3, 64, 4
    scores_np = rng.standard_normal((batch, seq_len)).astype(np.float32)
    scores_np += (
        np.arange(scores_np.size, dtype=np.float32).reshape(scores_np.shape)
        * np.float32(1e-5)
    )
    starts_np = np.array([4, 0, 8], dtype=np.int32)
    ends_np = np.array([40, 16, 48], dtype=np.int32)
    scores = mx.array(scores_np)
    starts = mx.array(starts_np)
    ends = mx.array(ends_np)

    out_c = topk_selector_tilelang(scores, k, starts=starts, ends=ends)
    assert out_c is not None, topk_selector_path_c_status().reason
    out_ref = topk_selector_reference(scores, k, starts=starts, ends=ends)
    out_b = topk_selector_metal(scores, k, starts=starts, ends=ends)
    assert out_b is not None
    assert _to_index_sets(out_c) == _to_index_sets(out_ref)
    assert _to_index_sets(out_c) == _to_index_sets(out_b)


def test_path_c_forward_parity_with_short_and_empty_intervals() -> None:
    scores = mx.array(np.array([
        [5.0, 4.0, 3.0, 2.0],
        [0.0, 1.0, 2.0, 3.0],
    ], dtype=np.float32))
    starts = mx.array(np.array([0, 3], dtype=np.int32))
    ends = mx.array(np.array([-1, 99], dtype=np.int32))

    out_c = topk_selector_tilelang(scores, k=2, starts=starts, ends=ends)
    assert out_c is not None, topk_selector_path_c_status().reason
    out_ref = topk_selector_reference(scores, k=2, starts=starts, ends=ends)
    mx.eval(out_c, out_ref)
    assert np.asarray(out_c).tolist() == [[-1, -1], [3, -1]]
    assert np.asarray(out_c).tolist() == np.asarray(out_ref).tolist()


def test_path_c_matches_path_b_exact_order_for_ties() -> None:
    scores = mx.array(np.array([
        [1.0, 3.0, 3.0, 2.0, 3.0, 0.0, -1.0, 3.0],
        [5.0, 5.0, 4.0, 4.0, 5.0, 5.0, 3.0, 2.0],
    ], dtype=np.float32))

    out_c = topk_selector_tilelang(scores, k=6)
    out_b = topk_selector_metal(scores, k=6)
    assert out_c is not None, topk_selector_path_c_status().reason
    assert out_b is not None
    mx.eval(out_c, out_b)

    expected = [
        [1, 2, 4, 7, 3, 0],
        [0, 1, 4, 5, 2, 3],
    ]
    assert np.asarray(out_b).tolist() == expected
    assert np.asarray(out_c).tolist() == expected


def test_path_c_matches_path_b_exact_order_for_masked_ties_and_sentinels() -> None:
    scores = mx.array(np.array([
        [9.0, 7.0, 7.0, 6.0, 7.0, 5.0, 7.0, 4.0],
        [1.0, 8.0, 8.0, 8.0, 3.0, 8.0, 2.0, 8.0],
        [0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 4.0],
    ], dtype=np.float32))
    starts = mx.array(np.array([1, 2, 6], dtype=np.int32))
    ends = mx.array(np.array([7, 6, 8], dtype=np.int32))

    out_c = topk_selector_tilelang(scores, k=4, starts=starts, ends=ends)
    out_b = topk_selector_metal(scores, k=4, starts=starts, ends=ends)
    assert out_c is not None, topk_selector_path_c_status().reason
    assert out_b is not None
    mx.eval(out_c, out_b)

    expected = [
        [1, 2, 4, 6],
        [2, 3, 5, 4],
        [6, 7, -1, -1],
    ]
    assert np.asarray(out_b).tolist() == expected
    assert np.asarray(out_c).tolist() == expected


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_path_c_acceptance_shape_512x64_dispatches(dtype: mx.Dtype) -> None:
    scores = _acceptance_scores(batch=4, seq_len=512, k=64, dtype=dtype, seed=1512)
    out_c = topk_selector_tilelang(scores, k=64)
    assert out_c is not None, topk_selector_path_c_status().reason
    out_ref = topk_selector_reference(scores, k=64)
    out_b = topk_selector_metal(scores, k=64)
    assert out_b is not None
    assert _to_index_sets(out_c) == _to_index_sets(out_ref)
    assert _to_index_sets(out_c) == _to_index_sets(out_b)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_path_c_acceptance_shape_4096x256_dispatches(dtype: mx.Dtype) -> None:
    scores = _acceptance_scores(batch=1, seq_len=4096, k=256, dtype=dtype, seed=14096)
    out_c = topk_selector_tilelang(scores, k=256)
    assert out_c is not None, topk_selector_path_c_status().reason
    out_ref = topk_selector_reference(scores, k=256)
    out_b = topk_selector_metal(scores, k=256)
    assert out_b is not None
    assert _to_index_sets(out_c) == _to_index_sets(out_ref)
    assert _to_index_sets(out_c) == _to_index_sets(out_b)


def test_public_entry_point_tilelang_backend_dispatches_kernel() -> None:
    rng = np.random.default_rng(29)
    scores_np = rng.standard_normal((2, 64)).astype(np.float32)
    scores_np += np.arange(scores_np.size, dtype=np.float32).reshape(scores_np.shape) * 1e-5
    scores = mx.array(scores_np)
    out_c = topk_selector(scores, k=8, backend="tilelang")
    out_ref = topk_selector(scores, k=8, backend="mlx")
    assert _to_index_sets(out_c) == _to_index_sets(out_ref)


def test_public_entry_point_auto_prefers_tilelang_path_c(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = mx.array(np.arange(64, dtype=np.float32).reshape(1, 64))
    sentinel = mx.array(np.array([[63, 62, 61, 60, 59, 58, 57, 56]], dtype=np.int32))

    def fake_path_c(
        scores_arg: mx.array,
        k_arg: int,
        *,
        starts: mx.array | None = None,
        ends: mx.array | None = None,
    ) -> mx.array:
        assert scores_arg is scores
        assert k_arg == 8
        assert starts is None
        assert ends is None
        return sentinel

    def fail_path_b(*args: object, **kwargs: object) -> None:
        raise AssertionError("receipt-backed backend='auto' must try TileLang Path C before Path B")

    monkeypatch.setattr("cppmega_mlx.nn._tilelang.topk_selector.topk_selector_tilelang", fake_path_c)
    monkeypatch.setattr("cppmega_mlx.nn._tilelang.topk_selector.topk_selector_metal", fail_path_b)
    out = topk_selector(scores, k=8, backend="auto")
    assert np.asarray(out).tolist() == [[63, 62, 61, 60, 59, 58, 57, 56]]


def test_public_entry_point_auto_prefers_tilelang_path_c_for_bfloat16_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = mx.array(np.arange(4 * 2048, dtype=np.float32).reshape(4, 2048)).astype(
        mx.bfloat16
    )
    sentinel = mx.full((4, 64), 7, dtype=mx.int32)

    def fake_path_c(
        scores_arg: mx.array,
        k_arg: int,
        *,
        starts: mx.array | None = None,
        ends: mx.array | None = None,
    ) -> mx.array:
        assert scores_arg is scores
        assert k_arg == 64
        assert starts is None
        assert ends is None
        return sentinel

    def fail_path_b(*args: object, **kwargs: object) -> None:
        raise AssertionError("bf16 receipt-backed backend='auto' must try TileLang Path C")

    monkeypatch.setattr("cppmega_mlx.nn._tilelang.topk_selector.topk_selector_tilelang", fake_path_c)
    monkeypatch.setattr("cppmega_mlx.nn._tilelang.topk_selector.topk_selector_metal", fail_path_b)
    out = topk_selector(scores, k=64, backend="auto")
    assert np.asarray(out).tolist() == [[7] * 64] * 4


def test_public_entry_point_auto_uses_path_b_first_for_unreceipted_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = mx.array(np.arange(16, dtype=np.float32).reshape(1, 16))
    sentinel = mx.array(np.array([[11, 10, 9, 8]], dtype=np.int32))

    def fail_path_c(*args: object, **kwargs: object) -> None:
        raise AssertionError("unreceipted backend='auto' must not try TileLang Path C first")

    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_tilelang",
        fail_path_c,
    )
    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_metal",
        lambda *args, **kwargs: sentinel,
    )
    out = topk_selector(scores, k=4, backend="auto")
    assert np.asarray(out).tolist() == [[11, 10, 9, 8]]


def test_public_entry_point_auto_uses_path_b_first_for_masked_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = mx.array(np.arange(64, dtype=np.float32).reshape(1, 64))
    starts = mx.array(np.array([8], dtype=np.int32))
    ends = mx.array(np.array([48], dtype=np.int32))
    sentinel = mx.array(np.array([[47, 46, 45, 44, 43, 42, 41, 40]], dtype=np.int32))

    def fail_path_c(*args: object, **kwargs: object) -> None:
        raise AssertionError("masked backend='auto' must not try TileLang Path C first")

    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_tilelang",
        fail_path_c,
    )
    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_metal",
        lambda *args, **kwargs: sentinel,
    )
    out = topk_selector(scores, k=8, starts=starts, ends=ends, backend="auto")
    assert np.asarray(out).tolist() == [[47, 46, 45, 44, 43, 42, 41, 40]]


def test_public_entry_point_auto_falls_back_to_path_b_after_receipted_path_c_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = mx.array(np.arange(64, dtype=np.float32).reshape(1, 64))
    sentinel = mx.array(np.array([[55, 54, 53, 52, 51, 50, 49, 48]], dtype=np.int32))

    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_tilelang",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_metal",
        lambda *args, **kwargs: sentinel,
    )
    out = topk_selector(scores, k=8, backend="auto")
    assert np.asarray(out).tolist() == [[55, 54, 53, 52, 51, 50, 49, 48]]


def test_topk_selector_bench_smoke_keeps_path_c_no_slower_than_path_b() -> None:
    from scripts.bench_tilelang_topk import _bench_shape, _row_strict_ok

    row = _bench_shape(
        batch=1,
        seq_len=512,
        k=32,
        dtype_name="float32",
        seed=20260504,
        warmup=2,
        iters=5,
    )

    assert _row_strict_ok(row, max_ratio=1.0), row
