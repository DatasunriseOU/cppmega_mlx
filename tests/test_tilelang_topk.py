"""Tests for the retired Path B and active Path C topk_selector ports.

The direct-MSL Path B Metal kernel is retired. Path C uses a TileLang DSL
PrimFunc lowered to Metal, then launched through the tvm-ffi owner-output
route when callers provide ``out``. The older no-``out`` MLX fast-kernel
wrapper is also retired.

The tests verify:

1. The pure-MLX reference returns the correct top-k indices (set-equality
   to a NumPy oracle) for a sweep of (B, T, k) shapes.
2. The retired direct-MSL Path B surface fails closed.
3. The TileLang DSL Path C kernel produces the same set of indices as the
   reference (set-equality, since all partition contracts are order-unspecified).
4. Output shape and dtype match the cppmega source contract.
5. Edge cases (k=1, k=seq_len, and start/end masking) are exercised.
"""

from __future__ import annotations

import numpy as np  # type: ignore[reportMissingImports]
import pytest

import mlx.core as mx

import cppmega_mlx.nn._tilelang.topk_selector as topk_selector_mod
from cppmega_mlx.nn._tilelang.topk_selector import (  # noqa: E402
    PathBStatus,
    PathCStatus,
    TopKPathCDirectError,
    _path_c_kernel_for,
    _path_c_threads_for,
    topk_selector,
    topk_selector_metal,
    topk_selector_path_b_status,
    topk_selector_path_c_status,
    topk_selector_reference,
    topk_selector_tilelang,
    topk_selector_tilelang_direct,
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


def _topk_tilelang_direct_output(
    scores: mx.array,
    k: int,
    *,
    starts: mx.array | None = None,
    ends: mx.array | None = None,
) -> mx.array:
    """Allocate an explicit test owner buffer and run the direct Path C route."""

    out = mx.full((int(scores.shape[0]), int(k)), -123, dtype=mx.int32)
    return topk_selector_tilelang_direct(
        scores,
        k,
        starts=starts,
        ends=ends,
        out=out,
    )


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


def test_path_b_status_reports_retired() -> None:
    status = topk_selector_path_b_status()
    assert isinstance(status, PathBStatus)
    assert status.available is False
    assert "retired" in status.reason
    assert "owner-output" in status.reason


def test_path_b_status_reason_is_stable() -> None:
    s1 = topk_selector_path_b_status()
    s2 = topk_selector_path_b_status()
    assert s1.reason == s2.reason


def test_path_b_status_documents_reference_fallback() -> None:
    reason = topk_selector_mod._PATH_B_RETIRED_REASON
    assert "direct-MSL Path B is retired" in reason
    assert "TileLang/tvm-ffi owner-output route" in reason
    assert "pure-MLX reference" in reason


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


def test_path_c_debug_lowering_uses_dispatch_returned_msl_without_fast_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega_mlx.nn._tilelang._msl_transform import TileLangMSLLowering

    sentinel = TileLangMSLLowering(
        header="// raw tilelang header\n",
        body="// raw tilelang body\n",
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        msl_text="kernel void topk() {}",
        buffer_param_names=[],
        kernel_name="topk",
    )
    calls: dict[str, object] = {}

    def fake_dispatch_lower(*args: object, **kwargs: object) -> TileLangMSLLowering:
        calls["return_msl"] = kwargs.get("return_msl")
        calls["pass_configs"] = kwargs.get("pass_configs")
        return sentinel

    monkeypatch.setattr(topk_selector_mod, "dispatch_lower", fake_dispatch_lower)
    monkeypatch.setattr(
        topk_selector_mod,
        "_topk_path_c_pass_configs",
        lambda: {"tl.z3_proof.barrier_minimization": True},
    )

    topk_selector_mod._path_c_kernel_for.cache_clear()
    try:
        kernel, lowering = topk_selector_mod._path_c_kernel_for(1, 8, 1, 1, "float32")
    finally:
        topk_selector_mod._path_c_kernel_for.cache_clear()

    assert kernel is None
    assert lowering is sentinel
    assert calls["return_msl"] is True
    assert calls["pass_configs"] == {"tl.z3_proof.barrier_minimization": True}


def test_path_c_lowering_avoids_break_in_insertion_scan() -> None:
    """Insertion scan must not emit a C/MSL ``break``.

    The direct tvm-ffi compile path can lower the TileLang insertion-loop
    ``break`` as a break from the outer row scan after static simplification,
    which skips later score candidates. The guarded form preserves the
    early-stop semantics without emitting a control-flow break.
    """

    threads = _path_c_threads_for(32)
    _, lowering = _path_c_kernel_for(1, 512, 32, threads, "float32")
    assert "break;" not in lowering.body
    assert "keep_scanning" in lowering.body


def test_path_c_direct_uses_owner_output_without_mlx_fast_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class RecordingKernel:
        def __call__(self, *args: object) -> object:
            calls.append(args)
            return args[1]

    def fail_legacy_kernel(*_: object, **__: object) -> object:
        raise AssertionError("direct owner-output topk must not build mx.fast fallback")

    monkeypatch.setattr(
        topk_selector_mod,
        "topk_selector_path_c_status",
        lambda: PathCStatus(True, "available"),
    )
    monkeypatch.setattr(
        topk_selector_mod,
        "_path_c_tvm_ffi_kernel_for",
        lambda *_, **__: RecordingKernel(),
    )
    monkeypatch.setattr(topk_selector_mod, "_path_c_kernel_for", fail_legacy_kernel)
    monkeypatch.setattr(mx, "synchronize", lambda: None)

    scores = mx.array(np.arange(8, dtype=np.float32).reshape(1, 8))
    out = mx.full((1, 2), -123, dtype=mx.int32)
    returned = topk_selector_tilelang_direct(scores, 2, out=out)

    assert returned is out
    assert len(calls) == 1
    assert calls[0][1] is out
    assert calls[0][2] is scores


def test_path_c_direct_propagates_typed_dlpack_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tilelang.contrib.mlx_interop import DLPackConversionError

    class FailingKernel:
        def __call__(self, *_: object) -> object:
            raise DLPackConversionError("MLX array import failed: wrong device")

    monkeypatch.setattr(
        topk_selector_mod,
        "topk_selector_path_c_status",
        lambda: PathCStatus(True, "available"),
    )
    monkeypatch.setattr(
        topk_selector_mod,
        "_path_c_tvm_ffi_kernel_for",
        lambda *_, **__: FailingKernel(),
    )

    scores = mx.array(np.arange(8, dtype=np.float32).reshape(1, 8))
    out = mx.full((1, 2), -123, dtype=mx.int32)
    with pytest.raises(DLPackConversionError, match="wrong device"):
        topk_selector_tilelang_direct(scores, 2, out=out)


def test_path_c_direct_rejects_bad_owner_output_abi() -> None:
    scores = mx.array(np.arange(8, dtype=np.float32).reshape(1, 8))

    with pytest.raises(ValueError, match="out shape"):
        topk_selector_tilelang_direct(
            scores,
            2,
            out=mx.zeros((1, 3), dtype=mx.int32),
        )
    with pytest.raises(ValueError, match="out dtype"):
        topk_selector_tilelang_direct(
            scores,
            2,
            out=mx.zeros((1, 2), dtype=mx.float32),
        )


def test_path_c_direct_rejects_hidden_bfloat16_score_cast() -> None:
    scores = mx.array(np.arange(8, dtype=np.float32).reshape(1, 8)).astype(
        mx.bfloat16
    )
    out = mx.zeros((1, 2), dtype=mx.int32)

    with pytest.raises(TopKPathCDirectError, match="without hidden casts"):
        topk_selector_tilelang_direct(scores, 2, out=out)


def test_path_c_no_out_public_route_fails_closed_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        topk_selector_mod,
        "topk_selector_path_c_status",
        lambda: PathCStatus(True, "available"),
    )
    monkeypatch.setattr(
        topk_selector_mod,
        "_path_c_kernel_for",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("no-out Path C wrapper must not be built by default")
        ),
    )

    scores = mx.array(np.arange(8, dtype=np.float32).reshape(1, 8))
    assert topk_selector_tilelang(scores, 2) is None


# ---------------------------------------------------------------------------
# Retired direct-MSL Path B surface.
# ---------------------------------------------------------------------------


def test_path_b_metal_surface_fails_closed() -> None:
    scores = mx.array(np.arange(16, dtype=np.float32).reshape(1, 16))
    assert topk_selector_metal(scores, k=4) is None


def test_path_b_metal_surface_still_validates_shape_and_k() -> None:
    with pytest.raises(ValueError):
        topk_selector_metal(mx.zeros((3,)), k=1)
    with pytest.raises(ValueError):
        topk_selector_metal(mx.zeros((2, 8)), k=0)
    with pytest.raises(ValueError):
        topk_selector_metal(mx.zeros((2, 8)), k=9)


def test_public_entry_point_metal_backend_fails_closed() -> None:
    rng = np.random.default_rng(11)
    scores = mx.array(rng.standard_normal((2, 64)).astype(np.float32))
    with pytest.raises(RuntimeError, match="direct-MSL Path B is retired"):
        topk_selector(scores, k=8, backend="metal")


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
    out_c = _topk_tilelang_direct_output(scores, k)
    assert out_c is not None, topk_selector_path_c_status().reason
    out_ref = topk_selector_reference(scores, k)
    assert _to_index_sets(out_c) == _to_index_sets(out_ref)
    assert out_c.dtype == mx.int32
    assert tuple(out_c.shape) == (batch, k)


def test_path_c_direct_tvm_ffi_reuses_owner_output_and_mutates_buffer() -> None:
    scores_np = np.array(
        [
            [0.0, 5.0, 1.0, 7.0, 2.0, 3.0, 4.0, 6.0],
            [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0],
        ],
        dtype=np.float32,
    )
    scores = mx.array(scores_np)
    out = mx.full((2, 2), -123, dtype=mx.int32)
    mx.eval(scores, out)

    returned = topk_selector_tilelang_direct(scores, 2, out=out)
    out_ref = topk_selector_reference(scores, 2)
    mx.eval(out_ref)

    assert returned is out
    assert np.asarray(out).tolist() != [[-123, -123], [-123, -123]]
    assert _to_index_sets(out) == _to_index_sets(out_ref)


def test_path_c_direct_rejects_starts_ends_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    out = mx.full((batch, k), -123, dtype=mx.int32)

    monkeypatch.setattr(
        topk_selector_mod,
        "_path_c_tvm_ffi_kernel_for",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("masked direct Path C must fail before compile")
        ),
    )

    with pytest.raises(TopKPathCDirectError, match="limited to unmasked rows"):
        topk_selector_tilelang_direct(scores, k, starts=starts, ends=ends, out=out)
    mx.eval(out)
    assert np.asarray(out).tolist() == [[-123] * k] * batch


@pytest.mark.parametrize(
    "starts_np,ends_np",
    [
        (np.array([0, 3], dtype=np.int32), None),
        (None, np.array([-1, 99], dtype=np.int32)),
    ],
)
def test_path_c_direct_rejects_partial_interval_mask(
    starts_np: np.ndarray | None,
    ends_np: np.ndarray | None,
) -> None:
    scores = mx.array(np.array([
        [5.0, 4.0, 3.0, 2.0],
        [0.0, 1.0, 2.0, 3.0],
    ], dtype=np.float32))
    starts = None if starts_np is None else mx.array(starts_np)
    ends = None if ends_np is None else mx.array(ends_np)
    out = mx.full((2, 2), -123, dtype=mx.int32)

    with pytest.raises(TopKPathCDirectError, match="limited to unmasked rows"):
        topk_selector_tilelang_direct(scores, k=2, starts=starts, ends=ends, out=out)
    mx.eval(out)
    assert np.asarray(out).tolist() == [[-123, -123], [-123, -123]]


def test_path_c_keeps_expected_exact_order_for_ties() -> None:
    scores = mx.array(np.array([
        [1.0, 3.0, 3.0, 2.0, 3.0, 0.0, -1.0, 3.0],
        [5.0, 5.0, 4.0, 4.0, 5.0, 5.0, 3.0, 2.0],
    ], dtype=np.float32))

    out_c = _topk_tilelang_direct_output(scores, k=6)
    assert out_c is not None, topk_selector_path_c_status().reason
    mx.eval(out_c)

    expected = [
        [1, 2, 4, 7, 3, 0],
        [0, 1, 4, 5, 2, 3],
    ]
    assert np.asarray(out_c).tolist() == expected


def test_path_c_direct_rejects_masked_ties_and_sentinels() -> None:
    scores = mx.array(np.array([
        [9.0, 7.0, 7.0, 6.0, 7.0, 5.0, 7.0, 4.0],
        [1.0, 8.0, 8.0, 8.0, 3.0, 8.0, 2.0, 8.0],
        [0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 4.0],
    ], dtype=np.float32))
    starts = mx.array(np.array([1, 2, 6], dtype=np.int32))
    ends = mx.array(np.array([7, 6, 8], dtype=np.int32))
    out = mx.full((3, 4), -123, dtype=mx.int32)

    with pytest.raises(TopKPathCDirectError, match="limited to unmasked rows"):
        topk_selector_tilelang_direct(scores, k=4, starts=starts, ends=ends, out=out)
    out_ref = topk_selector_reference(scores, k=4, starts=starts, ends=ends)
    mx.eval(out, out_ref)

    expected = [
        [1, 2, 4, 6],
        [2, 3, 5, 4],
        [6, 7, -1, -1],
    ]
    assert np.asarray(out_ref).tolist() == expected
    assert np.asarray(out).tolist() == [[-123] * 4] * 3


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_path_c_acceptance_shape_512x64_dispatches(dtype: mx.Dtype) -> None:
    scores = _acceptance_scores(batch=4, seq_len=512, k=64, dtype=dtype, seed=1512)
    out_c = _topk_tilelang_direct_output(scores, k=64)
    out_ref = topk_selector_reference(scores, k=64)
    assert _to_index_sets(out_c) == _to_index_sets(out_ref)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_path_c_acceptance_shape_4096x256_dispatches(dtype: mx.Dtype) -> None:
    scores = _acceptance_scores(batch=1, seq_len=4096, k=256, dtype=dtype, seed=14096)
    out_c = _topk_tilelang_direct_output(scores, k=256)
    out_ref = topk_selector_reference(scores, k=256)
    assert _to_index_sets(out_c) == _to_index_sets(out_ref)


def test_public_entry_point_tilelang_backend_allocates_owner_output_for_unmasked() -> None:
    rng = np.random.default_rng(29)
    scores_np = rng.standard_normal((2, 64)).astype(np.float32)
    scores_np += np.arange(scores_np.size, dtype=np.float32).reshape(scores_np.shape) * 1e-5
    scores = mx.array(scores_np)
    out = topk_selector(scores, k=8, backend="tilelang")
    out_ref = topk_selector_reference(scores, k=8)
    assert _to_index_sets(out) == _to_index_sets(out_ref)


def test_public_entry_point_auto_uses_path_c_owner_output_for_unmasked_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = mx.array(np.arange(64, dtype=np.float32).reshape(1, 64))
    sentinel = mx.array(np.array([[63, 62, 61, 60, 59, 58, 57, 56]], dtype=np.int32))

    def path_c_direct(*args: object, **kwargs: object) -> mx.array:
        return sentinel

    def fail_path_b(*args: object, **kwargs: object) -> None:
        raise AssertionError("backend='auto' must not call retired Path B")

    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_tilelang_direct",
        path_c_direct,
    )
    monkeypatch.setattr("cppmega_mlx.nn._tilelang.topk_selector.topk_selector_metal", fail_path_b)
    out = topk_selector(scores, k=8, backend="auto")
    assert np.asarray(out).tolist() == [[63, 62, 61, 60, 59, 58, 57, 56]]


def test_public_entry_point_auto_falls_back_to_reference_for_bfloat16_without_owner_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = mx.array(np.arange(4 * 2048, dtype=np.float32).reshape(4, 2048)).astype(
        mx.bfloat16
    )

    def fail_path_c(*args: object, **kwargs: object) -> None:
        raise TopKPathCDirectError("bf16 unsupported")

    def fail_path_b(*args: object, **kwargs: object) -> None:
        raise AssertionError("backend='auto' must not call retired Path B")

    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_tilelang_direct",
        fail_path_c,
    )
    monkeypatch.setattr("cppmega_mlx.nn._tilelang.topk_selector.topk_selector_metal", fail_path_b)
    out = topk_selector(scores, k=64, backend="auto")
    out_ref = topk_selector_reference(scores, k=64)
    assert _to_index_sets(out) == _to_index_sets(out_ref)


def test_public_entry_point_auto_falls_back_to_reference_when_path_c_direct_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = mx.array(np.arange(16, dtype=np.float32).reshape(1, 16))

    def fail_path_c(*args: object, **kwargs: object) -> None:
        raise TopKPathCDirectError("compile failed")

    def fail_path_b(*args: object, **kwargs: object) -> None:
        raise AssertionError("backend='auto' must not call retired Path B")

    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_tilelang_direct",
        fail_path_c,
    )
    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_metal",
        fail_path_b,
    )
    out = topk_selector(scores, k=4, backend="auto")
    out_ref = topk_selector_reference(scores, k=4)
    assert _to_index_sets(out) == _to_index_sets(out_ref)


def test_public_entry_point_auto_uses_reference_for_masked_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = mx.array(np.arange(64, dtype=np.float32).reshape(1, 64))
    starts = mx.array(np.array([8], dtype=np.int32))
    ends = mx.array(np.array([48], dtype=np.int32))

    def fail_path_c(*args: object, **kwargs: object) -> None:
        raise AssertionError("masked backend='auto' must not call unmasked Path C direct")

    def fail_path_b(*args: object, **kwargs: object) -> None:
        raise AssertionError("backend='auto' must not call retired Path B")

    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_tilelang_direct",
        fail_path_c,
    )
    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_metal",
        fail_path_b,
    )
    out = topk_selector(scores, k=8, starts=starts, ends=ends, backend="auto")
    out_ref = topk_selector_reference(scores, k=8, starts=starts, ends=ends)
    assert _to_index_sets(out) == _to_index_sets(out_ref)


def test_public_entry_point_auto_does_not_call_retired_path_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = mx.array(np.arange(64, dtype=np.float32).reshape(1, 64))
    sentinel = mx.array(np.array([[55, 54, 53, 52, 51, 50, 49, 48]], dtype=np.int32))

    def path_c_direct(*args: object, **kwargs: object) -> mx.array:
        return sentinel

    def fail_path_b(*args: object, **kwargs: object) -> None:
        raise AssertionError("backend='auto' must not call retired Path B")

    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_tilelang_direct",
        path_c_direct,
    )
    monkeypatch.setattr(
        "cppmega_mlx.nn._tilelang.topk_selector.topk_selector_metal",
        fail_path_b,
    )
    out = topk_selector(scores, k=8, backend="auto")
    assert np.asarray(out).tolist() == [[55, 54, 53, 52, 51, 50, 49, 48]]


def test_explicit_path_c_direct_keeps_no_hidden_output_allocation_boundary() -> None:
    scores = mx.array(np.arange(64, dtype=np.float32).reshape(1, 64))
    out = mx.full((1, 8), -123, dtype=mx.int32)
    returned = topk_selector_tilelang(scores, 8, out=out)
    assert returned is out
