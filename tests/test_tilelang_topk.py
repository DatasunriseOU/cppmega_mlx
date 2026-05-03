"""Tests for the Path B topk_selector port.

The Path B Metal kernel is currently blocked by tilelang 0.1.9's missing
``shared.dyn`` storage scope on the ``metal`` target (and a downstream
``LowerTileOp`` injective-layout failure when the histogram fill is sized
``RADIX+1``). See docs/tilelang_ports/topk_selector.md for the probe
transcript.

While that blocker is in place the tests verify:

1. The pure-MLX reference returns the correct top-k indices (set-equality
   to a NumPy oracle) for a sweep of (B, T, k) shapes.
2. Output shape and dtype match the cppmega source contract.
3. Edge cases (k=1, k=seq_len, and start/end masking) are exercised.
4. The Path B status helper reports the blocker reason and the public
   ``topk_selector`` entry point still produces the reference output.
5. The Metal kernel test surface is collected (skipped when tilelang is
   not importable or the codegen blocker is active).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang.topk_selector import (
    PathBStatus,
    topk_selector,
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


def test_path_b_status_records_blocker_reason() -> None:
    status = topk_selector_path_b_status()
    assert isinstance(status, PathBStatus)
    assert status.available is False
    # The reason must mention the codegen problem so future readers find it.
    assert "shared.dyn" in status.reason or "Loop layout" in status.reason


def test_path_b_status_reason_is_stable() -> None:
    s1 = topk_selector_path_b_status()
    s2 = topk_selector_path_b_status()
    assert s1.reason == s2.reason


# ---------------------------------------------------------------------------
# Path B kernel suite (skipped while shared.dyn / layout blocker is active).
# ---------------------------------------------------------------------------

def _tilelang_metal_topk_blocked() -> bool:
    try:
        importlib.import_module("tilelang")
    except Exception:
        return True
    return not topk_selector_path_b_status().available


@pytest.mark.skipif(
    _tilelang_metal_topk_blocked(),
    reason="Path B topk_selector kernel blocked by tilelang 0.1.9 metal shared.dyn gap",
)
def test_path_b_forward_parity() -> None:
    # Placeholder: the moment the blocker lifts, replace with a real parity
    # check between a Metal-backed topk_selector and the pure-MLX reference.
    pytest.skip("Metal-backed topk_selector is not yet implemented")
