"""V7-D25 (za1.2): topk_selector_metal parity vs sorted ground truth.

The cppmega CUDA `topk_selector` returns column indices of the k largest
scores per row. The audit asks for parity vs `torch.topk` at atol=0 on
(B=4, K=512, k=8) — since cppmega.mlx has no torch dependency on Apple
Silicon, we build the canonical ground-truth via `mx.argsort` (descending)
which is bit-exact equivalent. Test asserts the SET of returned indices
matches the top-k-by-score set (order-independent, per the docstring of
topk_selector_reference).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_mlx.kernels.topk_selector_metal import (
    topk_selector_metal, topk_selector_reference, topk,
)


def _ground_truth_topk_indices(scores: mx.array, k: int) -> mx.array:
    """Bit-exact ground truth: sort descending, take first k indices."""
    order = mx.argsort(-scores, axis=-1)
    return order[..., :k].astype(mx.int32)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_topk_selector_returns_topk_set_b4_k512_k8(seed: int):
    """B=4, K=512, k=8: returned set == ground-truth top-k set."""
    scores = mx.random.normal(shape=(4, 512), key=mx.random.key(seed))
    out = topk(scores, 8)
    assert out.shape == (4, 8)
    assert out.dtype == mx.int32

    truth = _ground_truth_topk_indices(scores, 8)
    for b in range(4):
        # Set membership comparison (argpartition order undefined).
        out_set = {int(x) for x in out[b].tolist()}
        truth_set = {int(x) for x in truth[b].tolist()}
        assert out_set == truth_set, (
            f"row {b}: out={sorted(out_set)} != truth={sorted(truth_set)}")


def test_topk_selector_scores_at_indices_match_ground_truth():
    """Strong parity: the SCORES at the returned indices equal the
    top-k sorted scores at atol=0."""
    scores = mx.random.normal(shape=(4, 512), key=mx.random.key(1))
    out = topk(scores, 8)
    # Gather scores at returned positions; sort descending; compare to
    # ground-truth descending top-k scores at atol=0.
    for b in range(4):
        gathered = mx.take(scores[b], out[b]).astype(mx.float32)
        gathered_sorted = mx.sort(-gathered).tolist()  # ascending of negatives = descending
        truth_top_scores = mx.sort(-scores[b])[:8].tolist()
        assert gathered_sorted == truth_top_scores, (
            f"row {b}: gathered={gathered_sorted} != truth={truth_top_scores}")


def test_topk_selector_metal_path_returns_none_post_retirement():
    """topk_selector_metal is the retired Path-B compatibility surface;
    after retirement it returns None for any valid input shape."""
    scores = mx.random.normal(shape=(4, 512), key=mx.random.key(0))
    out = topk_selector_metal(scores, 8)
    assert out is None


def test_topk_reference_and_topk_alias_agree():
    """`topk` alias and `topk_selector_reference` produce the same set."""
    scores = mx.random.normal(shape=(2, 64), key=mx.random.key(99))
    a = topk(scores, 4)
    b = topk_selector_reference(scores, 4)
    for r in range(2):
        assert {int(x) for x in a[r].tolist()} == {
            int(x) for x in b[r].tolist()}
