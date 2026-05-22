"""V7-B04: pipeline-parallel microbatch proxy tests."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime.pp_proxy import (
    pipeline_forward, pipeline_schedule, split_microbatches,
)


def test_v7_b04_split_microbatches_even_chunks():
    x = mx.arange(0, 24, dtype=mx.float32).reshape(8, 3)
    parts = split_microbatches(x, 4)
    assert len(parts) == 4
    for p in parts:
        assert p.shape == (2, 3)


def test_v7_b04_pipeline_forward_equivalent_to_sequential():
    x = mx.random.normal(shape=(8, 4), key=mx.random.key(0))
    stages = [
        lambda t: t * 2.0,
        lambda t: t + 1.0,
        lambda t: t - 0.5,
    ]
    # Direct sequential.
    direct = x
    for s in stages:
        direct = s(direct)
    # Pipeline with microbatching.
    pp = pipeline_forward(x, stages, num_microbatches=4)
    assert mx.allclose(direct, pp, atol=1e-6)


def test_v7_b04_pipeline_forward_empty_stages_identity():
    x = mx.array([[1.0, 2.0], [3.0, 4.0]])
    out = pipeline_forward(x, [], num_microbatches=1)
    assert mx.allclose(x, out, atol=0.0)


def test_v7_b04_split_rejects_non_divisible():
    with pytest.raises(ValueError):
        split_microbatches(mx.zeros((5, 3)), 2)


def test_v7_b04_pipeline_schedule_yields_mb_stage_pairs():
    pairs = list(pipeline_schedule(num_microbatches=2, num_stages=3))
    assert len(pairs) == 6
    assert pairs[0] == (0, 0)
    assert pairs[-1] == (1, 2)
