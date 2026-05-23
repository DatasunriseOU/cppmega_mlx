"""V7-B13: pp_proxy pipeline forward parity vs sequential forward."""

from __future__ import annotations

import mlx.core as mx

from cppmega_v4.runtime.pp_proxy import (
    pipeline_forward, pipeline_forward_real, split_microbatches,
)


def test_split_microbatches_preserves_concat_identity():
    mx.random.seed(0)
    x = mx.random.normal(shape=(8, 4), key=mx.random.key(0))
    for n in (1, 2, 4, 8):
        mbs = split_microbatches(x, n)
        recat = mx.concatenate(mbs, axis=0)
        assert mx.array_equal(recat, x).item(), f"split n={n} lost data"


def test_pipeline_forward_matches_sequential_on_simple_stages():
    """pipeline_forward(x, stages, num_microbatches=k) should match a
    bare sequential composition for any k that divides the batch."""
    mx.random.seed(7)
    x = mx.random.normal(shape=(8, 4), key=mx.random.key(1))

    def stage_a(y: mx.array) -> mx.array:
        return y + 1.0

    def stage_b(y: mx.array) -> mx.array:
        return y * 2.0

    def stage_c(y: mx.array) -> mx.array:
        return mx.exp(-y / 8.0)

    stages = [stage_a, stage_b, stage_c]
    seq = x
    for s in stages:
        seq = s(seq)

    for n in (1, 2, 4, 8):
        out = pipeline_forward(x, stages, num_microbatches=n)
        assert mx.allclose(out, seq, atol=1e-6).item(), (
            f"pp_proxy num_microbatches={n} drifted from sequential")


def test_pipeline_forward_real_single_process_matches_pipeline_forward():
    """pipeline_forward_real with 1f1b schedule reaches the same output
    as the legacy pipeline_forward when world_size==1."""
    x = mx.arange(16.0).reshape(8, 2)
    stages = [lambda y: y * 2, lambda y: y + 1]
    a = pipeline_forward(x, stages, num_microbatches=4)
    b = pipeline_forward_real(x, stages, num_microbatches=4,
                                schedule="1f1b")
    assert mx.allclose(a, b, atol=1e-6).item()
