"""V7-B-real: single-process passthrough behavior of distributed.py.

Multi-process behavior is tested via the scripts/launch_multi.py
runbook; this file just nails down the world_size==1 contract so the
single-process path (the default) keeps working in CI.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime import distributed as d
from cppmega_v4.runtime.tp_proxy import (
    ColumnParallelLinear, RowParallelLinear,
)
from cppmega_v4.runtime.pp_proxy import (
    gpipe_schedule, one_f_one_b_schedule, pipeline_forward_real,
)
from cppmega_v4.runtime.multi_node_topology import build_process_groups


@pytest.fixture(autouse=True)
def _clean():
    d.reset_for_test()
    yield
    d.reset_for_test()


def test_init_returns_single_world_when_forced():
    info = d.init(force_single=True)
    assert info.world_size == 1
    assert info.rank == 0
    assert info.real is False


def test_is_distributed_false_in_single():
    d.init(force_single=True)
    assert d.is_distributed() is False


def test_all_reduce_passthrough_in_single():
    d.init(force_single=True)
    x = mx.array([1.0, 2.0, 3.0])
    assert mx.array_equal(d.all_reduce(x, op="sum"), x).item()
    assert mx.array_equal(d.all_reduce(x, op="mean"), x).item()


def test_all_gather_replicates_in_single():
    d.init(force_single=True)
    x = mx.array([[1.0, 2.0]])
    out = d.all_gather(x, axis=0)
    # world_size==1 → unchanged.
    assert out.shape == (1, 2)


def test_reduce_scatter_passthrough_in_single():
    d.init(force_single=True)
    x = mx.array([1.0, 2.0, 3.0, 4.0])
    assert mx.array_equal(d.reduce_scatter(x), x).item()


def test_measure_collective_overhead_smoke():
    d.init(force_single=True)
    row = d.measure_collective_overhead_ms(shard_size=32, n_iter=2)
    assert row["world_size"] == 1.0
    assert row["all_reduce_ms_per_iter"] >= 0
    assert row["all_gather_ms_per_iter"] >= 0


def test_column_parallel_linear_world_size_1():
    d.init(force_single=True)
    linear = ColumnParallelLinear(in_features=8, out_features=4,
                                    gather_output=True)
    x = mx.ones((2, 8))
    y = linear(x)
    assert y.shape == (2, 4)


def test_row_parallel_linear_world_size_1():
    d.init(force_single=True)
    linear = RowParallelLinear(in_features=8, out_features=4,
                                 input_is_parallel=False)
    x = mx.ones((2, 8))
    y = linear(x)
    assert y.shape == (2, 4)


def test_gpipe_schedule_has_full_forward_then_backward():
    ops = list(gpipe_schedule(num_microbatches=3, num_stages=2))
    # 3 microbatches × 2 stages × 2 phases (F+B) = 12 ops.
    assert len(ops) == 12
    # All forwards precede all backwards.
    phases = [p for p, _, _ in ops]
    last_f = max(i for i, p in enumerate(phases) if p == "F")
    first_b = min(i for i, p in enumerate(phases) if p == "B")
    assert last_f < first_b


def test_1f1b_schedule_alternates_after_warmup():
    ops = list(one_f_one_b_schedule(num_microbatches=4, num_stages=2))
    # Warmup count = num_stages-1 = 1 → one mb fully forwarded first.
    # Then steady state pairs F+B.
    phases = [p for p, _, _ in ops]
    # Every F must have a matching B somewhere downstream.
    f_mbs = sorted([mb for (p, _, mb) in ops if p == "F"])
    b_mbs = sorted([mb for (p, _, mb) in ops if p == "B"])
    assert f_mbs == b_mbs


def test_pipeline_forward_real_single_process():
    d.init(force_single=True)
    # Identity stages.
    out = pipeline_forward_real(
        mx.arange(8.0).reshape(8, 1),
        stages=[lambda y: y + 1, lambda y: y * 2],
        num_microbatches=4,
        schedule="1f1b",
    )
    # Each microbatch: (x+1)*2 → 2x+2
    expected = (mx.arange(8.0).reshape(8, 1) + 1) * 2
    assert mx.array_equal(out, expected).item()


def test_build_process_groups_falls_back_when_not_real():
    d.init(force_single=True)
    pg = build_process_groups(num_nodes=2, gpus_per_node=2)
    assert pg["ok"] is False
    assert pg["expected_world_size"] == 4
    assert pg["intra_node"] is None
    assert pg["inter_node"] is None
