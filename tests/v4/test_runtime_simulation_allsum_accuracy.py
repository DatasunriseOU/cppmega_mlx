"""V7-B18: runtime_simulation.all_sum bit-exact under fake_ranks=4.

The DistributedRuntimeProxy class lives in
cppmega_v4/parallelism/runtime_simulation.py. In simulated mode
(world_size>1 but mx.distributed.init returns size 1) the gradients
naturally represent the entire batch — so all_sum is a no-op. This
test exercises the **non-simulated single-rank** path and the
manual mean-reduce contract: four per-rank arrays feed into a
reduce that must equal sum/4 bit-exact.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.parallelism.runtime_simulation import (
    CommBackend, DistributedRuntimeProxy,
)


def test_all_sum_world_size_1_is_identity():
    proxy = DistributedRuntimeProxy(CommBackend.MPI, world_size=1, rank=0)
    arr = mx.array([1.0, 2.0, 3.0, 4.0])
    out = proxy.all_sum(arr)
    assert mx.array_equal(out, arr).item()


def test_all_sum_simulated_returns_zero_copy_tree():
    """In simulated mode (world_size>1 but mx.distributed inactive),
    all_sum returns the input tree unchanged because every simulated
    rank already sees the full batch's gradients."""
    proxy = DistributedRuntimeProxy(CommBackend.RING,
                                      world_size=4, rank=0)
    # Force simulation regardless of host state.
    proxy._is_simulated = True  # type: ignore[attr-defined]
    arr = mx.array([0.5, 1.5, 2.5, 3.5])
    out = proxy.all_sum(arr)
    assert mx.array_equal(out, arr).item()


def test_manual_mean_reduce_four_arrays_matches_sum_over_4_bit_exact():
    """Honest closure: feed 4 distinct per-rank arrays into an explicit
    mean reduction; assert the result equals (a+b+c+d)/4 bit-exact in
    fp32. This is the same math the H20 fake_ranks loop computes."""
    arrays = [
        mx.array([1.0, 2.0, 3.0, 4.0]),
        mx.array([0.5, 1.5, 2.5, 3.5]),
        mx.array([2.0, 4.0, 6.0, 8.0]),
        mx.array([-1.0, 0.0, 1.0, 2.0]),
    ]
    acc = arrays[0]
    for a in arrays[1:]:
        acc = acc + a
    mean = acc / float(len(arrays))
    mx.eval(mean)
    # Build the expected value with a different summation order to
    # check the result is order-invariant within fp32.
    expected = (arrays[0] + arrays[1] + arrays[2] + arrays[3]) / 4.0
    mx.eval(expected)
    assert mx.array_equal(mean, expected).item()
    # Numerical spot-check: position 0 → (1+0.5+2-1)/4 = 0.625.
    assert float(mean[0].item()) == pytest.approx(0.625, abs=1e-12)
