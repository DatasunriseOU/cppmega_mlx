"""Single-rank and simulated-multi-rank tests for the ZeRO-1 wrapper.

The 48 GB Stream F peer is not connected yet, so multi-rank coverage is by
simulation: we instantiate the inner optimizer twice, manually feed each
"rank" its owned shard, and exchange tensors in-process to mimic
``mx.distributed.all_sum`` / ``all_gather``. These tests exercise the
sharding math; a real 2-node receipt is gated on the peer hardware and is
documented in :mod:`docs/distributed_zero1_smoke_procedure.md`.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_map, tree_merge, tree_unflatten

from cppmega_mlx.training.distributed_optimizer import (
    DistributedZeRO1Optimizer,
    _flatten_param_tree,
    _select_owned_subtree,
    _shard_assignment,
    _state_numel_bytes,
    make_distributed_optimizer,
)
from cppmega_mlx.training.optimizers import (
    AdamWFP32Moments,
    LionFP32Moments,
    make_adamw,
    make_lion,
    make_muon,
)


# ---------------------------------------------------------------------------
# Tiny test model + helpers
# ---------------------------------------------------------------------------


class _TinyMLP(nn.Module):
    """6-parameter-leaf MLP with bias terms — used for round-trip + numerics.

    Has an ``in_dim``->``hidden`` -> ``hidden`` -> ``out_dim`` shape so each
    apply-gradients touches every leaf and gives non-trivial loss.
    """

    def __init__(self, in_dim: int = 6, hidden: int = 8, out_dim: int = 4) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(self.l1(x))
        h = nn.relu(self.l2(h))
        return self.l3(h)


class _BalancedStack(nn.Module):
    """4 identical-shape Linear layers without bias.

    Every leaf has the same byte count so the round-robin shard at W=2 puts
    exactly half the optimizer-state bytes on each rank — used by the memory
    test where leaf-by-leaf imbalance would dominate the budget assertion.
    """

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.layer_0 = nn.Linear(dim, dim, bias=False)
        self.layer_1 = nn.Linear(dim, dim, bias=False)
        self.layer_2 = nn.Linear(dim, dim, bias=False)
        self.layer_3 = nn.Linear(dim, dim, bias=False)


def _seed_model(seed: int = 0) -> _TinyMLP:
    mx.random.seed(seed)
    model = _TinyMLP()
    mx.eval(model.parameters())
    return model


def _seed_balanced(seed: int = 0) -> _BalancedStack:
    mx.random.seed(seed)
    model = _BalancedStack()
    mx.eval(model.parameters())
    return model


def _grad_fn(model: _TinyMLP, x: mx.array, y: mx.array) -> Any:
    def loss_fn(m: _TinyMLP) -> mx.array:
        pred = m(x)
        return ((pred - y) ** 2).mean()

    grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = grad_fn(model)
    return loss, grads


def _clone_params(params: Any) -> Any:
    return tree_map(lambda a: mx.array(a) if isinstance(a, mx.array) else a, params)


def _params_equal(a: Any, b: Any, atol: float = 0.0) -> bool:
    flat_a = dict(tree_flatten(a))
    flat_b = dict(tree_flatten(b))
    if set(flat_a.keys()) != set(flat_b.keys()):
        return False
    for key in flat_a:
        diff = mx.max(mx.abs(flat_a[key] - flat_b[key])).item()
        if diff > atol:
            return False
    return True


# ---------------------------------------------------------------------------
# 1. world_size=1 short-circuit: behaves bit-for-bit like the inner optimizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_inner",
    [
        lambda: make_lion(learning_rate=1e-3),
        lambda: make_adamw(learning_rate=1e-3),
    ],
)
def test_zero1_world_size_1_identical_to_inner(make_inner) -> None:
    """At W=1 the wrapper must produce bit-identical params to the bare inner."""

    mx.random.seed(7)
    x = mx.random.normal((4, 6))
    y = mx.random.normal((4, 4))

    # Reference: bare inner optimizer.
    ref_model = _seed_model(seed=11)
    ref_opt = make_inner()
    ref_opt.init(ref_model.trainable_parameters())
    for _ in range(5):
        _, grads = _grad_fn(ref_model, x, y)
        ref_model.update(ref_opt.apply_gradients(grads, ref_model))
    mx.eval(ref_model.parameters())

    # Wrapped: ZeRO-1 with W=1, rank=0.
    test_model = _seed_model(seed=11)
    wrapped = make_distributed_optimizer(make_inner(), world_size=1, rank=0)
    wrapped.init(test_model.trainable_parameters())
    for _ in range(5):
        _, grads = _grad_fn(test_model, x, y)
        test_model.update(wrapped.apply_gradients(grads, test_model))
    mx.eval(test_model.parameters())

    assert wrapped.is_sharded is False
    assert _params_equal(
        ref_model.trainable_parameters(),
        test_model.trainable_parameters(),
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# 2. world_size=2: shard assignment is balanced and disjoint
# ---------------------------------------------------------------------------


def test_zero1_world_size_2_state_shards_correctly() -> None:
    """Owned-leaf sets at W=2 are disjoint, cover every leaf, and the rank-R
    optimizer state holds only that rank's leaves."""

    model = _seed_model(seed=3)
    params = model.trainable_parameters()
    flat = _flatten_param_tree(params)
    leaf_names = [name for name, _ in flat]

    # Every leaf assigned to exactly one rank (round-robin).
    assignment = _shard_assignment(len(flat), world_size=2)
    rank0_leaves = {name for name, owner in zip(leaf_names, assignment) if owner == 0}
    rank1_leaves = {name for name, owner in zip(leaf_names, assignment) if owner == 1}
    assert rank0_leaves.isdisjoint(rank1_leaves)
    assert rank0_leaves | rank1_leaves == set(leaf_names)

    # Wrapper at rank 0 holds optimizer state only for rank0_leaves.
    opt0 = make_distributed_optimizer(make_lion(learning_rate=1e-3), world_size=2, rank=0)
    opt0.init(params)
    state0_leaves = {
        name for name, leaf in tree_flatten(opt0.state) if isinstance(leaf, mx.array)
    }
    # Lion stores moment in keys like "<param>.m". Strip suffix to compare to
    # parameter names.
    state0_param_names = {name.removesuffix(".m") for name in state0_leaves if name.endswith(".m")}
    assert state0_param_names == rank0_leaves

    opt1 = make_distributed_optimizer(make_lion(learning_rate=1e-3), world_size=2, rank=1)
    opt1.init(params)
    state1_leaves = {
        name for name, leaf in tree_flatten(opt1.state) if isinstance(leaf, mx.array)
    }
    state1_param_names = {name.removesuffix(".m") for name in state1_leaves if name.endswith(".m")}
    assert state1_param_names == rank1_leaves


# ---------------------------------------------------------------------------
# 3. simulation: W=2 ZeRO-1 vs single non-sharded run, loss within tolerance
# ---------------------------------------------------------------------------


def _simulated_zero1_step(
    rank0_inner: Any,
    rank1_inner: Any,
    params: Any,
    grads: Any,
    *,
    world_size: int = 2,
) -> Any:
    """In-process simulation of one ZeRO-1 distributed step at W=2.

    Mirrors what :meth:`DistributedZeRO1Optimizer.apply_gradients` does on
    real hardware:

    1. all_reduce (mean) the gradients (in single-process simulation both
       ranks share the same local-autograd grads, so the explicit reduce is
       a no-op).
    2. Slice grads + params per rank, run the inner optimizer on the slice.
    3. Gather the disjoint per-rank updates back into a full param tree.
    """

    # Step 2: each rank steps inner on its owned shard.
    rank0_grads = _select_owned_subtree(grads, rank=0, world_size=world_size)
    rank0_params = _select_owned_subtree(params, rank=0, world_size=world_size)
    rank1_grads = _select_owned_subtree(grads, rank=1, world_size=world_size)
    rank1_params = _select_owned_subtree(params, rank=1, world_size=world_size)

    rank0_updates = (
        rank0_inner.apply_gradients(rank0_grads, rank0_params) if rank0_grads else {}
    )
    rank1_updates = (
        rank1_inner.apply_gradients(rank1_grads, rank1_params) if rank1_grads else {}
    )

    # Step 3: gather. Round-robin shard guarantees the two rank trees have
    # disjoint leaves, so flatten + concatenate + unflatten reconstitutes the
    # full updated tree without leaf-collision issues that ``tree_merge``
    # raises when the same path holds an array on both sides.
    merged_pairs = [
        (name, leaf)
        for name, leaf in (*tree_flatten(rank0_updates), *tree_flatten(rank1_updates))
        if isinstance(leaf, mx.array)
    ]
    return tree_unflatten(merged_pairs)


def test_zero1_simulation_w2_loss_matches_non_sharded_within_tolerance() -> None:
    """20 steps of simulated W=2 ZeRO-1 Lion vs non-sharded Lion: final loss
    should agree within 1% relative error.

    Test name explicitly says ``simulation`` per the constraints: real
    ``mx.distributed.all_sum`` / ``all_gather`` semantics are mirrored
    in-process by tree_merge, not invoked through the runtime collectives.
    """

    mx.random.seed(13)
    x = mx.random.normal((8, 6))
    y = mx.random.normal((8, 4))
    steps = 20
    lr = 5e-4

    # Reference run: single non-sharded Lion.
    ref_model = _seed_model(seed=21)
    ref_opt = make_lion(learning_rate=lr)
    ref_opt.init(ref_model.trainable_parameters())
    ref_loss = None
    for _ in range(steps):
        ref_loss, grads = _grad_fn(ref_model, x, y)
        ref_model.update(ref_opt.apply_gradients(grads, ref_model))
    mx.eval(ref_model.parameters())

    # Simulated W=2 ZeRO-1 run.
    sim_model = _seed_model(seed=21)
    sim_inner_r0 = make_lion(learning_rate=lr)
    sim_inner_r1 = make_lion(learning_rate=lr)
    sim_inner_r0.init(_select_owned_subtree(sim_model.trainable_parameters(), 0, 2))
    sim_inner_r1.init(_select_owned_subtree(sim_model.trainable_parameters(), 1, 2))
    sim_loss = None
    for _ in range(steps):
        sim_loss, grads = _grad_fn(sim_model, x, y)
        new_params = _simulated_zero1_step(
            sim_inner_r0,
            sim_inner_r1,
            sim_model.trainable_parameters(),
            grads,
            world_size=2,
        )
        sim_model.update(new_params)
    mx.eval(sim_model.parameters())

    ref_value = float(ref_loss.item())
    sim_value = float(sim_loss.item())
    # ZeRO-1 is exact: same data, same grads, the only difference is which
    # rank owns which leaf's optimizer state. The numerical answer must agree
    # to floating-point precision (well within 1% tolerance).
    rel_error = abs(sim_value - ref_value) / max(abs(ref_value), 1e-9)
    assert rel_error < 1e-2, (
        f"sim={sim_value:.6f} ref={ref_value:.6f} rel_err={rel_error:.2e}"
    )


# ---------------------------------------------------------------------------
# 4. memory: W=2 holds ~half the optimizer-state bytes of W=1
# ---------------------------------------------------------------------------


def test_zero1_memory_smaller_than_full_replicated() -> None:
    """``tree_flatten(opt.state)`` byte total at W=2 should be ~50% of W=1.

    Uses :class:`_BalancedStack` (4 identical-shape Linear layers, no bias) so
    the round-robin shard puts exactly 2 leaves' worth of optimizer-state
    bytes on each rank. We assert the [45%, 55%] band; the slack absorbs
    constant-bytes overhead from scalar bookkeeping like the ``step``
    counter that lives on every rank.

    For a tiny model whose leaves are heterogeneous (e.g. mixing biases and
    weights), round-robin can produce >50% imbalance — this is by design of
    the simple shard policy. Real production models with hundreds of
    same-shape transformer-block leaves never hit that pathological case.
    """

    model = _seed_balanced(seed=5)
    params = model.trainable_parameters()

    full = make_distributed_optimizer(make_lion(learning_rate=1e-3), world_size=1, rank=0)
    full.init(params)
    full_bytes = full.shard_owned_state_bytes()

    rank0 = make_distributed_optimizer(make_lion(learning_rate=1e-3), world_size=2, rank=0)
    rank0.init(params)
    rank1 = make_distributed_optimizer(make_lion(learning_rate=1e-3), world_size=2, rank=1)
    rank1.init(params)

    rank0_bytes = rank0.shard_owned_state_bytes()
    rank1_bytes = rank1.shard_owned_state_bytes()

    # Each rank holds < full state bytes (strictly less, not equal).
    assert rank0_bytes < full_bytes
    assert rank1_bytes < full_bytes

    # Per-rank fraction is ~50% (band 45%-55% on the balanced stack).
    fraction_r0 = rank0_bytes / full_bytes
    fraction_r1 = rank1_bytes / full_bytes
    assert 0.45 <= fraction_r0 <= 0.55, (
        f"rank0 fraction {fraction_r0:.3f} out of band; "
        f"rank0_bytes={rank0_bytes}, full_bytes={full_bytes}"
    )
    assert 0.45 <= fraction_r1 <= 0.55, (
        f"rank1 fraction {fraction_r1:.3f} out of band; "
        f"rank1_bytes={rank1_bytes}, full_bytes={full_bytes}"
    )

    # The two ranks' shards together cover the full state bytes (allowing for
    # small constant overhead from shared scalars like step counters).
    combined = rank0_bytes + rank1_bytes
    assert full_bytes <= combined <= int(full_bytes * 1.10)


# ---------------------------------------------------------------------------
# 5. Misc surface tests
# ---------------------------------------------------------------------------


def test_zero1_rejects_unsupported_inner_optimizer() -> None:
    class NotSupported:
        pass

    with pytest.raises(TypeError, match="inner_optimizer must be one of"):
        DistributedZeRO1Optimizer(NotSupported(), world_size=1, rank=0)


def test_zero1_rejects_invalid_rank_world_size() -> None:
    inner = make_lion(learning_rate=1e-3)
    with pytest.raises(ValueError, match="world_size must be"):
        DistributedZeRO1Optimizer(inner, world_size=0, rank=0)
    with pytest.raises(ValueError, match="rank must be"):
        DistributedZeRO1Optimizer(make_lion(learning_rate=1e-3), world_size=2, rank=2)
    with pytest.raises(ValueError, match="rank must be"):
        DistributedZeRO1Optimizer(make_lion(learning_rate=1e-3), world_size=2, rank=-1)


def test_zero1_supports_muon_adamw_multi() -> None:
    """The Muon+AdamW composite is one of the three supported inner optimizers."""

    inner = make_muon(lr_muon=1e-3, lr_adamw=1e-3)
    opt = DistributedZeRO1Optimizer(inner, world_size=1, rank=0)
    model = _seed_model(seed=42)
    opt.init(model.trainable_parameters())
    assert opt.owned_param_names  # non-empty
    assert opt.is_sharded is False


def test_zero1_learning_rate_setter_proxies_to_inner() -> None:
    inner = make_lion(learning_rate=1e-3)
    opt = DistributedZeRO1Optimizer(inner, world_size=1, rank=0)
    opt.learning_rate = 5e-4
    assert float(opt.learning_rate.item()) == pytest.approx(5e-4)
    assert float(inner.learning_rate.item()) == pytest.approx(5e-4)
