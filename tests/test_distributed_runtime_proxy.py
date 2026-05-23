"""Unit and integration tests for the unified DistributedRuntimeProxy and collective simulation layer."""

from __future__ import annotations

from typing import Any
import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from cppmega_v4.parallelism.sharding_spec import CommBackend, ShardingSpec, AxisAssignment, ParallelismKind
from cppmega_v4.parallelism.topology import DeviceTopology, DeviceSpec, DeviceKind
from cppmega_v4.parallelism.runtime_simulation import DistributedRuntimeProxy
from cppmega_v4.parallelism.gotcha_checker import check_gotchas
from cppmega_v4.buildspec.model_build_spec import ModelBuildSpec
from cppmega_v4.buildspec.loss_spec import LossSpec, LossKind
from cppmega_v4.buildspec.optim_spec import OptimSpec
from cppmega_v4.fusion.brick_graph import BrickGraph
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def test_proxy_init_and_simulation_mode() -> None:
    """Verify proxy initialization and correct simulated mode detection."""
    # Singe rank => not simulated
    proxy_single = DistributedRuntimeProxy(comm_backend="ring", world_size=1, rank=0)
    assert not proxy_single.is_simulated
    assert proxy_single.world_size == 1

    # Multi-rank on single process (no active mlx.distributed) => simulated
    proxy_sim = DistributedRuntimeProxy(comm_backend="ring", world_size=4, rank=1)
    assert proxy_sim.is_simulated
    assert proxy_sim.world_size == 4
    assert proxy_sim.rank == 1

    # Check CommBackend enum coercion
    assert proxy_sim.comm_backend == CommBackend.RING


def test_select_owned_and_round_robin_slicing() -> None:
    """Validate that round-robin slicing is disjoint and covers the entire tree."""
    model = nn.Sequential(
        nn.Linear(8, 8),
        nn.Linear(8, 8),
        nn.Linear(8, 8),
        nn.Linear(8, 8),
    )
    params = model.parameters()
    flat_params = tree_flatten(params)
    assert len(flat_params) == 8  # 4 weights + 4 biases

    world_size = 3
    proxies = [
        DistributedRuntimeProxy(comm_backend="ring", world_size=world_size, rank=r)
        for r in range(world_size)
    ]

    owned_subtrees = [proxies[r].select_owned(params) for r in range(world_size)]

    # Collect all flattened keys and verify disjointness
    seen_keys = set()
    total_owned_count = 0

    for r, subtree in enumerate(owned_subtrees):
        subtree_flat = tree_flatten(subtree)
        for name, leaf in subtree_flat:
            assert isinstance(leaf, mx.array)
            assert name not in seen_keys, f"Key {name} owned by rank {r} but already claimed!"
            seen_keys.add(name)
            total_owned_count += 1

    # Make sure we covered every leaf of the model parameters
    assert len(seen_keys) == len(flat_params)
    assert total_owned_count == len(flat_params)


def test_collective_simulations_zero_copy() -> None:
    """Ensure simulated collectives (all_sum, all_gather) maintain zero-copy references."""
    proxy = DistributedRuntimeProxy(comm_backend="ring", world_size=2, rank=0)

    # 1. all_sum (zero-copy reference)
    x = mx.array([1.0, 2.0, 3.0])
    y = proxy.all_sum(x)
    assert y is x  # Identical reference

    # 2. all_gather_simulated (reconstruction from disjoint virtual updates)
    model = nn.Sequential(
        nn.Linear(2, 2, bias=False),
        nn.Linear(2, 2, bias=False),
    )
    params = model.parameters()

    # Create updates for rank 0 and rank 1
    rank0_proxy = DistributedRuntimeProxy(comm_backend="ring", world_size=2, rank=0)
    rank1_proxy = DistributedRuntimeProxy(comm_backend="ring", world_size=2, rank=1)

    rank0_owned = rank0_proxy.select_owned(params)
    rank1_owned = rank1_proxy.select_owned(params)

    # Mock some updates
    flat_r0 = dict(tree_flatten(rank0_owned))
    flat_r1 = dict(tree_flatten(rank1_owned))

    # Perform updates by rank
    updated_r0 = {k: v + 1.0 for k, v in flat_r0.items()}
    updated_r1 = {k: v + 2.0 for k, v in flat_r1.items()}

    # Merge
    merged = rank0_proxy.all_gather_simulated([
        tree_unflatten(list(updated_r0.items())),
        tree_unflatten(list(updated_r1.items()))
    ])

    merged_flat = dict(tree_flatten(merged))
    for k in flat_r0:
        assert mx.allclose(merged_flat[k], flat_r0[k] + 1.0).item()
    for k in flat_r1:
        assert mx.allclose(merged_flat[k], flat_r1[k] + 2.0).item()


def test_gotcha_checker_rules() -> None:
    """Test gotchas related to communication backends and GPU NVLink interconnects."""
    # 1. incompatible_comm_backend (nccl on Apple Silicon)
    apple_topo = DeviceTopology(
        devices=(
            DeviceSpec(kind=DeviceKind.M3_ULTRA, hbm_bytes=100_000_000, interconnect="unified_memory", bandwidth_gbps=800.0),
        ),
        mesh_axes={"dp": 1},
    )
    sharding_nccl_on_apple = ShardingSpec(
        topology=apple_topo,
        axis_assignments=(
            AxisAssignment(axis_name="dp", kind=ParallelismKind.FSDP2, degree=1),
        ),
        comm_backend=CommBackend.NCCL,
    )
    from cppmega_v4.buildspec.optim_spec import adamw
    mock_build = ModelBuildSpec(
        graph=BrickGraph(nodes=()),
        dim_env={},
        optim=adamw(lr=1e-3),
        loss=LossSpec(kind=LossKind.CROSS_ENTROPY, head_outputs=("mlp",)),
    )
    fired = check_gotchas(sharding_nccl_on_apple, mock_build)
    assert any(g.gotcha_id == "incompatible_comm_backend" for g in fired)

    # 2. slow_loopback_ring (ring on GPU NVLink)
    nvlink_gpu_topo = DeviceTopology(
        devices=tuple(
            DeviceSpec(kind=DeviceKind.H100_80GB, hbm_bytes=80_000_000, interconnect="nvlink", bandwidth_gbps=900.0)
            for _ in range(4)
        ),
        mesh_axes={"dp": 4},
    )
    sharding_ring_on_gpu = ShardingSpec(
        topology=nvlink_gpu_topo,
        axis_assignments=(
            AxisAssignment(axis_name="dp", kind=ParallelismKind.FSDP2, degree=4),
        ),
        comm_backend=CommBackend.RING,
    )
    fired_gpu = check_gotchas(sharding_ring_on_gpu, mock_build)
    assert any(g.gotcha_id == "slow_loopback_ring" for g in fired_gpu)


def test_stage_train_integration_simulated_zero1() -> None:
    """Verify that stage_train completes successfully under simulated sharding."""
    d = {
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
        "sharding": {
            "topology": {"factory": "h100_8x", "kwargs": {}},
            "axis_assignments": [
                {"axis_name": "dp", "kind": "fsdp2", "degree": 2},
            ],
            "compile_mode": "regional",
            "fp8_enabled": False,
            "comm_backend": "ring",
        }
    }
    spec = VerifyParams.model_validate(d)

    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 3}},
    }))

    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"

    extras = tr.extras
    assert "comm_backend" in extras
    assert extras["comm_backend"] == "ring"
    assert "is_simulated" in extras
    assert extras["is_simulated"] is True
    assert "losses" in extras
    assert len(extras["losses"]) == 3
