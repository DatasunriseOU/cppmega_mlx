"""HybridTinyLM.make_path_c_physical_abi_bank_owner.

Verifies that the model exposes the model-owned physical ABI bank factory
the m04 Path C auto-install path requires. The check exercises the actual
production schedule (no monkeypatching) and asserts that the resulting
owner satisfies the structural contract enforced by
``cppmega_mlx.runtime.path_c_physical_abi.make_physical_abi_bank_owner``.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import pytest

from cppmega_mlx.recipes.model_factory import (
    build_local_gb10_quarter_tiny_smoke_model,
)
from cppmega_mlx.runtime.path_c_physical_abi import (
    PathCPhysicalAbiBankOwner,
)


@pytest.fixture(scope="module")
def model():
    return build_local_gb10_quarter_tiny_smoke_model()


def test_make_path_c_physical_abi_bank_owner_returns_validated_owner(model) -> None:
    owner = model.make_path_c_physical_abi_bank_owner(sequence_length=64)
    assert isinstance(owner, PathCPhysicalAbiBankOwner)
    assert owner.owner_name == "local_gb10_quarter.path_c_physical_abi_banks"
    assert owner.binding_payload["status"] == "ok"
    assert owner.hidden_packing_performed is False
    assert owner.no_hidden_allocation_policy is True
    # Bank specs must be non-empty and every bank buffer must already be
    # allocated as an mx.array with the spec's shape and dtype.
    assert owner.bank_specs, "expected at least one physical ABI bank"
    required = set(owner.required_bank_buffers)
    assert required.issubset(owner.buffers)
    for spec in owner.bank_specs:
        buf = owner.buffers[spec.name]
        assert isinstance(buf, mx.array)
        assert tuple(buf.shape) == spec.shape
        assert str(buf.dtype).rsplit(".", 1)[-1] == spec.dtype


def test_make_path_c_physical_abi_bank_owner_buffers_are_zero_initialised(model) -> None:
    owner = model.make_path_c_physical_abi_bank_owner(sequence_length=64)
    for spec in owner.bank_specs:
        buf = owner.buffers[spec.name]
        sample = buf.astype(mx.float32)
        mx.eval(sample)
        assert float(sample.sum().item()) == 0.0


def test_path_c_fused_train_block_prim_func_exposes_physical_abi_attrs(model) -> None:
    prim_func = model.path_c_fused_train_block_prim_func(sequence_length=64)
    assert prim_func is not None
    abi_map = dict(
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {}) or {}
    )
    abi_shapes = dict(
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_shapes", {}) or {}
    )
    # The route region for local_gb10_quarter must emit a banked physical ABI.
    assert abi_map, "physical ABI map must not be empty"
    assert abi_shapes, "physical ABI bank shapes must not be empty"
    # Bank names referenced by the placements must all appear in the shape map.
    placement_banks = {str(info["bank"]) for info in abi_map.values()}
    assert placement_banks <= set(abi_shapes)


def test_bank_owner_uses_generated_stage_union_abi(model) -> None:
    """Stage-generated kernels can require larger physical banks than monolithic ABI."""

    prim_func = model.path_c_fused_train_block_prim_func(sequence_length=64)
    assert prim_func is not None
    monolithic_shapes = dict(
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_shapes", {}) or {}
    )
    _abi_map, merged_shapes, prim_funcs = model.path_c_fused_train_block_physical_abi(
        sequence_length=64,
    )
    owner = model.make_path_c_physical_abi_bank_owner(sequence_length=64)
    assert owner is not None

    assert len(prim_funcs) > 1
    state_bank = "path_c_float32_state_abi_bank"
    assert tuple(merged_shapes[state_bank]) == tuple(owner.buffers[state_bank].shape)
    assert math.prod(tuple(merged_shapes[state_bank])) >= math.prod(
        tuple(monolithic_shapes[state_bank])
    )
    assert tuple(merged_shapes[state_bank]) != tuple(monolithic_shapes[state_bank])


def test_owner_unblocks_m04_fp8_path_c_runtime_binding(model) -> None:
    """The owner must let the m04 route_payload reach a bound runtime."""

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "m04_train_step", str(Path("scripts/m04_train_step.py").resolve())
    )
    assert spec is not None and spec.loader is not None
    m04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m04)

    args = m04.build_parser().parse_args([
        "--synthetic", "--dtype", "fp8_path_c",
        "--model-profile", "local_gb10_quarter",
        "--dry-run-json", "--output", "/tmp/test_owner_route.json",
        "--data-path", "/dev/null", "--data-format", "npz",
    ])
    route = m04.fp8_path_c_training_route_payload_for_model(
        args,
        model,
        auto_install_fused_train_block=True,
    )
    binding = route.get("path_c_fusion", {}).get("runtime_training_binding", {})
    assert binding.get("runtime_uses_fused_train_block") is True, route
    assert binding.get("physical_abi_binding_ready") is True, binding
    assert binding.get("fused_artifact_bound") is True, binding
    assert binding.get("missing_bank_buffers") == [], binding
    assert binding.get("bank_buffer_owner") == (
        "local_gb10_quarter.path_c_physical_abi_banks"
    )


def test_fused_train_block_runtime_is_not_auto_installed_by_default() -> None:
    """The monolithic fused train-block runtime is an explicit opt-in path."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "m04_train_step", str(Path("scripts/m04_train_step.py").resolve())
    )
    assert spec is not None and spec.loader is not None
    m04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m04)

    args = m04.build_parser().parse_args([
        "--synthetic", "--dtype", "fp8_path_c",
        "--model-profile", "local_gb10_quarter",
        "--dry-run-json", "--output", "/tmp/test_artifact_real_metal.json",
        "--data-path", "/dev/null", "--data-format", "npz",
    ])
    fresh_model = build_local_gb10_quarter_tiny_smoke_model()
    route = m04.fp8_path_c_training_route_payload_for_model(args, fresh_model)

    assert route["selected_action"] == "run_path_c_split_training_route"
    assert route["fused_train_block_training_runtime_available"] is False
    assert getattr(fresh_model, "path_c_fused_train_block_artifact", None) is None
    assert getattr(
        fresh_model,
        "path_c_fused_train_block_training_runtime",
        None,
    ) is None
    assert getattr(fresh_model, "path_c_physical_abi_bank_owner", None) is None


def test_path_c_fused_in_region_parameter_bank_aliases_covers_brick_params(
    model,
) -> None:
    aliases = model.path_c_fused_in_region_parameter_bank_aliases()
    # The local_gb10_quarter tiny smoke profile generates the brick_10_M /
    # brick_11_R / brick_12_A region. The replay/cotangent fused block covers
    # only parameters owned by that region; final_norm and lm_head stay in the
    # eager suffix and must not be hidden inside the physical ABI bank aliases.
    # Exact count check locks the discovery surface so future region drift
    # surfaces in tests.
    assert len(aliases) == 26
    expected_subset = {
        "layers.10.norm.weight",
        "layers.10.block.D",
        "layers.11.block.D",
        "layers.12.block.q_proj.weight",
    }
    assert expected_subset.issubset(aliases.keys())
    assert "norm.weight" not in aliases
    assert "lm_head.weight" not in aliases
    for name, info in aliases.items():
        assert "logical_name" in info
        assert "logical_grad_name" in info
        assert info["logical_grad_name"].endswith("_grad")
        assert info["bank"]
        assert info["dtype"] == "float32"
        assert isinstance(info["offset"], int)
        assert isinstance(info["size"], int)
        assert info["size"] > 0
        assert isinstance(info["logical_shape"], tuple)


def test_bind_path_c_in_region_parameter_views_into_bank_replaces_attributes(
    model,
) -> None:
    bank_owner = model.make_path_c_physical_abi_bank_owner()
    report = model.bind_path_c_in_region_parameter_views_into_bank(bank_owner)
    assert report["status"] == "ok"
    assert report["in_region_parameter_count"] == 26
    assert report["skipped"] == []
    # After binding, each in-region parameter must be backed by storage in
    # the same bank slot. Pick a small parameter and a large one to verify
    # the writes happened.
    aliases = report["in_region_parameter_bank_aliases"]
    sample_small = "layers.10.block.D"
    sample_info = aliases[sample_small]
    bank = bank_owner.buffers[sample_info["bank"]]
    slot = bank[sample_info["offset"] : sample_info["offset"] + sample_info["size"]]
    parts = sample_small.split(".")
    holder: Any = model
    for part in parts[:-1]:
        holder = holder[int(part)] if part.isdigit() else getattr(holder, part)
    param_value = getattr(holder, parts[-1])
    # Bank-resident slot equals current parameter value.
    assert mx.allclose(slot, param_value.flatten()).item() is True
    # And the parameter shape was preserved (logical_shape).
    assert param_value.shape == sample_info["logical_shape"]


def test_sync_path_c_in_region_parameters_into_bank_updates_bank_slots(
    model,
) -> None:
    bank_owner = model.make_path_c_physical_abi_bank_owner()
    aliases = model.path_c_fused_in_region_parameter_bank_aliases()
    # Pick layers.10.block.D — known to be size 4.
    name = "layers.10.block.D"
    info = aliases[name]
    bank_name = str(info["bank"])
    bank = bank_owner.buffers[bank_name]
    # Replace the model's parameter with a sentinel tensor and verify the
    # sync writes it into the bank slot in-place (no new bank allocation).
    sentinel = mx.array([7.0, 8.0, 9.0, 10.0], dtype=mx.float32)
    model.layers[10].block.D = sentinel
    bank_id_before = id(bank_owner.buffers[bank_name])
    report = model.sync_path_c_in_region_parameters_into_bank(
        bank_owner,
        in_region_aliases=aliases,
    )
    assert report["status"] == "ok"
    assert name in report["synced"]
    assert report["skipped"] == []
    # The bank object is the same Python-level object after slice assignment.
    assert id(bank_owner.buffers[bank_name]) == bank_id_before
    # And the slot now carries the sentinel values.
    bank = bank_owner.buffers[bank_name]
    slot = bank[info["offset"] : info["offset"] + info["size"]]
    assert mx.allclose(slot, sentinel).item() is True


def test_bind_path_c_syncs_sparse_mla_static_real_abi_inputs() -> None:
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = 127
    bank_owner = model.make_path_c_physical_abi_bank_owner(
        sequence_length=sequence_length,
    )
    assert bank_owner is not None
    report = model.bind_path_c_in_region_parameter_views_into_bank(
        bank_owner,
        sequence_length=sequence_length,
    )
    prim_func = model.path_c_fused_train_block_prim_func(
        sequence_length=sequence_length,
    )
    assert prim_func is not None
    abi_map = dict(
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
        or {}
    )

    def bank_slot(logical_name: str) -> mx.array:
        info = abi_map[logical_name]
        bank = bank_owner.buffers[str(info["bank"])]
        offset = int(info["offset"])
        size = int(info["size"])
        return bank[offset : offset + size]

    sm_scale_name = next(
        name for name in abi_map if name.endswith("sparse_mla_sm_scale")
    )
    sinks_name = next(
        name for name in abi_map if name.endswith("sparse_mla_sinks")
    )
    has_sinks_name = next(
        name for name in abi_map if name.endswith("sparse_mla_has_sinks")
    )
    sm_scale = bank_slot(sm_scale_name)
    sinks = bank_slot(sinks_name)
    has_sinks = bank_slot(has_sinks_name)
    mx.eval(sm_scale, sinks, has_sinks)

    assert report["static_real_abi_inputs_synced"] == [
        has_sinks_name,
        sinks_name,
        sm_scale_name,
    ]
    assert sm_scale.tolist() == [0.5]
    assert sinks.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert has_sinks.tolist() == [0]
