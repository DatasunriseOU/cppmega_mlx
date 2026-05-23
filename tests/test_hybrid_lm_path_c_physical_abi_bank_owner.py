"""HybridTinyLM.make_path_c_physical_abi_bank_owner.

Verifies that the model exposes the model-owned physical ABI bank factory
the m04 Path C auto-install path requires. The check exercises the actual
production schedule (no monkeypatching) and asserts that the resulting
owner satisfies the structural contract enforced by
``cppmega_mlx.runtime.path_c_physical_abi.make_physical_abi_bank_owner``.
"""

from __future__ import annotations

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
    assert set(owner.buffers) == required
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
    route = m04.fp8_path_c_training_route_payload_for_model(args, model)
    binding = route.get("path_c_fusion", {}).get("runtime_training_binding", {})
    assert binding.get("runtime_uses_fused_train_block") is True, route
    assert binding.get("physical_abi_binding_ready") is True, binding
    assert binding.get("fused_artifact_bound") is True, binding
    assert binding.get("missing_bank_buffers") == [], binding
    assert binding.get("bank_buffer_owner") == (
        "local_gb10_quarter.path_c_physical_abi_banks"
    )


def test_artifact_forward_runs_end_to_end_on_real_metal(model) -> None:
    """End-to-end live check: the fused TileLang kernel actually runs on Metal.

    Drives the production route to compile the artifact, then invokes
    ``artifact.forward(bank_owner=...)`` with the model-owned banks the
    factory produced, and confirms the call returns without raising and
    that lazy MLX work materialises.
    """
    import importlib.util
    from pathlib import Path

    import mlx.core as mx

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
    route = m04.fp8_path_c_training_route_payload_for_model(args, model)
    assert route["selected_action"] == "run_path_c_fused_train_block_route"

    artifact = model.path_c_fused_train_block_artifact
    bank_owner = model.path_c_physical_abi_bank_owner
    assert artifact is not None
    assert bank_owner is not None
    # The factory's banks must align with the artifact's expected shapes.
    assert artifact.physical_abi_shapes == {
        name: tuple(buf.shape) for name, buf in bank_owner.buffers.items()
    }
    # And the artifact's forward must complete and materialise lazy work.
    artifact.forward(bank_owner=bank_owner)
    mx.eval(*bank_owner.buffers.values())


def test_path_c_fused_in_region_parameter_bank_aliases_covers_brick_params(
    model,
) -> None:
    aliases = model.path_c_fused_in_region_parameter_bank_aliases()
    # The local_gb10_quarter tiny smoke profile generates the brick_10_M /
    # brick_11_R / brick_12_A region. The fused suffix also covers
    # final_norm + lm_head + per-layer residual norms. Exact count check
    # locks the discovery surface so future region drift surfaces in tests.
    assert len(aliases) == 27
    expected_subset = {
        "layers.10.block.D",
        "layers.11.block.D",
        "layers.12.block.q_proj.weight",
        "norm.weight",
        "lm_head.weight",
    }
    assert expected_subset.issubset(aliases.keys())
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
    assert report["in_region_parameter_count"] == 27
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
    holder: object = model
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
    bank_name = "path_c_float32_abi_bank"
    bank = bank_owner.buffers[bank_name]
    # Pick layers.10.block.D — known to be size 4.
    name = "layers.10.block.D"
    info = aliases[name]
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
