from __future__ import annotations

import pytest
import mlx.core as mx

from cppmega_mlx.runtime.path_c_physical_abi import (
    PathCLogicalBufferOwner,
    compose_path_c_logical_buffer_owner,
    make_physical_abi_bank_owner,
    physical_abi_bank_specs,
    physical_abi_full_runtime_kernel_args,
    physical_abi_runtime_kernel_args,
    normalize_physical_abi_map,
    plan_physical_abi_runtime_bridge,
    validate_physical_abi_map,
    validate_physical_abi_runtime_bindings,
    write_into_bank_slot,
)


def test_validate_physical_abi_map_accepts_disjoint_bank_ranges() -> None:
    mapping = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
        "out": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 4,
            "shape": (2,),
            "size": 2,
        },
        "flag": {
            "bank": "path_c_int32_abi_bank",
            "dtype": "int32",
            "offset": 0,
            "shape": (1,),
            "size": 1,
        },
    }

    payload = validate_physical_abi_map(
        mapping,
        {
            "path_c_float32_abi_bank": (8,),
            "path_c_int32_abi_bank": (1,),
        },
    )

    assert payload["status"] == "ok"
    assert payload["errors"] == []
    assert payload["logical_buffer_count"] == 3
    assert payload["bank_used_elements"]["path_c_float32_abi_bank"] == 6
    assert payload["bank_used_bytes"]["path_c_float32_abi_bank"] == 24


def test_validate_physical_abi_map_rejects_overlaps_and_overflow() -> None:
    mapping = {
        "a": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
        "b": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 3,
            "shape": (4,),
            "size": 4,
        },
    }

    payload = validate_physical_abi_map(
        mapping,
        {"path_c_float32_abi_bank": (6,)},
    )

    assert payload["status"] == "failed"
    assert any("overlaps" in error for error in payload["errors"])
    assert any("exceeds" in error for error in payload["errors"])


def test_normalize_physical_abi_map_rejects_non_mapping_entries() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        normalize_physical_abi_map({"hidden": "path_c_float32_abi_bank"})


def test_runtime_bridge_requires_caller_owned_banks_for_banked_abi() -> None:
    mapping = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
        "out": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 4,
            "shape": (2,),
            "size": 2,
        },
    }

    plan = plan_physical_abi_runtime_bridge(
        mapping,
        {"path_c_float32_abi_bank": (6,)},
    )

    assert plan["status"] == "prepacked_bank_buffers_required"
    assert plan["logical_tensor_binding_supported"] is False
    assert plan["prepacked_bank_binding_supported"] is True
    assert plan["required_bank_buffers"] == ["path_c_float32_abi_bank"]
    assert plan["no_hidden_allocation_policy"] is True
    assert "refuses to pack" in plan["reason"]


def test_physical_abi_bank_specs_report_order_dtype_shape_and_logicals() -> None:
    mapping = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
        "out": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 4,
            "shape": (2,),
            "size": 2,
        },
        "flags": {
            "bank": "path_c_int32_abi_bank",
            "dtype": "int32",
            "offset": 0,
            "shape": (1,),
            "size": 1,
        },
    }

    specs = physical_abi_bank_specs(
        mapping,
        {
            "path_c_float32_abi_bank": (6,),
            "path_c_int32_abi_bank": (1,),
        },
    )

    assert [spec.name for spec in specs] == [
        "path_c_float32_abi_bank",
        "path_c_int32_abi_bank",
    ]
    assert specs[0].dtype == "float32"
    assert specs[0].shape == (6,)
    assert specs[0].elements == 6
    assert specs[0].nbytes == 24
    assert specs[0].logical_buffers == ("hidden", "out")
    assert specs[1].dtype == "int32"
    assert specs[1].logical_buffers == ("flags",)


class _ArrayLike:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype


def test_compose_path_c_logical_buffer_owner_preserves_existing_refs() -> None:
    hidden = _ArrayLike((2, 4), "float32")
    hidden_grad = _ArrayLike((2, 4), "float32")

    owner = compose_path_c_logical_buffer_owner(
        "local_gb10_quarter.path_c_direct_fusion_chain_buffers",
        PathCLogicalBufferOwner(
            "local_gb10_quarter.path_c_model_parameter_buffers",
            {"hidden": hidden},
        ),
        PathCLogicalBufferOwner(
            "local_gb10_quarter.path_c_runtime_activation_buffers",
            {"hidden_grad": hidden_grad},
        ),
    )

    assert owner.owner_name == "local_gb10_quarter.path_c_direct_fusion_chain_buffers"
    assert owner.buffers["hidden"] is hidden
    assert owner.buffers["hidden_grad"] is hidden_grad
    assert owner.hidden_packing_performed is False
    assert owner.no_hidden_allocation_policy is True


def test_compose_path_c_logical_buffer_owner_rejects_conflicting_refs() -> None:
    left = _ArrayLike((2,), "float32")
    right = _ArrayLike((2,), "float32")

    with pytest.raises(ValueError, match="conflicting logical buffer"):
        compose_path_c_logical_buffer_owner(
            "local_gb10_quarter.path_c_direct_fusion_chain_buffers",
            PathCLogicalBufferOwner("left", {"hidden": left}),
            PathCLogicalBufferOwner("right", {"hidden": right}),
        )


def test_runtime_binding_accepts_existing_prepacked_bank_buffers() -> None:
    mapping = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
        "out": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 4,
            "shape": (2,),
            "size": 2,
        },
    }
    bank = _ArrayLike((6,), "float32")

    payload = validate_physical_abi_runtime_bindings(
        mapping,
        {"path_c_float32_abi_bank": (6,)},
        {"path_c_float32_abi_bank": bank},
    )

    assert payload["status"] == "ok"
    assert payload["ordered_kernel_buffers"] == ["path_c_float32_abi_bank"]
    assert payload["missing_bank_buffers"] == []
    assert payload["unexpected_buffers"] == []


def test_make_physical_abi_bank_owner_preserves_existing_buffers() -> None:
    mapping = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
        "flag": {
            "bank": "path_c_int32_abi_bank",
            "dtype": "int32",
            "offset": 0,
            "shape": (1,),
            "size": 1,
        },
    }
    float_bank = _ArrayLike((4,), "float32")
    int_bank = _ArrayLike((1,), "int32")

    owner = make_physical_abi_bank_owner(
        "HybridTinyLM.path_c_physical_abi_banks",
        mapping,
        {
            "path_c_float32_abi_bank": (4,),
            "path_c_int32_abi_bank": (1,),
        },
        {
            "path_c_float32_abi_bank": float_bank,
            "path_c_int32_abi_bank": int_bank,
        },
    )

    assert owner.owner_name == "HybridTinyLM.path_c_physical_abi_banks"
    assert owner.buffers["path_c_float32_abi_bank"] is float_bank
    assert owner.buffers["path_c_int32_abi_bank"] is int_bank
    assert owner.required_bank_buffers == (
        "path_c_float32_abi_bank",
        "path_c_int32_abi_bank",
    )
    assert owner.binding_payload["status"] == "ok"
    assert owner.hidden_packing_performed is False


def test_runtime_kernel_args_preserve_caller_owned_bank_identity() -> None:
    mapping = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
        "flag": {
            "bank": "path_c_int32_abi_bank",
            "dtype": "int32",
            "offset": 0,
            "shape": (1,),
            "size": 1,
        },
    }
    float_bank = _ArrayLike((4,), "float32")
    int_bank = _ArrayLike((1,), "int32")

    args = physical_abi_runtime_kernel_args(
        mapping,
        {
            "path_c_float32_abi_bank": (4,),
            "path_c_int32_abi_bank": (1,),
        },
        {
            "path_c_float32_abi_bank": float_bank,
            "path_c_int32_abi_bank": int_bank,
        },
    )

    assert args == (float_bank, int_bank)


def test_runtime_kernel_args_fail_instead_of_packing_logical_tensors() -> None:
    mapping = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
        "out": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 4,
            "shape": (2,),
            "size": 2,
        },
    }

    with pytest.raises(ValueError, match="missing caller-owned bank buffer"):
        physical_abi_runtime_kernel_args(
            mapping,
            {"path_c_float32_abi_bank": (6,)},
            {
                "hidden": _ArrayLike((4,), "float32"),
                "out": _ArrayLike((2,), "float32"),
            },
        )


def test_full_runtime_kernel_args_append_existing_scratch_buffers_in_kernel_order() -> None:
    mapping = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
        "flag": {
            "bank": "path_c_int32_abi_bank",
            "dtype": "int32",
            "offset": 0,
            "shape": (1,),
            "size": 1,
        },
    }
    float_bank = _ArrayLike((4,), "float32")
    int_bank = _ArrayLike((1,), "int32")
    scratch = _ArrayLike((8,), "float32")

    args = physical_abi_full_runtime_kernel_args(
        mapping,
        {
            "path_c_float32_abi_bank": (4,),
            "path_c_int32_abi_bank": (1,),
        },
        (
            "path_c_float32_abi_bank",
            "path_c_int32_abi_bank",
            "local_scratch",
        ),
        {
            "path_c_float32_abi_bank": float_bank,
            "path_c_int32_abi_bank": int_bank,
            "local_scratch": scratch,
        },
    )

    assert args == (float_bank, int_bank, scratch)


def test_full_runtime_kernel_args_reject_missing_scratch_buffer() -> None:
    mapping = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
    }

    with pytest.raises(ValueError, match="missing caller-owned kernel buffer"):
        physical_abi_full_runtime_kernel_args(
            mapping,
            {"path_c_float32_abi_bank": (4,)},
            ("path_c_float32_abi_bank", "local_scratch"),
            {"path_c_float32_abi_bank": _ArrayLike((4,), "float32")},
        )


def test_runtime_binding_rejects_logical_tensor_substitutes_for_banked_abi() -> None:
    mapping = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (4,),
            "size": 4,
        },
        "out": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 4,
            "shape": (2,),
            "size": 2,
        },
    }

    payload = validate_physical_abi_runtime_bindings(
        mapping,
        {"path_c_float32_abi_bank": (6,)},
        {"hidden": _ArrayLike((4,), "float32"), "out": _ArrayLike((2,), "float32")},
    )

    assert payload["status"] == "failed"
    assert payload["missing_bank_buffers"] == ["path_c_float32_abi_bank"]
    assert payload["unexpected_buffers"] == ["hidden", "out"]
    assert any("missing caller-owned bank buffer" in error for error in payload["errors"])


def test_write_into_bank_slot_rejects_dtype_mismatch() -> None:
    mapping = {
        "target_ids": {
            "bank": "path_c_int32_abi_bank",
            "dtype": "int32",
            "offset": 0,
            "shape": (4,),
            "logical_shape": (4,),
            "size": 4,
        },
    }
    buffers = {"path_c_int32_abi_bank": mx.zeros((4,), dtype=mx.int32)}

    with pytest.raises(ValueError, match="dtype 'float32'.*expected 'int32'"):
        write_into_bank_slot(
            mapping,
            buffers,
            "target_ids",
            mx.array([1.5, 2.5, 3.5, 4.5], dtype=mx.float32),
        )
