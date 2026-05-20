from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "path_c_fusion_compile_receipt.py"

SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "path_c_fusion_compile_receipt",
    SCRIPT,
)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
path_c_fusion_compile_receipt = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = path_c_fusion_compile_receipt
SCRIPT_SPEC.loader.exec_module(path_c_fusion_compile_receipt)


def test_compile_receipt_plans_model_derived_fused_schedule(tmp_path: Path) -> None:
    source_out = tmp_path / "generated.py"

    exit_code, payload = path_c_fusion_compile_receipt.build_compile_receipt(
        native_compile=False,
        source_out=source_out,
    )

    assert exit_code == 0
    assert payload["kind"] == "cppmega_path_c_fusion_compile_receipt"
    assert payload["status"] == "ok"
    assert payload["native_compile_requested"] is False
    assert payload["reporting_contract"] == {
        "matrix_measures_current_runtime_route": True,
        "compile_receipt_measures_fused_schedule_compile": True,
        "runtime_uses_fused_train_block": False,
        "path_c_default_allowed": False,
    }
    assert payload["schedule_target"]["implementation_kind"] == "production"
    assert payload["schedule_spec"]["production_fragments_complete"] is True
    assert payload["schedule_spec"]["real_abi_contract_complete"] is True
    assert payload["compile_plan"]["schedule_contract"]["status"] == (
        "registered_not_lowered"
    )
    assert payload["fusion_cache_key"]["status"] == "plan_only_no_lowered_module"
    assert payload["fusion_cache_key"]["cache_key_material"]
    assert payload["fusion_cache_key"]["cache_key_material_sha256"]
    assert payload["cache_key_recompile_audit"]["status"] == (
        "skipped_no_native_compile"
    )
    assert payload["cache_key_recompile_audit"]["primary"]["status"] == (
        "plan_only_no_lowered_module"
    )
    assert payload["cache_key_recompile_audit"]["second"] is None
    assert payload["runtime_execution_contract"]["status"] == (
        "compile_only_not_runtime_ready"
    )
    assert payload["runtime_execution_contract"]["runtime_route_uses_fused_region"] is False
    assert payload["runtime_execution_contract"]["single_thread_kernel"] is False
    assert payload["runtime_execution_contract"]["lane0_serial_fragments"] is True
    assert payload["runtime_execution_contract"]["lane0_fragment_markers"]
    assert "attention_qkv_projection" not in payload["runtime_execution_contract"][
        "lane0_production_fragments"
    ]
    assert "m2rnn" not in payload["runtime_execution_contract"][
        "lane0_production_fragments"
    ]
    assert "mamba3_mimo" not in payload["runtime_execution_contract"][
        "lane0_production_fragments"
    ]
    assert payload["runtime_execution_contract"]["lane_strided_row_loops"] is True
    assert payload["runtime_execution_contract"]["plan_single_kernel_fused"] is False
    assert payload["runtime_execution_contract"]["plan_schedule_status"] == "ready"
    assert payload["runtime_execution_contract"]["physical_abi_runtime_bridge_status"] == (
        "prepacked_bank_buffers_required"
    )
    assert payload["runtime_execution_contract"]["physical_abi_runtime_binding_status"] == (
        "not_bound"
    )
    assert payload["runtime_execution_contract"]["physical_abi_missing_bank_buffers"] == [
        "path_c_float32_abi_bank",
        "path_c_uint8_abi_bank",
        "path_c_int32_abi_bank",
    ]
    assert payload["runtime_execution_contract"]["schedule_contract_status"] == (
        "registered_not_lowered"
    )
    assert payload["runtime_execution_contract"]["schedule_contract_declared_kind"] == (
        "production"
    )
    assert payload["runtime_execution_contract"][
        "schedule_contract_declared_schedule_id"
    ].startswith("path_c_descriptor_chain_")
    assert payload["runtime_execution_contract"]["kernel_parameter_count"] <= (
        payload["runtime_execution_contract"]["metal_buffer_limit"]
    )
    assert payload["runtime_execution_contract"]["metal_buffer_limit"] == 31
    assert (
        payload["runtime_execution_contract"]["metal_buffer_limit_exceeded"]
        is False
    )
    assert payload["generated_source"]["physical_abi_policy"] == "banked_by_dtype"
    assert payload["generated_source"]["logical_parameter_count"] > (
        payload["runtime_execution_contract"]["kernel_parameter_count"]
    )
    assert payload["generated_source"]["logical_buffer_abi_map_count"] > (
        payload["runtime_execution_contract"]["kernel_parameter_count"]
    )
    hidden_mapping = payload["generated_source"]["physical_buffer_abi_map"]["hidden"]
    assert hidden_mapping["bank"] == "path_c_float32_abi_bank"
    assert hidden_mapping["offset"] == 0
    assert hidden_mapping["dtype"] == "float32"
    assert payload["generated_source"]["physical_abi_validation"]["status"] == "ok"
    assert (
        payload["generated_source"]["physical_abi_validation"]["logical_buffer_count"]
        == payload["generated_source"]["logical_buffer_abi_map_count"]
    )
    assert payload["generated_source"]["physical_abi_runtime_bridge"]["status"] == (
        "prepacked_bank_buffers_required"
    )
    assert payload["generated_source"]["physical_abi_runtime_bridge"][
        "logical_tensor_binding_supported"
    ] is False
    assert payload["generated_source"]["physical_abi_runtime_binding"]["status"] == (
        "not_bound"
    )
    assert payload["generated_source"]["physical_abi_runtime_binding"][
        "missing_bank_buffers"
    ] == [
        "path_c_float32_abi_bank",
        "path_c_uint8_abi_bank",
        "path_c_int32_abi_bank",
    ]
    assert payload["generated_source"]["spilled_shared_scratch_shapes"]
    assert payload["generated_source"]["shared_scratch_abi_bytes"] > 0
    assert "local_gb10_quarter_brick_10_M_delta" in (
        payload["generated_source"]["internal_scratch_abi_buffers"]
    )
    blockers = payload["runtime_execution_contract"]["blockers"]
    assert len(blockers) == 3
    assert blockers[0].startswith(
        "compile plan is not verified as the runtime single-kernel fused path"
    )
    assert "schedule_status=ready" in blockers[0]
    assert blockers[1].startswith(
        "schedule contract is not verified by this build"
    )
    assert "status=registered_not_lowered" in blockers[1]
    assert blockers[2].startswith(
        "physical ABI runtime binding is not ready"
    )
    assert payload["runtime_smoke"] == {
        "status": "not_requested",
        "mode": "none",
        "actually_executed": False,
    }
    assert payload["default_eligible"] is False
    assert source_out.exists()
    assert "def local_gb10_quarter_path_c_10_12" in source_out.read_text(
        encoding="utf-8"
    )


def test_compile_receipt_records_native_lowerer_artifact(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_lowerer(func_or_mod: object, **kwargs: object) -> object:
        captured["func_or_mod"] = func_or_mod
        captured["kwargs"] = kwargs
        return SimpleNamespace(name="fake-jit-kernel")

    exit_code, payload = path_c_fusion_compile_receipt.build_compile_receipt(
        native_compile=True,
        source_out=tmp_path / "generated.py",
        lowerer=fake_lowerer,
    )

    assert exit_code == 0
    assert payload["native_compile_requested"] is True
    assert payload["native_compile_ok"] is True
    assert payload["artifact"]["type"] == "SimpleNamespace"
    assert payload["compile_plan"]["single_kernel_fused"] is True
    assert payload["compile_plan"]["schedule_status"] == "ready"
    assert payload["compile_plan"]["schedule_contract"]["status"] == "verified"
    assert payload["runtime_execution_contract"]["plan_single_kernel_fused"] is True
    assert payload["runtime_execution_contract"]["schedule_contract_status"] == "verified"
    assert payload["fusion_cache_key"]["status"] in {
        "lowered_module_digest_recorded",
        "plan_only_no_lowered_module",
    }
    assert payload["cache_key_recompile_audit"]["status"] == "key_stable"
    assert payload["cache_key_recompile_audit"]["cache_hit_observed"] is None
    assert payload["cache_key_recompile_audit"]["cache_hit_status"] == (
        "not_observed_by_tilelang_api"
    )
    assert payload["cache_key_recompile_audit"]["second"]["native_compile_ok"] is True
    assert payload["generated_source"]["compile_pass_configs"] == {
        "tirx.disable_cse_tir": True,
    }
    assert captured["kwargs"]["target"] == "metal"


def test_compile_receipt_default_lowerer_uses_requested_execution_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_tilelang_lowerer(
        func_or_mod: object,
        *,
        target: str,
        execution_backend: str,
        **kwargs: object,
    ) -> object:
        captured["func_or_mod"] = func_or_mod
        captured["target"] = target
        captured["execution_backend"] = execution_backend
        captured["kwargs"] = kwargs
        return SimpleNamespace(name="fake-default-jit-kernel")

    monkeypatch.setattr(
        path_c_fusion_compile_receipt,
        "tilelang_single_entry_lowerer",
        fake_tilelang_lowerer,
    )

    exit_code, payload = path_c_fusion_compile_receipt.build_compile_receipt(
        native_compile=True,
        source_out=tmp_path / "generated.py",
        execution_backend="tvm",
    )

    assert exit_code == 0
    assert payload["native_compile_ok"] is True
    assert payload["execution_backend"] == "tvm"
    assert captured["target"] == "metal"
    assert captured["execution_backend"] == "tvm"


def test_compile_receipt_can_execute_tiny_banked_abi_runtime_smoke(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_artifact(*args: object) -> list[object]:
        captured["arg_count"] = len(args)
        captured["arg_shapes"] = [tuple(getattr(arg, "shape", ())) for arg in args]
        return []

    def fake_lowerer(func_or_mod: object, **kwargs: object) -> object:
        captured["func_or_mod"] = func_or_mod
        captured["kwargs"] = kwargs
        return fake_artifact

    exit_code, payload = path_c_fusion_compile_receipt.build_compile_receipt(
        native_compile=False,
        source_out=tmp_path / "generated.py",
        runtime_smoke="tiny",
        runtime_smoke_lowerer=fake_lowerer,
    )

    smoke = payload["runtime_smoke"]
    assert exit_code == 0
    assert smoke["status"] == "ok"
    assert smoke["mode"] == "tiny_mra"
    assert smoke["actually_executed"] is True
    assert smoke["physical_abi_policy"] == "banked_by_dtype"
    assert smoke["physical_abi_runtime_binding"]["status"] == "ok"
    assert smoke["physical_abi_runtime_binding"]["ordered_kernel_buffers"] == [
        "path_c_float32_abi_bank",
        "path_c_uint8_abi_bank",
        "path_c_int32_abi_bank",
    ]
    assert smoke["kernel_parameter_count"] == 3
    assert smoke["logical_parameter_count"] > smoke["kernel_parameter_count"]
    assert smoke["total_buffer_bytes"] < smoke["max_buffer_bytes"]
    assert [entry["name"] for entry in smoke["buffer_abi"]] == [
        "path_c_float32_abi_bank",
        "path_c_uint8_abi_bank",
        "path_c_int32_abi_bank",
    ]
    assert [entry["dtype"] for entry in smoke["buffer_abi"]] == [
        "float32",
        "uint8",
        "int32",
    ]
    assert captured["arg_count"] == 3
    assert captured["arg_shapes"] == [
        tuple(entry["shape"]) for entry in smoke["buffer_abi"]
    ]
    assert captured["kwargs"]["target"] == "metal"
