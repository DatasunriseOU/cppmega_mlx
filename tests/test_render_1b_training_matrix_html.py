from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_1b_training_matrix_html.py"
SPEC = importlib.util.spec_from_file_location("render_1b_training_matrix_html", SCRIPT)
assert SPEC is not None
renderer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


def _row(
    *,
    case_id: str,
    dtype: str,
    optimizer: str,
    path: str,
    status: str = "ok",
    tok_sec: float | None = None,
    peak_memory_gb: float | None = None,
    active_memory_gb: float | None = None,
    cache_memory_gb: float | None = None,
    reason: str = "ok",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "dtype": dtype,
        "optimizer": optimizer,
        "path": path,
        "status": status,
        "tok_sec": tok_sec,
        "step_sec": 1.0,
        "compile_time_s": 2.0,
        "peak_memory_gb": peak_memory_gb,
        "active_memory_gb": active_memory_gb,
        "cache_memory_gb": cache_memory_gb,
        "cache_hit": True,
        "selected_schedule": json.dumps({"path_counts": {"path_c": 1}}),
        "proof_result": json.dumps({"path": path}),
        "pass_fail_reason": reason,
        "command": f"run {case_id}",
        "receipt_path": f"/tmp/{case_id}.json",
    }


def test_renderer_marks_path_c_default_candidate_and_keep_path_b(tmp_path: Path) -> None:
    payload = {
        "config": {"batch_size": 1, "block_size": 2048, "steps": 20},
        "software": {
            "cppmega_sha": "cpp",
            "tilelang_sha": "tl",
            "mlx_version": "mlx",
        },
        "results": [
            _row(
                case_id="bf16_adamw_path_b",
                dtype="bf16",
                optimizer="adamw",
                path="path_b",
                tok_sec=100.0,
                peak_memory_gb=10.0,
                active_memory_gb=5.0,
                cache_memory_gb=2.0,
            ),
            _row(
                case_id="bf16_adamw_path_c_cold",
                dtype="bf16",
                optimizer="adamw",
                path="path_c_cold",
                tok_sec=95.0,
                peak_memory_gb=11.0,
            ),
            _row(
                case_id="bf16_adamw_path_c_warm",
                dtype="bf16",
                optimizer="adamw",
                path="path_c_warm",
                tok_sec=98.0,
                peak_memory_gb=11.0,
                active_memory_gb=5.0,
                cache_memory_gb=1.0,
            ),
            _row(
                case_id="bf16_lion_path_b",
                dtype="bf16",
                optimizer="lion",
                path="path_b",
                tok_sec=100.0,
                peak_memory_gb=10.0,
            ),
            _row(
                case_id="bf16_lion_path_c_cold",
                dtype="bf16",
                optimizer="lion",
                path="path_c_cold",
                tok_sec=80.0,
                peak_memory_gb=12.0,
            ),
            _row(
                case_id="bf16_lion_path_c_warm",
                dtype="bf16",
                optimizer="lion",
                path="path_c_warm",
                tok_sec=80.0,
                peak_memory_gb=12.0,
            ),
        ],
    }
    input_path = tmp_path / "matrix.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "matrix.html"
    rc = renderer.main(
        [
            "--input",
            str(input_path),
            "--out",
            str(out),
            "--dtypes",
            "bf16",
        ]
    )

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "Path C default candidate" in text
    assert "Keep Path B" in text
    assert "0.980x" in text
    assert "0.800x" in text
    assert "transient peak/high-water; active is flat" in text


def test_renderer_compares_fp8_path_b_baseline(tmp_path: Path) -> None:
    payload = {
        "config": {"batch_size": 1, "block_size": 2048, "steps": 20},
        "software": {
            "cppmega_sha": "cpp",
            "tilelang_sha": "tl",
            "mlx_version": "mlx",
        },
        "results": [
            _row(
                case_id="fp8_adamw_path_b",
                dtype="fp8",
                optimizer="adamw",
                path="path_b",
                tok_sec=100.0,
                peak_memory_gb=10.0,
            ),
            _row(
                case_id="fp8_adamw_path_c_cold",
                dtype="fp8",
                optimizer="adamw",
                path="path_c_cold",
                tok_sec=101.0,
                peak_memory_gb=11.0,
            ),
            _row(
                case_id="fp8_adamw_path_c_warm",
                dtype="fp8",
                optimizer="adamw",
                path="path_c_warm",
                tok_sec=102.0,
                peak_memory_gb=11.0,
            ),
        ],
    }
    input_path = tmp_path / "matrix.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "matrix.html"

    rc = renderer.main(
        [
            "--input",
            str(input_path),
            "--out",
            str(out),
            "--dtypes",
            "fp8",
        ]
    )

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "Path C default candidate" in text
    assert "1.020x" in text
    assert "not_applicable" not in text


def test_renderer_attaches_fused_compile_receipt(tmp_path: Path) -> None:
    payload = {
        "config": {"batch_size": 1, "block_size": 2048, "steps": 20},
        "software": {
            "cppmega_sha": "cpp",
            "tilelang_sha": "tl",
            "mlx_version": "mlx",
        },
        "results": [
            _row(
                case_id="bf16_adamw_path_b",
                dtype="bf16",
                optimizer="adamw",
                path="path_b",
                tok_sec=100.0,
                peak_memory_gb=10.0,
            ),
        ],
    }
    compile_receipt = {
        "status": "ok",
        "native_compile_requested": True,
        "native_compile_ok": True,
        "elapsed_s": 1.25,
        "schedule_target": {
            "schedule_id": "path_c_descriptor_chain_abc",
            "schedule_status": "ready",
        },
        "schedule_spec": {
            "implementation_kind": "production",
            "production_fragments_complete": True,
            "real_abi_contract_complete": True,
            "missing_real_abi_inputs": [],
        },
        "compile_plan": {
            "schedule_contract": {"status": "verified"}
        },
        "artifact": {"type": "JITKernel"},
        "generated_source": {
            "path": "reports/path_c_fusion_compile_source.py",
            "sha256": "abc123",
            "logical_buffer_abi_map_count": 78,
            "physical_abi_validation": {"status": "ok"},
            "physical_abi_runtime_bridge": {
                "status": "prepacked_bank_buffers_required",
                "logical_tensor_binding_supported": False,
                "prepacked_bank_binding_supported": True,
                "required_bank_buffers": [
                    "path_c_float32_abi_bank",
                    "path_c_uint8_abi_bank",
                ],
                "reason": "runtime bridge refuses to pack model tensors implicitly",
            },
            "physical_abi_runtime_binding": {
                "status": "not_bound",
                "missing_bank_buffers": [
                    "path_c_float32_abi_bank",
                    "path_c_uint8_abi_bank",
                ],
            },
            "spilled_shared_scratch_shapes": {
                "mamba3_delta": {
                    "shape": [3584],
                    "dtype": "float32",
                    "bytes": 14336,
                    "internal_scratch_abi": True,
                },
                "mamba3_projected_vec": {
                    "shape": [18784],
                    "dtype": "float32",
                    "bytes": 75136,
                    "internal_scratch_abi": False,
                },
            },
            "spilled_shared_scratch_count": 2,
            "shared_scratch_abi_bytes": 89472,
            "internal_scratch_abi_buffers": ["mamba3_delta"],
            "internal_scratch_abi_count": 1,
        },
        "fusion_cache_key": {
            "status": "lowered_module_digest_recorded",
            "lowered_module_digest": "digest123",
        },
        "cache_key_recompile_audit": {
            "status": "key_stable",
            "cache_hit_status": "not_observed_by_tilelang_api",
        },
        "runtime_execution_contract": {
            "status": "compile_only_not_runtime_ready",
            "plan_single_kernel_fused": True,
            "plan_schedule_status": "ready",
            "schedule_contract_status": "verified",
            "schedule_contract_reason": "native lowering verified the schedule contract",
            "schedule_contract_declared_schedule_id": "path_c_descriptor_chain_abc",
            "single_thread_kernel": False,
            "lane0_serial_fragments": True,
            "lane0_production_fragments": ["mamba3_mimo", "m2rnn"],
            "lane_strided_row_loops": True,
            "blockers": ["generated row-phased schedule still uses lane == 0"],
        },
        "runtime_smoke": {
            "mode": "tiny_mra",
            "status": "ok",
            "actually_executed": True,
            "schedule_id": "path_c_descriptor_chain_smoke",
            "kernel_parameter_count": 3,
            "total_buffer_bytes": 58300,
            "compile_elapsed_s": 0.25,
            "execute_elapsed_s": 0.01,
        },
        "reporting_contract": {
            "matrix_measures_current_runtime_route": True,
            "runtime_uses_fused_train_block": False,
            "path_c_default_allowed": False,
        },
    }
    input_path = tmp_path / "matrix.json"
    receipt_path = tmp_path / "compile.json"
    out = tmp_path / "matrix.html"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.write_text(json.dumps(compile_receipt), encoding="utf-8")

    rc = renderer.main(
        [
            "--input",
            str(input_path),
            "--compile-receipt",
            str(receipt_path),
            "--out",
            str(out),
            "--dtypes",
            "bf16",
        ]
    )

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "Fused Schedule Compile Receipt" in text
    assert "path_c_descriptor_chain_abc" in text
    assert "Runtime uses fused train block" in text
    assert "False" in text
    assert "Fusion cache-key status" in text
    assert "Recompile audit status" in text
    assert "key_stable" in text
    assert "not_observed_by_tilelang_api" in text
    assert "digest123" in text
    assert "Runtime execution status" in text
    assert "compile_only_not_runtime_ready" in text
    assert "Logical ABI map entries" in text
    assert "78" in text
    assert "Physical ABI validation" in text
    assert "Runtime ABI bridge" in text
    assert "prepacked_bank_buffers_required" in text
    assert "Runtime ABI binding" in text
    assert "not_bound" in text
    assert "path_c_float32_abi_bank, path_c_uint8_abi_bank" in text
    assert "Plan schedule status" in text
    assert "ready" in text
    assert "Contract runtime status" in text
    assert "verified" in text
    assert "native lowering verified the schedule contract" in text
    assert "Shared scratch ABI bytes" in text
    assert "89472" in text
    assert "Internal scratch names" in text
    assert "mamba3_delta" in text
    assert "mamba3_projected_vec" in text
    assert "Lane-0 serialized fragments" in text
    assert "Lane-0 production fragments" in text
    assert "mamba3_mimo, m2rnn" in text
    assert "Lane-strided row loops" in text
    assert "lane == 0" in text
    assert "Runtime Smoke" in text
    assert "path_c_descriptor_chain_smoke" in text
    assert "58300" in text


def test_compile_receipt_blocks_speed_only_rows_from_default_label(
    tmp_path: Path,
) -> None:
    payload = {
        "config": {
            "batch_size": 1,
            "block_size": 2048,
            "steps": 20,
            "mamba3_bwd": "path_c",
        },
        "software": {
            "cppmega_sha": "cpp",
            "tilelang_sha": "tl",
            "mlx_version": "mlx",
        },
        "results": [
            _row(
                case_id="bf16_adamw_path_b",
                dtype="bf16",
                optimizer="adamw",
                path="path_b",
                tok_sec=100.0,
                peak_memory_gb=10.0,
            ),
            _row(
                case_id="bf16_adamw_path_c_warm",
                dtype="bf16",
                optimizer="adamw",
                path="path_c_warm",
                tok_sec=110.0,
                peak_memory_gb=11.0,
            ),
        ],
    }
    compile_receipt = {
        "status": "ok",
        "compile_plan": {
            "schedule_contract": {"status": "verified"}
        },
        "reporting_contract": {
            "runtime_uses_fused_train_block": False,
            "production_runtime_smoke_uses_fused_train_block": True,
            "path_c_default_allowed": False,
        },
    }
    input_path = tmp_path / "matrix.json"
    receipt_path = tmp_path / "compile.json"
    out = tmp_path / "matrix.html"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.write_text(json.dumps(compile_receipt), encoding="utf-8")

    rc = renderer.main(
        [
            "--input",
            str(input_path),
            "--compile-receipt",
            str(receipt_path),
            "--out",
            str(out),
            "--dtypes",
            "bf16",
        ]
    )

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "Path C speed candidate" in text
    assert "Path C default candidate" not in text
    assert "train-step runtime fused train block is False" in text
    assert "production smoke fused block is True" in text
    assert "--mamba3-bwd path_c" in text
