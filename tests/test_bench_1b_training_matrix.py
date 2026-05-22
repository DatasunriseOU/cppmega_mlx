from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "bench_1b_training_matrix.py"
SPEC = importlib.util.spec_from_file_location("bench_1b_training_matrix", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def _args(tmp_path: Path, *extra: str):
    return matrix.build_parser().parse_args(
        [
            "--work-dir",
            str(tmp_path / "cells"),
            "--tilelang-cache-dir",
            str(tmp_path / "tilelang-cache"),
            "--out",
            str(tmp_path / "matrix.md"),
            "--csv",
            str(tmp_path / "matrix.csv"),
            "--json",
            str(tmp_path / "matrix.json"),
            *extra,
        ]
    )


def test_bench_1b_matrix_plan_covers_dtype_optimizer_path_cells(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    cells = matrix.build_cells(args)

    assert len(cells) == 42
    by_case = {cell.case_id: cell for cell in cells}
    assert by_case["bf16_adamw_path_b"].env["CPPMEGA_KERNEL_PATH"] == "path_b"
    assert by_case["bf16_adamw_path_b"].env["CPPMEGA_KERNEL_PATH__MAMBA3_MIMO"] == "path_b"
    assert by_case["bf16_adamw_path_b"].env["CPPMEGA_KERNEL_PATH__M2RNN"] == "path_b"
    assert by_case["bf16_adamw_path_c_cold"].env["CPPMEGA_KERNEL_PATH"] == "path_c"
    assert by_case["bf16_adamw_path_c_cold"].env["CPPMEGA_KERNEL_PATH__MAMBA3_MIMO"] == "path_c"
    assert by_case["bf16_adamw_path_c_cold"].env["CPPMEGA_KERNEL_PATH__M2RNN"] == "path_c"
    assert by_case["bf16_adamw_path_c_cold"].env["CPPMEGA_MAMBA3_PATH_C_BWD"] == "path_b"
    assert by_case["bf16_adamw_path_c_cold"].cache_mode == "cold"
    assert by_case["bf16_adamw_path_c_warm"].cache_mode == "warm"
    assert "--seq-len" in by_case["bf16_adamw_path_b"].command
    assert "2048" in by_case["bf16_adamw_path_b"].command
    assert by_case["bf16_lion_path_b"].cli_optimizer == "lion"
    assert by_case["bf16_lion8bit_path_b"].cli_optimizer == "lion8bit"
    assert by_case["bf16_adam8bit_path_b"].cli_optimizer == "adam8bit"
    assert by_case["bf16_muon_int8_path_c_cold"].cli_optimizer == "int8"
    assert by_case["fp8_adamw_path_b"].supported is True
    assert by_case["fp8_adamw_path_b"].dtype_arg == "fp8_path_b"
    assert by_case["fp8_lion8bit_path_b"].dtype_arg == "fp8_path_b"
    assert by_case["fp8_lion8bit_path_b"].cli_optimizer == "lion8bit"
    assert by_case["fp8_adamw_path_b"].env["CPPMEGA_KERNEL_PATH"] == "path_b"
    assert by_case["fp8_adamw_path_b"].env["CPPMEGA_KERNEL_PATH__SPARSE_MLA"] == "path_b"
    assert by_case["fp8_adamw_path_c_warm"].env["CPPMEGA_SPARSE_MLA_FP8_ROUTE"] == "path_c"
    assert "--use-path-c-direct-chain-runtime" not in by_case[
        "bf16_adamw_path_c_warm"
    ].command


def test_bench_1b_matrix_can_explicitly_force_full_mamba3_path_c_bwd(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        "--dtypes",
        "bf16",
        "--optimizers",
        "adamw",
        "--paths",
        "path_c_warm",
        "--mamba3-bwd",
        "path_c",
    )
    cell = matrix.build_cells(args)[0]

    assert cell.env["CPPMEGA_KERNEL_PATH__MAMBA3_MIMO"] == "path_c"
    assert cell.env["CPPMEGA_MAMBA3_PATH_C_BWD"] == "path_c"


def test_bench_1b_matrix_can_forward_direct_chain_runtime_flag(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        "--dtypes",
        "bf16",
        "--optimizers",
        "adamw",
        "--paths",
        "path_b,path_c_warm",
        "--use-path-c-direct-chain-runtime",
    )
    cells = matrix.build_cells(args)
    by_case = {cell.case_id: cell for cell in cells}

    assert "--use-path-c-direct-chain-runtime" not in by_case[
        "bf16_adamw_path_b"
    ].command
    assert "--use-path-c-direct-chain-runtime" in by_case[
        "bf16_adamw_path_c_warm"
    ].command


def test_bench_1b_matrix_can_plan_profile_capture_wrapped_command(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        "--capture-profiles",
        "--profile-capture-timeout-s",
        "123.5",
        "--dtypes",
        "bf16",
        "--optimizers",
        "adamw",
        "--paths",
        "path_c_warm",
    )
    cell = matrix.build_cells(args)[0]

    trace_path, capture_receipt_path = matrix.profile_capture_paths(cell, args)
    command = matrix.profile_capture_command(
        cell.command,
        trace_path=trace_path,
        timeout_s=args.profile_capture_timeout_s,
    )

    assert trace_path == (
        tmp_path / "cells" / "profiles" / "bf16_adamw_path_c_warm.gputrace"
    )
    assert capture_receipt_path == (
        tmp_path / "cells" / "profiles" / "bf16_adamw_path_c_warm_capture.json"
    )
    assert command[:2] == (sys.executable, "scripts/profile_capture.py")
    assert "--trace-path" in command
    assert str(trace_path) in command
    assert "--timeout-s" in command
    assert "123.5" in command
    assert command[-len(cell.command) :] == cell.command


def test_bench_1b_matrix_dry_run_writes_markdown_csv_and_json(
    tmp_path: Path,
) -> None:
    rc = matrix.main(
        [
            "--dry-run",
            "--dtypes",
            "bf16,fp8",
            "--optimizers",
            "adamw",
            "--paths",
            "path_b,path_c_warm",
            "--work-dir",
            str(tmp_path / "cells"),
            "--tilelang-cache-dir",
            str(tmp_path / "tilelang-cache"),
            "--out",
            str(tmp_path / "matrix.md"),
            "--csv",
            str(tmp_path / "matrix.csv"),
            "--json",
            str(tmp_path / "matrix.json"),
        ]
    )

    assert rc == 0
    markdown = (tmp_path / "matrix.md").read_text(encoding="utf-8")
    assert "cppmega SHA" in markdown
    assert "MLX SHA" in markdown
    assert "fp8_adamw_path_b" in markdown
    assert "--dtype fp8_path_b" in markdown
    assert "profile trace" in markdown
    rows = list(csv.DictReader((tmp_path / "matrix.csv").open(encoding="utf-8")))
    assert len(rows) == 4
    statuses = {row["case_id"]: row["status"] for row in rows}
    assert statuses["bf16_adamw_path_b"] == "planned"
    assert statuses["fp8_adamw_path_b"] == "planned"
    assert rows[0]["profiling_trace_captured"] == "False"
    assert rows[0]["profiling_trace_path"] == ""
    assert rows[0]["profiling_capture_receipt_path"] == ""
    assert rows[0]["profiling_capture_status"] == ""
    payload = json.loads((tmp_path / "matrix.json").read_text(encoding="utf-8"))
    assert payload["scope"] == "cppmega_1b_path_matrix"
    assert payload["config"]["block_size"] == 2048
    assert payload["config"]["mamba3_bwd"] == "path_b"
    first_result = payload["results"][0]
    assert first_result["profiling_trace_captured"] is False
    assert first_result["profiling_trace_path"] is None
    assert first_result["profiling_capture_receipt_path"] is None
    assert first_result["profiling_capture_status"] is None


def test_bench_1b_matrix_extracts_m04_receipt_metrics(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        "--dtypes",
        "bf16",
        "--optimizers",
        "adamw",
        "--paths",
        "path_c_cold",
    )
    cell = matrix.build_cells(args)[0]
    cell.output_json.parent.mkdir(parents=True)
    cell.output_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "timing": {
                    "step_times_s": [2.0, 1.0, 1.5],
                    "mean_step_time_s": 1.5,
                    "tokens_per_second": 2048.0,
                },
                "memory": {
                    "peak_memory_bytes": 1 << 30,
                    "after": {
                        "active_memory_bytes": 512 << 20,
                        "cache_memory_bytes": 768 << 20,
                    },
                },
                "profiling": {
                    "trace_path": "reports/profiling/bf16_adamw_path_c_cold.gputrace",
                    "capture_receipt_path": "reports/profiling/bf16_adamw_path_c_cold_capture.json",
                    "capture_status": "ok",
                },
                "training": {
                    "all_finite": True,
                    "kernel_dispatch": [
                        {
                            "op_name": "mamba3_mimo",
                            "path": "path_c",
                            "kernel_used": "path_c_tilelang_dsl",
                        }
                    ],
                    "fp8_path_c_training_route": {
                        "status": "m04_path_c_split_training_route_available",
                        "kernel_surface_available": True,
                        "path_c_fusion": {
                            "mode": "auto",
                            "status": "plan_ready_not_default",
                            "schedule_name": "fused:descriptor",
                            "schedule_status": "ready",
                            "single_kernel_fused": False,
                            "default_allowed": False,
                            "runtime_training_binding": {
                                "status": "model_owned_physical_abi_banks_missing",
                                "runtime_uses_fused_train_block": False,
                                "required_bank_buffers": [
                                    "path_c_float32_abi_bank",
                                    "path_c_uint8_abi_bank",
                                    "path_c_int32_abi_bank",
                                ],
                                "missing_bank_buffers": [
                                    "path_c_float32_abi_bank",
                                    "path_c_uint8_abi_bank",
                                    "path_c_int32_abi_bank",
                                ],
                            },
                            "schedule_blockers": [
                                {"kind": "fused_train_block_runtime_not_bound"}
                            ],
                            "production_schedule": {
                                "schedule_id": "path_c_descriptor_chain_abc",
                                "implementation_kind": "production",
                                "production_fragments_complete": True,
                                "real_abi_contract_complete": True,
                                "missing_real_abi_inputs": [],
                            },
                            "schedule_contract": {
                                "status": "registered_not_lowered"
                            },
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = matrix.extract_result(
        cell=cell,
        identity={
            "cppmega_sha": "cpp",
            "tilelang_sha": "tl",
            "mlx_sha": "mlx",
            "mlx_version": "0.0+mlx",
        },
        cache_state={
            "cache_mode": "cold",
            "cache_dir": str(tmp_path / "cache"),
            "cache_files_before": 0,
        },
        process=subprocess.CompletedProcess(cell.command, 0, "", ""),
        duration_s=3.0,
    )

    assert result.status == "ok"
    assert result.compile_time_s == 2.0
    assert result.step_sec == 1.25
    assert result.tok_sec == 2048.0
    assert result.peak_memory_gb == 1.0
    assert result.active_memory_gb == 0.5
    assert result.cache_memory_gb == 0.75
    assert result.selected_schedule["kernel_counts"] == {"path_c_tilelang_dsl": 1}
    assert result.selected_schedule["path_c_fusion"]["production_schedule_id"] == (
        "path_c_descriptor_chain_abc"
    )
    assert result.proof_result["path_c_requested"] is True
    assert result.proof_result["runtime_uses_fused_train_block"] is False
    assert result.proof_result["fused_train_block_runtime_available"] is False
    assert result.proof_result["path_c_fusion"]["runtime_binding_status"] == (
        "model_owned_physical_abi_banks_missing"
    )
    assert result.proof_result["path_c_fusion"]["schedule_blockers"] == [
        "fused_train_block_runtime_not_bound"
    ]
    assert result.profiling_trace_path == (
        "reports/profiling/bf16_adamw_path_c_cold.gputrace"
    )
    assert result.profiling_trace_captured is True
    assert result.profiling_capture_receipt_path == (
        "reports/profiling/bf16_adamw_path_c_cold_capture.json"
    )
    assert result.profiling_capture_status == "ok"
    row = result.to_row()
    assert row["profiling_trace_path"] == (
        "reports/profiling/bf16_adamw_path_c_cold.gputrace"
    )
    assert row["profiling_trace_captured"] is True
    assert row["profiling_capture_receipt_path"] == (
        "reports/profiling/bf16_adamw_path_c_cold_capture.json"
    )
    assert row["profiling_capture_status"] == "ok"


def test_bench_1b_matrix_extracts_profile_capture_receipt_when_m04_has_no_profile(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        "--dtypes",
        "bf16",
        "--optimizers",
        "adamw",
        "--paths",
        "path_c_warm",
    )
    cell = matrix.build_cells(args)[0]
    cell.output_json.parent.mkdir(parents=True)
    cell.output_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "timing": {"step_times_s": [1.0], "tokens_per_second": 1024.0},
                "memory": {"peak_memory_bytes": 1 << 30},
                "training": {"all_finite": True, "kernel_dispatch": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "profiles" / "bf16_adamw_path_c_warm.gputrace"
    capture_receipt_path = (
        tmp_path / "profiles" / "bf16_adamw_path_c_warm_capture.json"
    )

    result = matrix.extract_result(
        cell=cell,
        identity={
            "cppmega_sha": "cpp",
            "tilelang_sha": "tl",
            "mlx_sha": "mlx",
            "mlx_version": "0.0+mlx",
        },
        cache_state={
            "cache_mode": "warm",
            "cache_dir": str(tmp_path / "cache"),
            "cache_files_before": 1,
        },
        process=subprocess.CompletedProcess(cell.command, 0, "", ""),
        duration_s=3.0,
        profile_trace_path=trace_path,
        profile_capture_receipt_path=capture_receipt_path,
        profile_capture_receipt={
            "kind": "cppmega_mlx_metal_capture_receipt",
            "status": "ok",
            "capture_started": True,
            "capture_stopped": True,
            "trace_path": str(trace_path),
        },
    )

    assert result.profiling_trace_path == str(trace_path)
    assert result.profiling_trace_captured is True
    assert result.profiling_capture_receipt_path == str(capture_receipt_path)
    assert result.profiling_capture_status == "ok"


def test_path_c_fusion_summary_preserves_direct_chain_runtime_route() -> None:
    receipt = {
        "training": {
            "fp8_path_c_training_route": {
                "fused_train_block_runtime_available": True,
                "path_c_fusion": {
                    "mode": "auto",
                    "status": "plan_ready_not_default",
                    "runtime_training_binding": {
                        "status": "model_owned_physical_abi_banks_missing",
                        "runtime_uses_fused_train_block": False,
                    },
                    "direct_chained_fusion": {
                        "status": "ready",
                        "segment_count": 4,
                        "runtime_binding": {
                            "status": "ok",
                            "runtime_uses_direct_fusion_chain": True,
                        },
                        "training_runtime_contract": {
                            "status": "ok",
                            "training_runtime_available": True,
                            "critical_path_ready": True,
                        },
                    },
                    "schedule_blockers": [],
                    "production_schedule": {
                        "schedule_id": "path_c_descriptor_chain_abc",
                        "implementation_kind": "production",
                        "production_fragments_complete": True,
                        "real_abi_contract_complete": True,
                        "missing_real_abi_inputs": [],
                    },
                    "schedule_contract": {"status": "verified"},
                },
            },
        },
    }

    summary = matrix.path_c_fusion_summary_from_receipt(receipt)
    proof = matrix.proof_result_from_receipt(receipt, path="path_c_warm")

    assert summary["runtime_uses_fused_train_block"] is True
    assert summary["runtime_uses_direct_fusion_chain"] is True
    assert summary["runtime_binding_status"] == "ok"
    assert summary["direct_chain_runtime_binding_status"] == "ok"
    assert summary["direct_chain_training_runtime_status"] == "ok"
    assert summary["direct_chain_training_runtime_available"] is True
    assert proof["runtime_uses_fused_train_block"] is True


def test_path_c_fusion_summary_does_not_promote_standalone_direct_chain() -> None:
    receipt = {
        "training": {
            "fp8_path_c_training_route": {
                "fused_train_block_runtime_available": False,
                "path_c_fusion": {
                    "runtime_training_binding": {
                        "status": "model_owned_physical_abi_banks_missing",
                        "runtime_uses_fused_train_block": False,
                    },
                    "direct_chained_fusion": {
                        "status": "ready",
                        "segment_count": 4,
                        "runtime_binding": {
                            "status": "ok",
                            "runtime_uses_direct_fusion_chain": True,
                        },
                        "training_runtime_contract": {
                            "status": "direct_fusion_chain_training_runtime_incomplete",
                            "training_runtime_available": False,
                            "critical_path_ready": False,
                        },
                    },
                    "production_schedule": {
                        "schedule_id": "path_c_descriptor_chain_abc",
                        "implementation_kind": "production",
                        "production_fragments_complete": True,
                        "real_abi_contract_complete": True,
                        "missing_real_abi_inputs": [],
                    },
                    "schedule_contract": {"status": "verified"},
                },
            },
        },
    }

    summary = matrix.path_c_fusion_summary_from_receipt(receipt)
    proof = matrix.proof_result_from_receipt(receipt, path="path_c_warm")

    assert summary["runtime_uses_direct_fusion_chain"] is True
    assert summary["runtime_uses_fused_train_block"] is False
    assert summary["runtime_binding_status"] == (
        "direct_fusion_chain_training_runtime_incomplete"
    )
    assert summary["direct_chain_training_runtime_available"] is False
    assert proof["runtime_uses_fused_train_block"] is False


def test_bench_1b_matrix_reuses_existing_ok_receipt(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    args = _args(
        tmp_path,
        "--reuse-existing-ok",
        "--dtypes",
        "bf16",
        "--optimizers",
        "adamw",
        "--paths",
        "path_c_warm",
    )
    cell = matrix.build_cells(args)[0]
    cell.output_json.parent.mkdir(parents=True)
    cell.output_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "timing": {
                    "step_times_s": [1.0, 0.5],
                    "tokens_per_second": 4096.0,
                },
                "memory": {"peak_memory_bytes": 2 << 30},
                "training": {"all_finite": True, "kernel_dispatch": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_subprocess_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("existing ok receipt should not launch subprocess")

    monkeypatch.setattr(matrix.subprocess, "run", fail_subprocess_run)
    result = matrix.run_cell(
        cell,
        args=args,
        identity={
            "cppmega_sha": "cpp",
            "tilelang_sha": "tl",
            "mlx_sha": "mlx",
            "mlx_version": "0.0+mlx",
        },
    )

    assert result.status == "ok"
    assert result.duration_s == 0.0
    assert result.tok_sec == 4096.0
    assert result.cache_state["reused_existing_receipt"] is True
