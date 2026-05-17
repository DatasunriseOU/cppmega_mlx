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

    assert len(cells) == 36
    by_case = {cell.case_id: cell for cell in cells}
    assert by_case["bf16_adamw_path_b"].env["CPPMEGA_KERNEL_PATH"] == "auto"
    assert by_case["bf16_adamw_path_c_cold"].env["CPPMEGA_KERNEL_PATH"] == "path_c"
    assert by_case["bf16_adamw_path_c_cold"].env["CPPMEGA_MAMBA3_PATH_C_BWD"] == "path_b"
    assert by_case["bf16_adamw_path_c_cold"].cache_mode == "cold"
    assert by_case["bf16_adamw_path_c_warm"].cache_mode == "warm"
    assert "--seq-len" in by_case["bf16_adamw_path_b"].command
    assert "2048" in by_case["bf16_adamw_path_b"].command
    assert by_case["int8_lion_path_c_warm"].cli_optimizer == "lion8bit"
    assert by_case["int8_muon_path_c_cold"].cli_optimizer == "int8"
    assert by_case["fp8_adamw_path_b"].supported is True
    assert by_case["fp8_adamw_path_b"].dtype_arg == "fp8_path_b"
    assert by_case["fp8_adamw_path_b"].env["CPPMEGA_KERNEL_PATH"] == "auto"
    assert by_case["fp8_adamw_path_b"].env["CPPMEGA_KERNEL_PATH__SPARSE_MLA"] == "path_b"
    assert by_case["fp8_adamw_path_c_warm"].env["CPPMEGA_SPARSE_MLA_FP8_ROUTE"] == "path_c"


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
    rows = list(csv.DictReader((tmp_path / "matrix.csv").open(encoding="utf-8")))
    assert len(rows) == 4
    statuses = {row["case_id"]: row["status"] for row in rows}
    assert statuses["bf16_adamw_path_b"] == "planned"
    assert statuses["fp8_adamw_path_b"] == "planned"
    payload = json.loads((tmp_path / "matrix.json").read_text(encoding="utf-8"))
    assert payload["scope"] == "cppmega_1b_path_matrix"
    assert payload["config"]["block_size"] == 2048


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
                        "status": "m04_path_c_training_route_available",
                        "kernel_surface_available": True,
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
    assert result.proof_result["path_c_requested"] is True


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
