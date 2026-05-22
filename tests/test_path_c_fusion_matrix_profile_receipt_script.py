from __future__ import annotations

import json
from pathlib import Path

import scripts.m04_train_step as m04_train_step
from scripts import path_c_fusion_matrix_profile_receipt


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "path_c_fusion_matrix_profile_receipt.py"


def _complete_receipt_rows(*, cppmega_sha: str = "abc123") -> list[dict[str, object]]:
    return [
        {
            "dtype_route": dtype_route,
            "optimizer": optimizer,
            "status": "ok",
            "cppmega_sha": cppmega_sha,
            "path_b_status": "ok",
            "path_b_tok_sec": 100.0,
            "path_b_peak_memory_gb": 10.0,
            "path_c_warm_status": "ok",
            "path_c_warm_tok_sec": 125.0,
            "path_c_peak_memory_gb": 9.5,
            "path_c_warm_cache_hit": True,
            "path_c_cold_cache_hit": False,
            "profiling_trace_path": (
                f"reports/profiling/{dtype_route}_{optimizer}_path_c.json"
            ),
        }
        for dtype_route in m04_train_step.MATRIX_DTYPE_ROUTES
        for optimizer in m04_train_step.MATRIX_OPTIMIZERS
    ]


def _path_matrix_rows_without_profiles(
    *,
    cppmega_sha: str = "abc123",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dtype in ("bf16", "fp8"):
        for optimizer in (*m04_train_step.MATRIX_OPTIMIZERS, "muon_int8"):
            for path, cache_hit, tok_sec, peak_memory_gb in (
                ("path_b", None, 100.0, 10.0),
                ("path_c_cold", False, 110.0, 9.75),
                ("path_c_warm", True, 125.0, 9.5),
            ):
                rows.append(
                    {
                        "dtype": dtype,
                        "optimizer": optimizer,
                        "path": path,
                        "status": "ok",
                        "pass_fail_reason": "ok",
                        "cppmega_sha": cppmega_sha,
                        "tok_sec": tok_sec,
                        "peak_memory_gb": peak_memory_gb,
                        "cache_hit": cache_hit,
                    }
                )
    return rows


def test_matrix_profile_receipt_script_writes_verified_receipt(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "matrix_report.json"
    output_path = tmp_path / "matrix_profile_receipt.json"
    report_path.write_text(
        json.dumps(
            {
                "software": {"cppmega_sha": "abc123"},
                "results": _complete_receipt_rows(cppmega_sha="abc123"),
            }
        ),
        encoding="utf-8",
    )

    exit_code = path_c_fusion_matrix_profile_receipt.main(
        [
            "--matrix-report",
            str(report_path),
            "--out",
            str(output_path),
            "--schedule-id",
            "path_c_descriptor_chain_test",
            "--schedule-name",
            "test_schedule",
        ]
    )

    assert exit_code == 0
    receipt = json.loads(output_path.read_text(encoding="utf-8"))
    assert receipt["kind"] == "cppmega_path_c_fusion_matrix_profile_receipt"
    assert receipt["status"] == "ok"
    assert receipt["single_cppmega_commit"] is True
    assert receipt["cppmega_sha"] == "abc123"
    assert receipt["full_1b_matrix_captured"] is True
    assert receipt["profiling_traces_captured"] is True
    row_count = (
        len(m04_train_step.MATRIX_DTYPE_ROUTES)
        * len(m04_train_step.MATRIX_OPTIMIZERS)
    )
    assert receipt["row_check_summary"] == {
        "total_rows": row_count,
        "row_status_ok": row_count,
        "path_b_baseline_clean": row_count,
        "path_c_default_gate_passed": row_count,
        "path_c_peak_memory_non_regression": row_count,
        "path_c_warm_cache_hit_observed": row_count,
        "path_c_cold_cache_miss_observed": row_count,
        "profiling_trace_captured": row_count,
    }
    assert receipt["failed_rows_by_check"] == {}
    assert receipt["failed_checks"] == []


def test_matrix_profile_receipt_script_preserves_mismatch_exit_code(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "matrix_report.json"
    output_path = tmp_path / "matrix_profile_receipt.json"
    report_path.write_text(
        json.dumps(
            {
                "software": {"cppmega_sha": "abc123"},
                "results": _path_matrix_rows_without_profiles(cppmega_sha="abc123"),
            }
        ),
        encoding="utf-8",
    )

    exit_code = path_c_fusion_matrix_profile_receipt.main(
        [
            "--matrix-report",
            str(report_path),
            "--out",
            str(output_path),
            "--schedule-id",
            "path_c_descriptor_chain_test",
            "--schedule-name",
            "test_schedule",
        ]
    )

    assert exit_code == 2
    receipt = json.loads(output_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "mismatch"
    assert receipt["full_1b_matrix_captured"] is True
    assert receipt["profiling_traces_captured"] is False
    row_count = (
        len(m04_train_step.MATRIX_DTYPE_ROUTES)
        * len(m04_train_step.MATRIX_OPTIMIZERS)
    )
    assert receipt["row_check_summary"]["total_rows"] == row_count
    assert receipt["row_check_summary"]["profiling_trace_captured"] == 0
    assert receipt["failed_rows_by_check"]["profiling_trace_captured"] == [
        f"{dtype_route}:{optimizer}"
        for dtype_route in m04_train_step.MATRIX_DTYPE_ROUTES
        for optimizer in m04_train_step.MATRIX_OPTIMIZERS
    ]
    assert "profiling_traces_captured" in receipt["failed_checks"]
