from __future__ import annotations

import json
from pathlib import Path

from scripts import path_sanity_guard


ROOT = Path(__file__).resolve().parents[1]
PATH_B_BLOCKER = (
    "m2rnn: Path B kernel unavailable "
    "(direct-MSL Path B is retired; use m2rnn_path_c.py for native TileLang/TVM-FFI)"
)
DEPRECATION_WARNING = (
    "mx.metal.set_memory_limit is deprecated and will be removed in a future version. "
    "Use mx.set_memory_limit instead."
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _path_b_row(
    raw_receipt: Path,
    *,
    status: str = "failed",
    reason: str | None = DEPRECATION_WARNING,
) -> dict[str, object]:
    return {
        "case_id": "bf16_adamw_path_b",
        "dtype": "bf16",
        "optimizer": "adamw",
        "path": "path_b",
        "status": status,
        "pass_fail_reason": reason,
        "receipt_path": str(raw_receipt),
    }


def test_matrix_guard_rejects_masked_path_b_blocker(tmp_path: Path) -> None:
    raw = _write_json(
        tmp_path / "raw" / "bf16_adamw_path_b.json",
        {
            "status": "blocked",
            "blockers": [{"type": "RuntimeError", "reason": PATH_B_BLOCKER}],
        },
    )
    report = _write_json(
        tmp_path / "matrix.json",
        {"results": [_path_b_row(raw, reason=DEPRECATION_WARNING)]},
    )

    findings = path_sanity_guard.check_matrix_report(report, repo_root=tmp_path)

    assert any(f.code == "matrix_report_reason_masked" for f in findings)
    assert any(PATH_B_BLOCKER in f.detail for f in findings)


def test_matrix_guard_accepts_failed_path_b_when_reason_preserves_blocker(
    tmp_path: Path,
) -> None:
    raw = _write_json(
        tmp_path / "raw" / "bf16_adamw_path_b.json",
        {
            "status": "blocked",
            "blockers": [{"type": "RuntimeError", "reason": PATH_B_BLOCKER}],
        },
    )
    report = _write_json(
        tmp_path / "matrix.json",
        {"results": [_path_b_row(raw, reason=PATH_B_BLOCKER)]},
    )

    findings = path_sanity_guard.check_matrix_report(report, repo_root=tmp_path)

    assert [f for f in findings if f.code == "matrix_report_reason_masked"] == []
    assert [f for f in findings if f.code == "retired_broken_baseline_marked_ok"] == []


def test_matrix_guard_rejects_blocked_raw_baseline_marked_ok(tmp_path: Path) -> None:
    raw = _write_json(
        tmp_path / "raw" / "bf16_adamw_path_b.json",
        {
            "status": "blocked",
            "blockers": [{"type": "RuntimeError", "reason": PATH_B_BLOCKER}],
        },
    )
    report = _write_json(
        tmp_path / "matrix.json",
        {"results": [_path_b_row(raw, status="ok", reason="ok")]},
    )

    findings = path_sanity_guard.check_matrix_report(report, repo_root=tmp_path)

    assert any(f.code == "retired_broken_baseline_marked_ok" for f in findings)


def test_matrix_guard_rejects_path_c_comparison_without_path_b_baseline(
    tmp_path: Path,
) -> None:
    path_b_raw = _write_json(
        tmp_path / "raw" / "bf16_adamw_path_b.json",
        {"status": "blocked", "blockers": [{"reason": PATH_B_BLOCKER}]},
    )
    path_c_raw = _write_json(
        tmp_path / "raw" / "bf16_adamw_path_c_warm.json",
        {"status": "ok"},
    )
    report = _write_json(
        tmp_path / "matrix.json",
        {
            "results": [
                _path_b_row(path_b_raw, reason=PATH_B_BLOCKER),
                {
                    "case_id": "bf16_adamw_path_c_warm",
                    "dtype": "bf16",
                    "optimizer": "adamw",
                    "path": "path_c_warm",
                    "status": "ok",
                    "pass_fail_reason": "ok",
                    "receipt_path": str(path_c_raw),
                },
            ]
        },
    )

    findings = path_sanity_guard.check_matrix_report(report, repo_root=tmp_path)

    assert any(f.code == "path_c_without_runnable_path_b_baseline" for f in findings)


def test_matrix_guard_rejects_fused_path_c_claim_without_stepper_runtime(
    tmp_path: Path,
) -> None:
    path_b_raw = _write_json(tmp_path / "raw" / "bf16_adamw_path_b.json", {"status": "ok"})
    path_c_raw = _write_json(
        tmp_path / "raw" / "bf16_adamw_path_c_warm.json",
        {
            "status": "ok",
            "training": {
                "fp8_path_c_training_route": {
                    "fused_train_block_training_critical_path": True,
                    "fused_train_block_training_runtime_contract": {
                        "training_critical_path_verified": True,
                    },
                },
                "stepper_state": {
                    "path_c_training_runtime_installed": False,
                },
            },
        },
    )
    report = _write_json(
        tmp_path / "matrix.json",
        {
            "results": [
                _path_b_row(path_b_raw, status="ok", reason="ok"),
                {
                    "case_id": "bf16_adamw_path_c_warm",
                    "dtype": "bf16",
                    "optimizer": "adamw",
                    "path": "path_c_warm",
                    "status": "ok",
                    "pass_fail_reason": "ok",
                    "receipt_path": str(path_c_raw),
                },
            ]
        },
    )

    findings = path_sanity_guard.check_matrix_report(report, repo_root=tmp_path)

    assert any(
        f.code == "path_c_fused_runtime_claimed_but_not_attached"
        for f in findings
    )


def test_v4_matrix_guard_rejects_test_only_path_d_promotion(tmp_path: Path) -> None:
    report = _write_json(
        tmp_path / "v4_gdn_matrix.json",
        {
            "shape": {
                "block": "gdn",
                "path": "path_a",
                "batch": 1,
                "seq_len": 4,
                "num_heads": 2,
                "head_dim_k": 32,
                "head_dim_v": 32,
            },
            "cells": [
                {
                    "block": "gdn",
                    "path": "path_d",
                    "median_seconds": 0.1,
                    "iters": 2,
                    "backend_available": True,
                    "backend_reason": "test fixture",
                    "output_finite": True,
                    "measured_path": "path_d",
                    "fallback_used": False,
                }
            ],
            "promotion": {
                "winning_path": "path_d",
                "promotion_applied": True,
            },
        },
    )

    findings = path_sanity_guard.check_matrix_report(report, repo_root=ROOT)

    assert any(f.code == "v4_blocked_path_marked_available" for f in findings)
    assert any(f.code == "v4_blocked_path_promoted" for f in findings)


def test_v4_matrix_guard_rejects_available_fallback_cell(tmp_path: Path) -> None:
    report = _write_json(
        tmp_path / "v4_kda_matrix.json",
        {
            "shape": {
                "block": "kda",
                "path": "path_a",
                "batch": 1,
                "seq_len": 4,
                "num_heads": 2,
                "head_dim_k": 32,
                "head_dim_v": 32,
                "num_v_heads": 2,
            },
            "cells": [
                {
                    "block": "kda",
                    "path": "path_d",
                    "median_seconds": 0.1,
                    "iters": 2,
                    "backend_available": True,
                    "backend_reason": "test fixture",
                    "output_finite": True,
                    "measured_path": "path_a",
                    "fallback_used": True,
                }
            ],
            "promotion": {
                "winning_path": "path_a",
                "promotion_applied": False,
            },
        },
    )

    findings = path_sanity_guard.check_matrix_report(report, repo_root=ROOT)

    assert any(f.code == "v4_available_path_measured_fallback" for f in findings)


def test_v4_matrix_guard_rejects_path_e_ops_fallback_shape(tmp_path: Path) -> None:
    report = _write_json(
        tmp_path / "v4_gdn_path_e_small_shape.json",
        {
            "shape": {
                "block": "gdn",
                "path": "path_a",
                "batch": 1,
                "seq_len": 4,
                "num_heads": 2,
                "head_dim_k": 4,
                "head_dim_v": 4,
            },
            "cells": [
                {
                    "block": "gdn",
                    "path": "path_e",
                    "median_seconds": 0.1,
                    "iters": 2,
                    "backend_available": True,
                    "backend_reason": "import-only status",
                    "output_finite": True,
                    "measured_path": "path_e",
                    "fallback_used": False,
                }
            ],
            "promotion": {
                "winning_path": "path_e",
                "promotion_applied": True,
            },
        },
    )

    findings = path_sanity_guard.check_matrix_report(report, repo_root=ROOT)

    assert any(f.code == "v4_path_e_ops_fallback_marked_available" for f in findings)


def test_declared_path_contracts_cover_m04_and_v4_paths() -> None:
    declared = path_sanity_guard.discover_declared_paths(ROOT)

    assert declared["m04.training_matrix"] == (
        "path_b",
        "path_c_cold",
        "path_c_warm",
    )
    assert set(declared["v4.gdn"]) == {"path_a", "path_b", "path_c", "path_d", "path_e"}
    assert set(declared["v4.kda"]) == {"path_a", "path_b", "path_c", "path_d", "path_e"}

    findings = path_sanity_guard.check_path_contracts(ROOT)

    assert [f for f in findings if f.code == "missing_path_contract"] == []
    assert [f for f in findings if f.code == "missing_status_declaration"] == []
    assert [f for f in findings if f.code == "missing_dispatch_declaration"] == []
    assert [f for f in findings if f.code == "unsafe_path_d_import_probe"] == []
    assert [f for f in findings if f.code == "unsafe_path_d_fla_import_probe"] == []
    assert [f for f in findings if f.code == "missing_path_e_adapter"] == []
    assert [f for f in findings if f.code == "path_e_status_missing_adapter_import"] == []


def test_path_d_default_status_subprocess_guard_fails_closed() -> None:
    findings = path_sanity_guard.check_path_d_default_status_no_unsafe_imports(ROOT)

    assert findings == []
