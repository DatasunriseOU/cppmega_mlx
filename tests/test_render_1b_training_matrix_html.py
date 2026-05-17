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
