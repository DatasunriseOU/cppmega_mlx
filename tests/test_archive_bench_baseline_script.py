from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "archive_bench_baseline.py"

SCRIPT_SPEC = importlib.util.spec_from_file_location("archive_bench_baseline", SCRIPT)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
archive_bench_baseline = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = archive_bench_baseline
SCRIPT_SPEC.loader.exec_module(archive_bench_baseline)


def run_archive(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def valid_local_row() -> dict[str, object]:
    return {
        "hardware": "M4 Max",
        "commit": "abc1234",
        "dtype": "bfloat16",
        "batch_size": 2,
        "seq_len": 64,
        "route": "mamba3",
        "model": "hybrid-m",
        "mode": "eager",
        "tokens_per_second": 123.5,
        "local_only": True,
        "gb10_parity_claim": False,
    }


def test_module_is_importable() -> None:
    assert archive_bench_baseline.build_parser() is not None


def test_archives_json_row_and_prints_compact_receipt(tmp_path: Path) -> None:
    input_path = tmp_path / "row.json"
    output_dir = tmp_path / "baselines"
    input_path.write_text(json.dumps(valid_local_row()), encoding="utf-8")

    result = run_archive(
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "ok"
    assert receipt["kind"] == "cppmega.mlx.benchmark_baseline_archive_receipt"
    assert receipt["archive_kind"] == "cppmega.mlx.benchmark_baseline"
    assert receipt["index_kind"] == "cppmega.mlx.benchmark_baseline_index"
    assert receipt["local_only"] is True
    assert receipt["gb10_parity_claim"] is False
    assert receipt["entry"]["tokens_per_second"] == 123.5
    assert "GB10 parity claims require" in receipt["parity_evidence_policy"]

    archive_path = Path(receipt["path"])
    index_path = Path(receipt["index_path"])
    assert archive_path.exists()
    assert index_path == output_dir / "index.json"
    archived = json.loads(archive_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert archived["row"] == valid_local_row()
    assert index["entries"] == [receipt["entry"]]


def test_stdin_input_archives_row(tmp_path: Path) -> None:
    result = run_archive(
        "--input",
        "-",
        "--output-dir",
        str(tmp_path / "baselines"),
        input_text=json.dumps(valid_local_row()),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert Path(receipt["path"]).exists()


def test_unsafe_gb10_parity_claim_fails_closed(tmp_path: Path) -> None:
    row = valid_local_row()
    row["local_only"] = False
    row["gb10_parity_claim"] = True
    input_path = tmp_path / "row.json"
    input_path.write_text(json.dumps(row), encoding="utf-8")

    result = run_archive(
        "--input",
        str(input_path),
        "--output-dir",
        str(tmp_path / "baselines"),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert "matched evidence" in payload["error"]
    assert "GB10 parity claims require" in payload["parity_evidence_policy"]
    assert not (tmp_path / "baselines").exists()


def test_non_object_input_fails_closed(tmp_path: Path) -> None:
    input_path = tmp_path / "row.json"
    input_path.write_text(json.dumps([valid_local_row()]), encoding="utf-8")

    result = run_archive(
        "--input",
        str(input_path),
        "--output-dir",
        str(tmp_path / "baselines"),
    )

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["error"] == "input JSON must be one benchmark row object"
