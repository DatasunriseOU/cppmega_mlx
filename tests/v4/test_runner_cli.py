"""F-A.2 CLI tests — argparse + json output + exit codes.

Uses :func:`cppmega_v4.runner.cli.main` in-process for fast tests; a
separate subprocess test verifies the console-script entry point is
actually installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cppmega_v4.runner.cli import main


_DIM_ENV = {"B": 1, "S": 4, "H": 64, "nh": 2, "nkv": 1, "head_dim": 32,
            "num_experts": 8, "top_k": 2}


def _spec_dict() -> dict:
    return {
        "graph": {
            "nodes": [{"id": "a", "kind": "mlp"}, {"id": "b", "kind": "mlp"}],
            "edges": [{"src": "a", "dst": "b"}],
        },
        "dim_env": _DIM_ENV,
        "loss": {"kind": "cross_entropy", "head_outputs": ["b"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 3e-4,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    }


def _write_spec(tmp_path: Path) -> Path:
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_spec_dict()))
    return p


# ---------------------------------------------------------------------------
# Argparse + exit codes.
# ---------------------------------------------------------------------------


def test_cli_smoke_pipeline_exits_zero(tmp_path: Path, capsys):
    p = _write_spec(tmp_path)
    rc = main([str(p)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "[PASS] parse" in captured.out
    assert "overall: ok" in captured.out


def test_cli_json_output(tmp_path: Path, capsys):
    p = _write_spec(tmp_path)
    rc = main([str(p), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "ok"
    assert any(s["name"] == "parse" for s in payload["stages"])


def test_cli_stages_smoke_alias(tmp_path: Path, capsys):
    p = _write_spec(tmp_path)
    rc = main([str(p), "--stages", "smoke", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(payload["stages"]) == 8


def test_cli_stages_all_alias(tmp_path: Path, capsys):
    p = _write_spec(tmp_path)
    rc = main([str(p), "--stages", "all", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(payload["stages"]) == 11


def test_cli_stages_csv_subset(tmp_path: Path, capsys):
    p = _write_spec(tmp_path)
    rc = main([str(p), "--stages", "parse,verify_build_spec", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    names = [s["name"] for s in payload["stages"]]
    assert names == ["parse", "verify_build_spec"]


def test_cli_invalid_stage_exits_with_error(tmp_path: Path, capsys):
    p = _write_spec(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main([str(p), "--stages", "bogus_stage"])
    assert exc.value.code != 0


def test_cli_missing_spec_returns_2(tmp_path: Path, capsys):
    rc = main([str(tmp_path / "missing.json")])
    assert rc == 2


def test_cli_invalid_json_spec_returns_2(tmp_path: Path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc = main([str(bad)])
    assert rc == 2


def test_cli_invalid_spec_validation_returns_2(tmp_path: Path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"missing": "required fields"}))
    rc = main([str(bad)])
    assert rc == 2


def test_cli_failure_exits_one(tmp_path: Path, capsys):
    p = tmp_path / "fail.json"
    d = _spec_dict()
    d["graph"] = {"nodes": [{"id": "x", "kind": "not_a_real_brick"}], "edges": []}
    p.write_text(json.dumps(d))
    rc = main([str(p), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["overall_status"] == "fail"


def test_cli_pipeline_yaml_round_trip(tmp_path: Path, capsys):
    p = _write_spec(tmp_path)
    y = tmp_path / "pipe.yaml"
    y.write_text("stages:\n  - parse\n  - verify_build_spec\n")
    rc = main([str(p), "--pipeline", str(y), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [s["name"] for s in payload["stages"]] == ["parse", "verify_build_spec"]


# ---------------------------------------------------------------------------
# Console-script subprocess (system test).
# ---------------------------------------------------------------------------


def test_cppmega_run_executable_invokes(tmp_path: Path):
    """End-to-end: spawn the CLI module via -m and check exit code."""
    p = _write_spec(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "cppmega_v4.runner.cli", str(p), "--json"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["overall_status"] == "ok"
