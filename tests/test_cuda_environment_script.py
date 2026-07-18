from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_cuda_environment.py"


def test_cuda_preflight_has_stable_json_contract() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    payload = json.loads(result.stdout)
    assert payload["kind"] == "cppmega_mlx_cuda_environment"
    assert payload["schema_version"] == 1
    assert set(payload["versions"]) == {"torch", "triton", "tilelang"}
    assert isinstance(payload["errors"], list)
    assert result.returncode == (0 if payload["ok"] else 1)


def test_cuda_preflight_script_compiles() -> None:
    assert importlib.util.spec_from_file_location("cuda_preflight", SCRIPT)
