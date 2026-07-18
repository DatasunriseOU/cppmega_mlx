from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.quantize_for_inference", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def test_quantize_cli_ignores_dirty_tilelang_loader_environment(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DYLD_LIBRARY_PATH": "/Volumes/external/sources/tilelang/build/lib",
            "TVM_LIBRARY_PATH": "/Volumes/external/sources/tilelang/build/lib",
            "TILELANG_DEV_BUILD_ROOT": "/Volumes/external/sources/tilelang/build",
            # Keep the repository importable for ``python -m`` while still
            # placing a dirty TileLang path in front of the loader contract.
            "PYTHONPATH": os.pathsep.join(
                (str(REPO), "/Volumes/external/sources/tilelang")
            ),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.quantize_for_inference",
            "--json",
            "--out",
            str(tmp_path / "clean.json"),
            "--preset",
            "smoke_attention",
        ],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["hardware"]["mlx_version"] == "0.32.0"


def test_quantize_for_inference_writes_manifest_and_forward_check(tmp_path):
    out_path = tmp_path / "quantized_manifest.json"

    result = run_script(
        "--json",
        "--out",
        str(out_path),
        "--preset",
        "smoke_attention",
        "--bits",
        "4",
        "--group-size",
        "32",
        "--kv-bits",
        "4",
        "--kv-group-size",
        "32",
        "--dtype",
        "float32",
        "--check-forward",
        "--seed",
        "178",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == json.loads(out_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["receipt_scope"] == "local_inference_quantization_manifest"
    assert payload["local_only"] is True
    assert payload["training_quantization_claim"] is False
    assert payload["full_checkpoint_converter_claim"] is False
    assert payload["gb10_parity_claim"] is False

    quant = payload["linear_quantization"]
    assert quant["bits"] == 4
    assert quant["group_size"] == 32
    assert quant["quantized_linear_modules"] > 0
    assert quant["remaining_linear_modules"] >= 1
    assert quant["embed_lm_head_skipped"] is True

    kv = payload["kv_cache"]
    assert kv["bits"] == 4
    assert kv["group_size"] == 32
    assert kv["quantized"] is True

    forward = payload["forward_check"]
    assert forward["enabled"] is True
    assert forward["finite"] is True
    assert forward["max_abs_diff"] >= 0.0


def test_quantize_for_inference_rejects_invalid_quant_args(tmp_path):
    result = run_script(
        "--json",
        "--out",
        str(tmp_path / "blocked.json"),
        "--bits",
        "3",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "bits" in payload["error"]


def test_quantize_for_inference_help_lists_manifest_and_kv_flags():
    result = run_script("--help")

    assert result.returncode == 0
    assert "--preset" in result.stdout
    assert "--check-forward" in result.stdout
    assert "--kv-bits" in result.stdout
    assert "--kv-group-size" in result.stdout
