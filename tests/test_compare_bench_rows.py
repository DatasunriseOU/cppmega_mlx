from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_bench_rows.py"
PYTHON = ROOT / ".venv" / "bin" / "python"


def tiny_route_plan() -> dict[str, Any]:
    return {
        "model_route": "tiny",
        "route_symbols": "tiny",
        "route_roles": ["attention", "ffn"],
        "pattern": "tiny",
    }


def mlx_tiny_backend_plan(*, n_layers: int = 1) -> dict[str, Any]:
    return {
        "execution_backend": "mlx",
        "layer_backends": ["mlx.nn.MultiHeadAttention"] * n_layers,
        "backend_summary": {"mlx.nn.MultiHeadAttention": n_layers},
        "attention_modes": [],
        "attention_backends": [],
        "attention_sparse_reference": [],
    }


def hybrid_backend_plan(backend: str) -> dict[str, Any]:
    return {
        "execution_backend": "mlx",
        "layer_backends": [backend],
        "backend_summary": {backend: 1},
        "attention_modes": [],
        "attention_backends": [],
        "attention_sparse_reference": [],
    }


def torch_backend_plan(backend: str) -> dict[str, Any]:
    return {
        "execution_backend": backend,
        "layer_backends": [backend],
        "backend_summary": {backend: 1},
        "attention_modes": [],
        "attention_backends": [],
        "attention_sparse_reference": [],
    }


def run_compare(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_benchmark_package_modules_are_importable() -> None:
    assert importlib.import_module("scripts.bench_matrix") is not None
    assert importlib.import_module("scripts.compare_bench_rows") is not None


def workload_key(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("config", {})
    include_structure = row.get(
        "include_structure",
        config.get("include_structure", False),
    )
    return {
        "profile": row.get("profile", config.get("profile")),
        "route": row.get("route", config.get("route")),
        "dtype": row.get("dtype", config.get("dtype")),
        "batch_size": row.get("batch_size", config.get("batch_size")),
        "seq_len": row.get("seq_len", config.get("seq_len")),
        "vocab_size": row.get("vocab_size", config.get("vocab_size")),
        "d_model": row.get("d_model", config.get("d_model")),
        "n_heads": row.get("n_heads", config.get("n_heads")),
        "n_layers": row.get("n_layers", config.get("n_layers")),
        "mlp_dim": row.get("mlp_dim", config.get("mlp_dim")),
        "warmup_steps": row.get("warmup_steps", config.get("warmup_steps")),
        "measured_steps": row.get("measured_steps", config.get("steps")),
        "compile": row.get("compile", config.get("compile")),
        "include_structure": include_structure,
        "learning_rate": row.get("learning_rate", config.get("learning_rate")),
        "seed": row.get("seed", config.get("seed")),
        "model_source": row.get("model_source", config.get("model_source")),
        "model_route": row.get("model_route", config.get("model_route")),
        "route_plan": row.get("route_plan", config.get("route_plan")),
        "backend_plan": row.get("backend_plan", config.get("backend_plan")),
        "backend": row.get("backend", "mlx"),
        "data_contract": row.get(
            "data_contract",
            "synthetic_structure" if include_structure else "synthetic_plain",
        ),
    }


def mlx_software_key(row: dict[str, Any]) -> dict[str, Any]:
    device = row.get("device", {})
    return {
        "framework": "mlx",
        "backend": row.get("backend", "metal"),
        "python_version": device.get("python"),
        "mlx_version": device.get("mlx"),
        "mlx_lm_version": device.get("mlx_lm"),
        "mlx_metal_version": device.get("mlx_metal"),
        "device_name": row.get("device_name", "Apple M4 Max"),
    }


def add_mlx_comparison_key(row: dict[str, Any]) -> dict[str, Any]:
    row["comparison_key"] = {
        "schema_version": 1,
        "workload": workload_key(row),
        "software": mlx_software_key(row),
    }
    return row


def torch_workload_key(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": row.get("profile"),
        "route": row.get("route"),
        "dtype": row.get("dtype"),
        "batch_size": row.get("batch_size"),
        "seq_len": row.get("seq_len"),
        "vocab_size": row.get("vocab_size"),
        "d_model": row.get("d_model"),
        "n_heads": row.get("n_heads"),
        "n_layers": row.get("n_layers"),
        "mlp_dim": row.get("mlp_dim"),
        "warmup_steps": row.get("warmup_steps"),
        "measured_steps": row.get("measured_steps"),
        "compile": row.get("compile"),
        "include_structure": row.get("include_structure"),
        "learning_rate": row.get("learning_rate"),
        "seed": row.get("seed"),
        "model_source": row.get("model_source"),
        "model_route": row.get("model_route"),
        "route_plan": row.get("route_plan"),
        "backend_plan": row.get("backend_plan"),
        "backend": row.get("backend"),
        "data_contract": row.get("data_contract"),
    }


def torch_software_key(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "framework": row.get("framework"),
        "backend": row.get("backend"),
        "torch_version": row.get("torch"),
        "cuda_version": row.get("cuda"),
        "driver_version": row.get("driver_version"),
        "device_name": row.get("device"),
        "device_capability": row.get("capability"),
    }


def add_torch_comparison_key(row: dict[str, Any]) -> dict[str, Any]:
    row["comparison_key"] = {
        "schema_version": 1,
        "workload": torch_workload_key(row),
        "software": torch_software_key(row),
    }
    return row


def unreceipted_bench_row(hardware_label: str, **kwargs: Any) -> dict[str, Any]:
    row = bench_row(hardware_label, **kwargs)
    row.pop("comparison_key", None)
    return row


def bench_row(
    hardware_label: str,
    *,
    tokens_per_second: float = 100.0,
    peak_memory_bytes: int = 1_000,
    seq_len: int = 8,
    mlx_version: str = "0.31.0",
) -> dict[str, Any]:
    row = {
        "case_id": f"smoke-route_plain-b1-s{seq_len}-bfloat16-eager-{hardware_label}",
        "profile": "smoke",
        "route": "plain",
        "hardware_label": hardware_label,
        "dtype": "bfloat16",
        "batch_size": 1,
        "seq_len": seq_len,
        "warmup_steps": 1,
        "measured_steps": 3,
        "compile": False,
        "include_structure": False,
        "tokens_per_second": tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
        "compile_time_s": 0.0,
        "mean_step_time_s": 0.01,
        "model_source": "cppmega_mlx.models.tiny_lm",
        "model_route": "tiny",
        "route_plan": tiny_route_plan(),
        "backend_plan": mlx_tiny_backend_plan(),
        "config": {
            "hardware_label": hardware_label,
            "batch_size": 1,
            "seq_len": seq_len,
            "vocab_size": 32,
            "d_model": 8,
            "n_heads": 1,
            "n_layers": 1,
            "mlp_dim": 16,
            "dtype": "bfloat16",
            "learning_rate": 0.001,
            "warmup_steps": 1,
            "steps": 3,
            "seed": 0,
            "compile": False,
            "include_structure": False,
            "model_route": "tiny",
            "route_plan": tiny_route_plan(),
            "backend_plan": mlx_tiny_backend_plan(),
        },
        "device": {
            "python": "3.13.0",
            "mlx": mlx_version,
            "mlx_lm": "0.31.0",
            "mlx_metal": "0.31.0",
        },
    }
    return add_mlx_comparison_key(row)


def aliased_hybrid_row(
    hardware_label: str,
    *,
    route: str,
    model_route: str,
    backend: str,
    symbols: str,
    tokens_per_second: float = 100.0,
) -> dict[str, Any]:
    row = bench_row(hardware_label, tokens_per_second=tokens_per_second)
    route_plan = {
        "model_route": model_route,
        "route_symbols": symbols,
        "route_roles": [backend],
        "pattern": symbols,
    }
    backend_plan = hybrid_backend_plan(backend)
    row.update(
        {
            "case_id": f"hybrid-smoke-route_{route}-b1-s8-bfloat16-eager-{hardware_label}",
            "profile": "hybrid-smoke",
            "route": route,
            "include_structure": True,
            "model_source": "cppmega_mlx.models.hybrid_lm",
            "model_route": model_route,
            "route_plan": route_plan,
            "backend_plan": backend_plan,
            "backend": "mlx",
        }
    )
    row["config"].update(
        {
            "include_structure": True,
            "model_route": model_route,
            "route_plan": route_plan,
            "backend_plan": backend_plan,
        }
    )
    return add_mlx_comparison_key(row)


def torch_cuda_row(
    hardware_label: str,
    *,
    tokens_per_sec: float = 100.0,
    max_alloc_gib: float = 1.0,
    backend: str = "torch_sdpa",
    torch_version: str = "2.12.0.dev20260430+cu132",
    cuda_version: str = "13.2",
    driver_version: str = "590.44",
    device: str = "NVIDIA GB10 DGX Spark",
    capability: tuple[int, int] = (12, 1),
) -> dict[str, Any]:
    row = {
        "name": backend,
        "case_id": f"smoke-route_plain-b1-s8-bfloat16-eager-{hardware_label}-{backend}",
        "profile": "smoke",
        "route": "plain",
        "hardware_label": hardware_label,
        "framework": "torch",
        "backend": backend,
        "dtype": "bfloat16",
        "batch_size": 1,
        "seq_len": 8,
        "vocab_size": 32,
        "d_model": 8,
        "n_heads": 1,
        "n_layers": 1,
        "mlp_dim": 16,
        "warmup_steps": 1,
        "measured_steps": 3,
        "compile": False,
        "include_structure": False,
        "learning_rate": 0.001,
        "seed": 0,
        "model_source": "cppmega_mlx.models.tiny_lm",
        "model_route": "tiny",
        "route_plan": tiny_route_plan(),
        "backend_plan": torch_backend_plan(backend),
        "data_contract": "synthetic_tokens",
        "tokens_per_sec": tokens_per_sec,
        "max_alloc_gib": max_alloc_gib,
        "mean_step_time_s": 0.01,
        "torch": torch_version,
        "cuda": cuda_version,
        "driver_version": driver_version,
        "device": device,
        "capability": capability,
    }
    return add_torch_comparison_key(row)


def receipt_only_row(
    hardware_label: str,
    *,
    tokens_per_second: float = 100.0,
    peak_memory_bytes: int = 1_000,
    route: str = "mamba3",
    model_route: str = "hybrid-m",
    backend: str = "mamba3",
    data_contract: str = "synthetic_tokens",
    mlx_version: str = "0.31.0",
) -> dict[str, Any]:
    route_plan = {
        "model_route": model_route,
        "route_symbols": "M" if backend == "mamba3" else "R",
        "route_roles": [backend],
        "pattern": "M" if backend == "mamba3" else "R",
    }
    workload = {
        "profile": "hybrid-smoke",
        "route": route,
        "dtype": "float32",
        "batch_size": 1,
        "seq_len": 8,
        "vocab_size": 32,
        "d_model": 8,
        "n_heads": 1,
        "n_layers": 4,
        "mlp_dim": 16,
        "warmup_steps": 0,
        "measured_steps": 1,
        "compile": False,
        "include_structure": True,
        "learning_rate": 0.001,
        "seed": 0,
        "model_source": "cppmega_mlx.models.hybrid_lm",
        "model_route": model_route,
        "route_plan": route_plan,
        "backend_plan": hybrid_backend_plan(backend),
        "backend": "mlx",
        "data_contract": data_contract,
    }
    software = {
        "framework": "mlx",
        "backend": "mlx",
        "execution_backend": "mlx",
        "framework_backend": "metal",
        "python_version": "3.13.0",
        "platform": "macOS-15-arm64-arm-64bit",
        "machine": "arm64",
        "mlx_version": mlx_version,
        "mlx_lm_version": "0.31.0",
        "mlx_metal_version": "0.31.0",
        "default_device": "Device(gpu, 0)",
        "device_name": "Apple M4 Max",
        "metal": {"available": True, "capture_supported": True},
    }
    return {
        "hardware_label": hardware_label,
        "tokens_per_second": tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
        "bench_receipt": {
            "schema_version": 1,
            "hardware_label": hardware_label,
            "timing": {
                "warmup_steps": 0,
                "measured_steps": 1,
                "compile": False,
                "tokens_per_second": tokens_per_second,
                "mean_step_time_s": 0.002,
                "wall_time_s": 0.002,
                "tokens_per_second_or_step_time": True,
                "synchronized_timing": True,
            },
            "workload": workload,
            "software": software,
            "comparison_key": {
                "schema_version": 1,
                "workload": workload,
                "software": software,
            },
        },
    }


def bench_tiny_json_row(
    hardware_label: str,
    *,
    tokens_per_second: float = 100.0,
    peak_memory_bytes: int = 1_000,
) -> dict[str, Any]:
    workload = {
        "dtype": "float32",
        "batch_size": 1,
        "seq_len": 4,
        "vocab_size": 32,
        "d_model": 8,
        "n_heads": 1,
        "n_layers": 1,
        "mlp_dim": 16,
        "warmup_steps": 0,
        "measured_steps": 1,
        "compile": False,
        "include_structure": False,
        "learning_rate": 0.001,
        "model_source": "cppmega_mlx.models.tiny_lm",
        "model_route": "tiny",
        "route_plan": tiny_route_plan(),
        "backend_plan": mlx_tiny_backend_plan(),
        "backend": "mlx",
        "data_contract": "synthetic_tokens",
    }
    software = {
        "framework": "mlx",
        "backend": "mlx",
        "execution_backend": "mlx",
        "framework_backend": "metal",
        "python_version": "3.13.0",
        "platform": "macOS-15-arm64-arm-64bit",
        "machine": "arm64",
        "mlx_version": "0.31.0",
        "mlx_lm_version": "0.31.0",
        "mlx_metal_version": "0.31.0",
        "default_device": "Device(gpu, 0)",
        "device_name": "Apple M4 Max",
        "metal": {"available": True, "capture_supported": True},
    }
    comparison_key = {
        "schema_version": 1,
        "workload": workload,
        "software": software,
    }
    return {
        "status": "ok",
        "receipt_schema_version": 1,
        "receipt_scope": "local_only",
        "local_only": True,
        "gb10_parity_claim": False,
        "hardware_label": hardware_label,
        "dtype": "float32",
        "batch_size": 1,
        "seq_len": 4,
        "warmup_steps": 0,
        "measured_steps": 1,
        "compile": False,
        "include_structure": False,
        "tokens_per_second": tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
        "mean_step_time_s": 0.002,
        "model_source": "cppmega_mlx.models.tiny_lm",
        "model_route": "tiny",
        "route_plan": tiny_route_plan(),
        "backend_plan": mlx_tiny_backend_plan(),
        "backend": "mlx",
        "config": {
            "hardware_label": hardware_label,
            "batch_size": 1,
            "seq_len": 4,
            "vocab_size": 32,
            "d_model": 8,
            "n_heads": 1,
            "n_layers": 1,
            "mlp_dim": 16,
            "dtype": "float32",
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "steps": 1,
            "seed": 0,
            "compile": False,
            "include_structure": False,
            "model_route": "tiny",
        },
        "device": {
            "python": "3.13.0",
            "platform": "macOS-15-arm64-arm-64bit",
            "machine": "arm64",
            "mlx": "0.31.0",
            "mlx_lm": "0.31.0",
            "mlx_metal": "0.31.0",
            "default_device": "Device(gpu, 0)",
            "mlx_device_info": {"device_name": "Apple M4 Max"},
            "metal": {"available": True, "capture_supported": True},
        },
        "run_metadata": {
            "schema_version": 1,
            "workload": {**workload, "hardware_label": hardware_label},
            "framework": {
                "python": "3.13.0",
                "platform": "macOS-15-arm64-arm-64bit",
                "machine": "arm64",
                "mlx": "0.31.0",
                "mlx_lm": "0.31.0",
                "mlx_metal": "0.31.0",
                "default_device": "Device(gpu, 0)",
                "mlx_device_info": {"device_name": "Apple M4 Max"},
                "metal": {"available": True, "capture_supported": True},
            },
            "matched_run": {
                "key": workload,
                "receipt_scope": "local_only",
                "local_only": True,
                "gb10_parity_claim": False,
            },
        },
        "matched_run": {
            "key": workload,
            "receipt_scope": "local_only",
            "local_only": True,
            "gb10_parity_claim": False,
        },
        "matched_run_key": workload,
        "workload_key": workload,
        "software_key": software,
        "comparison_key": comparison_key,
        "bench_receipt": {
            "schema_version": 1,
            "receipt_scope": "local_only",
            "local_only": True,
            "gb10_parity_claim": False,
            "hardware_label": hardware_label,
            "software": software,
            "workload": workload,
            "comparison_key": comparison_key,
            "timing": {
                "warmup_steps": 0,
                "measured_steps": 1,
                "compile": False,
                "tokens_per_second": tokens_per_second,
                "mean_step_time_s": 0.002,
                "wall_time_s": 0.002,
                "tokens_per_second_or_step_time": True,
                "synchronized_timing": True,
            },
        },
    }


def test_json_summary_reports_ratios_for_matched_rows(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "cases": [
                    bench_row("M4 Max", tokens_per_second=120.0, peak_memory_bytes=900),
                    bench_row("GB10", tokens_per_second=100.0, peak_memory_bytes=1_800),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["conditional_required_match_fields"] == {
        "mlx": ["mlx_version", "mlx_metal_version"],
        "torch": ["torch_version", "device_name"],
        "cuda": ["cuda_version", "driver_version", "device_capability"],
    }
    assert payload["matched_comparison_count"] == 1
    comparison = payload["comparisons"][0]
    assert comparison["status"] == "matched"
    assert comparison["match"]["seq_len"] == 8
    assert comparison["match"]["backend"] == "metal"
    assert comparison["match"]["mlx_version"] == "0.31.0"
    assert comparison["ratios"]["m4_tokens_per_second_over_gb10"] == 1.2
    assert comparison["ratios"]["gb10_peak_memory_bytes_over_m4"] == 2.0


def test_jsonl_input_outputs_comparison_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.ndjson"
    path.write_text(
        "\n".join(
            [
                json.dumps(bench_row("m4 max local", tokens_per_second=40.0)),
                json.dumps(bench_row("gb10 local", tokens_per_second=20.0)),
            ]
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path), "--jsonl")

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "matched"
    assert rows[0]["ratios"]["m4_tokens_per_second_over_gb10"] == 2.0


def test_bench_tiny_json_shape_matches_when_explicit_keys_agree(tmp_path: Path) -> None:
    path = tmp_path / "bench-tiny-shaped.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    bench_tiny_json_row("M4 Max", tokens_per_second=48.0),
                    bench_tiny_json_row("GB10", tokens_per_second=24.0),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["matched_comparison_count"] == 1
    comparison = payload["comparisons"][0]
    assert comparison["status"] == "matched"
    assert comparison["m4"]["matched_comparison_key_sources"] == [
        "comparison_key",
        "bench_receipt.comparison_key",
        "workload_key+software_key",
    ]
    assert comparison["gb10"]["matched_comparison_key_sources"] == [
        "comparison_key",
        "bench_receipt.comparison_key",
        "workload_key+software_key",
    ]
    assert comparison["ratios"]["m4_tokens_per_second_over_gb10"] == 2.0


def test_package_dir_writes_matched_report_manifest_and_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "matched.json"
    package_dir = tmp_path / "matched_package"
    input_path.write_text(
        json.dumps(
            {
                "cases": [
                    bench_row("M4 Max", tokens_per_second=120.0),
                    bench_row("GB10", tokens_per_second=100.0),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare(
        "--input",
        str(input_path),
        "--package-dir",
        str(package_dir),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["compare_package"]["matched_comparison_count"] == 1
    assert payload["compare_package"]["unmatched_pair_count"] == 0
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "cppmega.mlx.gb10_matched_compare_package"
    assert manifest["status"] == "ok"
    assert manifest["artifacts"] == {
        "compare_report": "compare_report.json",
        "matched_comparisons": "matched_comparisons.jsonl",
        "refused_pairs": "refused_pairs.jsonl",
        "manifest": "manifest.json",
    }
    assert manifest["parity_claim_refused"] is False
    assert "Only matched_comparisons.jsonl may contain ratios" in manifest["artifact_policy"]
    report = json.loads((package_dir / "compare_report.json").read_text(encoding="utf-8"))
    assert report["matched_comparison_count"] == 1
    assert report["ratio_artifact"] == "matched_comparisons.jsonl"
    assert "intentionally omits ratios" in report["artifact_policy"]
    assert "ratios" not in report["comparisons"][0]
    matched_rows = [
        json.loads(line)
        for line in (package_dir / "matched_comparisons.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    refused_text = (package_dir / "refused_pairs.jsonl").read_text(encoding="utf-8")
    assert len(matched_rows) == 1
    assert matched_rows[0]["status"] == "matched"
    assert matched_rows[0]["ratios"]["m4_tokens_per_second_over_gb10"] == 1.2
    assert refused_text == ""


def test_package_dir_writes_refusal_artifact_without_ratios(tmp_path: Path) -> None:
    input_path = tmp_path / "refused.json"
    package_dir = tmp_path / "refused_package"
    input_path.write_text(
        json.dumps(
            {
                "cases": [
                    bench_row("M4 Max", mlx_version="0.31.0"),
                    bench_row("GB10", mlx_version="0.32.0"),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare(
        "--input",
        str(input_path),
        "--package-dir",
        str(package_dir),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["compare_package"]["parity_claim_refused"] is True
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "no_matching_rows"
    assert manifest["matched_comparison_count"] == 0
    assert manifest["unmatched_pair_count"] == 1
    assert manifest["parity_claim_refused"] is True
    assert "identical selected comparison_key.workload" in manifest[
        "workload_software_key_guard"
    ]
    report = json.loads((package_dir / "compare_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "no_matching_rows"
    assert report["matched_comparison_count"] == 0
    assert report["unmatched_pair_count"] == 1
    assert report["parity_claim_refused"] is True
    assert report["comparisons"] == []
    assert "ratios" not in report["unmatched_pairs"][0]
    matched_text = (package_dir / "matched_comparisons.jsonl").read_text(
        encoding="utf-8"
    )
    refused_rows = [
        json.loads(line)
        for line in (package_dir / "refused_pairs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert matched_text == ""
    assert len(refused_rows) == 1
    assert refused_rows[0]["status"] == "unmatched"
    assert refused_rows[0]["parity_claim_refused"] is True
    assert "ratios" not in refused_rows[0]


def test_single_machine_status_is_insufficient_matched_rows(tmp_path: Path) -> None:
    path = tmp_path / "m4.json"
    path.write_text(json.dumps({"cases": [bench_row("M4 Max")]}), encoding="utf-8")

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "insufficient_matched_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 0, "ignored": 0}
    assert payload["parity_claim_refused"] is True
    assert payload["unmatched_pair_count"] == 0
    assert payload["unmatched_pairs"] == []
    assert payload["comparisons"] == []


def test_package_dir_refuses_m4_only_missing_gb10_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "m4-only.json"
    package_dir = tmp_path / "m4_only_package"
    input_path.write_text(
        json.dumps({"cases": [bench_row("M4 Max", tokens_per_second=120.0)]}),
        encoding="utf-8",
    )

    result = run_compare(
        "--input",
        str(input_path),
        "--package-dir",
        str(package_dir),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "insufficient_matched_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 0, "ignored": 0}
    assert payload["parity_claim_refused"] is True
    assert payload["matched_comparison_count"] == 0
    assert payload["unmatched_pair_count"] == 0
    assert payload["comparisons"] == []
    assert payload["unmatched_pairs"] == []

    assert sorted(path.name for path in package_dir.iterdir()) == [
        "compare_report.json",
        "manifest.json",
        "matched_comparisons.jsonl",
        "refused_pairs.jsonl",
    ]
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "insufficient_matched_rows"
    assert manifest["host_counts"]["m4"] == 1
    assert manifest["host_counts"]["gb10"] == 0
    assert manifest["matched_comparison_count"] == 0
    assert manifest["unmatched_pair_count"] == 0
    assert manifest["parity_claim_refused"] is True
    assert "must not be used for M4-vs-GB10" in manifest["artifact_policy"]

    report = json.loads((package_dir / "compare_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "insufficient_matched_rows"
    assert report["host_counts"]["gb10"] == 0
    assert report["parity_claim_refused"] is True
    assert report["comparisons"] == []
    assert report["unmatched_pairs"] == []
    assert report["ratio_artifact"] == "matched_comparisons.jsonl"
    assert (package_dir / "matched_comparisons.jsonl").read_text(encoding="utf-8") == ""
    assert (package_dir / "refused_pairs.jsonl").read_text(encoding="utf-8") == ""


def test_mismatched_framework_versions_do_not_report_ratios(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    bench_row("M4 Max", mlx_version="0.31.0"),
                    bench_row("GB10", mlx_version="0.32.0"),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    assert payload["unmatched_pair_count"] == 1
    refusal = payload["unmatched_pairs"][0]
    assert refusal["status"] == "unmatched"
    assert refusal["parity_claim_refused"] is True
    assert refusal["reason"] == "matched_comparison_key_mismatch"
    assert "ratios" not in refusal
    assert refusal["missing_required_fields"] == {"m4": [], "gb10": []}
    assert {field["field"] for field in refusal["mismatched_fields"]} == {"mlx_version"}
    assert payload["comparisons"] == []


def test_explicit_workload_key_mismatch_blocks_normalized_field_match(tmp_path: Path) -> None:
    m4 = bench_row("M4 Max", tokens_per_second=120.0)
    gb10 = bench_row("GB10", tokens_per_second=100.0)
    gb10["comparison_key"]["workload"] = {
        **gb10["comparison_key"]["workload"],
        "seq_len": 16,
    }
    path = tmp_path / "explicit-workload-key-mismatch.json"
    path.write_text(json.dumps({"cases": [m4, gb10]}), encoding="utf-8")

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    refusal = payload["unmatched_pairs"][0]
    assert refusal["reason"] == "matched_comparison_key_mismatch"
    assert refusal["mismatched_fields"] == []
    assert [field["field"] for field in refusal["mismatched_comparison_keys"]] == [
        "comparison_key.workload"
    ]
    assert refusal["m4"]["matched_comparison_key_source"] == "comparison_key"
    assert refusal["gb10"]["matched_comparison_key_source"] == "comparison_key"
    assert "ratios" not in refusal
    assert payload["comparisons"] == []


def test_explicit_software_key_mismatch_blocks_normalized_field_match(tmp_path: Path) -> None:
    m4 = bench_row("M4 Max", tokens_per_second=120.0)
    gb10 = bench_row("GB10", tokens_per_second=100.0)
    gb10["comparison_key"]["software"] = {
        **gb10["comparison_key"]["software"],
        "extra_runtime_flag": "gb10-only",
    }
    path = tmp_path / "explicit-software-key-mismatch.json"
    path.write_text(json.dumps({"cases": [m4, gb10]}), encoding="utf-8")

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    refusal = payload["unmatched_pairs"][0]
    assert refusal["reason"] == "matched_comparison_key_mismatch"
    assert refusal["mismatched_fields"] == []
    assert [field["field"] for field in refusal["mismatched_comparison_keys"]] == [
        "comparison_key.software"
    ]
    assert "ratios" not in refusal
    assert payload["comparisons"] == []


def test_conflicting_key_sources_inside_row_refuse_ratios(tmp_path: Path) -> None:
    m4 = bench_row("M4 Max", tokens_per_second=120.0)
    gb10 = bench_row("GB10", tokens_per_second=100.0)
    for row in (m4, gb10):
        row["workload_key"] = row["comparison_key"]["workload"]
        row["software_key"] = row["comparison_key"]["software"]
        row["bench_receipt"] = {
            "comparison_key": {
                "schema_version": 1,
                "workload": row["comparison_key"]["workload"],
                "software": row["comparison_key"]["software"],
            }
        }
    gb10["bench_receipt"]["comparison_key"]["software"] = {
        **gb10["comparison_key"]["software"],
        "extra_runtime_flag": "conflicting-receipt-only",
    }
    path = tmp_path / "conflicting-key-sources.json"
    path.write_text(json.dumps({"cases": [m4, gb10]}), encoding="utf-8")

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    assert payload["missing_matched_comparison_key_counts"] == {"m4": 0, "gb10": 1}
    assert payload["matched_comparison_key_conflict_counts"] == {"m4": 0, "gb10": 1}
    refusal = payload["unmatched_pairs"][0]
    assert refusal["reason"] == "missing_matched_comparison_key"
    assert refusal["missing_matched_comparison_key"] == {"m4": False, "gb10": True}
    assert refusal["mismatched_fields"] == []
    assert refusal["mismatched_comparison_keys"] == [
        {
            "field": "comparison_key.workload",
            "m4": m4["comparison_key"]["workload"],
            "gb10": "<missing>",
        },
        {
            "field": "comparison_key.software",
            "m4": m4["comparison_key"]["software"],
            "gb10": "<missing>",
        },
    ]
    conflicts = refusal["matched_comparison_key_conflicts"]["gb10"]
    assert len(conflicts) == 1
    assert conflicts[0]["section"] == "software"
    assert conflicts[0]["left_source"] == "comparison_key"
    assert conflicts[0]["right_source"] == "bench_receipt.comparison_key"
    assert conflicts[0]["right"]["extra_runtime_flag"] == "conflicting-receipt-only"
    assert refusal["gb10"]["matched_comparison_key_source"] is None
    assert refusal["gb10"]["matched_comparison_key_conflicts"] == conflicts
    assert "ratios" not in refusal
    assert payload["comparisons"] == []


def test_unreceipted_m4_gb10_rows_refuse_ratios_even_when_fields_match(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unreceipted-matched-fields.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    unreceipted_bench_row(
                        "M4 Max",
                        tokens_per_second=120.0,
                        peak_memory_bytes=900,
                    ),
                    unreceipted_bench_row(
                        "GB10",
                        tokens_per_second=100.0,
                        peak_memory_bytes=1_800,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["incomplete_match_metadata_counts"] == {"m4": 0, "gb10": 0}
    assert payload["missing_matched_comparison_key_counts"] == {"m4": 1, "gb10": 1}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    assert payload["comparisons"] == []
    refusal = payload["unmatched_pairs"][0]
    assert refusal["reason"] == "missing_matched_comparison_key"
    assert refusal["missing_required_fields"] == {"m4": [], "gb10": []}
    assert refusal["missing_matched_comparison_key"] == {"m4": True, "gb10": True}
    assert refusal["mismatched_fields"] == []
    assert "ratios" not in refusal


def test_alias_route_label_is_strict_workload_identity(tmp_path: Path) -> None:
    path = tmp_path / "alias-routes.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    aliased_hybrid_row(
                        "M4 Max",
                        route="mamba3",
                        model_route="hybrid-m",
                        backend="mamba3",
                        symbols="M",
                        tokens_per_second=40.0,
                    ),
                    aliased_hybrid_row(
                        "GB10",
                        route="hybrid-m",
                        model_route="hybrid-m",
                        backend="mamba3",
                        symbols="M",
                        tokens_per_second=20.0,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    assert payload["unmatched_pair_count"] == 1
    assert {field["field"] for field in payload["unmatched_pairs"][0]["mismatched_fields"]} == {
        "route"
    }
    assert payload["comparisons"] == []


def test_alias_route_rows_match_when_alias_and_stack_are_identical(tmp_path: Path) -> None:
    path = tmp_path / "matched-alias-routes.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    aliased_hybrid_row(
                        "M4 Max",
                        route="m2rnn",
                        model_route="hybrid-r",
                        backend="m2rnn",
                        symbols="R",
                        tokens_per_second=30.0,
                    ),
                    aliased_hybrid_row(
                        "GB10",
                        route="m2rnn",
                        model_route="hybrid-r",
                        backend="m2rnn",
                        symbols="R",
                        tokens_per_second=15.0,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    comparison = payload["comparisons"][0]
    assert comparison["match"]["profile"] == "hybrid-smoke"
    assert comparison["match"]["route"] == "m2rnn"
    assert comparison["match"]["model_route"] == "hybrid-r"
    assert comparison["match"]["backend_plan"]["backend_summary"] == {"m2rnn": 1}
    assert comparison["ratios"]["m4_tokens_per_second_over_gb10"] == 2.0


def test_zero_valued_match_fields_are_preserved(tmp_path: Path) -> None:
    m4 = bench_row("M4 Max", tokens_per_second=10.0)
    gb10 = bench_row("GB10", tokens_per_second=5.0)
    for row in (m4, gb10):
        row["warmup_steps"] = 0
        row["seed"] = 0
        row["config"]["warmup_steps"] = 99
        row["config"]["seed"] = 99
    path = tmp_path / "zero-fields.json"
    path.write_text(json.dumps({"cases": [m4, gb10]}), encoding="utf-8")

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    match = payload["comparisons"][0]["match"]
    assert match["warmup_steps"] == 0
    assert match["seed"] == 0
    assert payload["comparisons"][0]["ratios"]["m4_tokens_per_second_over_gb10"] == 2.0


def test_run_metadata_fields_are_used_for_matching(tmp_path: Path) -> None:
    m4 = bench_row("M4 Max", tokens_per_second=9.0)
    gb10 = bench_row("GB10", tokens_per_second=3.0)
    for row in (m4, gb10):
        for field in (
            "model_source",
            "vocab_size",
            "d_model",
            "n_heads",
            "n_layers",
            "mlp_dim",
            "learning_rate",
            "seed",
        ):
            row.pop(field, None)
        row["config"] = {}
        row["matched_run"] = {
            "key": {
                "dtype": "bfloat16",
                "batch_size": 1,
                "seq_len": 8,
                "vocab_size": 32,
                "d_model": 8,
                "n_heads": 1,
                "n_layers": 1,
                "mlp_dim": 16,
                "warmup_steps": 1,
                "measured_steps": 3,
                "compile": False,
                "include_structure": False,
                "learning_rate": 0.001,
                "model_source": "cppmega_mlx.models.tiny_lm",
                "model_route": "tiny",
                "route_plan": tiny_route_plan(),
                "backend_plan": mlx_tiny_backend_plan(),
                "backend": "mlx",
                "data_contract": "synthetic_tokens",
            }
        }
        row["run_metadata"] = {
            "framework": {
                "python": "3.13.0",
                "mlx": "0.31.0",
                "mlx_lm": "0.31.0",
                "mlx_metal": "0.31.0",
            },
            "workload": {"seed": 0},
        }
    path = tmp_path / "metadata-only.ndjson"
    path.write_text(f"{json.dumps(m4)}\n{json.dumps(gb10)}\n", encoding="utf-8")

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["comparisons"][0]["match"]["data_contract"] == "synthetic_tokens"
    assert payload["comparisons"][0]["match"]["framework"] == "mlx"
    assert payload["comparisons"][0]["match"]["backend"] == "metal"
    assert payload["comparisons"][0]["ratios"]["m4_tokens_per_second_over_gb10"] == 3.0


def test_receipt_only_rows_match_from_nested_bench_receipt(tmp_path: Path) -> None:
    path = tmp_path / "receipt-only.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    receipt_only_row(
                        "M4 Max",
                        tokens_per_second=48.0,
                        peak_memory_bytes=900,
                    ),
                    receipt_only_row(
                        "GB10",
                        tokens_per_second=24.0,
                        peak_memory_bytes=1_800,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["matched_comparison_count"] == 1
    comparison = payload["comparisons"][0]
    assert comparison["match"]["profile"] == "hybrid-smoke"
    assert comparison["match"]["route"] == "mamba3"
    assert comparison["match"]["model_route"] == "hybrid-m"
    assert comparison["match"]["backend_plan"]["backend_summary"] == {"mamba3": 1}
    assert comparison["match"]["data_contract"] == "synthetic_tokens"
    assert comparison["match"]["framework"] == "mlx"
    assert comparison["match"]["mlx_version"] == "0.31.0"
    assert comparison["m4"]["tokens_per_second"] == 48.0
    assert comparison["gb10"]["tokens_per_second"] == 24.0
    assert comparison["ratios"]["m4_tokens_per_second_over_gb10"] == 2.0
    assert comparison["ratios"]["gb10_peak_memory_bytes_over_m4"] == 2.0


def test_receipt_only_parquet_data_contract_mismatch_blocks_match(tmp_path: Path) -> None:
    path = tmp_path / "receipt-only-parquet-mismatch.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    receipt_only_row("M4 Max", data_contract="synthetic_tokens"),
                    receipt_only_row("GB10", data_contract="parquet_clang_v10_code"),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    refusal = payload["unmatched_pairs"][0]
    assert refusal["reason"] == "matched_comparison_key_mismatch"
    assert {field["field"] for field in refusal["mismatched_fields"]} == {"data_contract"}
    assert "ratios" not in refusal
    assert payload["comparisons"] == []


def test_package_dir_keeps_parquet_contract_refusal_ratio_free(tmp_path: Path) -> None:
    input_path = tmp_path / "receipt-only-parquet-mismatch.json"
    package_dir = tmp_path / "parquet_refused_package"
    input_path.write_text(
        json.dumps(
            {
                "cases": [
                    receipt_only_row("M4 Max", data_contract="synthetic_tokens"),
                    receipt_only_row("GB10", data_contract="parquet_clang_v10_code"),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare(
        "--input",
        str(input_path),
        "--package-dir",
        str(package_dir),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["compare_package"]["matched_comparison_count"] == 0
    assert payload["compare_package"]["unmatched_pair_count"] == 1
    assert payload["compare_package"]["parity_claim_refused"] is True

    report = json.loads((package_dir / "compare_report.json").read_text(encoding="utf-8"))
    assert report["comparisons"] == []
    assert report["unmatched_pair_count"] == 1
    assert "ratios" not in report["unmatched_pairs"][0]

    assert (package_dir / "matched_comparisons.jsonl").read_text(encoding="utf-8") == ""
    refused_rows = [
        json.loads(line)
        for line in (package_dir / "refused_pairs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(refused_rows) == 1
    refusal = refused_rows[0]
    assert refusal["status"] == "unmatched"
    assert refusal["parity_claim_refused"] is True
    assert {field["field"] for field in refusal["mismatched_fields"]} == {"data_contract"}
    assert "ratios" not in refusal


def test_receipt_only_software_mismatch_blocks_match(tmp_path: Path) -> None:
    path = tmp_path / "receipt-only-software-mismatch.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    receipt_only_row("M4 Max", mlx_version="0.31.0"),
                    receipt_only_row("GB10", mlx_version="0.32.0"),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    refusal = payload["unmatched_pairs"][0]
    assert refusal["reason"] == "matched_comparison_key_mismatch"
    assert {field["field"] for field in refusal["mismatched_fields"]} == {"mlx_version"}
    assert payload["comparisons"] == []


def test_explicit_comparison_key_mismatch_is_refusal_reason_when_fields_also_differ(
    tmp_path: Path,
) -> None:
    m4 = receipt_only_row(
        "M4 Max",
        data_contract="synthetic_tokens",
        mlx_version="0.31.0",
    )
    gb10 = receipt_only_row(
        "GB10",
        data_contract="parquet_clang_v10_code",
        mlx_version="0.32.0",
    )
    path = tmp_path / "explicit-key-and-field-mismatch.json"
    path.write_text(json.dumps({"cases": [m4, gb10]}), encoding="utf-8")

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    assert payload["comparisons"] == []
    refusal = payload["unmatched_pairs"][0]
    assert refusal["reason"] == "matched_comparison_key_mismatch"
    assert {field["field"] for field in refusal["mismatched_comparison_keys"]} == {
        "comparison_key.workload",
        "comparison_key.software",
    }
    assert {field["field"] for field in refusal["mismatched_fields"]} == {
        "data_contract",
        "mlx_version",
    }
    assert "ratios" not in refusal


def test_gb10_torch_cuda_row_is_ingested_but_not_matched_to_mlx(tmp_path: Path) -> None:
    path = tmp_path / "mixed-framework.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    bench_row("M4 Max", tokens_per_second=120.0),
                    torch_cuda_row("GB10", tokens_per_sec=80.0),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["incomplete_match_metadata_counts"] == {"m4": 0, "gb10": 0}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    refusal = payload["unmatched_pairs"][0]
    assert refusal["reason"] == "matched_comparison_key_mismatch"
    assert "ratios" not in refusal
    assert refusal["missing_required_fields"] == {"m4": [], "gb10": []}
    assert {"framework", "backend"} <= {
        field["field"] for field in refusal["mismatched_fields"]
    }
    assert payload["comparisons"] == []


def test_gb10_torch_cuda_rows_match_only_with_same_stack_metadata(tmp_path: Path) -> None:
    m4 = torch_cuda_row(
        "M4 Max external torch baseline",
        tokens_per_sec=50.0,
        max_alloc_gib=0.5,
    )
    gb10 = torch_cuda_row("GB10 DGX Spark", tokens_per_sec=100.0, max_alloc_gib=1.0)
    path = tmp_path / "torch-cuda.ndjson"
    path.write_text(f"{json.dumps(m4)}\n{json.dumps(gb10)}\n", encoding="utf-8")

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    comparison = payload["comparisons"][0]
    assert comparison["match"]["framework"] == "torch"
    assert comparison["match"]["backend"] == "torch_sdpa"
    assert comparison["match"]["torch_version"] == "2.12.0.dev20260430+cu132"
    assert comparison["match"]["cuda_version"] == "13.2"
    assert comparison["match"]["driver_version"] == "590.44"
    assert comparison["gb10"]["device_name"] == "NVIDIA GB10 DGX Spark"
    assert comparison["gb10"]["device_capability"] == "12.1"
    assert comparison["gb10"]["peak_memory_bytes"] == 1024**3
    assert comparison["ratios"]["gb10_tokens_per_second_over_m4"] == 2.0
    assert comparison["ratios"]["gb10_peak_memory_bytes_over_m4"] == 2.0


def test_gb10_row_without_required_workload_metadata_does_not_report_ratios(tmp_path: Path) -> None:
    gb10 = torch_cuda_row("GB10", tokens_per_sec=100.0)
    gb10.pop("model_source")
    gb10["comparison_key"]["workload"].pop("model_source")
    path = tmp_path / "incomplete-gb10.json"
    path.write_text(
        json.dumps({"cases": [bench_row("M4 Max"), gb10]}),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["incomplete_match_metadata_counts"] == {"m4": 0, "gb10": 1}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    refusal = payload["unmatched_pairs"][0]
    assert refusal["reason"] == "missing_required_match_fields"
    assert refusal["missing_required_fields"] == {"m4": [], "gb10": ["model_source"]}
    assert payload["comparisons"] == []


def test_gb10_torch_cuda_row_without_device_metadata_is_incomplete(tmp_path: Path) -> None:
    gb10 = torch_cuda_row("GB10", tokens_per_sec=100.0)
    for field in ("device", "capability", "cuda", "driver_version"):
        gb10.pop(field)
    for field in ("device_name", "device_capability", "cuda_version", "driver_version"):
        gb10["comparison_key"]["software"].pop(field)
    path = tmp_path / "incomplete-gb10-device.json"
    path.write_text(
        json.dumps({"cases": [bench_row("M4 Max"), gb10]}),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_matching_rows"
    assert payload["host_counts"] == {"m4": 1, "gb10": 1, "ignored": 0}
    assert payload["incomplete_match_metadata_counts"] == {"m4": 0, "gb10": 1}
    assert payload["matched_comparison_count"] == 0
    assert payload["parity_claim_refused"] is True
    refusal = payload["unmatched_pairs"][0]
    assert refusal["reason"] == "missing_required_match_fields"
    assert refusal["missing_required_fields"] == {
        "m4": [],
        "gb10": ["device_name", "cuda_version", "driver_version", "device_capability"],
    }
    assert payload["comparisons"] == []


def test_jsonl_no_match_marks_refused_parity_claim(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    bench_row("M4 Max", mlx_version="0.31.0"),
                    bench_row("GB10", mlx_version="0.32.0"),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_compare("--input", str(path), "--jsonl")

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "no_matching_rows"
    assert row["parity_claim_refused"] is True
    assert row["matched_comparison_count"] == 0
    assert row["unmatched_pair_count"] == 1
    assert "ratios" not in row["unmatched_pairs"][0]
    assert {field["field"] for field in row["unmatched_pairs"][0]["mismatched_fields"]} == {
        "mlx_version"
    }
