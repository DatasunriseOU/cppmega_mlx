from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

import scripts.m04_train_step as m04_train_step
from cppmega_mlx.training.optimizers import (
    ADAMW_BASE_CLASS,
    ADAMW_FP32_MOMENTS_CLASS,
    AdamWFP32Moments,
    collect_adamw_moment_dtypes,
    dtype_name,
    make_adamw,
)
from scripts.m04_train_step import (
    OBSERVED_OPTIMIZER_IDENTITY,
    GRAD_CHECKPOINT_EXPECTATION,
    REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
    REQUIRED_DTYPE,
    REQUIRED_MODEL_GEOMETRY,
    REQUIRED_MODEL_SOURCE,
    acceptance_gate_payload,
    applied_memory_limit_api_path_from_payload,
    local_gb10_quarter_preflight_payload,
    target_dataset_path,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m04_train_step.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
LOCAL_MLX_PYTHON = Path("/Volumes/external/sources/mlx/python")
LOCAL_MLX_LIB = LOCAL_MLX_PYTHON / "mlx" / "lib"
BASELINE_RECEIPT = ROOT / "bench" / "baselines" / "m04_train_step.json"
GB10_SAMPLE = (
    ROOT
    / "data"
    / "parquet_samples"
    / "gb10"
    / "clang_semantic_4k_v10"
    / "val_00000.parquet"
)
TARGET_PARQUET = "data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet"
REAL_PARQUET_COLUMNS = (
    "token_ids",
    "structure_ids",
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
)


def strip_known_tvm_stderr_noise(stderr: str) -> str:
    lines = [
        line
        for line in stderr.splitlines()
        if not (
            "arm_aprofile.cc:125: Warning: Cannot parse Arm(R)-based target features"
            in line
            and "without LLVM support" in line
        )
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def canonical_allocation_probe(**overrides: Any) -> dict[str, Any]:
    probe = {
        "status": "ok",
        "allocation_ready": True,
        "source": REQUIRED_MODEL_SOURCE,
        "allocation_mode": "full_profile_allocation_probe",
        "required_geometry": REQUIRED_MODEL_GEOMETRY,
        "profile_geometry": REQUIRED_MODEL_GEOMETRY,
        "geometry_matches_required": True,
        "profile_name": "local_gb10_quarter",
        "model_class": "HybridTinyLM",
        "eval_scope": "parameters_only_no_forward_no_training",
        "forward_executed": False,
        "training_executed": False,
        "memory_before": {"active_memory_bytes": 0},
        "memory_after": {"active_memory_bytes": 1024},
    }
    probe.update(overrides)
    return probe


def test_fp8_path_policies_set_explicit_runtime_routes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(m04_train_step, "ensure_tilelang_dev_env_for_path_c", lambda: None)
    for key in (
        "CPPMEGA_KERNEL_PATH__SPARSE_MLA",
        "CPPMEGA_SPARSE_MLA_FP8_ROUTE",
        "CPPMEGA_MAMBA3_PATH_C_BWD",
    ):
        monkeypatch.delenv(key, raising=False)

    path_c_args = m04_train_step.build_parser().parse_args(
        ["--synthetic", "--dtype", "fp8_path_c", "--output", str(tmp_path / "c.json")]
    )
    with m04_train_step.fp8_path_c_kernel_policy(path_c_args):
        assert os.environ["CPPMEGA_KERNEL_PATH__SPARSE_MLA"] == "path_c"
        assert os.environ["CPPMEGA_SPARSE_MLA_FP8_ROUTE"] == "path_c"
        assert os.environ["CPPMEGA_MAMBA3_PATH_C_BWD"] == "path_c"
    assert "CPPMEGA_SPARSE_MLA_FP8_ROUTE" not in os.environ
    assert "CPPMEGA_MAMBA3_PATH_C_BWD" not in os.environ

    path_b_args = m04_train_step.build_parser().parse_args(
        ["--synthetic", "--dtype", "fp8_path_b", "--output", str(tmp_path / "b.json")]
    )
    with m04_train_step.fp8_path_b_kernel_policy(path_b_args):
        assert os.environ["CPPMEGA_KERNEL_PATH__SPARSE_MLA"] == "path_b"
        assert os.environ["CPPMEGA_SPARSE_MLA_FP8_ROUTE"] == "path_b"
        assert "CPPMEGA_MAMBA3_PATH_C_BWD" not in os.environ


def test_tilelang_dev_env_points_to_build_root_and_runtime_libs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "tl_apache_tvm_swap"
    build_root = source_root / "build"
    (source_root / "tilelang").mkdir(parents=True)
    (source_root / "3rdparty" / "tvm" / "python").mkdir(parents=True)
    (build_root / "lib").mkdir(parents=True)
    (build_root / "tvm").mkdir(parents=True)
    monkeypatch.setenv("TILELANG_DEV_BUILD_ROOT", str(source_root))
    monkeypatch.delenv("TVM_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)

    m04_train_step.ensure_tilelang_dev_env_for_path_c()

    assert os.environ["TILELANG_DEV_BUILD_ROOT"] == str(build_root)
    assert os.environ["TVM_LIBRARY_PATH"] == str(build_root / "lib")
    assert str(build_root / "lib") in os.environ["DYLD_LIBRARY_PATH"].split(os.pathsep)


def test_m04_import_preserves_recipes_package_exports() -> None:
    import cppmega_mlx.recipes as recipes

    assert recipes is sys.modules["cppmega_mlx.recipes"]
    assert isinstance(recipes.__all__, list)
    assert "local_gb10_quarter" in recipes.__all__
    assert hasattr(recipes, "local_gb10_quarter")


def test_profile_hold_seconds_is_opt_in(tmp_path: Path) -> None:
    args = m04_train_step.build_parser().parse_args(
        ["--synthetic", "--output", str(tmp_path / "receipt.json")]
    )

    assert args.profile_hold_seconds == 0.0


class _FakeCacheLimitMLX:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def set_cache_limit(self, limit: int) -> int:
        self.calls.append(limit)
        return 987654321


def test_path_c_local_gb10_defaults_cache_limit_to_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    args = m04_train_step.build_parser().parse_args(
        [
            "--model-profile",
            "local_gb10_quarter",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    fake = _FakeCacheLimitMLX()

    payload = m04_train_step.apply_cache_limit_payload(args, mx_module=fake)

    assert fake.calls == [0]
    assert payload == {
        "configured": True,
        "applied": True,
        "limit_bytes": 0,
        "source": "path_c_default",
        "api_path": "mx.set_cache_limit",
        "previous_limit_bytes": 987654321,
    }


def test_non_path_c_keeps_mlx_cache_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CPPMEGA_KERNEL_PATH", raising=False)
    monkeypatch.delenv("CPPMEGA_MLX_CACHE_LIMIT_BYTES", raising=False)
    args = m04_train_step.build_parser().parse_args(
        [
            "--model-profile",
            "local_gb10_quarter",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    fake = _FakeCacheLimitMLX()

    payload = m04_train_step.apply_cache_limit_payload(args, mx_module=fake)

    assert fake.calls == []
    assert payload["configured"] is False
    assert payload["source"] == "mlx_default"


def run_script(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pythonpath = [str(ROOT)]
    if LOCAL_MLX_PYTHON.exists():
        pythonpath.insert(0, str(LOCAL_MLX_PYTHON))
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if LOCAL_MLX_LIB.exists():
        dyld_path = [str(LOCAL_MLX_LIB)]
        if env.get("DYLD_LIBRARY_PATH"):
            dyld_path.append(env["DYLD_LIBRARY_PATH"])
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(dyld_path)
    result = subprocess.run(
        [str(PYTHON if PYTHON.exists() else sys.executable), str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return subprocess.CompletedProcess(
        args=result.args,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=strip_known_tvm_stderr_noise(result.stderr),
    )


def load_json_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def copy_real_parquet_head(
    source_path: Path, sample_path: Path, *, row_count: int = 4
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet_file = pq.ParquetFile(source_path)
    batch = next(
        parquet_file.iter_batches(
            batch_size=row_count,
            columns=list(REAL_PARQUET_COLUMNS),
        )
    )
    pq.write_table(pa.Table.from_batches([batch]), sample_path)


def tiny_args(output: Path, *, steps: int = 1) -> list[str]:
    return [
        "--synthetic",
        "--steps",
        str(steps),
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--vocab-size",
        "32",
        "--hidden-size",
        "8",
        "--pattern",
        "M",
        "--depth",
        "1",
        "--output",
        str(output),
        "--json",
    ]


def assert_m04_receipt_contract(payload: dict[str, Any]) -> None:
    assert payload["receipt_schema_version"] == 1
    assert payload["receipt_scope"] == "local_mlx_m04_train_step"
    assert payload["issue"]["id"] == "cppmega-mlx-t8f.4"
    assert payload["local_only"] is True
    assert payload["gb10_training_correctness_claim"] is False
    assert payload["m4_vs_gb10_throughput_parity_claim"] is False
    assert payload["full_m0_4_acceptance_claim"] is False
    gate = payload["acceptance_gate"]
    assert gate["full_target_dataset"] == TARGET_PARQUET
    assert gate["full_target_dataset_100_step_required"] is True
    assert gate["local_gb10_quarter_required"] is True
    assert gate["required_model_profile"] == "local_gb10_quarter"
    assert gate["required_dtype"] == REQUIRED_DTYPE
    assert gate["observed_dtype"] == "bfloat16"
    assert gate["dtype_ok"] is True
    assert gate["required_optimizer_name"] == "AdamW"
    assert gate["grad_checkpoint_required"] is True
    assert gate["full_local_gb10_quarter_gate_required"] is True
    for key in (
        "real_parquet_source_identity",
        "target_parquet_path_ok",
        "dataset_name_ok",
        "dataset_format_ok",
        "dtype_ok",
        "local_gb10_quarter_preflight",
        "local_gb10_quarter_preflight_ok",
        "model_identity_ok",
        "model_identity",
        "optimizer_identity_ok",
        "optimizer_identity",
        "required_adamw_master_moment_dtype",
        "observed_adamw_master_moment_dtypes",
        "fp32_adamw_master_moments_ok",
        "adamw_ok",
        "grad_checkpoint_expectation_ok",
        "grad_checkpoint_identity",
        "step_count_ok",
        "loss_decrease_ok",
        "loss_fields_ok",
        "all_finite_ok",
        "optimizer_update_ok",
        "m4_runtime_metadata",
        "m4_runtime_metadata_ok",
        "full_local_gb10_quarter_gate_completed",
        "full_local_gb10_quarter_gate_blockers",
    ):
        assert key in gate
    assert gate["real_parquet_source_identity"]["required_path"] == TARGET_PARQUET
    assert (
        payload["local_gb10_quarter_preflight"] == gate["local_gb10_quarter_preflight"]
    )
    preflight = payload["local_gb10_quarter_preflight"]
    assert preflight["profile_name"] == "local_gb10_quarter"
    assert preflight["source"] == REQUIRED_MODEL_SOURCE
    assert preflight["required_geometry"] == REQUIRED_MODEL_GEOMETRY
    assert preflight["profile_geometry"] == REQUIRED_MODEL_GEOMETRY
    assert preflight["geometry_matches_required"] is True
    assert preflight["tokenizer_contract"]["resolved"] is True
    assert preflight["tokenizer_contract"]["expected_vocab_size"] == 65_536
    assert preflight["tokenizer_contract"]["blocker_id"] == "cppmega-mlx-t8f.1"
    assert preflight["tokenizer_contract"]["milestone"] == "M0.1"
    assert (
        "<FIM_INSTRUCTION>"
        in preflight["tokenizer_contract"]["required_special_tokens"]
    )
    assert "CODE_START" in preflight["tokenizer_contract"]["reason"]
    assert "M0.1 is closed" in preflight["tokenizer_contract"]["reason"]
    if payload["workload"]["probe_local_gb10_quarter_allocation"]:
        assert preflight["allocation_attempted"] is True
        assert preflight["allocation_ready"] is True
        assert preflight["allocation_mode"] == "full_profile_allocation_probe"
        assert preflight["allocation_probe"]["status"] == "ok"
        assert preflight["allocation_probe"]["source"] == REQUIRED_MODEL_SOURCE
        assert preflight["allocation_probe"]["allocation_mode"] == (
            "full_profile_allocation_probe"
        )
        assert preflight["allocation_probe"]["required_geometry"] == (
            REQUIRED_MODEL_GEOMETRY
        )
        assert preflight["allocation_probe"]["profile_geometry"] == (
            REQUIRED_MODEL_GEOMETRY
        )
        assert preflight["allocation_probe"]["geometry_matches_required"] is True
        assert preflight["allocation_probe"]["model_class"] == "HybridTinyLM"
        assert preflight["allocation_probe"]["forward_executed"] is False
        assert preflight["allocation_probe"]["training_executed"] is False
        assert preflight["ok"] is True
        assert preflight["blockers"] == []
        assert gate["local_gb10_quarter_preflight_ok"] is True
    else:
        assert preflight["allocation_attempted"] is False
        assert preflight["allocation_ready"] is False
        assert preflight["allocation_mode"] == "allocation_free_preflight"
        assert preflight["ok"] is False
        assert {"allocation_attempted", "allocation_ready"}.issubset(
            set(preflight["blockers"])
        )
        assert "tokenizer_contract_resolved" not in preflight["blockers"]
        assert gate["local_gb10_quarter_preflight_ok"] is False
    if gate["full_target_dataset_100_step_completed"]:
        assert gate["uses_full_target_dataset"] is True
        assert gate["real_parquet_source_identity"]["ok"] is True
        assert gate["full_target_dataset_blocker"] is None
    else:
        assert gate["full_target_dataset_blocker"]
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert gate["full_local_gb10_quarter_gate_blockers"]
    model_identity = gate["model_identity"]
    assert model_identity["required_name"] == "local_gb10_quarter"
    assert model_identity["required_source"] == REQUIRED_MODEL_SOURCE
    assert model_identity["required_profile"] == "local_gb10_quarter"
    assert model_identity["required_geometry"] == REQUIRED_MODEL_GEOMETRY
    assert model_identity["ok"] is gate["model_identity_ok"]
    optimizer_identity = gate["optimizer_identity"]
    assert gate["required_adamw_master_moment_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert optimizer_identity["required_master_moment_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert optimizer_identity["master_moment_evidence"]["required_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert (
        optimizer_identity["master_moment_dtype_ok"]
        is (gate["fp32_adamw_master_moments_ok"])
    )
    assert payload["workload"]["dtype"] == "bfloat16"
    model_payload = payload["model"]
    if model_payload.get("metadata_only"):
        assert payload["workload"]["mode"] == "metadata_only_no_forward_no_training"
        assert model_payload["source"] is None
        assert model_payload["name"] is None
        assert model_payload["required_source"] == REQUIRED_MODEL_SOURCE
        assert model_payload["required_profile"] == "local_gb10_quarter"
        assert model_payload["profile_matches_required"] is False
        assert model_payload["local_gb10_quarter_preflight"] == preflight
        assert model_payload["forward_executed"] is False
        assert model_payload["training_executed"] is False
    else:
        assert model_payload["source"] == "cppmega_mlx.models.hybrid_lm"
        assert model_payload["name"] == "HybridTinyLM"
        assert model_payload["required_profile"] == "local_gb10_quarter"
        assert model_payload["profile_matches_required"] is False
        assert model_payload["local_gb10_quarter_preflight"] == preflight
    assert payload["training"]["optimizer"]["name"] == "AdamW"
    assert payload["training"]["optimizer"]["class"] == (ADAMW_FP32_MOMENTS_CLASS)
    assert payload["training"]["optimizer"]["base_class"] == ADAMW_BASE_CLASS
    assert payload["training"]["optimizer"]["adamw"] is True
    assert payload["training"]["optimizer"]["required_master_moment_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert (
        payload["training"]["optimizer"]["master_moment_evidence"]
        == (optimizer_identity["master_moment_evidence"])
    )
    grad_checkpoint_expected = bool(payload["workload"].get("grad_checkpoint", False))
    assert payload["training"]["grad_checkpoint"]["required"] is True
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is (
        grad_checkpoint_expected
    )
    assert payload["training"]["grad_checkpoint"]["expectation_satisfied"] is (
        grad_checkpoint_expected
    )
    assert gate["grad_checkpoint_observed_enabled"] is grad_checkpoint_expected
    assert gate["grad_checkpoint_expectation_ok"] is grad_checkpoint_expected
    assert gate["grad_checkpoint_identity"]["observed_enabled"] is (
        grad_checkpoint_expected
    )
    assert gate["grad_checkpoint_identity"]["expectation_satisfied"] is (
        grad_checkpoint_expected
    )
    assert gate["grad_checkpoint_identity"]["ok"] is grad_checkpoint_expected
    expected_model = (
        "metadata_only_no_observed_model"
        if model_payload.get("metadata_only")
        else "HybridTinyLM"
    )
    expected_route = (
        "metadata_only_no_forward_no_training"
        if model_payload.get("metadata_only")
        else model_payload["route_symbols"]
    )
    assert payload["baseline_row"] == {
        "batch_size": payload["workload"]["batch_size"],
        "commit": payload["software"]["git_commit"] or "unknown",
        "dtype": "bfloat16",
        "gb10_parity_claim": False,
        "hardware": payload["baseline_row"]["hardware"],
        "local_only": True,
        "mode": payload["workload"]["mode"],
        "model": expected_model,
        "route": expected_route,
        "seq_len": payload["workload"]["seq_len"],
        "tokens_per_second": payload["timing"]["tokens_per_second"] or 0.0,
    }


def assert_regression_report_matches_payload(payload: dict[str, Any]) -> None:
    report = payload["regression_report"]
    route_dispatch = report["route_dispatch"]
    producer_gate = report["fp8_path_c_producer_gate"]

    assert route_dispatch["raw"] == payload["training"].get("kernel_dispatch", [])
    assert "fallback_reason" in route_dispatch
    assert report["fallback_reason"] == route_dispatch["fallback_reason"]
    assert report["dtype"]["requested"] == payload["workload"]["dtype"]
    assert report["optimizer"]["key"] == payload["workload"]["optimizer"]["key"]
    assert report["memory"]["peak_memory_bytes"] == payload["memory"][
        "peak_memory_bytes"
    ]
    assert report["training"]["all_finite"] == payload["training"]["all_finite"]
    assert report["training"]["initial_loss"] == payload["training"]["initial_loss"]
    assert report["training"]["final_loss"] == payload["training"]["final_loss"]
    assert report["training"]["mean_loss"] == payload["training"].get("mean_loss")
    assert report["training"]["loss_decreased"] == payload["training"][
        "loss_decreased"
    ]
    assert report["gate_summary"]["dtype"] == payload["workload"]["dtype"]
    assert report["gate_summary"]["optimizer"] == payload["workload"]["optimizer"][
        "key"
    ]
    assert report["gate_summary"]["fallback_reason"] == route_dispatch[
        "fallback_reason"
    ]
    assert report["gate_summary"]["fp8_path_c_producer_status"] == producer_gate[
        "status"
    ]
    assert report["gate_summary"]["fp8_path_c_producer_ok"] == producer_gate["ok"]
    assert producer_gate["name"] == "fp8_path_c_sparse_mla_producer"
    assert producer_gate["large_tensor_staging_allowed"] is False
    assert producer_gate["hidden_wrapper_quantization_allowed"] is False
    assert producer_gate["kernel_boundary_quantization_allowed"] is False
    assert "regression_report.fp8_path_c_producer_gate" in producer_gate[
        "receipt_field_paths"
    ]
    if payload["workload"]["dtype"] == "fp8_path_c":
        assert producer_gate["required"] is True
        assert producer_gate["fallback_to_path_b_allowed"] is False
    else:
        assert producer_gate["required"] is False
        assert producer_gate["status"] == "not_requested"
    assert "path_b_observed" in route_dispatch
    assert "path_c_observed" in route_dispatch
    assert "path_summary" in route_dispatch
    assert report["throughput"]["tokens_per_second"] == payload["timing"][
        "tokens_per_second"
    ]
    assert report["throughput"]["claim_gate"]["ok"] is True
    assert report["gate_summary"]["tokens_per_second_claim_ok"] is True
    assert report["gate_summary"]["bogus_tok_sec_claim_detected"] is False
    assert report["visibility_gate"] == {
        "route_dispatch_visible": True,
        "dtype_visible": True,
        "optimizer_visible": True,
        "memory_peak_visible": True,
        "tokens_per_second_visible": payload["timing"]["tokens_per_second"]
        is not None,
        "finite_visible": True,
        "loss_visible": True,
        "fallback_reason_visible": True,
    }


def assert_m04_20step_matrix_plan(payload: dict[str, Any]) -> None:
    matrix = payload["m04_20step_matrix"]

    assert matrix["name"] == "m04_local_gb10_20step_dtype_optimizer_matrix"
    assert matrix["status"] == "commands_prepared_not_executed_by_this_receipt"
    assert matrix["profile"] == "local_gb10_quarter"
    assert matrix["dataset"] == TARGET_PARQUET
    assert matrix["steps"] == 20
    assert matrix["acceptance_steps"] == 100
    assert matrix["batch_size"] == 1
    assert matrix["seq_len"] == 4096
    assert matrix["dtype_routes"] == ["bf16", "fp8_path_b", "fp8_path_c", "int8"]
    assert matrix["optimizers"] == [
        "adamw",
        "muon",
        "muon_adamw",
        "lion",
        "lion8bit",
        "adam8bit",
    ]
    baseline = matrix["baseline_comparison"]
    assert baseline["baseline_tokens_per_second"] == 900.0
    assert baseline["baseline_kind"] == (
        "existing_real_parquet_bs1_seq4096_20step_receipts"
    )
    assert baseline["baseline_scope"] == "local_m4_only_not_gb10_parity"
    assert {
        row["case_id"] for row in baseline["reference_receipts"]
    } == {
        "lion8bit_sym_lr1e-4",
        "adam8bit_sym_lr1e-4",
        "adam8bit_dyn_lr1e-4",
    }
    assert any(
        row["meets_900_tok_s_baseline"] is True
        for row in baseline["reference_receipts"]
    )
    assert matrix["command_sets"] == [
        "dry_run",
        "smoke_1step",
        "real_20step",
        "real_100step",
    ]
    cases = {case["case_id"]: case for case in matrix["cases"]}
    assert len(cases) == 24
    assert len(matrix["real_20step_commands"]) == 24
    assert len(matrix["real_100step_commands"]) == 24
    assert len(matrix["dry_run_commands"]) == 24
    assert len(matrix["smoke_commands"]) == 24
    assert cases["bf16_adamw_20step"]["supported"] is True
    assert "--dtype bfloat16" in cases["bf16_adamw_20step"]["command"]
    assert "--optimizer adamw" in cases["bf16_adamw_20step"]["command"]
    assert "--dry-run-json" in cases["bf16_adamw_20step"]["dry_run_command"]
    assert "--steps 1" in cases["bf16_adamw_20step"]["smoke_command"]
    assert "--steps 20" in cases["bf16_adamw_20step"]["real_20step_command"]
    assert "--steps 100" in matrix["real_100step_commands"][0]
    assert "--require-loss-decrease" in matrix["real_100step_commands"][0]
    assert cases["bf16_muon_20step"]["supported"] is True
    assert "--optimizer muon" in cases["bf16_muon_20step"]["command"]
    assert cases["fp8_path_b_muon_adamw_20step"]["supported"] is True
    assert "--dtype fp8_path_b" in cases["fp8_path_b_muon_adamw_20step"]["command"]
    assert "--optimizer muon_adamw" in cases[
        "fp8_path_b_muon_adamw_20step"
    ]["command"]
    assert cases["fp8_path_c_muon_adamw_20step"]["supported"] is True
    assert "--dtype fp8_path_c" in cases["fp8_path_c_muon_adamw_20step"]["command"]
    assert "--optimizer muon_adamw" in cases[
        "fp8_path_c_muon_adamw_20step"
    ]["command"]
    assert cases["fp8_path_c_muon_20step"]["supported"] is True
    assert "--dtype fp8_path_c" in cases["fp8_path_c_muon_20step"]["command"]
    assert "--optimizer muon" in cases["fp8_path_c_muon_20step"]["command"]
    assert cases["int8_muon_20step"]["supported"] is True
    assert "--dtype bfloat16" in cases["int8_muon_20step"]["command"]
    assert "--optimizer int8" in cases["int8_muon_20step"]["command"]
    assert cases["int8_muon_adamw_20step"]["supported"] is True
    assert "--dtype bfloat16" in cases["int8_muon_adamw_20step"]["command"]
    assert "--optimizer int8" in cases["int8_muon_adamw_20step"]["command"]
    assert cases["int8_adam8bit_20step"]["supported"] is True
    assert cases["int8_lion8bit_20step"]["supported"] is True
    assert cases["int8_adamw_20step"]["supported"] is True
    assert "--dtype bfloat16" in cases["int8_adamw_20step"]["command"]
    assert "--optimizer adam8bit" in cases["int8_adamw_20step"]["command"]
    assert cases["int8_lion_20step"]["supported"] is True
    assert "--dtype bfloat16" in cases["int8_lion_20step"]["command"]
    assert "--optimizer lion8bit" in cases["int8_lion_20step"]["command"]


def _receipt_args_for_regression_report(
    tmp_path: Path,
    *,
    dtype: str = "bfloat16",
    optimizer: str = "adamw",
    pattern: str = "M",
    depth: str = "1",
    dsa_a_layer_ranks: str = "",
) -> argparse.Namespace:
    return m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            dtype,
            "--optimizer",
            optimizer,
            "--pattern",
            pattern,
            "--depth",
            depth,
            "--dsa-a-layer-ranks",
            dsa_a_layer_ranks,
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )


def test_regression_report_rejects_bogus_tokens_per_second_claim(
    tmp_path: Path,
) -> None:
    args = _receipt_args_for_regression_report(tmp_path)
    train_payload = {
        "step_metrics": [
            {
                "loss": 2.0,
                "seconds": 1.0,
                "ntokens": 128,
                "tokens_per_second": 999_999.0,
                "updated": True,
            }
        ],
        "kernel_dispatch": [
            {
                "op_name": "mamba3_mimo",
                "path": "path_b",
                "kernel_used": "metal_kernel_fwd_v1",
            }
        ],
    }

    report = m04_train_step.regression_report_payload(
        args,
        config=args,
        train_payload=train_payload,
        optimizer=adamw_identity(),
        memory_after={"peak_memory_bytes": 4096},
        tokens_per_second=999_999.0,
        status="ok",
    )

    claim_gate = report["throughput"]["claim_gate"]
    step_check = claim_gate["step_checks"][0]
    assert claim_gate["ok"] is False
    assert claim_gate["bogus_tok_sec_claim_detected"] is True
    assert claim_gate["reported_tokens_per_second_finite"] is True
    assert claim_gate["step_rates_consistent"] is False
    assert step_check["expected_tokens_per_second"] == 128.0
    assert step_check["reported_tokens_per_second"] == 999_999.0
    assert step_check["rate_consistent_with_ntokens_and_seconds"] is False
    assert report["gate_summary"]["tokens_per_second_claim_ok"] is False
    assert report["gate_summary"]["bogus_tok_sec_claim_detected"] is True


@pytest.mark.parametrize(
    (
        "dtype",
        "optimizer",
        "pattern",
        "depth",
        "dsa_a_layer_ranks",
        "kernel_dispatch",
        "expected_path_b",
        "expected_path_c",
        "expected_fallback",
    ),
    [
        (
            "bfloat16",
            "adamw",
            "M",
            "1",
            "",
            [
                {
                    "op_name": "mamba3_mimo",
                    "path": "path_b",
                    "kernel_used": "metal_kernel_fwd_v1",
                }
            ],
            True,
            False,
            None,
        ),
        (
            "fp8_path_c",
            "lion",
            "A",
            "1",
            "0",
            [
                {
                    "op_name": "mamba3_mimo",
                    "path": "path_c",
                    "kernel_used": "mamba3_mimo_path_c",
                },
                {
                    "op_name": "m2rnn",
                    "path": "path_c",
                    "kernel_used": "path_c_tilelang_dsl_packed",
                },
                {
                    "op_name": "sparse_mla",
                    "path": "path_c",
                    "kernel_used": "sparse_mla_fp8_path_c_apply",
                },
            ],
            False,
            True,
            None,
        ),
    ],
)
def test_regression_report_records_path_b_vs_path_c_receipt_gate_fields(
    tmp_path: Path,
    dtype: str,
    optimizer: str,
    pattern: str,
    depth: str,
    dsa_a_layer_ranks: str,
    kernel_dispatch: list[dict[str, Any]],
    expected_path_b: bool,
    expected_path_c: bool,
    expected_fallback: str | None,
) -> None:
    args = _receipt_args_for_regression_report(
        tmp_path,
        dtype=dtype,
        optimizer=optimizer,
        pattern=pattern,
        depth=depth,
        dsa_a_layer_ranks=dsa_a_layer_ranks,
    )
    train_payload = {
        "mean_loss": 1.5,
        "step_metrics": [
            {
                "loss": 2.0,
                "seconds": 0.5,
                "ntokens": 128,
                "tokens_per_second": 256.0,
                "updated": True,
            },
            {
                "loss": 1.0,
                "seconds": 0.25,
                "ntokens": 128,
                "tokens_per_second": 512.0,
                "updated": True,
            },
        ],
        "kernel_dispatch": kernel_dispatch,
    }

    report = m04_train_step.regression_report_payload(
        args,
        config=args,
        train_payload=train_payload,
        optimizer=adamw_identity(name="Lion" if optimizer == "lion" else "AdamW"),
        memory_after={"peak_memory_bytes": 123_456},
        tokens_per_second=384.0,
        status="ok",
    )

    route = report["route_dispatch"]
    summary = report["gate_summary"]
    assert report["dtype"]["requested"] == dtype
    assert report["optimizer"]["key"] == optimizer
    assert report["memory"]["peak_memory_bytes"] == 123_456
    assert route["path_b_observed"] is expected_path_b
    assert route["path_c_observed"] is expected_path_c
    assert route["fallback_reason"] == expected_fallback
    assert summary["dtype"] == dtype
    assert summary["optimizer"] == optimizer
    assert summary["path_b_observed"] is expected_path_b
    assert summary["path_c_observed"] is expected_path_c
    assert summary["fallback_reason"] == expected_fallback
    assert summary["tokens_per_second_claim_ok"] is True
    assert summary["bogus_tok_sec_claim_detected"] is False
    if dtype == "fp8_path_c":
        assert route["requested_path_c_ops"] == ["m2rnn", "mamba3_mimo", "sparse_mla"]
        assert route["unobserved_requested_path_c_ops"] == []
        assert route["producer_missing"] is False
        assert route["producer_unobserved"] is False
        assert report["fp8_path_c_producer_gate"]["ok"] is True
        assert summary["fp8_path_c_producer_status"] == (
            m04_train_step.FP8_PATH_C_NATIVE_PRODUCER_STATUS
        )
    else:
        assert route["requested_path_c_ops"] == []
        assert report["fp8_path_c_producer_gate"]["required"] is False
        assert summary["fp8_path_c_producer_status"] == "not_requested"


def test_checked_in_receipt_records_parquet_smoke_without_m0_4_claim() -> None:
    payload = json.loads(BASELINE_RECEIPT.read_text())

    assert_m04_receipt_contract(payload)
    assert payload["status"] == "ok"
    assert payload["full_m0_4_acceptance_claim"] is False
    assert payload["workload"]["synthetic"] is False
    assert payload["workload"]["data_format"] == "parquet"
    assert payload["workload"]["mode"] == "eager"
    assert payload["workload"]["steps_requested"] == 100
    assert payload["workload"]["batch_size"] == 1
    assert payload["workload"]["seq_len"] == 64
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is True
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is True
    assert payload["acceptance_gate"]["real_parquet_source_identity"]["ok"] is True
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is True
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["model_identity_ok"] is False
    assert payload["acceptance_gate"]["optimizer_identity_ok"] is True
    assert payload["acceptance_gate"]["adamw_ok"] is True
    assert payload["acceptance_gate"]["fp32_adamw_master_moments_ok"] is True
    assert payload["acceptance_gate"]["observed_adamw_master_moment_dtypes"]
    assert payload["acceptance_gate"]["grad_checkpoint_expectation_ok"] is False
    assert payload["acceptance_gate"]["m4_runtime_metadata_ok"] is True
    assert set(payload["acceptance_gate"]["full_local_gb10_quarter_gate_blockers"]) == {
        "grad_checkpoint_expectation_ok",
        "model_identity_ok",
    }
    assert payload["acceptance_gate"]["full_target_dataset_blocker"] is None
    assert payload["training"]["steps_completed"] == 100
    assert payload["training"]["all_finite"] is True
    assert payload["training"]["optimizer_updated"] is True
    assert payload["training"]["loss_decreased"] is True
    assert payload["training"]["loss_decrease_satisfied"] is True
    assert payload["training"]["final_loss"] < payload["training"]["initial_loss"]
    assert payload["training"]["optimizer"]["name"] == "AdamW"
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is False
    assert payload["model"]["name"] == "HybridTinyLM"
    assert payload["model"]["profile_matches_required"] is False
    assert payload["baseline_row"]["model"] == "HybridTinyLM"
    assert {item["id"] for item in payload["acceptance_blockers"]} == {
        "cppmega-mlx-t8f.4.local_gb10_quarter_gate",
    }


def test_synthetic_one_step_writes_finite_receipt(tmp_path: Path) -> None:
    output = tmp_path / "m04_train_step.json"
    result = run_script(*tiny_args(output))
    payload = load_json_result(result)

    assert output.exists()
    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert_regression_report_matches_payload(payload)
    assert_m04_20step_matrix_plan(payload)
    assert payload["status"] == "ok"
    assert payload["workload"]["synthetic"] is True
    assert payload["workload"]["data_format"] == "npz"
    assert payload["workload"]["model_profile"] == "hybrid_tiny"
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is False
    assert payload["training"]["steps_completed"] == 1
    assert payload["training"]["optimizer_updated"] is True
    assert payload["training"]["all_finite"] is True
    assert payload["training"]["loss_decrease_satisfied"] is True
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is False
    assert payload["acceptance_gate"]["real_parquet_source_identity"]["ok"] is False
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["full_target_dataset_blocker"]
    assert payload["training"]["final_loss"] > 0
    assert payload["training"]["step_metrics"][0]["updated"] is True
    interpretation = payload["timing"]["throughput_interpretation"]
    assert interpretation["reported_tokens_per_second_kind"] == (
        "loss_target_tokens_per_second"
    )
    assert interpretation["denominator"] == "sum(step_metrics[].ntokens)"
    assert interpretation["input_tokens_per_step"] == 4
    assert interpretation["nominal_target_tokens_per_step"] == 3
    assert interpretation["measured_target_tokens_per_step"] == [3]
    assert interpretation["workload_scope"] == "tiny_or_hybrid_smoke"
    assert interpretation["production_shape"] is False
    assert interpretation["excluded_from_step_timer"] == [
        "dataset construction",
        "next(batches) parquet/npz batch fetch",
        "model allocation",
        "optimizer initialization",
        "receipt JSON serialization",
        "post-step cache clear cadence",
    ]
    assert payload["memory"]["peak_memory_bytes"] is None or (
        payload["memory"]["peak_memory_bytes"] >= 0
    )


def test_throughput_interpretation_marks_short_local_gb10_sequence() -> None:
    config = m04_train_step.TrainHybridTinyConfig(
        model_profile="local_gb10_quarter",
        data_format="parquet",
        batch_size=1,
        seq_len=1024,
        steps=1,
        dtype="bfloat16",
        grad_checkpoint=True,
    )
    interpretation = m04_train_step.throughput_interpretation_payload(
        config,
        train_payload={"tokens_per_second": 480.0},
        step_metrics=[
            {
                "ntokens": 1023,
                "seconds": 2.0,
                "tokens_per_second": 511.5,
            }
        ],
        tokens_per_second_values=[511.5],
    )

    assert interpretation["workload_scope"] == "short_sequence_full_profile_smoke"
    assert interpretation["production_seq_len"] == 4096
    assert interpretation["production_shape"] is False
    assert interpretation["input_tokens_per_step"] == 1024
    assert interpretation["nominal_target_tokens_per_step"] == 1023
    assert interpretation["total_input_tokens"] == 1024
    assert interpretation["total_target_tokens"] == 1023
    assert interpretation["input_tokens_per_second"] == 512.0
    assert interpretation["target_tokens_per_second"] == 511.5
    assert "underfills" in interpretation["warning"]


def test_synthetic_grad_checkpoint_receipt_marks_gate_without_m0_4_claim(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_train_step.json"
    result = run_script(*tiny_args(output), "--grad-checkpoint")
    payload = load_json_result(result)

    assert output.exists()
    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert payload["status"] == "ok"
    assert payload["workload"]["model_profile"] == "hybrid_tiny"
    assert payload["workload"]["grad_checkpoint"] is True
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is True
    assert payload["training"]["grad_checkpoint"]["expectation_satisfied"] is True
    assert payload["acceptance_gate"]["grad_checkpoint_expectation_ok"] is True
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert {
        "local_gb10_quarter_preflight_ok",
        "model_identity_ok",
    }.issubset(set(payload["acceptance_gate"]["full_local_gb10_quarter_gate_blockers"]))


def test_require_loss_decrease_fails_single_step_but_writes_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_train_step.json"
    result = run_script(*tiny_args(output), "--require-loss-decrease")

    assert result.returncode == 2
    assert not result.stderr
    payload = json.loads(result.stdout)
    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert payload["status"] == "failed"
    assert payload["training"]["steps_completed"] == 1
    assert payload["training"]["loss_decrease_required"] is True
    assert payload["training"]["loss_decrease_satisfied"] is False


def test_missing_dataset_dry_run_reports_blocked_receipt(tmp_path: Path) -> None:
    output = tmp_path / "m04_train_step.json"
    missing = tmp_path / "missing.parquet"

    result = run_script(
        "--data-path",
        str(missing),
        "--dry-run-json",
        "--output",
        str(output),
    )
    payload = load_json_result(result)

    assert json.loads(output.read_text()) == payload
    assert payload["status"] == "blocked"
    assert_regression_report_matches_payload(payload)
    assert_m04_20step_matrix_plan(payload)
    assert payload["regression_report"]["fallback_reason"].startswith(
        "missing_dataset:"
    )
    assert payload["blockers"][0]["type"] == "missing_dataset"
    assert payload["training"]["steps_completed"] == 0
    assert payload["workload"]["data_path"] == str(missing)
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is False
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_fp8_path_c_training_dtype_route_blocks_missing_sparse_mla_producer(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_train_step.json"

    result = run_script(
        *tiny_args(output),
        "--dtype",
        "fp8_path_c",
    )

    assert result.returncode == 2, result.stderr
    assert not result.stderr
    payload = json.loads(result.stdout)
    assert json.loads(output.read_text()) == payload
    assert payload["status"] == "blocked"
    assert_regression_report_matches_payload(payload)
    assert_m04_20step_matrix_plan(payload)
    assert payload["blockers"][0]["type"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert payload["blockers"][0]["reason"].startswith("fp8_path_c requested")
    assert payload["training"]["steps_completed"] == 0
    assert payload["training"]["kernel_dispatch"] == []
    assert payload["workload"]["dtype"] == "fp8_path_c"
    assert payload["workload"]["optimizer"]["key"] == "adamw"
    precision_route = payload["workload"]["precision_route"]
    assert precision_route["requested"] == "fp8_path_c"
    assert precision_route["kind"] == "fp8_path_c"
    assert precision_route["status"] == m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    assert precision_route["blocker_type"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert precision_route["carrier_dtype"] == "bfloat16"
    assert precision_route["native_fp8_producer_status"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert precision_route["kernel_surface_status"] == (
        m04_train_step.FP8_PATH_C_KERNEL_SURFACE_STATUS
    )
    assert precision_route["kernel_surface_available"] is True
    assert precision_route["full_end_to_end_training_available"] is False
    assert precision_route["bridge_target"] == m04_train_step.FP8_PATH_C_BRIDGE_TARGET
    assert precision_route["bridge_status"] == m04_train_step.FP8_PATH_C_BRIDGE_STATUS
    assert precision_route["zero_copy_required"] is True
    assert precision_route["large_tensor_staging_allowed"] is False
    assert precision_route["hidden_wrapper_quantization_allowed"] is False
    assert precision_route["kernel_boundary_quantization_allowed"] is False
    assert precision_route["prepared_buffers_configured"] is False
    producer = precision_route["sparse_mla_fp8_producer"]
    assert producer["configured"] is False
    assert producer["prepared_buffers_configured"] is False
    assert producer["status"] == m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    assert producer["required_prepared_buffers"] == [
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    ]
    assert producer["hidden_wrapper_quantization_allowed"] is False
    assert producer["kernel_boundary_quantization_allowed"] is False
    assert producer["producer_stage"] == m04_train_step.FP8_PATH_C_PRODUCER_STAGE
    assert producer["producer_quantization"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_QUANTIZATION
    )
    route = payload["training"]["fp8_path_c_training_route"]
    assert route["requested"] is True
    assert route["status"] == m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    assert route["blocker_type"] == m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    assert route["reason"].startswith("producer_missing:")
    assert route["carrier_dtype"] == "bfloat16"
    assert route["native_fp8_producer_status"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert route["sparse_mla_fp8_producer"] == producer
    assert route["kernel_surface_status"] == (
        m04_train_step.FP8_PATH_C_KERNEL_SURFACE_STATUS
    )
    assert route["kernel_surface_available"] is True
    assert route["full_end_to_end_training_available"] is False
    assert route["end_to_end_training_status"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert route["direct_mx_array_artifact_call_status"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert route["bridge_target"] == m04_train_step.FP8_PATH_C_BRIDGE_TARGET
    assert route["bridge_status"] == m04_train_step.FP8_PATH_C_BRIDGE_STATUS
    assert route["bridge_evidence"] == {
        "mlx_array_exports_dlpack": True,
        "mlx_public_from_dlpack_available": False,
        "tvm_ffi_from_dlpack_available": True,
        "mlx_metal_dlpack_device": "kDLMetal:0",
        "tvm_from_dlpack_device": "metal:0",
        "native_mlx_array_wrapper_linked": True,
        "native_tvm_ffi_graph_outputs": True,
        "dlpack_used_for_path_c_graph_bridge": False,
        "standalone_mlx_to_tvm_metal_kernel_verified": True,
        "m04_bridge_wired": True,
    }
    assert route["zero_copy_required"] is True
    assert route["large_tensor_staging_allowed"] is False
    assert route["hidden_wrapper_quantization_allowed"] is False
    assert route["kernel_boundary_quantization_allowed"] is False
    assert route["prepared_buffers_configured"] is False
    assert route["hidden_dtype_cast_allowed"] is False
    assert route["hidden_shape_staging_allowed"] is False
    assert route["fallback_to_path_b_allowed"] is False
    assert route["selected_action"] == "fail_closed_producer_missing"
    assert route["kernel_policy_env"] == {
        "CPPMEGA_KERNEL_PATH__MAMBA3_MIMO": "path_c",
        "CPPMEGA_KERNEL_PATH__M2RNN": "path_c",
        "CPPMEGA_KERNEL_PATH__SPARSE_MLA": "path_c",
    }
    assert {
        "fp8_scaled_vecmat_path_c",
        "mamba3_mimo_path_c",
        "m2rnn_path_c",
        "sparse_mla_fp8_path_c_apply",
        "matmul_tl_fp8_scaled_matmul",
    } == {surface["name"] for surface in route["available_path_c_surfaces"]}
    surfaces = {
        surface["name"]: surface for surface in route["available_path_c_surfaces"]
    }
    assert surfaces["matmul_tl_fp8_scaled_matmul"]["kernel_surface_available"] is True
    assert surfaces["mamba3_mimo_path_c"]["training_surface"] is True
    assert surfaces["mamba3_mimo_path_c"]["fp8_route_auto_selected"] is True
    assert surfaces["m2rnn_path_c"]["training_surface"] is True
    assert surfaces["m2rnn_path_c"]["fallback_to_path_b_allowed"] is False
    assert surfaces["m2rnn_path_c"]["fp8_route_auto_selected"] is True
    assert surfaces["sparse_mla_fp8_path_c_apply"]["training_surface"] is False
    assert surfaces["sparse_mla_fp8_path_c_apply"]["producer_required"] is True
    assert surfaces["sparse_mla_fp8_path_c_apply"]["producer_status"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert (
        surfaces["sparse_mla_fp8_path_c_apply"]["backward_surface"]
        == "native_tvm_ffi_graph_output_scatter"
    )
    assert (
        "FP8 parameter/weight producers that create the required dtype/layout "
        "before matmul kernel boundaries" in route["missing_training_surfaces"]
    )
    assert (
        "absorbed MLA producer split for NoPE/RoPE KV layout and calibrated "
        "separate K/V scale lifecycle" in route["missing_training_surfaces"]
    )
    assert (
        route["higher_level_owner"]["sparse_mla_fp8_next_owner"]
        == m04_train_step.FP8_PATH_C_PRODUCER_OWNER
    )
    assert "without DSA Sparse-MLA producer" in (
        route["higher_level_owner"]["current_m04_route_owner"]
    )
    dispatch_report = payload["regression_report"]["route_dispatch"]
    assert dispatch_report["requested_path_c_ops"] == ["m2rnn", "mamba3_mimo", "sparse_mla"]
    assert dispatch_report["observed_path_c_ops"] == []
    assert dispatch_report["unobserved_requested_path_c_ops"] == [
        "m2rnn",
        "mamba3_mimo",
        "sparse_mla",
    ]
    assert dispatch_report["producer_missing"] is True
    assert dispatch_report["fp8_sparse_mla_producer"] == producer
    assert dispatch_report["fallback_detected"] is True
    assert dispatch_report["fallback_reason"].startswith("producer_missing:")
    producer_gate = payload["regression_report"]["fp8_path_c_producer_gate"]
    assert producer_gate["required"] is True
    assert producer_gate["ok"] is False
    assert producer_gate["status"] == m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    assert producer_gate["fail_closed"] is True
    assert producer_gate["producer"] == producer
    assert producer_gate["reason"].startswith("producer_missing:")


def test_fp8_path_c_dsa_attention_route_metadata_is_configured(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")

    producer = m04_train_step.sparse_mla_fp8_producer_payload(config)
    producer_gate = m04_train_step.fp8_path_c_producer_gate_payload(config)
    route = m04_train_step.fp8_path_c_training_route_payload(config)

    assert config.dsa_a_layer_ranks == (0,)
    assert producer["configured"] is True
    assert producer["status"] == m04_train_step.FP8_PATH_C_NATIVE_PRODUCER_STATUS
    assert producer["dsa_layer_numbers"] == [1]
    assert producer["owner"] == m04_train_step.FP8_PATH_C_PRODUCER_OWNER
    assert producer["prepared_buffers_configured"] is True
    assert producer["producer_stage"] == m04_train_step.FP8_PATH_C_PRODUCER_STAGE
    assert producer["producer_quantization"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_QUANTIZATION
    )
    assert producer["hidden_wrapper_quantization_allowed"] is False
    assert producer["kernel_boundary_quantization_allowed"] is False
    assert producer["required_prepared_buffers"] == [
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    ]
    assert route["status"] == m04_train_step.FP8_PATH_C_E2E_TRAINING_STATUS
    assert route["blocker_type"] is None
    assert route["prepared_buffers_configured"] is True
    assert route["hidden_wrapper_quantization_allowed"] is False
    assert route["kernel_boundary_quantization_allowed"] is False
    assert route["full_end_to_end_training_available"] is True
    assert route["direct_mx_array_artifact_call_status"] == "m04_uses_model_graph_route"
    assert route["selected_action"] == "run_path_c_training_route"
    assert producer_gate["required"] is True
    assert producer_gate["ok"] is True
    assert producer_gate["status"] == m04_train_step.FP8_PATH_C_NATIVE_PRODUCER_STATUS
    assert producer_gate["fail_closed"] is False
    assert producer_gate["producer"] == producer


def test_fp8_path_c_local_gb10_profile_uses_model_factory_dsa_producer() -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--model-profile",
            "local_gb10_quarter",
            "--dtype",
            "fp8_path_c",
        ]
    )

    producer = m04_train_step.sparse_mla_fp8_producer_payload(args)
    producer_gate = m04_train_step.fp8_path_c_producer_gate_payload(args)

    assert producer["configured"] is True
    assert producer["route_source"] == (
        "cppmega_mlx.recipes.model_factory.local_gb10_quarter"
    )
    assert producer["status"] == m04_train_step.FP8_PATH_C_NATIVE_PRODUCER_STATUS
    assert producer["dsa_a_layer_ranks"] == [1, 2, 3]
    assert producer["dsa_layer_numbers"] == [5, 9, 13]
    assert producer["prepared_buffers_configured"] is True
    assert producer["hidden_wrapper_quantization_allowed"] is False
    assert producer["kernel_boundary_quantization_allowed"] is False
    assert producer["reason"] is None
    assert producer_gate["required"] is True
    assert producer_gate["ok"] is True
    assert producer_gate["status"] == m04_train_step.FP8_PATH_C_NATIVE_PRODUCER_STATUS
    assert producer_gate["producer"] == producer


def assert_local_gb10_metadata_dry_run_contract(payload: dict[str, Any]) -> None:
    status = payload["status"]
    assert status in {"dry_run", "failed"}
    assert payload["receipt_schema_version"] == 1
    assert payload["receipt_scope"] == "local_mlx_m04_train_step"
    assert payload["local_only"] is True
    assert payload["gb10_training_correctness_claim"] is False
    assert payload["m4_vs_gb10_throughput_parity_claim"] is False
    assert payload["full_m0_4_acceptance_claim"] is False
    assert "blockers" not in payload
    assert_regression_report_matches_payload(payload)
    assert_m04_20step_matrix_plan(payload)
    assert {item["id"] for item in payload["acceptance_blockers"]} == {
        "cppmega-mlx-t8f.4.local_gb10_quarter_gate",
    }
    assert payload["workload"]["model_profile"] == "local_gb10_quarter"
    assert payload["workload"]["mode"] == "metadata_only_no_forward_no_training"
    assert payload["training"]["steps_completed"] == 0
    assert payload["training"]["optimizer_updated"] is False
    assert payload["training"]["losses"] == []
    assert payload["training"]["optimizer"]["update_observed"] is False
    assert payload["training"]["optimizer"]["master_moment_evidence"]["skipped"] is True
    assert payload["model"]["source"] is None
    assert payload["model"]["name"] is None
    assert payload["model"]["observed_source"] is None
    assert payload["model"]["observed_name"] is None
    assert payload["model"]["required_source"] == REQUIRED_MODEL_SOURCE
    assert payload["model"]["required_name"] == "local_gb10_quarter"
    assert payload["model"]["requested_profile"] == "local_gb10_quarter"
    assert payload["model"]["profile"] is None
    assert payload["model"]["requested_profile_matches_required"] is True
    assert payload["model"]["profile_matches_required"] is False
    assert payload["model"]["metadata_only"] is True
    assert payload["model"]["forward_executed"] is False
    assert payload["model"]["training_executed"] is False
    assert payload["baseline_row"]["model"] == "metadata_only_no_observed_model"
    assert payload["baseline_row"]["tokens_per_second"] == 0.0
    assert payload["baseline_row"]["local_only"] is True
    assert payload["baseline_row"]["gb10_parity_claim"] is False
    gate = payload["acceptance_gate"]
    assert gate["required_model_profile"] == "local_gb10_quarter"
    assert gate["observed_model_name"] is None
    assert gate["observed_model_source"] is None
    assert gate["model_identity_ok"] is False
    assert gate["optimizer_update_ok"] is False
    assert gate["adamw_ok"] is False
    assert gate["full_target_dataset_100_step_completed"] is False
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert {
        "real_parquet_source_identity_ok",
        "target_parquet_path_ok",
        "dataset_name_ok",
        "dataset_format_ok",
        "model_identity_ok",
        "optimizer_update_ok",
        "loss_decrease_ok",
        "loss_fields_ok",
        "all_finite_ok",
    }.issubset(set(gate["full_local_gb10_quarter_gate_blockers"]))


def test_local_gb10_quarter_dry_run_is_metadata_only_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_route(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("training and dry-run routes must not be called")

    def fail_allocation_probe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("full local_gb10_quarter allocation must be opt-in")

    monkeypatch.setattr(m04_train_step, "dry_run_payload", fail_route)
    monkeypatch.setattr(m04_train_step, "train_hybrid_tiny", fail_route)
    monkeypatch.setattr(m04_train_step, "local_gb10_quarter", fail_allocation_probe)
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--dry-run-json",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert exit_code == 0
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is False
    assert payload["local_gb10_quarter_preflight"]["allocation_attempted"] is False
    assert payload["local_gb10_quarter_preflight"]["allocation_mode"] == (
        "allocation_free_preflight"
    )
    assert payload["acceptance_gate"]["local_gb10_quarter_preflight_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_quarter_dry_run_require_loss_decrease_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_route(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("training and dry-run routes must not be called")

    monkeypatch.setattr(m04_train_step, "dry_run_payload", fail_route)
    monkeypatch.setattr(m04_train_step, "train_hybrid_tiny", fail_route)
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--dry-run-json",
            "--require-loss-decrease",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert exit_code == 2
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["status"] == "failed"
    assert payload["training"]["loss_decreased"] is False
    assert payload["training"]["loss_decrease_required"] is True
    assert payload["training"]["loss_decrease_satisfied"] is False
    assert payload["acceptance_gate"]["loss_decrease_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_quarter_dry_run_cli_writes_requested_output_only(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_local_gb10_metadata.json"
    baseline_before = (
        BASELINE_RECEIPT.read_text(encoding="utf-8")
        if BASELINE_RECEIPT.exists()
        else None
    )

    result = run_script(
        "--synthetic",
        "--model-profile",
        "local_gb10_quarter",
        "--dry-run-json",
        "--output",
        str(output),
        "--json",
    )

    payload = load_json_result(result)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert_local_gb10_metadata_dry_run_contract(payload)
    if baseline_before is None:
        assert not BASELINE_RECEIPT.exists()
    else:
        assert BASELINE_RECEIPT.read_text(encoding="utf-8") == baseline_before


def test_local_gb10_quarter_dry_run_records_non_default_optimizer_metadata(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_local_gb10_lion.json"
    result = run_script(
        "--synthetic",
        "--model-profile",
        "local_gb10_quarter",
        "--dry-run-json",
        "--optimizer",
        "lion",
        "--output",
        str(output),
        "--json",
    )

    payload = load_json_result(result)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["workload"]["optimizer"] == {
        "requested": "lion",
        "key": "lion",
        "quant_scheme": "dynamic_int8_v1",
        "source": "cli",
    }
    optimizer = payload["training"]["optimizer"]
    assert optimizer["name"] == "Lion"
    assert optimizer["key"] == "lion"
    assert optimizer["class"] == "cppmega_mlx.training.optimizers.LionFP32Moments"
    assert optimizer["adamw"] is False
    assert optimizer["master_moment_evidence"]["skipped"] is True
    assert payload["acceptance_gate"]["observed_optimizer_name"] == "Lion"
    assert payload["acceptance_gate"]["optimizer_identity_ok"] is False
    assert payload["acceptance_gate"]["adamw_ok"] is False


def test_non_default_optimizer_is_blocked_outside_local_gb10_route(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_hybrid_lion.json"
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--optimizer",
            "lion",
            "--output",
            str(output),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["type"] == "unsupported_optimizer_route"
    assert payload["workload"]["optimizer"]["key"] == "lion"
    assert payload["training"]["optimizer"]["name"] == "Lion"


def test_local_gb10_quarter_training_routes_to_monkeypatchable_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    route_calls: list[tuple[m04_train_step.TrainHybridTinyConfig, Path]] = []

    def fail_train(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("HybridTinyLM training route must not be called")

    def fake_local_gb10_route(
        _args: Any,
        *,
        config: m04_train_step.TrainHybridTinyConfig,
        data_path: Path,
    ) -> tuple[dict[str, Any], int]:
        route_calls.append((config, data_path))
        return (
            m04_train_step.blocked_receipt(
                _args,
                "unit-test local_gb10_quarter route called",
                "unit_test_route_called",
            ),
            2,
        )

    monkeypatch.setattr(m04_train_step, "train_hybrid_tiny", fail_train)
    monkeypatch.setattr(
        m04_train_step,
        "run_local_gb10_quarter_training",
        fake_local_gb10_route,
    )
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert len(route_calls) == 1
    config, data_path = route_calls[0]
    assert config.model_profile == "local_gb10_quarter"
    assert config.grad_checkpoint is False
    assert config.data_format == "npz"
    assert data_path.suffix == ".npz"
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["full_m0_4_acceptance_claim"] is False
    assert payload["blockers"][0]["type"] == "unit_test_route_called"
    assert (
        payload["blockers"][0]["reason"] == "unit-test local_gb10_quarter route called"
    )
    assert payload["workload"]["model_profile"] == "local_gb10_quarter"
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["training"]["steps_completed"] == 0
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["model_identity_ok"] is False


def test_local_gb10_quarter_grad_checkpoint_routes_to_monkeypatchable_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    route_calls: list[tuple[m04_train_step.TrainHybridTinyConfig, Path]] = []

    def fail_train(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("HybridTinyLM training route must not be called")

    def fake_local_gb10_route(
        _args: Any,
        *,
        config: m04_train_step.TrainHybridTinyConfig,
        data_path: Path,
    ) -> tuple[dict[str, Any], int]:
        route_calls.append((config, data_path))
        return (
            m04_train_step.blocked_receipt(
                _args,
                "unit-test local_gb10_quarter grad-checkpoint route called",
                "unit_test_route_called",
            ),
            2,
        )

    monkeypatch.setattr(m04_train_step, "train_hybrid_tiny", fail_train)
    monkeypatch.setattr(
        m04_train_step,
        "run_local_gb10_quarter_training",
        fake_local_gb10_route,
    )
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--grad-checkpoint",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert len(route_calls) == 1
    config, data_path = route_calls[0]
    assert config.model_profile == "local_gb10_quarter"
    assert config.grad_checkpoint is True
    assert config.data_format == "npz"
    assert data_path.suffix == ".npz"
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["type"] == "unit_test_route_called"
    assert payload["blockers"][0]["reason"] == (
        "unit-test local_gb10_quarter grad-checkpoint route called"
    )
    assert payload["workload"]["model_profile"] == "local_gb10_quarter"
    assert payload["workload"]["grad_checkpoint"] is True
    assert payload["training"]["steps_completed"] == 0
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is True
    assert payload["acceptance_gate"]["grad_checkpoint_expectation_ok"] is True
    assert payload["acceptance_gate"]["model_identity_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_quarter_dry_run_with_allocation_probe_is_preflight_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe_called = False

    def fake_probe() -> dict[str, Any]:
        nonlocal probe_called
        probe_called = True
        return canonical_allocation_probe()

    def fail_route(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("training and dry-run routes must not be called")

    monkeypatch.setattr(
        m04_train_step, "probe_local_gb10_quarter_allocation", fake_probe
    )
    monkeypatch.setattr(m04_train_step, "dry_run_payload", fail_route)
    monkeypatch.setattr(m04_train_step, "train_hybrid_tiny", fail_route)
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--probe-local-gb10-quarter-allocation",
            "--dry-run-json",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert probe_called is True
    assert exit_code == 0
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is True
    preflight = payload["local_gb10_quarter_preflight"]
    assert preflight["allocation_attempted"] is True
    assert preflight["allocation_ready"] is True
    assert preflight["allocation_mode"] == "full_profile_allocation_probe"
    assert preflight["allocation_probe"]["forward_executed"] is False
    assert preflight["allocation_probe"]["training_executed"] is False
    assert preflight["ok"] is True
    assert payload["acceptance_gate"]["local_gb10_quarter_preflight_ok"] is True
    assert payload["acceptance_gate"]["model_identity_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_allocation_probe_success_is_preflight_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe() -> dict[str, Any]:
        return canonical_allocation_probe()

    monkeypatch.setattr(
        m04_train_step, "probe_local_gb10_quarter_allocation", fake_probe
    )
    args = m04_train_step.build_parser().parse_args(
        [
            "--probe-local-gb10-quarter-allocation",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    preflight = m04_train_step.local_gb10_quarter_preflight_from_args(args)
    gate = local_gb10_gate(
        model_name="HybridTinyLM",
        grad_checkpoint=grad_checkpoint_identity(enabled=False),
        local_gb10_quarter_preflight=preflight,
    )

    assert preflight["allocation_attempted"] is True
    assert preflight["allocation_ready"] is True
    assert preflight["allocation_mode"] == "full_profile_allocation_probe"
    assert preflight["allocation_probe"]["forward_executed"] is False
    assert preflight["allocation_probe"]["training_executed"] is False
    assert preflight["ok"] is True
    assert preflight["blockers"] == []
    assert gate["local_gb10_quarter_preflight_ok"] is True
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert {
        "model_identity_ok",
        "grad_checkpoint_expectation_ok",
    }.issubset(set(gate["full_local_gb10_quarter_gate_blockers"]))


def test_local_gb10_allocation_probe_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe() -> dict[str, Any]:
        return canonical_allocation_probe(
            status="blocked",
            allocation_ready=False,
            memory_after={"active_memory_bytes": 0},
            error_type="RuntimeError",
            error="synthetic allocation failure",
        )

    monkeypatch.setattr(
        m04_train_step, "probe_local_gb10_quarter_allocation", fake_probe
    )
    args = m04_train_step.build_parser().parse_args(
        [
            "--probe-local-gb10-quarter-allocation",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    preflight = m04_train_step.local_gb10_quarter_preflight_from_args(args)
    gate = local_gb10_gate(local_gb10_quarter_preflight=preflight)

    assert preflight["allocation_attempted"] is True
    assert preflight["allocation_ready"] is False
    assert preflight["allocation_mode"] == "full_profile_allocation_probe"
    assert preflight["allocation_probe"]["error_type"] == "RuntimeError"
    assert preflight["allocation_probe"]["error"] == "synthetic allocation failure"
    assert preflight["ok"] is False
    assert preflight["blockers"] == ["allocation_ready"]
    assert gate["local_gb10_quarter_preflight_ok"] is False
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert (
        "local_gb10_quarter_preflight_ok"
        in (gate["full_local_gb10_quarter_gate_blockers"])
    )


def test_applied_memory_limit_api_path_preserves_actual_fallback_path() -> None:
    payload = {
        "applied": True,
        "metal_limit_api_path": "mx.set_memory_limit",
    }

    assert applied_memory_limit_api_path_from_payload(payload) == "mx.set_memory_limit"


class _Bf16Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.ones((2, 2), dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.sum(x @ self.weight)


def _run_bf16_probe_update(
    optimizer: optim.Optimizer,
) -> tuple[_Bf16Probe, dict[str, str]]:
    model = _Bf16Probe()

    def loss_fn(probe: _Bf16Probe, x: mx.array) -> mx.array:
        return probe(x)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    _, grads = loss_and_grad(model, mx.ones((2, 2), dtype=mx.bfloat16))
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return model, collect_adamw_moment_dtypes(optimizer.state)


@pytest.mark.parametrize(
    ("optimizer_key", "expected_key", "expected_name", "quantized"),
    [
        ("adamw", "adamw", "AdamW", False),
        ("muon_adamw", "muon_adamw", "MuonAdamW", False),
        ("nam56r", "muon_adamw", "MuonAdamW", False),
        ("lion", "lion", "Lion", False),
        ("adam8bit", "adam8bit", "Adam8bit", True),
        ("lion8bit", "lion8bit", "Lion8bit", True),
        ("int8", "int8", "MuonAdamWInt8", True),
    ],
)
def test_local_gb10_optimizer_selector_initializes_supported_variants(
    optimizer_key: str,
    expected_key: str,
    expected_name: str,
    quantized: bool,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--model-profile",
            "local_gb10_quarter",
            "--optimizer",
            optimizer_key,
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=GB10_SAMPLE)
    model = _Bf16Probe()

    optimizer = m04_train_step.make_local_gb10_optimizer(
        args,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    identity = m04_train_step.optimizer_identity_for_selected_optimizer(
        args,
        config,
        optimizer,
        model,
        optimizer_updated=True,
    )

    assert identity["key"] == expected_key
    assert identity["name"] == expected_name
    assert identity["quantized_state"] is quantized
    assert identity["learning_rate"] == config.learning_rate
    assert identity["weight_decay"] == config.weight_decay
    assert identity["variant"]["requested"] == optimizer_key
    expected_quant_scheme = None if optimizer_key == "adamw" else "dynamic_int8_v1"
    assert identity["variant"]["quant_scheme"] == expected_quant_scheme
    if optimizer_key == "adamw":
        assert identity["adamw"] is True
        assert identity["name_matches_required"] is True
        assert identity["master_moment_dtype_ok"] is True
    else:
        assert identity["adamw"] is False
        assert identity["name_matches_required"] is False
        assert identity["master_moment_dtype_ok"] is False
        assert identity["state_evidence"]["state_dtype_breakdown_bytes"]


def test_stock_mlx_adamw_uses_bf16_moments_for_bf16_params() -> None:
    model, moment_dtypes = _run_bf16_probe_update(
        optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    )

    assert dtype_name(model.weight) == "bfloat16"
    assert moment_dtypes == {
        "weight/m": "bfloat16",
        "weight/v": "bfloat16",
    }


def test_repo_local_adamw_keeps_bf16_params_with_fp32_moments() -> None:
    optimizer = make_adamw(learning_rate=1e-3, weight_decay=0.0)
    model, moment_dtypes = _run_bf16_probe_update(optimizer)

    assert isinstance(optimizer, AdamWFP32Moments)
    assert dtype_name(model.weight) == "bfloat16"
    assert moment_dtypes == {
        "weight/m": "float32",
        "weight/v": "float32",
    }


def test_repo_local_adamw_weight_decay_preserves_fp32_moments() -> None:
    optimizer = make_adamw(learning_rate=1e-3, weight_decay=0.1)
    model, moment_dtypes = _run_bf16_probe_update(optimizer)

    assert isinstance(optimizer, AdamWFP32Moments)
    assert dtype_name(model.weight) == "bfloat16"
    assert bool(mx.all(mx.isfinite(model.weight)).item())
    assert moment_dtypes == {
        "weight/m": "float32",
        "weight/v": "float32",
    }


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"applied": False, "metal_limit_api_path": "mx.set_memory_limit"},
        {"applied": True},
        {"applied": True, "metal_limit_api_path": ""},
    ],
)
def test_applied_memory_limit_api_path_requires_applied_recorded_path(
    payload: Any,
) -> None:
    assert applied_memory_limit_api_path_from_payload(payload) is None


def test_real_parquet_dry_run_uses_gb10_sample_receipt(tmp_path: Path) -> None:
    if not GB10_SAMPLE.exists():
        pytest.skip(f"GB10 parquet sample is not present: {GB10_SAMPLE}")

    sample_path = tmp_path / "clang_semantic_4k_v10_head.parquet"
    copy_real_parquet_head(GB10_SAMPLE, sample_path)
    output = tmp_path / "m04_train_step.json"
    result = run_script(
        "--data-path",
        str(sample_path),
        "--dry-run-json",
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--seq-len",
        "64",
        "--hidden-size",
        "8",
        "--pattern",
        "M",
        "--depth",
        "1",
        "--output",
        str(output),
    )
    payload = load_json_result(result)

    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert payload["status"] == "dry_run"
    assert payload["workload"]["synthetic"] is False
    assert payload["workload"]["data_format"] == "parquet"
    assert payload["workload"]["data_path"] == str(sample_path)
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is False
    assert payload["acceptance_gate"]["real_parquet_source_identity"]["ok"] is False
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["full_target_dataset_blocker"]
    assert payload["dataset"]["metadata"]["source_format"] == "parquet"
    assert payload["dataset"]["dataset_receipt"]["source_dataset_name"] == (
        "clang_semantic_4k_v10"
    )
    assert payload["training"]["steps_completed"] == 0


def target_dataset_receipt(
    *,
    source_path: str = TARGET_PARQUET,
    source_format: str = "parquet",
    source_dataset_name: str = "clang_semantic_4k_v10",
) -> dict[str, Any]:
    return {
        "path": source_path,
        "dataset_receipt": {
            "source_path": source_path,
            "source_format": source_format,
            "source_dataset_name": source_dataset_name,
        },
        "metadata": {"source_format": source_format},
    }


def adamw_moment_evidence(*, moment_dtype: str = "float32") -> dict[str, Any]:
    return {
        "required_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "observed_parameter_dtype": REQUIRED_DTYPE,
        "observed_moment_dtypes": {
            "weight/m": moment_dtype,
            "weight/v": moment_dtype,
        },
        "optimizer_class": ADAMW_FP32_MOMENTS_CLASS,
        "optimizer_base_class": ADAMW_BASE_CLASS,
        "state_keys": ["learning_rate", "step", "weight"],
        "ok": moment_dtype == REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
    }


def adamw_identity(
    *,
    name: str = "AdamW",
    updated: bool = True,
    moment_dtype: str = "float32",
) -> dict[str, Any]:
    master_moment_evidence = adamw_moment_evidence(moment_dtype=moment_dtype)
    return {
        **OBSERVED_OPTIMIZER_IDENTITY,
        "name": name,
        "required_name": "AdamW",
        "name_matches_required": name == "AdamW",
        "adamw": name == "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "update_observed": updated,
        "required_master_moment_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "master_moment_evidence": master_moment_evidence,
        "master_moment_dtype_ok": master_moment_evidence["ok"],
    }


def grad_checkpoint_identity(*, enabled: bool = True) -> dict[str, Any]:
    return {
        "required": True,
        "observed_enabled": enabled,
        "source": GRAD_CHECKPOINT_EXPECTATION["source"],
        "expectation_satisfied": enabled,
    }


def m4_device_metadata(*, device_name: str = "Apple M4 Max") -> dict[str, Any]:
    return {
        "machine": "arm64",
        "metal_available": True,
        "platform": "macOS-26.4.1-arm64-arm-64bit-Mach-O",
        "mlx_device_info": {
            "device_name": device_name,
            "memory_size": 137438953472,
        },
    }


def local_gb10_model_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "profile": "local_gb10_quarter",
        **REQUIRED_MODEL_GEOMETRY,
    }
    config["mtp"] = dict(REQUIRED_MODEL_GEOMETRY["mtp"])
    config.update(overrides)
    return config


def resolved_local_gb10_preflight(**overrides: Any) -> dict[str, Any]:
    allocation_probe = canonical_allocation_probe()
    preflight = local_gb10_quarter_preflight_payload(
        allocation_attempted=True,
        allocation_ready=True,
        allocation_mode="full_profile_allocation_probe",
        allocation_probe=allocation_probe,
    )
    preflight["tokenizer_contract"] = {
        **preflight["tokenizer_contract"],
        "resolved": True,
    }
    preflight["ok"] = True
    preflight["blockers"] = []
    preflight.update(overrides)
    return preflight


def local_gb10_gate(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "data_path": target_dataset_path(),
        "data_format": "parquet",
        "dtype": REQUIRED_DTYPE,
        "dataset": target_dataset_receipt(),
        "steps_requested": 100,
        "steps_completed": 100,
        "loss_decreased": True,
        "all_finite": True,
        "optimizer_updated": True,
        "model_name": "local_gb10_quarter",
        "model_source": REQUIRED_MODEL_SOURCE,
        "model_config": local_gb10_model_config(),
        "optimizer": adamw_identity(),
        "grad_checkpoint": grad_checkpoint_identity(),
        "device": m4_device_metadata(),
        "local_gb10_quarter_preflight": resolved_local_gb10_preflight(),
    }
    payload.update(overrides)
    return acceptance_gate_payload(**payload)


def test_acceptance_gate_accepts_complete_local_gb10_quarter_evidence() -> None:
    gate = local_gb10_gate()

    assert gate["real_parquet_source_identity"]["ok"] is True
    assert gate["full_target_dataset_100_step_completed"] is True
    assert gate["dtype_ok"] is True
    assert gate["local_gb10_quarter_preflight_ok"] is True
    assert gate["local_gb10_quarter_preflight"]["source"] == REQUIRED_MODEL_SOURCE
    assert gate["local_gb10_quarter_preflight"]["required_geometry"] == (
        REQUIRED_MODEL_GEOMETRY
    )
    assert gate["local_gb10_quarter_preflight"]["profile_geometry"] == (
        REQUIRED_MODEL_GEOMETRY
    )
    assert gate["model_identity_ok"] is True
    assert gate["optimizer_identity_ok"] is True
    assert gate["required_adamw_master_moment_dtype"] == "float32"
    assert gate["observed_adamw_master_moment_dtypes"] == {
        "weight/m": "float32",
        "weight/v": "float32",
    }
    assert gate["fp32_adamw_master_moments_ok"] is True
    assert gate["adamw_ok"] is True
    assert gate["grad_checkpoint_expectation_ok"] is True
    assert gate["m4_runtime_metadata_ok"] is True
    assert gate["full_local_gb10_quarter_gate_completed"] is True
    assert gate["full_local_gb10_quarter_gate_blockers"] == []


@pytest.mark.parametrize(
    ("overrides", "failed_checks"),
    [
        (
            {
                "dataset": target_dataset_receipt(
                    source_path="/tmp/fake/clang_semantic_4k_v10/val_00000.parquet"
                )
            },
            {"real_parquet_source_identity_ok", "target_parquet_path_ok"},
        ),
        (
            {"dataset": target_dataset_receipt(source_dataset_name="not_clang")},
            {"real_parquet_source_identity_ok", "dataset_name_ok"},
        ),
        (
            {
                "data_format": "npz",
                "dataset": target_dataset_receipt(source_format="npz"),
            },
            {"real_parquet_source_identity_ok", "dataset_format_ok"},
        ),
        (
            {"dtype": "float32"},
            {"dtype_ok"},
        ),
        (
            {
                "model_name": "HybridTinyLM",
                "model_config": local_gb10_model_config(),
            },
            {"model_identity_ok"},
        ),
        (
            {
                "model_name": "local_gb10_quarter",
                "model_config": local_gb10_model_config(profile="HybridTinyLM"),
            },
            {"model_identity_ok"},
        ),
        (
            {"model_source": "fake.local_gb10_quarter"},
            {"model_identity_ok"},
        ),
        (
            {"model_config": local_gb10_model_config(hidden_size=16)},
            {"model_identity_ok"},
        ),
        (
            {
                "model_config": local_gb10_model_config(
                    mtp={"depth": 1, "beta": 0.6, "loss_weight": 0.3}
                )
            },
            {"model_identity_ok"},
        ),
        (
            {
                "local_gb10_quarter_preflight": resolved_local_gb10_preflight(
                    source="fake.local_gb10_quarter"
                )
            },
            {"local_gb10_quarter_preflight_ok"},
        ),
        (
            {
                "local_gb10_quarter_preflight": resolved_local_gb10_preflight(
                    profile_geometry={
                        **REQUIRED_MODEL_GEOMETRY,
                        "hidden_size": 16,
                    }
                )
            },
            {"local_gb10_quarter_preflight_ok"},
        ),
        (
            {
                "local_gb10_quarter_preflight": resolved_local_gb10_preflight(
                    tokenizer_contract={
                        **resolved_local_gb10_preflight()["tokenizer_contract"],
                        "resolved": False,
                    }
                )
            },
            {"local_gb10_quarter_preflight_ok"},
        ),
        (
            {"optimizer": adamw_identity(name="SGD")},
            {"optimizer_identity_ok", "adamw_ok"},
        ),
        (
            {"optimizer": {**adamw_identity(), "class": "fake.AdamW"}},
            {"optimizer_identity_ok", "adamw_ok"},
        ),
        (
            {"optimizer": adamw_identity(moment_dtype="bfloat16")},
            {"fp32_adamw_master_moments_ok"},
        ),
        (
            {"grad_checkpoint": grad_checkpoint_identity(enabled=False)},
            {"grad_checkpoint_expectation_ok"},
        ),
        (
            {
                "grad_checkpoint": {
                    **grad_checkpoint_identity(),
                    "source": "unit-test-local-gb10-quarter",
                }
            },
            {"grad_checkpoint_expectation_ok"},
        ),
        (
            {"device": m4_device_metadata(device_name="Apple M3 Max")},
            {"m4_runtime_metadata_ok"},
        ),
        (
            {"steps_completed": 99},
            {"step_count_ok"},
        ),
        (
            {"steps_requested": 101, "steps_completed": 100},
            {"step_count_ok"},
        ),
        (
            {"loss_decreased": False},
            {"loss_decrease_ok", "loss_fields_ok"},
        ),
        (
            {"all_finite": False},
            {"all_finite_ok", "loss_fields_ok"},
        ),
    ],
)
def test_acceptance_gate_fail_closes_on_fake_or_incomplete_evidence(
    overrides: dict[str, Any],
    failed_checks: set[str],
) -> None:
    gate = local_gb10_gate(**overrides)

    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert failed_checks.issubset(set(gate["full_local_gb10_quarter_gate_blockers"]))
    if any(
        check in failed_checks
        for check in (
            "real_parquet_source_identity_ok",
            "target_parquet_path_ok",
            "dataset_name_ok",
            "dataset_format_ok",
            "dtype_ok",
            "step_count_ok",
            "loss_fields_ok",
            "optimizer_update_ok",
        )
    ):
        assert gate["full_target_dataset_100_step_completed"] is False


@pytest.mark.parametrize(
    "preflight_overrides",
    [
        {"allocation_probe": None},
        {"allocation_probe": canonical_allocation_probe(status="blocked")},
        {"allocation_probe": canonical_allocation_probe(allocation_ready=False)},
        {
            "allocation_probe": canonical_allocation_probe(
                source="fake.local_gb10_quarter"
            )
        },
        {"allocation_probe": canonical_allocation_probe(source=None)},
        {
            "allocation_probe": canonical_allocation_probe(
                allocation_mode="caller_supplied_allocation_evidence"
            )
        },
        {"allocation_probe": canonical_allocation_probe(allocation_mode=None)},
        {"allocation_probe": canonical_allocation_probe(profile_name="HybridTinyLM")},
        {"allocation_probe": canonical_allocation_probe(model_class="FakeTinyLM")},
        {"allocation_probe": canonical_allocation_probe(model_class=None)},
        {"allocation_probe": canonical_allocation_probe(eval_scope="forward_smoke")},
        {"allocation_probe": canonical_allocation_probe(forward_executed=True)},
        {"allocation_probe": canonical_allocation_probe(training_executed=True)},
        {
            "allocation_probe": canonical_allocation_probe(
                geometry_matches_required=False
            )
        },
        {
            "allocation_probe": canonical_allocation_probe(
                required_geometry={**REQUIRED_MODEL_GEOMETRY, "hidden_size": 16}
            )
        },
        {
            "allocation_probe": canonical_allocation_probe(
                profile_geometry={**REQUIRED_MODEL_GEOMETRY, "hidden_size": 16}
            )
        },
        {
            "allocation_mode": "caller_supplied_allocation_evidence",
            "allocation_probe": canonical_allocation_probe(),
        },
    ],
)
def test_acceptance_gate_requires_canonical_allocation_probe(
    preflight_overrides: dict[str, Any],
) -> None:
    gate = local_gb10_gate(
        local_gb10_quarter_preflight=resolved_local_gb10_preflight(
            **preflight_overrides
        )
    )

    assert gate["local_gb10_quarter_preflight_ok"] is False
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert (
        "local_gb10_quarter_preflight_ok"
        in (gate["full_local_gb10_quarter_gate_blockers"])
    )
