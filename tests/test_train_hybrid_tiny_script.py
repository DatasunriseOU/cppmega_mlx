from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_hybrid_tiny.py"
GB10_SAMPLE_ROOT = ROOT / "data" / "parquet_samples" / "gb10"
REAL_PARQUET_COLUMNS = (
    "token_ids",
    "structure_ids",
)
FULL_SIDE_CHANNELS = [
    "ast_depth_ids",
    "attention_mask",
    "dep_levels",
    "node_type_ids",
    "sibling_index_ids",
    "structure_ids",
]
STRUCTURE_MODEL_KWARG_NAMES = [
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
]
THREADED_STRUCTURE_SIDE_CHANNELS = [
    "ast_depth_ids",
    "dep_levels",
    "node_type_ids",
    "sibling_index_ids",
    "structure_ids",
]


def write_npz(
    path: Path,
    *,
    vocab_size: int = 32,
    include_structure: bool = True,
    full_structure: bool = False,
) -> None:
    tokens = (np.arange(32, dtype=np.int32) % vocab_size).reshape(4, 8)
    arrays = {
        "tokens": tokens,
        "attention_mask": np.ones_like(tokens, dtype=np.float32),
        "vocab_size": np.array(vocab_size, dtype=np.int64),
        "tokenizer_contract": np.array("local_profile"),
    }
    if include_structure or full_structure:
        arrays["structure_ids"] = (tokens % 7).astype(np.int32)
        arrays["dep_levels"] = (tokens % 3).astype(np.int32)
    if full_structure:
        arrays["ast_depth_ids"] = (tokens % 5).astype(np.int32)
        arrays["sibling_index_ids"] = (tokens % 11).astype(np.int32)
        arrays["node_type_ids"] = (tokens % 13).astype(np.int32)
    np.savez(path, **arrays)


def run_script(
    *args: str,
    timeout: int = 45,
    env: dict[str, str | None] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env is None or "CPPMEGA_OPTIMIZER" not in env:
        process_env.pop("CPPMEGA_OPTIMIZER", None)
    if env is not None:
        for key, value in env.items():
            if value is None:
                process_env.pop(key, None)
            else:
                process_env[key] = value
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=process_env,
        timeout=timeout,
        check=False,
    )


def _copy_real_parquet_head(
    source_path: Path,
    sample_path: Path,
    *,
    row_count: int = 4,
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


def _tiny_route_args(symbol: str, *, steps: int = 1) -> list[str]:
    args = [
        "--json",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--steps",
        str(steps),
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        symbol,
        "--depth",
        "1",
    ]
    if symbol == "M":
        args.extend(
            [
                "--mamba-expand",
                "1",
                "--mamba-head-dim",
                "4",
                "--mamba-state-dim",
                "4",
                "--mamba-groups",
                "1",
                "--mamba-chunk-size",
                "4",
            ]
        )
    elif symbol == "R":
        args.extend(
            [
                "--m2rnn-k-head-dim",
                "2",
                "--m2rnn-v-head-dim",
                "2",
                "--m2rnn-num-v-heads",
                "1",
                "--m2rnn-num-f-heads",
                "1",
                "--m2rnn-chunk-size",
                "4",
            ]
        )
    return args


def _tiny_mixed_mr_args(*, steps: int = 1) -> list[str]:
    return [
        "--json",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--steps",
        str(steps),
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "MR",
        "--depth",
        "2",
        "--mamba-expand",
        "1",
        "--mamba-head-dim",
        "4",
        "--mamba-state-dim",
        "4",
        "--mamba-groups",
        "1",
        "--mamba-chunk-size",
        "4",
        "--m2rnn-k-head-dim",
        "2",
        "--m2rnn-v-head-dim",
        "2",
        "--m2rnn-num-v-heads",
        "1",
        "--m2rnn-num-f-heads",
        "1",
        "--m2rnn-chunk-size",
        "4",
    ]


def _load_json_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def _assert_training_optimizer(
    payload: dict[str, Any],
    *,
    name: str,
    source: str = "default",
    learning_rate: float | None = None,
    weight_decay: float | None = None,
) -> None:
    optimizer = payload["training_optimizer"]
    assert optimizer["name"] == name
    assert optimizer["source"] == source
    if learning_rate is not None:
        assert optimizer["learning_rate"] == learning_rate
    if weight_decay is not None:
        assert optimizer["weight_decay"] == weight_decay

    groups = optimizer["groups"]
    if name == "adamw":
        assert optimizer["class"] == "cppmega_mlx.training.optimizers.AdamWFP32Moments"
        assert optimizer["factory"] == "cppmega_mlx.training.optimizers.make_adamw"
        assert groups == {
            "adamw": {
                "betas": [0.9, 0.999],
                "learning_rate": optimizer["learning_rate"],
                "moment_dtype": "float32",
                "weight_decay": optimizer["weight_decay"],
            },
        }
    elif name == "muon":
        assert optimizer["class"] == "cppmega_mlx.training.optimizers.MuonAdamWMulti"
        assert optimizer["factory"] == "cppmega_mlx.training.optimizers.make_muon"
        assert optimizer["cppmega_cuda_parity"] is False
        assert groups["muon"] == {
            "learning_rate": optimizer["learning_rate"],
            "momentum": 0.95,
            "nesterov": True,
            "ns_carrier": "fp32",
            "ns_steps": 5,
            "weight_decay": optimizer["weight_decay"],
        }
        assert groups["adamw"] == {
            "betas": [0.9, 0.95],
            "learning_rate": 1e-4,
            "moment_dtype": "float32",
            "weight_decay": optimizer["weight_decay"],
        }
    else:
        raise AssertionError(f"unexpected optimizer under test: {name}")


def _assert_training_loss(
    payload: dict[str, Any],
    *,
    backend: str = "cross_entropy",
    chunk_rows: int | None = None,
) -> None:
    loss = payload["training_loss"]
    assert loss["backend"] == backend
    assert loss["eval_backend"] == "cross_entropy"
    assert loss["manual_chunked_backward"] is False
    if backend == "cross_entropy":
        assert loss == {
            "backend": "cross_entropy",
            "chunk_rows": None,
            "default": True,
            "eval_backend": "cross_entropy",
            "forward_memory_saving_claim": False,
            "manual_chunked_backward": False,
            "source": "cppmega_mlx.training.loss.next_token_cross_entropy",
        }
    elif backend == "cce":
        assert loss == {
            "backend": "cce",
            "chunk_rows": chunk_rows,
            "default": False,
            "eval_backend": "cross_entropy",
            "forward_memory_saving_claim": True,
            "manual_chunked_backward": False,
            "source": "cppmega_mlx.training.loss.next_token_cut_cross_entropy",
        }
    else:
        raise AssertionError(f"unexpected loss backend under test: {backend}")


def _tiny_ngram_hash_args() -> list[str]:
    return [
        "--ngram-hash",
        "--ngram-hash-orders",
        "2",
        "--ngram-hash-heads",
        "1",
        "--ngram-hash-table-size",
        "16",
        "--ngram-hash-embed-dim",
        "2",
        "--ngram-hash-seed",
        "7",
    ]


def _assert_ngram_hash_contract(dataset: dict[str, Any]) -> None:
    contract = dataset["side_channel_contract"]
    assert contract["unsupported_sidecars_fail_closed"] is True
    ngram_hash = contract["ngram_hash"]
    assert ngram_hash == {
        "batch_slice": "tokens[:, :-1]",
        "enabled": True,
        "heads": 1,
        "model_derived": True,
        "orders": [2],
        "sidecars_supported": False,
        "source": "input_ids",
        "threaded_to_model": "HybridTinyLM.__call__(input_ids)",
    }


def _assert_full_structure_contract(dataset: dict[str, Any]) -> None:
    assert dataset["side_channels"] == FULL_SIDE_CHANNELS
    structure = dataset["side_channel_contract"]["structure_side_channels"]
    assert structure["threaded_to_model"] == THREADED_STRUCTURE_SIDE_CHANNELS
    assert structure["model_kwarg_names"] == STRUCTURE_MODEL_KWARG_NAMES
    assert structure["batch_slice"] == "tokens[:, :-1]"
    assert structure["attention_mask_is_loss_only"] is True


def _assert_finite_mixed_mr_training_payload(
    payload: dict[str, Any],
    *,
    compiled: bool,
    expected_steps: int = 1,
) -> None:
    expected_tokens_per_step = 3
    assert payload["status"] == "ok"
    assert payload["compile"] is compiled
    assert payload["compile_enabled"] is compiled
    compile_plan = payload["compile_plan"]
    assert compile_plan["requested"] is compiled
    assert compile_plan["enabled"] is compiled
    assert compile_plan["disabled_by_env"] is False
    assert compile_plan["backend"] == (
        "mlx.core.compile" if compiled else "eager"
    )
    assert compile_plan["pattern"] == (
        "mlx_lm_tuner_stateful_step" if compiled else "python_eager_step"
    )
    assert compile_plan["state_inputs_outputs"] == (
        ["model.state", "optimizer.state", "mx.random.state"] if compiled else []
    )
    assert compile_plan["fixed_batch_signature"] is compiled
    assert payload["route_symbols"] == "MR"
    assert payload["route_roles"] == ["mamba3", "m2rnn"]
    assert payload["steps"] == expected_steps
    assert payload["tokens_per_step"] == expected_tokens_per_step
    assert payload["trained_tokens"] == expected_steps * expected_tokens_per_step
    assert payload["start_step"] == 0
    assert payload["end_step"] == expected_steps
    assert payload["model_source"] == "cppmega_mlx.models.hybrid_lm"
    assert payload["parameter_count"] > 0
    assert isinstance(payload["device"]["metal_available"], bool)
    assert "mlx_metal" in payload["device"]
    _assert_memory_receipt(payload)

    final_loss = payload["final_loss"]
    assert isinstance(final_loss, float)
    assert math.isfinite(final_loss)
    assert final_loss > 0

    assert payload["backend_plan"] == {
        "attention_backends": [],
        "backend_summary": {"m2rnn": 1, "mamba3": 1},
        "execution_backend": "mlx",
        "layer_backends": ["mamba3", "m2rnn"],
        "route_roles": ["mamba3", "m2rnn"],
        "route_symbols": "MR",
    }

    step_metrics = payload["step_metrics"]
    assert isinstance(step_metrics, list)
    assert len(step_metrics) == expected_steps
    for index, step in enumerate(step_metrics, start=1):
        assert step["compiled"] is compiled
        assert step["ntokens"] == expected_tokens_per_step
        assert step["step"] == index
        assert step["trained_tokens"] == index * expected_tokens_per_step
        assert step["tokens_per_second"] > 0
        assert math.isfinite(step["loss"])
        assert step["loss"] > 0


def _assert_finite_route_training_payload(
    payload: dict[str, Any],
    *,
    symbol: str,
    backend: str,
    compiled: bool,
    expected_steps: int = 1,
) -> None:
    expected_tokens_per_step = 3
    assert payload["status"] == "ok"
    assert payload["compile"] is compiled
    assert payload["compile_enabled"] is compiled
    compile_plan = payload["compile_plan"]
    assert compile_plan["requested"] is compiled
    assert compile_plan["enabled"] is compiled
    assert compile_plan["disabled_by_env"] is False
    assert compile_plan["backend"] == (
        "mlx.core.compile" if compiled else "eager"
    )
    assert compile_plan["pattern"] == (
        "mlx_lm_tuner_stateful_step" if compiled else "python_eager_step"
    )
    assert compile_plan["state_inputs_outputs"] == (
        ["model.state", "optimizer.state", "mx.random.state"] if compiled else []
    )
    assert compile_plan["fixed_batch_signature"] is compiled
    assert payload["route_symbols"] == symbol
    assert payload["route_roles"] == [backend]
    assert payload["steps"] == expected_steps
    assert payload["tokens_per_step"] == expected_tokens_per_step
    assert payload["trained_tokens"] == expected_steps * expected_tokens_per_step
    assert payload["start_step"] == 0
    assert payload["end_step"] == expected_steps
    assert payload["model_source"] == "cppmega_mlx.models.hybrid_lm"
    assert payload["parameter_count"] > 0
    assert isinstance(payload["device"]["metal_available"], bool)
    assert "mlx_metal" in payload["device"]
    _assert_memory_receipt(payload)

    final_loss = payload["final_loss"]
    assert isinstance(final_loss, float)
    assert math.isfinite(final_loss)
    assert final_loss > 0

    expected_attention_backends = ["mlx.fast.sdpa"] if backend == "attention" else []
    expected_execution_backend = (
        "mlx+mlx.fast.sdpa" if backend == "attention" else "mlx"
    )
    backend_plan = payload["backend_plan"]
    assert backend_plan == {
        "attention_backends": expected_attention_backends,
        "backend_summary": {backend: 1},
        "execution_backend": expected_execution_backend,
        "layer_backends": [backend],
        "route_roles": [backend],
        "route_symbols": symbol,
    }

    step_metrics = payload["step_metrics"]
    assert isinstance(step_metrics, list)
    assert len(step_metrics) == expected_steps
    for index, step in enumerate(step_metrics, start=1):
        assert step["compiled"] is compiled
        assert step["ntokens"] == expected_tokens_per_step
        assert step["step"] == index
        assert step["trained_tokens"] == index * expected_tokens_per_step
        assert step["tokens_per_second"] > 0
        assert math.isfinite(step["loss"])
        assert step["loss"] > 0


def _assert_optional_non_negative_int(value: Any) -> None:
    assert value is None or (isinstance(value, int) and value >= 0)


def _assert_memory_receipt(payload: dict[str, Any]) -> None:
    memory = payload["memory"]
    assert {
        "after",
        "before",
        "peak_memory_bytes",
        "peak_memory_reset",
    }.issubset(memory)
    assert isinstance(memory["peak_memory_reset"], bool)
    assert memory["peak_memory_bytes"] == memory["after"]["peak_memory_bytes"]
    for scope in ("before", "after"):
        assert set(memory[scope]) == {
            "active_memory_bytes",
            "cache_memory_bytes",
            "peak_memory_bytes",
        }
        for key in ("active_memory_bytes", "cache_memory_bytes", "peak_memory_bytes"):
            _assert_optional_non_negative_int(memory[scope][key])
    if "clear_cache_every_steps" in memory:
        cadence = memory["clear_cache_every_steps"]
        assert cadence is None or (isinstance(cadence, int) and cadence > 0)
    if "clear_cache_events" in memory:
        assert isinstance(memory["clear_cache_events"], list)
    if "clear_cache_event_count" in memory:
        assert memory["clear_cache_event_count"] == len(memory.get("clear_cache_events", []))


def _assert_update_boundary_training_state(
    manifest: dict[str, Any],
    *,
    step: int,
    trained_tokens: int,
    compiled: bool,
) -> None:
    assert manifest["training_state"] == {
        "compiled": compiled,
        "grad_accum_steps": 1,
        "gradient_accumulator": {
            "file": None,
            "num_tensors": 0,
            "present": False,
            "tensors": [],
        },
        "gradient_accumulator_present": False,
        "pending_microbatches": 0,
        "state": {
            "step": step,
            "trained_tokens": trained_tokens,
        },
    }


def _assert_snapshot_rng_contract(manifest: dict[str, Any]) -> None:
    rng = manifest["rng"]
    assert isinstance(rng, dict)
    assert rng["mode"] == "snapshot"
    snapshot = rng["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["version"] == 1
    assert snapshot["scope"] == "single_process_local"
    assert set(snapshot) == {
        "version",
        "scope",
        "python_random",
        "numpy_random",
        "mlx_random",
    }


def test_help_lists_hybrid_training_flags() -> None:
    result = run_script("--help")

    assert result.returncode == 0
    assert "npz_path" in result.stdout
    assert "HybridTinyLM" in result.stdout
    assert "--dry-run-json" in result.stdout
    assert "--compile" in result.stdout
    assert "--no-compile" in result.stdout
    assert "--pattern" in result.stdout
    assert "A=attention/transformer" in result.stdout
    assert "--depth" in result.stdout
    assert "--moe-num-experts" in result.stdout
    assert "--mamba-state-dim" in result.stdout
    assert "--m2rnn-v-head-dim" in result.stdout
    assert "--valid-npz-path" in result.stdout
    assert "--valid-dataset-path" in result.stdout
    assert "--valid-dataset-format" in result.stdout
    assert "--eval-batches" in result.stdout
    assert "--optimizer" in result.stdout
    assert "adamw" in result.stdout
    assert "muon" in result.stdout
    assert "CPPMEGA_OPTIMIZER" in result.stdout
    assert "--loss-backend" in result.stdout
    assert "--cce-chunk-rows" in result.stdout
    assert "--memory-limit-total-bytes" in result.stdout
    assert "--apply-memory-limit-plan" in result.stdout


def test_dry_run_json_reports_synthetic_hybrid_plan() -> None:
    result = run_script("--dry-run-json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["synthetic_npz"] is True
    assert payload["config"]["npz_path"] is None
    assert payload["config"]["optimizer"] == "adamw"
    assert payload["config"]["optimizer_source"] == "default"
    assert payload["config"]["loss_backend"] == "cross_entropy"
    assert payload["config"]["cce_chunk_rows"] > 0
    _assert_training_optimizer(payload, name="adamw")
    _assert_training_loss(payload)
    assert payload["dataset"]["num_batches"] >= 1
    _assert_full_structure_contract(payload["dataset"])
    assert payload["model_source"] == "cppmega_mlx.models.hybrid_lm"
    assert payload["model_config"]["vocab_size"] == 32
    assert payload["route_symbols"] == "AEMR"
    assert payload["tokens_per_step"] == 7
    assert payload["planned_steps"] == 1
    assert payload["compile"] is False
    assert payload["compile_enabled"] is False
    assert payload["compile_plan"]["backend"] == "eager"
    assert payload["compile_plan"]["state_inputs_outputs"] == []
    assert isinstance(payload["device"]["metal_available"], bool)
    assert "mlx_metal" in payload["device"]
    assert payload["parameter_count"] > 0
    assert "default_device" in payload["device"]
    assert payload["backend_plan"]["backend_summary"] == {
        "attention": 1,
        "m2rnn": 1,
        "mamba3": 1,
        "moe": 1,
    }


def test_dry_run_json_reports_opt_in_cce_loss_backend() -> None:
    result = run_script(
        "--dry-run-json",
        "--loss-backend",
        "cce",
        "--cce-chunk-rows",
        "4",
    )

    payload = _load_json_result(result)
    assert payload["status"] == "dry_run"
    assert payload["config"]["loss_backend"] == "cce"
    assert payload["config"]["cce_chunk_rows"] == 4
    _assert_training_loss(payload, backend="cce", chunk_rows=4)


def test_dry_run_json_selects_muon_from_cli() -> None:
    result = run_script(
        "--dry-run-json",
        "--optimizer",
        "muon",
        "--lr",
        "0.002",
        "--weight-decay",
        "0.01",
    )

    payload = _load_json_result(result)
    assert payload["config"]["optimizer"] == "muon"
    assert payload["config"]["optimizer_source"] == "cli"
    _assert_training_optimizer(
        payload,
        name="muon",
        source="cli",
        learning_rate=0.002,
        weight_decay=0.01,
    )


def test_dry_run_json_selects_muon_from_environment() -> None:
    result = run_script(
        "--dry-run-json",
        env={"CPPMEGA_OPTIMIZER": "muon"},
    )

    payload = _load_json_result(result)
    assert payload["config"]["optimizer"] == "muon"
    assert payload["config"]["optimizer_source"] == "env"
    _assert_training_optimizer(payload, name="muon", source="env")


def test_invalid_optimizer_env_returns_error_json() -> None:
    result = run_script(
        "--json",
        env={"CPPMEGA_OPTIMIZER": "lion"},
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert "unsupported optimizer='lion'" in payload["error"]
    assert payload["config"]["optimizer"] == "lion"
    assert payload["config"]["optimizer_source"] == "env"
    assert payload["training_optimizer"]["name"] == "lion"
    assert "unsupported optimizer='lion'" in payload["training_optimizer"]["error"]


def test_dry_run_json_reports_memory_limit_plan_without_applying() -> None:
    result = run_script(
        "--dry-run-json",
        "--memory-limit-total-bytes",
        "2000",
        "--memory-limit-wired-ratio",
        "0.5",
        "--memory-limit-metal-ratio",
        "0.75",
        "--apply-memory-limit-plan",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    memory_limit = payload["memory_limit"]
    assert memory_limit["mode"] == "planned"
    assert memory_limit["apply_requested"] is True
    assert memory_limit["applied"] is False
    assert memory_limit["metal_limit_api_path"] is None
    assert memory_limit["total_bytes_source"] == "cli"
    assert memory_limit["plan"] == {
        "metal_limit_bytes": 1500,
        "metal_ratio": 0.75,
        "total_bytes": 2000,
        "wired_limit_bytes": 1000,
        "wired_ratio": 0.5,
    }


def test_dry_run_json_reports_ngram_hash_model_derived_side_channel_contract(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(
        str(npz_path),
        "--dry-run-json",
        *_tiny_route_args("A"),
        *_tiny_ngram_hash_args(),
    )

    payload = _load_json_result(result)
    assert payload["status"] == "dry_run"
    assert payload["model_config"]["ngram_hash_enabled"] is True
    _assert_ngram_hash_contract(payload["dataset"])


def test_dry_run_json_reports_npz_hybrid_plan(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(
        str(npz_path),
        "--dry-run-json",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--steps",
        "2",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "AEMR",
        "--depth",
        "4",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["synthetic_npz"] is False
    assert payload["dataset"]["path"] == str(npz_path)
    assert payload["dataset"]["num_samples"] == 8
    assert payload["dataset"]["num_batches"] == 8
    assert payload["model_config"]["pattern"] == "AEMR"
    assert payload["route_symbols"] == "AEMR"
    assert payload["tokens_per_step"] == 3
    assert payload["planned_steps"] == 2


@pytest.mark.parametrize(
    "dataset_name",
    [
        "clang_semantic_4k_v10",
        "clang_commits_4k_v1",
    ],
)
def test_real_gb10_parquet_cli_smoke_trains_token_only_after_retokenize(
    tmp_path: Path,
    dataset_name: str,
) -> None:
    source_path = GB10_SAMPLE_ROOT / dataset_name / "val_00000.parquet"
    if not source_path.exists():
        pytest.skip(f"GB10 parquet sample is not present: {source_path}")

    sample_path = tmp_path / f"{dataset_name}_head.parquet"
    _copy_real_parquet_head(source_path, sample_path)

    result = run_script(
        str(sample_path),
        "--json",
        "--data-format",
        "parquet",
        "--token-key",
        "token_ids",
        "--batch-size",
        "1",
        "--seq-len",
        "64",
        "--steps",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "M",
        "--depth",
        "1",
        "--mamba-expand",
        "1",
        "--mamba-head-dim",
        "4",
        "--mamba-state-dim",
        "4",
        "--mamba-groups",
        "1",
        "--mamba-chunk-size",
        "4",
        "--vocab-size",
        "131072",
    )

    payload = _load_json_result(result)
    assert payload["synthetic_npz"] is False
    assert payload["dataset"]["path"] == str(sample_path)
    assert payload["dataset"]["metadata"]["source_format"] == "parquet"
    assert payload["dataset"]["token_key"] == "token_ids"
    assert payload["dataset"]["num_samples"] >= 4
    assert payload["dataset"]["side_channels"] == []
    assert payload["dataset"]["side_channel_contract"]["structure_side_channels"][
        "threaded_to_model"
    ] == []
    receipt = payload["dataset"]["dataset_receipt"]
    assert receipt["parquet_receipt"] == {
        "source_format": "parquet",
        "columns": sorted(REAL_PARQUET_COLUMNS),
        "column_types": {
            "token_ids": "large_list<element: uint32>",
            "structure_ids": "large_list<element: int8>",
        },
        "token_source": {
            "mode": "token_column",
            "column": "token_ids",
            "type": "large_list<element: uint32>",
        },
        "side_channel_sources": {},
        "skipped_side_channel_columns": [
            {
                "field": "structure_ids",
                "column": "structure_ids",
                "type": "large_list<element: int8>",
                "reason": "not_token_aligned",
            },
        ],
    }
    assert receipt == {
        "batch_size": 1,
        "dropped_samples": payload["dataset"]["dropped_samples"],
        "num_batches": payload["dataset"]["num_batches"],
        "num_samples": payload["dataset"]["num_samples"],
        "parquet_receipt": receipt["parquet_receipt"],
        "seq_len": 64,
        "side_channels": payload["dataset"]["side_channels"],
        "source_dataset_name": dataset_name,
        "source_format": "parquet",
        "source_path": str(sample_path),
        "token_key": "token_ids",
    }
    assert payload["route_symbols"] == "M"
    assert payload["route_roles"] == ["mamba3"]
    assert payload["tokens_per_step"] == 63
    assert payload["trained_tokens"] == 63
    assert payload["final_loss"] > 0
    assert payload["step_metrics"][0]["ntokens"] == 63


def test_dry_run_json_accepts_explicit_no_compile() -> None:
    result = run_script(
        "--dry-run-json",
        "--no-compile",
        "--pattern",
        "A",
        "--depth",
        "1",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["compile"] is False
    assert payload["compile_enabled"] is False
    assert payload["compile_plan"]["enabled"] is False
    assert payload["compile_plan"]["backend"] == "eager"
    assert payload["route_symbols"] == "A"
    assert payload["route_roles"] == ["attention"]


@pytest.mark.parametrize(
    ("symbol", "backend"),
    [
        ("M", "mamba3"),
        ("R", "m2rnn"),
    ],
)
def test_single_route_cli_smoke_eager_reports_finite_route_metadata(
    tmp_path: Path,
    symbol: str,
    backend: str,
) -> None:
    npz_path = tmp_path / f"tokens_{symbol}.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(str(npz_path), *_tiny_route_args(symbol))
    payload = _load_json_result(result)

    _assert_finite_route_training_payload(
        payload,
        symbol=symbol,
        backend=backend,
        compiled=False,
    )
    assert payload["synthetic_npz"] is False
    assert payload["dataset"]["path"] == str(npz_path)


def test_cce_loss_backend_cli_smoke_eager_reports_finite_training(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens_cce.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(
        str(npz_path),
        *_tiny_route_args("A"),
        "--loss-backend",
        "cce",
        "--cce-chunk-rows",
        "2",
    )
    payload = _load_json_result(result)

    _assert_finite_route_training_payload(
        payload,
        symbol="A",
        backend="attention",
        compiled=False,
    )
    assert payload["config"]["loss_backend"] == "cce"
    assert payload["config"]["cce_chunk_rows"] == 2
    _assert_training_loss(payload, backend="cce", chunk_rows=2)


@pytest.mark.parametrize(
    ("symbol", "backend"),
    [
        ("M", "mamba3"),
        ("R", "m2rnn"),
    ],
)
def test_single_route_cli_smoke_compiled_reports_finite_route_metadata(
    tmp_path: Path,
    symbol: str,
    backend: str,
) -> None:
    npz_path = tmp_path / f"tokens_{symbol}.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(str(npz_path), *_tiny_route_args(symbol, steps=2), "--compile")
    payload = _load_json_result(result)

    _assert_finite_route_training_payload(
        payload,
        symbol=symbol,
        backend=backend,
        compiled=True,
        expected_steps=2,
    )


def test_mixed_mamba3_m2rnn_cli_smoke_eager_reports_finite_route_metadata(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens_MR.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(str(npz_path), *_tiny_mixed_mr_args())
    payload = _load_json_result(result)

    _assert_finite_mixed_mr_training_payload(payload, compiled=False)
    assert payload["synthetic_npz"] is False
    assert payload["dataset"]["path"] == str(npz_path)


def test_mixed_mamba3_m2rnn_cli_smoke_compiled_reports_finite_route_metadata(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens_MR.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(str(npz_path), *_tiny_mixed_mr_args(steps=2), "--compile")
    payload = _load_json_result(result)

    _assert_finite_mixed_mr_training_payload(
        payload,
        compiled=True,
        expected_steps=2,
    )


def test_transformer_route_cli_smoke_compiled_reports_finite_metadata(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens_A.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(str(npz_path), *_tiny_route_args("A", steps=2), "--compile")
    payload = _load_json_result(result)

    _assert_finite_route_training_payload(
        payload,
        symbol="A",
        backend="attention",
        compiled=True,
        expected_steps=2,
    )
    assert payload["backend_plan"]["attention_backends"] == ["mlx.fast.sdpa"]
    assert payload["backend_plan"]["execution_backend"] == "mlx+mlx.fast.sdpa"


@pytest.mark.parametrize(
    ("symbol", "backend", "compiled"),
    [
        ("M", "mamba3", False),
        ("R", "m2rnn", False),
        ("M", "mamba3", True),
        ("R", "m2rnn", True),
    ],
)
def test_single_route_checkpoint_resume_preserves_batch_cursor(
    tmp_path: Path,
    symbol: str,
    backend: str,
    compiled: bool,
) -> None:
    npz_path = tmp_path / f"tokens_{symbol}.npz"
    compile_suffix = "compiled" if compiled else "eager"
    checkpoint_dir = tmp_path / f"checkpoints_{symbol}_{compile_suffix}"
    resumed_final = tmp_path / f"resumed_final_{symbol}_{compile_suffix}"
    write_npz(npz_path, vocab_size=32)
    compile_args = ["--compile"] if compiled else []

    first = run_script(
        str(npz_path),
        *_tiny_route_args(symbol),
        *compile_args,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-save-interval",
        "1",
    )
    first_payload = _load_json_result(first)
    _assert_finite_route_training_payload(
        first_payload,
        symbol=symbol,
        backend=backend,
        compiled=compiled,
    )
    assert first_payload["checkpoints"]["saved"][0]["step"] == 1

    checkpoint_path = checkpoint_dir / "checkpoint-000001"
    metadata_path = checkpoint_path / "metadata.json"
    manifest = json.loads(metadata_path.read_text())
    assert manifest["model_config"]["pattern"] == symbol
    assert manifest["batch_cursor"]["global_batch_offset"] == 1
    assert manifest["batch_cursor"]["batch_offset"] == 1
    assert manifest["trained_tokens"] == 3
    _assert_update_boundary_training_state(
        manifest,
        step=1,
        trained_tokens=3,
        compiled=compiled,
    )

    second = run_script(
        str(npz_path),
        *_tiny_route_args(symbol),
        *compile_args,
        "--resume-from",
        str(checkpoint_path),
        "--checkpoint-path",
        str(resumed_final),
    )
    second_payload = _load_json_result(second)

    assert second_payload["status"] == "ok"
    assert second_payload["compile"] is compiled
    assert second_payload["route_symbols"] == symbol
    assert second_payload["route_roles"] == [backend]
    assert second_payload["resume"]["loaded"] is True
    assert second_payload["resume"]["step"] == 1
    assert second_payload["resume"]["trained_tokens"] == 3
    assert second_payload["resume"]["batch_cursor"]["global_batch_offset"] == 1
    assert second_payload["start_step"] == 1
    assert second_payload["end_step"] == 2
    assert second_payload["step_metrics"][0]["compiled"] is compiled
    assert second_payload["step_metrics"][0]["step"] == 2
    assert second_payload["trained_tokens"] == 6

    resumed_manifest = json.loads((resumed_final / "metadata.json").read_text())
    assert resumed_manifest["model_config"]["pattern"] == symbol
    assert resumed_manifest["batch_cursor"]["global_batch_offset"] == 2
    assert resumed_manifest["batch_cursor"]["batch_offset"] == 2
    assert resumed_manifest["trained_tokens"] == 6
    _assert_update_boundary_training_state(
        resumed_manifest,
        step=2,
        trained_tokens=6,
        compiled=compiled,
    )


@pytest.mark.parametrize(
    ("symbol", "backend"),
    [
        ("M", "mamba3"),
        ("R", "m2rnn"),
    ],
)
def test_single_route_checkpoint_resume_eval_records_structure_receipts(
    tmp_path: Path,
    symbol: str,
    backend: str,
) -> None:
    npz_path = tmp_path / f"tokens_{symbol}.npz"
    valid_npz_path = tmp_path / f"valid_tokens_{symbol}.npz"
    checkpoint_dir = tmp_path / f"checkpoints_{symbol}"
    resumed_final = tmp_path / f"resumed_final_{symbol}"
    write_npz(npz_path, vocab_size=32, full_structure=True)
    write_npz(valid_npz_path, vocab_size=32, full_structure=True)

    first = run_script(
        str(npz_path),
        *_tiny_route_args(symbol),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-save-interval",
        "1",
    )
    first_payload = _load_json_result(first)
    _assert_finite_route_training_payload(
        first_payload,
        symbol=symbol,
        backend=backend,
        compiled=False,
    )
    _assert_full_structure_contract(first_payload["dataset"])

    checkpoint_path = checkpoint_dir / "checkpoint-000001"
    manifest = json.loads((checkpoint_path / "metadata.json").read_text())
    _assert_full_structure_contract(manifest["dataset"])
    assert manifest["resume_cursor"] == {
        "batch_cursor": {
            "batch_offset": 1,
            "epoch": 0,
            "global_batch_offset": 1,
        },
        "step": 1,
        "trained_tokens": 3,
    }

    second = run_script(
        str(npz_path),
        *_tiny_route_args(symbol),
        "--valid-dataset-path",
        str(valid_npz_path),
        "--valid-dataset-format",
        "npz",
        "--eval-batches",
        "1",
        "--resume-from",
        str(checkpoint_path),
        "--checkpoint-path",
        str(resumed_final),
    )
    second_payload = _load_json_result(second)

    assert second_payload["route_symbols"] == symbol
    assert second_payload["route_roles"] == [backend]
    assert second_payload["resume"]["resume_cursor"] == {
        "batch_cursor": {
            "batch_offset": 1,
            "epoch": 0,
            "global_batch_offset": 1,
        },
        "step": 1,
        "trained_tokens": 3,
    }
    assert second_payload["start_step"] == 1
    assert second_payload["end_step"] == 2
    _assert_full_structure_contract(second_payload["dataset"])
    assert second_payload["evaluation"]["dataset"]["path"] == str(valid_npz_path)
    _assert_full_structure_contract(second_payload["evaluation"]["dataset"])
    assert second_payload["evaluation"]["requested_batches"] == 1
    assert second_payload["evaluation"]["planned_batches"] == 1
    assert second_payload["evaluation"]["evaluated_batches"] == 1
    assert second_payload["evaluation"]["metrics"]["batches"] == 1
    assert second_payload["evaluation"]["metrics"]["ntokens"] == 3
    assert second_payload["evaluation"]["metrics"]["loss"] > 0

    resumed_manifest = json.loads((resumed_final / "metadata.json").read_text())
    _assert_full_structure_contract(resumed_manifest["dataset"])
    assert resumed_manifest["resume_cursor"] == {
        "batch_cursor": {
            "batch_offset": 2,
            "epoch": 0,
            "global_batch_offset": 2,
        },
        "step": 2,
        "trained_tokens": 6,
    }
    assert resumed_manifest["evaluation"]["dataset"]["path"] == str(valid_npz_path)
    _assert_full_structure_contract(resumed_manifest["evaluation"]["dataset"])
    assert resumed_manifest["evaluation"]["requested_batches"] == 1
    assert resumed_manifest["evaluation"]["planned_batches"] == 1
    assert resumed_manifest["evaluation"]["evaluated_batches"] == 1
    assert resumed_manifest["evaluation"]["metrics"]["batches"] == 1
    assert resumed_manifest["evaluation"]["metrics"]["ntokens"] == 3
    assert resumed_manifest["evaluation"]["metrics"]["loss"] > 0


def test_one_eager_training_step_reports_machine_readable_metrics(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(
        str(npz_path),
        "--json",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "AEMR",
        "--depth",
        "4",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["compile"] is False
    assert payload["compile_enabled"] is False
    assert payload["dtype"] == "float32"
    assert payload["steps"] == 1
    assert payload["tokens_per_step"] == 3
    assert payload["trained_tokens"] == 3
    assert payload["final_loss"] > 0
    assert payload["mean_step_time_s"] > 0
    assert payload["tokens_per_second"] > 0
    assert payload["parameter_count"] > 0
    assert payload["model_source"] == "cppmega_mlx.models.hybrid_lm"
    assert payload["route_symbols"] == "AEMR"
    assert payload["step_metrics"][0]["compiled"] is False


def test_training_reports_validation_eval_metrics(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    valid_npz_path = tmp_path / "valid_tokens.npz"
    write_npz(npz_path, vocab_size=32)
    write_npz(valid_npz_path, vocab_size=32)

    result = run_script(
        str(npz_path),
        "--json",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--valid-npz-path",
        str(valid_npz_path),
        "--eval-batches",
        "1",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["evaluation"]["dataset"]["path"] == str(valid_npz_path)
    assert payload["evaluation"]["requested_batches"] == 1
    assert payload["evaluation"]["evaluated_batches"] == 1
    assert payload["evaluation"]["metrics"]["batches"] == 1
    assert payload["evaluation"]["metrics"]["ntokens"] == 3
    assert payload["evaluation"]["metrics"]["loss"] > 0
    assert payload["evaluation"]["metrics"]["tokens_per_second"] > 0


@pytest.mark.parametrize(
    ("symbol", "backend"),
    [
        ("A", "attention"),
        ("E", "moe"),
        ("M", "mamba3"),
        ("R", "m2rnn"),
    ],
)
def test_route_training_validation_dataset_path_reports_full_side_channel_contract(
    tmp_path: Path,
    symbol: str,
    backend: str,
) -> None:
    npz_path = tmp_path / f"tokens_{symbol}.npz"
    valid_npz_path = tmp_path / f"valid_tokens_{symbol}.npz"
    write_npz(npz_path, vocab_size=32, full_structure=True)
    write_npz(valid_npz_path, vocab_size=32, full_structure=True)

    result = run_script(
        str(npz_path),
        *_tiny_route_args(symbol),
        "--valid-dataset-path",
        str(valid_npz_path),
        "--valid-dataset-format",
        "npz",
        "--eval-batches",
        "1",
    )

    payload = _load_json_result(result)
    _assert_finite_route_training_payload(
        payload,
        symbol=symbol,
        backend=backend,
        compiled=False,
    )
    _assert_full_structure_contract(payload["dataset"])
    assert payload["dataset"]["path"] == str(npz_path)
    assert payload["evaluation"]["dataset"]["path"] == str(valid_npz_path)
    _assert_full_structure_contract(payload["evaluation"]["dataset"])
    assert payload["evaluation"]["requested_batches"] == 1
    assert payload["evaluation"]["evaluated_batches"] == 1
    assert payload["evaluation"]["metrics"]["batches"] == 1
    assert payload["evaluation"]["metrics"]["ntokens"] == 3
    assert payload["evaluation"]["metrics"]["loss"] > 0


def test_training_eval_reports_ngram_hash_side_channel_contract(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    valid_npz_path = tmp_path / "valid_tokens.npz"
    write_npz(npz_path, vocab_size=32)
    write_npz(valid_npz_path, vocab_size=32)

    result = run_script(
        str(npz_path),
        *_tiny_route_args("A"),
        *_tiny_ngram_hash_args(),
        "--valid-npz-path",
        str(valid_npz_path),
        "--eval-batches",
        "1",
    )

    payload = _load_json_result(result)
    _assert_ngram_hash_contract(payload["dataset"])
    assert payload["evaluation"]["dataset"]["path"] == str(valid_npz_path)
    _assert_ngram_hash_contract(payload["evaluation"]["dataset"])


def test_dry_run_validation_dataset_path_reports_full_side_channel_plan(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    valid_npz_path = tmp_path / "valid_tokens.npz"
    write_npz(npz_path, vocab_size=32, full_structure=True)
    write_npz(valid_npz_path, vocab_size=32, full_structure=True)

    result = run_script(
        str(npz_path),
        "--dry-run-json",
        *_tiny_route_args("A"),
        "--valid-dataset-path",
        str(valid_npz_path),
        "--valid-dataset-format",
        "npz",
        "--eval-batches",
        "2",
    )

    payload = _load_json_result(result)
    assert payload["status"] == "dry_run"
    _assert_full_structure_contract(payload["dataset"])
    assert payload["evaluation"]["dataset"]["path"] == str(valid_npz_path)
    _assert_full_structure_contract(payload["evaluation"]["dataset"])
    assert payload["evaluation"]["requested_batches"] == 2
    assert payload["evaluation"]["planned_batches"] == 2


def test_no_structure_fails_closed_for_validation_side_channels(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    valid_npz_path = tmp_path / "valid_tokens.npz"
    write_npz(npz_path, vocab_size=32, include_structure=False)
    write_npz(valid_npz_path, vocab_size=32, include_structure=True)

    result = run_script(
        str(npz_path),
        "--json",
        "--no-structure",
        "--valid-npz-path",
        str(valid_npz_path),
        "--eval-batches",
        "1",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert "--no-structure" in payload["error"]
    assert "structure side channels" in payload["error"]


def test_ngram_hash_side_channel_contract_is_saved_to_training_checkpoints(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    final_checkpoint = tmp_path / "final"
    write_npz(npz_path, vocab_size=32)

    result = run_script(
        str(npz_path),
        *_tiny_route_args("A"),
        *_tiny_ngram_hash_args(),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-save-interval",
        "1",
        "--checkpoint-path",
        str(final_checkpoint),
    )

    payload = _load_json_result(result)
    _assert_ngram_hash_contract(payload["dataset"])

    periodic_manifest = json.loads(
        (checkpoint_dir / "checkpoint-000001" / "metadata.json").read_text()
    )
    _assert_ngram_hash_contract(periodic_manifest["dataset"])

    final_manifest = json.loads((final_checkpoint / "metadata.json").read_text())
    _assert_ngram_hash_contract(final_manifest["dataset"])


def test_checkpoint_save_and_resume_reports_hybrid_cursor(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    final_checkpoint = tmp_path / "final"
    write_npz(npz_path, vocab_size=32)

    first = run_script(
        str(npz_path),
        "--json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--dtype",
        "float32",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "AEMR",
        "--depth",
        "4",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-save-interval",
        "1",
        "--checkpoint-path",
        str(final_checkpoint),
    )

    assert first.returncode == 0, first.stderr
    first_payload = json.loads(first.stdout)
    assert first_payload["status"] == "ok"
    _assert_training_optimizer(first_payload, name="adamw")
    _assert_training_loss(first_payload)
    assert first_payload["start_step"] == 0
    assert first_payload["end_step"] == 1
    assert first_payload["checkpoints"]["saved"][0]["step"] == 1
    assert first_payload["checkpoints"]["final"]["step"] == 1

    checkpoint_path = checkpoint_dir / "checkpoint-000001"
    assert (checkpoint_path / "model.safetensors").exists()
    assert (checkpoint_path / "optimizer.safetensors").exists()
    manifest = json.loads((checkpoint_path / "metadata.json").read_text())
    assert manifest["step"] == 1
    assert manifest["trained_tokens"] == first_payload["trained_tokens"]
    assert manifest["batch_cursor"]["global_batch_offset"] == 1
    assert manifest["batch_cursor"]["batch_offset"] == 1
    assert manifest["optimizer"]["present"] is True
    _assert_training_optimizer(manifest, name="adamw")
    _assert_training_loss(manifest)
    assert manifest["model_config"]["pattern"] == "AEMR"
    _assert_snapshot_rng_contract(manifest)
    _assert_update_boundary_training_state(
        manifest,
        step=1,
        trained_tokens=first_payload["trained_tokens"],
        compiled=False,
    )

    second = run_script(
        str(npz_path),
        "--json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--dtype",
        "float32",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "AEMR",
        "--depth",
        "4",
        "--resume-from",
        str(checkpoint_path),
        "--checkpoint-path",
        str(tmp_path / "resumed-final"),
    )

    assert second.returncode == 0, second.stderr
    second_payload = json.loads(second.stdout)
    assert second_payload["status"] == "ok"
    _assert_training_optimizer(second_payload, name="adamw")
    _assert_training_loss(second_payload)
    assert second_payload["resume"]["loaded"] is True
    assert second_payload["resume"]["step"] == 1
    assert second_payload["resume"]["trained_tokens"] == first_payload["trained_tokens"]
    assert second_payload["resume"]["batch_cursor"]["global_batch_offset"] == 1
    assert second_payload["start_step"] == 1
    assert second_payload["end_step"] == 2
    assert second_payload["step_metrics"][0]["step"] == 2
    assert second_payload["trained_tokens"] == first_payload["trained_tokens"] * 2
    assert second_payload["checkpoints"]["final"]["step"] == 2

    resumed_manifest = json.loads(
        (tmp_path / "resumed-final" / "metadata.json").read_text()
    )
    assert resumed_manifest["step"] == 2
    assert resumed_manifest["batch_cursor"]["global_batch_offset"] == 2
    _assert_training_optimizer(resumed_manifest, name="adamw")
    _assert_training_loss(resumed_manifest)
    _assert_snapshot_rng_contract(resumed_manifest)
    _assert_update_boundary_training_state(
        resumed_manifest,
        step=2,
        trained_tokens=second_payload["trained_tokens"],
        compiled=False,
    )


def test_muon_checkpoint_receipt_and_resume_optimizer_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    write_npz(npz_path, vocab_size=32)

    first = run_script(
        str(npz_path),
        "--json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "A",
        "--depth",
        "1",
        "--optimizer",
        "muon",
        "--lr",
        "0.002",
        "--weight-decay",
        "0.01",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-save-interval",
        "1",
    )

    first_payload = _load_json_result(first)
    _assert_training_optimizer(
        first_payload,
        name="muon",
        source="cli",
        learning_rate=0.002,
        weight_decay=0.01,
    )

    checkpoint_path = checkpoint_dir / "checkpoint-000001"
    manifest = json.loads((checkpoint_path / "metadata.json").read_text())
    _assert_training_optimizer(
        manifest,
        name="muon",
        source="cli",
        learning_rate=0.002,
        weight_decay=0.01,
    )

    resumed = run_script(
        str(npz_path),
        "--json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "A",
        "--depth",
        "1",
        "--optimizer",
        "adamw",
        "--resume-from",
        str(checkpoint_path),
    )

    assert resumed.returncode == 2
    payload = json.loads(resumed.stdout)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert "checkpoint optimizer 'muon'" in payload["error"]
    assert "requested optimizer 'adamw'" in payload["error"]
    _assert_training_optimizer(payload, name="adamw", source="cli")


def test_resume_checkpoint_cursor_advances_from_nonzero_global_offset(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    resumed_final = tmp_path / "resumed-final"
    write_npz(npz_path, vocab_size=32)

    first = run_script(
        str(npz_path),
        "--json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "2",
        "--dtype",
        "float32",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "AEMR",
        "--depth",
        "4",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-save-interval",
        "2",
    )
    assert first.returncode == 0, first.stderr
    first_payload = json.loads(first.stdout)
    assert first_payload["status"] == "ok"
    assert first_payload["end_step"] == 2
    assert first_payload["checkpoints"]["saved"][0]["step"] == 2

    checkpoint_path = checkpoint_dir / "checkpoint-000002"
    manifest = json.loads((checkpoint_path / "metadata.json").read_text())
    assert manifest["step"] == 2
    assert manifest["batch_cursor"]["global_batch_offset"] == 2
    assert manifest["batch_cursor"]["batch_offset"] == 2

    second = run_script(
        str(npz_path),
        "--json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--dtype",
        "float32",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "AEMR",
        "--depth",
        "4",
        "--resume-from",
        str(checkpoint_path),
        "--checkpoint-path",
        str(resumed_final),
    )
    assert second.returncode == 0, second.stderr
    second_payload = json.loads(second.stdout)
    assert second_payload["status"] == "ok"
    assert second_payload["resume"]["step"] == 2
    assert second_payload["resume"]["batch_cursor"]["global_batch_offset"] == 2
    assert second_payload["start_step"] == 2
    assert second_payload["end_step"] == 3
    assert second_payload["step_metrics"][0]["step"] == 3

    resumed_manifest = json.loads((resumed_final / "metadata.json").read_text())
    assert resumed_manifest["step"] == 3
    assert resumed_manifest["trained_tokens"] == second_payload["trained_tokens"]
    assert resumed_manifest["batch_cursor"]["global_batch_offset"] == 3
    assert resumed_manifest["batch_cursor"]["batch_offset"] == 3
    _assert_update_boundary_training_state(
        resumed_manifest,
        step=3,
        trained_tokens=second_payload["trained_tokens"],
        compiled=False,
    )


def test_invalid_steps_returns_error_json_with_device_and_compile_metadata() -> None:
    result = run_script("--json", "--compile", "--steps", "0")

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert payload["error"] == "steps must be positive"
    assert payload["compile"] is True
    assert payload["config"]["compile"] is True
    assert payload["compile_plan"]["requested"] is True
    assert "default_device" in payload["device"]
    assert "metal_available" in payload["device"]


def test_invalid_cce_chunk_rows_returns_error_json_with_loss_metadata() -> None:
    result = run_script("--json", "--loss-backend", "cce", "--cce-chunk-rows", "0")

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert "cce_chunk_rows must be positive" in payload["error"]
    assert payload["config"]["loss_backend"] == "cce"
    assert payload["config"]["cce_chunk_rows"] == 0
    assert payload["training_loss"]["backend"] == "cce"
    assert payload["training_loss"]["chunk_rows"] == 0


def test_valid_dataset_format_without_validation_path_returns_error_json() -> None:
    result = run_script("--json", "--valid-dataset-format", "npz")

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert (
        payload["error"]
        == "valid_dataset_format requires valid_dataset_path or valid_npz_path"
    )
    assert payload["compile"] is False
    assert payload["compile_enabled"] is False
    assert payload["compile_plan"]["enabled"] is False


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"step": "1"}, "step must be a non-negative integer"),
        ({"trained_tokens": True}, "trained_tokens must be a non-negative integer"),
        (
            {"batch_cursor": {"global_batch_offset": "1"}},
            "batch_cursor.global_batch_offset must be a non-negative integer",
        ),
    ],
)
def test_resume_rejects_coerced_checkpoint_cursor_metadata(
    tmp_path: Path,
    mutation: dict[str, Any],
    error: str,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    write_npz(npz_path, vocab_size=32)

    first = run_script(
        str(npz_path),
        "--json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "A",
        "--depth",
        "1",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-save-interval",
        "1",
    )
    assert first.returncode == 0, first.stderr

    checkpoint_path = checkpoint_dir / "checkpoint-000001"
    metadata_path = checkpoint_path / "metadata.json"
    manifest = json.loads(metadata_path.read_text())
    if "batch_cursor" in mutation:
        manifest["batch_cursor"].update(mutation["batch_cursor"])
    else:
        manifest.update(mutation)
    metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    resumed = run_script(
        str(npz_path),
        "--json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "A",
        "--depth",
        "1",
        "--resume-from",
        str(checkpoint_path),
    )

    assert resumed.returncode == 2
    payload = json.loads(resumed.stdout)
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert error in payload["error"]
    assert payload["compile"] is False
    assert payload["compile_enabled"] is False


def test_mlx_disable_compile_env_reports_requested_but_eager_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npz_path = tmp_path / "tokens_M.npz"
    write_npz(npz_path, vocab_size=32)
    monkeypatch.setenv("MLX_DISABLE_COMPILE", "1")

    result = run_script(str(npz_path), *_tiny_route_args("M"), "--compile")
    payload = _load_json_result(result)

    assert payload["compile"] is True
    assert payload["compile_enabled"] is False
    assert payload["compile_plan"] == {
        "backend": "eager",
        "disabled_by_env": True,
        "enabled": False,
        "fixed_batch_signature": False,
        "mlx_disable_compile": "1",
        "pattern": "python_eager_step",
        "requested": True,
        "state_inputs_outputs": [],
    }
    assert payload["step_metrics"][0]["compiled"] is False
    assert payload["route_symbols"] == "M"
    assert payload["route_roles"] == ["mamba3"]
