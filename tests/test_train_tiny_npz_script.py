from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_tiny_npz.py"
_HEADER = b"MMIDIDX\x00\x00"
_DTYPE_CODES = {
    np.dtype(np.uint16): 8,
    np.dtype(np.int32): 4,
    np.dtype(np.int64): 5,
}
FULL_SIDE_CHANNELS = [
    "ast_depth_ids",
    "attention_mask",
    "dep_levels",
    "node_type_ids",
    "sibling_index_ids",
    "structure_ids",
]


def write_npz(
    path: Path,
    *,
    vocab_size: int = 32,
    include_structure: bool = False,
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


def write_mmididx(
    prefix: Path,
    *,
    dtype: np.dtype = np.dtype(np.int32),
    vocab_size: int = 32,
) -> None:
    docs = [
        (np.arange(16, dtype=np.int64) % vocab_size).astype(dtype),
        (np.arange(16, 32, dtype=np.int64) % vocab_size).astype(dtype),
    ]
    flat = np.concatenate(docs)
    flat.tofile(prefix.with_suffix(".bin"))

    lengths = np.asarray([len(doc) for doc in docs], dtype=np.int32)
    pointers = np.empty(len(docs), dtype=np.int64)
    offset = 0
    for i, length in enumerate(lengths):
        pointers[i] = offset
        offset += int(length) * dtype.itemsize
    documents = np.arange(len(docs) + 1, dtype=np.int64)

    with prefix.with_suffix(".idx").open("wb") as fh:
        fh.write(_HEADER)
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<B", _DTYPE_CODES[dtype]))
        fh.write(struct.pack("<Q", len(docs)))
        fh.write(struct.pack("<Q", len(documents)))
        lengths.tofile(fh)
        pointers.tofile(fh)
        documents.tofile(fh)

    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "vocab_size": vocab_size,
                "tokenizer_contract": "local_profile",
            }
        ),
        encoding="utf-8",
    )


def run_script(
    *args: str,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _assert_update_boundary_training_state(
    manifest: dict[str, object],
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


def test_help_lists_npz_training_flags() -> None:
    result = run_script("--help")

    assert result.returncode == 0
    assert "npz_path" in result.stdout
    assert "--batch-size" in result.stdout
    assert "--seq-len" in result.stdout
    assert "--steps" in result.stdout
    assert "--dtype" in result.stdout
    assert "--no-compile" in result.stdout
    assert "--checkpoint-dir" in result.stdout
    assert "--checkpoint-path" in result.stdout
    assert "--checkpoint-save-interval" in result.stdout
    assert "--resume-from" in result.stdout
    assert "--valid-npz-path" in result.stdout
    assert "--dataset-format" in result.stdout
    assert "--valid-dataset-path" in result.stdout
    assert "--valid-dataset-format" in result.stdout
    assert "--eval-batches" in result.stdout
    assert "--dry-run-json" in result.stdout


def test_dry_run_json_reports_dataset_and_model_plan(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, vocab_size=32, include_structure=True)

    result = run_script(
        str(npz_path),
        "--dry-run-json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "3",
        "--dtype",
        "float32",
        "--hidden-size",
        "8",
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["config"]["npz_path"] == str(npz_path)
    assert payload["config"]["compile"] is True
    assert payload["dataset"]["num_samples"] == 8
    assert payload["dataset"]["num_batches"] == 4
    assert payload["dataset"]["dropped_samples"] == 0
    assert payload["dataset"]["metadata"]["vocab_size"] == 32
    assert payload["dataset"]["side_channels"] == [
        "attention_mask",
        "dep_levels",
        "structure_ids",
    ]
    assert payload["model_config"]["vocab_size"] == 32
    assert payload["tokens_per_step"] == 6
    assert payload["planned_steps"] == 3
    assert payload["dtype"] == "float32"
    assert "default_device" in payload["device"]


def test_dry_run_json_accepts_suffixless_megatron_prefix(tmp_path: Path) -> None:
    prefix = tmp_path / "clang_semantic_4k_v10_train"
    write_mmididx(prefix, dtype=np.dtype(np.int32), vocab_size=32)

    result = run_script(
        str(prefix),
        "--dataset-format",
        "megatron",
        "--dry-run-json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "3",
        "--dtype",
        "float32",
        "--hidden-size",
        "8",
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["config"]["npz_path"] == str(prefix)
    assert payload["config"]["dataset_format"] == "megatron"
    assert payload["dataset"]["path"] == str(prefix)
    assert payload["dataset"]["metadata"]["source_format"] == "megatron"
    assert payload["dataset"]["metadata"]["vocab_size"] == 32
    assert payload["dataset"]["index_metadata"]["dtype"] == "int32"
    assert payload["dataset"]["index_metadata"]["sequence_count"] == 2
    assert payload["dataset"]["index_metadata"]["document_count"] == 2
    assert payload["dataset"]["side_channels"] == []
    assert "megatron_indexed_receipt" not in payload["dataset"]
    receipt = payload["dataset"]["dataset_receipt"]
    assert receipt["source_format"] == "megatron"
    assert receipt["source_dataset_name"] == "clang_semantic_4k_v10_train"
    assert receipt["source_path"] == str(prefix)
    assert receipt["token_key"] == "tokens"
    assert receipt["seq_len"] == 4
    assert receipt["batch_size"] == 2
    assert receipt["num_samples"] == payload["dataset"]["num_samples"]
    assert receipt["num_batches"] == payload["dataset"]["num_batches"]
    assert receipt["dropped_samples"] == payload["dataset"]["dropped_samples"]
    assert receipt["side_channels"] == []
    assert receipt["index_metadata"] == payload["dataset"]["index_metadata"]
    megatron_receipt = receipt["megatron_indexed_receipt"]
    assert megatron_receipt["ingress"] == "MegatronIndexedDataset"
    assert megatron_receipt["path_accepts_suffixless_prefix"] is True
    assert megatron_receipt["sidecar_schema"] == (
        "explicit_token_aligned_binary_side_channel_paths"
    )
    assert megatron_receipt["local_only"] is True
    assert megatron_receipt["receipt_scope"] == "local_mlx_training_ingress"
    assert megatron_receipt["megatron_runtime_imported"] is False
    assert megatron_receipt["distributed_megatron_parity_claim"] is False
    assert megatron_receipt["gb10_training_correctness_claim"] is False
    assert megatron_receipt["m4_vs_gb10_throughput_parity_claim"] is False
    assert payload["model_config"]["vocab_size"] == 32


def test_invalid_token_vocab_returns_error_json(tmp_path: Path) -> None:
    npz_path = tmp_path / "bad_tokens.npz"
    write_npz(npz_path, vocab_size=16)

    result = run_script(
        str(npz_path),
        "--dry-run-json",
        "--vocab-size",
        "8",
        "--seq-len",
        "4",
        "--hidden-size",
        "8",
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "exceeds vocab_size=8" in payload["error"]
    assert payload["compile_enabled"] is True
    assert payload["compile_plan"]["enabled"] is True


def test_valid_dataset_format_without_validation_path_returns_error_json(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(
        str(npz_path),
        "--json",
        "--valid-dataset-format",
        "npz",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert (
        payload["error"]
        == "valid_dataset_format requires valid_dataset_path or valid_npz_path"
    )
    assert payload["compile"] is True
    assert payload["compile_enabled"] is True
    assert payload["compile_plan"]["enabled"] is True


def test_one_eager_training_step_reports_machine_readable_metrics(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, vocab_size=32, include_structure=True)

    result = run_script(
        str(npz_path),
        "--json",
        "--no-compile",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["compile"] is False
    assert payload["dtype"] == "float32"
    assert payload["steps"] == 1
    assert payload["tokens_per_step"] == 6
    assert payload["trained_tokens"] == 6
    assert payload["final_loss"] > 0
    assert payload["mean_step_time_s"] > 0
    assert payload["tokens_per_second"] > 0
    assert payload["parameter_count"] > 0
    assert payload["model_source"] == "cppmega_mlx.models.tiny_lm"
    assert payload["step_metrics"][0]["compiled"] is False


def test_one_eager_training_step_accepts_suffixless_megatron_prefix(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "clang_semantic_4k_v10_train"
    write_mmididx(prefix, dtype=np.dtype(np.int32), vocab_size=32)

    result = run_script(
        str(prefix),
        "--json",
        "--no-compile",
        "--dataset-format",
        "megatron",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["dataset"]["path"] == str(prefix)
    assert payload["dataset"]["metadata"]["source_format"] == "megatron"
    assert payload["dataset"]["index_metadata"]["dtype"] == "int32"
    assert payload["dataset"]["side_channels"] == []
    assert "megatron_indexed_receipt" not in payload["dataset"]
    receipt = payload["dataset"]["dataset_receipt"]
    assert receipt["source_format"] == "megatron"
    assert receipt["source_dataset_name"] == "clang_semantic_4k_v10_train"
    assert receipt["source_path"] == str(prefix)
    assert receipt["token_key"] == "tokens"
    assert receipt["seq_len"] == 4
    assert receipt["batch_size"] == 2
    assert receipt["num_samples"] == payload["dataset"]["num_samples"]
    assert receipt["num_batches"] == payload["dataset"]["num_batches"]
    assert receipt["dropped_samples"] == payload["dataset"]["dropped_samples"]
    assert receipt["side_channels"] == []
    assert receipt["index_metadata"] == payload["dataset"]["index_metadata"]
    megatron_receipt = receipt["megatron_indexed_receipt"]
    assert megatron_receipt["ingress"] == "MegatronIndexedDataset"
    assert megatron_receipt["path_accepts_suffixless_prefix"] is True
    assert megatron_receipt["sidecar_schema"] == (
        "explicit_token_aligned_binary_side_channel_paths"
    )
    assert megatron_receipt["local_only"] is True
    assert megatron_receipt["receipt_scope"] == "local_mlx_training_ingress"
    assert megatron_receipt["megatron_runtime_imported"] is False
    assert megatron_receipt["distributed_megatron_parity_claim"] is False
    assert megatron_receipt["gb10_training_correctness_claim"] is False
    assert megatron_receipt["m4_vs_gb10_throughput_parity_claim"] is False
    assert payload["compile"] is False
    assert payload["tokens_per_step"] == 6
    assert payload["trained_tokens"] == 6
    assert payload["step_metrics"][0]["compiled"] is False


def test_training_reports_validation_eval_metrics(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    valid_npz_path = tmp_path / "valid_tokens.npz"
    write_npz(npz_path, vocab_size=32, include_structure=True)
    write_npz(valid_npz_path, vocab_size=32, include_structure=True)

    result = run_script(
        str(npz_path),
        "--json",
        "--no-compile",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
        "--valid-npz-path",
        str(valid_npz_path),
        "--eval-batches",
        "1",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["dataset"]["path"] == str(npz_path)
    assert payload["evaluation"]["dataset"]["path"] == str(valid_npz_path)
    assert payload["evaluation"]["requested_batches"] == 1
    assert payload["evaluation"]["evaluated_batches"] == 1
    assert payload["evaluation"]["metrics"]["batches"] == 1
    assert payload["evaluation"]["metrics"]["ntokens"] == 6
    assert payload["evaluation"]["metrics"]["loss"] > 0
    assert payload["evaluation"]["metrics"]["tokens_per_second"] > 0
    assert "evaluation" not in payload["checkpoints"]


def test_training_validation_reports_full_side_channel_variant(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    valid_npz_path = tmp_path / "valid_tokens.npz"
    write_npz(npz_path, vocab_size=32, full_structure=True)
    write_npz(valid_npz_path, vocab_size=32, full_structure=True)

    result = run_script(
        str(npz_path),
        "--json",
        "--no-compile",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
        "--valid-dataset-path",
        str(valid_npz_path),
        "--valid-dataset-format",
        "npz",
        "--eval-batches",
        "1",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["dataset"]["side_channels"] == FULL_SIDE_CHANNELS
    assert payload["evaluation"]["dataset"]["path"] == str(valid_npz_path)
    assert payload["evaluation"]["dataset"]["side_channels"] == FULL_SIDE_CHANNELS
    assert payload["evaluation"]["metrics"]["batches"] == 1
    assert payload["evaluation"]["metrics"]["ntokens"] == 6


def test_dry_run_json_reports_validation_eval_plan(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    valid_npz_path = tmp_path / "valid_tokens.npz"
    write_npz(npz_path, vocab_size=32)
    write_npz(valid_npz_path, vocab_size=32)

    result = run_script(
        str(npz_path),
        "--dry-run-json",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
        "--valid-npz-path",
        str(valid_npz_path),
        "--eval-batches",
        "2",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["evaluation"]["dataset"]["path"] == str(valid_npz_path)
    assert payload["evaluation"]["requested_batches"] == 2
    assert payload["evaluation"]["planned_batches"] == 2


def test_dry_run_json_reports_full_validation_side_channel_plan(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    valid_npz_path = tmp_path / "valid_tokens.npz"
    write_npz(npz_path, vocab_size=32, full_structure=True)
    write_npz(valid_npz_path, vocab_size=32, full_structure=True)

    result = run_script(
        str(npz_path),
        "--dry-run-json",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
        "--valid-dataset-path",
        str(valid_npz_path),
        "--valid-dataset-format",
        "npz",
        "--eval-batches",
        "2",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["dataset"]["side_channels"] == FULL_SIDE_CHANNELS
    assert payload["evaluation"]["dataset"]["path"] == str(valid_npz_path)
    assert payload["evaluation"]["dataset"]["side_channels"] == FULL_SIDE_CHANNELS
    assert payload["evaluation"]["requested_batches"] == 2
    assert payload["evaluation"]["planned_batches"] == 2


def test_one_compiled_training_step_reports_compiled_metrics(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, vocab_size=32)

    result = run_script(
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
        timeout=45,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["compile"] is True
    assert payload["tokens_per_step"] == 6
    assert payload["trained_tokens"] == 6
    assert payload["step_metrics"][0]["compiled"] is True


def test_mlx_disable_compile_env_reports_requested_but_eager_execution(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, vocab_size=32)
    env = {**os.environ, "MLX_DISABLE_COMPILE": "1"}

    result = run_script(
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
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


def test_checkpoint_save_and_resume_reports_manifest_contract(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    final_checkpoint = tmp_path / "final"
    write_npz(npz_path, vocab_size=32, include_structure=True)

    first = run_script(
        str(npz_path),
        "--json",
        "--no-compile",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
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
    assert first_payload["start_step"] == 0
    assert first_payload["end_step"] == 1
    assert first_payload["checkpoints"]["saved"][0]["step"] == 1
    assert first_payload["checkpoints"]["final"]["step"] == 1
    assert (checkpoint_dir / "checkpoint-000001" / "model.safetensors").exists()
    manifest_path = checkpoint_dir / "checkpoint-000001" / "metadata.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["step"] == 1
    assert manifest["trained_tokens"] == first_payload["trained_tokens"]
    assert manifest["optimizer"]["present"] is True
    assert manifest["batch_cursor"]["global_batch_offset"] == 1
    _assert_update_boundary_training_state(
        manifest,
        step=1,
        trained_tokens=first_payload["trained_tokens"],
        compiled=False,
    )

    second = run_script(
        str(npz_path),
        "--json",
        "--no-compile",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
        "--resume-from",
        str(checkpoint_dir / "checkpoint-000001"),
        "--checkpoint-path",
        str(tmp_path / "resumed-final"),
    )

    assert second.returncode == 0, second.stderr
    second_payload = json.loads(second.stdout)
    assert second_payload["status"] == "ok"
    assert second_payload["resume"]["loaded"] is True
    assert second_payload["resume"]["step"] == 1
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
    assert resumed_manifest["trained_tokens"] == second_payload["trained_tokens"]
    assert resumed_manifest["batch_cursor"]["global_batch_offset"] == 2
    assert resumed_manifest["batch_cursor"]["batch_offset"] == 2
    _assert_update_boundary_training_state(
        resumed_manifest,
        step=2,
        trained_tokens=second_payload["trained_tokens"],
        compiled=False,
    )


def test_resume_checkpoint_cursor_advances_from_nonzero_global_offset(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "tokens.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    resumed_final = tmp_path / "resumed-final"
    write_npz(npz_path, vocab_size=32, include_structure=True)

    first = run_script(
        str(npz_path),
        "--json",
        "--no-compile",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
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
        "--no-compile",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
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
        "--no-compile",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
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
        "--no-compile",
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
        "--num-heads",
        "1",
        "--ffn-hidden-size",
        "16",
        "--resume-from",
        str(checkpoint_path),
    )

    assert resumed.returncode == 2
    payload = json.loads(resumed.stdout)
    assert payload["status"] == "error"
    assert error in payload["error"]
    assert payload["compile"] is False
    assert payload["compile_enabled"] is False
