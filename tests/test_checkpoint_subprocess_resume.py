from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_hybrid_tiny.py"


def _write_tiny_npz(path: Path, *, samples: int = 128, seq_len: int = 4) -> None:
    vocab_size = 32
    tokens = (
        np.arange(samples * seq_len, dtype=np.int32) % vocab_size
    ).reshape(samples, seq_len)
    np.savez(
        path,
        tokens=tokens,
        attention_mask=np.ones_like(tokens, dtype=np.float32),
        structure_ids=(tokens % 7).astype(np.int32),
        dep_levels=(tokens % 3).astype(np.int32),
        ast_depth_ids=(tokens % 11).astype(np.int32),
        sibling_index_ids=(tokens % 13).astype(np.int32),
        node_type_ids=(tokens % 17).astype(np.int32),
        vocab_size=np.array(vocab_size, dtype=np.int64),
        tokenizer_contract=np.array("local_profile"),
    )


def _run_train(*args: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    assert payload["status"] == "ok"
    return payload


def _tiny_hybrid_args(npz_path: Path, *, steps: int) -> list[str]:
    return [
        str(npz_path),
        "--json",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        str(steps),
        "--dtype",
        "float32",
        "--seed",
        "17",
        "--shuffle",
        "--lr",
        "0.001",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "AEMR",
        "--depth",
        "4",
    ]


def test_subprocess_checkpoint_resume_matches_uninterrupted_100_step_suffix(
    tmp_path: Path,
) -> None:
    interrupt_steps = 37
    post_resume_steps = 100
    total_steps = interrupt_steps + post_resume_steps
    npz_path = tmp_path / "tokens.npz"
    checkpoint_dir = tmp_path / "checkpoints"
    resumed_final = tmp_path / "resumed-final"
    _write_tiny_npz(npz_path)

    uninterrupted = _run_train(*_tiny_hybrid_args(npz_path, steps=total_steps))
    first = _run_train(
        *_tiny_hybrid_args(npz_path, steps=interrupt_steps),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-save-interval",
        str(interrupt_steps),
    )
    checkpoint_path = checkpoint_dir / f"checkpoint-{interrupt_steps:06d}"
    resumed = _run_train(
        *_tiny_hybrid_args(npz_path, steps=post_resume_steps),
        "--resume-from",
        str(checkpoint_path),
        "--checkpoint-path",
        str(resumed_final),
    )

    assert first["step_metrics"][-1]["step"] == interrupt_steps
    assert resumed["resume"]["loaded"] is True
    assert resumed["resume"]["step"] == interrupt_steps
    assert resumed["start_step"] == interrupt_steps
    assert resumed["end_step"] == total_steps
    assert [item["step"] for item in resumed["step_metrics"]] == list(
        range(interrupt_steps + 1, total_steps + 1)
    )
    np.testing.assert_allclose(
        [item["loss"] for item in resumed["step_metrics"]],
        [item["loss"] for item in uninterrupted["step_metrics"][interrupt_steps:]],
        rtol=1e-5,
        atol=1e-5,
    )

    resumed_manifest = json.loads((resumed_final / "metadata.json").read_text())
    assert resumed_manifest["step"] == total_steps
    assert resumed_manifest["batch_cursor"]["global_batch_offset"] == total_steps
    assert resumed_manifest["resume_cursor"]["step"] == total_steps
    assert (
        resumed_manifest["resume_cursor"]["batch_cursor"]["global_batch_offset"]
        == total_steps
    )
    assert resumed_manifest["training_state"]["state"]["step"] == total_steps
    assert resumed_manifest["optimizer"]["present"] is True
    assert resumed_manifest["rng"]["mode"] == "snapshot"
    assert resumed_manifest["rng"]["snapshot"]["scope"] == "single_process_local"
