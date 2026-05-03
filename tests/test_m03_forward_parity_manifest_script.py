from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cppmega_mlx.training.parity import (
    M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT,
    M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT,
    M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH,
    M03_FORWARD_PARITY_LOGITS_NUMEL,
    M03_FORWARD_PARITY_LOGITS_SHAPE,
    M03_FORWARD_PARITY_LOGITS_TENSOR_NAME,
    validate_m03_forward_parity_manifest_dict,
)
from scripts.m03_forward_parity_manifest import deterministic_input_tokens, input_tokens_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m03_forward_parity_manifest.py"
PYTHON = ROOT / ".venv" / "bin" / "python"


def run_script(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON if PYTHON.exists() else sys.executable), str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def load_json_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def fixed_input_hash() -> str:
    tokens = deterministic_input_tokens(seed=3003, batch_size=1, seq_len=512, vocab_size=65_536)
    return input_tokens_sha256(tokens)


def valid_cuda_artifact_payload(*, input_hash: str | None = None) -> dict[str, Any]:
    return {
        "format": M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT,
        "profile": "local_gb10_quarter",
        "tensor_name": M03_FORWARD_PARITY_LOGITS_TENSOR_NAME,
        "seed": 3003,
        "batch_size": 1,
        "seq_len": 512,
        "vocab_size": 65_536,
        "shape": list(M03_FORWARD_PARITY_LOGITS_SHAPE),
        "dtype": "bf16",
        "logits_dtype": "bf16",
        "input_tokens_sha256": input_hash or fixed_input_hash(),
        "logits_sha256": "1" * 64,
        "source_commit": "cuda-ref-abc123",
        "hardware": "GB10 CUDA reference",
        "cuda_runtime": "CUDA 13.0",
        "logits_summary": {
            "numel": M03_FORWARD_PARITY_LOGITS_NUMEL,
            "min": -1.0,
            "max": 1.0,
            "mean": 0.0,
            "std": 0.5,
            "l2_norm": 10.0,
            "max_abs": 1.0,
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_deterministic_input_hash_is_seed_stable() -> None:
    first = deterministic_input_tokens(seed=3003, batch_size=1, seq_len=8, vocab_size=64)
    second = deterministic_input_tokens(seed=3003, batch_size=1, seq_len=8, vocab_size=64)
    different = deterministic_input_tokens(seed=3004, batch_size=1, seq_len=8, vocab_size=64)

    assert first.dtype == np.dtype(np.uint32)
    assert first.shape == (1, 8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert input_tokens_sha256(first) == input_tokens_sha256(second)
    assert input_tokens_sha256(first) != input_tokens_sha256(different)


def test_script_writes_blocked_manifest_without_cuda_artifact(tmp_path: Path) -> None:
    output = tmp_path / "m03_random_init.json"
    result = run_script("--no-cuda-reference-artifact", "--output", str(output), "--json")
    payload = load_json_result(result)

    assert output.exists()
    assert json.loads(output.read_text()) == payload
    validate_m03_forward_parity_manifest_dict(payload)
    assert payload["status"] == "blocked"
    assert payload["issue"]["id"] == "cppmega-mlx-t8f.3"
    assert payload["profile"] == "local_gb10_quarter"
    assert payload["num_receipts"] == 0
    assert payload["acceptance_status"] == "not_evaluated"
    assert payload["m0_3_closed"] is False
    assert payload["full_m0_3_acceptance_claim"] is False
    assert payload["cuda_weight_import"] is False
    assert payload["warm_start"] is False
    assert payload["gb10_forward_parity_claim"] is False
    assert payload["m4_vs_gb10_forward_parity_claim"] is False
    assert payload["distributed_megatron_forward_parity_claim"] is False
    assert "no GB10-vs-M4" in payload["claim_boundary"]
    assert payload["cuda_reference"]["status"] == "blocked_missing_artifact"
    assert payload["cuda_reference"]["artifact"] is None
    assert payload["cuda_reference"]["artifact_supplied"] is False
    assert payload["cuda_reference"]["artifact_preflight_status"] == "not_supplied"
    assert payload["cuda_reference"]["required_artifact"] == M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH
    assert payload["cuda_reference"]["artifact_contract"] == M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT
    assert payload["cuda_reference"]["artifact_error"] == "No CUDA reference artifact path was supplied."
    assert payload["cuda_reference"]["evaluates_logits"] is False
    assert payload["cuda_reference"]["preflight_is_acceptance"] is False
    assert payload["cuda_reference"]["evaluated_by_this_manifest"] is False
    assert payload["mlx_reference"]["status"] == "not_evaluated"
    assert payload["mlx_reference"]["evaluated_by_this_manifest"] is False
    assert payload["mlx_reference"]["readiness_status"] == "pass"
    assert payload["mlx_reference"]["local_mlx_forward_executed"] is True
    assert payload["mlx_reference"]["local_mlx_forward_scope"] == "tiny_smoke_only"
    assert payload["mlx_reference"]["readiness_is_cuda_parity"] is False
    assert payload["mlx_reference"]["full_profile_allocation_executed"] is False
    assert payload["mlx_reference"]["full_profile_forward_executed"] is False
    assert payload["mlx_reference"]["tiny_smoke_forward_executed"] is True
    assert (
        payload["mlx_reference"]["closure_required_mlx_forward_scope"]
        == "full_local_gb10_quarter_logits"
    )
    assert "does not allocate or forward the full profile" in payload["mlx_reference"]["execution_note"]
    assert payload["mlx_reference"]["full_profile"]["name"] == "local_gb10_quarter"
    assert payload["mlx_reference"]["full_profile"]["depth"] == 13
    assert payload["mlx_reference"]["full_profile"]["hidden_size"] == 3584
    assert payload["mlx_reference"]["full_profile"]["route_symbols"] == [
        "A",
        "E",
        "M",
        "E",
        "A",
        "E",
        "M",
        "E",
        "A",
        "E",
        "M",
        "R",
        "A",
    ]
    smoke = payload["mlx_reference"]["tiny_smoke_forward"]
    assert smoke["seed"] == 3003
    assert smoke["input_tokens_modulo_vocab"] is True
    assert smoke["logits_shape"] == [1, 512, 256]
    assert smoke["logits_all_finite"] is True
    assert smoke["logits_sha256"]
    assert payload["acceptance_gate"]["requires_cuda_reference_artifact"] is True
    assert payload["acceptance_gate"]["requires_external_cuda_logits"] is True
    assert payload["acceptance_gate"]["requires_mlx_forward"] is True
    assert payload["acceptance_gate"]["requires_separate_numerical_harness"] is True
    assert payload["acceptance_gate"]["cuda_reference_artifact_supplied"] is False
    assert payload["acceptance_gate"]["cuda_reference_artifact_preflight_status"] == "not_supplied"
    assert payload["acceptance_gate"]["receipts_evaluated_by_this_manifest"] is False
    assert payload["acceptance_gate"]["full_m0_3_acceptance"] is False
    assert payload["metadata"]["scaffold_only"] is True
    assert payload["metadata"]["local_mlx_readiness_status"] == "pass"
    assert payload["input_batch"]["batch_size"] == 1
    assert payload["input_batch"]["seq_len"] == 512
    assert payload["input_batch"]["seed"] == 3003
    assert payload["input_batch"]["vocab_size"] == 65_536
    assert payload["input_batch"]["tokens_sha256"]


def test_script_refuses_missing_default_cuda_artifact(tmp_path: Path) -> None:
    output = tmp_path / "m03_random_init.json"

    result = run_script("--output", str(output), "--json")
    payload = load_json_result(result)

    assert json.loads(output.read_text()) == payload
    validate_m03_forward_parity_manifest_dict(payload)
    assert payload["status"] == "refused"
    assert payload["m0_3_closed"] is False
    assert payload["full_m0_3_acceptance_claim"] is False
    assert payload["cuda_reference"]["status"] == "refused_missing_artifact"
    assert payload["cuda_reference"]["artifact"].endswith(M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH)
    assert payload["cuda_reference"]["artifact_supplied"] is True
    assert payload["cuda_reference"]["artifact_preflight_status"] == "missing"
    assert "not found" in payload["cuda_reference"]["artifact_error"]
    assert payload["cuda_reference"]["evaluated_by_this_manifest"] is False
    assert payload["cuda_reference"]["evaluates_logits"] is False
    assert payload["cuda_reference"]["preflight_is_acceptance"] is False
    assert payload["acceptance_gate"]["cuda_reference_artifact_supplied"] is True
    assert payload["acceptance_gate"]["cuda_reference_artifact_preflight_status"] == "missing"
    assert payload["acceptance_gate"]["full_m0_3_acceptance"] is False


def test_script_refuses_invalid_cuda_artifact(tmp_path: Path) -> None:
    cuda_artifact = tmp_path / "cuda_logits.json"
    output = tmp_path / "m03_random_init.json"
    cuda_artifact.write_text('{"tensor_name": "local_gb10_quarter.logits"}\n', encoding="utf-8")

    result = run_script(
        "--cuda-reference-artifact",
        str(cuda_artifact),
        "--output",
        str(output),
        "--json",
    )
    payload = load_json_result(result)

    assert json.loads(output.read_text()) == payload
    validate_m03_forward_parity_manifest_dict(payload)
    assert payload["status"] == "refused"
    assert payload["acceptance_status"] == "not_evaluated"
    assert payload["m0_3_closed"] is False
    assert payload["full_m0_3_acceptance_claim"] is False
    assert payload["num_receipts"] == 0
    assert payload["receipts"] == []
    assert payload["cuda_reference"]["status"] == "refused_invalid_artifact"
    assert payload["cuda_reference"]["artifact"] == str(cuda_artifact)
    assert payload["cuda_reference"]["artifact_supplied"] is True
    assert payload["cuda_reference"]["artifact_preflight_status"] == "invalid"
    assert "format" in payload["cuda_reference"]["artifact_error"]
    assert payload["cuda_reference"]["required_artifact"] == M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH
    assert payload["cuda_reference"]["artifact_contract"] == M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT
    assert payload["cuda_reference"]["evaluated_by_this_manifest"] is False
    assert payload["cuda_reference"]["evaluates_logits"] is False
    assert payload["cuda_reference"]["preflight_is_acceptance"] is False
    assert payload["mlx_reference"]["status"] == "not_evaluated"
    assert payload["mlx_reference"]["readiness_status"] == "pass"
    assert payload["mlx_reference"]["local_mlx_forward_executed"] is True
    assert payload["mlx_reference"]["local_mlx_forward_scope"] == "tiny_smoke_only"
    assert payload["mlx_reference"]["readiness_is_cuda_parity"] is False
    assert payload["mlx_reference"]["full_profile_allocation_executed"] is False
    assert payload["mlx_reference"]["full_profile_forward_executed"] is False
    assert payload["mlx_reference"]["tiny_smoke_forward_executed"] is True
    assert payload["acceptance_gate"]["cuda_reference_artifact_supplied"] is True
    assert payload["acceptance_gate"]["receipt_count"] == 0
    assert payload["acceptance_gate"]["logits_receipts"] == 0
    assert payload["acceptance_gate"]["cuda_reference_artifact_preflight_status"] == "invalid"
    assert payload["acceptance_gate"]["receipts_evaluated_by_this_manifest"] is False
    assert payload["acceptance_gate"]["full_m0_3_acceptance"] is False
    assert payload["input_batch"]["tokens_sha256"]


def test_script_refuses_malformed_and_non_object_cuda_artifacts(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    non_object = tmp_path / "array.json"
    malformed.write_text("{not-json", encoding="utf-8")
    non_object.write_text("[1, 2, 3]\n", encoding="utf-8")

    malformed_payload = load_json_result(
        run_script(
            "--cuda-reference-artifact",
            str(malformed),
            "--output",
            str(tmp_path / "malformed_out.json"),
            "--json",
        )
    )
    non_object_payload = load_json_result(
        run_script(
            "--cuda-reference-artifact",
            str(non_object),
            "--output",
            str(tmp_path / "array_out.json"),
            "--json",
        )
    )

    assert malformed_payload["cuda_reference"]["status"] == "refused_invalid_artifact"
    assert malformed_payload["cuda_reference"]["artifact_preflight_status"] == "invalid"
    assert "not valid JSON" in malformed_payload["cuda_reference"]["artifact_error"]
    assert non_object_payload["cuda_reference"]["status"] == "refused_invalid_artifact"
    assert non_object_payload["cuda_reference"]["artifact_preflight_status"] == "invalid"
    assert "must be an object" in non_object_payload["cuda_reference"]["artifact_error"]


def test_script_refuses_stale_and_non_numeric_cuda_artifacts(tmp_path: Path) -> None:
    stale = tmp_path / "stale.json"
    non_numeric = tmp_path / "non_numeric.json"
    stale_payload = valid_cuda_artifact_payload(input_hash="2" * 64)
    non_numeric_payload = valid_cuda_artifact_payload()
    non_numeric_payload["logits_summary"]["mean"] = "0.0"
    write_json(stale, stale_payload)
    write_json(non_numeric, non_numeric_payload)

    stale_manifest = load_json_result(
        run_script(
            "--cuda-reference-artifact",
            str(stale),
            "--output",
            str(tmp_path / "stale_out.json"),
            "--json",
        )
    )
    non_numeric_manifest = load_json_result(
        run_script(
            "--cuda-reference-artifact",
            str(non_numeric),
            "--output",
            str(tmp_path / "non_numeric_out.json"),
            "--json",
        )
    )

    assert stale_manifest["cuda_reference"]["artifact_preflight_status"] == "invalid"
    assert "stale" in stale_manifest["cuda_reference"]["artifact_error"]
    assert non_numeric_manifest["cuda_reference"]["artifact_preflight_status"] == "invalid"
    assert "mean must be a finite number" in non_numeric_manifest["cuda_reference"]["artifact_error"]


def test_script_refuses_forged_valid_cuda_artifact_hardware(tmp_path: Path) -> None:
    cuda_artifact = tmp_path / "forged_hardware.json"
    output = tmp_path / "forged_hardware_out.json"
    payload = valid_cuda_artifact_payload()
    payload["hardware"] = "GB10 CUDA Apple fallback"
    write_json(cuda_artifact, payload)

    manifest = load_json_result(
        run_script(
            "--cuda-reference-artifact",
            str(cuda_artifact),
            "--output",
            str(output),
            "--json",
        )
    )

    assert json.loads(output.read_text()) == manifest
    validate_m03_forward_parity_manifest_dict(manifest)
    assert manifest["status"] == "refused"
    assert manifest["cuda_reference"]["status"] == "refused_invalid_artifact"
    assert manifest["cuda_reference"]["artifact_preflight_status"] == "invalid"
    assert "hardware" in manifest["cuda_reference"]["artifact_error"]
    assert "GB10 and CUDA" in manifest["cuda_reference"]["artifact_error"]
    assert manifest["acceptance_gate"]["cuda_reference_artifact_preflight_status"] == "invalid"
    assert manifest["acceptance_gate"]["full_m0_3_acceptance"] is False


def test_script_refuses_forged_valid_cuda_artifact_runtime(tmp_path: Path) -> None:
    cuda_artifact = tmp_path / "forged_runtime.json"
    output = tmp_path / "forged_runtime_out.json"
    payload = valid_cuda_artifact_payload()
    payload["cuda_runtime"] = "CUDA 13.0 via MLX fallback"
    write_json(cuda_artifact, payload)

    manifest = load_json_result(
        run_script(
            "--cuda-reference-artifact",
            str(cuda_artifact),
            "--output",
            str(output),
            "--json",
        )
    )

    assert json.loads(output.read_text()) == manifest
    validate_m03_forward_parity_manifest_dict(manifest)
    assert manifest["status"] == "refused"
    assert manifest["cuda_reference"]["status"] == "refused_invalid_artifact"
    assert manifest["cuda_reference"]["artifact_preflight_status"] == "invalid"
    assert "cuda_runtime" in manifest["cuda_reference"]["artifact_error"]
    assert manifest["acceptance_gate"]["cuda_reference_artifact_preflight_status"] == "invalid"
    assert manifest["acceptance_gate"]["full_m0_3_acceptance"] is False


def test_script_records_valid_cuda_artifact_but_still_refuses_closure(tmp_path: Path) -> None:
    cuda_artifact = tmp_path / "cuda_logits.json"
    output = tmp_path / "m03_random_init.json"
    write_json(cuda_artifact, valid_cuda_artifact_payload())

    result = run_script(
        "--cuda-reference-artifact",
        str(cuda_artifact),
        "--output",
        str(output),
        "--json",
    )
    payload = load_json_result(result)

    assert json.loads(output.read_text()) == payload
    assert payload["status"] == "refused"
    assert payload["acceptance_status"] == "not_evaluated"
    assert payload["m0_3_closed"] is False
    assert payload["full_m0_3_acceptance_claim"] is False
    assert payload["cuda_reference"]["status"] == "refused_not_evaluated"
    assert payload["cuda_reference"]["artifact_preflight_status"] == "valid_not_evaluated"
    assert payload["cuda_reference"]["artifact"] == str(cuda_artifact)
    assert payload["cuda_reference"]["artifact_format"] == M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT
    assert payload["cuda_reference"]["artifact_tensor_name"] == M03_FORWARD_PARITY_LOGITS_TENSOR_NAME
    assert payload["cuda_reference"]["artifact_logits_sha256"] == "1" * 64
    assert payload["cuda_reference"]["artifact_source_commit"] == "cuda-ref-abc123"
    assert payload["cuda_reference"]["artifact_hardware"] == "GB10 CUDA reference"
    assert payload["cuda_reference"]["artifact_cuda_runtime"] == "CUDA 13.0"
    assert payload["cuda_reference"]["artifact_error"] is None
    assert payload["cuda_reference"]["evaluated_by_this_manifest"] is False
    assert payload["cuda_reference"]["evaluates_logits"] is False
    assert payload["cuda_reference"]["preflight_is_acceptance"] is False
    assert payload["acceptance_gate"]["cuda_reference_artifact_supplied"] is True
    assert (
        payload["acceptance_gate"]["cuda_reference_artifact_preflight_status"]
        == "valid_not_evaluated"
    )
    assert payload["acceptance_gate"]["receipts_evaluated_by_this_manifest"] is False
    assert payload["acceptance_gate"]["full_m0_3_acceptance"] is False
    validate_m03_forward_parity_manifest_dict(payload)


def test_script_can_skip_mlx_readiness_but_stays_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "m03_random_init.json"
    result = run_script("--skip-mlx-readiness", "--output", str(output), "--json")
    payload = load_json_result(result)

    validate_m03_forward_parity_manifest_dict(payload)
    assert payload["status"] == "refused"
    assert payload["cuda_reference"]["status"] == "refused_missing_artifact"
    assert payload["cuda_reference"]["artifact_preflight_status"] == "missing"
    assert payload["acceptance_status"] == "not_evaluated"
    assert payload["full_m0_3_acceptance_claim"] is False
    assert payload["mlx_reference"]["status"] == "not_evaluated"
    assert payload["mlx_reference"]["readiness_status"] == "skipped"
    assert payload["mlx_reference"]["local_mlx_forward_executed"] is False
    assert payload["mlx_reference"]["local_mlx_forward_scope"] == "skipped"
    assert payload["mlx_reference"]["readiness_is_cuda_parity"] is False
    assert payload["mlx_reference"]["full_profile_allocation_executed"] is False
    assert payload["mlx_reference"]["full_profile_forward_executed"] is False
    assert payload["mlx_reference"]["tiny_smoke_forward_executed"] is False
    assert (
        payload["mlx_reference"]["closure_required_mlx_forward_scope"]
        == "full_local_gb10_quarter_logits"
    )
    assert payload["acceptance_gate"]["full_m0_3_acceptance"] is False
