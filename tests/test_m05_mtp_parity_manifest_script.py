from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cppmega_mlx.training.parity import (
    M05_MTP_BETA,
    M05_MTP_CUDA_ARTIFACT_CONTRACT,
    M05_MTP_CUDA_ARTIFACT_FORMAT,
    M05_MTP_CUDA_ARTIFACT_PATH,
    M05_MTP_CUDA_REFERENCE_SOURCES,
    M05_MTP_DEPTH,
    M05_MTP_LAMBDA,
    M05_MTP_PARITY_ISSUE_ID,
    M05_MTP_PARITY_OUTPUT,
    M05_MTP_PARITY_PROFILE,
    m05_loss_values_sha256,
    validate_m05_mtp_parity_manifest_dict,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m05_mtp_parity_manifest.py"
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


def valid_cuda_artifact_payload() -> dict[str, Any]:
    loss_values = {
        "next_token_loss": 2.0,
        "mtp_loss": 1.4375,
        "total_loss": 2.43125,
        "mtp_depth_1_loss": 1.25,
        "mtp_depth_2_loss": 1.75,
        "grad_norm": 3.5,
    }
    return {
        "format": M05_MTP_CUDA_ARTIFACT_FORMAT,
        "profile": M05_MTP_PARITY_PROFILE,
        "mtp_depth": M05_MTP_DEPTH,
        "mtp_beta": M05_MTP_BETA,
        "mtp_lambda": M05_MTP_LAMBDA,
        "cuda_reference_sources": list(M05_MTP_CUDA_REFERENCE_SOURCES),
        "source_sha256": {
            "cppmega/megatron/fastmtp_layer.py": "1" * 64,
            "cppmega/megatron/mtp_native_hopper_ce.py": "2" * 64,
        },
        "loss_values": loss_values,
        "loss_values_sha256": m05_loss_values_sha256(loss_values),
        "source_commit": "cuda-fastmtp-abc123",
        "hardware": "GB10 CUDA reference",
        "cuda_runtime": "CUDA 13.0",
        "acceptance_claim": False,
        "numerical_harness_passed": False,
        "numerical_parity_passed": False,
        "full_m0_5_acceptance": False,
        "full_m0_5_acceptance_claim": False,
        "gb10_mtp_parity_claim": False,
        "m4_vs_gb10_mtp_parity_claim": False,
        "distributed_megatron_mtp_parity_claim": False,
        "evaluated_by_numerical_harness": False,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def assert_m05_fail_closed_invariants(payload: dict[str, Any]) -> None:
    validate_m05_mtp_parity_manifest_dict(payload)
    assert payload["issue"]["id"] == M05_MTP_PARITY_ISSUE_ID
    assert payload["profile"] == M05_MTP_PARITY_PROFILE
    assert payload["m0_5_closed"] is False
    assert payload["full_m0_5_acceptance_claim"] is False
    assert payload["gb10_mtp_parity_claim"] is False
    assert payload["m4_vs_gb10_mtp_parity_claim"] is False
    assert payload["distributed_megatron_mtp_parity_claim"] is False
    assert payload["acceptance_status"] == "not_evaluated"
    assert payload["num_receipts"] == 0
    assert payload["receipts"] == []
    assert "no FastMTP numerical parity" in payload["claim_boundary"]
    assert payload["mtp_config"] == {
        "depth": M05_MTP_DEPTH,
        "beta": M05_MTP_BETA,
        "lambda": M05_MTP_LAMBDA,
        "profile": M05_MTP_PARITY_PROFILE,
    }
    identity = payload["cuda_reference_identity"]
    assert identity["required_sources"] == list(M05_MTP_CUDA_REFERENCE_SOURCES)
    assert "cppmega/megatron/fastmtp_layer.py" in identity["required_sources"]
    assert "cppmega/megatron/mtp_native_hopper_ce.py" in identity["required_sources"]
    assert identity["artifact_contract"] == M05_MTP_CUDA_ARTIFACT_CONTRACT
    assert identity["requires_hopper_liger_fused_ce_receipt"] is True
    assert payload["cuda_reference"]["required_artifact"] == M05_MTP_CUDA_ARTIFACT_PATH
    assert payload["cuda_reference"]["artifact_contract"] == M05_MTP_CUDA_ARTIFACT_CONTRACT
    assert payload["cuda_reference"]["evaluates_mtp_losses"] is False
    assert payload["cuda_reference"]["preflight_is_acceptance"] is False
    assert payload["cuda_reference"]["evaluated_by_this_manifest"] is False
    assert payload["mlx_reference"]["status"] == "not_evaluated"
    assert payload["mlx_reference"]["evaluated_by_this_manifest"] is False
    assert payload["mlx_reference"]["local_mlx_mtp_losses_evaluated"] is False
    gate = payload["acceptance_gate"]
    assert gate["requires_cuda_reference_artifact"] is True
    assert gate["requires_external_cuda_mtp_losses"] is True
    assert gate["requires_external_cuda_grad_norm"] is True
    assert gate["requires_mlx_mtp_losses"] is True
    assert gate["requires_matching_mtp_depth"] is True
    assert gate["requires_matching_beta_lambda"] is True
    assert gate["requires_separate_numerical_harness"] is True
    assert gate["requires_hopper_liger_fused_ce_receipt"] is True
    assert gate["receipts_evaluated_by_this_manifest"] is False
    assert gate["full_m0_5_acceptance"] is False
    assert gate["receipt_count"] == 0
    assert gate["mtp_receipts"] == 0


def test_script_writes_blocked_manifest_without_cuda_artifact(tmp_path: Path) -> None:
    output = tmp_path / "m05_fastmtp.json"
    result = run_script("--no-cuda-reference-artifact", "--output", str(output), "--json")
    payload = load_json_result(result)

    assert output.exists()
    assert json.loads(output.read_text()) == payload
    assert_m05_fail_closed_invariants(payload)
    assert payload["status"] == "blocked"
    assert payload["cuda_reference"]["status"] == "blocked_missing_artifact"
    assert payload["cuda_reference"]["artifact"] is None
    assert payload["cuda_reference"]["artifact_supplied"] is False
    assert payload["cuda_reference"]["artifact_preflight_status"] == "not_supplied"
    assert payload["acceptance_gate"]["cuda_reference_artifact_supplied"] is False
    assert payload["acceptance_gate"]["cuda_reference_artifact_preflight_status"] == "not_supplied"
    assert "No CUDA FastMTP reference artifact" in payload["acceptance_gate"]["blocker"]


def test_script_refuses_missing_default_cuda_artifact(tmp_path: Path) -> None:
    output = tmp_path / "m05_fastmtp.json"
    result = run_script("--output", str(output), "--json")
    payload = load_json_result(result)

    assert json.loads(output.read_text()) == payload
    assert_m05_fail_closed_invariants(payload)
    assert payload["status"] == "refused"
    assert payload["cuda_reference"]["status"] == "refused_missing_artifact"
    assert payload["cuda_reference"]["artifact"].endswith(M05_MTP_CUDA_ARTIFACT_PATH)
    assert payload["cuda_reference"]["artifact_supplied"] is True
    assert payload["cuda_reference"]["artifact_preflight_status"] == "missing"
    assert "not found" in payload["cuda_reference"]["artifact_error"]
    assert payload["acceptance_gate"]["cuda_reference_artifact_supplied"] is True
    assert payload["acceptance_gate"]["cuda_reference_artifact_preflight_status"] == "missing"
    assert payload["acceptance_gate"]["full_m0_5_acceptance"] is False


def test_checked_in_m05_manifest_is_fail_closed_missing_cuda_artifact() -> None:
    manifest_path = ROOT / M05_MTP_PARITY_OUTPUT
    payload = json.loads(manifest_path.read_text())

    assert_m05_fail_closed_invariants(payload)
    assert payload["status"] == "refused"
    assert payload["m0_5_closed"] is False
    assert payload["acceptance_status"] == "not_evaluated"
    assert payload["cuda_reference"]["artifact_preflight_status"] == "missing"
    assert payload["cuda_reference"]["status"] == "refused_missing_artifact"
    assert payload["acceptance_gate"]["full_m0_5_acceptance"] is False


def test_script_records_valid_cuda_artifact_but_still_refuses_closure(tmp_path: Path) -> None:
    cuda_artifact = tmp_path / "cuda_fastmtp.json"
    output = tmp_path / "m05_fastmtp.json"
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
    assert_m05_fail_closed_invariants(payload)
    assert payload["status"] == "refused"
    assert payload["cuda_reference"]["status"] == "refused_not_evaluated"
    assert payload["cuda_reference"]["artifact"] == str(cuda_artifact)
    assert payload["cuda_reference"]["artifact_supplied"] is True
    assert payload["cuda_reference"]["artifact_preflight_status"] == "valid_not_evaluated"
    assert payload["cuda_reference"]["artifact_format"] == M05_MTP_CUDA_ARTIFACT_FORMAT
    assert (
        payload["cuda_reference"]["artifact_loss_values_sha256"]
        == valid_cuda_artifact_payload()["loss_values_sha256"]
    )
    assert payload["cuda_reference"]["artifact_source_commit"] == "cuda-fastmtp-abc123"
    assert payload["cuda_reference"]["artifact_hardware"] == "GB10 CUDA reference"
    assert payload["cuda_reference"]["artifact_cuda_runtime"] == "CUDA 13.0"
    assert payload["cuda_reference"]["artifact_error"] is None
    assert payload["acceptance_gate"]["cuda_reference_artifact_supplied"] is True
    assert (
        payload["acceptance_gate"]["cuda_reference_artifact_preflight_status"]
        == "valid_not_evaluated"
    )
    assert "not compared" in payload["acceptance_gate"]["blocker"]
    assert payload["acceptance_gate"]["full_m0_5_acceptance"] is False


@pytest.mark.parametrize(
    ("artifact_payload", "error"),
    [
        ("not-json", "not valid JSON"),
        ([], "JSON must be an object"),
        (
            {
                **valid_cuda_artifact_payload(),
                "loss_values": {
                    **valid_cuda_artifact_payload()["loss_values"],
                    "mtp_loss": 1.5,
                    "total_loss": 2.45,
                },
            },
            "beta-normalized weighted per-depth MTP loss",
        ),
        (
            {
                **valid_cuda_artifact_payload(),
                "loss_values": {
                    **valid_cuda_artifact_payload()["loss_values"],
                    "total_loss": 2.46,
                },
            },
            "total_loss",
        ),
        (
            {
                **valid_cuda_artifact_payload(),
                "cuda_reference_sources": [
                    *M05_MTP_CUDA_REFERENCE_SOURCES,
                    "cppmega/megatron/extra_fastmtp.py",
                ],
            },
            "exactly match",
        ),
        (
            {
                **valid_cuda_artifact_payload(),
                "hardware": "M4 CUDA reference",
            },
            "GB10 and CUDA",
        ),
        (
            {
                **valid_cuda_artifact_payload(),
                "hardware": "GB10 Metal reference",
            },
            "GB10 and CUDA",
        ),
        (
            {
                **valid_cuda_artifact_payload(),
                "cuda_runtime": "not CUDA",
            },
            "cuda_runtime",
        ),
        (
            {
                **valid_cuda_artifact_payload(),
                "loss_values_sha256": "3" * 64,
            },
            "loss_values_sha256",
        ),
    ],
)
def test_script_refuses_invalid_cuda_artifact_contract(
    tmp_path: Path,
    artifact_payload: Any,
    error: str,
) -> None:
    cuda_artifact = tmp_path / "cuda_fastmtp.json"
    output = tmp_path / "m05_fastmtp.json"
    if artifact_payload == "not-json":
        cuda_artifact.write_text("{not json\n", encoding="utf-8")
    else:
        write_json(cuda_artifact, artifact_payload)

    result = run_script(
        "--cuda-reference-artifact",
        str(cuda_artifact),
        "--output",
        str(output),
        "--json",
    )
    payload = load_json_result(result)

    assert_m05_fail_closed_invariants(payload)
    assert payload["status"] == "refused"
    assert payload["cuda_reference"]["status"] == "refused_invalid_artifact"
    assert json.loads(output.read_text()) == payload
    assert payload["cuda_reference"]["artifact_preflight_status"] == "invalid"
    assert error in payload["cuda_reference"]["artifact_error"]
    assert payload["acceptance_gate"]["cuda_reference_artifact_preflight_status"] == "invalid"
    assert payload["acceptance_gate"]["full_m0_5_acceptance"] is False


def test_script_refuses_cuda_artifact_directory(tmp_path: Path) -> None:
    output = tmp_path / "m05_fastmtp.json"
    result = run_script(
        "--cuda-reference-artifact",
        str(tmp_path),
        "--output",
        str(output),
        "--json",
    )
    payload = load_json_result(result)

    assert_m05_fail_closed_invariants(payload)
    assert payload["status"] == "refused"
    assert payload["cuda_reference"]["status"] == "refused_invalid_artifact"
    assert payload["cuda_reference"]["artifact_preflight_status"] == "invalid"
    assert "not a regular file" in payload["cuda_reference"]["artifact_error"]
    assert payload["acceptance_gate"]["full_m0_5_acceptance"] is False
