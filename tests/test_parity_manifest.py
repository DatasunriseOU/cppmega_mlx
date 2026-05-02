from __future__ import annotations

import json
from typing import Any, cast

import pytest

from cppmega_mlx.training.parity import (
    JsonObject,
    LOCAL_ONLY_POLICY,
    M03_FORWARD_PARITY_ATOL,
    M03_FORWARD_PARITY_BATCH_SIZE,
    M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT,
    M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT,
    M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH,
    M03_FORWARD_PARITY_LOGITS_NUMEL,
    M03_FORWARD_PARITY_LOGITS_SHAPE,
    M03_FORWARD_PARITY_LOGITS_TENSOR_NAME,
    M03_FORWARD_PARITY_FIXED_INPUT,
    M03_FORWARD_PARITY_INPUT_TOKENS_SHA256,
    M03_FORWARD_PARITY_ISSUE_ID,
    M03_FORWARD_PARITY_OUTPUT,
    M03_FORWARD_PARITY_PROFILE,
    M03_FORWARD_PARITY_PROFILE_METADATA,
    M03_FORWARD_PARITY_RECEIPT_SCOPE,
    M03_FORWARD_PARITY_RTOL,
    M03_FORWARD_PARITY_SEED,
    M03_FORWARD_PARITY_SEQ_LEN,
    M03_FORWARD_PARITY_TOLERANCES,
    M03_FORWARD_PARITY_VOCAB_SIZE,
    PARITY_MANIFEST_FORMAT,
    PARITY_MANIFEST_VERSION,
    PARITY_RECEIPT_SCOPE,
    TensorParityReceipt,
    build_m03_forward_parity_manifest,
    build_parity_manifest,
    validate_m03_cuda_reference_artifact_dict,
    validate_m03_forward_parity_manifest_dict,
    validate_parity_manifest_dict,
    validate_parity_receipt_dict,
    write_m03_forward_parity_manifest_json,
    write_parity_manifest_json,
    write_parity_manifest_jsonl,
)


INPUT_TOKENS_SHA256 = M03_FORWARD_PARITY_INPUT_TOKENS_SHA256


def _receipt(name: str = "layers.0.attn.q_proj.weight") -> TensorParityReceipt:
    return TensorParityReceipt(
        tensor_name=name,
        shape=(2, 4),
        dtype="bf16",
        atol=1e-1,
        rtol=1e-3,
        max_abs_error=5e-2,
        max_rel_error=5e-4,
        source_commit="cuda-ref-abc123",
        mlx_version="0.31.0",
        mlx_commit="mlx-def456",
        hardware="M4 Max local",
        status="pass",
        metadata={"route": "A", "non_json": object()},
    )


def _valid_m03_cuda_artifact(
    *,
    input_tokens_sha256: str = INPUT_TOKENS_SHA256,
) -> dict[str, Any]:
    return {
        "format": M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT,
        "profile": M03_FORWARD_PARITY_PROFILE,
        "tensor_name": M03_FORWARD_PARITY_LOGITS_TENSOR_NAME,
        "seed": M03_FORWARD_PARITY_SEED,
        "batch_size": M03_FORWARD_PARITY_BATCH_SIZE,
        "seq_len": M03_FORWARD_PARITY_SEQ_LEN,
        "vocab_size": M03_FORWARD_PARITY_VOCAB_SIZE,
        "shape": list(M03_FORWARD_PARITY_LOGITS_SHAPE),
        "dtype": "bf16",
        "logits_dtype": "bf16",
        "input_tokens_sha256": input_tokens_sha256,
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


def _valid_m03_cuda_preflight() -> dict[str, Any]:
    return {
        "artifact_preflight_status": "valid_not_evaluated",
        "artifact_error": None,
        "artifact_format": M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT,
        "artifact_tensor_name": M03_FORWARD_PARITY_LOGITS_TENSOR_NAME,
        "artifact_logits_sha256": "1" * 64,
        "artifact_source_commit": "cuda-ref-abc123",
        "artifact_hardware": "GB10 CUDA reference",
        "artifact_cuda_runtime": "CUDA 13.0",
    }


def _valid_m03_mlx_readiness() -> dict[str, Any]:
    return {
        "readiness_status": "pass",
        "local_mlx_forward_executed": True,
        "local_mlx_forward_scope": "tiny_smoke_only",
        "readiness_is_cuda_parity": False,
        "full_profile_allocation_executed": False,
        "full_profile_forward_executed": False,
        "tiny_smoke_forward_executed": True,
        "closure_required_mlx_forward_scope": "full_local_gb10_quarter_logits",
    }


def test_parity_manifest_writes_json_and_jsonl(tmp_path) -> None:
    receipts = [_receipt(), _receipt("layers.0.mlp.up.weight")]

    manifest = write_parity_manifest_json(
        tmp_path / "parity" / "manifest.json",
        receipts,
        source="unit-test",
        metadata={"owner": "wave14"},
    )
    jsonl_rows = write_parity_manifest_jsonl(tmp_path / "parity" / "receipts.jsonl", receipts)

    loaded_manifest = json.loads((tmp_path / "parity" / "manifest.json").read_text())
    loaded_rows = [
        json.loads(line)
        for line in (tmp_path / "parity" / "receipts.jsonl").read_text().splitlines()
    ]

    assert loaded_manifest == manifest
    assert loaded_rows == jsonl_rows
    assert manifest["format"] == PARITY_MANIFEST_FORMAT
    assert manifest["version"] == PARITY_MANIFEST_VERSION
    assert manifest["receipt_scope"] == PARITY_RECEIPT_SCOPE
    assert manifest["num_receipts"] == 2
    assert manifest["source"] == "unit-test"
    assert manifest["metadata"] == {"owner": "wave14"}
    assert LOCAL_ONLY_POLICY in cast(str, manifest["local_only_policy"])
    for row in jsonl_rows:
        assert row["receipt_scope"] == PARITY_RECEIPT_SCOPE
        assert row["within_tolerance"] is True
        assert row["local_only"] is True
        assert row["gb10_parity_claim"] is False
        assert row["m4_vs_gb10_parity_claim"] is False
        assert row["distributed_megatron_parity_claim"] is False
        metadata = cast(dict[str, Any], row["metadata"])
        assert isinstance(metadata["non_json"], str)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda row: row.pop("tensor_name"), "missing required fields: tensor_name"),
        (lambda row: row.update({"status": "ok"}), "status must be one of"),
        (lambda row: row.update({"shape": [2, -1]}), "shape"),
        (lambda row: row.update({"atol": float("nan")}), "atol"),
        (lambda row: row.update({"gb10_parity_claim": True}), "gb10_parity_claim"),
        (lambda row: row.update({"m4_vs_gb10_parity_claim": True}), "m4_vs_gb10"),
        (
            lambda row: row.update({"distributed_megatron_parity_claim": True}),
            "distributed_megatron",
        ),
    ],
)
def test_parity_receipt_validation_fails_closed(mutation, error) -> None:
    row = _receipt().to_dict()
    mutation(row)

    with pytest.raises(ValueError, match=error):
        validate_parity_receipt_dict(row)


def test_parity_receipt_refuses_inconsistent_tolerance_summary() -> None:
    row = _receipt().to_dict()
    row["max_abs_error"] = 2.0
    row["within_tolerance"] = True

    with pytest.raises(ValueError, match="within_tolerance"):
        validate_parity_receipt_dict(row)


def test_manifest_validation_refuses_local_only_parity_claims() -> None:
    manifest = build_parity_manifest([_receipt()])

    assert manifest["local_only"] is True
    assert manifest["gb10_parity_claim"] is False
    assert manifest["m4_vs_gb10_parity_claim"] is False
    assert manifest["distributed_megatron_parity_claim"] is False

    for claim_field in (
        "gb10_parity_claim",
        "m4_vs_gb10_parity_claim",
        "distributed_megatron_parity_claim",
    ):
        mutated = dict(manifest)
        mutated[claim_field] = True
        with pytest.raises(ValueError, match=claim_field):
            validate_parity_manifest_dict(mutated)


def test_manifest_validation_checks_receipt_count_and_rows() -> None:
    manifest = build_parity_manifest([_receipt()])
    mutated_count = dict(manifest)
    mutated_count["num_receipts"] = 2

    with pytest.raises(ValueError, match="num_receipts"):
        validate_parity_manifest_dict(mutated_count)

    mutated_row = dict(manifest)
    receipts = cast(list[JsonObject], mutated_row["receipts"])
    row = dict(receipts[0])
    row["local_only"] = False
    mutated_row["receipts"] = [row]

    with pytest.raises(ValueError, match=r"receipts\[0\].*local_only"):
        validate_parity_manifest_dict(mutated_row)


def test_m03_cuda_reference_artifact_contract_accepts_valid_metadata() -> None:
    artifact = _valid_m03_cuda_artifact()

    validate_m03_cuda_reference_artifact_dict(
        artifact,
        input_tokens_sha256=INPUT_TOKENS_SHA256,
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda artifact: artifact.update({"input_tokens_sha256": "2" * 64}), "stale"),
        (lambda artifact: artifact.update({"shape": [1, 512, 256]}), "shape"),
        (lambda artifact: artifact.update({"logits_sha256": "not-sha"}), "logits_sha256"),
        (
            lambda artifact: cast(dict[str, Any], artifact["logits_summary"]).update(
                {"mean": "0.0"}
            ),
            "mean must be a finite number",
        ),
        (
            lambda artifact: cast(dict[str, Any], artifact["logits_summary"]).update(
                {"std": float("nan")}
            ),
            "std must be a finite number",
        ),
        (
            lambda artifact: cast(dict[str, Any], artifact["logits_summary"]).update(
                {"l2_norm": -1.0}
            ),
            "l2_norm must be non-negative",
        ),
        (
            lambda artifact: cast(dict[str, Any], artifact["logits_summary"]).update(
                {"numel": 256}
            ),
            "numel",
        ),
        (lambda artifact: artifact.update({"source_commit": ""}), "source_commit"),
    ],
)
def test_m03_cuda_reference_artifact_contract_rejects_bad_metadata(
    mutation,
    error: str,
) -> None:
    artifact = _valid_m03_cuda_artifact()
    mutation(artifact)

    with pytest.raises(ValueError, match=error):
        validate_m03_cuda_reference_artifact_dict(
            artifact,
            input_tokens_sha256=INPUT_TOKENS_SHA256,
        )


def test_m03_forward_parity_manifest_refuses_artifact_receipts_without_evaluation(tmp_path) -> None:
    receipt = TensorParityReceipt(
        tensor_name="local_gb10_quarter.logits",
        shape=(
            M03_FORWARD_PARITY_BATCH_SIZE,
            M03_FORWARD_PARITY_SEQ_LEN,
            M03_FORWARD_PARITY_VOCAB_SIZE,
        ),
        dtype="bf16",
        atol=M03_FORWARD_PARITY_ATOL,
        rtol=M03_FORWARD_PARITY_RTOL,
        max_abs_error=5e-2,
        max_rel_error=5e-3,
        source_commit="cuda-ref-abc123",
        mlx_version="0.31.0",
        mlx_commit="mlx-def456",
        hardware="M4 Max local",
        status="pass",
    )

    manifest = write_m03_forward_parity_manifest_json(
        tmp_path / M03_FORWARD_PARITY_OUTPUT,
        [receipt],
        seed=M03_FORWARD_PARITY_SEED,
        batch_size=M03_FORWARD_PARITY_BATCH_SIZE,
        seq_len=M03_FORWARD_PARITY_SEQ_LEN,
        source="unit-test",
        input_tokens_sha256=INPUT_TOKENS_SHA256,
        cuda_reference_artifact="external_cuda_logits.json",
        cuda_reference={"script": "external_cuda_forward.py"},
        cuda_reference_preflight=_valid_m03_cuda_preflight(),
        mlx_reference={
            "script": "scripts/m03_forward_parity_manifest.py",
            **_valid_m03_mlx_readiness(),
        },
        metadata={"owner": "m0.3"},
    )

    assert json.loads((tmp_path / M03_FORWARD_PARITY_OUTPUT).read_text()) == manifest
    assert manifest["m0_3_receipt_scope"] == M03_FORWARD_PARITY_RECEIPT_SCOPE
    assert cast(dict[str, Any], manifest["issue"])["id"] == M03_FORWARD_PARITY_ISSUE_ID
    assert manifest["profile"] == M03_FORWARD_PARITY_PROFILE
    assert manifest["profile_metadata"] == M03_FORWARD_PARITY_PROFILE_METADATA
    input_batch = cast(dict[str, Any], manifest["input_batch"])
    assert input_batch["batch_size"] == M03_FORWARD_PARITY_FIXED_INPUT["batch_size"]
    assert input_batch["seq_len"] == M03_FORWARD_PARITY_FIXED_INPUT["seq_len"]
    assert input_batch["seed"] == M03_FORWARD_PARITY_FIXED_INPUT["seed"]
    assert input_batch["vocab_size"] == M03_FORWARD_PARITY_FIXED_INPUT["vocab_size"]
    assert input_batch["fixed"] is True
    assert M03_FORWARD_PARITY_FIXED_INPUT["tokens_sha256"] == INPUT_TOKENS_SHA256
    assert input_batch["tokens_sha256"] == INPUT_TOKENS_SHA256
    assert manifest["tolerances"] == M03_FORWARD_PARITY_TOLERANCES
    assert manifest["status"] == "refused"
    assert manifest["acceptance_status"] == "not_evaluated"
    assert manifest["m0_3_closed"] is False
    assert manifest["full_m0_3_acceptance_claim"] is False
    assert manifest["cuda_weight_import"] is False
    assert manifest["warm_start"] is False
    assert manifest["gb10_forward_parity_claim"] is False
    assert manifest["m4_vs_gb10_forward_parity_claim"] is False
    assert manifest["distributed_megatron_forward_parity_claim"] is False
    assert "no GB10-vs-M4" in cast(str, manifest["claim_boundary"])
    cuda_reference = cast(dict[str, Any], manifest["cuda_reference"])
    assert cuda_reference["status"] == "refused_not_evaluated"
    assert cuda_reference["artifact"] == "external_cuda_logits.json"
    assert cuda_reference["artifact_supplied"] is True
    assert cuda_reference["artifact_preflight_status"] == "valid_not_evaluated"
    assert cuda_reference["required_artifact"] == M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH
    assert cuda_reference["artifact_contract"] == M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT
    assert cuda_reference["artifact_error"] is None
    assert cuda_reference["artifact_format"] == M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT
    assert cuda_reference["artifact_tensor_name"] == M03_FORWARD_PARITY_LOGITS_TENSOR_NAME
    assert cuda_reference["artifact_logits_sha256"] == "1" * 64
    assert cuda_reference["artifact_source_commit"] == "cuda-ref-abc123"
    assert cuda_reference["artifact_hardware"] == "GB10 CUDA reference"
    assert cuda_reference["artifact_cuda_runtime"] == "CUDA 13.0"
    assert cuda_reference["evaluates_logits"] is False
    assert cuda_reference["preflight_is_acceptance"] is False
    assert cuda_reference["evaluated_by_this_manifest"] is False
    mlx_reference = cast(dict[str, Any], manifest["mlx_reference"])
    assert mlx_reference["status"] == "not_evaluated"
    assert mlx_reference["evaluated_by_this_manifest"] is False
    assert mlx_reference["readiness_is_cuda_parity"] is False
    assert mlx_reference["full_profile_allocation_executed"] is False
    assert mlx_reference["full_profile_forward_executed"] is False
    assert mlx_reference["local_mlx_forward_scope"] == "tiny_smoke_only"
    acceptance_gate = cast(dict[str, Any], manifest["acceptance_gate"])
    assert acceptance_gate["logits_receipts"] == 1
    assert acceptance_gate["receipt_count"] == 1
    assert acceptance_gate["cuda_reference_artifact_supplied"] is True
    assert acceptance_gate["cuda_reference_artifact_preflight_status"] == "valid_not_evaluated"
    assert acceptance_gate["receipts_evaluated_by_this_manifest"] is False
    assert acceptance_gate["full_m0_3_acceptance"] is False


def test_m03_forward_parity_manifest_blocks_without_receipts() -> None:
    manifest = build_m03_forward_parity_manifest(
        [],
        seed=M03_FORWARD_PARITY_SEED,
        batch_size=M03_FORWARD_PARITY_BATCH_SIZE,
        seq_len=M03_FORWARD_PARITY_SEQ_LEN,
        source="unit-test",
    )

    assert manifest["status"] == "blocked"
    assert manifest["acceptance_status"] == "not_evaluated"
    assert manifest["m0_3_closed"] is False
    assert manifest["num_receipts"] == 0
    assert manifest["full_m0_3_acceptance_claim"] is False
    cuda_reference = cast(dict[str, Any], manifest["cuda_reference"])
    assert cuda_reference["status"] == "blocked_missing_artifact"
    assert cuda_reference["artifact"] is None
    assert cuda_reference["artifact_supplied"] is False
    assert cuda_reference["artifact_preflight_status"] == "not_supplied"
    assert cuda_reference["required_artifact"] == M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH
    assert cuda_reference["artifact_contract"] == M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT
    assert cuda_reference["artifact_error"] == "No CUDA reference artifact was supplied."
    assert cuda_reference["evaluated_by_this_manifest"] is False
    acceptance_gate = cast(dict[str, Any], manifest["acceptance_gate"])
    assert acceptance_gate["requires_cuda_reference_artifact"] is True
    assert acceptance_gate["requires_external_cuda_logits"] is True
    assert acceptance_gate["requires_mlx_forward"] is True
    assert acceptance_gate["cuda_reference_artifact_preflight_status"] == "not_supplied"
    assert acceptance_gate["receipts_evaluated_by_this_manifest"] is False
    assert acceptance_gate["full_m0_3_acceptance"] is False
    assert "CUDA reference artifact" in cast(str, acceptance_gate["blocker"])


def test_m03_forward_parity_manifest_rejects_valid_preflight_without_metadata() -> None:
    with pytest.raises(ValueError, match="artifact_format"):
        build_m03_forward_parity_manifest(
            [],
            seed=M03_FORWARD_PARITY_SEED,
            batch_size=M03_FORWARD_PARITY_BATCH_SIZE,
            seq_len=M03_FORWARD_PARITY_SEQ_LEN,
            cuda_reference_artifact="external_cuda_logits.json",
            cuda_reference_preflight={
                "artifact_preflight_status": "valid_not_evaluated",
                "artifact_error": None,
            },
        )


def test_m03_forward_parity_manifest_fails_closed_for_overclaims() -> None:
    manifest = build_m03_forward_parity_manifest(
        [],
        seed=M03_FORWARD_PARITY_SEED,
        batch_size=M03_FORWARD_PARITY_BATCH_SIZE,
        seq_len=M03_FORWARD_PARITY_SEQ_LEN,
    )

    for field in (
        "cuda_weight_import",
        "warm_start",
        "gb10_forward_parity_claim",
        "m4_vs_gb10_forward_parity_claim",
        "distributed_megatron_forward_parity_claim",
    ):
        mutated = dict(manifest)
        mutated[field] = True
        with pytest.raises(ValueError, match=field):
            validate_m03_forward_parity_manifest_dict(mutated)

    mutated_acceptance = dict(manifest)
    mutated_acceptance["full_m0_3_acceptance_claim"] = True
    with pytest.raises(ValueError, match="full_m0_3_acceptance_claim"):
        validate_m03_forward_parity_manifest_dict(mutated_acceptance)

    mutated_acceptance_status = dict(manifest)
    mutated_acceptance_status["acceptance_status"] = "pass"
    with pytest.raises(ValueError, match="acceptance_status"):
        validate_m03_forward_parity_manifest_dict(mutated_acceptance_status)

    mutated_closed = dict(manifest)
    mutated_closed["m0_3_closed"] = True
    with pytest.raises(ValueError, match="m0_3_closed"):
        validate_m03_forward_parity_manifest_dict(mutated_closed)

    mutated_status = dict(manifest)
    mutated_status["status"] = "pass"
    with pytest.raises(ValueError, match="status must be blocked or refused"):
        validate_m03_forward_parity_manifest_dict(mutated_status)

    mutated_gate_acceptance = dict(manifest)
    gate_acceptance = dict(cast(dict[str, Any], mutated_gate_acceptance["acceptance_gate"]))
    gate_acceptance["full_m0_3_acceptance"] = True
    mutated_gate_acceptance["acceptance_gate"] = gate_acceptance
    with pytest.raises(ValueError, match="acceptance_gate.full_m0_3_acceptance"):
        validate_m03_forward_parity_manifest_dict(mutated_gate_acceptance)

    mutated_gate_receipts = dict(manifest)
    gate_receipts = dict(cast(dict[str, Any], mutated_gate_receipts["acceptance_gate"]))
    gate_receipts["receipts_evaluated_by_this_manifest"] = True
    mutated_gate_receipts["acceptance_gate"] = gate_receipts
    with pytest.raises(ValueError, match="receipts_evaluated_by_this_manifest"):
        validate_m03_forward_parity_manifest_dict(mutated_gate_receipts)

    mutated_cuda = dict(manifest)
    cuda_reference = dict(cast(dict[str, Any], mutated_cuda["cuda_reference"]))
    cuda_reference["evaluated_by_this_manifest"] = True
    mutated_cuda["cuda_reference"] = cuda_reference
    with pytest.raises(ValueError, match="cuda_reference.evaluated_by_this_manifest"):
        validate_m03_forward_parity_manifest_dict(mutated_cuda)

    mutated_cuda_preflight = dict(manifest)
    cuda_reference_preflight = dict(cast(dict[str, Any], mutated_cuda_preflight["cuda_reference"]))
    cuda_reference_preflight["artifact_preflight_status"] = "valid_not_evaluated"
    mutated_cuda_preflight["cuda_reference"] = cuda_reference_preflight
    with pytest.raises(ValueError, match="artifact_preflight_status mismatch"):
        validate_m03_forward_parity_manifest_dict(mutated_cuda_preflight)

    mutated_cuda_supplied = dict(manifest)
    cuda_reference_supplied = dict(cast(dict[str, Any], mutated_cuda_supplied["cuda_reference"]))
    cuda_reference_supplied["artifact_preflight_status"] = "valid_not_evaluated"
    cuda_reference_supplied["status"] = "refused_not_evaluated"
    gate_supplied = dict(cast(dict[str, Any], mutated_cuda_supplied["acceptance_gate"]))
    gate_supplied["cuda_reference_artifact_preflight_status"] = "valid_not_evaluated"
    mutated_cuda_supplied["cuda_reference"] = cuda_reference_supplied
    mutated_cuda_supplied["acceptance_gate"] = gate_supplied
    with pytest.raises(ValueError, match="artifact_supplied"):
        validate_m03_forward_parity_manifest_dict(mutated_cuda_supplied)

    mutated_cuda_logits = dict(manifest)
    cuda_reference_logits = dict(cast(dict[str, Any], mutated_cuda_logits["cuda_reference"]))
    cuda_reference_logits["evaluates_logits"] = True
    mutated_cuda_logits["cuda_reference"] = cuda_reference_logits
    with pytest.raises(ValueError, match="cuda_reference.evaluates_logits"):
        validate_m03_forward_parity_manifest_dict(mutated_cuda_logits)

    mutated_cuda_acceptance = dict(manifest)
    cuda_reference_acceptance = dict(cast(dict[str, Any], mutated_cuda_acceptance["cuda_reference"]))
    cuda_reference_acceptance["preflight_is_acceptance"] = True
    mutated_cuda_acceptance["cuda_reference"] = cuda_reference_acceptance
    with pytest.raises(ValueError, match="preflight_is_acceptance"):
        validate_m03_forward_parity_manifest_dict(mutated_cuda_acceptance)

    mutated_cuda_contract = dict(manifest)
    cuda_reference_contract = dict(cast(dict[str, Any], mutated_cuda_contract["cuda_reference"]))
    contract = dict(cast(dict[str, Any], cuda_reference_contract["artifact_contract"]))
    contract["shape"] = [1, 512, 256]
    cuda_reference_contract["artifact_contract"] = contract
    mutated_cuda_contract["cuda_reference"] = cuda_reference_contract
    with pytest.raises(ValueError, match="artifact_contract"):
        validate_m03_forward_parity_manifest_dict(mutated_cuda_contract)

    mutated_gate_preflight = dict(manifest)
    gate_preflight = dict(cast(dict[str, Any], mutated_gate_preflight["acceptance_gate"]))
    gate_preflight["cuda_reference_artifact_preflight_status"] = "invalid"
    mutated_gate_preflight["acceptance_gate"] = gate_preflight
    with pytest.raises(ValueError, match="artifact_preflight_status mismatch"):
        validate_m03_forward_parity_manifest_dict(mutated_gate_preflight)

    mutated_mlx = dict(manifest)
    mlx_reference = dict(cast(dict[str, Any], mutated_mlx["mlx_reference"]))
    mlx_reference["evaluated_by_this_manifest"] = True
    mutated_mlx["mlx_reference"] = mlx_reference
    with pytest.raises(ValueError, match="mlx_reference.evaluated_by_this_manifest"):
        validate_m03_forward_parity_manifest_dict(mutated_mlx)

    mutated_input_hash = dict(manifest)
    input_batch = dict(cast(dict[str, Any], mutated_input_hash["input_batch"]))
    input_batch["tokens_sha256"] = "0" * 64
    mutated_input_hash["input_batch"] = input_batch
    with pytest.raises(ValueError, match="tokens_sha256 must match fixed input hash"):
        validate_m03_forward_parity_manifest_dict(mutated_input_hash)


def test_m03_forward_parity_manifest_rejects_nested_m03_overclaims() -> None:
    manifest = build_m03_forward_parity_manifest(
        [],
        seed=M03_FORWARD_PARITY_SEED,
        batch_size=M03_FORWARD_PARITY_BATCH_SIZE,
        seq_len=M03_FORWARD_PARITY_SEQ_LEN,
        cuda_reference_artifact="external_cuda_logits.json",
        cuda_reference_preflight=_valid_m03_cuda_preflight(),
        mlx_reference=_valid_m03_mlx_readiness(),
    )

    cuda_mutations: list[tuple[str, Any, str]] = [
        ("artifact_format", "malicious_acceptance_format", "artifact_format"),
        ("artifact_tensor_name", "local_gb10_quarter.accepted_logits", "artifact_tensor_name"),
        ("artifact_logits_sha256", "not-sha", "artifact_logits_sha256"),
        ("artifact_source_commit", "", "artifact_source_commit"),
        ("artifact_hardware", "", "artifact_hardware"),
        ("artifact_cuda_runtime", "", "artifact_cuda_runtime"),
    ]
    for field_name, value, error in cuda_mutations:
        mutated = dict(manifest)
        cuda_reference = dict(cast(dict[str, Any], mutated["cuda_reference"]))
        cuda_reference[field_name] = value
        mutated["cuda_reference"] = cuda_reference
        with pytest.raises(ValueError, match=error):
            validate_m03_forward_parity_manifest_dict(mutated)

    mlx_mutations: list[tuple[str, Any, str]] = [
        ("readiness_is_cuda_parity", True, "readiness_is_cuda_parity"),
        ("full_profile_allocation_executed", True, "full_profile_allocation_executed"),
        ("full_profile_forward_executed", True, "full_profile_forward_executed"),
        ("local_mlx_forward_scope", "full_local_gb10_quarter_logits", "local_mlx_forward_scope"),
        ("closure_required_mlx_forward_scope", "tiny_smoke_only", "closure_required"),
    ]
    for field_name, value, error in mlx_mutations:
        mutated = dict(manifest)
        mlx_reference = dict(cast(dict[str, Any], mutated["mlx_reference"]))
        mlx_reference[field_name] = value
        mutated["mlx_reference"] = mlx_reference
        with pytest.raises(ValueError, match=error):
            validate_m03_forward_parity_manifest_dict(mutated)

    skipped_mutated = dict(manifest)
    skipped_mlx = dict(cast(dict[str, Any], skipped_mutated["mlx_reference"]))
    skipped_mlx["readiness_status"] = "skipped"
    skipped_mlx["local_mlx_forward_scope"] = "skipped"
    skipped_mlx["local_mlx_forward_executed"] = True
    skipped_mutated["mlx_reference"] = skipped_mlx
    with pytest.raises(ValueError, match="local_mlx_forward_executed"):
        validate_m03_forward_parity_manifest_dict(skipped_mutated)


def test_m03_forward_parity_manifest_does_not_evaluate_ambiguous_logits_receipts() -> None:
    receipt = TensorParityReceipt(
        tensor_name="local_gb10_quarter.logits",
        shape=(
            M03_FORWARD_PARITY_BATCH_SIZE,
            M03_FORWARD_PARITY_SEQ_LEN,
            M03_FORWARD_PARITY_VOCAB_SIZE,
        ),
        dtype="bf16",
        atol=M03_FORWARD_PARITY_ATOL,
        rtol=M03_FORWARD_PARITY_RTOL,
        max_abs_error=5e-2,
        max_rel_error=5e-3,
        source_commit="cuda-ref-abc123",
        mlx_version="0.31.0",
        hardware="M4 Max local",
        status="pass",
    )

    manifest = build_m03_forward_parity_manifest(
        [receipt, receipt],
        seed=M03_FORWARD_PARITY_SEED,
        batch_size=M03_FORWARD_PARITY_BATCH_SIZE,
        seq_len=M03_FORWARD_PARITY_SEQ_LEN,
    )

    assert manifest["status"] == "blocked"
    assert manifest["acceptance_status"] == "not_evaluated"
    assert manifest["m0_3_closed"] is False
    assert manifest["full_m0_3_acceptance_claim"] is False
    acceptance_gate = cast(dict[str, Any], manifest["acceptance_gate"])
    assert acceptance_gate["logits_receipts"] == 2
    assert acceptance_gate["receipts_evaluated_by_this_manifest"] is False
    assert acceptance_gate["full_m0_3_acceptance"] is False
