from __future__ import annotations

from collections.abc import Mapping

import pytest

import cppmega_mlx.training as training
from cppmega_mlx.training import (
    BASELINE_ARCHIVE_KIND,
    BASELINE_INDEX_FILENAME,
    LOCAL_ONLY_POLICY,
    M05_MTP_BETA,
    M05_MTP_DEPTH,
    M05_MTP_LAMBDA,
    M05_MTP_PARITY_POLICY,
    M05_MTP_PARITY_PROFILE,
    PARITY_MANIFEST_FORMAT,
    PARITY_RECEIPT_SCOPE,
    REQUIRED_BASELINE_ROW_KEYS,
    REQUIRED_RECEIPT_FIELDS,
    BaselineValidationError,
    TensorParityReceipt,
    build_m05_mtp_parity_manifest,
    build_parity_manifest,
    m05_loss_values_sha256,
    validate_baseline_row,
    validate_m05_mtp_parity_manifest_dict,
    validate_parity_manifest_dict,
    validate_parity_receipt_dict,
)


EXPECTED_HELPER_EXPORTS = {
    "BASELINE_ARCHIVE_KIND",
    "BASELINE_INDEX_FILENAME",
    "BASELINE_INDEX_KIND",
    "BASELINE_INDEX_SCHEMA_VERSION",
    "BASELINE_ROW_SCHEMA_VERSION",
    "BaselineValidationError",
    "M05_MTP_BETA",
    "M05_MTP_CUDA_ARTIFACT_CONTRACT",
    "M05_MTP_CUDA_ARTIFACT_FORMAT",
    "M05_MTP_CUDA_ARTIFACT_PATH",
    "M05_MTP_CUDA_ARTIFACT_PREFLIGHT_STATUSES",
    "M05_MTP_CUDA_LOSS_VALUE_FIELDS",
    "M05_MTP_CUDA_REFERENCE_SOURCES",
    "M05_MTP_DEPTH",
    "M05_MTP_LAMBDA",
    "M05_MTP_PARITY_ISSUE_ID",
    "M05_MTP_PARITY_OUTPUT",
    "M05_MTP_PARITY_POLICY",
    "M05_MTP_PARITY_PROFILE",
    "M05_MTP_PARITY_RECEIPT_SCOPE",
    "PARITY_EVIDENCE_POLICY",
    "REQUIRED_BASELINE_ROW_KEYS",
    "VALID_MODES",
    "archive_baseline_row",
    "baseline_filename",
    "validate_baseline_row",
    "LOCAL_ONLY_POLICY",
    "PARITY_MANIFEST_FORMAT",
    "PARITY_MANIFEST_VERSION",
    "PARITY_RECEIPT_SCOPE",
    "REQUIRED_RECEIPT_FIELDS",
    "TensorParityReceipt",
    "VALID_PARITY_STATUSES",
    "build_m05_mtp_parity_manifest",
    "build_parity_manifest",
    "coerce_parity_receipt",
    "m05_loss_values_sha256",
    "validate_m05_cuda_reference_artifact_dict",
    "validate_m05_mtp_parity_manifest_dict",
    "validate_parity_manifest_dict",
    "validate_parity_receipt_dict",
    "write_m05_mtp_parity_manifest_json",
    "write_parity_manifest_json",
    "write_parity_manifest_jsonl",
}

UNSTABLE_OR_OVERCLAIMING_NAMES = {
    "BASELINE_MATCHED_PARITY",
    "M4GB10ParityReport",
    "ParityBenchmarkRunner",
    "assert_distributed_megatron_parity",
    "compare_gb10_performance",
    "cuda_reference_loader",
    "gb10_parity",
    "matched_m4_gb10_parity",
    "torch",
}


def _valid_baseline_row() -> dict[str, object]:
    return {
        "hardware": "M4 Max",
        "commit": "abc1234",
        "dtype": "bfloat16",
        "batch_size": 2,
        "seq_len": 64,
        "route": "mamba3",
        "model": "hybrid-m",
        "mode": "eager",
        "tokens_per_second": 123.5,
        "local_only": True,
        "gb10_parity_claim": False,
    }


def _valid_receipt() -> TensorParityReceipt:
    return TensorParityReceipt(
        tensor_name="layers.0.attn.q_proj.weight",
        shape=(2, 4),
        dtype="bf16",
        atol=1e-1,
        rtol=1e-3,
        max_abs_error=5e-2,
        max_rel_error=5e-4,
        source_commit="cuda-ref-abc123",
        mlx_version="0.31.0",
        hardware="M4 Max local",
        status="pass",
    )


def test_training_root_reexports_stable_helper_apis_only() -> None:
    exports = set(training.__all__)

    assert EXPECTED_HELPER_EXPORTS <= exports
    for name in EXPECTED_HELPER_EXPORTS:
        assert getattr(training, name) is not None
    for name in UNSTABLE_OR_OVERCLAIMING_NAMES:
        assert name not in exports
        assert not hasattr(training, name), name


def test_training_helper_exports_import_cleanly_and_preserve_policies() -> None:
    assert BASELINE_ARCHIVE_KIND == "cppmega.mlx.benchmark_baseline"
    assert BASELINE_INDEX_FILENAME == "index.json"
    assert "gb10_parity_claim" in REQUIRED_BASELINE_ROW_KEYS
    assert PARITY_MANIFEST_FORMAT == "cppmega_mlx_tensor_parity_manifest_v1"
    assert PARITY_RECEIPT_SCOPE == "local_cuda_mlx_tensor"
    assert "tensor_name" in REQUIRED_RECEIPT_FIELDS
    assert "not M4-vs-GB10 parity evidence" in LOCAL_ONLY_POLICY
    assert M05_MTP_DEPTH == 2
    assert M05_MTP_BETA == 0.6
    assert M05_MTP_LAMBDA == 0.3
    assert M05_MTP_PARITY_PROFILE == "local_gb10_quarter"
    assert "fail-closed" in M05_MTP_PARITY_POLICY
    assert "Closure requires a real GB10 CUDA FastMTP golden artifact" in M05_MTP_PARITY_POLICY
    assert "separate numerical MLX-vs-CUDA comparison harness" in M05_MTP_PARITY_POLICY
    assert isinstance(
        m05_loss_values_sha256(
            {
                "next_token_loss": 2.0,
                "mtp_loss": 1.4375,
                "total_loss": 2.43125,
                "mtp_depth_1_loss": 1.25,
                "mtp_depth_2_loss": 1.75,
                "grad_norm": 3.5,
            }
        ),
        str,
    )


def test_training_root_fail_closed_helpers_remain_available() -> None:
    row = _valid_baseline_row()
    assert validate_baseline_row(row)["local_only"] is True

    bad_row = dict(row)
    bad_row["gb10_parity_claim"] = True
    with pytest.raises(BaselineValidationError, match="local_only"):
        validate_baseline_row(bad_row)

    receipt = _valid_receipt().to_dict()
    validate_parity_receipt_dict(receipt)

    bad_receipt = dict(receipt)
    bad_receipt["distributed_megatron_parity_claim"] = True
    with pytest.raises(ValueError, match="distributed_megatron"):
        validate_parity_receipt_dict(bad_receipt)

    manifest = build_parity_manifest([receipt])
    validate_parity_manifest_dict(manifest)
    assert manifest["gb10_parity_claim"] is False
    assert manifest["m4_vs_gb10_parity_claim"] is False
    assert manifest["distributed_megatron_parity_claim"] is False

    m05_manifest = build_m05_mtp_parity_manifest([])
    validate_m05_mtp_parity_manifest_dict(m05_manifest)
    assert m05_manifest["status"] == "blocked"
    assert m05_manifest["m0_5_closed"] is False
    acceptance_gate = m05_manifest["acceptance_gate"]
    assert isinstance(acceptance_gate, Mapping)
    assert acceptance_gate["full_m0_5_acceptance"] is False
