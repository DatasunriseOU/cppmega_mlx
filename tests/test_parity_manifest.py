from __future__ import annotations

import json
from typing import Any, cast

import pytest

from cppmega_mlx.training.parity import (
    JsonObject,
    LOCAL_ONLY_POLICY,
    PARITY_MANIFEST_FORMAT,
    PARITY_MANIFEST_VERSION,
    PARITY_RECEIPT_SCOPE,
    TensorParityReceipt,
    build_parity_manifest,
    validate_parity_manifest_dict,
    validate_parity_receipt_dict,
    write_parity_manifest_json,
    write_parity_manifest_jsonl,
)


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
