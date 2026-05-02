"""Fail-closed tensor-parity receipt manifests.

The helpers in this module describe local CUDA-reference versus MLX tensor
diffs. They intentionally do not import CUDA, Torch, MLX, or sibling repos at
runtime; callers provide already-computed error metrics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


JsonScalar = bool | int | float | str | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

PARITY_MANIFEST_FORMAT = "cppmega_mlx_tensor_parity_manifest_v1"
PARITY_MANIFEST_VERSION = 1
PARITY_RECEIPT_SCOPE = "local_cuda_mlx_tensor"
VALID_PARITY_STATUSES = ("pass", "fail", "blocked", "refused")
LOCAL_ONLY_POLICY = (
    "Local CUDA/MLX tensor diff receipts are implementation evidence only. "
    "They are not M4-vs-GB10 parity evidence and do not prove distributed "
    "Megatron parity."
)
REQUIRED_RECEIPT_FIELDS = (
    "tensor_name",
    "shape",
    "dtype",
    "atol",
    "rtol",
    "max_abs_error",
    "max_rel_error",
    "source_commit",
    "mlx_version",
    "hardware",
    "status",
)
_FALSE_CLAIM_FIELDS = (
    "gb10_parity_claim",
    "m4_vs_gb10_parity_claim",
    "distributed_megatron_parity_claim",
)


def _json_safe(value: Any) -> JsonValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return str(value)


def _json_safe_mapping(value: Mapping[str, Any]) -> JsonObject:
    return {str(key): _json_safe(item) for key, item in value.items()}


def _require_mapping(payload: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_non_empty_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"parity receipt {field_name} must be a non-empty string")
    return value


def _require_non_negative_number(payload: Mapping[str, Any], field_name: str) -> float:
    value = payload.get(field_name)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"parity receipt {field_name} must be a non-negative finite number")
    return float(value)


def _require_shape(payload: Mapping[str, Any]) -> tuple[int, ...]:
    value = payload.get("shape")
    if not isinstance(value, list | tuple):
        raise ValueError("parity receipt shape must be a list of non-negative integers")
    shape: list[int] = []
    for dim in value:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError("parity receipt shape must be a list of non-negative integers")
        shape.append(dim)
    return tuple(shape)


def _validate_no_local_parity_claims(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("local_only") is not True:
        raise ValueError(f"{label} local_only must be true")
    for field_name in _FALSE_CLAIM_FIELDS:
        if payload.get(field_name) is not False:
            raise ValueError(f"{label} {field_name} must be false")


@dataclass(frozen=True)
class TensorParityReceipt:
    """One local tensor diff receipt comparing a CUDA reference with MLX output."""

    tensor_name: str
    shape: tuple[int, ...]
    dtype: str
    atol: float
    rtol: float
    max_abs_error: float
    max_rel_error: float
    source_commit: str
    mlx_version: str
    hardware: str
    status: str
    mlx_commit: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_parity_receipt_dict(self.to_dict(validate=False))

    @property
    def within_tolerance(self) -> bool:
        return self.max_abs_error <= self.atol and self.max_rel_error <= self.rtol

    def to_dict(self, *, validate: bool = True) -> JsonObject:
        payload: JsonObject = {
            "tensor_name": self.tensor_name,
            "shape": [int(dim) for dim in self.shape],
            "dtype": self.dtype,
            "atol": float(self.atol),
            "rtol": float(self.rtol),
            "max_abs_error": float(self.max_abs_error),
            "max_rel_error": float(self.max_rel_error),
            "source_commit": self.source_commit,
            "mlx_version": self.mlx_version,
            "hardware": self.hardware,
            "status": self.status,
            "within_tolerance": self.within_tolerance,
            "receipt_scope": PARITY_RECEIPT_SCOPE,
            "local_only": True,
            "gb10_parity_claim": False,
            "m4_vs_gb10_parity_claim": False,
            "distributed_megatron_parity_claim": False,
        }
        if self.mlx_commit is not None:
            payload["mlx_commit"] = self.mlx_commit
        if self.notes is not None:
            payload["notes"] = self.notes
        if self.metadata:
            payload["metadata"] = _json_safe_mapping(self.metadata)
        if validate:
            validate_parity_receipt_dict(payload)
        return payload


def validate_parity_receipt_dict(payload: Mapping[str, Any]) -> None:
    """Validate a JSON-like parity receipt.

    Returns ``None`` so callers can use it as a pure assertion helper without
    accidentally normalizing or trusting malformed caller-owned dictionaries.
    """

    _require_mapping(payload, label="parity receipt")
    missing = [field_name for field_name in REQUIRED_RECEIPT_FIELDS if field_name not in payload]
    if missing:
        raise ValueError(f"parity receipt missing required fields: {', '.join(missing)}")

    _require_non_empty_string(payload, "tensor_name")
    _require_shape(payload)
    dtype = _require_non_empty_string(payload, "dtype")
    atol = _require_non_negative_number(payload, "atol")
    rtol = _require_non_negative_number(payload, "rtol")
    max_abs_error = _require_non_negative_number(payload, "max_abs_error")
    max_rel_error = _require_non_negative_number(payload, "max_rel_error")
    source_commit = _require_non_empty_string(payload, "source_commit")
    mlx_version = _require_non_empty_string(payload, "mlx_version")
    hardware = _require_non_empty_string(payload, "hardware")
    status = _require_non_empty_string(payload, "status")
    if status not in VALID_PARITY_STATUSES:
        raise ValueError(
            "parity receipt status must be one of "
            f"{', '.join(VALID_PARITY_STATUSES)}; got {status!r}"
        )

    mlx_commit = payload.get("mlx_commit")
    if mlx_commit is not None and (not isinstance(mlx_commit, str) or not mlx_commit):
        raise ValueError("parity receipt mlx_commit must be a non-empty string when present")

    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("parity receipt notes must be a string when present")

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("parity receipt metadata must be an object when present")

    if payload.get("receipt_scope") != PARITY_RECEIPT_SCOPE:
        raise ValueError(f"parity receipt receipt_scope must be {PARITY_RECEIPT_SCOPE!r}")
    _validate_no_local_parity_claims(payload, label="parity receipt")

    within_tolerance = payload.get("within_tolerance")
    expected_within_tolerance = max_abs_error <= atol and max_rel_error <= rtol
    if within_tolerance is not None and within_tolerance is not expected_within_tolerance:
        raise ValueError("parity receipt within_tolerance does not match recorded errors")

    _ = (dtype, source_commit, mlx_version, hardware, mlx_commit, notes, metadata)
    return None


def coerce_parity_receipt(payload: TensorParityReceipt | Mapping[str, Any]) -> JsonObject:
    if isinstance(payload, TensorParityReceipt):
        return payload.to_dict()
    validate_parity_receipt_dict(payload)
    return _json_safe_mapping(payload)


def build_parity_manifest(
    receipts: Iterable[TensorParityReceipt | Mapping[str, Any]],
    *,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> JsonObject:
    rows: list[JsonValue] = [coerce_parity_receipt(receipt) for receipt in receipts]
    manifest: JsonObject = {
        "format": PARITY_MANIFEST_FORMAT,
        "version": PARITY_MANIFEST_VERSION,
        "receipt_scope": PARITY_RECEIPT_SCOPE,
        "local_only": True,
        "gb10_parity_claim": False,
        "m4_vs_gb10_parity_claim": False,
        "distributed_megatron_parity_claim": False,
        "local_only_policy": LOCAL_ONLY_POLICY,
        "required_receipt_fields": list(REQUIRED_RECEIPT_FIELDS),
        "valid_statuses": list(VALID_PARITY_STATUSES),
        "num_receipts": len(rows),
        "receipts": rows,
    }
    if source is not None:
        if not source:
            raise ValueError("parity manifest source must be non-empty when present")
        manifest["source"] = source
    if metadata:
        manifest["metadata"] = _json_safe_mapping(metadata)
    validate_parity_manifest_dict(manifest)
    return manifest


def validate_parity_manifest_dict(payload: Mapping[str, Any]) -> None:
    _require_mapping(payload, label="parity manifest")
    if payload.get("format") != PARITY_MANIFEST_FORMAT:
        raise ValueError(
            f"parity manifest format must be {PARITY_MANIFEST_FORMAT!r}; "
            f"got {payload.get('format')!r}"
        )
    if payload.get("version") != PARITY_MANIFEST_VERSION:
        raise ValueError(
            f"parity manifest version must be {PARITY_MANIFEST_VERSION}; "
            f"got {payload.get('version')!r}"
        )
    if payload.get("receipt_scope") != PARITY_RECEIPT_SCOPE:
        raise ValueError(f"parity manifest receipt_scope must be {PARITY_RECEIPT_SCOPE!r}")
    _validate_no_local_parity_claims(payload, label="parity manifest")

    receipts = payload.get("receipts")
    if not isinstance(receipts, list):
        raise ValueError("parity manifest receipts must be a list")
    num_receipts = payload.get("num_receipts")
    if isinstance(num_receipts, bool) or not isinstance(num_receipts, int):
        raise ValueError("parity manifest num_receipts must be an integer")
    if num_receipts != len(receipts):
        raise ValueError("parity manifest num_receipts does not match receipts length")
    for index, receipt in enumerate(receipts):
        try:
            validate_parity_receipt_dict(_require_mapping(receipt, label="parity receipt"))
        except ValueError as exc:
            raise ValueError(f"parity manifest receipts[{index}]: {exc}") from exc


def write_parity_manifest_json(
    path: str | Path,
    receipts: Iterable[TensorParityReceipt | Mapping[str, Any]],
    *,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> JsonObject:
    manifest = build_parity_manifest(receipts, source=source, metadata=metadata)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_parity_manifest_jsonl(
    path: str | Path,
    receipts: Iterable[TensorParityReceipt | Mapping[str, Any]],
) -> list[JsonObject]:
    rows = [coerce_parity_receipt(receipt) for receipt in receipts]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


__all__ = [
    "LOCAL_ONLY_POLICY",
    "PARITY_MANIFEST_FORMAT",
    "PARITY_MANIFEST_VERSION",
    "PARITY_RECEIPT_SCOPE",
    "REQUIRED_RECEIPT_FIELDS",
    "TensorParityReceipt",
    "VALID_PARITY_STATUSES",
    "build_parity_manifest",
    "coerce_parity_receipt",
    "validate_parity_manifest_dict",
    "validate_parity_receipt_dict",
    "write_parity_manifest_json",
    "write_parity_manifest_jsonl",
]
