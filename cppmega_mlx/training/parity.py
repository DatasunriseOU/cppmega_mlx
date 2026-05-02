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
M03_FORWARD_PARITY_RECEIPT_SCOPE = "local_cuda_mlx_m03_forward_logits"
M03_FORWARD_PARITY_ISSUE_ID = "cppmega-mlx-t8f.3"
M03_FORWARD_PARITY_OUTPUT = "bench/parity/m03_random_init.json"
M03_FORWARD_PARITY_PROFILE = "local_gb10_quarter"
M03_FORWARD_PARITY_SEED = 3003
M03_FORWARD_PARITY_BATCH_SIZE = 1
M03_FORWARD_PARITY_SEQ_LEN = 512
M03_FORWARD_PARITY_VOCAB_SIZE = 65_536
M03_FORWARD_PARITY_INPUT_TOKENS_SHA256 = (
    "c645ca4053e5206dcbe58c13aa26f4a9e56c5aa2aee90a4d4778bbc9d9c33549"
)
M03_FORWARD_PARITY_ATOL = 1e-1
M03_FORWARD_PARITY_RTOL = 1e-2
M03_FORWARD_PARITY_POLICY = (
    "M0.3 is random-init, seed-matched, forward-only architecture evidence. "
    "This helper is a scaffold only: it records the required local_gb10_quarter "
    "profile, fixed input metadata, and tolerance policy, but it does not run or "
    "evaluate CUDA/MLX logits and never closes M0.3 by itself. Closure requires "
    "an external CUDA artifact plus a separate numerical parity harness."
)
M03_FORWARD_PARITY_PROFILE_METADATA: JsonObject = {
    "name": M03_FORWARD_PARITY_PROFILE,
    "pattern": "AEMEAEMEAEMR",
    "depth": 13,
    "hidden_size": 3584,
    "ffn_hidden_size": 18_944,
    "num_attention_heads": 28,
    "head_dim": 128,
    "vocab_size": M03_FORWARD_PARITY_VOCAB_SIZE,
    "mtp": {"depth": 2, "beta": 0.6, "loss_weight": 0.3},
}
M03_FORWARD_PARITY_FIXED_INPUT: JsonObject = {
    "batch_size": M03_FORWARD_PARITY_BATCH_SIZE,
    "seq_len": M03_FORWARD_PARITY_SEQ_LEN,
    "seed": M03_FORWARD_PARITY_SEED,
    "vocab_size": M03_FORWARD_PARITY_VOCAB_SIZE,
    "token_pattern": "numpy.default_rng(seed).integers(0, vocab_size, dtype=uint32)",
    "tokens_sha256": M03_FORWARD_PARITY_INPUT_TOKENS_SHA256,
    "tokens_sha256_required_for_closure": True,
}
M03_FORWARD_PARITY_TOLERANCES: JsonObject = {
    "bf16_single_matmul": {"atol": 1e-1, "rtol": 1e-3},
    "chained_bf16": {"atol": 1e-1, "rtol": 1e-2},
    "logits": {"atol": M03_FORWARD_PARITY_ATOL, "rtol": M03_FORWARD_PARITY_RTOL},
    "attention_rmsnorm": {"atol": 5e-2},
    "full_step_grad": {"atol": 1e-1},
}
M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH = (
    "bench/parity/cuda/m03_local_gb10_quarter_seed3003_logits.json"
)
M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT = "cppmega_cuda_m03_forward_logits_v1"
M03_FORWARD_PARITY_CUDA_ARTIFACT_PREFLIGHT_STATUSES = (
    "not_supplied",
    "missing",
    "invalid",
    "valid_not_evaluated",
)
M03_FORWARD_PARITY_LOGITS_TENSOR_NAME = f"{M03_FORWARD_PARITY_PROFILE}.logits"
M03_FORWARD_PARITY_LOGITS_SHAPE = (
    M03_FORWARD_PARITY_BATCH_SIZE,
    M03_FORWARD_PARITY_SEQ_LEN,
    M03_FORWARD_PARITY_VOCAB_SIZE,
)
M03_FORWARD_PARITY_LOGITS_NUMEL = math.prod(M03_FORWARD_PARITY_LOGITS_SHAPE)
M03_FORWARD_PARITY_CUDA_LOGITS_DTYPES = ("bf16", "bfloat16", "torch.bfloat16")
M03_FORWARD_PARITY_CUDA_SUMMARY_FIELDS = (
    "min",
    "max",
    "mean",
    "std",
    "l2_norm",
    "max_abs",
)
M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT: JsonObject = {
    "format": M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT,
    "required_artifact": M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH,
    "tensor_name": M03_FORWARD_PARITY_LOGITS_TENSOR_NAME,
    "profile": M03_FORWARD_PARITY_PROFILE,
    "seed": M03_FORWARD_PARITY_SEED,
    "batch_size": M03_FORWARD_PARITY_BATCH_SIZE,
    "seq_len": M03_FORWARD_PARITY_SEQ_LEN,
    "vocab_size": M03_FORWARD_PARITY_VOCAB_SIZE,
    "shape": [int(dim) for dim in M03_FORWARD_PARITY_LOGITS_SHAPE],
    "dtype": "bf16",
    "allowed_logits_dtype": list(M03_FORWARD_PARITY_CUDA_LOGITS_DTYPES),
    "required_logits_summary_fields": ["numel", *M03_FORWARD_PARITY_CUDA_SUMMARY_FIELDS],
    "requires_logits_sha256": True,
    "requires_input_tokens_sha256": True,
    "requires_finite_numeric_summary": True,
}
_M03_CUDA_STATUS_BY_PREFLIGHT_STATUS = {
    "not_supplied": "blocked_missing_artifact",
    "missing": "refused_missing_artifact",
    "invalid": "refused_invalid_artifact",
    "valid_not_evaluated": "refused_not_evaluated",
}
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


def _require_finite_number(payload: Mapping[str, Any], field_name: str, *, label: str) -> float:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{label} {field_name} must be a finite number")
    return float(value)


def _require_exact_int(
    payload: Mapping[str, Any],
    field_name: str,
    expected: int,
    *,
    label: str,
) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{label} {field_name} must be {expected}")
    return value


def _require_sha256(payload: Mapping[str, Any], field_name: str, *, label: str) -> str:
    value = _require_non_empty_string(payload, field_name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{label} {field_name} must be a 64-character hex sha256")
    return value


def _require_false_when_present(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    label: str,
) -> None:
    if field_name in payload and payload.get(field_name) is not False:
        raise ValueError(f"{label} {field_name} must be false")


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


def _require_shape_value(payload: Mapping[str, Any], *, label: str) -> tuple[int, ...]:
    value = payload.get("shape")
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} shape must be a list of non-negative integers")
    shape: list[int] = []
    for dim in value:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"{label} shape must be a list of non-negative integers")
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


def validate_m03_cuda_reference_artifact_dict(
    payload: Mapping[str, Any],
    *,
    input_tokens_sha256: str,
    dtype: str = "bf16",
) -> None:
    """Validate the external CUDA logits artifact contract for M0.3.

    This validates artifact metadata and finite summary/checksum fields only; it
    does not compare logits or close M0.3.
    """

    _require_mapping(payload, label="M0.3 CUDA reference artifact")
    if payload.get("format") != M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT:
        raise ValueError(
            "M0.3 CUDA reference artifact format must be "
            f"{M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT!r}"
        )
    if payload.get("profile") != M03_FORWARD_PARITY_PROFILE:
        raise ValueError(
            f"M0.3 CUDA reference artifact profile must be {M03_FORWARD_PARITY_PROFILE!r}"
        )
    if payload.get("tensor_name") != M03_FORWARD_PARITY_LOGITS_TENSOR_NAME:
        raise ValueError(
            "M0.3 CUDA reference artifact tensor_name must be "
            f"{M03_FORWARD_PARITY_LOGITS_TENSOR_NAME!r}"
        )
    _require_exact_int(
        payload,
        "seed",
        M03_FORWARD_PARITY_SEED,
        label="M0.3 CUDA reference artifact",
    )
    _require_exact_int(
        payload,
        "batch_size",
        M03_FORWARD_PARITY_BATCH_SIZE,
        label="M0.3 CUDA reference artifact",
    )
    _require_exact_int(
        payload,
        "seq_len",
        M03_FORWARD_PARITY_SEQ_LEN,
        label="M0.3 CUDA reference artifact",
    )
    _require_exact_int(
        payload,
        "vocab_size",
        M03_FORWARD_PARITY_VOCAB_SIZE,
        label="M0.3 CUDA reference artifact",
    )
    shape = _require_shape_value(payload, label="M0.3 CUDA reference artifact")
    if shape != M03_FORWARD_PARITY_LOGITS_SHAPE:
        raise ValueError(
            "M0.3 CUDA reference artifact shape must be "
            f"{list(M03_FORWARD_PARITY_LOGITS_SHAPE)!r}"
        )
    artifact_dtype = _require_non_empty_string(payload, "dtype")
    if artifact_dtype != dtype:
        raise ValueError(f"M0.3 CUDA reference artifact dtype must be {dtype!r}")
    logits_dtype = _require_non_empty_string(payload, "logits_dtype")
    if logits_dtype not in M03_FORWARD_PARITY_CUDA_LOGITS_DTYPES:
        allowed = ", ".join(M03_FORWARD_PARITY_CUDA_LOGITS_DTYPES)
        raise ValueError(f"M0.3 CUDA reference artifact logits_dtype must be one of {allowed}")
    artifact_tokens_sha256 = _require_sha256(
        payload,
        "input_tokens_sha256",
        label="M0.3 CUDA reference artifact",
    )
    if artifact_tokens_sha256 != input_tokens_sha256:
        raise ValueError("M0.3 CUDA reference artifact input_tokens_sha256 is stale")
    _require_sha256(payload, "logits_sha256", label="M0.3 CUDA reference artifact")
    _require_non_empty_string(payload, "source_commit")
    _require_non_empty_string(payload, "hardware")
    _require_non_empty_string(payload, "cuda_runtime")

    summary = _require_mapping(
        payload.get("logits_summary"),
        label="M0.3 CUDA reference artifact logits_summary",
    )
    _require_exact_int(
        summary,
        "numel",
        M03_FORWARD_PARITY_LOGITS_NUMEL,
        label="M0.3 CUDA reference artifact logits_summary",
    )
    for field_name in M03_FORWARD_PARITY_CUDA_SUMMARY_FIELDS:
        value = _require_finite_number(
            summary,
            field_name,
            label="M0.3 CUDA reference artifact logits_summary",
        )
        if field_name in {"std", "l2_norm", "max_abs"} and value < 0:
            raise ValueError(
                f"M0.3 CUDA reference artifact logits_summary {field_name} "
                "must be non-negative"
            )


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


def build_m03_forward_parity_manifest(
    receipts: Iterable[TensorParityReceipt | Mapping[str, Any]],
    *,
    seed: int = M03_FORWARD_PARITY_SEED,
    batch_size: int = M03_FORWARD_PARITY_BATCH_SIZE,
    seq_len: int = M03_FORWARD_PARITY_SEQ_LEN,
    profile: str = M03_FORWARD_PARITY_PROFILE,
    dtype: str = "bf16",
    source: str | None = None,
    input_tokens_sha256: str | None = None,
    cuda_reference_artifact: str | Path | None = None,
    cuda_reference: Mapping[str, Any] | None = None,
    cuda_reference_preflight: Mapping[str, Any] | None = None,
    mlx_reference: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> JsonObject:
    """Build the fail-closed M0.3 forward-parity scaffold manifest."""

    _require_non_negative_int("seed", seed)
    _require_positive_int("batch_size", batch_size)
    _require_positive_int("seq_len", seq_len)
    if profile != M03_FORWARD_PARITY_PROFILE:
        raise ValueError(f"M0.3 profile must be {M03_FORWARD_PARITY_PROFILE!r}")
    if not dtype:
        raise ValueError("M0.3 dtype must be non-empty")

    manifest = build_parity_manifest(
        receipts,
        source=source,
        metadata=metadata,
    )
    rows = manifest["receipts"]
    if not isinstance(rows, list):  # pragma: no cover - build_parity_manifest invariant.
        raise ValueError("M0.3 receipts must be a list")
    logits_receipts = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("tensor_name", "")).endswith("logits")
    ]
    artifact_text = str(cuda_reference_artifact) if cuda_reference_artifact is not None else None
    artifact_supplied = bool(artifact_text)
    cuda_preflight_payload = _json_safe_mapping(cuda_reference_preflight or {})
    preflight_status = cuda_preflight_payload.get("artifact_preflight_status")
    if not isinstance(preflight_status, str):
        preflight_status = "missing" if artifact_supplied else "not_supplied"
        cuda_preflight_payload["artifact_preflight_status"] = preflight_status
    cuda_status = _M03_CUDA_STATUS_BY_PREFLIGHT_STATUS.get(preflight_status)
    if cuda_status is None:
        allowed = ", ".join(M03_FORWARD_PARITY_CUDA_ARTIFACT_PREFLIGHT_STATUSES)
        raise ValueError(f"M0.3 artifact_preflight_status must be one of {allowed}")
    status = "blocked" if preflight_status == "not_supplied" else "refused"
    if "required_artifact" not in cuda_preflight_payload:
        cuda_preflight_payload["required_artifact"] = M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH
    if "artifact_contract" not in cuda_preflight_payload:
        cuda_preflight_payload["artifact_contract"] = _json_safe_mapping(
            M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT
        )
    if "artifact_error" not in cuda_preflight_payload:
        cuda_preflight_payload["artifact_error"] = (
            "No CUDA reference artifact was supplied."
            if preflight_status == "not_supplied"
            else "CUDA reference artifact preflight was not run by the caller."
        )
    if "evaluates_logits" not in cuda_preflight_payload:
        cuda_preflight_payload["evaluates_logits"] = False
    if "preflight_is_acceptance" not in cuda_preflight_payload:
        cuda_preflight_payload["preflight_is_acceptance"] = False
    if preflight_status == "valid_not_evaluated":
        blocker = (
            "CUDA reference artifact passed metadata preflight, but logits were not compared; "
            "M0.3 closure requires a separate numerical parity harness."
        )
    elif preflight_status == "not_supplied":
        blocker = (
            "No CUDA reference artifact was supplied; M0.3 closure requires the exact external "
            f"CUDA logits artifact at {M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH} plus a separate "
            "numerical parity harness."
        )
    elif preflight_status == "missing":
        blocker = (
            "CUDA reference artifact path was supplied but the artifact is missing; "
            "M0.3 closure requires the artifact contract plus a separate numerical parity harness."
        )
    else:
        blocker = (
            "CUDA reference artifact preflight failed; M0.3 closure requires a valid artifact "
            "contract plus a separate numerical parity harness."
        )
    resolved_input_tokens_sha256 = (
        input_tokens_sha256
        if input_tokens_sha256 is not None
        else M03_FORWARD_PARITY_INPUT_TOKENS_SHA256
    )
    input_batch = _json_safe_mapping(M03_FORWARD_PARITY_FIXED_INPUT)
    input_batch.update(
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "seed": seed,
            "tokens_sha256": resolved_input_tokens_sha256,
            "fixed": True,
        }
    )
    cuda_reference_payload = _json_safe_mapping(cuda_reference or {})
    cuda_reference_payload.update(cuda_preflight_payload)
    cuda_reference_payload.update(
        {
            "status": cuda_status,
            "artifact": artifact_text,
            "artifact_supplied": artifact_supplied,
            "evaluated_by_this_manifest": False,
            "required_for_closure": True,
            "required_artifact": cuda_preflight_payload["required_artifact"],
            "artifact_contract": cuda_preflight_payload["artifact_contract"],
            "artifact_error": cuda_preflight_payload["artifact_error"],
        }
    )
    mlx_reference_payload = _json_safe_mapping(mlx_reference or {})
    mlx_reference_payload.update(
        {
            "status": "not_evaluated",
            "evaluated_by_this_manifest": False,
            "required_for_closure": True,
        }
    )

    manifest.update(
        {
            "m0_3_schema_version": 1,
            "m0_3_receipt_scope": M03_FORWARD_PARITY_RECEIPT_SCOPE,
            "status": status,
            "issue": {
                "id": M03_FORWARD_PARITY_ISSUE_ID,
                "title": "M0.3: random-init seed-matched forward parity vs CUDA reference",
            },
            "m0_3_policy": M03_FORWARD_PARITY_POLICY,
            "profile": profile,
            "profile_metadata": _json_safe_mapping(M03_FORWARD_PARITY_PROFILE_METADATA),
            "dtype": dtype,
            "seed": seed,
            "input_batch": input_batch,
            "tolerances": _json_safe_mapping(M03_FORWARD_PARITY_TOLERANCES),
            "random_init_seed_matched": True,
            "cuda_weight_import": False,
            "warm_start": False,
            "acceptance_status": "not_evaluated",
            "m0_3_closed": False,
            "full_m0_3_acceptance_claim": False,
            "gb10_forward_parity_claim": False,
            "m4_vs_gb10_forward_parity_claim": False,
            "distributed_megatron_forward_parity_claim": False,
            "claim_boundary": (
                "This scaffold makes no GB10-vs-M4 throughput, GB10 numerical, "
                "M4-vs-GB10 parity, or distributed Megatron parity claim."
            ),
            "cuda_reference": cuda_reference_payload,
            "mlx_reference": mlx_reference_payload,
            "acceptance_gate": {
                "requires_cuda_reference_artifact": True,
                "requires_external_cuda_logits": True,
                "requires_mlx_forward": True,
                "requires_same_seed": True,
                "requires_no_cuda_weight_import": True,
                "requires_fixed_input_batch": True,
                "requires_separate_numerical_harness": True,
                "cuda_reference_artifact_supplied": artifact_supplied,
                "cuda_reference_artifact_preflight_status": preflight_status,
                "receipt_count": len(rows),
                "logits_receipts": len(logits_receipts),
                "receipts_evaluated_by_this_manifest": False,
                "full_m0_3_acceptance": False,
                "blocker": blocker,
            },
        }
    )
    validate_m03_forward_parity_manifest_dict(manifest)
    return manifest


def validate_m03_forward_parity_manifest_dict(payload: Mapping[str, Any]) -> None:
    """Validate the M0.3 wrapper and its generic parity manifest payload."""

    validate_parity_manifest_dict(payload)
    if payload.get("m0_3_schema_version") != 1:
        raise ValueError("M0.3 manifest m0_3_schema_version must be 1")
    if payload.get("m0_3_receipt_scope") != M03_FORWARD_PARITY_RECEIPT_SCOPE:
        raise ValueError(
            f"M0.3 manifest m0_3_receipt_scope must be {M03_FORWARD_PARITY_RECEIPT_SCOPE!r}"
        )
    issue = _require_mapping(payload.get("issue"), label="M0.3 manifest issue")
    if issue.get("id") != M03_FORWARD_PARITY_ISSUE_ID:
        raise ValueError(f"M0.3 manifest issue.id must be {M03_FORWARD_PARITY_ISSUE_ID!r}")
    status = _require_non_empty_string(payload, "status")
    if status not in ("blocked", "refused"):
        raise ValueError("M0.3 manifest status must be blocked or refused")
    if payload.get("random_init_seed_matched") is not True:
        raise ValueError("M0.3 manifest random_init_seed_matched must be true")
    for field_name in (
        "cuda_weight_import",
        "warm_start",
        "gb10_forward_parity_claim",
        "m4_vs_gb10_forward_parity_claim",
        "distributed_megatron_forward_parity_claim",
    ):
        if payload.get(field_name) is not False:
            raise ValueError(f"M0.3 manifest {field_name} must be false")

    if payload.get("seed") != M03_FORWARD_PARITY_SEED:
        raise ValueError("M0.3 manifest seed must match fixed scaffold seed")
    if payload.get("profile") != M03_FORWARD_PARITY_PROFILE:
        raise ValueError(f"M0.3 manifest profile must be {M03_FORWARD_PARITY_PROFILE!r}")
    _require_non_empty_string(payload, "dtype")
    profile_metadata = _require_mapping(
        payload.get("profile_metadata"),
        label="M0.3 profile_metadata",
    )
    if profile_metadata != M03_FORWARD_PARITY_PROFILE_METADATA:
        raise ValueError("M0.3 profile_metadata must match local_gb10_quarter scaffold metadata")
    input_batch = _require_mapping(payload.get("input_batch"), label="M0.3 input_batch")
    if input_batch.get("batch_size") != M03_FORWARD_PARITY_BATCH_SIZE:
        raise ValueError("M0.3 input_batch.batch_size must match fixed scaffold batch size")
    if input_batch.get("seq_len") != M03_FORWARD_PARITY_SEQ_LEN:
        raise ValueError("M0.3 input_batch.seq_len must match fixed scaffold sequence length")
    if input_batch.get("seed") != M03_FORWARD_PARITY_SEED:
        raise ValueError("M0.3 input_batch.seed must match fixed scaffold seed")
    if input_batch.get("vocab_size") != M03_FORWARD_PARITY_VOCAB_SIZE:
        raise ValueError("M0.3 input_batch.vocab_size must match fixed scaffold vocab size")
    if input_batch.get("fixed") is not True:
        raise ValueError("M0.3 input_batch.fixed must be true")
    if input_batch.get("tokens_sha256_required_for_closure") is not True:
        raise ValueError("M0.3 input_batch.tokens_sha256_required_for_closure must be true")
    tokens_sha256 = input_batch.get("tokens_sha256")
    if tokens_sha256 != M03_FORWARD_PARITY_INPUT_TOKENS_SHA256:
        raise ValueError(
            "M0.3 input_batch.tokens_sha256 must match fixed input hash "
            f"{M03_FORWARD_PARITY_INPUT_TOKENS_SHA256}"
        )
    tolerances = _require_mapping(payload.get("tolerances"), label="M0.3 tolerances")
    if tolerances != M03_FORWARD_PARITY_TOLERANCES:
        raise ValueError("M0.3 tolerances must match the scaffold tolerance policy")

    if payload.get("acceptance_status") != "not_evaluated":
        raise ValueError("M0.3 acceptance_status must be not_evaluated")
    if payload.get("m0_3_closed") is not False:
        raise ValueError("M0.3 m0_3_closed must be false")
    if payload.get("full_m0_3_acceptance_claim") is not False:
        raise ValueError("M0.3 full_m0_3_acceptance_claim must be false for scaffold manifests")

    gate = _require_mapping(payload.get("acceptance_gate"), label="M0.3 acceptance_gate")
    for field_name in (
        "requires_cuda_reference_artifact",
        "requires_external_cuda_logits",
        "requires_mlx_forward",
        "requires_same_seed",
        "requires_no_cuda_weight_import",
        "requires_fixed_input_batch",
        "requires_separate_numerical_harness",
    ):
        if gate.get(field_name) is not True:
            raise ValueError(f"M0.3 acceptance_gate.{field_name} must be true")
    if gate.get("receipts_evaluated_by_this_manifest") is not False:
        raise ValueError("M0.3 acceptance_gate.receipts_evaluated_by_this_manifest must be false")
    if gate.get("full_m0_3_acceptance") is not False:
        raise ValueError("M0.3 acceptance_gate.full_m0_3_acceptance must be false")
    blocker = gate.get("blocker")
    if not isinstance(blocker, str) or not blocker:
        raise ValueError("M0.3 acceptance_gate.blocker must be a non-empty string")

    cuda_reference = _require_mapping(payload.get("cuda_reference"), label="M0.3 cuda_reference")
    artifact_supplied = cuda_reference.get("artifact_supplied")
    if not isinstance(artifact_supplied, bool):
        raise ValueError("M0.3 cuda_reference.artifact_supplied must be boolean")
    if gate.get("cuda_reference_artifact_supplied") is not artifact_supplied:
        raise ValueError("M0.3 acceptance_gate.cuda_reference_artifact_supplied mismatch")
    preflight_status = cuda_reference.get("artifact_preflight_status")
    if preflight_status not in M03_FORWARD_PARITY_CUDA_ARTIFACT_PREFLIGHT_STATUSES:
        allowed = ", ".join(M03_FORWARD_PARITY_CUDA_ARTIFACT_PREFLIGHT_STATUSES)
        raise ValueError(f"M0.3 cuda_reference.artifact_preflight_status must be one of {allowed}")
    if gate.get("cuda_reference_artifact_preflight_status") != preflight_status:
        raise ValueError("M0.3 acceptance_gate.cuda_reference_artifact_preflight_status mismatch")
    expected_supplied = preflight_status != "not_supplied"
    if artifact_supplied is not expected_supplied:
        raise ValueError("M0.3 cuda_reference.artifact_supplied must match preflight status")
    expected_cuda_status = _M03_CUDA_STATUS_BY_PREFLIGHT_STATUS[preflight_status]
    if cuda_reference.get("status") != expected_cuda_status:
        raise ValueError(f"M0.3 cuda_reference.status must be {expected_cuda_status!r}")
    if cuda_reference.get("evaluated_by_this_manifest") is not False:
        raise ValueError("M0.3 cuda_reference.evaluated_by_this_manifest must be false")
    if cuda_reference.get("required_for_closure") is not True:
        raise ValueError("M0.3 cuda_reference.required_for_closure must be true")
    if cuda_reference.get("evaluates_logits") is not False:
        raise ValueError("M0.3 cuda_reference.evaluates_logits must be false")
    if cuda_reference.get("preflight_is_acceptance") is not False:
        raise ValueError("M0.3 cuda_reference.preflight_is_acceptance must be false")
    if cuda_reference.get("required_artifact") != M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH:
        raise ValueError(
            "M0.3 cuda_reference.required_artifact must be "
            f"{M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH!r}"
        )
    contract = _require_mapping(
        cuda_reference.get("artifact_contract"),
        label="M0.3 cuda_reference.artifact_contract",
    )
    if contract != M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT:
        raise ValueError("M0.3 cuda_reference.artifact_contract must match required contract")
    artifact_error = cuda_reference.get("artifact_error")
    if preflight_status in {"missing", "invalid"}:
        if not isinstance(artifact_error, str) or not artifact_error:
            raise ValueError("M0.3 cuda_reference.artifact_error must describe refusal")
    elif artifact_error is not None and not isinstance(artifact_error, str):
        raise ValueError("M0.3 cuda_reference.artifact_error must be null or a string")
    if preflight_status == "valid_not_evaluated":
        if artifact_error is not None:
            raise ValueError("M0.3 cuda_reference.artifact_error must be null after valid preflight")
        if cuda_reference.get("artifact_format") != M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT:
            raise ValueError(
                "M0.3 cuda_reference.artifact_format must be "
                f"{M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT!r}"
            )
        if cuda_reference.get("artifact_tensor_name") != M03_FORWARD_PARITY_LOGITS_TENSOR_NAME:
            raise ValueError(
                "M0.3 cuda_reference.artifact_tensor_name must be "
                f"{M03_FORWARD_PARITY_LOGITS_TENSOR_NAME!r}"
            )
        _require_sha256(
            cuda_reference,
            "artifact_logits_sha256",
            label="M0.3 cuda_reference",
        )
        _require_non_empty_string(cuda_reference, "artifact_source_commit")
        _require_non_empty_string(cuda_reference, "artifact_hardware")
        _require_non_empty_string(cuda_reference, "artifact_cuda_runtime")
    artifact = cuda_reference.get("artifact")
    if artifact_supplied:
        if status != "refused":
            raise ValueError("M0.3 artifact-supplied scaffold status must be refused")
        if not isinstance(artifact, str) or not artifact:
            raise ValueError("M0.3 cuda_reference.artifact must be non-empty when supplied")
    else:
        if status != "blocked":
            raise ValueError("M0.3 missing-artifact scaffold status must be blocked")
        if artifact is not None:
            raise ValueError("M0.3 cuda_reference.artifact must be null when not supplied")

    mlx_reference = _require_mapping(payload.get("mlx_reference"), label="M0.3 mlx_reference")
    if mlx_reference.get("status") != "not_evaluated":
        raise ValueError("M0.3 mlx_reference.status must be not_evaluated")
    if mlx_reference.get("evaluated_by_this_manifest") is not False:
        raise ValueError("M0.3 mlx_reference.evaluated_by_this_manifest must be false")
    if mlx_reference.get("required_for_closure") is not True:
        raise ValueError("M0.3 mlx_reference.required_for_closure must be true")
    readiness_status = mlx_reference.get("readiness_status")
    if readiness_status is not None and readiness_status not in {"pass", "fail", "skipped"}:
        raise ValueError("M0.3 mlx_reference.readiness_status must be pass, fail, or skipped")
    _require_false_when_present(
        mlx_reference,
        "readiness_is_cuda_parity",
        label="M0.3 mlx_reference",
    )
    _require_false_when_present(
        mlx_reference,
        "full_profile_allocation_executed",
        label="M0.3 mlx_reference",
    )
    _require_false_when_present(
        mlx_reference,
        "full_profile_forward_executed",
        label="M0.3 mlx_reference",
    )
    mlx_scope = mlx_reference.get("local_mlx_forward_scope")
    if mlx_scope is not None and mlx_scope not in {"tiny_smoke_only", "skipped"}:
        raise ValueError(
            "M0.3 mlx_reference.local_mlx_forward_scope must be tiny_smoke_only or skipped"
        )
    if mlx_reference.get("closure_required_mlx_forward_scope") not in (
        None,
        "full_local_gb10_quarter_logits",
    ):
        raise ValueError(
            "M0.3 mlx_reference.closure_required_mlx_forward_scope must be "
            "'full_local_gb10_quarter_logits'"
        )
    if mlx_scope == "skipped":
        _require_false_when_present(
            mlx_reference,
            "local_mlx_forward_executed",
            label="M0.3 mlx_reference",
        )
        _require_false_when_present(
            mlx_reference,
            "tiny_smoke_forward_executed",
            label="M0.3 mlx_reference",
        )


def write_m03_forward_parity_manifest_json(
    path: str | Path,
    receipts: Iterable[TensorParityReceipt | Mapping[str, Any]],
    *,
    seed: int = M03_FORWARD_PARITY_SEED,
    batch_size: int = M03_FORWARD_PARITY_BATCH_SIZE,
    seq_len: int = M03_FORWARD_PARITY_SEQ_LEN,
    profile: str = M03_FORWARD_PARITY_PROFILE,
    dtype: str = "bf16",
    source: str | None = None,
    input_tokens_sha256: str | None = None,
    cuda_reference_artifact: str | Path | None = None,
    cuda_reference: Mapping[str, Any] | None = None,
    cuda_reference_preflight: Mapping[str, Any] | None = None,
    mlx_reference: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> JsonObject:
    manifest = build_m03_forward_parity_manifest(
        receipts,
        seed=seed,
        batch_size=batch_size,
        seq_len=seq_len,
        profile=profile,
        dtype=dtype,
        source=source,
        input_tokens_sha256=input_tokens_sha256,
        cuda_reference_artifact=cuda_reference_artifact,
        cuda_reference=cuda_reference,
        cuda_reference_preflight=cuda_reference_preflight,
        mlx_reference=mlx_reference,
        metadata=metadata,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "LOCAL_ONLY_POLICY",
    "M03_FORWARD_PARITY_ATOL",
    "M03_FORWARD_PARITY_BATCH_SIZE",
    "M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT",
    "M03_FORWARD_PARITY_CUDA_ARTIFACT_FORMAT",
    "M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH",
    "M03_FORWARD_PARITY_CUDA_ARTIFACT_PREFLIGHT_STATUSES",
    "M03_FORWARD_PARITY_CUDA_LOGITS_DTYPES",
    "M03_FORWARD_PARITY_CUDA_SUMMARY_FIELDS",
    "M03_FORWARD_PARITY_FIXED_INPUT",
    "M03_FORWARD_PARITY_INPUT_TOKENS_SHA256",
    "M03_FORWARD_PARITY_ISSUE_ID",
    "M03_FORWARD_PARITY_LOGITS_NUMEL",
    "M03_FORWARD_PARITY_LOGITS_SHAPE",
    "M03_FORWARD_PARITY_LOGITS_TENSOR_NAME",
    "M03_FORWARD_PARITY_OUTPUT",
    "M03_FORWARD_PARITY_POLICY",
    "M03_FORWARD_PARITY_PROFILE",
    "M03_FORWARD_PARITY_PROFILE_METADATA",
    "M03_FORWARD_PARITY_RECEIPT_SCOPE",
    "M03_FORWARD_PARITY_RTOL",
    "M03_FORWARD_PARITY_SEED",
    "M03_FORWARD_PARITY_SEQ_LEN",
    "M03_FORWARD_PARITY_TOLERANCES",
    "M03_FORWARD_PARITY_VOCAB_SIZE",
    "PARITY_MANIFEST_FORMAT",
    "PARITY_MANIFEST_VERSION",
    "PARITY_RECEIPT_SCOPE",
    "REQUIRED_RECEIPT_FIELDS",
    "TensorParityReceipt",
    "VALID_PARITY_STATUSES",
    "build_m03_forward_parity_manifest",
    "build_parity_manifest",
    "coerce_parity_receipt",
    "validate_m03_cuda_reference_artifact_dict",
    "validate_m03_forward_parity_manifest_dict",
    "validate_parity_manifest_dict",
    "validate_parity_receipt_dict",
    "write_m03_forward_parity_manifest_json",
    "write_parity_manifest_json",
    "write_parity_manifest_jsonl",
]
