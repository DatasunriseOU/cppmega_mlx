"""Production-only ingress for immutable cppmega Megatron bundles."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from cppmega_mlx.config.model import (
    LOCAL_PROFILE_VOCAB_SIZE,
    MEGACPP_TOKENIZER_VOCAB_SIZE,
)
from cppmega_mlx.data.dataset_metadata import TokenDatasetMetadata
from cppmega_mlx.data.domain_schema import DOMAIN_SCHEMA_SHA256
from cppmega_mlx.data.nanochat_pipeline.packed_rows_schema import (
    LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1,
)
from cppmega_mlx.data.tokenizer_contract import (
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256,
)
from cppmega_mlx.data.graph_recipe import (
    stage1_graph_recipe_binding,
    validate_stage1_graph_contract,
)

if TYPE_CHECKING:
    from cppmega_mlx.data.megatron_indexed import MegatronIndexedDataset


_BUNDLE_SCHEMA = "cppmega_megatron_bundle_v3"
_BUNDLE_TOKENIZER_CONTRACT = "megacpp-vocab-65536"
_PREFIX_TOKENIZER_CONTRACT = "megacpp"
_EXPECTED_VOCAB_SIZE = 65536
_TRAINING_CONTRACT = "objective_materialized"
_IMPLEMENTATION_BINDING_SCHEMA = "cppmega_implementation_binding_v1"
_PRODUCER_IMPLEMENTATION_COMPONENTS = (
    "cppmega",
    "cppmega_mlx",
    "clang_indexer",
)
_TRAINING_IMPLEMENTATION_COMPONENTS = (
    "cppmega",
    "megatron",
    "cppmega_mlx",
    "clang_indexer",
)
_IMPLEMENTATION_COMPONENT_FIELDS = {
    "cppmega": frozenset({"commit", "tree_sha256"}),
    "cppmega_mlx": frozenset({"commit", "tree_sha256"}),
    "clang_indexer": frozenset(
        {"source_sha256", "dependency_closure_sha256"}
    ),
    "megatron": frozenset({"commit"}),
}
_OBJECTIVE_BUCKETS_SCHEMA = "cppmega_bucketed_objective_materializations_v1"
_OBJECTIVE_CONTRACT_SCHEMA = "cppmega_pre_materialized_objectives_v1"
_OBJECTIVE_ARTIFACT_SCHEMA = "cppmega_objective_materialization_artifact_v2"
_LEGACY_OBJECTIVE_ARTIFACT_SCHEMA = "cppmega_objective_materialization_artifact_v1"
_OBJECTIVE_SOURCE_SCHEMA = "cppmega_objective_source_snapshot_v1"
_LEGACY_SOURCE_SAMPLING_MODE = "deterministic_epoch_shuffle_v1"
_BOUNDED_SOURCE_SAMPLING_MODE = (
    "deterministic_shard_row_group_record_batch_shuffle_v2"
)
_BOUNDED_SOURCE_ORDERING = {
    "permutation": "sha256_sort_key_v1",
    "epochs": "ascending",
    "shards": "seeded_permutation_per_epoch",
    "row_groups": "seeded_permutation_per_shard_epoch",
    "record_batches": "physical_order_within_row_group",
    "rows": "seeded_permutation_within_record_batch",
}
_BOUNDED_SOURCE_PRODUCER = "pyarrow.parquet.ParquetFile.iter_batches"
_BOUNDED_SOURCE_PRODUCER_VERSION = 1
_BOUNDED_SOURCE_CURSOR_FIELDS = {
    "epoch",
    "shard_position",
    "shard_index",
    "row_group_position",
    "row_group_index",
    "record_batch_index",
    "row_shuffle_position",
    "row_index_in_record_batch",
    "source_index",
}
_GRAPH_SIDECAR_SCHEMA = "cppmega_graph_routes_v2"
_CASE5_RECEIPT_KEY = "case5_domain_ingestion_receipt"
_CASE5_SCHEMA = "case5_domain_routes_v1"
_SOURCE_IDENTITY_REGISTRY_SCHEMA = "cppmega_source_identity_registry_v1"
_RESTORE_RECEIPT_SCHEMA = "cppmega_megatron_restore_receipt_v1"
_RESTORE_BINDING_SCHEMA = "cppmega_case6_receipt_binding_v2"
_SOURCE_COMPOSITION_SCHEMA = "cppmega_source_conveyor_composition_v1"
_GLOBAL_DEDUP_RECEIPT_SCHEMA = "cppmega_global_dedup_store_receipt_v1"
_SOURCE_FULL_LAUNCH_SCHEMA = "cppmega.canonical_source_launch_v1"
_SOURCE_FULL_EXIT_SCHEMA = "cppmega.canonical_source_exit_v1"
_SOURCE_TARGETED_LAUNCH_SCHEMA = (
    "cppmega.canonical_source_targeted_retry_launch_v1"
)
_SOURCE_TARGETED_EXIT_SCHEMA = (
    "cppmega.canonical_source_targeted_retry_exit_v1"
)
_CI_PRODUCTION_EXPORT_SCHEMA = "cppmega_ci_content_store_case5_export_v3"
_CI_FETCH_STATE_SCHEMA = "cppmega_ci_stream_fetch_v4"
_CI_PRODUCTION_MERGE_SCHEMA = "cppmega_ci_stream_shard_union_receipt_v3"
_CI_COMPLETION_MODE = "inventory-exhaustive"
_CI_MIN_EXACT_UNIQUE_PAYLOAD_TOKENS = 20_000_000_000
_CI_PROVENANCE_LEDGER_NAMES = {
    "representative_ledger": "representative_ledger.jsonl",
    "fragment_ledger": "fragment_ledger.jsonl",
    "dropped_graph_edges": "dropped_graph_edges.jsonl",
    "representative_metadata": "representative_metadata.jsonl",
    "excluded_opaque_artifacts": "excluded_opaque_artifacts.jsonl",
    "source_binding_projection": "source_binding_projection.jsonl",
}
_SOURCE_EXIT_SCHEMA_BY_LAUNCH = {
    _SOURCE_FULL_LAUNCH_SCHEMA: _SOURCE_FULL_EXIT_SCHEMA,
    _SOURCE_TARGETED_LAUNCH_SCHEMA: _SOURCE_TARGETED_EXIT_SCHEMA,
}
_GLOBAL_DEDUP_TABLES = frozenset(
    {
        "exact",
        "lsh",
        "minhash",
        "dedup_meta",
        "chunk_claims",
        "dedup_stages",
        "exact_stage",
        "minhash_stage",
        "lsh_stage",
        "chunk_claims_stage",
    }
)
_SOURCE_COMPOSITION_RUN_ARTIFACTS = frozenset(
    {
        "launch",
        "exit",
        "manifest",
        "archive_sha256_receipt",
        "archive_inventory",
        "repo_list",
        "source_quarantine_manifest",
        "tokenizer",
    }
)
_NO_CHECKPOINT_SHA256 = hashlib.sha256(b"cppmega:no-checkpoint:v1").hexdigest()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RELEVANT_UNLISTED_SUFFIXES = frozenset({".bin", ".idx", ".json"})

_DOMAIN_ROUTE_COLUMNS = (
    "token_domain_ids",
    "token_role_ids",
    "token_entity_ids",
    "token_scope_ids",
    "token_source_doc_ids",
    "token_source_identity_ids",
    "token_confidence_ids",
)
_GRAPH_ROUTE_COLUMNS = (
    "token_call_edges",
    "token_type_edges",
    "token_domain_edges",
    "token_build_edges",
    "token_shell_edges",
    "token_diagnostic_edges",
    "token_cross_domain_edges",
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
)
_TOKEN_SIDECAR_DTYPES = {
    "loss_mask": "uint8",
    "doc_ids": "uint32",
    "token_domain_ids": "uint16",
    "token_role_ids": "uint16",
    "token_entity_ids": "uint32",
    "token_scope_ids": "uint32",
    "token_source_doc_ids": "uint32",
    "token_source_identity_ids": "uint64",
    "token_confidence_ids": "uint8",
    "token_structure_ids": "uint8",
    "token_dep_levels": "uint16",
    "token_ast_depth": "uint16",
    "token_sibling_index": "uint16",
    "token_ast_node_type": "uint16",
    "token_symbol_ids": "uint64",
    "token_call_targets": "uint64",
    "token_type_refs": "uint64",
    "token_def_use": "uint8",
    "token_change_mask_pre": "uint8",
    "token_change_mask_post": "uint8",
}
_OBJECTIVE_DESCRIPTOR_FIELDS = {
    "artifact_path",
    "artifact_schema",
    "artifact_set_sha256",
    "artifact_file_sha256",
    "contract_path",
    "contract_schema",
    "contract_sha256",
    "contract_file_sha256",
    "source_snapshot",
}


@dataclass(frozen=True)
class ProductionMegatronDatasetMetadata(TokenDatasetMetadata):
    """Validated immutable-bundle provenance carried by a production dataset."""

    training_contract: str = ""
    bucket: int = 0
    bundle_id: str = ""
    bundle_prefix: str = ""
    artifact_set_sha256: str = ""
    logical_manifest_sha256: str = ""
    prefix_manifest_sha256: str = ""
    restore_receipt_sha256: str = ""
    restore_run_id: str = ""
    storage_bucket: str = ""
    source_snapshot_artifact_set_sha256: str = ""
    source_manifest_sha256: str = ""
    repaired_source_manifest_sha256: str = ""
    source_composition_receipt_sha256: str = ""
    source_producer_set_sha256: str = ""
    global_dedup_receipt_sha256: str = ""
    global_dedup_logical_sha256: str = ""
    ci_manifest_sha256: str = ""
    ci_source_inventory_sha256: str = ""
    ci_acquisition_provenance_sha256: str = ""
    ci_exact_unique_payload_tokens: int = 0
    objective_contract_sha256: str = ""
    objective_contract_file_sha256: str = ""
    objective_artifact_set_sha256: str = ""
    objective_artifact_file_sha256: str = ""
    tokenizer_artifact_set_sha256: str = ""
    tokenizer_contract_sha256: str = ""
    domain_schema_sha256: str = ""
    case5_delimiter_contract_sha256: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_contract != _TRAINING_CONTRACT:
            raise ValueError("production metadata has an invalid training contract")
        if self.bucket < 1:
            raise ValueError("production metadata bucket must be positive")
        if not _IDENTIFIER_RE.fullmatch(self.bundle_id):
            raise ValueError("production metadata bundle_id is invalid")
        if not self.bundle_prefix:
            raise ValueError("production metadata bundle_prefix is required")
        if not self.storage_bucket:
            raise ValueError("production metadata storage_bucket is required")
        if not _RUN_ID_RE.fullmatch(self.restore_run_id):
            raise ValueError("production metadata restore_run_id is invalid")
        for field_name in (
            "artifact_set_sha256",
            "logical_manifest_sha256",
            "prefix_manifest_sha256",
            "restore_receipt_sha256",
            "source_snapshot_artifact_set_sha256",
            "source_manifest_sha256",
            "repaired_source_manifest_sha256",
            "source_composition_receipt_sha256",
            "source_producer_set_sha256",
            "global_dedup_receipt_sha256",
            "global_dedup_logical_sha256",
            "ci_manifest_sha256",
            "ci_source_inventory_sha256",
            "ci_acquisition_provenance_sha256",
            "objective_contract_sha256",
            "objective_contract_file_sha256",
            "objective_artifact_set_sha256",
            "objective_artifact_file_sha256",
            "tokenizer_artifact_set_sha256",
            "tokenizer_contract_sha256",
            "domain_schema_sha256",
            "case5_delimiter_contract_sha256",
        ):
            _require_sha256(getattr(self, field_name), where=f"metadata.{field_name}")
        if self.ci_exact_unique_payload_tokens < _CI_MIN_EXACT_UNIQUE_PAYLOAD_TOKENS:
            raise ValueError(
                "production metadata lacks the minimum exact unique CI payload"
            )

    def provenance_receipt(self) -> dict[str, object]:
        """Return the stable fields that every run/checkpoint receipt must retain."""

        return {
            "training_contract": self.training_contract,
            "bucket": self.bucket,
            "bundle_id": self.bundle_id,
            "bundle_prefix": self.bundle_prefix,
            "artifact_set_sha256": self.artifact_set_sha256,
            "logical_manifest_sha256": self.logical_manifest_sha256,
            "prefix_manifest_sha256": self.prefix_manifest_sha256,
            "restore_receipt_sha256": self.restore_receipt_sha256,
            "restore_run_id": self.restore_run_id,
            "storage_bucket": self.storage_bucket,
            "source_snapshot_artifact_set_sha256": (
                self.source_snapshot_artifact_set_sha256
            ),
            "source_manifest_sha256": self.source_manifest_sha256,
            "repaired_source_manifest_sha256": (self.repaired_source_manifest_sha256),
            "source_composition_receipt_sha256": (
                self.source_composition_receipt_sha256
            ),
            "source_producer_set_sha256": self.source_producer_set_sha256,
            "global_dedup_receipt_sha256": self.global_dedup_receipt_sha256,
            "global_dedup_logical_sha256": self.global_dedup_logical_sha256,
            "ci_manifest_sha256": self.ci_manifest_sha256,
            "ci_source_inventory_sha256": self.ci_source_inventory_sha256,
            "ci_acquisition_provenance_sha256": (
                self.ci_acquisition_provenance_sha256
            ),
            "ci_exact_unique_payload_tokens": self.ci_exact_unique_payload_tokens,
            "objective_contract_sha256": self.objective_contract_sha256,
            "objective_contract_file_sha256": self.objective_contract_file_sha256,
            "objective_artifact_set_sha256": self.objective_artifact_set_sha256,
            "objective_artifact_file_sha256": self.objective_artifact_file_sha256,
            "tokenizer_artifact_set_sha256": self.tokenizer_artifact_set_sha256,
            "tokenizer_contract_sha256": self.tokenizer_contract_sha256,
            "domain_schema_sha256": self.domain_schema_sha256,
            "case5_delimiter_contract_sha256": (self.case5_delimiter_contract_sha256),
        }


@dataclass(frozen=True)
class _ObjectiveValidation:
    contract: dict[str, Any]
    artifact: dict[str, Any]
    source_summary: dict[str, object]
    descriptor: dict[str, Any]


@dataclass(frozen=True)
class _PrefixValidation:
    prefix: Path
    relative: str
    manifest_sha256: str


@dataclass(frozen=True)
class _RestoreValidation:
    receipt_sha256: str
    run_id: str
    storage_bucket: str


@dataclass(frozen=True)
class _SourceCompositionValidation:
    receipt_sha256: str
    producer_set_sha256: str
    dedup_receipt_sha256: str
    dedup_logical_sha256: str


@dataclass(frozen=True)
class _CIValidation:
    manifest_sha256: str
    source_inventory_sha256: str
    acquisition_provenance_sha256: str
    exact_unique_payload_tokens: int


@dataclass(frozen=True)
class _ValidatedBundle:
    manifest: dict[str, Any]
    artifacts: dict[str, dict[str, object]]
    artifact_stats: dict[str, tuple[int, int, int, int, int]]
    logical_manifest_sha256: str
    tokenizer_artifact_set_sha256: str
    source_manifest_sha256: str
    repaired_source_manifest_sha256: str
    source_composition: _SourceCompositionValidation
    ci: _CIValidation
    objectives: dict[int, _ObjectiveValidation]
    prefixes: dict[int, _PrefixValidation]
    restore: _RestoreValidation


def open_production_megatron_bundle(
    bundle_root: str | Path,
    bucket: int,
    expected_bundle_id: str,
    *,
    restore_receipt: str | Path | None,
    seq_len: int,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    loop: bool = False,
    resume_batch: int = 0,
    hash_jobs: int = 4,
) -> MegatronIndexedDataset:
    """Validate and open one immutable production bundle bucket.

    ``restore_receipt`` must name the retained ``restore_receipt.json`` inside
    ``bundle_root``. All manifest-listed files are byte-verified before this
    function invokes the mmap-backed bare reader.
    """

    if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket < 1:
        raise ValueError("production Megatron bucket must be a positive integer")
    if seq_len != bucket:
        raise ValueError(
            f"production seq_len must equal requested bucket: {seq_len} != {bucket}"
        )
    if not isinstance(expected_bundle_id, str) or not _IDENTIFIER_RE.fullmatch(
        expected_bundle_id
    ):
        raise ValueError("expected_bundle_id is missing or invalid")
    if restore_receipt is None:
        raise ValueError("production Megatron ingress requires a restore receipt")
    if isinstance(hash_jobs, bool) or not isinstance(hash_jobs, int) or hash_jobs < 1:
        raise ValueError("hash_jobs must be a positive integer")

    root = Path(os.path.abspath(os.fspath(bundle_root)))
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"production bundle root must be a regular directory: {root}")
    root = root.resolve()
    receipt_path = Path(restore_receipt)
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    receipt_path = Path(os.path.abspath(os.fspath(receipt_path)))
    expected_receipt_path = root / "restore_receipt.json"
    if receipt_path != expected_receipt_path:
        raise ValueError(
            "restore receipt must be the retained bundle-root restore_receipt.json"
        )

    validated = _validate_bundle(
        root,
        bucket=bucket,
        expected_bundle_id=expected_bundle_id,
        restore_receipt_path=receipt_path,
        hash_jobs=hash_jobs,
    )
    selected_objective = validated.objectives[bucket]
    selected_prefix = validated.prefixes[bucket]
    metadata = ProductionMegatronDatasetMetadata(
        vocab_size=int(validated.manifest["vocab_size"]),
        tokenizer_contract="megacpp",
        local_profile_vocab_size=LOCAL_PROFILE_VOCAB_SIZE,
        megacpp_tokenizer_vocab_size=MEGACPP_TOKENIZER_VOCAB_SIZE,
        source_format="megatron-production-bundle",
        training_contract=_TRAINING_CONTRACT,
        bucket=bucket,
        bundle_id=expected_bundle_id,
        bundle_prefix=selected_prefix.relative,
        artifact_set_sha256=str(validated.manifest["artifact_set_sha256"]),
        logical_manifest_sha256=validated.logical_manifest_sha256,
        prefix_manifest_sha256=selected_prefix.manifest_sha256,
        restore_receipt_sha256=validated.restore.receipt_sha256,
        restore_run_id=validated.restore.run_id,
        storage_bucket=validated.restore.storage_bucket,
        source_snapshot_artifact_set_sha256=str(
            selected_objective.source_summary["artifact_set_sha256"]
        ),
        source_manifest_sha256=validated.source_manifest_sha256,
        repaired_source_manifest_sha256=validated.repaired_source_manifest_sha256,
        source_composition_receipt_sha256=(
            validated.source_composition.receipt_sha256
        ),
        source_producer_set_sha256=(
            validated.source_composition.producer_set_sha256
        ),
        global_dedup_receipt_sha256=(
            validated.source_composition.dedup_receipt_sha256
        ),
        global_dedup_logical_sha256=(
            validated.source_composition.dedup_logical_sha256
        ),
        ci_manifest_sha256=validated.ci.manifest_sha256,
        ci_source_inventory_sha256=validated.ci.source_inventory_sha256,
        ci_acquisition_provenance_sha256=(
            validated.ci.acquisition_provenance_sha256
        ),
        ci_exact_unique_payload_tokens=validated.ci.exact_unique_payload_tokens,
        objective_contract_sha256=str(selected_objective.descriptor["contract_sha256"]),
        objective_contract_file_sha256=str(
            selected_objective.descriptor["contract_file_sha256"]
        ),
        objective_artifact_set_sha256=str(
            selected_objective.descriptor["artifact_set_sha256"]
        ),
        objective_artifact_file_sha256=str(
            selected_objective.descriptor["artifact_file_sha256"]
        ),
        tokenizer_artifact_set_sha256=(validated.tokenizer_artifact_set_sha256),
        tokenizer_contract_sha256=TOKENIZER_CONTRACT_SHA256,
        domain_schema_sha256=DOMAIN_SCHEMA_SHA256,
        case5_delimiter_contract_sha256=DOMAIN_DELIMITER_CONTRACT_SHA256,
    )

    _assert_artifacts_unchanged(root, validated.artifact_stats)
    from cppmega_mlx.data.megatron_indexed import (
        MegatronIndexedDataset,
        open_megatron_indexed_dataset,
    )

    dataset = open_megatron_indexed_dataset(
        selected_prefix.prefix,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        loop=loop,
        resume_batch=resume_batch,
        metadata=metadata,
    )
    if not isinstance(dataset, MegatronIndexedDataset):
        raise RuntimeError(
            "production bundle prefix opened as an unexpected dataset type"
        )
    _assert_artifacts_unchanged(root, validated.artifact_stats)
    return dataset


def _validate_bundle(
    root: Path,
    *,
    bucket: int,
    expected_bundle_id: str,
    restore_receipt_path: Path,
    hash_jobs: int,
) -> _ValidatedBundle:
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_bytes, manifest = _load_json_object(manifest_path, where="bundle manifest")
    artifacts = _validate_logical_manifest(
        root,
        manifest,
        expected_bundle_id=expected_bundle_id,
        bucket=bucket,
    )
    _reject_symlinks_and_unlisted_relevant_files(
        root,
        listed=set(artifacts),
        restore_receipt_path=restore_receipt_path,
    )
    artifact_stats = _verify_all_artifacts(
        root, artifacts=artifacts, hash_jobs=hash_jobs
    )

    tokenizer_digest = _validate_tokenizer(root, manifest, artifacts)
    _validate_data_contracts(root, manifest, artifacts)
    source_composition = _validate_source_composition(
        root, manifest, artifacts
    )
    source_sha, repaired_sha, repaired_by_bucket = _validate_source_manifests(
        root, manifest, artifacts
    )
    ci = _validate_ci_production_acquisition(root, manifest, artifacts)
    buckets = [int(value) for value in manifest["buckets"]]
    objectives = _validate_objectives(
        root,
        manifest,
        artifacts,
        buckets=buckets,
        repaired_by_bucket=repaired_by_bucket,
    )
    prefixes = _validate_bucket_prefixes(
        root,
        manifest,
        artifacts,
        buckets=buckets,
        objectives=objectives,
    )
    logical_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    restore = _validate_restore_receipt(
        root,
        restore_receipt_path,
        manifest=manifest,
        artifacts=artifacts,
        prefixes=prefixes,
        expected_bundle_id=expected_bundle_id,
        logical_manifest_sha256=logical_manifest_sha256,
    )
    return _ValidatedBundle(
        manifest=manifest,
        artifacts=artifacts,
        artifact_stats=artifact_stats,
        logical_manifest_sha256=logical_manifest_sha256,
        tokenizer_artifact_set_sha256=tokenizer_digest,
        source_manifest_sha256=source_sha,
        repaired_source_manifest_sha256=repaired_sha,
        source_composition=source_composition,
        ci=ci,
        objectives=objectives,
        prefixes=prefixes,
        restore=restore,
    )


def _validate_logical_manifest(
    root: Path,
    manifest: dict[str, Any],
    *,
    expected_bundle_id: str,
    bucket: int,
) -> dict[str, dict[str, object]]:
    if manifest.get("schema") != _BUNDLE_SCHEMA:
        raise ValueError(
            f"unsupported production bundle schema: {manifest.get('schema')!r}"
        )
    if manifest.get("training_contract") != _TRAINING_CONTRACT:
        raise ValueError(
            "production bundle must use training_contract='objective_materialized'"
        )
    if manifest.get("known_limitations") != []:
        raise ValueError("complete production bundle known_limitations must be empty")
    _validate_implementation_binding(
        manifest.get("implementation"),
        where="production bundle",
        required_components=_PRODUCER_IMPLEMENTATION_COMPONENTS,
    )
    if manifest.get("bundle_id") != expected_bundle_id:
        raise ValueError(
            f"bundle_id mismatch: {manifest.get('bundle_id')!r} != {expected_bundle_id!r}"
        )
    if manifest.get("tokenizer_contract") != _BUNDLE_TOKENIZER_CONTRACT:
        raise ValueError("production bundle tokenizer contract is unsupported")
    if int(manifest.get("vocab_size", -1)) != _EXPECTED_VOCAB_SIZE:
        raise ValueError("production bundle vocab_size is unsupported")
    expected_layout = {
        "token_column": "input_ids",
        "length_column": "valid_token_count",
        "writer_backend": "mmididx",
    }
    layout_drift = {
        name: manifest.get(name)
        for name, expected in expected_layout.items()
        if manifest.get(name) != expected
    }
    if layout_drift:
        raise ValueError(f"production bundle layout contract drifted: {layout_drift}")

    raw_buckets = manifest.get("buckets")
    if (
        not isinstance(raw_buckets, list)
        or not raw_buckets
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in raw_buckets
        )
        or len(set(raw_buckets)) != len(raw_buckets)
    ):
        raise ValueError("bundle buckets must be unique positive integers")
    if bucket not in raw_buckets:
        raise ValueError(f"requested bucket {bucket} is absent from the bundle")

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("bundle manifest has no artifacts")
    artifacts: dict[str, dict[str, object]] = {}
    for raw_record in raw_artifacts:
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "path",
            "size",
            "sha256",
        }:
            raise ValueError("bundle artifact records must contain path/size/sha256")
        relative = raw_record.get("path")
        if not isinstance(relative, str) or relative == "manifest.json":
            raise ValueError("bundle artifact path is invalid")
        _safe_bundle_path(root, relative, where="artifact")
        if relative in artifacts:
            raise ValueError(f"duplicate bundle artifact path: {relative}")
        size = raw_record.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"artifact {relative} has invalid size")
        digest = _require_sha256(raw_record.get("sha256"), where=relative)
        artifacts[relative] = {"path": relative, "size": size, "sha256": digest}
    if int(manifest.get("artifact_count", -1)) != len(artifacts):
        raise ValueError("bundle artifact_count does not match artifact list")
    if int(manifest.get("artifact_bytes", -1)) != sum(
        int(record["size"]) for record in artifacts.values()
    ):
        raise ValueError("bundle artifact_bytes does not match artifact list")
    artifact_set_sha256 = _artifact_set_sha256(artifacts.values())
    if manifest.get("artifact_set_sha256") != artifact_set_sha256:
        raise ValueError("bundle artifact_set_sha256 does not match artifact list")
    if not expected_bundle_id.endswith(artifact_set_sha256[:16]):
        raise ValueError("expected bundle_id is not bound to the artifact set")
    _validate_source_composition_descriptor(manifest, artifacts)
    return artifacts


def _validate_source_composition_descriptor(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
) -> None:
    source_snapshot = _require_mapping(
        manifest.get("source_snapshot"), where="source_snapshot"
    )
    descriptor = _require_mapping(
        source_snapshot.get("source_composition"),
        where="source_snapshot.source_composition",
    )
    expected_fields = {
        "schema",
        "receipt",
        "plan",
        "dedup_receipt",
        "dedup_verifier",
        "runs",
    }
    if (
        set(descriptor) != expected_fields
        or descriptor.get("schema") != _SOURCE_COMPOSITION_SCHEMA
    ):
        raise ValueError("source composition descriptor fields/schema drifted")

    bound_paths: set[str] = set()

    def validate_artifact(
        raw_binding: object,
        *,
        where: str,
        with_size: bool,
    ) -> None:
        binding = _require_mapping(raw_binding, where=where)
        required = {"path", "sha256"}
        if with_size:
            required.add("size_bytes")
        if set(binding) != required:
            raise ValueError(f"{where} artifact binding fields drifted")
        relative = _require_relative_string(binding.get("path"), where=f"{where}.path")
        if not relative.startswith("provenance/source_composition/"):
            raise ValueError(f"{where} escapes source composition provenance")
        if relative in bound_paths:
            raise ValueError(f"duplicate source composition artifact: {relative}")
        bound_paths.add(relative)
        digest = _require_sha256(binding.get("sha256"), where=f"{where}.sha256")
        artifact = _require_artifact(artifacts, relative, where=where)
        if artifact["sha256"] != digest:
            raise ValueError(f"{where} SHA-256 is not artifact-bound")
        if with_size:
            size = binding.get("size_bytes")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or artifact["size"] != size
            ):
                raise ValueError(f"{where} size is not artifact-bound")

    for name in ("receipt", "plan", "dedup_receipt", "dedup_verifier"):
        validate_artifact(
            descriptor[name],
            where=f"source composition {name}",
            with_size=False,
        )
    raw_runs = descriptor.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("source composition descriptor has no runs")
    run_ids: set[str] = set()
    for raw_run in raw_runs:
        run = _require_mapping(raw_run, where="source composition run")
        if set(run) != {"run_id", "artifacts"}:
            raise ValueError("source composition run descriptor fields drifted")
        run_id = run.get("run_id")
        if (
            not isinstance(run_id, str)
            or not _RUN_ID_RE.fullmatch(run_id)
            or run_id in run_ids
        ):
            raise ValueError("source composition run_id is invalid or duplicated")
        run_ids.add(run_id)
        run_artifacts = _require_mapping(
            run.get("artifacts"), where=f"source composition run {run_id} artifacts"
        )
        if set(run_artifacts) != _SOURCE_COMPOSITION_RUN_ARTIFACTS:
            raise ValueError(
                f"source composition run {run_id} artifact inventory drifted"
            )
        for name, binding in run_artifacts.items():
            validate_artifact(
                binding,
                where=f"source composition run {run_id}/{name}",
                with_size=True,
            )
    composition_artifacts = {
        relative
        for relative in artifacts
        if relative.startswith("provenance/source_composition/")
    }
    if composition_artifacts != bound_paths:
        raise ValueError(
            "source composition provenance contains unbound artifacts: "
            f"missing={sorted(bound_paths - composition_artifacts)} "
            f"extra={sorted(composition_artifacts - bound_paths)}"
        )


def _validate_source_producer(
    raw_producer: object,
    *,
    where: str,
) -> Mapping[str, Any]:
    producer = _require_mapping(raw_producer, where=where)
    if set(producer) != {"cppmega", "clang_indexer"}:
        raise ValueError(f"{where} component set drifted")
    cppmega = _require_mapping(producer.get("cppmega"), where=f"{where} cppmega")
    indexer = _require_mapping(
        producer.get("clang_indexer"), where=f"{where} clang indexer"
    )
    if (
        set(cppmega) != {"commit", "tree_sha256"}
        or not isinstance(cppmega.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", cppmega["commit"]) is None
    ):
        raise ValueError(f"{where} cppmega identity drifted")
    _require_sha256(cppmega.get("tree_sha256"), where=f"{where} cppmega tree")
    if set(indexer) != {"source_sha256", "dependency_closure_sha256"}:
        raise ValueError(f"{where} clang indexer fields drifted")
    _require_sha256(indexer.get("source_sha256"), where=f"{where} indexer source")
    _require_sha256(
        indexer.get("dependency_closure_sha256"),
        where=f"{where} indexer dependency closure",
    )
    return producer


def _validate_source_composition(
    root: Path,
    manifest: Mapping[str, Any],
    _artifacts: Mapping[str, Mapping[str, object]],
) -> _SourceCompositionValidation:
    descriptor = _require_mapping(
        _require_mapping(manifest.get("source_snapshot"), where="source_snapshot").get(
            "source_composition"
        ),
        where="source_snapshot.source_composition",
    )

    def load_json(
        raw_binding: object,
        *,
        where: str,
        max_bytes: int,
    ) -> tuple[bytes, dict[str, Any]]:
        binding = _require_mapping(raw_binding, where=f"{where} binding")
        relative = _require_relative_string(binding.get("path"), where=f"{where} path")
        artifact = _require_artifact(_artifacts, relative, where=where)
        if int(artifact["size"]) > max_bytes:
            raise ValueError(f"{where} exceeds the production size limit")
        raw, payload = _load_json_object(
            _safe_bundle_path(root, relative, where=where),
            where=where,
        )
        if len(raw) > max_bytes:
            raise ValueError(f"{where} exceeds the production size limit")
        if hashlib.sha256(raw).hexdigest() != binding.get("sha256"):
            raise ValueError(f"{where} payload SHA-256 drifted")
        return raw, payload

    receipt_raw, receipt = load_json(
        descriptor.get("receipt"),
        where="source composition receipt",
        max_bytes=4 * 1024 * 1024,
    )
    expected_receipt_fields = {
        "schema",
        "status",
        "plan_sha256",
        "buckets",
        "archive",
        "dedup",
        "runs",
        "source_producers",
        "source_producer_set_sha256",
        "coverage",
    }
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("schema") != _SOURCE_COMPOSITION_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("buckets") != manifest.get("buckets")
    ):
        raise ValueError("source composition receipt contract drifted")
    plan_binding = _require_mapping(
        descriptor.get("plan"), where="source composition plan binding"
    )
    if receipt.get("plan_sha256") != plan_binding.get("sha256"):
        raise ValueError("source composition plan binding drifted")

    dedup_raw, dedup_receipt = load_json(
        descriptor.get("dedup_receipt"),
        where="global dedup receipt",
        max_bytes=4 * 1024 * 1024,
    )
    portable_dedup = dict(dedup_receipt)
    portable_database = _require_mapping(
        portable_dedup.get("database"), where="global dedup database"
    )
    portable_database = dict(portable_database)
    portable_database.pop("path", None)
    portable_dedup["database"] = portable_database
    dedup_receipt_sha256 = hashlib.sha256(dedup_raw).hexdigest()
    portable_dedup["receipt_sha256"] = dedup_receipt_sha256
    if receipt.get("dedup") != portable_dedup:
        raise ValueError("source composition portable dedup binding drifted")
    if set(dedup_receipt) != {
        "schema",
        "status",
        "created_at",
        "database",
        "checkpoint",
        "integrity_check",
        "sqlite_schema_sha256",
        "logical_hash_algorithm",
        "logical_sha256",
        "tables",
        "policy",
        "verifier",
    }:
        raise ValueError("global dedup receipt fields drifted")
    database = _require_mapping(
        dedup_receipt.get("database"), where="global dedup database"
    )
    database_size = database.get("size_bytes")
    if (
        set(database) != {"path", "size_bytes", "sha256"}
        or not isinstance(database.get("path"), str)
        or not database["path"]
        or isinstance(database_size, bool)
        or not isinstance(database_size, int)
        or database_size < 1
    ):
        raise ValueError("global dedup database binding drifted")
    _require_sha256(database.get("sha256"), where="global dedup database")
    _require_sha256(
        dedup_receipt.get("sqlite_schema_sha256"),
        where="global dedup SQLite schema",
    )
    if (
        dedup_receipt.get("schema") != _GLOBAL_DEDUP_RECEIPT_SCHEMA
        or dedup_receipt.get("status") != "verified"
        or not isinstance(dedup_receipt.get("created_at"), str)
        or not dedup_receipt["created_at"]
        or dedup_receipt.get("integrity_check") != "ok"
        or dedup_receipt.get("logical_hash_algorithm")
        != "cppmega_sqlite_rows_lenprefixed_v1"
        or dedup_receipt.get("checkpoint")
        != {
            "mode": "TRUNCATE",
            "busy": 0,
            "log_frames": 0,
            "checkpointed_frames": 0,
            "wal_size_bytes": 0,
        }
    ):
        raise ValueError("global dedup receipt is not a completed production snapshot")
    dedup_logical_sha256 = _require_sha256(
        dedup_receipt.get("logical_sha256"), where="global dedup logical SHA-256"
    )
    tables = _require_mapping(dedup_receipt.get("tables"), where="global dedup tables")
    if set(tables) != _GLOBAL_DEDUP_TABLES:
        raise ValueError("global dedup table inventory drifted")
    for name, raw_table in tables.items():
        table = _require_mapping(raw_table, where=f"global dedup {name}")
        rows = table.get("rows")
        if (
            set(table) != {"rows", "logical_sha256"}
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 0
        ):
            raise ValueError(f"global dedup {name} receipt fields drifted")
        _require_sha256(
            table.get("logical_sha256"),
            where=f"global dedup {name} logical SHA-256",
        )
    for name in (
        "dedup_stages",
        "exact_stage",
        "minhash_stage",
        "lsh_stage",
        "chunk_claims_stage",
    ):
        table = _require_mapping(tables.get(name), where=f"global dedup {name}")
        if table.get("rows") != 0:
            raise ValueError(f"global dedup {name} contains unpromoted rows")
    for name in ("exact", "lsh", "minhash", "dedup_meta", "chunk_claims"):
        table = _require_mapping(tables.get(name), where=f"global dedup {name}")
        rows = table.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError(f"global dedup production table {name} is empty")
    policy = _require_mapping(dedup_receipt.get("policy"), where="global dedup policy")
    if (
        set(policy) != {"exact", "chunk", "near"}
        or policy.get("exact") != "sha1_token_ids_v1"
        or policy.get("chunk") != "tokenized_chunk_claims_v1"
        or policy.get("near")
        != {
            "enabled": True,
            "threshold": 0.7,
            "num_perm": 256,
            "shingle_k": 5,
        }
    ):
        raise ValueError("global dedup production exact+near policy drifted")
    verifier = _require_mapping(
        dedup_receipt.get("verifier"), where="global dedup verifier"
    )
    verifier_binding = _require_mapping(
        descriptor.get("dedup_verifier"),
        where="source composition dedup verifier binding",
    )
    if (
        set(verifier) != {"repository_identity", "script", "script_sha256"}
        or verifier.get("repository_identity") != "cppmega"
        or verifier.get("script") != "scripts/data/verify_global_dedup_store.py"
        or verifier.get("script_sha256") != verifier_binding.get("sha256")
    ):
        raise ValueError("global dedup verifier artifact binding drifted")

    archive = _require_mapping(receipt.get("archive"), where="source archive")
    coverage = _require_mapping(receipt.get("coverage"), where="source coverage")
    if set(archive) != {
        "repository_count",
        "repository_names_sha256",
        "input_binding_sha256",
        "archive_identity_sha256",
    }:
        raise ValueError("source composition archive binding fields drifted")
    for name in (
        "repository_names_sha256",
        "input_binding_sha256",
        "archive_identity_sha256",
    ):
        _require_sha256(archive.get(name), where=f"source archive {name}")
    if set(coverage) != {
        "expected_repositories",
        "code_success_repositories",
        "commit_success_repositories",
        "failed_repositories_observed",
        "failed_units_observed",
        "unresolved_failed_units",
        "repository_set_sha256",
        "allowlist_counts",
    }:
        raise ValueError("source composition coverage fields drifted")
    _require_sha256(
        coverage.get("repository_set_sha256"),
        where="source composition repository set",
    )
    for name in ("failed_repositories_observed", "failed_units_observed"):
        value = coverage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"source composition {name} is invalid")
    expected_repositories = coverage.get("expected_repositories")
    if (
        isinstance(expected_repositories, bool)
        or not isinstance(expected_repositories, int)
        or expected_repositories < 1
        or archive.get("repository_count") != expected_repositories
        or archive.get("repository_names_sha256")
        != coverage.get("repository_set_sha256")
        or coverage.get("code_success_repositories") != expected_repositories
        or coverage.get("commit_success_repositories") != expected_repositories
        or coverage.get("unresolved_failed_units") != 0
    ):
        raise ValueError("source composition does not prove full repository coverage")
    allowlist_counts = _require_mapping(
        coverage.get("allowlist_counts"), where="source composition allowlist"
    )
    expected_allowlist = {
        f"{kind}/{bundle_bucket}"
        for kind in ("code", "commits")
        for bundle_bucket in manifest["buckets"]
    }
    if set(allowlist_counts) != expected_allowlist or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in allowlist_counts.values()
    ):
        raise ValueError("source composition allowlist coverage drifted")

    source_producers = receipt.get("source_producers")
    if not isinstance(source_producers, list) or not source_producers:
        raise ValueError("source composition has no source producers")
    producer_digests: set[str] = set()
    ordered_producer_digests: list[str] = []
    for raw_producer in source_producers:
        producer = _validate_source_producer(
            raw_producer,
            where="source producer",
        )
        digest = _canonical_sha256(producer)
        if digest in producer_digests:
            raise ValueError("source composition contains a duplicate producer")
        producer_digests.add(digest)
        ordered_producer_digests.append(digest)
    if ordered_producer_digests != sorted(ordered_producer_digests):
        raise ValueError("source composition producer order drifted")
    producer_set_sha256 = _canonical_sha256(source_producers)
    if receipt.get("source_producer_set_sha256") != producer_set_sha256:
        raise ValueError("source producer set SHA-256 drifted")

    receipt_runs = receipt.get("runs")
    descriptor_runs = descriptor.get("runs")
    if (
        not isinstance(receipt_runs, list)
        or not isinstance(descriptor_runs, list)
        or len(receipt_runs) != len(descriptor_runs)
    ):
        raise ValueError("source composition run set drifted")
    descriptor_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_run in descriptor_runs:
        descriptor_run = _require_mapping(raw_run, where="source run descriptor")
        descriptor_by_id[str(descriptor_run["run_id"])] = descriptor_run
    input_file_keys = {
        "archive_sha256_receipt": "archive_sha256_receipt",
        "archive_inventory": "archive_inventory_receipt",
        "repo_list": "repo_list",
        "source_quarantine_manifest": "source_quarantine_manifest",
        "tokenizer": "tokenizer",
    }
    receipt_run_ids: set[str] = set()
    full_code_receipts = 0
    full_commit_receipts = 0
    for raw_run in receipt_runs:
        run = _require_mapping(raw_run, where="source composition receipt run")
        if set(run) != {
            "run_id",
            "launch",
            "exit",
            "manifest",
            "streams",
            "selected_repositories",
            "terminal_repositories",
            "terminal_repository_set_sha256",
            "input_artifacts",
            "code_revision",
            "allowlist_counts",
        }:
            raise ValueError("source composition receipt run fields drifted")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or run_id not in descriptor_by_id:
            raise ValueError("source composition receipt run is not staged")
        if run_id in receipt_run_ids:
            raise ValueError("source composition receipt run_id is duplicated")
        receipt_run_ids.add(run_id)
        staged = _require_mapping(
            descriptor_by_id[run_id].get("artifacts"),
            where=f"source composition run {run_id} artifacts",
        )
        launch = _require_mapping(run.get("launch"), where=f"{run_id} launch")
        exit_receipt = _require_mapping(run.get("exit"), where=f"{run_id} exit")
        run_manifest = _require_mapping(run.get("manifest"), where=f"{run_id} manifest")
        inputs = _require_mapping(
            run.get("input_artifacts"), where=f"{run_id} input artifacts"
        )
        if set(inputs) != {
            "archive_sha256_receipt",
            "archive_inventory_receipt",
            "repo_list",
            "source_quarantine_manifest",
            "tokenizer",
        }:
            raise ValueError(f"source composition run {run_id} input fields drifted")
        if set(launch) != {"schema", "sha256"}:
            raise ValueError(f"source composition run {run_id} launch fields drifted")
        if set(exit_receipt) != {"schema", "sha256", "exit_code"}:
            raise ValueError(f"source composition run {run_id} exit fields drifted")
        launch_schema = launch.get("schema")
        if (
            launch_schema not in _SOURCE_EXIT_SCHEMA_BY_LAUNCH
            or exit_receipt.get("schema")
            != _SOURCE_EXIT_SCHEMA_BY_LAUNCH[launch_schema]
        ):
            raise ValueError(
                f"source composition run {run_id} supervisor schemas drifted"
            )
        if set(run_manifest) != {
            "sha256",
            "done_units",
            "failed_units",
            "done_unit_set_sha256",
            "failed_unit_set_sha256",
        }:
            raise ValueError(f"source composition run {run_id} manifest fields drifted")
        for name in ("done_units", "failed_units"):
            value = run_manifest.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"source composition run {run_id} {name} is invalid")
        for name in ("done_unit_set_sha256", "failed_unit_set_sha256"):
            _require_sha256(
                run_manifest.get(name),
                where=f"source composition run {run_id} {name}",
            )
        if exit_receipt.get("exit_code") != 0:
            raise ValueError(f"source composition run {run_id} did not exit cleanly")
        if run.get("streams") not in {"code", "commits", "both"}:
            raise ValueError(f"source composition run {run_id} streams drifted")
        streams = str(run["streams"])
        for name in (
            "selected_repositories",
            "terminal_repositories",
        ):
            values = run.get(name)
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value for value in values)
                or values != sorted(set(values))
            ):
                raise ValueError(f"source composition run {run_id} {name} is invalid")
        _require_sha256(
            run.get("terminal_repository_set_sha256"),
            where=f"source composition run {run_id} terminal repository set",
        )
        terminal_repositories = run["terminal_repositories"]
        if run.get("terminal_repository_set_sha256") != _canonical_sha256(
            terminal_repositories
        ):
            raise ValueError(
                f"source composition run {run_id} terminal repository digest drifted"
            )
        selected_repositories = run["selected_repositories"]
        if launch_schema == _SOURCE_FULL_LAUNCH_SCHEMA and selected_repositories:
            raise ValueError(
                f"source composition full run {run_id} selects repositories"
            )
        if (
            launch_schema == _SOURCE_TARGETED_LAUNCH_SCHEMA
            and not selected_repositories
        ):
            raise ValueError(
                f"source composition targeted run {run_id} has no selection"
            )
        if launch_schema == _SOURCE_FULL_LAUNCH_SCHEMA:
            if len(terminal_repositories) != expected_repositories or run.get(
                "terminal_repository_set_sha256"
            ) != coverage.get("repository_set_sha256"):
                raise ValueError(
                    f"source composition full run {run_id} archive coverage drifted"
                )
            if streams in {"code", "both"}:
                full_code_receipts += 1
            if streams in {"commits", "both"}:
                full_commit_receipts += 1
        run_allowlist = _require_mapping(
            run.get("allowlist_counts"),
            where=f"source composition run {run_id} allowlist",
        )
        if not set(run_allowlist).issubset(expected_allowlist) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in run_allowlist.values()
        ):
            raise ValueError(f"source composition run {run_id} allowlist drifted")
        run_producer = _validate_source_producer(
            run.get("code_revision"),
            where=f"source composition run {run_id} code revision",
        )
        if _canonical_sha256(run_producer) not in producer_digests:
            raise ValueError(
                f"source composition run {run_id} producer is absent from producer set"
            )
        expected_hashes = {
            "launch": launch.get("sha256"),
            "exit": exit_receipt.get("sha256"),
            "manifest": run_manifest.get("sha256"),
            **{
                file_key: inputs.get(input_key)
                for file_key, input_key in input_file_keys.items()
            },
        }
        for name, expected_sha256 in expected_hashes.items():
            _require_sha256(
                expected_sha256, where=f"source composition run {run_id}/{name}"
            )
            staged_binding = _require_mapping(
                staged.get(name), where=f"staged source run {run_id}/{name}"
            )
            if staged_binding.get("sha256") != expected_sha256:
                raise ValueError(
                    f"source composition run artifact binding drifted: {run_id}/{name}"
                )
    if receipt_run_ids != set(descriptor_by_id):
        raise ValueError("source composition receipt omits a staged run")
    if full_code_receipts < 1 or full_commit_receipts < 1:
        raise ValueError("source composition lacks full code and commit receipts")

    return _SourceCompositionValidation(
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        producer_set_sha256=producer_set_sha256,
        dedup_receipt_sha256=dedup_receipt_sha256,
        dedup_logical_sha256=dedup_logical_sha256,
    )


def _reject_symlinks_and_unlisted_relevant_files(
    root: Path,
    *,
    listed: set[str],
    restore_receipt_path: Path,
) -> None:
    allowed_unlisted = {
        "manifest.json",
        restore_receipt_path.relative_to(root).as_posix(),
    }
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError(f"production bundle contains a symlink: {path}")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError(f"production bundle contains a symlink: {path}")
            relative = path.relative_to(root).as_posix()
            if (
                path.suffix.lower() in _RELEVANT_UNLISTED_SUFFIXES
                and relative not in listed
                and relative not in allowed_unlisted
            ):
                raise ValueError(
                    f"production bundle contains unlisted relevant artifact: {relative}"
                )


def _verify_all_artifacts(
    root: Path,
    *,
    artifacts: Mapping[str, Mapping[str, object]],
    hash_jobs: int,
) -> dict[str, tuple[int, int, int, int, int]]:
    def verify(
        item: tuple[str, Mapping[str, object]],
    ) -> tuple[str, tuple[int, int, int, int, int]]:
        relative, record = item
        path = _safe_bundle_path(root, relative, where="artifact")
        digest, signature = _stable_file_sha256(path)
        if signature[2] != int(record["size"]):
            raise ValueError(
                f"artifact size mismatch for {relative}: {signature[2]} != {record['size']}"
            )
        if digest != record["sha256"]:
            raise ValueError(f"artifact sha256 mismatch for {relative}")
        return relative, signature

    with ThreadPoolExecutor(max_workers=hash_jobs) as pool:
        verified = pool.map(verify, sorted(artifacts.items()))
        return dict(verified)


def _validate_tokenizer(
    root: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
) -> str:
    descriptor = _require_mapping(manifest.get("tokenizer"), where="tokenizer")
    if descriptor.get("contract") != _BUNDLE_TOKENIZER_CONTRACT:
        raise ValueError("bundle tokenizer descriptor contract mismatch")
    if int(descriptor.get("vocab_size", -1)) != _EXPECTED_VOCAB_SIZE:
        raise ValueError("bundle tokenizer descriptor vocab size mismatch")
    tokenizer_relative = descriptor.get("path")
    if not isinstance(tokenizer_relative, str):
        raise ValueError("bundle tokenizer path must be a string")
    tokenizer_root = _safe_bundle_path(
        root, tokenizer_relative, where="tokenizer", require_file=False
    )
    if not tokenizer_root.is_dir():
        raise FileNotFoundError(tokenizer_root)
    records = descriptor.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("bundle tokenizer descriptor has no files")
    canonical = _canonical_artifact_records(records)
    tokenizer_digest = _artifact_set_sha256(canonical)
    if descriptor.get("artifact_set_sha256") != tokenizer_digest:
        raise ValueError("bundle tokenizer artifact-set SHA-256 mismatch")
    referenced: set[str] = set()
    for record in canonical:
        relative = str(record["path"])
        if artifacts.get(relative) != record:
            raise ValueError(f"tokenizer file is not artifact-bound: {relative}")
        path = _safe_bundle_path(root, relative, where="tokenizer artifact")
        if tokenizer_root not in path.parents:
            raise ValueError(f"tokenizer artifact escapes tokenizer root: {relative}")
        referenced.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in tokenizer_root.iterdir()
        if path.is_file()
    }
    if actual != referenced:
        raise ValueError("tokenizer directory differs from its descriptor allowlist")
    required_contract = f"{tokenizer_relative}/tokenizer_contract_v1.json"
    contract_record = artifacts.get(required_contract)
    if (
        contract_record is None
        or contract_record["sha256"] != TOKENIZER_CONTRACT_SHA256
    ):
        raise ValueError(
            "bundle tokenizer contract does not match the local frozen hash"
        )
    return tokenizer_digest


def _validate_data_contracts(
    root: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
) -> None:
    descriptors = _require_mapping(
        manifest.get("data_contracts"), where="data_contracts"
    )
    expected = {
        "domain_schema": DOMAIN_SCHEMA_SHA256,
        "tokenizer_contract": TOKENIZER_CONTRACT_SHA256,
    }
    if set(descriptors) != set(expected):
        raise ValueError("bundle data_contracts descriptor is incomplete")
    for name, expected_sha256 in expected.items():
        descriptor = _require_mapping(descriptors[name], where=f"data_contracts.{name}")
        if set(descriptor) != {"path", "size", "sha256"}:
            raise ValueError(f"bundle data contract {name} descriptor is invalid")
        relative = descriptor.get("path")
        if not isinstance(relative, str):
            raise ValueError(f"bundle data contract {name} path is invalid")
        _safe_bundle_path(root, relative, where=f"data contract {name}")
        expected_record = {
            "path": relative,
            "size": descriptor.get("size"),
            "sha256": descriptor.get("sha256"),
        }
        if artifacts.get(relative) != expected_record:
            raise ValueError(f"bundle data contract {name} is not artifact-bound")
        if descriptor.get("sha256") != expected_sha256:
            raise ValueError(f"bundle data contract {name} has a stale frozen hash")


def _validate_source_manifests(
    root: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
) -> tuple[str, str, dict[int, list[dict[str, Any]]]]:
    descriptor = _require_mapping(
        manifest.get("source_snapshot"), where="source_snapshot"
    )
    source_relative = descriptor.get("manifest")
    repaired_relative = descriptor.get("repaired_manifest")
    if not isinstance(source_relative, str) or not isinstance(repaired_relative, str):
        raise ValueError("source snapshot manifest paths must be strings")
    source_record = _require_artifact(
        artifacts, source_relative, where="source manifest"
    )
    repaired_record = _require_artifact(
        artifacts, repaired_relative, where="repaired source manifest"
    )
    _, source = _load_json_object(
        _safe_bundle_path(root, source_relative, where="source manifest"),
        where="source manifest",
    )
    _, repaired = _load_json_object(
        _safe_bundle_path(root, repaired_relative, where="repaired source manifest"),
        where="repaired source manifest",
    )
    if source.get("schema") != "cppmega_parquet_snapshot_v1":
        raise ValueError("source snapshot manifest schema is unsupported")
    if repaired.get("schema") != "cppmega_repaired_parquet_snapshot_v1":
        raise ValueError("repaired source snapshot manifest schema is unsupported")
    source_files = source.get("files")
    repaired_files = repaired.get("files")
    if not isinstance(source_files, list) or not isinstance(repaired_files, list):
        raise ValueError("source snapshot manifests require files lists")
    expected_count = int(descriptor.get("file_count", -1))
    if (
        expected_count < 1
        or int(source.get("file_count", -1)) != expected_count
        or int(repaired.get("file_count", -1)) != expected_count
        or len(source_files) != expected_count
        or len(repaired_files) != expected_count
    ):
        raise ValueError("source snapshot file counts disagree")

    source_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for raw_record in source_files:
        record = dict(_require_mapping(raw_record, where="source snapshot file"))
        key = _source_record_key(record, where="source snapshot file")
        if key in source_by_key:
            raise ValueError(f"duplicate source snapshot file: {key}")
        _require_positive_int(record.get("size"), where="source snapshot size")
        _require_sha256(record.get("sha256"), where="source snapshot sha256")
        source_by_key[key] = record

    repaired_by_bucket: dict[int, list[dict[str, Any]]] = {}
    repaired_keys: set[tuple[str, int, str]] = set()
    for raw_record in repaired_files:
        record = dict(_require_mapping(raw_record, where="repaired snapshot file"))
        key = _source_record_key(record, where="repaired snapshot file")
        if key in repaired_keys or key not in source_by_key:
            raise ValueError(f"repaired source snapshot file identity drifted: {key}")
        repaired_keys.add(key)
        source_entry = source_by_key[key]
        _require_positive_int(record.get("size"), where="repaired snapshot size")
        if record.get("source_sha256") != source_entry.get("sha256"):
            raise ValueError(
                "repaired snapshot source hash does not match source manifest"
            )
        _require_sha256(record.get("snapshot_sha256"), where="repaired snapshot sha256")
        repaired_by_bucket.setdefault(key[1], []).append(record)
    if repaired_keys != set(source_by_key):
        raise ValueError("repaired source snapshot omits source files")
    return (
        str(source_record["sha256"]),
        str(repaired_record["sha256"]),
        repaired_by_bucket,
    )


def _validate_ci_production_acquisition(
    root: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
) -> _CIValidation:
    """Bind MLX training ingress to the exhaustive cppmega CI-v4 proof chain."""

    source_snapshot = _require_mapping(
        manifest.get("source_snapshot"), where="source_snapshot"
    )
    descriptor = _require_mapping(
        source_snapshot.get("ci_manifest"), where="source_snapshot.ci_manifest"
    )
    expected_descriptor_fields = {
        "path",
        "sha256",
        "source_inventory_sha256",
        "input_docs",
        "fragments",
        "valid_tokens",
        "cross_boundary_chunk_edges",
        "cross_boundary_token_edges",
    }
    if set(descriptor) != expected_descriptor_fields:
        raise ValueError("CI manifest descriptor fields drifted")
    relative = _require_relative_string(
        descriptor.get("path"), where="CI manifest path"
    )
    if relative != "provenance/ci_manifest.json":
        raise ValueError("CI manifest must use the canonical provenance path")
    manifest_record = _require_artifact(artifacts, relative, where="CI export receipt")
    manifest_sha256 = _require_sha256(
        descriptor.get("sha256"), where="CI manifest sha256"
    )
    if manifest_record["sha256"] != manifest_sha256:
        raise ValueError("CI manifest descriptor is not artifact-bound")
    raw, ci_manifest = _load_json_object(
        _safe_bundle_path(root, relative, where="CI manifest"),
        where="CI manifest",
    )
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("CI manifest bytes drifted from the descriptor")
    if (
        ci_manifest.get("schema") != _CI_PRODUCTION_EXPORT_SCHEMA
        or ci_manifest.get("status") != "complete"
        or ci_manifest.get("completion_mode") != _CI_COMPLETION_MODE
        or ci_manifest.get("production_complete") is not True
    ):
        raise ValueError(
            "production bundle requires an inventory-exhaustive CI export v3"
        )
    _require_sha256(
        ci_manifest.get("exporter_script_sha256"),
        where="CI exporter script sha256",
    )

    validation = _require_mapping(
        ci_manifest.get("validation"), where="CI export validation"
    )
    required_validation = (
        "all_passed",
        "fixed_widths",
        "zero_overflow",
        "payload_conserved",
        "payload_identity_and_order_verified",
        "post_normalize_pack_sidecars_and_edges_verified",
    )
    if any(validation.get(name) is not True for name in required_validation):
        raise ValueError("CI export validation is not green")

    buckets = [int(value) for value in manifest["buckets"]]
    case5 = _require_mapping(
        ci_manifest.get("case5_contract"), where="CI CASE5 contract"
    )
    if (
        case5.get("buckets") != buckets
        or case5.get("overflow_rows") != 0
        or case5.get("parquet_shard_max_rows") != 512
        or case5.get("parquet_layout") != "bucket-first-split-in-filename-v1"
    ):
        raise ValueError("CI CASE5 bucket contract drifted")

    input_store = _require_mapping(
        ci_manifest.get("input_store"), where="CI input store"
    )
    input_fetch_state = _require_mapping(
        ci_manifest.get("input_fetch_state"), where="CI fetch state"
    )
    fetch_artifact = _require_mapping(
        input_fetch_state.get("artifact"), where="CI fetch-state artifact"
    )
    if (
        input_store.get("schema") != "cppmega_ci_content_store_v1"
        or input_store.get("receipt_schema") != "cppmega_ci_content_store_receipt_v1"
        or input_store.get("verified_before_export") is not True
        or input_store.get("unchanged_after_export") is not True
        or input_fetch_state.get("schema") != _CI_FETCH_STATE_SCHEMA
    ):
        raise ValueError("CI immutable input store/fetch contract drifted")
    for name in (
        "receipt_sha256",
        "policy_sha256",
        "sqlite_schema_sha256",
        "logical_content_set_sha256",
        "logical_token_sequence_set_sha256",
        "occurrence_set_sha256",
        "sqlite_logical_sha256",
    ):
        _require_sha256(input_store.get(name), where=f"CI input store {name}")
    if (
        not isinstance(input_store.get("pack_hashes"), list)
        or not input_store["pack_hashes"]
    ):
        raise ValueError("CI input store pack inventory is empty")
    _require_sha256(
        fetch_artifact.get("sha256"), where="CI fetch-state artifact sha256"
    )
    for name in (
        "sqlite_schema_sha256",
        "sqlite_logical_sha256",
        "sidecar_set_sha256",
    ):
        _require_sha256(input_fetch_state.get(name), where=f"CI fetch state {name}")
    fetch_settings = _require_mapping(
        input_fetch_state.get("settings"), where="CI fetch settings"
    )
    for name in (
        "fetcher_script_sha256",
        "parser_script_sha256",
        "content_store_script_sha256",
    ):
        _require_sha256(fetch_settings.get(name), where=f"CI fetch setting {name}")

    eligibility = _require_mapping(
        ci_manifest.get("eligibility"), where="CI eligibility"
    )
    eligible = _require_mapping(eligibility.get("eligible"), where="CI eligible corpus")
    conservation = _require_mapping(
        eligibility.get("conservation"), where="CI eligibility conservation"
    )
    counts = _require_mapping(ci_manifest.get("counts"), where="CI counts")
    representatives = _require_mapping(
        ci_manifest.get("representatives"), where="CI representatives"
    )
    graph = _require_mapping(
        ci_manifest.get("graph_accounting"), where="CI graph accounting"
    )

    def require_int(value: object, *, where: str, positive: bool = False) -> int:
        minimum = 1 if positive else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{where} must be an integer >= {minimum}")
        return value

    target_tokens = require_int(
        eligibility.get("target_exact_unique_payload_tokens"),
        where="CI target exact unique payload tokens",
    )
    eligible_tokens = require_int(
        eligible.get("exact_unique_payload_tokens"),
        where="CI eligible exact unique payload tokens",
        positive=True,
    )
    representative_count = require_int(
        representatives.get("count"), where="CI representative count", positive=True
    )
    fragment_count = require_int(
        counts.get("fragments"), where="CI fragment count", positive=True
    )
    valid_tokens = require_int(
        counts.get("valid_tokens"), where="CI valid token count", positive=True
    )
    cross_chunk_edges = require_int(
        graph.get("cross_chunk_outbound_edges_dropped"),
        where="CI cross-chunk edge count",
    )
    cross_fragment_edges = require_int(
        graph.get("cross_fragment_edges_dropped"),
        where="CI cross-fragment edge count",
    )
    if (
        eligibility.get("target_met") is not True
        or conservation.get("exact_unique_payload_tokens") is not True
        or conservation.get("unique_token_sequences") is not True
        or target_tokens < _CI_MIN_EXACT_UNIQUE_PAYLOAD_TOKENS
        or eligible_tokens < target_tokens
        or counts.get("payload_tokens") != eligible_tokens
        or counts.get("representatives") != representative_count
        or eligible.get("unique_token_sequences") != representative_count
        or descriptor.get("input_docs") != representative_count
        or descriptor.get("fragments") != fragment_count
        or descriptor.get("valid_tokens") != valid_tokens
        or descriptor.get("cross_boundary_chunk_edges") != cross_chunk_edges
        or descriptor.get("cross_boundary_token_edges") != cross_fragment_edges
    ):
        raise ValueError("CI exact-token or aggregate accounting drifted")

    representative_ledger_sha256 = _require_sha256(
        representatives.get("ledger_sha256"),
        where="CI representative ledger sha256",
    )
    source_binding = {
        "fetch_state_sqlite_logical_sha256": input_fetch_state["sqlite_logical_sha256"],
        "fetch_state_sidecar_set_sha256": input_fetch_state["sidecar_set_sha256"],
        "representative_ledger_sha256": representative_ledger_sha256,
    }
    source_inventory_sha256 = _canonical_sha256(source_binding)
    if descriptor.get("source_inventory_sha256") != source_inventory_sha256:
        raise ValueError("CI source inventory binding drifted")

    acquisition = _require_mapping(
        ci_manifest.get("acquisition_provenance"),
        where="CI acquisition provenance",
    )
    if (
        acquisition.get("completion_mode") != _CI_COMPLETION_MODE
        or acquisition.get("production_complete") is not True
    ):
        raise ValueError("CI acquisition provenance is not production-complete")
    inventory = _require_mapping(
        acquisition.get("inventory"), where="CI acquisition inventory"
    )
    fetch = _require_mapping(acquisition.get("fetch"), where="CI acquisition fetch")
    store = _require_mapping(acquisition.get("store"), where="CI acquisition store")
    merge = _require_mapping(acquisition.get("merge"), where="CI acquisition merge")
    for where, value in (
        ("inventory sha256", inventory.get("sha256")),
        ("inventory logical sha256", inventory.get("logical_sha256")),
        ("inventory receipt sha256", inventory.get("receipt_sha256")),
        ("fetch state sha256", fetch.get("state_sha256")),
        ("fetch receipt sha256", fetch.get("receipt_sha256")),
        ("fetch attempt set sha256", fetch.get("attempt_set_sha256")),
        ("fetch terminal proof sha256", fetch.get("terminal_proof_sha256")),
        ("store receipt sha256", store.get("receipt_sha256")),
        ("merge receipt sha256", merge.get("receipt_sha256")),
    ):
        _require_sha256(value, where=f"CI acquisition {where}")
    if (
        fetch.get("state_sha256") != fetch_artifact.get("sha256")
        or store.get("receipt_sha256") != input_store.get("receipt_sha256")
        or merge.get("schema") != _CI_PRODUCTION_MERGE_SCHEMA
    ):
        raise ValueError("CI acquisition proof chain is not mutually bound")

    source_manifest_relative = _require_relative_string(
        source_snapshot.get("manifest"), where="source manifest path"
    )
    _, source_manifest = _load_json_object(
        _safe_bundle_path(root, source_manifest_relative, where="source manifest"),
        where="source manifest",
    )
    source_ci_artifacts: set[tuple[int, str, int, str]] = set()
    for raw_record in source_manifest.get("files", []):
        record = _require_mapping(raw_record, where="source manifest file")
        if record.get("kind") != "ci":
            continue
        snapshot = _require_relative_string(
            record.get("snapshot"), where="CI source snapshot path"
        )
        snapshot_parts = PurePosixPath(snapshot).parts
        bucket = _require_positive_int(record.get("bucket"), where="CI source bucket")
        size = _require_positive_int(record.get("size"), where="CI source size")
        digest = _require_sha256(record.get("sha256"), where="CI source sha256")
        if (
            len(snapshot_parts) != 3
            or snapshot_parts[0] != "ci"
            or snapshot_parts[1] != str(bucket)
        ):
            raise ValueError("CI source snapshot path is not canonical")
        source_ci_artifacts.add((bucket, snapshot_parts[2], size, digest))

    raw_ci_artifacts = ci_manifest.get("artifacts")
    if not isinstance(raw_ci_artifacts, list) or not raw_ci_artifacts:
        raise ValueError("CI export artifact inventory is empty")
    export_ci_artifacts: set[tuple[int, str, int, str]] = set()
    observed_ledgers: set[str] = set()
    observed_paths: set[str] = set()
    for raw_record in raw_ci_artifacts:
        record = _require_mapping(raw_record, where="CI export artifact")
        artifact_relative = _require_relative_string(
            record.get("path"), where="CI export artifact path"
        )
        if artifact_relative in observed_paths:
            raise ValueError("CI export artifact path is duplicated")
        observed_paths.add(artifact_relative)
        size = require_int(record.get("byte_size"), where="CI export artifact size")
        digest = _require_sha256(
            record.get("sha256"), where="CI export artifact sha256"
        )
        kind = record.get("kind")
        if kind == "case5_parquet":
            export_path = PurePosixPath(artifact_relative)
            bucket = _require_positive_int(
                record.get("bucket"), where="CI export bucket"
            )
            if (
                len(export_path.parts) != 2
                or export_path.parts[0] != str(bucket)
                or bucket not in buckets
            ):
                raise ValueError("CI export Parquet path is not canonical")
            export_ci_artifacts.add((bucket, export_path.name, size, digest))
            continue
        expected_name = (
            _CI_PROVENANCE_LEDGER_NAMES.get(kind) if isinstance(kind, str) else None
        )
        if (
            expected_name is None
            or artifact_relative != expected_name
            or kind in observed_ledgers
        ):
            raise ValueError("CI export provenance ledger inventory drifted")
        observed_ledgers.add(kind)
        staged_relative = f"provenance/ci_export/{artifact_relative}"
        staged = _require_artifact(
            artifacts, staged_relative, where=f"CI provenance ledger {kind}"
        )
        if staged["size"] != size or staged["sha256"] != digest:
            raise ValueError(f"CI provenance ledger {kind} is not artifact-bound")
    if observed_ledgers != set(_CI_PROVENANCE_LEDGER_NAMES):
        raise ValueError("CI export provenance ledger inventory is incomplete")
    if not export_ci_artifacts or export_ci_artifacts != source_ci_artifacts:
        raise ValueError("CI export Parquet inventory differs from source snapshot")

    return _CIValidation(
        manifest_sha256=manifest_sha256,
        source_inventory_sha256=source_inventory_sha256,
        acquisition_provenance_sha256=_canonical_sha256(acquisition),
        exact_unique_payload_tokens=eligible_tokens,
    )


def _validate_objectives(
    root: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
    *,
    buckets: list[int],
    repaired_by_bucket: Mapping[int, list[dict[str, Any]]],
) -> dict[int, _ObjectiveValidation]:
    objective_root = _require_mapping(
        manifest.get("objective_materialization"), where="objective_materialization"
    )
    if objective_root.get("schema") != _OBJECTIVE_BUCKETS_SCHEMA:
        raise ValueError("bundle objective materialization schema is unsupported")
    raw_descriptors = _require_mapping(
        objective_root.get("buckets"), where="objective_materialization.buckets"
    )
    if set(raw_descriptors) != {str(bucket) for bucket in buckets}:
        raise ValueError("objective descriptor buckets do not match bundle buckets")

    validated: dict[int, _ObjectiveValidation] = {}
    for bucket in buckets:
        descriptor = dict(
            _require_mapping(
                raw_descriptors[str(bucket)], where=f"objective bucket {bucket}"
            )
        )
        if set(descriptor) != _OBJECTIVE_DESCRIPTOR_FIELDS:
            raise ValueError(f"objective descriptor fields drifted for bucket {bucket}")
        if descriptor.get("artifact_schema") == _LEGACY_OBJECTIVE_ARTIFACT_SCHEMA:
            raise ValueError(
                f"objective artifact schema for bucket {bucket} is legacy; migration "
                "required: regenerate the objective artifact and bundle"
            )
        if descriptor.get("artifact_schema") != _OBJECTIVE_ARTIFACT_SCHEMA:
            raise ValueError(f"objective artifact schema drifted for bucket {bucket}")
        if descriptor.get("contract_schema") != _OBJECTIVE_CONTRACT_SCHEMA:
            raise ValueError(f"objective contract schema drifted for bucket {bucket}")
        for name in (
            "artifact_set_sha256",
            "artifact_file_sha256",
            "contract_sha256",
            "contract_file_sha256",
        ):
            _require_sha256(descriptor.get(name), where=f"objective {bucket}.{name}")

        contract_relative = _require_relative_string(
            descriptor.get("contract_path"), where=f"objective {bucket} contract path"
        )
        contract_record = _require_artifact(
            artifacts, contract_relative, where=f"objective {bucket} contract"
        )
        if contract_record["sha256"] != descriptor["contract_file_sha256"]:
            raise ValueError(
                f"objective contract file hash drifted for bucket {bucket}"
            )
        contract_path = _safe_bundle_path(
            root, contract_relative, where=f"objective {bucket} contract"
        )
        _, contract = _load_json_object(
            contract_path, where=f"objective {bucket} contract"
        )
        if contract.get("schema") != _OBJECTIVE_CONTRACT_SCHEMA:
            raise ValueError(
                f"objective contract payload schema drifted for bucket {bucket}"
            )
        contract_sha256 = _canonical_sha256(contract)
        if contract_sha256 != descriptor["contract_sha256"]:
            raise ValueError(
                f"objective contract payload hash drifted for bucket {bucket}"
            )
        totals = _require_mapping(
            contract.get("totals"), where=f"objective {bucket} totals"
        )
        _require_positive_int(
            totals.get("samples"), where=f"objective {bucket} samples"
        )
        materialization = _require_mapping(
            contract.get("materialization"), where=f"objective {bucket} materialization"
        )
        if materialization.get("format") != "shifted_lm_document_v1":
            raise ValueError(
                f"objective materialization format drifted for bucket {bucket}"
            )
        if (
            materialization.get("loss_mask_alignment")
            != LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1
        ):
            raise ValueError(
                f"objective loss-mask alignment drifted for bucket {bucket}"
            )
        graph = _require_mapping(
            contract.get("graph_auxiliary"), where=f"objective {bucket} graph"
        )
        validate_stage1_graph_contract(graph)
        if (
            graph.get("pair_mask") != "causal_same_document_upstream_v1"
            or graph.get("chunk_edge_expansion") != "cartesian_token_spans_v1"
        ):
            raise ValueError(f"objective graph contract drifted for bucket {bucket}")

        source_summary, source_records = _objective_source_summary(
            contract.get("source_snapshot"), bucket=bucket
        )
        if descriptor.get("source_snapshot") != source_summary:
            raise ValueError(f"objective source summary drifted for bucket {bucket}")
        snapshot_records = repaired_by_bucket.get(bucket, [])
        snapshot_counter = Counter(
            (int(record["size"]), str(record["snapshot_sha256"]))
            for record in snapshot_records
        )
        if source_records != snapshot_counter:
            raise ValueError(
                f"objective source snapshot does not match repaired snapshot for bucket {bucket}"
            )

        artifact_relative = _require_relative_string(
            descriptor.get("artifact_path"), where=f"objective {bucket} artifact path"
        )
        artifact_record = _require_artifact(
            artifacts, artifact_relative, where=f"objective {bucket} artifact"
        )
        if artifact_record["sha256"] != descriptor["artifact_file_sha256"]:
            raise ValueError(
                f"objective artifact file hash drifted for bucket {bucket}"
            )
        _, artifact = _load_json_object(
            _safe_bundle_path(
                root, artifact_relative, where=f"objective {bucket} artifact"
            ),
            where=f"objective {bucket} artifact",
        )
        _validate_objective_artifact_shape(artifact, bucket=bucket)
        artifact_payload = dict(artifact)
        artifact_digest = artifact_payload.pop("artifact_set_sha256", None)
        if artifact_digest != descriptor[
            "artifact_set_sha256"
        ] or artifact_digest != _canonical_sha256(artifact_payload):
            raise ValueError(
                f"objective artifact payload hash drifted for bucket {bucket}"
            )
        contract_ref = _require_mapping(
            artifact.get("objective_contract"),
            where=f"objective {bucket} artifact contract",
        )
        if (
            contract_ref.get("sha256") != descriptor["contract_sha256"]
            or contract_ref.get("file_sha256") != descriptor["contract_file_sha256"]
            or int(artifact.get("documents", -1)) != int(totals["samples"])
        ):
            raise ValueError(
                f"objective artifact contract binding drifted for bucket {bucket}"
            )
        validated[bucket] = _ObjectiveValidation(
            contract=contract,
            artifact=artifact,
            source_summary=source_summary,
            descriptor=descriptor,
        )
    return validated


def _validate_objective_artifact_shape(
    artifact: Mapping[str, object], *, bucket: int
) -> None:
    """Validate the versioned artifact envelope before hash-bound consumption."""

    if artifact.get("schema") == _LEGACY_OBJECTIVE_ARTIFACT_SCHEMA:
        raise ValueError(
            f"objective artifact payload for bucket {bucket} is legacy; migration "
            "required: regenerate the objective artifact and bundle"
        )
    if artifact.get("schema") != _OBJECTIVE_ARTIFACT_SCHEMA:
        raise ValueError(
            f"objective artifact payload schema drifted for bucket {bucket}"
        )
    expected_fields = {
        "schema",
        "graph_recipe",
        "documents",
        "objective_contract",
        "parquet_shards",
        "converter",
        "artifact_set_sha256",
    }
    if set(artifact) != expected_fields:
        raise ValueError(
            f"objective artifact fields drifted for bucket {bucket}: "
            f"{sorted(artifact)}"
        )
    if artifact.get("graph_recipe") != stage1_graph_recipe_binding():
        raise ValueError(
            f"objective artifact graph recipe drifted for bucket {bucket}; "
            "regenerate the objective artifact and bundle"
        )


def _validate_bucket_prefixes(
    root: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
    *,
    buckets: list[int],
    objectives: Mapping[int, _ObjectiveValidation],
) -> dict[int, _PrefixValidation]:
    raw_results = manifest.get("bucket_results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("bundle bucket_results must be a non-empty list")
    result_buckets = [
        result.get("bucket") if isinstance(result, dict) else None
        for result in raw_results
    ]
    if result_buckets != buckets:
        raise ValueError("bundle bucket_results must exactly match ordered buckets")

    validated: dict[int, _PrefixValidation] = {}
    expected_idx_paths: set[str] = set()
    for raw_result in raw_results:
        result = dict(_require_mapping(raw_result, where="bucket result"))
        bucket = _require_positive_int(
            result.get("bucket"), where="bucket result bucket"
        )
        relative = _require_relative_string(
            result.get("prefix"), where=f"bucket {bucket} prefix"
        )
        if PurePosixPath(relative).suffix:
            raise ValueError(f"bucket {bucket} prefix must not have a suffix")
        prefix = _safe_bundle_path(
            root, relative, where=f"bucket {bucket} prefix", require_file=False
        )
        token_relative = f"{relative}.bin"
        index_relative = f"{relative}.idx"
        sidecar_relative = f"{relative}.json"
        for path in (token_relative, index_relative, sidecar_relative):
            _require_artifact(artifacts, path, where=f"bucket {bucket} prefix")
            _safe_bundle_path(root, path, where=f"bucket {bucket} prefix")
        expected_idx_paths.add(index_relative)
        _, sidecar = _load_json_object(
            root / sidecar_relative, where=f"bucket {bucket} prefix manifest"
        )
        if result.get("manifest") != sidecar:
            raise ValueError(f"bucket {bucket} embedded prefix manifest drifted")
        _validate_prefix_manifest(
            root,
            relative=relative,
            bucket=bucket,
            sidecar=sidecar,
            artifacts=artifacts,
            objective=objectives[bucket],
        )
        validated[bucket] = _PrefixValidation(
            prefix=prefix,
            relative=relative,
            manifest_sha256=str(artifacts[sidecar_relative]["sha256"]),
        )
    listed_idx_paths = {
        relative for relative in artifacts if PurePosixPath(relative).suffix == ".idx"
    }
    if listed_idx_paths != expected_idx_paths:
        raise ValueError("bundle contains an indexed prefix absent from bucket_results")
    return validated


def _validate_prefix_manifest(
    root: Path,
    *,
    relative: str,
    bucket: int,
    sidecar: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
    objective: _ObjectiveValidation,
) -> None:
    if sidecar.get("tokenizer_contract") != _PREFIX_TOKENIZER_CONTRACT:
        raise ValueError(f"bucket {bucket} prefix tokenizer contract drifted")
    if int(sidecar.get("vocab_size", -1)) != _EXPECTED_VOCAB_SIZE:
        raise ValueError(f"bucket {bucket} prefix vocab size drifted")
    if int(sidecar.get("token_count", -1)) < 1 or int(
        sidecar.get("document_count", -1)
    ) != int(objective.contract["totals"]["samples"]):
        raise ValueError(f"bucket {bucket} prefix token/document counts are invalid")

    case5 = _require_mapping(
        sidecar.get(_CASE5_RECEIPT_KEY), where=f"bucket {bucket} CASE5 receipt"
    )
    expected_case5 = {
        "schema": _CASE5_SCHEMA,
        "status": "success",
        "delimiter_contract_sha256": DOMAIN_DELIMITER_CONTRACT_SHA256,
        "domain_schema_sha256": DOMAIN_SCHEMA_SHA256,
        "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
        "domain_route_columns": list(_DOMAIN_ROUTE_COLUMNS),
        "graph_route_columns": list(_GRAPH_ROUTE_COLUMNS),
        "graph_sidecars_written": True,
        "source_identity_registry_schema": _SOURCE_IDENTITY_REGISTRY_SCHEMA,
    }
    drift = {
        name: case5.get(name)
        for name, expected in expected_case5.items()
        if case5.get(name) != expected
    }
    if drift:
        raise ValueError(
            f"bucket {bucket} prefix has stale CASE5 receipt fields: {drift}"
        )

    side_paths = _require_mapping(
        sidecar.get("side_channel_paths"), where=f"bucket {bucket} side channels"
    )
    missing_sidecars = sorted(set(_TOKEN_SIDECAR_DTYPES) - set(side_paths))
    if missing_sidecars:
        raise ValueError(
            f"bucket {bucket} prefix is missing sidecars {missing_sidecars}"
        )
    for name, expected_dtype in _TOKEN_SIDECAR_DTYPES.items():
        spec = _require_mapping(
            side_paths[name], where=f"bucket {bucket} sidecar {name}"
        )
        if spec.get("dtype") != expected_dtype:
            raise ValueError(f"bucket {bucket} sidecar {name} dtype drifted")
        _require_referenced_artifact(
            root,
            artifacts,
            spec.get("path"),
            where=f"bucket {bucket} sidecar {name}",
            base=PurePosixPath(relative).parent,
        )

    if sidecar.get("graph_sidecar_schema") != _GRAPH_SIDECAR_SCHEMA:
        raise ValueError(f"bucket {bucket} graph sidecar schema drifted")
    graph_paths = _require_mapping(
        sidecar.get("graph_sidecar_paths"), where=f"bucket {bucket} graph sidecars"
    )
    missing_graph = sorted(set(_GRAPH_ROUTE_COLUMNS) - set(graph_paths))
    if missing_graph:
        raise ValueError(
            f"bucket {bucket} prefix is missing graph sidecars {missing_graph}"
        )
    for name in _GRAPH_ROUTE_COLUMNS:
        spec = _require_mapping(
            graph_paths[name], where=f"bucket {bucket} graph sidecar {name}"
        )
        for field in ("offsets_path", "data_path"):
            _require_referenced_artifact(
                root,
                artifacts,
                spec.get(field),
                where=f"bucket {bucket} graph sidecar {name}.{field}",
                base=PurePosixPath(relative).parent,
            )

    registry = _require_mapping(
        sidecar.get("source_identity_registry"),
        where=f"bucket {bucket} source identity registry",
    )
    if (
        registry.get("schema") != _SOURCE_IDENTITY_REGISTRY_SCHEMA
        or registry.get("id_encoding") != "uint64_be"
        or registry.get("canonical_digest") != "sha256"
        or registry.get("token_foreign_key_sidecar") != "token_source_identity_ids"
    ):
        raise ValueError(f"bucket {bucket} source identity registry contract drifted")
    _require_referenced_artifact(
        root,
        artifacts,
        registry.get("path"),
        where=f"bucket {bucket} source identity registry",
        base=PurePosixPath(relative).parent,
    )

    source_platform = _require_mapping(
        sidecar.get("source_platform_sidecar"),
        where=f"bucket {bucket} source platform sidecar",
    )
    if source_platform.get("schema") != "cppmega_source_platform_v1":
        raise ValueError(f"bucket {bucket} source platform schema drifted")
    for field in (
        "sequence_doc_offsets_path",
        "doc_platform_offsets_path",
        "platform_ids_path",
    ):
        _require_referenced_artifact(
            root,
            artifacts,
            source_platform.get(field),
            where=f"bucket {bucket} source platform {field}",
            base=PurePosixPath(relative).parent,
        )

    objective_wrapper = _require_mapping(
        sidecar.get("objective_contract"), where=f"bucket {bucket} objective contract"
    )
    objective_descriptor = objective.descriptor
    if (
        objective_wrapper.get("schema") != _OBJECTIVE_CONTRACT_SCHEMA
        or objective_wrapper.get("sha256") != objective_descriptor["contract_sha256"]
        or objective_wrapper.get("payload") != objective.contract
    ):
        raise ValueError(f"bucket {bucket} prefix objective contract drifted")
    objective_id = _require_mapping(
        objective_wrapper.get("objective_id_sidecar"),
        where=f"bucket {bucket} objective ID sidecar",
    )
    if (
        objective_id.get("dtype") != "uint8"
        or objective_id.get("document_aligned") is not True
    ):
        raise ValueError(f"bucket {bucket} objective ID sidecar contract drifted")
    _require_referenced_artifact(
        root,
        artifacts,
        objective_id.get("path"),
        where=f"bucket {bucket} objective ID sidecar",
        base=PurePosixPath(relative).parent,
    )
    expected_materialization = {
        **objective.artifact,
        "artifact_file_sha256": objective_descriptor["artifact_file_sha256"],
    }
    if sidecar.get("objective_materialization") != expected_materialization:
        raise ValueError(f"bucket {bucket} prefix objective artifact drifted")


def _validate_restore_receipt(
    root: Path,
    receipt_path: Path,
    *,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, object]],
    prefixes: Mapping[int, _PrefixValidation],
    expected_bundle_id: str,
    logical_manifest_sha256: str,
) -> _RestoreValidation:
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt_bytes, receipt = _load_json_object(
        receipt_path, where="Megatron restore receipt"
    )
    if receipt.get("schema") != _RESTORE_RECEIPT_SCHEMA:
        raise ValueError("unsupported Megatron restore receipt schema")
    if receipt.get("status") not in {"restored_verified", "already_verified"}:
        raise ValueError("Megatron restore receipt is not verified")
    expected_top = {
        "bundle_id": expected_bundle_id,
        "logical_manifest_sha256": logical_manifest_sha256,
        "artifact_set_sha256": manifest["artifact_set_sha256"],
        "artifact_count": manifest["artifact_count"],
        "artifact_bytes": manifest["artifact_bytes"],
    }
    drift = {
        name: receipt.get(name)
        for name, expected in expected_top.items()
        if receipt.get(name) != expected
    }
    if drift:
        raise ValueError(f"Megatron restore receipt bundle binding drifted: {drift}")
    _require_sha256(receipt.get("transport_sha256"), where="restore transport")

    transport = receipt.get("transport")
    if not isinstance(transport, str):
        raise ValueError("restore receipt transport must be an S3 URI")
    parsed = urlparse(transport)
    expected_suffix = f"/transports/{expected_bundle_id}/transport.json"
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(expected_suffix)
    ):
        raise ValueError("restore receipt transport URI is not bound to the bundle")

    binding = _require_mapping(receipt.get("binding"), where="restore binding")
    expected_binding_fields = {
        "schema",
        "bundle_id",
        "artifact_set_sha256",
        "prefix_manifest_sha256s",
        "checkpoint_sha256",
        "config_sha256",
        "command_sha256",
        "run_id",
        "implementation",
    }
    if set(binding) != expected_binding_fields:
        raise ValueError("restore receipt binding fields drifted")
    if (
        binding.get("schema") != _RESTORE_BINDING_SCHEMA
        or binding.get("bundle_id") != expected_bundle_id
        or binding.get("artifact_set_sha256") != manifest["artifact_set_sha256"]
        or binding.get("checkpoint_sha256") != _NO_CHECKPOINT_SHA256
    ):
        raise ValueError("restore receipt identity binding drifted")
    for name in ("config_sha256", "command_sha256"):
        _require_sha256(binding.get(name), where=f"restore binding {name}")
    run_id = binding.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("restore receipt run_id is invalid")
    expected_prefix_hashes = {
        f"{prefix.relative}.json": str(artifacts[f"{prefix.relative}.json"]["sha256"])
        for prefix in prefixes.values()
    }
    if binding.get("prefix_manifest_sha256s") != dict(
        sorted(expected_prefix_hashes.items())
    ):
        raise ValueError("restore receipt prefix-manifest binding drifted")
    _validate_training_implementation_extension(
        manifest.get("implementation"),
        binding.get("implementation"),
    )
    return _RestoreValidation(
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        run_id=run_id,
        storage_bucket=parsed.netloc,
    )


def _objective_source_summary(
    raw: object, *, bucket: int
) -> tuple[dict[str, object], Counter[tuple[int, str]]]:
    source = _require_mapping(raw, where=f"objective {bucket} source snapshot")
    expected_fields = {
        "schema",
        "sequence_length",
        "file_count",
        "row_count",
        "files",
        "sampling",
        "artifact_set_sha256",
    }
    if set(source) != expected_fields:
        raise ValueError(
            f"objective source snapshot fields drifted for bucket {bucket}"
        )
    if (
        source.get("schema") != _OBJECTIVE_SOURCE_SCHEMA
        or source.get("sequence_length") != bucket
    ):
        raise ValueError(
            f"objective source snapshot schema/bucket drifted for {bucket}"
        )
    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"objective source snapshot has no files for bucket {bucket}")
    records: list[dict[str, object]] = []
    source_counter: Counter[tuple[int, str]] = Counter()
    file_row_counts: list[int] = []
    row_count = 0
    for raw_record in files:
        record = _require_mapping(
            raw_record, where=f"objective source file for bucket {bucket}"
        )
        if set(record) != {"path", "size_bytes", "sha256", "rows"}:
            raise ValueError(
                f"objective source file fields drifted for bucket {bucket}"
            )
        path = record.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(
                f"objective source path for bucket {bucket} must be a non-empty string"
            )
        size = _require_positive_int(
            record.get("size_bytes"), where=f"objective source size for bucket {bucket}"
        )
        digest = _require_sha256(
            record.get("sha256"), where=f"objective source hash for bucket {bucket}"
        )
        rows = _require_positive_int(
            record.get("rows"), where=f"objective source rows for bucket {bucket}"
        )
        records.append({"path": path, "size": size, "sha256": digest})
        file_row_counts.append(rows)
        source_counter[(size, digest)] += 1
        row_count += rows
    if source.get("file_count") != len(files) or source.get("row_count") != row_count:
        raise ValueError(
            f"objective source snapshot counts drifted for bucket {bucket}"
        )
    artifact_set_sha256 = _artifact_set_sha256(records)
    if source.get("artifact_set_sha256") != artifact_set_sha256:
        raise ValueError(
            f"objective source artifact-set hash drifted for bucket {bucket}"
        )
    sampling = _validate_objective_source_sampling(
        source.get("sampling"),
        row_count=row_count,
        file_count=len(files),
        file_row_counts=file_row_counts,
        bucket=bucket,
    )
    return (
        {
            "schema": _OBJECTIVE_SOURCE_SCHEMA,
            "artifact_set_sha256": artifact_set_sha256,
            "file_count": len(files),
            "row_count": row_count,
            "sampling": dict(sampling),
        },
        source_counter,
    )


def _validate_objective_source_sampling(
    raw_sampling: object,
    *,
    row_count: int,
    file_count: int,
    file_row_counts: list[int],
    bucket: int,
) -> dict[str, object]:
    sampling = _require_mapping(
        raw_sampling, where=f"objective source sampling for bucket {bucket}"
    )
    common_fields = {
        "mode",
        "seed",
        "requested_samples",
        "full_passes",
        "tail_rows",
        "min_row_reuse",
        "max_row_reuse",
    }
    mode = sampling.get("mode")
    if mode == _LEGACY_SOURCE_SAMPLING_MODE:
        expected_fields = common_fields
    elif mode == _BOUNDED_SOURCE_SAMPLING_MODE:
        expected_fields = common_fields | {
            "record_batch_rows",
            "producer",
            "ordering",
            "cursor_semantics",
            "final_cursor",
        }
    else:
        raise ValueError(
            f"objective source sampling mode is unsupported for bucket {bucket}: "
            f"{mode!r}"
        )
    if set(sampling) != expected_fields:
        raise ValueError(
            f"objective source sampling fields drifted for bucket {bucket}"
        )

    seed = sampling.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"objective source seed must be an integer for bucket {bucket}")
    requested = _require_positive_int(
        sampling.get("requested_samples"),
        where=f"objective requested samples for bucket {bucket}",
    )
    full_passes, tail_rows = divmod(requested, row_count)
    if (
        sampling.get("full_passes") != full_passes
        or sampling.get("tail_rows") != tail_rows
        or sampling.get("min_row_reuse") != full_passes
        or sampling.get("max_row_reuse") != full_passes + int(tail_rows > 0)
    ):
        raise ValueError(f"objective source sampling drifted for bucket {bucket}")

    if mode == _LEGACY_SOURCE_SAMPLING_MODE:
        return dict(sampling)

    record_batch_rows = _require_positive_int(
        sampling.get("record_batch_rows"),
        where=f"objective source record_batch_rows for bucket {bucket}",
    )
    producer = _require_mapping(
        sampling.get("producer"),
        where=f"objective source producer for bucket {bucket}",
    )
    if set(producer) != {"name", "version", "row_group_rows"}:
        raise ValueError(
            f"objective source producer fields drifted for bucket {bucket}"
        )
    if (
        producer.get("name") != _BOUNDED_SOURCE_PRODUCER
        or producer.get("version") != _BOUNDED_SOURCE_PRODUCER_VERSION
    ):
        raise ValueError(f"objective source producer drifted for bucket {bucket}")
    raw_row_groups = producer.get("row_group_rows")
    if not isinstance(raw_row_groups, list) or len(raw_row_groups) != file_count:
        raise ValueError(
            f"objective source producer shard layout drifted for bucket {bucket}"
        )
    row_group_rows: list[list[int]] = []
    for shard_index, (raw_groups, expected_rows) in enumerate(
        zip(raw_row_groups, file_row_counts, strict=True)
    ):
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError(
                f"objective source producer row groups are invalid for bucket "
                f"{bucket}, shard {shard_index}"
            )
        groups: list[int] = []
        for raw_rows in raw_groups:
            groups.append(
                _require_positive_int(
                    raw_rows,
                    where=(
                        f"objective source producer row-group size for bucket "
                        f"{bucket}, shard {shard_index}"
                    ),
                )
            )
        if sum(groups) != expected_rows:
            raise ValueError(
                f"objective source producer row count drifted for bucket {bucket}, "
                f"shard {shard_index}"
            )
        row_group_rows.append(groups)
    if sampling.get("ordering") != _BOUNDED_SOURCE_ORDERING:
        raise ValueError(
            f"objective source bounded ordering drifted for bucket {bucket}"
        )
    if sampling.get("cursor_semantics") != "last_yielded_row_v1":
        raise ValueError(
            f"objective source cursor semantics drifted for bucket {bucket}"
        )
    cursor = _require_mapping(
        sampling.get("final_cursor"),
        where=f"objective source final cursor for bucket {bucket}",
    )
    if set(cursor) != _BOUNDED_SOURCE_CURSOR_FIELDS:
        raise ValueError(
            f"objective source final cursor fields drifted for bucket {bucket}"
        )
    for name in _BOUNDED_SOURCE_CURSOR_FIELDS:
        value = cursor.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"objective source final cursor {name} must be a non-negative "
                f"integer for bucket {bucket}"
            )
    if cursor["source_index"] != requested - 1:
        raise ValueError(
            f"objective source final cursor index drifted for bucket {bucket}"
        )
    if cursor["epoch"] != (requested - 1) // row_count:
        raise ValueError(
            f"objective source final cursor epoch drifted for bucket {bucket}"
        )
    if cursor["shard_position"] >= file_count or cursor["shard_index"] >= file_count:
        raise ValueError(
            f"objective source final cursor shard drifted for bucket {bucket}"
        )
    shard_row_groups = row_group_rows[cursor["shard_index"]]
    if (
        cursor["row_group_position"] >= len(shard_row_groups)
        or cursor["row_group_index"] >= len(shard_row_groups)
    ):
        raise ValueError(
            f"objective source final cursor row group drifted for bucket {bucket}"
        )
    group_rows = shard_row_groups[cursor["row_group_index"]]
    record_batch_count = (group_rows + record_batch_rows - 1) // record_batch_rows
    if cursor["record_batch_index"] >= record_batch_count:
        raise ValueError(
            f"objective source final cursor record batch drifted for bucket {bucket}"
        )
    batch_start = cursor["record_batch_index"] * record_batch_rows
    batch_rows = min(record_batch_rows, group_rows - batch_start)
    if (
        cursor["row_shuffle_position"] >= batch_rows
        or cursor["row_index_in_record_batch"] >= batch_rows
    ):
        raise ValueError(
            f"objective source final cursor row drifted for bucket {bucket}"
        )
    return dict(sampling)


def _require_referenced_artifact(
    root: Path,
    artifacts: Mapping[str, Mapping[str, object]],
    raw_path: object,
    *,
    where: str,
    base: PurePosixPath,
) -> str:
    relative = _require_relative_string(raw_path, where=where)
    joined = (base / PurePosixPath(relative)).as_posix()
    _safe_bundle_path(root, joined, where=where)
    _require_artifact(artifacts, joined, where=where)
    return joined


def _source_record_key(
    record: Mapping[str, Any], *, where: str
) -> tuple[str, int, str]:
    kind = record.get("kind")
    snapshot = record.get("snapshot")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"{where} kind is invalid")
    bucket = _require_positive_int(record.get("bucket"), where=f"{where} bucket")
    snapshot_path = _require_relative_string(snapshot, where=f"{where} snapshot")
    return kind, bucket, snapshot_path


def _safe_bundle_path(
    root: Path,
    relative: str,
    *,
    where: str,
    require_file: bool = True,
) -> Path:
    posix = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or posix.is_absolute()
        or posix.as_posix() != relative
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"unsafe {where} path: {relative!r}")
    path = root.joinpath(*posix.parts)
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise ValueError(f"{where} path escapes bundle: {relative!r}") from error
    if require_file and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_json_object(path: Path, *, where: str) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{where} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{where} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{where} must be a JSON object: {path}")
    return raw, payload


def _stable_file_sha256(path: Path) -> tuple[str, tuple[int, int, int, int, int]]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"bundle artifact must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    before_signature = _stat_signature(before)
    after_signature = _stat_signature(after)
    if before_signature != after_signature:
        raise ValueError(f"bundle artifact changed while hashing: {path}")
    return digest.hexdigest(), after_signature


def _assert_artifacts_unchanged(
    root: Path, signatures: Mapping[str, tuple[int, int, int, int, int]]
) -> None:
    for relative, expected in signatures.items():
        path = _safe_bundle_path(root, relative, where="validated artifact")
        if _stat_signature(path.lstat()) != expected:
            raise ValueError(f"bundle artifact changed after validation: {relative}")


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_artifact_records(
    records: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    canonical: list[dict[str, object]] = []
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "size", "sha256"}:
            raise ValueError("artifact records must contain exactly path/size/sha256")
        path = raw.get("path")
        size = raw.get("size")
        if not isinstance(path, str) or not path:
            raise ValueError("artifact record path must be a non-empty string")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"artifact {path} size is invalid")
        digest = _require_sha256(raw.get("sha256"), where=f"artifact {path}")
        canonical.append({"path": path, "size": size, "sha256": digest})
    paths = [str(record["path"]) for record in canonical]
    if len(paths) != len(set(paths)):
        raise ValueError("artifact records contain duplicate paths")
    return sorted(canonical, key=lambda record: str(record["path"]))


def _artifact_set_sha256(records: Iterable[Mapping[str, object]]) -> str:
    payload = json.dumps(
        _canonical_artifact_records(records),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: object, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    return value


def _require_artifact(
    artifacts: Mapping[str, Mapping[str, object]], relative: str, *, where: str
) -> Mapping[str, object]:
    record = artifacts.get(relative)
    if record is None:
        raise ValueError(f"{where} is absent from the bundle artifact set: {relative}")
    return record


def _require_relative_string(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty relative path")
    posix = PurePosixPath(value)
    if (
        "\\" in value
        or posix.is_absolute()
        or posix.as_posix() != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"{where} is unsafe: {value!r}")
    return value


def _require_positive_int(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _validate_implementation_binding(
    value: object,
    *,
    where: str,
    required_components: Iterable[str],
) -> dict[str, object]:
    binding = dict(
        _require_mapping(value, where=f"{where} implementation binding")
    )
    expected_fields = {"schema", "components"}
    if set(binding) != expected_fields:
        raise ValueError(
            f"{where} implementation binding fields drifted: "
            f"missing={sorted(expected_fields - set(binding))} "
            f"extra={sorted(set(binding) - expected_fields)}"
        )
    if binding.get("schema") != _IMPLEMENTATION_BINDING_SCHEMA:
        raise ValueError(f"{where} implementation binding schema drifted")
    components = _require_mapping(
        binding.get("components"),
        where=f"{where} implementation components",
    )
    expected_components = set(required_components)
    actual_components = set(components)
    if actual_components != expected_components:
        raise ValueError(
            f"{where} implementation components drifted: "
            f"missing={sorted(expected_components - actual_components)} "
            f"extra={sorted(actual_components - expected_components)}"
        )

    normalized: dict[str, dict[str, str]] = {}
    for name, raw_component in components.items():
        if not isinstance(name, str) or _RUN_ID_RE.fullmatch(name) is None:
            raise ValueError(f"{where} implementation component name is invalid")
        component = dict(
            _require_mapping(
                raw_component,
                where=f"{where} implementation component {name}",
            )
        )
        expected_fields = _IMPLEMENTATION_COMPONENT_FIELDS[name]
        actual_fields = set(component)
        if actual_fields != expected_fields:
            raise ValueError(
                f"{where} implementation component {name} fields drifted: "
                f"missing={sorted(expected_fields - actual_fields)} "
                f"extra={sorted(actual_fields - expected_fields)}"
            )
        normalized_component: dict[str, str] = {}
        if "commit" in component:
            commit = component["commit"]
            if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
                raise ValueError(
                    f"{where} implementation {name}.commit must be a lowercase "
                    "Git commit SHA"
                )
            normalized_component["commit"] = commit
        for field in (
            "tree_sha256",
            "source_sha256",
            "dependency_closure_sha256",
        ):
            if field in component:
                normalized_component[field] = _require_sha256(
                    component[field],
                    where=f"{where} implementation {name}.{field}",
                )
        normalized[name] = dict(sorted(normalized_component.items()))
    return {
        "schema": _IMPLEMENTATION_BINDING_SCHEMA,
        "components": dict(sorted(normalized.items())),
    }


def _validate_training_implementation_extension(
    producer_value: object,
    training_value: object,
) -> None:
    producer = _validate_implementation_binding(
        producer_value,
        where="production bundle",
        required_components=_PRODUCER_IMPLEMENTATION_COMPONENTS,
    )
    training = _validate_implementation_binding(
        training_value,
        where="restore receipt",
        required_components=_TRAINING_IMPLEMENTATION_COMPONENTS,
    )
    producer_components = producer["components"]
    training_components = training["components"]
    expected_training_components = set(producer_components) | {"megatron"}
    if set(training_components) != expected_training_components:
        raise ValueError(
            "restore receipt implementation component set does not exactly "
            "extend the bundle producer"
        )
    for name, producer_component in producer_components.items():
        if training_components.get(name) != producer_component:
            raise ValueError(
                "restore receipt implementation does not extend the bundle "
                f"producer for {name}"
            )


__all__ = [
    "ProductionMegatronDatasetMetadata",
    "open_production_megatron_bundle",
]
