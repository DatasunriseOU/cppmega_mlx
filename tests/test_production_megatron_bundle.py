from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
import pytest

import cppmega_mlx.data.production_bundle as production_bundle
from cppmega_mlx.data.domain_schema import DOMAIN_SCHEMA_SHA256
from cppmega_mlx.data.graph_recipe import (
    stage1_graph_recipe_binding,
    stage1_graph_recipe_payload,
)
from cppmega_mlx.data.production_bundle import (
    ProductionMegatronDatasetMetadata,
    open_production_megatron_bundle,
)
from cppmega_mlx.data.tokenizer_contract import (
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_PATH,
    TOKENIZER_CONTRACT_SHA256,
)


_BUCKET = 8
_STORAGE_BUCKET = "cppmega-production-test"
_PREFIX_RELATIVE = "data/seq8/cppmega_macro_routes_seq8_train"
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
_DTYPES = {
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
    "uint64": np.dtype(np.uint64),
}


@dataclass(frozen=True)
class _BundleFixture:
    root: Path
    bundle_id: str
    prefix: Path
    restore_receipt: Path
    sidecar_paths: dict[str, Path]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _producer_implementation() -> dict[str, Any]:
    return {
        "schema": "cppmega_implementation_binding_v1",
        "components": {
            "cppmega": {
                "commit": "1" * 40,
                "tree_sha256": "2" * 64,
            },
            "cppmega_mlx": {
                "commit": "3" * 40,
                "tree_sha256": "4" * 64,
            },
            "clang_indexer": {
                "source_sha256": "5" * 64,
                "dependency_closure_sha256": "6" * 64,
            },
        },
    }


def _training_implementation() -> dict[str, Any]:
    implementation = _producer_implementation()
    implementation["components"]["megatron"] = {"commit": "7" * 40}
    return implementation


def _artifact_records(root: Path) -> list[dict[str, object]]:
    excluded = {root / "manifest.json", root / "restore_receipt.json"}
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path not in excluded
    ]


def _artifact_set_sha256(records: list[dict[str, object]]) -> str:
    canonical = sorted(records, key=lambda record: str(record["path"]))
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_source_composition_provenance(root: Path) -> dict[str, object]:
    provenance = root / "provenance" / "source_composition"
    provenance.mkdir(parents=True, exist_ok=True)

    def write(name: str, payload: object) -> dict[str, object]:
        path = provenance / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            _write_json(path, payload)
        return {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    plan = write(
        "plan.json",
        {
            "schema": "cppmega_source_conveyor_composition_plan_v1",
            "runs": [{"run_id": "full"}],
            "dedup_receipt": "/fixture/global_dedup_receipt.json",
        },
    )
    verifier = write("verify_global_dedup_store.py", b"# fixture verifier\n")
    dedup_payload = {
        "schema": "cppmega_global_dedup_store_receipt_v1",
        "status": "verified",
        "created_at": "2026-07-28T00:00:00Z",
        "database": {
            "path": "/fixture/dedup.sqlite",
            "size_bytes": 1,
            "sha256": "1" * 64,
        },
        "checkpoint": {
            "mode": "TRUNCATE",
            "busy": 0,
            "log_frames": 0,
            "checkpointed_frames": 0,
            "wal_size_bytes": 0,
        },
        "integrity_check": "ok",
        "sqlite_schema_sha256": "2" * 64,
        "logical_hash_algorithm": "cppmega_sqlite_rows_lenprefixed_v1",
        "logical_sha256": "3" * 64,
        "tables": {
            name: {
                "rows": (0 if name.endswith("_stage") or name == "dedup_stages" else 1),
                "logical_sha256": hashlib.sha256(b"").hexdigest(),
            }
            for name in (
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
            )
        },
        "policy": {
            "exact": "sha1_token_ids_v1",
            "chunk": "tokenized_chunk_claims_v1",
            "near": {
                "enabled": True,
                "threshold": 0.7,
                "num_perm": 256,
                "shingle_k": 5,
            },
        },
        "verifier": {
            "repository_identity": "cppmega",
            "script": "scripts/data/verify_global_dedup_store.py",
            "script_sha256": verifier["sha256"],
        },
    }
    dedup = write("global_dedup_receipt.json", dedup_payload)
    inputs = {
        "archive_sha256_receipt": write(
            "runs/full/archive_sha256_receipt.json", {"fixture": "archive"}
        ),
        "archive_inventory_receipt": write(
            "runs/full/archive_inventory.json", {"fixture": "inventory"}
        ),
        "repo_list": write("runs/full/repo_list.json", {"fixture": "repos"}),
        "source_quarantine_manifest": write(
            "runs/full/source_quarantine_manifest.json",
            {"fixture": "quarantine"},
        ),
        "tokenizer": write("runs/full/tokenizer.json", {"fixture": "tokenizer"}),
    }
    run_artifacts = {
        "launch": write("runs/full/launch.json", {"fixture": "launch"}),
        "exit": write("runs/full/exit.json", {"fixture": "exit"}),
        "manifest": write("runs/full/manifest.json", {"fixture": "manifest"}),
        "archive_sha256_receipt": inputs["archive_sha256_receipt"],
        "archive_inventory": inputs["archive_inventory_receipt"],
        "repo_list": inputs["repo_list"],
        "source_quarantine_manifest": inputs["source_quarantine_manifest"],
        "tokenizer": inputs["tokenizer"],
    }
    producer = {
        "cppmega": {"commit": "a" * 40, "tree_sha256": "b" * 64},
        "clang_indexer": {
            "source_sha256": "c" * 64,
            "dependency_closure_sha256": "d" * 64,
        },
    }
    portable_dedup = dict(dedup_payload)
    portable_database = dict(dedup_payload["database"])
    portable_database.pop("path")
    portable_dedup["database"] = portable_database
    portable_dedup["receipt_sha256"] = dedup["sha256"]
    repository_set_sha256 = _canonical_sha256(["repo"])
    run_receipt = {
        "run_id": "full",
        "launch": {
            "schema": "cppmega.canonical_source_launch_v1",
            "sha256": run_artifacts["launch"]["sha256"],
        },
        "exit": {
            "schema": "cppmega.canonical_source_exit_v1",
            "sha256": run_artifacts["exit"]["sha256"],
            "exit_code": 0,
        },
        "manifest": {
            "sha256": run_artifacts["manifest"]["sha256"],
            "done_units": 2,
            "failed_units": 0,
            "done_unit_set_sha256": "4" * 64,
            "failed_unit_set_sha256": "5" * 64,
        },
        "streams": "both",
        "selected_repositories": [],
        "terminal_repositories": ["repo"],
        "terminal_repository_set_sha256": repository_set_sha256,
        "input_artifacts": {
            name: descriptor["sha256"] for name, descriptor in inputs.items()
        },
        "code_revision": producer,
        "allowlist_counts": {f"code/{_BUCKET}": 1, f"commits/{_BUCKET}": 1},
    }
    receipt_payload = {
        "schema": "cppmega_source_conveyor_composition_v1",
        "status": "complete",
        "plan_sha256": plan["sha256"],
        "buckets": [_BUCKET],
        "archive": {
            "repository_count": 1,
            "repository_names_sha256": repository_set_sha256,
            "input_binding_sha256": "8" * 64,
            "archive_identity_sha256": "9" * 64,
        },
        "dedup": portable_dedup,
        "runs": [run_receipt],
        "source_producers": [producer],
        "source_producer_set_sha256": _canonical_sha256([producer]),
        "coverage": {
            "expected_repositories": 1,
            "code_success_repositories": 1,
            "commit_success_repositories": 1,
            "failed_repositories_observed": 0,
            "failed_units_observed": 0,
            "unresolved_failed_units": 0,
            "repository_set_sha256": repository_set_sha256,
            "allowlist_counts": {
                f"code/{_BUCKET}": 1,
                f"commits/{_BUCKET}": 1,
            },
        },
    }
    receipt = write("receipt.json", receipt_payload)
    return {
        "schema": receipt_payload["schema"],
        "receipt": {key: receipt[key] for key in ("path", "sha256")},
        "plan": {key: plan[key] for key in ("path", "sha256")},
        "dedup_receipt": {key: dedup[key] for key in ("path", "sha256")},
        "dedup_verifier": {key: verifier[key] for key in ("path", "sha256")},
        "runs": [{"run_id": "full", "artifacts": run_artifacts}],
    }


def _write_ci_production_provenance(
    root: Path,
    *,
    source_size: int,
    source_sha256: str,
) -> dict[str, object]:
    provenance = root / "provenance"
    staged_ledgers = provenance / "ci_export"
    staged_ledgers.mkdir(parents=True, exist_ok=True)
    ledger_names = {
        "representative_ledger": "representative_ledger.parquet",
        "fragment_ledger": "fragment_ledger.parquet",
        "dropped_graph_edges": "dropped_graph_edges.parquet",
        "representative_metadata": "representative_metadata.parquet",
        "excluded_opaque_artifacts": "excluded_opaque_artifacts.parquet",
        "excluded_training_scope": "excluded_training_scope.parquet",
        "source_binding_projection": "source_binding_projection.parquet",
        "occurrence_metadata": "occurrence_metadata.parquet",
    }
    artifacts: list[dict[str, object]] = [
        {
            "path": f"{_BUCKET}/ci-case5-train-{_BUCKET}-000000.parquet",
            "kind": "case5_parquet",
            "bucket": _BUCKET,
            "split": "train",
            "rows": 1,
            "byte_size": source_size,
            "sha256": source_sha256,
        }
    ]
    for kind, name in ledger_names.items():
        path = staged_ledgers / name
        path.write_text("{}\n", encoding="utf-8")
        artifacts.append(
            {
                "path": name,
                "kind": kind,
                "rows": 1,
                "byte_size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    fetch_state_sha256 = "1" * 64
    fetch_state_logical_sha256 = "2" * 64
    fetch_sidecar_set_sha256 = "3" * 64
    representative_ledger_sha256 = "4" * 64
    store_receipt_sha256 = "5" * 64
    source_binding = {
        "fetch_state_sqlite_logical_sha256": fetch_state_logical_sha256,
        "fetch_state_sidecar_set_sha256": fetch_sidecar_set_sha256,
        "representative_ledger_sha256": representative_ledger_sha256,
    }
    exact_tokens = 20_000_000_123
    acquisition = {
        "completion_mode": "inventory-exhaustive",
        "production_complete": True,
        "inventory": {
            "path": "/immutable-ci/inventory.sqlite3",
            "sha256": "6" * 64,
            "logical_sha256": "7" * 64,
            "receipt_path": "/immutable-ci/inventory_receipt.json",
            "receipt_sha256": "8" * 64,
        },
        "fetch": {
            "state_path": "/immutable-ci/fetch_state.sqlite3",
            "state_sha256": fetch_state_sha256,
            "receipt_path": "/immutable-ci/fetch_receipt.json",
            "receipt_sha256": "9" * 64,
            "attempt_set_sha256": "a" * 64,
            "terminal_proof_sha256": "b" * 64,
        },
        "store": {
            "path": "/immutable-ci/store",
            "receipt_path": "/immutable-ci/store_receipt.json",
            "receipt_sha256": store_receipt_sha256,
        },
        "merge": {
            "receipt_path": "/immutable-ci/merge_receipt.json",
            "receipt_sha256": "c" * 64,
            "schema": "cppmega_ci_stream_shard_union_receipt_v3",
        },
    }
    payload = {
        "schema": "cppmega_ci_content_store_case5_export_v4",
        "status": "complete",
        "completion_mode": "inventory-exhaustive",
        "production_complete": True,
        "exporter_script_sha256": "d" * 64,
        "input_store": {
            "schema": "cppmega_ci_content_store_v1",
            "receipt_schema": "cppmega_ci_content_store_receipt_v1",
            "receipt_sha256": store_receipt_sha256,
            "policy_sha256": "e" * 64,
            "sqlite_schema_sha256": "f" * 64,
            "logical_content_set_sha256": "0" * 64,
            "logical_token_sequence_set_sha256": "1" * 64,
            "occurrence_set_sha256": "2" * 64,
            "sqlite_logical_sha256": "3" * 64,
            "pack_hashes": [{"path": "pack-00000000.cicp", "sha256": "4" * 64}],
            "verified_before_export": True,
            "unchanged_after_export": True,
        },
        "input_fetch_state": {
            "schema": "cppmega_ci_stream_fetch_v4",
            "artifact": {
                "path": "/immutable-ci/fetch_state.sqlite3",
                "sha256": fetch_state_sha256,
            },
            "sqlite_schema_sha256": "5" * 64,
            "sqlite_logical_sha256": fetch_state_logical_sha256,
            "sidecar_set_sha256": fetch_sidecar_set_sha256,
            "settings": {
                "fetcher_script_sha256": "6" * 64,
                "parser_script_sha256": "7" * 64,
                "content_store_script_sha256": "8" * 64,
            },
        },
        "case5_contract": {
            "buckets": [_BUCKET],
            "overflow_rows": 0,
            "parquet_shard_max_rows": 512,
            "parquet_layout": "bucket-first-split-in-filename-v1",
            "parquet_compression": {"codec": "zstd", "level": 9},
        },
        "eligibility": {
            "target_exact_unique_payload_tokens": 20_000_000_000,
            "target_met": True,
            "eligible": {
                "unique_token_sequences": 1,
                "exact_unique_payload_tokens": exact_tokens,
            },
            "conservation": {
                "exact_unique_payload_tokens": True,
                "unique_token_sequences": True,
            },
        },
        "representatives": {
            "count": 1,
            "ledger_sha256": representative_ledger_sha256,
        },
        "counts": {
            "representatives": 1,
            "fragments": 1,
            "payload_tokens": exact_tokens,
            "valid_tokens": _BUCKET - 1,
            "trained_tokens": _BUCKET - 1,
            "capacity_tokens": _BUCKET,
        },
        "graph_accounting": {
            "cross_chunk_outbound_edges_dropped": 0,
            "cross_fragment_edges_dropped": 0,
        },
        "artifacts": artifacts,
        "validation": {
            "all_passed": True,
            "fixed_widths": True,
            "zero_overflow": True,
            "payload_conserved": True,
            "payload_identity_and_order_verified": True,
            "all_case5_parquet_zstd": True,
            "post_normalize_pack_sidecars_and_edges_verified": True,
        },
        "acquisition_provenance": acquisition,
    }
    path = provenance / "ci_manifest.json"
    _write_json(path, payload)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "source_inventory_sha256": _canonical_sha256(source_binding),
        "input_docs": 1,
        "fragments": 1,
        "valid_tokens": _BUCKET - 1,
        "cross_boundary_chunk_edges": 0,
        "cross_boundary_token_edges": 0,
    }


def _write_mmididx(prefix: Path, tokens: np.ndarray) -> None:
    tokens.astype(np.int32, copy=False).tofile(prefix.with_suffix(".bin"))
    with prefix.with_suffix(".idx").open("wb") as handle:
        handle.write(b"MMIDIDX\x00\x00")
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", 4))
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<Q", 2))
        np.asarray([len(tokens)], dtype=np.int32).tofile(handle)
        np.asarray([0], dtype=np.int64).tofile(handle)
        np.asarray([0, 1], dtype=np.int64).tofile(handle)


def _sidecar_values(name: str) -> np.ndarray:
    if name == "loss_mask":
        return np.ones(_BUCKET, dtype=np.uint8)
    if name == "doc_ids":
        return np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
    if name == "token_domain_ids":
        return np.full(_BUCKET, 3, dtype=np.uint16)
    if name == "token_role_ids":
        return np.full(_BUCKET, 2, dtype=np.uint16)
    if name == "token_source_doc_ids":
        return np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
    if name == "token_source_identity_ids":
        return np.ones(_BUCKET, dtype=np.uint64)
    if name == "token_confidence_ids":
        return np.ones(_BUCKET, dtype=np.uint8)
    if name == "token_structure_ids":
        return np.arange(1, _BUCKET + 1, dtype=np.uint8)
    return np.zeros(_BUCKET, dtype=_DTYPES[_TOKEN_SIDECAR_DTYPES[name]])


def _write_token_sidecars(
    prefix: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Path]]:
    specs: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for name, dtype in _TOKEN_SIDECAR_DTYPES.items():
        path = prefix.with_name(f"{prefix.name}_{name}.bin")
        _sidecar_values(name).tofile(path)
        specs[name] = {"path": path.name, "dtype": dtype}
        paths[name] = path
    return specs, paths


def _write_graph_sidecars(prefix: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, np.ndarray] = {
        "token_call_edges": np.asarray([[0, 1]], dtype=np.int32),
        "token_type_edges": np.zeros((0, 2), dtype=np.int32),
        "token_domain_edges": np.asarray([[6, 5, 60]], dtype=np.int32),
        "token_build_edges": np.zeros((0, 3), dtype=np.int32),
        "token_shell_edges": np.zeros((0, 3), dtype=np.int32),
        "token_diagnostic_edges": np.zeros((0, 3), dtype=np.int32),
        "token_cross_domain_edges": np.zeros((0, 3), dtype=np.int32),
        "token_chunk_starts": np.asarray([0, 4], dtype=np.uint32),
        "token_chunk_ends": np.asarray([4, 8], dtype=np.uint32),
        "token_chunk_kinds": np.asarray([1, 2], dtype=np.uint8),
        "token_chunk_dep_levels": np.asarray([0, 1], dtype=np.uint16),
    }
    dtypes = {
        "token_call_edges": "int32",
        "token_type_edges": "int32",
        "token_domain_edges": "int32",
        "token_build_edges": "int32",
        "token_shell_edges": "int32",
        "token_diagnostic_edges": "int32",
        "token_cross_domain_edges": "int32",
        "token_chunk_starts": "uint32",
        "token_chunk_ends": "uint32",
        "token_chunk_kinds": "uint8",
        "token_chunk_dep_levels": "uint16",
    }
    specs: dict[str, dict[str, object]] = {}
    for name, values in rows.items():
        item_count = int(values.shape[0])
        offsets_path = prefix.with_name(f"{prefix.name}_{name}_offsets.bin")
        data_path = prefix.with_name(f"{prefix.name}_{name}_data.bin")
        np.asarray([0, item_count], dtype=np.int64).tofile(offsets_path)
        values.tofile(data_path)
        if name in {"token_call_edges", "token_type_edges"}:
            kind = "edge_pairs"
            shape_tail = [2]
            coordinate_space = "chunk_index"
        elif name.endswith("_edges"):
            kind = "edge_triples"
            shape_tail = [3]
            coordinate_space = "token_index"
        else:
            kind = "ragged_1d"
            shape_tail = [1]
            coordinate_space = (
                "chunk_index"
                if name in {"token_chunk_kinds", "token_chunk_dep_levels"}
                else "token_index"
            )
        specs[name] = {
            "kind": kind,
            "offsets_path": offsets_path.name,
            "data_path": data_path.name,
            "offset_dtype": "int64",
            "dtype": dtypes[name],
            "item_count": item_count,
            "shape_tail": shape_tail,
            "coordinate_space": coordinate_space,
        }
    return specs


def _build_bundle(
    tmp_path: Path, *, bounded_source_sampling: bool = False
) -> _BundleFixture:
    root = tmp_path / "bundle"
    prefix = root / _PREFIX_RELATIVE
    prefix.parent.mkdir(parents=True)
    tokens = np.asarray([2, 3, 5, 7, 11, 13, 17, 19], dtype=np.int32)
    _write_mmididx(prefix, tokens)
    sidecar_specs, sidecar_paths = _write_token_sidecars(prefix)
    graph_specs = _write_graph_sidecars(prefix)

    registry_path = prefix.with_name(f"{prefix.name}_source_identity.sqlite")
    registry_path.write_bytes(b"verified-registry-fixture")
    objective_ids_path = prefix.with_name(f"{prefix.name}_objective_ids.bin")
    objective_ids_path.write_bytes(b"\x01")
    sequence_platform_path = prefix.with_name(
        f"{prefix.name}_source_platform_sequence_offsets.bin"
    )
    document_platform_path = prefix.with_name(
        f"{prefix.name}_source_platform_document_offsets.bin"
    )
    platform_ids_path = prefix.with_name(f"{prefix.name}_source_platform_ids.bin")
    np.asarray([0, 1], dtype=np.int64).tofile(sequence_platform_path)
    np.asarray([0, 1], dtype=np.int64).tofile(document_platform_path)
    np.asarray([1], dtype=np.uint16).tofile(platform_ids_path)

    source_bytes = b"src"
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    source_sampling: dict[str, object] = {
        "mode": "deterministic_epoch_shuffle_v1",
        "seed": 7,
        "requested_samples": 1,
        "full_passes": 1,
        "tail_rows": 0,
        "min_row_reuse": 1,
        "max_row_reuse": 1,
    }
    if bounded_source_sampling:
        source_sampling.update(
            {
                "mode": "deterministic_shard_row_group_record_batch_shuffle_v2",
                "record_batch_rows": 64,
                "producer": {
                    "name": "pyarrow.parquet.ParquetFile.iter_batches",
                    "version": 1,
                    "row_group_rows": [[1]],
                },
                "ordering": {
                    "permutation": "sha256_sort_key_v1",
                    "epochs": "ascending",
                    "shards": "seeded_permutation_per_epoch",
                    "row_groups": "seeded_permutation_per_shard_epoch",
                    "record_batches": "physical_order_within_row_group",
                    "rows": "seeded_permutation_within_record_batch",
                },
                "cursor_semantics": "last_yielded_row_v1",
                "final_cursor": {
                    "epoch": 0,
                    "shard_position": 0,
                    "shard_index": 0,
                    "row_group_position": 0,
                    "row_group_index": 0,
                    "record_batch_index": 0,
                    "row_shuffle_position": 0,
                    "row_index_in_record_batch": 0,
                    "source_index": 0,
                },
            }
        )
    source_snapshot = {
        "schema": "cppmega_objective_source_snapshot_v1",
        "sequence_length": _BUCKET,
        "file_count": 1,
        "row_count": 1,
        "files": [
            {
                "path": "/immutable-source/code_8.parquet",
                "size_bytes": len(source_bytes),
                "sha256": source_digest,
                "rows": 1,
            }
        ],
        "sampling": source_sampling,
    }
    source_snapshot["artifact_set_sha256"] = _artifact_set_sha256(
        [
            {
                "path": source_snapshot["files"][0]["path"],
                "size": len(source_bytes),
                "sha256": source_digest,
            }
        ]
    )
    contract = {
        "schema": "cppmega_pre_materialized_objectives_v1",
        "totals": {"samples": 1, "input_tokens": 7, "loss_tokens": 7},
        "materialization": {
            "format": "shifted_lm_document_v1",
            "loss_mask_alignment": "source_token_predicts_next_v1",
        },
        "graph_auxiliary": {
            **stage1_graph_recipe_payload(),
            "recipe": stage1_graph_recipe_binding(),
        },
        "source_snapshot": source_snapshot,
    }
    provenance = root / "provenance"
    contract_path = provenance / "objective_contract_seq8.json"
    _write_json(contract_path, contract)
    contract_sha256 = _canonical_sha256(contract)
    parquet_binding = {
        "path": "materialized_00000.parquet",
        "size_bytes": 1,
        "sha256": hashlib.sha256(b"p").hexdigest(),
    }
    artifact: dict[str, Any] = {
        "schema": "cppmega_objective_materialization_artifact_v2",
        "graph_recipe": stage1_graph_recipe_binding(),
        "documents": 1,
        "objective_contract": {
            "path": "objective_contract.json",
            "sha256": contract_sha256,
            "size_bytes": contract_path.stat().st_size,
            "file_sha256": _sha256(contract_path),
        },
        "parquet_shards": [parquet_binding],
        "converter": {
            "source_platform_sidecar": "require",
            "loss_mask_alignment": "source_token_predicts_next_v1",
        },
    }
    artifact["artifact_set_sha256"] = _canonical_sha256(artifact)
    artifact_path = provenance / "objective_artifact_seq8.json"
    _write_json(artifact_path, artifact)

    source_manifest = {
        "schema": "cppmega_parquet_snapshot_v1",
        "file_count": 1,
        "files": [
            {
                "kind": "ci",
                "bucket": _BUCKET,
                "source": (
                    f"/immutable-source/{_BUCKET}/"
                    f"ci-case5-train-{_BUCKET}-000000.parquet"
                ),
                "snapshot": (f"ci/{_BUCKET}/ci-case5-train-{_BUCKET}-000000.parquet"),
                "size": len(source_bytes),
                "mtime_ns": 1,
                "sha256": source_digest,
            }
        ],
    }
    repaired_manifest = {
        "schema": "cppmega_repaired_parquet_snapshot_v1",
        "file_count": 1,
        "changed_files": 0,
        "files": [
            {
                "kind": "ci",
                "bucket": _BUCKET,
                "snapshot": (f"ci/{_BUCKET}/ci-case5-train-{_BUCKET}-000000.parquet"),
                "size": len(source_bytes),
                "source_sha256": source_digest,
                "snapshot_sha256": source_digest,
                "boundary_repaired": False,
            }
        ],
    }
    source_manifest_path = provenance / "source_manifest.json"
    repaired_manifest_path = provenance / "repaired_snapshot_manifest.json"
    _write_json(source_manifest_path, source_manifest)
    _write_json(repaired_manifest_path, repaired_manifest)
    ci_manifest = _write_ci_production_provenance(
        root,
        source_size=len(source_bytes),
        source_sha256=source_digest,
    )

    tokenizer_root = root / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer_contract_v1.json").write_bytes(
        TOKENIZER_CONTRACT_PATH.read_bytes()
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        _write_json(tokenizer_root / name, {"fixture": name})
    contracts_root = root / "contracts"
    contracts_root.mkdir()
    domain_contract = contracts_root / "domain_schema_v1.json"
    domain_contract.write_bytes(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("cppmega_mlx/data/domain_schema_v1.json")
        .read_bytes()
    )
    tokenizer_contract = contracts_root / "tokenizer_contract_v1.json"
    tokenizer_contract.write_bytes(TOKENIZER_CONTRACT_PATH.read_bytes())

    objective_descriptor = {
        "artifact_path": artifact_path.relative_to(root).as_posix(),
        "artifact_schema": "cppmega_objective_materialization_artifact_v2",
        "artifact_set_sha256": artifact["artifact_set_sha256"],
        "artifact_file_sha256": _sha256(artifact_path),
        "contract_path": contract_path.relative_to(root).as_posix(),
        "contract_schema": "cppmega_pre_materialized_objectives_v1",
        "contract_sha256": contract_sha256,
        "contract_file_sha256": _sha256(contract_path),
        "source_snapshot": {
            "schema": "cppmega_objective_source_snapshot_v1",
            "artifact_set_sha256": source_snapshot["artifact_set_sha256"],
            "file_count": 1,
            "row_count": 1,
            "sampling": source_snapshot["sampling"],
        },
    }
    sidecar = {
        "source_format": "megatron-objective-materialized",
        "dtype": "int32",
        "token_count": _BUCKET,
        "document_count": 1,
        "vocab_size": 65536,
        "tokenizer_contract": "megacpp",
        "symbol_identity_schema_version": 3,
        "loss_mask_alignment": "source_token_predicts_next_v1",
        "side_channel_paths": sidecar_specs,
        "graph_sidecar_schema": "cppmega_graph_routes_v2",
        "graph_sidecar_paths": graph_specs,
        "case5_domain_ingestion_receipt": {
            "schema": "case5_domain_routes_v1",
            "status": "success",
            "delimiter_contract_sha256": DOMAIN_DELIMITER_CONTRACT_SHA256,
            "domain_schema_sha256": DOMAIN_SCHEMA_SHA256,
            "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
            "domain_route_columns": list(_DOMAIN_ROUTE_COLUMNS),
            "graph_route_columns": list(_GRAPH_ROUTE_COLUMNS),
            "graph_sidecars_written": True,
            "source_identity_registry_schema": ("cppmega_source_identity_registry_v1"),
        },
        "source_identity_registry": {
            "schema": "cppmega_source_identity_registry_v1",
            "id_encoding": "uint64_be",
            "canonical_digest": "sha256",
            "sequence_count": 1,
            "token_foreign_key_sidecar": "token_source_identity_ids",
            "path": registry_path.name,
        },
        "source_platform_sidecar": {
            "schema": "cppmega_source_platform_v1",
            "source_document_count": 1,
            "platform_id_count": 1,
            "sequence_doc_offsets_path": sequence_platform_path.name,
            "doc_platform_offsets_path": document_platform_path.name,
            "platform_ids_path": platform_ids_path.name,
        },
        "objective_contract": {
            "schema": "cppmega_pre_materialized_objectives_v1",
            "sha256": contract_sha256,
            "payload": contract,
            "objective_id_sidecar": {
                "path": objective_ids_path.name,
                "dtype": "uint8",
                "document_aligned": True,
            },
        },
        "objective_materialization": {
            **artifact,
            "artifact_file_sha256": _sha256(artifact_path),
        },
    }
    _write_json(prefix.with_suffix(".json"), sidecar)

    tokenizer_records = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(tokenizer_root.iterdir())
    ]
    source_composition = _write_source_composition_provenance(root)
    artifacts = _artifact_records(root)
    artifact_set_sha256 = _artifact_set_sha256(artifacts)
    bundle_id = f"fixture-bundle-{artifact_set_sha256[:16]}"
    manifest = {
        "schema": "cppmega_megatron_bundle_v3",
        "bundle_id": bundle_id,
        "implementation": _producer_implementation(),
        "tokenizer_contract": "megacpp-vocab-65536",
        "vocab_size": 65536,
        "token_column": "input_ids",
        "length_column": "valid_token_count",
        "writer_backend": "mmididx",
        "tokenizer": {
            "path": "tokenizer",
            "contract": "megacpp-vocab-65536",
            "vocab_size": 65536,
            "files": tokenizer_records,
            "artifact_set_sha256": _artifact_set_sha256(tokenizer_records),
        },
        "data_contracts": {
            "domain_schema": {
                "path": domain_contract.relative_to(root).as_posix(),
                "size": domain_contract.stat().st_size,
                "sha256": _sha256(domain_contract),
            },
            "tokenizer_contract": {
                "path": tokenizer_contract.relative_to(root).as_posix(),
                "size": tokenizer_contract.stat().st_size,
                "sha256": _sha256(tokenizer_contract),
            },
        },
        "training_contract": "objective_materialized",
        "known_limitations": [],
        "objective_materialization": {
            "schema": "cppmega_bucketed_objective_materializations_v1",
            "buckets": {str(_BUCKET): objective_descriptor},
        },
        "source_snapshot": {
            "file_count": 1,
            "manifest": source_manifest_path.relative_to(root).as_posix(),
            "repaired_manifest": repaired_manifest_path.relative_to(root).as_posix(),
            "local_snapshot_retained": False,
            "source_composition": source_composition,
            "ci_manifest": ci_manifest,
        },
        "buckets": [_BUCKET],
        "bucket_results": [
            {"bucket": _BUCKET, "prefix": _PREFIX_RELATIVE, "manifest": sidecar}
        ],
        "artifacts": artifacts,
        "artifact_set_sha256": artifact_set_sha256,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(int(record["size"]) for record in artifacts),
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    prefix_manifest_path = prefix.with_suffix(".json")
    restore_receipt = root / "restore_receipt.json"
    binding = {
        "schema": "cppmega_case6_receipt_binding_v2",
        "bundle_id": bundle_id,
        "artifact_set_sha256": artifact_set_sha256,
        "prefix_manifest_sha256s": {
            prefix_manifest_path.relative_to(root).as_posix(): _sha256(
                prefix_manifest_path
            )
        },
        "checkpoint_sha256": hashlib.sha256(b"cppmega:no-checkpoint:v1").hexdigest(),
        "config_sha256": hashlib.sha256(b"config").hexdigest(),
        "command_sha256": hashlib.sha256(b"command").hexdigest(),
        "run_id": "fixture-restore",
        "implementation": _training_implementation(),
    }
    _write_json(
        restore_receipt,
        {
            "schema": "cppmega_megatron_restore_receipt_v1",
            "status": "restored_verified",
            "bundle_id": bundle_id,
            "transport": (
                f"s3://{_STORAGE_BUCKET}/cppmega/transports/{bundle_id}/transport.json"
            ),
            "transport_sha256": hashlib.sha256(b"transport").hexdigest(),
            "logical_manifest_sha256": _sha256(manifest_path),
            "artifact_set_sha256": artifact_set_sha256,
            "artifact_count": len(artifacts),
            "artifact_bytes": sum(int(record["size"]) for record in artifacts),
            "binding": binding,
            "restored_at": "2026-07-14T00:00:00+00:00",
        },
    )
    return _BundleFixture(
        root=root,
        bundle_id=bundle_id,
        prefix=prefix,
        restore_receipt=restore_receipt,
        sidecar_paths=sidecar_paths,
    )


def _open(fixture: _BundleFixture):
    return open_production_megatron_bundle(
        fixture.root,
        _BUCKET,
        fixture.bundle_id,
        restore_receipt=fixture.restore_receipt,
        seq_len=_BUCKET,
        batch_size=1,
        hash_jobs=2,
    )


def _forbid_mmap(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            f"mmap opened before validation: args={args}, kwargs={kwargs}"
        )

    monkeypatch.setattr(np, "memmap", forbidden)


def _rewrite_ci_manifest_for_validation(
    fixture: _BundleFixture,
    update: Callable[[dict[str, Any]], None],
) -> tuple[dict[str, Any], dict[str, dict[str, object]]]:
    manifest = json.loads((fixture.root / "manifest.json").read_text(encoding="utf-8"))
    descriptor = manifest["source_snapshot"]["ci_manifest"]
    path = fixture.root / descriptor["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    _write_json(path, payload)
    descriptor["sha256"] = _sha256(path)
    artifacts = {str(record["path"]): dict(record) for record in manifest["artifacts"]}
    artifacts[descriptor["path"]]["size"] = path.stat().st_size
    artifacts[descriptor["path"]]["sha256"] = descriptor["sha256"]
    return manifest, artifacts


@pytest.mark.mlx_runtime
def test_open_production_megatron_bundle_records_validated_provenance(
    tmp_path: Path,
) -> None:
    fixture = _build_bundle(tmp_path)

    dataset = _open(fixture)
    batch = next(dataset.iter_batches(loop=False))

    assert isinstance(dataset.metadata, ProductionMegatronDatasetMetadata)
    assert dataset.metadata.bundle_id == fixture.bundle_id
    assert dataset.metadata.bucket == _BUCKET
    assert dataset.metadata.storage_bucket == _STORAGE_BUCKET
    assert dataset.metadata.training_contract == "objective_materialized"
    assert dataset.metadata.source_snapshot_artifact_set_sha256
    source_composition = json.loads(
        (fixture.root / "manifest.json").read_text(encoding="utf-8")
    )["source_snapshot"]["source_composition"]
    source_receipt = fixture.root / source_composition["receipt"]["path"]
    dedup_receipt = fixture.root / source_composition["dedup_receipt"]["path"]
    source_payload = json.loads(source_receipt.read_text(encoding="utf-8"))
    dedup_payload = json.loads(dedup_receipt.read_text(encoding="utf-8"))
    assert dataset.metadata.source_composition_receipt_sha256 == _sha256(source_receipt)
    assert (
        dataset.metadata.source_producer_set_sha256
        == source_payload["source_producer_set_sha256"]
    )
    assert dataset.metadata.global_dedup_receipt_sha256 == _sha256(dedup_receipt)
    assert (
        dataset.metadata.global_dedup_logical_sha256 == dedup_payload["logical_sha256"]
    )
    ci_descriptor = json.loads(
        (fixture.root / "manifest.json").read_text(encoding="utf-8")
    )["source_snapshot"]["ci_manifest"]
    assert dataset.metadata.ci_manifest_sha256 == ci_descriptor["sha256"]
    assert (
        dataset.metadata.ci_source_inventory_sha256
        == ci_descriptor["source_inventory_sha256"]
    )
    assert dataset.metadata.ci_acquisition_provenance_sha256
    assert dataset.metadata.ci_exact_unique_payload_tokens == 20_000_000_123
    assert dataset.metadata.objective_contract_sha256
    assert dataset.metadata.objective_artifact_set_sha256
    assert dataset.metadata.tokenizer_contract_sha256 == TOKENIZER_CONTRACT_SHA256
    assert dataset.metadata.domain_schema_sha256 == DOMAIN_SCHEMA_SHA256
    assert tuple(batch.tokens.shape) == (1, _BUCKET)
    assert batch.document_ids is not None
    assert batch.graph_batch is not None


def test_ci_ingress_rejects_legacy_fetch_state(tmp_path: Path) -> None:
    fixture = _build_bundle(tmp_path)
    manifest, artifacts = _rewrite_ci_manifest_for_validation(
        fixture,
        lambda payload: payload["input_fetch_state"].update(
            {"schema": "cppmega_ci_stream_fetch_v3"}
        ),
    )

    with pytest.raises(ValueError, match="input store/fetch contract drifted"):
        production_bundle._validate_ci_production_acquisition(
            fixture.root, manifest, artifacts
        )


def test_ci_ingress_rejects_legacy_export_schema(tmp_path: Path) -> None:
    fixture = _build_bundle(tmp_path)
    manifest, artifacts = _rewrite_ci_manifest_for_validation(
        fixture,
        lambda payload: payload.update(
            {"schema": "cppmega_ci_content_store_case5_export_v3"}
        ),
    )

    with pytest.raises(ValueError, match="inventory-exhaustive CI export v4"):
        production_bundle._validate_ci_production_acquisition(
            fixture.root, manifest, artifacts
        )


def test_ci_ingress_rejects_non_zstd_case5_contract(tmp_path: Path) -> None:
    fixture = _build_bundle(tmp_path)
    manifest, artifacts = _rewrite_ci_manifest_for_validation(
        fixture,
        lambda payload: payload["case5_contract"].update(
            {"parquet_compression": {"codec": "none", "level": 0}}
        ),
    )

    with pytest.raises(ValueError, match="CASE5 bucket contract drifted"):
        production_bundle._validate_ci_production_acquisition(
            fixture.root, manifest, artifacts
        )


def test_ci_ingress_requires_receipt_bound_manifest(tmp_path: Path) -> None:
    fixture = _build_bundle(tmp_path)
    manifest = json.loads(
        (fixture.root / "manifest.json").read_text(encoding="utf-8")
    )
    del manifest["source_snapshot"]["ci_manifest"]
    artifacts = {
        str(record["path"]): dict(record) for record in manifest["artifacts"]
    }

    with pytest.raises(ValueError, match="source_snapshot.ci_manifest"):
        production_bundle._validate_ci_production_acquisition(
            fixture.root, manifest, artifacts
        )


def test_ci_ingress_rejects_less_than_twenty_billion_tokens(
    tmp_path: Path,
) -> None:
    fixture = _build_bundle(tmp_path)
    manifest, artifacts = _rewrite_ci_manifest_for_validation(
        fixture,
        lambda payload: payload["eligibility"].update(
            {"target_exact_unique_payload_tokens": 19_999_999_999}
        ),
    )

    with pytest.raises(ValueError, match="exact-token or aggregate accounting"):
        production_bundle._validate_ci_production_acquisition(
            fixture.root, manifest, artifacts
        )


def test_ci_ingress_rejects_legacy_merge_receipt(tmp_path: Path) -> None:
    fixture = _build_bundle(tmp_path)
    manifest, artifacts = _rewrite_ci_manifest_for_validation(
        fixture,
        lambda payload: payload["acquisition_provenance"]["merge"].update(
            {"schema": "cppmega_ci_stream_shard_union_receipt_v2"}
        ),
    )

    with pytest.raises(ValueError, match="proof chain is not mutually bound"):
        production_bundle._validate_ci_production_acquisition(
            fixture.root, manifest, artifacts
        )


def test_ci_ingress_rejects_unbound_parquet_inventory(tmp_path: Path) -> None:
    fixture = _build_bundle(tmp_path)

    def replace_parquet_digest(payload: dict[str, Any]) -> None:
        parquet = next(
            record
            for record in payload["artifacts"]
            if record["kind"] == "case5_parquet"
        )
        parquet["sha256"] = "f" * 64

    manifest, artifacts = _rewrite_ci_manifest_for_validation(
        fixture, replace_parquet_digest
    )

    with pytest.raises(ValueError, match="differs from source snapshot"):
        production_bundle._validate_ci_production_acquisition(
            fixture.root, manifest, artifacts
        )


def test_legacy_objective_artifact_shape_requires_regeneration() -> None:
    with pytest.raises(ValueError, match="legacy.*migration required.*regenerate"):
        production_bundle._validate_objective_artifact_shape(
            {"schema": "cppmega_objective_materialization_artifact_v1"},
            bucket=_BUCKET,
        )


@pytest.mark.parametrize(
    "legacy_schema",
    ["cppmega_megatron_bundle_v1", "cppmega_megatron_bundle_v2"],
)
def test_legacy_bundle_schema_is_rejected(
    tmp_path: Path,
    legacy_schema: str,
) -> None:
    fixture = _build_bundle(tmp_path)
    manifest_path = fixture.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = legacy_schema

    with pytest.raises(ValueError, match="unsupported production bundle schema"):
        production_bundle._validate_logical_manifest(
            fixture.root,
            manifest,
            expected_bundle_id=fixture.bundle_id,
            bucket=_BUCKET,
        )


def test_bundle_without_source_composition_is_rejected(tmp_path: Path) -> None:
    fixture = _build_bundle(tmp_path)
    manifest = json.loads((fixture.root / "manifest.json").read_text(encoding="utf-8"))
    del manifest["source_snapshot"]["source_composition"]

    with pytest.raises(ValueError, match="source_snapshot.source_composition"):
        production_bundle._validate_logical_manifest(
            fixture.root,
            manifest,
            expected_bundle_id=fixture.bundle_id,
            bucket=_BUCKET,
        )


def test_incomplete_source_composition_coverage_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _build_bundle(tmp_path)
    manifest = json.loads((fixture.root / "manifest.json").read_text(encoding="utf-8"))
    descriptor = manifest["source_snapshot"]["source_composition"]
    receipt_path = fixture.root / descriptor["receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["coverage"]["code_success_repositories"] = 0
    _write_json(receipt_path, receipt)
    descriptor["receipt"]["sha256"] = _sha256(receipt_path)

    with pytest.raises(ValueError, match="full repository coverage"):
        production_bundle._validate_source_composition(
            fixture.root,
            manifest,
            {str(record["path"]): record for record in manifest["artifacts"]},
        )


def test_weakened_source_composition_near_dedup_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _build_bundle(tmp_path)
    manifest = json.loads((fixture.root / "manifest.json").read_text(encoding="utf-8"))
    descriptor = manifest["source_snapshot"]["source_composition"]
    dedup_path = fixture.root / descriptor["dedup_receipt"]["path"]
    dedup = json.loads(dedup_path.read_text(encoding="utf-8"))
    dedup["policy"]["near"]["enabled"] = False
    _write_json(dedup_path, dedup)
    descriptor["dedup_receipt"]["sha256"] = _sha256(dedup_path)

    receipt_path = fixture.root / descriptor["receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    portable_dedup = dict(dedup)
    portable_database = dict(dedup["database"])
    portable_database.pop("path")
    portable_dedup["database"] = portable_database
    portable_dedup["receipt_sha256"] = _sha256(dedup_path)
    receipt["dedup"] = portable_dedup
    _write_json(receipt_path, receipt)
    descriptor["receipt"]["sha256"] = _sha256(receipt_path)

    with pytest.raises(ValueError, match=r"exact\+near policy drifted"):
        production_bundle._validate_source_composition(
            fixture.root,
            manifest,
            {str(record["path"]): record for record in manifest["artifacts"]},
        )


def test_nonempty_bundle_known_limitations_are_rejected(tmp_path: Path) -> None:
    fixture = _build_bundle(tmp_path)
    manifest = json.loads((fixture.root / "manifest.json").read_text(encoding="utf-8"))
    manifest["known_limitations"] = ["source coverage incomplete"]

    with pytest.raises(ValueError, match="known_limitations must be empty"):
        production_bundle._validate_logical_manifest(
            fixture.root,
            manifest,
            expected_bundle_id=fixture.bundle_id,
            bucket=_BUCKET,
        )


def test_bundle_without_implementation_is_rejected(tmp_path: Path) -> None:
    fixture = _build_bundle(tmp_path)
    manifest_path = fixture.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["implementation"]

    with pytest.raises(ValueError, match="implementation binding"):
        production_bundle._validate_logical_manifest(
            fixture.root,
            manifest,
            expected_bundle_id=fixture.bundle_id,
            bucket=_BUCKET,
        )


def test_restore_implementation_must_extend_bundle_producer() -> None:
    training = _training_implementation()
    training["components"]["cppmega"]["commit"] = "8" * 40

    with pytest.raises(
        ValueError,
        match="implementation does not extend the bundle producer for cppmega",
    ):
        production_bundle._validate_training_implementation_extension(
            _producer_implementation(),
            training,
        )


def test_bundle_implementation_rejects_training_component(tmp_path: Path) -> None:
    fixture = _build_bundle(tmp_path)
    manifest = json.loads(
        (fixture.root / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["implementation"]["components"]["megatron"] = {"commit": "7" * 40}

    with pytest.raises(ValueError, match=r"components drifted: .*extra=\['megatron'\]"):
        production_bundle._validate_logical_manifest(
            fixture.root,
            manifest,
            expected_bundle_id=fixture.bundle_id,
            bucket=_BUCKET,
        )


def test_bundle_implementation_rejects_extra_component_field() -> None:
    producer = _producer_implementation()
    producer["components"]["cppmega"]["source_sha256"] = "8" * 64

    with pytest.raises(
        ValueError,
        match=r"component cppmega fields drifted: .*extra=\['source_sha256'\]",
    ):
        production_bundle._validate_implementation_binding(
            producer,
            where="production bundle",
            required_components=production_bundle._PRODUCER_IMPLEMENTATION_COMPONENTS,
        )


@pytest.mark.mlx_runtime
def test_open_production_megatron_bundle_accepts_bounded_source_sampling(
    tmp_path: Path,
) -> None:
    fixture = _build_bundle(tmp_path, bounded_source_sampling=True)

    dataset = _open(fixture)
    batch = next(dataset.iter_batches(loop=False))

    assert tuple(batch.tokens.shape) == (1, _BUCKET)
    assert batch.graph_batch is not None


def test_bounded_source_sampling_producer_is_fully_bound() -> None:
    sampling = {
        "mode": "deterministic_shard_row_group_record_batch_shuffle_v2",
        "seed": 7,
        "requested_samples": 3,
        "full_passes": 1,
        "tail_rows": 1,
        "min_row_reuse": 1,
        "max_row_reuse": 2,
        "record_batch_rows": 2,
        "producer": {
            "name": "pyarrow.parquet.ParquetFile.iter_batches",
            "version": 1,
            "row_group_rows": [[2]],
        },
        "ordering": dict(production_bundle._BOUNDED_SOURCE_ORDERING),
        "cursor_semantics": "last_yielded_row_v1",
        "final_cursor": {
            "epoch": 1,
            "shard_position": 0,
            "shard_index": 0,
            "row_group_position": 0,
            "row_group_index": 0,
            "record_batch_index": 0,
            "row_shuffle_position": 0,
            "row_index_in_record_batch": 0,
            "source_index": 2,
        },
    }

    validated = production_bundle._validate_objective_source_sampling(
        sampling,
        row_count=2,
        file_count=1,
        file_row_counts=[2],
        bucket=8,
    )
    assert validated["producer"] == sampling["producer"]

    for mutation, message in (
        ({"producer": None}, "producer"),
        ({"producer": {**sampling["producer"], "version": 2}}, "producer drifted"),
        (
            {
                "producer": {
                    **sampling["producer"],
                    "row_group_rows": [[1]],
                }
            },
            "row count drifted",
        ),
    ):
        candidate = {**sampling, **mutation}
        with pytest.raises((TypeError, ValueError), match=message):
            production_bundle._validate_objective_source_sampling(
                candidate,
                row_count=2,
                file_count=1,
                file_row_counts=[2],
                bucket=8,
            )


def test_one_byte_mutation_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_bundle(tmp_path)
    payload = bytearray(fixture.prefix.with_suffix(".bin").read_bytes())
    payload[0] ^= 1
    fixture.prefix.with_suffix(".bin").write_bytes(payload)
    _forbid_mmap(monkeypatch)

    with pytest.raises(ValueError, match="artifact sha256 mismatch"):
        _open(fixture)


@pytest.mark.parametrize("wrong_bucket", [4, 16])
def test_wrong_sequence_bucket_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong_bucket: int
) -> None:
    fixture = _build_bundle(tmp_path)
    _forbid_mmap(monkeypatch)

    with pytest.raises(ValueError, match="requested bucket"):
        open_production_megatron_bundle(
            fixture.root,
            wrong_bucket,
            fixture.bundle_id,
            restore_receipt=fixture.restore_receipt,
            seq_len=wrong_bucket,
            batch_size=1,
        )


def test_wrong_bundle_id_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_bundle(tmp_path)
    _forbid_mmap(monkeypatch)

    with pytest.raises(ValueError, match="bundle_id mismatch"):
        open_production_megatron_bundle(
            fixture.root,
            _BUCKET,
            "different-bundle-0000000000000000",
            restore_receipt=fixture.restore_receipt,
            seq_len=_BUCKET,
            batch_size=1,
        )


def test_unlisted_index_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_bundle(tmp_path)
    (fixture.root / "data/unlisted.idx").write_bytes(b"unlisted")
    _forbid_mmap(monkeypatch)

    with pytest.raises(ValueError, match="unlisted relevant artifact"):
        _open(fixture)


def test_missing_sidecar_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_bundle(tmp_path)
    fixture.sidecar_paths["token_domain_ids"].unlink()
    _forbid_mmap(monkeypatch)

    with pytest.raises(FileNotFoundError):
        _open(fixture)


def test_mixed_sidecar_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_bundle(tmp_path)
    fixture.sidecar_paths["token_domain_ids"].write_bytes(
        fixture.sidecar_paths["token_role_ids"].read_bytes()
    )
    _forbid_mmap(monkeypatch)

    with pytest.raises(ValueError, match="artifact sha256 mismatch"):
        _open(fixture)


def test_stale_source_snapshot_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_bundle(tmp_path)
    repaired = fixture.root / "provenance/repaired_snapshot_manifest.json"
    payload = json.loads(repaired.read_text(encoding="utf-8"))
    payload["files"][0]["snapshot_sha256"] = hashlib.sha256(b"stale").hexdigest()
    _write_json(repaired, payload)
    _forbid_mmap(monkeypatch)

    with pytest.raises(ValueError, match="artifact sha256 mismatch"):
        _open(fixture)


def test_stale_objective_contract_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_bundle(tmp_path)
    contract = fixture.root / "provenance/objective_contract_seq8.json"
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["totals"]["loss_tokens"] = 6
    _write_json(contract, payload)
    _forbid_mmap(monkeypatch)

    with pytest.raises(ValueError, match="artifact sha256 mismatch"):
        _open(fixture)


def test_missing_restore_receipt_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_bundle(tmp_path)
    _forbid_mmap(monkeypatch)

    with pytest.raises(ValueError, match="requires a restore receipt"):
        open_production_megatron_bundle(
            fixture.root,
            _BUCKET,
            fixture.bundle_id,
            restore_receipt=None,
            seq_len=_BUCKET,
            batch_size=1,
        )


def test_restore_receipt_prefix_binding_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_bundle(tmp_path)
    receipt = json.loads(fixture.restore_receipt.read_text(encoding="utf-8"))
    prefix_hashes = receipt["binding"]["prefix_manifest_sha256s"]
    prefix_hashes[next(iter(prefix_hashes))] = hashlib.sha256(b"stale").hexdigest()
    _write_json(fixture.restore_receipt, receipt)
    _forbid_mmap(monkeypatch)

    with pytest.raises(ValueError, match="prefix-manifest binding"):
        _open(fixture)


def test_symlinked_artifact_fails_before_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_bundle(tmp_path)
    target = fixture.sidecar_paths["token_domain_ids"]
    replacement = target.with_name("domain-copy")
    replacement.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(replacement.name)
    _forbid_mmap(monkeypatch)

    with pytest.raises(ValueError, match="symlink"):
        _open(fixture)
