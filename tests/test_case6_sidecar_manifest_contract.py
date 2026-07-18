from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.check_environment import readiness_failures
from scripts.sidecar_manifest_contract import (
    AUDIT_FILENAME,
    AUDIT_REMOTE,
    AUDIT_SCHEMA,
    GRAPH_CONTRACT,
    OBJECTIVE_CONTRACT,
    audit_contract,
    finalize_manifest,
    inventory_directory,
    inventory_sha256,
    resolve_s3_env,
    selection_policy,
    validate_audit_receipt,
    validate_manifest,
    verify_inventory,
)


PARQUET_REMOTES = [
    f"parquet/{kind}/{bucket}"
    for kind in ("code", "commits", "pr")
    for bucket in (1024, 2048, 4096, 8192, 16384)
    if not (kind == "code" and bucket == 16384)
    and not (kind == "pr")
]


def _green_manifest(tmp_path: Path) -> tuple[dict, Path]:
    selections: list[dict[str, object]] = []
    by_kind_bucket: dict[str, dict[str, int]] = {}
    for index, remote in enumerate(PARQUET_REMOTES, 1):
        local = tmp_path / "source" / remote.replace("/", "_")
        local.mkdir(parents=True)
        (local / f"shard-{index:03d}.parquet").write_bytes(f"shard-{index}".encode())
        inventory = inventory_directory(local, remote=remote)
        selections.append({"remote": remote, "local": local.relative_to(tmp_path).as_posix(), **inventory})
        by_kind_bucket[remote.removeprefix("parquet/")] = {
            "files": 1,
            "valid_tokens": 10,
            "bad_files": 0,
            "bad_rows": 0,
        }

    audit_dir = tmp_path / "source" / "audit"
    audit_dir.mkdir(parents=True)
    report = {
        "schema": AUDIT_SCHEMA,
        "status": "verified",
        "contract": audit_contract(),
        "total": {"bad_files": 0, "bad_rows": 0},
        "by_kind_bucket": by_kind_bucket,
    }
    audit_path = audit_dir / AUDIT_FILENAME
    audit_path.write_text(json.dumps(report), encoding="utf-8")
    audit_inventory = inventory_directory(audit_dir, remote=AUDIT_REMOTE)
    selections.append(
        {"remote": AUDIT_REMOTE, "local": audit_dir.relative_to(tmp_path).as_posix(), **audit_inventory}
    )
    selected_keys = [remote.removeprefix("parquet/") for remote in PARQUET_REMOTES]
    validate_audit_receipt(report, selected_keys=selected_keys)
    audit_record = next(
        record for record in audit_inventory["files"] if record["path"] == AUDIT_FILENAME
    )
    payload = {
        "schema": "cppmega_verified_sidecar_manifest_v2",
        "bucket": "bucket",
        "prefix": "prefix/run-1",
        "endpoint_url": "https://storage.example",
        "verified_valid_tokens": len(PARQUET_REMOTES) * 10,
        "profile": "code_commits_integrated_pr",
        "standalone_pr_included": False,
        "selections": selections,
        "inventory_sha256": inventory_sha256(selections),
        "selection_policy": selection_policy(
            [str(item["remote"]) for item in selections],
            include_standalone_pr=False,
        ),
        "audit_receipt": {
            "schema": AUDIT_SCHEMA,
            "status": "verified",
            "remote": AUDIT_REMOTE,
            "path": AUDIT_FILENAME,
            "sha256": audit_record["sha256"],
        },
        "graph_contract": GRAPH_CONTRACT,
        "objective_contract": OBJECTIVE_CONTRACT,
    }
    return finalize_manifest(payload), tmp_path


def test_manifest_binds_inventory_and_explicit_bucket_exclusions(tmp_path: Path) -> None:
    manifest, root = _green_manifest(tmp_path)
    validated = validate_manifest(
        manifest,
        expected_bucket="bucket",
        expected_prefix="prefix/run-1",
        expected_endpoint_url="https://storage.example",
    )
    verification = verify_inventory(root, validated)

    assert verification["status"] == "verified"
    assert verification["files"] == len(PARQUET_REMOTES) + 1
    excluded = validated["selection_policy"]["excluded"]
    assert {item["remote"] for item in excluded} >= {
        "parquet/code/16384",
        "parquet/pr/1024",
    }


def test_objective_manifest_is_explicitly_a_source_for_v2_materialization() -> None:
    assert OBJECTIVE_CONTRACT["schema"] == "cppmega_mlx_objective_source_v2"
    assert OBJECTIVE_CONTRACT["status"] == "source_for_pre_materialization"
    assert OBJECTIVE_CONTRACT["materialized_in_parquet"] is False
    assert OBJECTIVE_CONTRACT["materializer"] == (
        "scripts.materialize_megatron_objectives"
    )
    assert OBJECTIVE_CONTRACT["artifact_schema"] == (
        "cppmega_objective_materialization_artifact_v2"
    )
    assert "TaskMixer" not in str(OBJECTIVE_CONTRACT.get("runtime", ""))


def test_manifest_rejects_payload_tampering(tmp_path: Path) -> None:
    manifest, _root = _green_manifest(tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered["verified_valid_tokens"] += 1

    with pytest.raises(ValueError, match="payload SHA-256"):
        validate_manifest(tampered)


def test_download_inventory_rejects_extra_file(tmp_path: Path) -> None:
    manifest, root = _green_manifest(tmp_path)
    validated = validate_manifest(manifest)
    first = validated["selections"][0]
    local = root / first["local"]
    (local / "unexpected.parquet").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_inventory(root, validated)


def test_audit_receipt_must_be_schema_bound_and_green() -> None:
    with pytest.raises(ValueError, match="schema-bound"):
        validate_audit_receipt(
            {"total": {"bad_files": 0, "bad_rows": 0}, "by_kind_bucket": {}},
            selected_keys=[],
        )

    with pytest.raises(ValueError, match="contract"):
        validate_audit_receipt(
            {
                "schema": AUDIT_SCHEMA,
                "status": "verified",
                "contract": {},
                "total": {"bad_files": 0, "bad_rows": 0},
                "by_kind_bucket": {},
            },
            selected_keys=[],
        )


def test_manifest_profile_and_audit_inventory_binding_are_exact(tmp_path: Path) -> None:
    manifest, _root = _green_manifest(tmp_path)
    wrong_profile = copy.deepcopy(manifest)
    wrong_profile["profile"] = "code_commits_plus_standalone_pr"
    with pytest.raises(ValueError, match="profile"):
        validate_manifest(wrong_profile)

    wrong_audit = copy.deepcopy(manifest)
    wrong_audit["audit_receipt"]["sha256"] = "0" * 64
    wrong_audit = finalize_manifest(wrong_audit)
    with pytest.raises(ValueError, match="inventory binding"):
        validate_manifest(wrong_audit)


def test_nebius_credentials_are_atomic() -> None:
    env = resolve_s3_env(
        {
            "NEBIUS_S3_ACCESS_KEY_ID": "a",
            "NEBIUS_S3_SECRET_ACCESS_KEY": "s",
            "AWS_ACCESS_KEY_ID": "old-a",
            "AWS_SECRET_ACCESS_KEY": "old-s",
        }
    )
    assert env["AWS_ACCESS_KEY_ID"] == "a"
    with pytest.raises(SystemExit, match="complete Nebius"):
        resolve_s3_env({"NEBIUS_S3_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "s"})


def test_environment_readiness_is_explicit() -> None:
    failures = readiness_failures(
        {
            "macos": {"is_macos": True},
            "mlx": {
                "installed": True,
                "import_error": None,
                "default_device": "gpu",
            },
            "metal": {"module_present": True, "available": True},
            "file_descriptors": {"meets_recommended_min": False},
        }
    )
    assert failures == ["file descriptor soft limit is below 65536"]
