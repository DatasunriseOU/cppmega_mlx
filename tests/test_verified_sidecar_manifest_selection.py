from __future__ import annotations

from contextlib import chdir
import json
from pathlib import Path

import pytest

from tests.test_verified_sidecar_download_verification import (
    _write_green_shard as _write_case5_shard,
)
from scripts import download_verified_sidecar_from_nebius_s3 as download
from scripts import upload_verified_sidecar_to_nebius_s3 as upload
from scripts.sidecar_manifest_contract import (
    AUDIT_FILENAME,
    AUDIT_REMOTE,
    AUDIT_SCHEMA,
    GRAPH_CONTRACT,
    MANIFEST_SCHEMA,
    OBJECTIVE_CONTRACT,
    audit_contract,
    build_semantic_audit_binding,
    finalize_manifest,
    inventory_directory,
    inventory_sha256,
    selection_policy,
    validate_manifest,
)


def _remotes_from_upload(
    *,
    include_standalone_pr: bool = False,
    code_commits_only: bool = False,
) -> set[str]:
    return {
        remote
        for remote, _local in upload._selection_items(
            include_standalone_pr=include_standalone_pr,
            code_commits_only=code_commits_only,
        )
    }


def _remotes_from_download(
    *,
    include_standalone_pr: bool = False,
    code_commits_only: bool = False,
) -> set[str]:
    manifest = download._default_manifest(
        "bucket",
        "prefix",
        "https://storage.eu-north1.nebius.cloud",
        include_standalone_pr=include_standalone_pr,
        code_commits_only=code_commits_only,
    )
    return {item["remote"] for item in manifest["selections"]}


def test_upload_default_excludes_standalone_pr_diagnostics() -> None:
    remotes = _remotes_from_upload()

    assert "parquet/code/1024" in remotes
    assert "parquet/code/8192" in remotes
    assert "parquet/commits/1024" in remotes
    assert "parquet/commits/16384" in remotes
    assert not any(remote.startswith("parquet/pr/") for remote in remotes)
    assert "audits/sidecar_audit_all_final_poststop_valid" in remotes


def test_upload_standalone_pr_is_explicit_opt_in() -> None:
    remotes = _remotes_from_upload(include_standalone_pr=True)

    assert "parquet/code/8192" in remotes
    assert "parquet/commits/16384" in remotes
    assert "parquet/pr/1024" in remotes
    assert "parquet/pr/16384" in remotes


def test_download_default_manifest_matches_upload_training_policy() -> None:
    default_remotes = _remotes_from_download()
    standalone_remotes = _remotes_from_download(include_standalone_pr=True)

    assert not any(remote.startswith("parquet/pr/") for remote in default_remotes)
    assert "parquet/pr/1024" in standalone_remotes
    assert "parquet/pr/16384" in standalone_remotes


# --- verified token-total accounting (real receipt JSON, no mocks) -----------

# Distinct, easily-summed per-bucket valid_tokens keyed by the receipt's
# ``by_kind_bucket`` keys (``<kind>/<bucket>``). Code+commits=1+...; pr is large
# so a subset that wrongly reuses the all-valid total is obvious.
_BUCKET_VALID_TOKENS = {
    "code/1024": 1,
    "code/2048": 2,
    "code/4096": 4,
    "code/8192": 8,
    "commits/1024": 16,
    "commits/2048": 32,
    "commits/4096": 64,
    "commits/8192": 128,
    "commits/16384": 256,
    "pr/1024": 1_000,
    "pr/2048": 2_000,
    "pr/4096": 4_000,
    "pr/8192": 8_000,
    "pr/16384": 16_000,
}


def _write_green_shard(
    path: Path,
    *,
    bucket: int,
    valid_tokens: int,
    source_id: int,
) -> None:
    _write_case5_shard(
        path,
        length=bucket,
        source_id=source_id,
        valid_tokens=valid_tokens,
    )


def _build_receipt(
    path: Path,
    *,
    valid_tokens: dict[str, int],
    bad: dict[str, tuple[int, int]] | None = None,
    omit: set[str] | None = None,
) -> Path:
    """Write a real audit receipt JSON with a ``by_kind_bucket`` map.

    ``bad`` maps a bucket key to ``(bad_files, bad_rows)``; ``omit`` drops keys
    entirely (stale/narrow receipt). The aggregate ``total`` is kept green on
    purpose to prove the gate is per-selected-bucket, not the coarse aggregate.
    """
    bad = bad or {}
    omit = omit or set()
    by_kind_bucket: dict[str, dict[str, int]] = {}
    for key, tokens in valid_tokens.items():
        if key in omit:
            continue
        bf, br = bad.get(key, (0, 0))
        by_kind_bucket[key] = {
            "valid_tokens": tokens,
            "files": 1,
            "bad_files": bf,
            "bad_rows": br,
        }
    payload = {
        "schema": AUDIT_SCHEMA,
        "status": "verified",
        "contract": audit_contract(),
        "total": {
            "valid_tokens": sum(valid_tokens.values()),
            "bad_files": 0,
            "bad_rows": 0,
        },
        "by_kind_bucket": by_kind_bucket,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _download_fixture(
    tmp_path: Path,
    *,
    include_standalone_pr: bool = False,
    bad: dict[str, tuple[int, int]] | None = None,
) -> tuple[dict, Path]:
    """Build a complete v3 manifest and matching local inventory."""

    manifest_plan = download._default_manifest(
        "bucket",
        "prefix",
        "https://storage.eu-north1.nebius.cloud",
        include_standalone_pr=include_standalone_pr,
    )
    selected_keys = [
        item["remote"][len("parquet/"):]
        for item in manifest_plan["selections"]
        if item["remote"].startswith("parquet/")
    ]
    receipt_source = tmp_path / "receipt-source" / AUDIT_FILENAME
    receipt_source.parent.mkdir(parents=True)
    _build_receipt(receipt_source, valid_tokens=_BUCKET_VALID_TOKENS, bad=bad)

    root = tmp_path / "dl"
    selections: list[dict[str, object]] = []
    for index, item in enumerate(manifest_plan["selections"], 1):
        remote = str(item["remote"])
        local_relative = str(item["local"])
        local = root / local_relative
        local.mkdir(parents=True)
        if remote == AUDIT_REMOTE:
            (local / AUDIT_FILENAME).write_bytes(receipt_source.read_bytes())
        else:
            bucket = int(remote.rsplit("/", 1)[1])
            key = remote.removeprefix("parquet/")
            _write_green_shard(
                local / f"part-{index:03d}.parquet",
                bucket=bucket,
                valid_tokens=_BUCKET_VALID_TOKENS[key],
                source_id=index,
            )
        inventory = inventory_directory(local, remote=remote)
        selections.append({"remote": remote, "local": local_relative, **inventory})

    audit_selection = next(item for item in selections if item["remote"] == AUDIT_REMOTE)
    audit_record = next(
        record for record in audit_selection["files"] if record["path"] == AUDIT_FILENAME
    )
    token_total = sum(_BUCKET_VALID_TOKENS[key] for key in selected_keys)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "bucket": "bucket",
        "prefix": "prefix",
        "endpoint_url": "https://storage.eu-north1.nebius.cloud",
        "verified_valid_tokens": token_total,
        "profile": (
            "code_commits_plus_standalone_pr"
            if include_standalone_pr
            else "code_commits_integrated_pr"
        ),
        "standalone_pr_included": include_standalone_pr,
        "selections": selections,
        "inventory_sha256": inventory_sha256(selections),
        "selection_policy": selection_policy(
            [str(item["remote"]) for item in selections],
            include_standalone_pr=include_standalone_pr,
        ),
        "audit_receipt": {
            "schema": AUDIT_SCHEMA,
            "status": "verified",
            "remote": AUDIT_REMOTE,
            "path": AUDIT_FILENAME,
            "sha256": audit_record["sha256"],
        },
        "semantic_audit": build_semantic_audit_binding(
            selections=selections,
            audited_files=upload._audit_selected_parquet_files(
                [
                    {
                        **item,
                        "local": str(root / str(item["local"])),
                    }
                    for item in selections
                ]
            ),
            source_receipt_sha256=str(audit_record["sha256"]),
        ),
        "graph_contract": GRAPH_CONTRACT,
        "objective_contract": OBJECTIVE_CONTRACT,
    }
    return finalize_manifest(payload), root


def _populate_upload_sources(selections: tuple[tuple[str, Path], ...]) -> None:
    for index, (remote, local) in enumerate(selections, 1):
        local.mkdir(parents=True, exist_ok=True)
        if remote.startswith("parquet/"):
            key = remote.removeprefix("parquet/")
            bucket = int(remote.rsplit("/", 1)[1])
            _write_green_shard(
                local / f"part-{index:03d}.parquet",
                bucket=bucket,
                valid_tokens=_BUCKET_VALID_TOKENS[key],
                source_id=index,
            )


def test_token_total_is_profile_aware_subset(tmp_path: Path) -> None:
    receipt = _build_receipt(
        tmp_path / "sidecar_parquet_audit.json",
        valid_tokens=_BUCKET_VALID_TOKENS,
    )
    receipts = (receipt,)

    default_sel = upload._selection_items()
    standalone_sel = upload._selection_items(include_standalone_pr=True)

    default_total = upload._load_verified_token_total(receipts, default_sel)
    standalone_total = upload._load_verified_token_total(receipts, standalone_sel)

    code_commits_sum = sum(
        v for k, v in _BUCKET_VALID_TOKENS.items() if not k.startswith("pr/")
    )
    pr_sum = sum(v for k, v in _BUCKET_VALID_TOKENS.items() if k.startswith("pr/"))

    assert default_total == code_commits_sum
    assert standalone_total == sum(_BUCKET_VALID_TOKENS.values())
    assert standalone_total - default_total == pr_sum
    assert default_total < standalone_total


def test_token_total_raises_when_selected_bucket_uncovered(tmp_path: Path) -> None:
    # A stale/narrow receipt that omits a selected bucket must NOT silently pass
    # even though its aggregate total is green.
    receipt = _build_receipt(
        tmp_path / "sidecar_parquet_audit.json",
        valid_tokens=_BUCKET_VALID_TOKENS,
        omit={"commits/16384"},
    )
    cc_sel = upload._selection_items()

    with pytest.raises(SystemExit) as exc:
        upload._load_verified_token_total((receipt,), cc_sel)
    assert "commits/16384" in str(exc.value)


def test_token_total_raises_when_selected_bucket_not_green(tmp_path: Path) -> None:
    receipt = _build_receipt(
        tmp_path / "sidecar_parquet_audit.json",
        valid_tokens=_BUCKET_VALID_TOKENS,
        bad={"code/4096": (0, 3)},
    )
    cc_sel = upload._selection_items()

    with pytest.raises(SystemExit) as exc:
        upload._load_verified_token_total((receipt,), cc_sel)
    assert "code/4096" in str(exc.value)


def test_token_total_gate_is_scoped_to_selected_buckets(tmp_path: Path) -> None:
    # A non-green PR bucket must not block the default training upload, because
    # standalone PR parquet is diagnostic-only. It must block the explicit
    # standalone PR profile that includes it.
    receipt = _build_receipt(
        tmp_path / "sidecar_parquet_audit.json",
        valid_tokens=_BUCKET_VALID_TOKENS,
        bad={"pr/8192": (1, 0)},
    )
    receipts = (receipt,)

    cc_sel = upload._selection_items()
    code_commits_sum = sum(
        v for k, v in _BUCKET_VALID_TOKENS.items() if not k.startswith("pr/")
    )
    assert upload._load_verified_token_total(receipts, cc_sel) == code_commits_sum

    all_sel = upload._selection_items(include_standalone_pr=True)
    with pytest.raises(SystemExit) as exc:
        upload._load_verified_token_total(receipts, all_sel)
    assert "pr/8192" in str(exc.value)


def test_token_total_raises_on_ambiguous_multi_receipt_coverage(tmp_path: Path) -> None:
    # Same bucket covered by two receipts would be double-counted; fail loud.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    r1 = _build_receipt(
        tmp_path / "a" / "sidecar_parquet_audit.json",
        valid_tokens=_BUCKET_VALID_TOKENS,
    )
    r2 = _build_receipt(
        tmp_path / "b" / "sidecar_parquet_audit.json",
        valid_tokens=_BUCKET_VALID_TOKENS,
    )
    cc_sel = upload._selection_items()
    with pytest.raises(SystemExit) as exc:
        upload._load_verified_token_total((r1, r2), cc_sel)
    assert "multiple receipts" in str(exc.value)


# --- producer/consumer round-trip invariant ---------------------------------

def test_upload_download_remote_sets_match_for_both_profiles() -> None:
    """The uploader's selection set and the downloader's default-manifest set
    must be identical for BOTH profiles, or produced and consumed bucket sets
    silently diverge.
    """
    assert _remotes_from_upload() == _remotes_from_download()
    assert _remotes_from_upload(
        include_standalone_pr=True
    ) == _remotes_from_download(
        include_standalone_pr=True
    )
    only_in_standalone_profile = _remotes_from_upload(
        include_standalone_pr=True
    ) - _remotes_from_upload()
    assert only_in_standalone_profile == {
        "parquet/pr/1024",
        "parquet/pr/2048",
        "parquet/pr/4096",
        "parquet/pr/8192",
        "parquet/pr/16384",
    }


def test_legacy_code_commits_only_alias_matches_default_training_profile() -> None:
    assert _remotes_from_upload(code_commits_only=True) == _remotes_from_upload()
    assert _remotes_from_download(code_commits_only=True) == _remotes_from_download()


def test_standalone_pr_flag_conflicts_with_legacy_code_commits_only() -> None:
    with pytest.raises(SystemExit):
        upload._selection_items(
            include_standalone_pr=True,
            code_commits_only=True,
        )
    with pytest.raises(SystemExit):
        download._default_manifest(
            "bucket",
            "prefix",
            "https://storage.eu-north1.nebius.cloud",
            include_standalone_pr=True,
            code_commits_only=True,
        )


# --- _s3_env(): credential RAISE + NEBIUS_* -> AWS_* mapping -----------------

# The autouse conftest fixture scrubs TileLang/MLX env vars but NOT S3
# credentials, so each cred test clears these explicitly for a hermetic baseline.
_CRED_ENV_VARS = (
    "NEBIUS_S3_ACCESS_KEY_ID",
    "NEBIUS_S3_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


@pytest.mark.parametrize("module", [upload, download])
def test_s3_env_raises_without_credentials(monkeypatch, module) -> None:
    for var in _CRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(SystemExit) as exc:
        module._s3_env()
    assert "NEBIUS_S3_ACCESS_KEY_ID" in str(exc.value)


@pytest.mark.parametrize("module", [upload, download])
def test_s3_env_maps_nebius_credentials_to_aws(monkeypatch, module) -> None:
    for var in _CRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # NEBIUS_* take precedence over any AWS_* and are exported as AWS_* for the
    # subprocess S3 client.
    monkeypatch.setenv("NEBIUS_S3_ACCESS_KEY_ID", "nebius-access")
    monkeypatch.setenv("NEBIUS_S3_SECRET_ACCESS_KEY", "nebius-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "should-be-overridden")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-be-overridden")

    env = module._s3_env()
    assert env["AWS_ACCESS_KEY_ID"] == "nebius-access"
    assert env["AWS_SECRET_ACCESS_KEY"] == "nebius-secret"


@pytest.mark.parametrize("module", [upload, download])
def test_s3_env_falls_back_to_existing_aws_credentials(monkeypatch, module) -> None:
    for var in _CRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")

    env = module._s3_env()
    assert env["AWS_ACCESS_KEY_ID"] == "aws-access"
    assert env["AWS_SECRET_ACCESS_KEY"] == "aws-secret"


# --- _existing_sources(): missing-input RAISE vs all-present pass-through -----

def test_existing_sources_raises_listing_missing_inputs(tmp_path: Path) -> None:
    present = tmp_path / "present"
    present.mkdir()
    missing = tmp_path / "missing"  # never created
    selections = (
        ("parquet/code/1024", present),
        ("parquet/code/2048", missing),
    )
    with pytest.raises(SystemExit) as exc:
        upload._existing_sources(selections)
    assert str(missing) in str(exc.value)
    assert str(present) not in str(exc.value)


def test_existing_sources_returns_present_pairs(tmp_path: Path) -> None:
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    selections = (("r1", a), ("r2", b))
    assert upload._existing_sources(selections) == [("r1", a), ("r2", b)]


# --- main(['--dry-run']): S3 URI template, no real S3 call, no creds ----------

def test_upload_main_dry_run_writes_manifest_and_prints_targets(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    # Every selected source dir must exist or _existing_sources RAISEs; this
    # also creates the audit dir that holds the receipt.
    _populate_upload_sources(upload.ALL_VALID_SELECTIONS)
    # Real temp receipt at the relative path the script reads (_audit_receipts),
    # covering every selected parquet bucket and green.
    _build_receipt(upload._audit_receipts()[0], valid_tokens=_BUCKET_VALID_TOKENS)

    manifest_path = tmp_path / "manifest.json"
    rc = upload.main(
        [
            "--bucket",
            "mybucket",
            "--prefix",
            "myprefix",
            "--dry-run",
            "--manifest",
            str(manifest_path),
            "--env-file",
            str(tmp_path / "nonexistent.env"),
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "aws s3 sync" in out
    # URI template: s3://{bucket}/{prefix}/{remote}/
    assert "s3://mybucket/myprefix/parquet/code/1024/" in out
    assert "parquet/pr/" not in out
    assert "s3://mybucket/myprefix/audits/sidecar_audit_all_final_poststop_valid/" in out
    # The manifest itself is cp'd to s3://.../manifest.json.
    assert "s3://mybucket/myprefix/manifest.json" in out

    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(
        written,
        expected_bucket="mybucket",
        expected_prefix="myprefix",
        expected_endpoint_url="https://storage.eu-north1.nebius.cloud",
    )
    assert written["bucket"] == "mybucket"
    assert written["prefix"] == "myprefix"
    assert written["profile"] == "code_commits_integrated_pr"
    assert written["standalone_pr_included"] is False
    assert written["verified_valid_tokens"] == sum(
        v for k, v in _BUCKET_VALID_TOKENS.items() if not k.startswith("pr/")
    )
    assert {s["remote"] for s in written["selections"]} == _remotes_from_upload()
    assert any(
        item["remote"].startswith("parquet/pr/")
        for item in written["selection_policy"]["excluded"]
    )


def test_upload_main_dry_run_does_not_gate_default_on_standalone_pr_bucket(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _populate_upload_sources(upload.ALL_VALID_SELECTIONS)
    _build_receipt(
        upload._audit_receipts()[0],
        valid_tokens=_BUCKET_VALID_TOKENS,
        bad={"pr/8192": (1, 0)},
    )

    rc = upload.main(
        [
            "--bucket",
            "mybucket",
            "--dry-run",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--env-file",
            str(tmp_path / "nonexistent.env"),
        ]
    )
    assert rc == 0


def test_upload_main_dry_run_raises_on_non_green_explicit_standalone_pr_bucket(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _populate_upload_sources(
        upload.CODE_COMMIT_SELECTIONS + upload.STANDALONE_PR_SELECTIONS
    )
    _build_receipt(
        upload._audit_receipts()[0],
        valid_tokens=_BUCKET_VALID_TOKENS,
        bad={"pr/8192": (1, 0)},
    )

    with pytest.raises(SystemExit) as exc:
        upload.main(
            [
                "--bucket",
                "mybucket",
                "--dry-run",
                "--include-standalone-pr",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--env-file",
                str(tmp_path / "nonexistent.env"),
            ]
        )
    assert "pr/8192" in str(exc.value)


def test_upload_main_records_failed_receipt_before_local_preflight(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "sidecar_upload_receipt.json"
    with chdir(tmp_path):
        upload._audit_receipts()[0].parent.mkdir(parents=True, exist_ok=True)
        _build_receipt(
            upload._audit_receipts()[0], valid_tokens=_BUCKET_VALID_TOKENS
        )

        with pytest.raises(SystemExit, match="missing verified upload inputs"):
            upload.main(
                [
                    "--bucket",
                    "mybucket",
                    "--dry-run",
                    "--manifest",
                    str(manifest_path),
                    "--env-file",
                    str(tmp_path / "nonexistent.env"),
                ]
            )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "missing verified upload inputs" in receipt["error"]
    assert not manifest_path.exists()


def test_download_main_dry_run_default_manifest_prints_source_uris(
    tmp_path: Path, capsys
) -> None:
    dest = tmp_path / "dl"
    rc = download.main(
        [
            "--bucket",
            "mybucket",
            "--prefix",
            "myprefix",
            "--dest",
            str(dest),
            "--dry-run",
            "--use-default-manifest",
            "--env-file",
            str(tmp_path / "nonexistent.env"),
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "aws s3 sync" in out
    assert "s3://mybucket/myprefix/parquet/code/1024/" in out
    assert str(dest / "outputs/reindexed/1024") in out
    assert "parquet/pr/" not in out
    assert "s3://mybucket/myprefix/audits/sidecar_audit_all_final_poststop_valid/" in out


def test_download_verify_scopes_token_total_and_green_gate_to_selected_buckets(
    tmp_path: Path,
) -> None:
    manifest, root = _download_fixture(
        tmp_path,
        bad={"pr/8192": (1, 0)},
    )
    code_commits_sum = sum(
        v for k, v in _BUCKET_VALID_TOKENS.items() if not k.startswith("pr/")
    )

    result = download._verify_downloaded_set(
        dest=root,
        manifest=manifest,
        require_token_total=True,
    )

    assert result["verified_valid_tokens"] == code_commits_sum
    assert result["bad_files"] == 0
    assert result["bad_rows"] == 0


def test_download_verify_rejects_non_green_explicit_standalone_pr_bucket(
    tmp_path: Path,
) -> None:
    manifest, root = _download_fixture(
        tmp_path,
        include_standalone_pr=True,
        bad={"pr/8192": (1, 0)},
    )

    with pytest.raises(ValueError) as exc:
        download._verify_downloaded_set(
            dest=root,
            manifest=manifest,
            require_token_total=True,
        )
    assert "pr/8192" in str(exc.value)


def test_download_main_dry_run_explicit_standalone_pr_prints_pr_uris(
    tmp_path: Path, capsys
) -> None:
    dest = tmp_path / "dl"
    rc = download.main(
        [
            "--bucket",
            "mybucket",
            "--prefix",
            "myprefix",
            "--dest",
            str(dest),
            "--dry-run",
            "--use-default-manifest",
            "--include-standalone-pr",
            "--env-file",
            str(tmp_path / "nonexistent.env"),
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "s3://mybucket/myprefix/parquet/pr/16384/" in out


def test_download_main_dry_run_code_commits_only_excludes_pr_uris(
    tmp_path: Path, capsys
) -> None:
    dest = tmp_path / "dl"
    rc = download.main(
        [
            "--bucket",
            "mybucket",
            "--prefix",
            "myprefix",
            "--dest",
            str(dest),
            "--dry-run",
            "--use-default-manifest",
            "--code-commits-only",
            "--env-file",
            str(tmp_path / "nonexistent.env"),
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "s3://mybucket/myprefix/parquet/code/8192/" in out
    assert "s3://mybucket/myprefix/parquet/commits/16384/" in out
    assert "parquet/pr/" not in out


def test_download_main_dry_run_copies_manifest_uri(
    tmp_path: Path, capsys
) -> None:
    # Without --use-default-manifest the script first cp's the remote
    # manifest.json; under --dry-run that command is only printed (never read).
    dest = tmp_path / "dl"
    rc = download.main(
        [
            "--bucket",
            "mybucket",
            "--prefix",
            "myprefix",
            "--dest",
            str(dest),
            "--dry-run",
            "--env-file",
            str(tmp_path / "nonexistent.env"),
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "aws s3 cp" in out
    assert "s3://mybucket/myprefix/manifest.json" in out
    # Falls back to the default training manifest under --dry-run and still
    # syncs code/commit buckets, but not standalone PR diagnostics.
    assert "s3://mybucket/myprefix/parquet/code/1024/" in out
    assert "parquet/pr/" not in out
