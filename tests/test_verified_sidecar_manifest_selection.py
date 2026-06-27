from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import download_verified_sidecar_from_nebius_s3 as download
from scripts import upload_verified_sidecar_to_nebius_s3 as upload


def _remotes_from_upload(*, code_commits_only: bool = False) -> set[str]:
    return {
        remote
        for remote, _local in upload._selection_items(
            code_commits_only=code_commits_only
        )
    }


def _remotes_from_download(*, code_commits_only: bool = False) -> set[str]:
    manifest = download._default_manifest(
        "bucket",
        "prefix",
        "https://storage.eu-north1.nebius.cloud",
        code_commits_only=code_commits_only,
    )
    return {item["remote"] for item in manifest["selections"]}


def test_upload_default_includes_every_final_audit_valid_bucket() -> None:
    remotes = _remotes_from_upload()

    assert "parquet/code/1024" in remotes
    assert "parquet/code/8192" in remotes
    assert "parquet/commits/1024" in remotes
    assert "parquet/commits/16384" in remotes
    assert "parquet/pr/1024" in remotes
    assert "parquet/pr/16384" in remotes
    assert "audits/sidecar_audit_all_final_poststop_valid" in remotes


def test_upload_code_commits_only_is_explicit_opt_out_from_pr() -> None:
    remotes = _remotes_from_upload(code_commits_only=True)

    assert "parquet/code/8192" in remotes
    assert "parquet/commits/16384" in remotes
    assert not any(remote.startswith("parquet/pr/") for remote in remotes)


def test_download_default_manifest_matches_upload_all_valid_policy() -> None:
    default_remotes = _remotes_from_download()
    code_commits_only_remotes = _remotes_from_download(code_commits_only=True)

    assert "parquet/pr/1024" in default_remotes
    assert "parquet/pr/16384" in default_remotes
    assert not any(remote.startswith("parquet/pr/") for remote in code_commits_only_remotes)


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
            "bad_files": bf,
            "bad_rows": br,
        }
    payload = {
        "total": {
            "valid_tokens": sum(valid_tokens.values()),
            "bad_files": 0,
            "bad_rows": 0,
        },
        "by_kind_bucket": by_kind_bucket,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_token_total_is_profile_aware_subset(tmp_path: Path) -> None:
    receipt = _build_receipt(
        tmp_path / "sidecar_parquet_audit.json",
        valid_tokens=_BUCKET_VALID_TOKENS,
    )
    receipts = (receipt,)

    all_sel = upload._selection_items(code_commits_only=False)
    cc_sel = upload._selection_items(code_commits_only=True)

    all_total = upload._load_verified_token_total(receipts, all_sel)
    cc_total = upload._load_verified_token_total(receipts, cc_sel)

    code_commits_sum = sum(
        v for k, v in _BUCKET_VALID_TOKENS.items() if not k.startswith("pr/")
    )
    pr_sum = sum(v for k, v in _BUCKET_VALID_TOKENS.items() if k.startswith("pr/"))

    # all-valid sums every parquet bucket; the subset must drop the PR tokens
    # entirely instead of reusing the full all-valid total.
    assert all_total == sum(_BUCKET_VALID_TOKENS.values())
    assert cc_total == code_commits_sum
    assert all_total - cc_total == pr_sum
    assert cc_total < all_total


def test_token_total_raises_when_selected_bucket_uncovered(tmp_path: Path) -> None:
    # A stale/narrow receipt that omits a selected bucket must NOT silently pass
    # even though its aggregate total is green.
    receipt = _build_receipt(
        tmp_path / "sidecar_parquet_audit.json",
        valid_tokens=_BUCKET_VALID_TOKENS,
        omit={"commits/16384"},
    )
    cc_sel = upload._selection_items(code_commits_only=True)

    with pytest.raises(SystemExit) as exc:
        upload._load_verified_token_total((receipt,), cc_sel)
    assert "commits/16384" in str(exc.value)


def test_token_total_raises_when_selected_bucket_not_green(tmp_path: Path) -> None:
    receipt = _build_receipt(
        tmp_path / "sidecar_parquet_audit.json",
        valid_tokens=_BUCKET_VALID_TOKENS,
        bad={"code/4096": (0, 3)},
    )
    cc_sel = upload._selection_items(code_commits_only=True)

    with pytest.raises(SystemExit) as exc:
        upload._load_verified_token_total((receipt,), cc_sel)
    assert "code/4096" in str(exc.value)


def test_token_total_gate_is_scoped_to_selected_buckets(tmp_path: Path) -> None:
    # A non-green PR bucket must not block a code+commits-only upload, but must
    # block the all-valid upload that includes it.
    receipt = _build_receipt(
        tmp_path / "sidecar_parquet_audit.json",
        valid_tokens=_BUCKET_VALID_TOKENS,
        bad={"pr/8192": (1, 0)},
    )
    receipts = (receipt,)

    cc_sel = upload._selection_items(code_commits_only=True)
    code_commits_sum = sum(
        v for k, v in _BUCKET_VALID_TOKENS.items() if not k.startswith("pr/")
    )
    assert upload._load_verified_token_total(receipts, cc_sel) == code_commits_sum

    all_sel = upload._selection_items(code_commits_only=False)
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
    cc_sel = upload._selection_items(code_commits_only=True)
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
    assert _remotes_from_upload(code_commits_only=True) == _remotes_from_download(
        code_commits_only=True
    )
    # The only difference between the two profiles is the standalone PR buckets.
    only_in_all_valid = _remotes_from_upload() - _remotes_from_upload(
        code_commits_only=True
    )
    assert only_in_all_valid == {
        "parquet/pr/1024",
        "parquet/pr/2048",
        "parquet/pr/4096",
        "parquet/pr/8192",
        "parquet/pr/16384",
    }


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
    for _remote, local in upload.ALL_VALID_SELECTIONS:
        local.mkdir(parents=True, exist_ok=True)
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
    assert "s3://mybucket/myprefix/parquet/pr/16384/" in out
    assert "s3://mybucket/myprefix/audits/sidecar_audit_all_final_poststop_valid/" in out
    # The manifest itself is cp'd to s3://.../manifest.json.
    assert "s3://mybucket/myprefix/manifest.json" in out

    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["bucket"] == "mybucket"
    assert written["prefix"] == "myprefix"
    assert written["profile"] == "all_valid"
    assert written["standalone_pr_included"] is True
    assert written["verified_valid_tokens"] == sum(_BUCKET_VALID_TOKENS.values())
    assert {s["remote"] for s in written["selections"]} == _remotes_from_upload()


def test_upload_main_dry_run_raises_on_non_green_selected_bucket(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for _remote, local in upload.ALL_VALID_SELECTIONS:
        local.mkdir(parents=True, exist_ok=True)
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
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--env-file",
                str(tmp_path / "nonexistent.env"),
            ]
        )
    assert "pr/8192" in str(exc.value)


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
    assert "s3://mybucket/myprefix/parquet/pr/16384/" in out
    assert "s3://mybucket/myprefix/audits/sidecar_audit_all_final_poststop_valid/" in out


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
    # Falls back to the default manifest under --dry-run and still syncs buckets.
    assert "s3://mybucket/myprefix/parquet/pr/16384/" in out
