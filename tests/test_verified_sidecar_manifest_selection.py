from __future__ import annotations

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
