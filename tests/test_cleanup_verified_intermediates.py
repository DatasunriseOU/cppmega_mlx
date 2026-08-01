from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cleanup_verified_intermediates as cleanup


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, cleanup.Proof]:
    repo = "sample"
    cache_root = tmp_path / "cache"
    repo_dir = cache_root / repo
    repo_dir.mkdir(parents=True)
    (repo_dir / ".conveyor-cache.lock").touch()
    source = repo_dir / f"{repo}_commits.jsonl"
    source.write_bytes(b'{"commit": 1}\n{"commit": 2}\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _write_json(
        Path(f"{source}.extract-checkpoint") / "publication.json",
        {
            "schema_version": 1,
            "contract_version": 1,
            "status": "done",
            "output_path": str(source.resolve()),
            "output": {
                "size_bytes": source.stat().st_size,
                "line_count": 2,
                "sha256": digest,
            },
            "job_fingerprint": "fixture",
        },
    )

    parquet_root = tmp_path / "parquet"
    parquet_path = parquet_root / "1024" / f"{repo}_r0.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "valid_token_count": pa.array([3, 4], type=pa.int32()),
                "trained_token_count": pa.array([2, 3], type=pa.int32()),
                "slack_tokens": pa.array([1021, 1020], type=pa.int32()),
            }
        ),
        parquet_path,
        compression="zstd",
    )
    manifest_path = tmp_path / "done.json"
    _write_json(
        manifest_path,
        {
            "done": {
                f"{repo}::commit_plan": {
                    "source": "commit_plan",
                    "n_records": 2,
                },
                f"{repo}::r0": {
                    "source": "commits",
                    "repo": repo,
                    "range": [0, 2],
                    "n_records": 2,
                    "dedup_stage_promoted": {
                        "stage_id": f"commit:{repo}:r0:2",
                    },
                    "lengths": {
                        "1024": {
                            "rows": 2,
                            "capacity_tokens": 2048,
                            "valid_tokens": 7,
                            "pad_tokens": 2041,
                        }
                    },
                },
                f"{repo}::commits": {
                    "source": "commits",
                    "n_records": 2,
                    "range_count": 1,
                    "complete": True,
                    "completion_proof": "commit_plan_exact_range_coverage",
                },
            }
        },
    )
    proof = cleanup.load_proofs([(manifest_path, parquet_root)])[0]
    return cache_root, proof


def test_build_plan_requires_full_receipts_and_validates_parquet(
    tmp_path: Path,
) -> None:
    cache_root, proof = _fixture(tmp_path)
    verified, blocked, locks = cleanup.build_plan(cache_root, [proof])
    try:
        assert blocked == []
        assert verified[0]["repo"] == "sample"
        assert verified[0]["source"]["lines"] == 2
        assert verified[0]["parquet"] == {
            "files": 1,
            "bytes": (tmp_path / "parquet" / "1024" / "sample_r0.parquet")
            .stat()
            .st_size,
            "rows": 2,
            "valid_tokens": 7,
            "trained_tokens": 5,
            "pad_tokens": 2041,
        }
    finally:
        for lock in locks:
            lock.close()


def test_build_plan_blocks_old_code_only_completion(tmp_path: Path) -> None:
    cache_root, proof = _fixture(tmp_path)
    proof.done.pop("sample::commits")
    verified, blocked, locks = cleanup.build_plan(cache_root, [proof])
    assert locks == []
    assert verified == []
    assert blocked == [
        {
            "repo": "sample",
            "path": str(cache_root / "sample"),
            "reason": "no_exact_range_completion",
        }
    ]


def test_build_plan_fails_closed_on_publication_or_parquet_mismatch(
    tmp_path: Path,
) -> None:
    cache_root, proof = _fixture(tmp_path)
    source = cache_root / "sample" / "sample_commits.jsonl"
    source.write_bytes(b'{"commit": 9}\n{"commit": 2}\n')
    with pytest.raises(cleanup.VerificationError, match="digest/lines"):
        cleanup.build_plan(cache_root, [proof])

    cache_root, proof = _fixture(tmp_path / "second")
    proof.done["sample::r0"]["lengths"]["1024"]["valid_tokens"] = 8
    proof.done["sample::r0"]["lengths"]["1024"]["pad_tokens"] = 2040
    with pytest.raises(cleanup.VerificationError, match="token totals"):
        cleanup.build_plan(cache_root, [proof])


def test_parquet_row_token_invariants_are_fail_closed(tmp_path: Path) -> None:
    cache_root, proof = _fixture(tmp_path)
    parquet_path = tmp_path / "parquet" / "1024" / "sample_r0.parquet"
    pq.write_table(
        pa.table(
            {
                "valid_token_count": pa.array([3, 4], type=pa.int32()),
                "trained_token_count": pa.array([4, 3], type=pa.int32()),
                "slack_tokens": pa.array([1021, 1020], type=pa.int32()),
            }
        ),
        parquet_path,
        compression="zstd",
    )
    with pytest.raises(cleanup.VerificationError, match="trained_token_count"):
        cleanup.build_plan(cache_root, [proof])


def test_manifest_artifact_filename_selects_case_safe_parquet(
    tmp_path: Path,
) -> None:
    cache_root, proof = _fixture(tmp_path)
    encoded_cache = cache_root / "%53ample"
    (cache_root / "sample").rename(encoded_cache)
    source = encoded_cache / "sample_commits.jsonl"
    publication_path = (
        Path(f"{source}.extract-checkpoint") / "publication.json"
    )
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["output_path"] = str(source.resolve())
    _write_json(publication_path, publication)
    legacy = tmp_path / "parquet" / "1024" / "sample_r0.parquet"
    encoded = legacy.with_name("%53ample_r0.parquet")
    legacy.rename(encoded)
    proof.done["sample::r0"]["artifact_filename"] = encoded.name

    verified, blocked, locks = cleanup.build_plan(cache_root, [proof])
    try:
        assert blocked == []
        assert verified[0]["parquet"]["files"] == 1
    finally:
        for lock in locks:
            lock.close()


def test_empty_after_dedup_range_does_not_require_a_stage(tmp_path: Path) -> None:
    cache_root, proof = _fixture(tmp_path)
    proof.done["sample::r0"] = {
        "source": "commits",
        "repo": "sample",
        "range": [0, 2],
        "n_records": 2,
        "empty_after_dedup": True,
        "lengths": {},
    }
    verified, blocked, locks = cleanup.build_plan(cache_root, [proof])
    try:
        assert blocked == []
        assert verified[0]["parquet"]["files"] == 0
    finally:
        for lock in locks:
            lock.close()


def test_execute_cleanup_removes_only_verified_direct_child_and_writes_receipt(
    tmp_path: Path,
) -> None:
    cache_root, proof = _fixture(tmp_path)
    verified, blocked, locks = cleanup.build_plan(cache_root, [proof])
    receipt = {
        "schema_version": 1,
        "mode": "execute",
        "status": "planned",
        "cache_root": str(cache_root.resolve()),
        "blocked": blocked,
        "verified": verified,
    }
    receipt_path = tmp_path / "receipts" / "cleanup.json"
    try:
        cleanup.execute_cleanup(cache_root, receipt_path, receipt)
    finally:
        for lock in locks:
            lock.close()

    assert not (cache_root / "sample").exists()
    persisted = json.loads(receipt_path.read_text())
    assert persisted["status"] == "complete"
    assert persisted["verified"][0]["status"] == "removed"
