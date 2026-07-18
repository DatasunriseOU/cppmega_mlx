from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest


MLX_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = MLX_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _repo_identity_map(tmp_path: Path) -> Path:
    path = tmp_path / "repo_list.json"
    path.write_text(
        json.dumps(
            {"repos": [{"name": "repo", "owner_repo": "tests/repo"}]}
        ),
        encoding="utf-8",
    )
    return path


def test_code_project_identity_claims_deduplicate_aliases() -> None:
    import streaming_conveyor

    claims: dict[str, str] = {}

    assert streaming_conveyor.claim_code_project_identity(
        claims,
        "open-watcom",
        "open-watcom/open-watcom-v2",
    ) is None
    assert streaming_conveyor.claim_code_project_identity(
        claims,
        "open-watcom-v2",
        "open-watcom/open-watcom-v2",
    ) == "open-watcom"


def test_libclang_preflight_fails_before_staging_with_child_error(tmp_path) -> None:
    import streaming_conveyor

    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\necho 'clang bindings unavailable' >&2\nexit 17\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    with pytest.raises(
        SystemExit,
        match=(
            r"libclang preflight failed: .*exit=17: "
            r"clang bindings unavailable"
        ),
    ):
        streaming_conveyor.verify_libclang_preflight(fake_python)


def test_unmapped_code_repo_is_quarantined_without_blocking_next_submission(
    tmp_path,
) -> None:
    import threading

    import streaming_conveyor

    repo_list = tmp_path / "repo_list.json"
    repo_list.write_text(
        json.dumps(
            {
                "repos": [
                    {
                        "name": "public-repo",
                        "owner_repo": "tests/public-repo",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = streaming_conveyor.Manifest(tmp_path / "_done.json")
    progress_path = tmp_path / "progress.jsonl"
    progress = streaming_conveyor.ProgressWriter(progress_path)
    claims: dict[str, str] = {}
    lock = threading.Lock()

    assert not streaming_conveyor.claim_code_repo_for_submission(
        repo="unmapped archive",
        repo_list=repo_list,
        project_identity_claims=claims,
        manifest=manifest,
        manifest_lock=lock,
        progress=progress,
    )
    assert manifest.failed["unmapped archive::code"]["stage"] == "project_identity"
    assert claims == {}

    assert streaming_conveyor.claim_code_repo_for_submission(
        repo="public-repo",
        repo_list=repo_list,
        project_identity_claims=claims,
        manifest=manifest,
        manifest_lock=lock,
        progress=progress,
    )
    assert claims == {"tests/public-repo": "public-repo"}

    events = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 1
    assert events[0]["event"] == "unit_failed"
    assert events[0]["repo"] == "unmapped archive"
    assert events[0]["unit"] == "unmapped archive::code"
    assert events[0]["stream"] == "code"
    assert events[0]["stage"] == "project_identity"
    assert events[0]["detail"] == (
        f"{repo_list}: no canonical project identity for bare repo "
        "'unmapped archive'"
    )


def test_progress_writer_appends_jsonl(tmp_path):
    import streaming_conveyor

    path = tmp_path / "progress.jsonl"
    writer = streaming_conveyor.ProgressWriter(path)
    writer.emit("unit_done", stream="code", repo="repo", valid_tokens=1024)
    writer.emit("unit_failed", stream="commits", repo="repo", stage="test")

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == ["unit_done", "unit_failed"]
    assert rows[0]["stream"] == "code"
    assert rows[0]["valid_tokens"] == 1024
    assert rows[1]["stage"] == "test"


def test_progress_writer_tracks_extract_cache_rates(tmp_path):
    import streaming_conveyor

    path = tmp_path / "progress.jsonl"
    writer = streaming_conveyor.ProgressWriter(path)

    writer.emit("extract_cache", repo="a", stream="commits", status="hit")
    writer.emit("extract_cache", repo="b", stream="commits", status="fresh")
    writer.emit("extract_cache", repo="c", stream="commits", status="adopt")

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert rows[-1]["extract_cache_seen"] == 3
    assert rows[-1]["extract_cache_hits"] == 1
    assert rows[-1]["extract_cache_hit_rate"] == 1 / 3
    assert rows[-1]["extract_cache_reused"] == 2
    assert rows[-1]["extract_cache_reuse_rate"] == 2 / 3
    assert rows[-1]["extract_cache_status_counts"] == {
        "adopt": 1,
        "fresh": 1,
        "hit": 1,
    }
    assert writer.extract_cache_metrics()["status_counts"]["hit"] == 1


def test_populate_code_source_cache_emits_ready_and_summary(tmp_path, monkeypatch):
    import streaming_conveyor

    progress_path = tmp_path / "progress.jsonl"
    progress = streaming_conveyor.ProgressWriter(progress_path)
    cache = tmp_path / "source_cache"
    observed = {}

    def fake_populate(
        work_root,
        should_process,
        source_cache_dir,
        *,
        max_repos,
        on_repo_ready,
    ):
        observed["work_root"] = work_root
        observed["source_cache_dir"] = source_cache_dir
        observed["max_repos"] = max_repos
        observed["has_callback"] = on_repo_ready is not None
        repos = []
        for repo in ("repo-a", "repo.bare", "repo-b"):
            if should_process(repo):
                repos.append({"repo": repo, "path": str(source_cache_dir / repo)})
        repos = repos[:max_repos]
        for idx, item in enumerate(repos, start=1):
            on_repo_ready(item["repo"], Path(item["path"]), idx)
        return {
            "source_cache_dir": str(source_cache_dir),
            "repos": repos,
            "repo_count": len(repos),
        }

    monkeypatch.setattr(streaming_conveyor.sr, "populate_source_cache", fake_populate)

    report = streaming_conveyor.populate_code_source_cache(
        tmp_path / "work",
        cache,
        streaming_conveyor.sr.is_code_worktree_repo,
        progress,
        max_repos=1,
    )

    assert observed == {
        "work_root": tmp_path / "work",
        "source_cache_dir": cache,
        "max_repos": 1,
        "has_callback": True,
    }
    assert report["repo_count"] == 1
    rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "source_cache_repo_ready",
        "source_cache_populated",
    ]
    assert rows[0]["repo"] == "repo-a"
    assert rows[0]["repo_dir"] == str(cache / "repo-a")
    assert rows[1]["repo_count"] == 1
    assert rows[1]["repos"] == [{"repo": "repo-a", "path": str(cache / "repo-a")}]


def test_background_recompressor_runs_and_surfaces_completion(tmp_path, monkeypatch):
    import streaming_conveyor

    calls = []

    def fake_recompress(path):
        calls.append(Path(path).name)

    monkeypatch.setattr(
        streaming_conveyor.src,
        "recompress_zstd_max",
        fake_recompress,
    )

    recompressor = streaming_conveyor.BackgroundRecompressor(max_workers=2)
    recompressor.submit(tmp_path / "a.parquet")
    recompressor.submit(tmp_path / "b.parquet")
    recompressor.shutdown()

    assert sorted(calls) == ["a.parquet", "b.parquet"]


def test_manifest_complete_commit_ranges_uses_authoritative_record_count(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "repo::commit_plan": {"source": "commit_plan", "n_records": 823},
            "repo::r0": {"range": [0, 500], "source": "commits"},
            "repo::r500": {"range": [500, 823], "source": "commits"},
        },
        failed={},
    )

    assert streaming_conveyor.manifest_complete_commit_ranges(
        "repo", manifest, 500
    ) == ((0, 500), (500, 823))


def test_manifest_complete_commit_ranges_ignores_stale_aggregate_failure(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "repo::commit_plan": {"source": "commit_plan", "n_records": 823},
            "repo::r0": {"range": [0, 500], "source": "commits"},
            "repo::r500": {"range": [500, 823], "source": "commits"},
        },
        failed={
            "repo::commits": {
                "stage": "extract_git_history",
                "detail": "stale failure from an earlier attempt",
            }
        },
    )

    assert streaming_conveyor.manifest_complete_commit_ranges(
        "repo", manifest, 500
    ) == ((0, 500), (500, 823))


def test_manifest_complete_commit_ranges_rejects_failed_range(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "repo::commit_plan": {"source": "commit_plan", "n_records": 823},
            "repo::r0": {"range": [0, 500], "source": "commits"},
            "repo::r500": {"range": [500, 823], "source": "commits"},
        },
        failed={
            "repo::r500": {
                "stage": "pack",
                "detail": "range is not authoritatively complete",
            }
        },
    )

    assert streaming_conveyor.manifest_complete_commit_ranges(
        "repo", manifest, 500
    ) is None


def test_manifest_complete_commit_ranges_refuses_exact_multiple(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "repo::r0": {"range": [0, 500], "source": "commits"},
            "repo::r500": {"range": [500, 1000], "source": "commits"},
        },
        failed={},
    )

    assert streaming_conveyor.manifest_complete_commit_ranges(
        "repo", manifest, 500
    ) is None


def test_manifest_complete_commit_ranges_refuses_adaptive_short_nonfinal_range(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "repo::commit_plan": {"source": "commit_plan", "n_records": 1200},
            "repo::r0": {"range": [0, 468], "source": "commits"},
        },
        failed={},
    )

    assert streaming_conveyor.manifest_complete_commit_ranges(
        "repo", manifest, 500
    ) is None


def test_missing_commit_subranges_respects_split_done_intervals(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "repo::r0": {"range": [0, 250], "source": "commits"},
            "repo::r500": {"range": [500, 750], "source": "commits"},
        },
        failed={},
    )

    intervals = streaming_conveyor.manifest_done_commit_intervals("repo", manifest)

    assert streaming_conveyor.missing_commit_subranges(0, 500, intervals) == (
        (250, 500),
    )
    assert streaming_conveyor.missing_commit_subranges(500, 1000, intervals) == (
        (750, 1000),
    )
    assert not streaming_conveyor.manifest_covers_commit_span("repo", manifest, 1000)


def test_plan_commit_ranges_fixed_record_count_mode(tmp_path):
    import streaming_conveyor

    records = tmp_path / "records.jsonl"
    records.write_text("".join("{}\n" for _ in range(7)), encoding="utf-8")

    assert streaming_conveyor.plan_commit_ranges(
        records,
        7,
        max_records=3,
        target_bytes=0,
    ) == ((0, 3), (3, 6), (6, 7))


def test_plan_commit_ranges_balances_by_raw_record_bytes(tmp_path):
    import streaming_conveyor

    records = tmp_path / "records.jsonl"
    lines = [
        b"a" * 5 + b"\n",
        b"b" * 5 + b"\n",
        b"c" * 5 + b"\n",
        b"d" * 5 + b"\n",
        b"e" * 5 + b"\n",
    ]
    records.write_bytes(b"".join(lines))

    assert streaming_conveyor.plan_commit_ranges(
        records,
        5,
        max_records=500,
        target_bytes=12,
    ) == ((0, 2), (2, 4), (4, 5))


def test_plan_commit_ranges_keeps_oversized_record_as_single_range(tmp_path):
    import streaming_conveyor

    records = tmp_path / "records.jsonl"
    records.write_bytes(b"x" * 20 + b"\n" + b"y\n")

    assert streaming_conveyor.plan_commit_ranges(
        records,
        2,
        max_records=500,
        target_bytes=10,
    ) == ((0, 1), (1, 2))


def test_manifest_covers_commit_span_with_adaptive_splits(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "repo::r0": {"range": [0, 250], "source": "commits"},
            "repo::r250": {"range": [250, 500], "source": "commits"},
            "repo::r500": {"range": [500, 750], "source": "commits"},
            "repo::r750": {"range": [750, 1000], "source": "commits"},
        },
        failed={},
    )

    assert streaming_conveyor.manifest_covers_commit_span("repo", manifest, 1000)


def test_empty_after_dedup_commit_range_counts_as_done_interval(tmp_path):
    import streaming_conveyor
    import streaming_reindex_commits

    info = streaming_reindex_commits.empty_after_dedup_info("repo", 500, 750, 250)
    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "repo::r0": {"range": [0, 500], "source": "commits"},
            "repo::r500": info,
            "repo::r750": {"range": [750, 1000], "source": "commits"},
        },
        failed={},
    )

    assert info["lengths"] == {}
    assert info["empty_after_dedup"] is True
    assert streaming_conveyor.missing_commit_subranges(
        500,
        750,
        streaming_conveyor.manifest_done_commit_intervals("repo", manifest),
    ) == ()
    assert streaming_conveyor.manifest_covers_commit_span("repo", manifest, 1000)


def test_process_range_adaptive_splits_peak_oom(tmp_path):
    import streaming_conveyor

    calls = []

    def runner(
        repo,
        repo_dir,
        records_jsonl,
        start,
        end,
        lengths_sorted,
        repo_work,
        dedup_db,
        dedup_near,
        pr_store,
        repo_list,
        memory_limit_gb,
        analysis_cache_entries,
    ):
        calls.append((start, end, analysis_cache_entries))
        if end - start > 2:
            raise streaming_conveyor.RepoFailure(
                repo,
                "process_commits",
                "exit code 137\nERROR: process_commits exceeded memory limit",
            )
        return {
            "source": "commits",
            "repo": repo,
            "range": [start, end],
            "lengths": {"1024": _stat(rows=1, valid=1024, target_length=1024)},
        }

    result = streaming_conveyor.process_range_adaptive(
        "repo",
        tmp_path,
        tmp_path / "records.jsonl",
        0,
        8,
        (1024,),
        tmp_path / "work",
        None,
        False,
        None,
        None,
        8.0,
        analysis_cache_entries=37,
        min_range_size=1,
        runner=runner,
    )

    assert result["failed"] == []
    assert [item[:2] for item in result["done"]] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
    ]
    assert (0, 8, 37) in calls
    assert {call[2] for call in calls} == {37}


def test_run_code_half_adaptive_retries_single_parse_worker_on_peak_oom(tmp_path):
    import streaming_conveyor

    calls = []

    def runner(
        repo,
        project_id,
        repo_dir,
        lengths_code,
        work_root,
        dedup_db,
        dedup_near,
        global_symbol_index,
        memory_limit_gb,
        parse_workers,
        index_timeout_s,
        index_stall_timeout_s,
        recompressor,
    ):
        calls.append(
            (parse_workers, dedup_db, dedup_near, index_timeout_s, index_stall_timeout_s, recompressor)
        )
        if parse_workers > 1:
            raise streaming_conveyor.RepoFailure(
                repo,
                "index_project",
                "exit code 137\nERROR: index_project exceeded memory limit",
            )
        return {
            "source": "code",
            "repo": repo,
            "lengths": {"1024": _stat(rows=1, valid=1024, target_length=1024)},
        }

    info = streaming_conveyor.run_code_half_adaptive(
        "repo",
        "tests/repo",
        tmp_path,
        (1024,),
        tmp_path / "work",
        tmp_path / "global.sqlite",
        True,
        None,
        8.0,
        2,
        7200,
        900,
        runner=runner,
    )

    assert calls == [
        (2, tmp_path / "global.sqlite", True, 7200, 900, None),
        (1, tmp_path / "global.sqlite", False, 7200, 900, None),
    ]
    assert info["lengths"]["1024"]["valid_tokens"] == 1024


def test_run_code_half_adaptive_retries_single_parse_worker_on_timeout(tmp_path):
    import streaming_conveyor

    calls = []

    def runner(
        repo,
        project_id,
        repo_dir,
        lengths_code,
        work_root,
        dedup_db,
        dedup_near,
        global_symbol_index,
        memory_limit_gb,
        parse_workers,
        index_timeout_s,
        index_stall_timeout_s,
        recompressor,
    ):
        calls.append((parse_workers, dedup_db, dedup_near, index_timeout_s, index_stall_timeout_s))
        if parse_workers > 1:
            raise streaming_conveyor.RepoFailure(
                repo,
                "index_project",
                "timed out after 7200s",
            )
        return {
            "source": "code",
            "repo": repo,
            "lengths": {"1024": _stat(rows=1, valid=1024, target_length=1024)},
        }

    info = streaming_conveyor.run_code_half_adaptive(
        "repo",
        "tests/repo",
        tmp_path,
        (1024,),
        tmp_path / "work",
        tmp_path / "global.sqlite",
        True,
        None,
        8.0,
        2,
        7200,
        900,
        runner=runner,
    )

    assert calls == [
        (2, tmp_path / "global.sqlite", True, 7200, 900),
        (1, tmp_path / "global.sqlite", False, 7200, 900),
    ]
    assert info["lengths"]["1024"]["valid_tokens"] == 1024


def test_run_code_half_removes_outputs_when_dedup_promote_fails(tmp_path, monkeypatch):
    import streaming_conveyor

    repo = "repo"
    output_root = tmp_path / "out"
    dest = output_root / "1024" / f"{repo}.parquet"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"partial parquet already appended")
    monkeypatch.setattr(streaming_conveyor.sr, "OUTPUT_ROOT", output_root)

    def fake_process_one_repo(*_args, **_kwargs):
        return {
            "source": "code",
            "lengths": {"1024": _stat(rows=1, valid=1024, target_length=1024)},
            "stage_timings_s": {},
        }

    def fail_promote(*_args, **_kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(streaming_conveyor.sr, "process_one_repo", fake_process_one_repo)
    monkeypatch.setattr(streaming_conveyor.src, "recompress_zstd_max", lambda _path: None)
    monkeypatch.setattr(streaming_conveyor.sr, "promote_dedup_stage", fail_promote)

    try:
        streaming_conveyor.run_code_half(
            repo,
            "tests/repo",
            tmp_path / "src",
            (1024,),
            tmp_path / "work",
            tmp_path / "dedup.sqlite",
            True,
        )
    except streaming_conveyor.RepoFailure as exc:
        assert exc.stage == "dedup_promote"
        assert "OperationalError" in exc.detail
    else:
        raise AssertionError("run_code_half should fail when dedup promote fails")

    assert not dest.exists()


def test_run_code_half_recompresses_before_dedup_promote(tmp_path, monkeypatch):
    import streaming_conveyor

    repo = "repo"
    output_root = tmp_path / "out"
    dest = output_root / "1024" / f"{repo}.parquet"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"parquet")
    monkeypatch.setattr(streaming_conveyor.sr, "OUTPUT_ROOT", output_root)
    events: list[str] = []

    monkeypatch.setattr(
        streaming_conveyor.sr,
        "process_one_repo",
        lambda *_args, **_kwargs: {
            "source": "code",
            "lengths": {"1024": _stat(rows=1, valid=1024, target_length=1024)},
            "stage_timings_s": {},
        },
    )
    monkeypatch.setattr(
        streaming_conveyor.src,
        "recompress_zstd_max",
        lambda _path: events.append("recompress"),
    )

    def promote(*_args, **_kwargs):
        events.append("promote")
        return {}

    monkeypatch.setattr(streaming_conveyor.sr, "promote_dedup_stage", promote)
    recompressor = streaming_conveyor.BackgroundRecompressor(max_workers=1)
    try:
        streaming_conveyor.run_code_half(
            repo,
            "tests/repo",
            tmp_path / "src",
            (1024,),
            tmp_path / "work",
            tmp_path / "dedup.sqlite",
            True,
            recompressor=recompressor,
        )
    finally:
        recompressor.shutdown()

    assert events == ["recompress", "promote"]


def test_run_code_half_recompress_failure_rolls_back_before_promote(tmp_path, monkeypatch):
    import streaming_conveyor

    repo = "repo"
    output_root = tmp_path / "out"
    dest = output_root / "1024" / f"{repo}.parquet"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"parquet")
    monkeypatch.setattr(streaming_conveyor.sr, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        streaming_conveyor.sr,
        "process_one_repo",
        lambda *_args, **_kwargs: {
            "source": "code",
            "lengths": {"1024": _stat(rows=1, valid=1024, target_length=1024)},
            "stage_timings_s": {},
        },
    )
    monkeypatch.setattr(
        streaming_conveyor.src,
        "recompress_zstd_max",
        lambda _path: (_ for _ in ()).throw(RuntimeError("zstd failed")),
    )
    monkeypatch.setattr(
        streaming_conveyor.sr,
        "promote_dedup_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dedup must not promote before recompress")
        ),
    )

    with pytest.raises(streaming_conveyor.RepoFailure, match="zstd failed") as exc_info:
        streaming_conveyor.run_code_half(
            repo,
            "tests/repo",
            tmp_path / "src",
            (1024,),
            tmp_path / "work",
            tmp_path / "dedup.sqlite",
            True,
        )

    assert exc_info.value.stage == "recompress"
    assert not dest.exists()


def test_run_code_half_skips_no_trainable_source(tmp_path, monkeypatch):
    import streaming_conveyor

    def no_trainable_source(*_args, **_kwargs):
        raise streaming_conveyor.RepoFailure(
            "repo",
            "index_project",
            "no training docs (no_trainable_source): empty enriched jsonl",
        )

    monkeypatch.setattr(streaming_conveyor.sr, "process_one_repo", no_trainable_source)

    info = streaming_conveyor.run_code_half(
        "repo",
        "tests/repo",
        tmp_path / "src",
        (1024,),
        tmp_path / "work",
        tmp_path / "dedup.sqlite",
        True,
    )

    assert info["skipped"] is True
    assert info["skip_reason"] == "no_trainable_source"
    assert info["lengths"] == {}


def test_run_code_half_skips_dedup_exhausted_and_discards_stage(
    tmp_path, monkeypatch
):
    import streaming_conveyor

    def dedup_exhausted(*_args, **_kwargs):
        raise streaming_conveyor.RepoFailure(
            "repo",
            "index_project",
            "no training docs (dedup_exhausted): all functions already claimed",
        )

    discarded = []
    monkeypatch.setattr(streaming_conveyor.sr, "process_one_repo", dedup_exhausted)
    monkeypatch.setattr(
        streaming_conveyor.sr,
        "discard_dedup_stage",
        lambda *args: discarded.append(args),
    )

    info = streaming_conveyor.run_code_half(
        "repo",
        "tests/repo",
        tmp_path / "src",
        (1024,),
        tmp_path / "work",
        tmp_path / "dedup.sqlite",
        True,
    )

    assert info["skipped"] is True
    assert info["skip_reason"] == "dedup_exhausted"
    assert len(discarded) == 1


def test_failed_code_unit_was_index_memory_reads_manifest_receipt(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={},
        failed={
            "repo::code": {
                "stage": "index_project",
                "detail": "exit code 137\nERROR: index_project exceeded memory limit",
            }
        },
    )

    assert streaming_conveyor.failed_code_unit_was_index_memory("repo", manifest)
    assert not streaming_conveyor.failed_code_unit_was_index_memory("other", manifest)


def test_failed_code_unit_memory_retry_ignores_older_higher_parse_worker_failure(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={},
        failed={
            "repo::code": {
                "stage": "index_project",
                "detail": (
                    "Using 8 parse workers\n"
                    "ERROR: index_project exceeded memory limit: "
                    "max_rss=10.68 GiB limit=10.00 GiB"
                ),
            }
        },
    )

    assert not streaming_conveyor.failed_code_unit_was_index_memory(
        "repo",
        manifest,
        parse_workers=2,
    )
    assert streaming_conveyor.failed_code_unit_was_index_memory(
        "repo",
        manifest,
        parse_workers=8,
    )


def test_run_commits_half_skips_extract_when_manifest_proves_complete(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "repo::commit_plan": {"source": "commit_plan", "n_records": 612},
            "repo::r0": {"range": [0, 500], "source": "commits"},
            "repo::r500": {"range": [500, 612], "source": "commits"},
        },
        failed={
            "repo::commits": {
                "stage": "extract_git_history",
                "detail": "stale failure from an earlier attempt",
            }
        },
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        done, failed, all_done = streaming_conveyor.run_commits_half(
            repo="repo",
            repo_dir=tmp_path / "missing_repo_dir",
            repo_work=tmp_path / "repo_work",
            work_root=tmp_path / "work",
            work_parent=tmp_path / "work_parent",
            lengths_commits=(1024,),
            range_size=500,
            pool=pool,
            manifest=manifest,
            manifest_lock=streaming_conveyor.threading.Lock(),
            resume=True,
            cumulative={"valid": 0},
            dedup_db=None,
            dedup_near=False,
            pr_store=None,
            repo_list=_repo_identity_map(tmp_path),
        )

    assert (done, failed, all_done) == (0, 0, True)
    assert manifest.done["repo::commits"] == {
        "source": "commits",
        "repo": "repo",
        "complete": True,
        "completion_proof": "commit_plan_exact_range_coverage",
        "n_records": 612,
        "range_count": 2,
    }
    assert "repo::commits" not in manifest.failed


def test_process_one_repo_cleans_partial_intermediates_by_default(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import streaming_conveyor

    repo = "repo"
    work_root = tmp_path / "work"
    repo_work = work_root / repo
    repo_dir = repo_work / "_src"
    repo_dir.mkdir(parents=True)
    (repo_dir / "main.cpp").write_text("int f() { return 1; }\n", encoding="utf-8")
    cache_root = tmp_path / "extract_cache"
    cache_dir = cache_root / repo
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{repo}_commits.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(streaming_conveyor, "EXTRACT_CACHE_ROOT", cache_root)

    def fake_code_half(*_args, **_kwargs):
        return {
            "source": "code",
            "lengths": {"1024": _stat(rows=1, valid=32, target_length=1024)},
        }

    def fake_commits_half(*_args, **_kwargs):
        return 0, 0, False

    monkeypatch.setattr(streaming_conveyor, "run_code_half_adaptive", fake_code_half)
    monkeypatch.setattr(streaming_conveyor, "run_commits_half", fake_commits_half)
    manifest = streaming_conveyor.Manifest(tmp_path / "_done.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        streaming_conveyor.process_one_repo(
            repo=repo,
            repo_dir=repo_dir,
            lengths_code=(1024,),
            lengths_commits=(1024,),
            range_size=500,
            range_target_bytes=0,
            work_root=work_root,
            work_parent=tmp_path / "parent",
            pool=pool,
            manifest=manifest,
            manifest_lock=streaming_conveyor.threading.Lock(),
            resume=True,
            cumulative={"valid": 0},
            keep_temp=False,
            dedup_db=None,
            dedup_near=False,
            pr_store=None,
            repo_list=_repo_identity_map(tmp_path),
            streams="both",
        )

    assert manifest.is_done("repo::code")
    assert not repo_work.exists()
    assert not cache_dir.exists()


def test_process_one_repo_records_unresolved_project_identity_without_aborting_pool(
    tmp_path,
    monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor

    import streaming_conveyor

    repo = "unmapped archive"
    work_root = tmp_path / "work"
    repo_dir = work_root / repo / "_src"
    repo_dir.mkdir(parents=True)
    (repo_dir / "main.cpp").write_text("int value = 1;\n", encoding="utf-8")
    manifest = streaming_conveyor.Manifest(tmp_path / "_done.json")
    called = False

    def fail_if_code_runs(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("code stage must not run without canonical identity")

    monkeypatch.setattr(streaming_conveyor, "run_code_half_adaptive", fail_if_code_runs)
    monkeypatch.setattr(
        streaming_conveyor.sr,
        "resolve_project_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            streaming_conveyor.SymbolIdentityError("no canonical owner/repo identity")
        ),
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = streaming_conveyor.process_one_repo(
            repo=repo,
            repo_dir=repo_dir,
            lengths_code=(1024,),
            lengths_commits=(),
            range_size=500,
            range_target_bytes=0,
            work_root=work_root,
            work_parent=tmp_path / "parent",
            pool=pool,
            manifest=manifest,
            manifest_lock=streaming_conveyor.threading.Lock(),
            resume=True,
            cumulative={"valid": 0},
            keep_temp=False,
            dedup_db=None,
            dedup_near=False,
            pr_store=None,
            repo_list=tmp_path / "repo_list.json",
            streams="code",
        )

    assert not called
    assert result["code"] == "failed"
    assert manifest.failed[f"{repo}::code"]["stage"] == "project_identity"


def test_discover_existing_jsonl_never_adopts_unvalidated_stable_cache(
    tmp_path,
    monkeypatch,
):
    import streaming_conveyor

    repo = "repo"
    cache_root = tmp_path / "extract_cache"
    cache_dir = cache_root / repo
    cache_dir.mkdir(parents=True)
    unvalidated = cache_dir / f"{repo}_commits.jsonl"
    unvalidated.write_text('{"partial": true}\n', encoding="utf-8")
    work_root = tmp_path / "work"
    work_parent = tmp_path / "work_parent"
    monkeypatch.setattr(streaming_conveyor, "EXTRACT_CACHE_ROOT", cache_root)
    monkeypatch.setattr(streaming_conveyor, "DEFAULT_WORK_PARENT", work_parent)

    assert (
        streaming_conveyor._discover_existing_jsonl(
            repo,
            work_root,
            work_parent,
        )
        is None
    )


def test_ensure_commit_records_reextracts_unproven_legacy_jsonl(
    tmp_path,
    monkeypatch,
):
    import streaming_conveyor

    repo = "repo"
    cache_root = tmp_path / "extract_cache"
    work_root = tmp_path / "work"
    work_parent = tmp_path / "work_parent"
    legacy = work_root / repo / f"{repo}_commits.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"partial": true}\n', encoding="utf-8")
    monkeypatch.setattr(streaming_conveyor, "EXTRACT_CACHE_ROOT", cache_root)
    monkeypatch.setattr(streaming_conveyor, "DEFAULT_WORK_PARENT", work_parent)
    monkeypatch.setattr(streaming_conveyor, "get_commit_list", lambda _repo_dir: ["c1"])
    calls = []

    def fresh_extract(name, _repo_dir, cache_dir):
        calls.append(name)
        output = cache_dir / f"{name}_commits.jsonl"
        output.write_text('{"fresh": true}\n', encoding="utf-8")
        return output

    monkeypatch.setattr(streaming_conveyor, "stage_extract_commits", fresh_extract)
    manifest = streaming_conveyor.Manifest(tmp_path / "_done.json")

    output, count, status = streaming_conveyor.ensure_commit_records(
        repo,
        tmp_path / "repo_src",
        work_root,
        work_parent,
        manifest,
        resume=True,
    )

    assert calls == [repo]
    assert output.read_text(encoding="utf-8") == '{"fresh": true}\n'
    assert legacy.read_text(encoding="utf-8") == '{"partial": true}\n'
    assert (count, status) == (1, "fresh")
    assert streaming_conveyor._read_valid_sentinel(output) == 1


def test_process_one_repo_retains_transactional_extract_checkpoint(
    tmp_path,
    monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor

    import streaming_conveyor

    repo = "repo"
    work_root = tmp_path / "work"
    repo_work = work_root / repo
    repo_dir = repo_work / "_src"
    repo_dir.mkdir(parents=True)
    cache_root = tmp_path / "extract_cache"
    cache_dir = cache_root / repo
    checkpoint_db = (
        cache_dir
        / f"{repo}_commits.jsonl.extract-checkpoint"
        / "repo-0000-deadbeef"
        / "checkpoint.sqlite3"
    )
    checkpoint_db.parent.mkdir(parents=True)
    checkpoint_db.write_bytes(b"durable checkpoint")
    monkeypatch.setattr(streaming_conveyor, "EXTRACT_CACHE_ROOT", cache_root)

    monkeypatch.setattr(
        streaming_conveyor,
        "run_code_half_adaptive",
        lambda *_args, **_kwargs: {
            "source": "code",
            "lengths": {"1024": _stat(rows=1, valid=32, target_length=1024)},
        },
    )
    monkeypatch.setattr(
        streaming_conveyor,
        "run_commits_half",
        lambda *_args, **_kwargs: (0, 0, False),
    )
    manifest = streaming_conveyor.Manifest(tmp_path / "_done.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        streaming_conveyor.process_one_repo(
            repo=repo,
            repo_dir=repo_dir,
            lengths_code=(1024,),
            lengths_commits=(1024,),
            range_size=500,
            range_target_bytes=0,
            work_root=work_root,
            work_parent=tmp_path / "parent",
            pool=pool,
            manifest=manifest,
            manifest_lock=streaming_conveyor.threading.Lock(),
            resume=True,
            cumulative={"valid": 0},
            keep_temp=False,
            dedup_db=None,
            dedup_near=False,
            pr_store=None,
            repo_list=_repo_identity_map(tmp_path),
            streams="both",
        )

    assert not repo_work.exists()
    assert checkpoint_db.read_bytes() == b"durable checkpoint"


def test_process_one_repo_can_retain_partial_work_for_zero_rework_resume(
    tmp_path,
    monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor

    import streaming_conveyor

    repo = "repo"
    work_root = tmp_path / "work"
    repo_work = work_root / repo
    repo_dir = repo_work / "_src"
    repo_dir.mkdir(parents=True)
    cache_root = tmp_path / "extract_cache"
    cache_dir = cache_root / repo
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(streaming_conveyor, "EXTRACT_CACHE_ROOT", cache_root)

    def fake_code_half(*_args, **_kwargs):
        return {
            "source": "code",
            "lengths": {"1024": _stat(rows=1, valid=32, target_length=1024)},
        }

    def fake_commits_half(*_args, **_kwargs):
        return 0, 0, False

    monkeypatch.setattr(streaming_conveyor, "run_code_half_adaptive", fake_code_half)
    monkeypatch.setattr(streaming_conveyor, "run_commits_half", fake_commits_half)
    manifest = streaming_conveyor.Manifest(tmp_path / "_done.json")

    with ThreadPoolExecutor(max_workers=1) as pool:
        streaming_conveyor.process_one_repo(
            repo=repo,
            repo_dir=repo_dir,
            lengths_code=(1024,),
            lengths_commits=(1024,),
            range_size=500,
            range_target_bytes=0,
            work_root=work_root,
            work_parent=tmp_path / "parent",
            pool=pool,
            manifest=manifest,
            manifest_lock=streaming_conveyor.threading.Lock(),
            resume=True,
            cumulative={"valid": 0},
            keep_temp=False,
            dedup_db=None,
            dedup_near=False,
            pr_store=None,
            repo_list=_repo_identity_map(tmp_path),
            streams="both",
            retain_partial_work=True,
        )

    assert repo_work.exists()
    assert cache_dir.exists()


def test_process_one_repo_interrupted_keeps_extract_cache_without_temp(
    tmp_path,
    monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor

    import streaming_conveyor

    repo = "repo"
    work_root = tmp_path / "work"
    repo_work = work_root / repo
    repo_dir = repo_work / "_src"
    repo_dir.mkdir(parents=True)
    cache_root = tmp_path / "extract_cache"
    cache_dir = cache_root / repo
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{repo}_commits.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(streaming_conveyor, "EXTRACT_CACHE_ROOT", cache_root)

    def fake_code_half(*_args, **_kwargs):
        return {
            "source": "code",
            "lengths": {"1024": _stat(rows=1, valid=32, target_length=1024)},
        }

    def fake_commits_half(*_args, **_kwargs):
        return 0, 0, False

    monkeypatch.setattr(streaming_conveyor, "run_code_half_adaptive", fake_code_half)
    monkeypatch.setattr(streaming_conveyor, "run_commits_half", fake_commits_half)
    manifest = streaming_conveyor.Manifest(tmp_path / "_done.json")

    streaming_conveyor.STOP_EVENT.set()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            streaming_conveyor.process_one_repo(
                repo=repo,
                repo_dir=repo_dir,
                lengths_code=(1024,),
                lengths_commits=(1024,),
                range_size=500,
                range_target_bytes=0,
                work_root=work_root,
                work_parent=tmp_path / "parent",
                pool=pool,
                manifest=manifest,
                manifest_lock=streaming_conveyor.threading.Lock(),
                resume=True,
                cumulative={"valid": 0},
                keep_temp=False,
                dedup_db=None,
                dedup_near=False,
                pr_store=None,
                repo_list=_repo_identity_map(tmp_path),
                streams="both",
                retain_partial_work=False,
            )
    finally:
        streaming_conveyor.STOP_EVENT.clear()

    assert not repo_work.exists()
    assert cache_dir.exists()


def test_run_commits_half_batches_deferred_dedup_promotions(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    import streaming_conveyor

    clang_indexer = MLX_ROOT / "tools" / "clang_indexer"
    if str(clang_indexer) not in sys.path:
        sys.path.insert(0, str(clang_indexer))
    from dedup_store import DedupStore

    repo = "repo"
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(json.dumps({"i": i}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    manifest = streaming_conveyor.Manifest(tmp_path / "_done.json")
    progress = streaming_conveyor.ProgressWriter(tmp_path / "progress.jsonl")
    db = tmp_path / "dedup.sqlite"
    DedupStore(str(db), near=False, commit_every=1).close()
    tokens_by_start: dict[int, list[int]] = {}
    observed_kwargs: list[dict] = []

    def records_provider(_repo, _repo_dir, _work_root, _work_parent, _manifest, _resume):
        return records, 3, "hit"

    def runner(
        range_repo,
        _repo_dir,
        _records_jsonl,
        start,
        end,
        lengths_sorted,
        _repo_work,
        range_dedup_db,
        _dedup_near,
        _pr_store,
        _repo_list,
        _memory_limit_gb,
        _analysis_cache_entries,
        **kwargs,
    ):
        observed_kwargs.append(kwargs)
        assert range_dedup_db == db
        assert kwargs["defer_promote"] is True
        stage_id = f"commit:{range_repo}:r{start}:{end}"
        stage_db = kwargs["deferred_stage_dir"] / f"stage_{start}.sqlite"
        tokens = [10_000 + start * 100 + idx for idx in range(6)]
        tokens_by_start[start] = tokens
        staged = DedupStore(
            str(range_dedup_db),
            near=False,
            commit_every=1,
            stage_id=stage_id,
            stage_db_path=str(stage_db),
        )
        try:
            assert staged.seen_exact_tokens(tokens) is False
        finally:
            staged.close()
        return {
            "source": "commits",
            "repo": range_repo,
            "range": [start, end],
            "lengths": {
                str(lengths_sorted[0]): {
                    "rows": 1,
                    "valid_tokens": 11,
                    "pad_tokens": 0,
                    "capacity_tokens": 11,
                }
            },
            "stage_timings_s": {
                "process_commits_s": 0.01,
                "materialize_s": 0.01,
                "pack_s": 0.01,
                "promote_deferred": 1.0,
            },
            "dedup_stage": {
                "stage_id": stage_id,
                "stage_db": str(stage_db),
            },
        }

    with ThreadPoolExecutor(max_workers=2) as pool:
        done, failed, all_done = streaming_conveyor.run_commits_half(
            repo=repo,
            repo_dir=tmp_path / "repo_src",
            repo_work=tmp_path / "repo_work",
            work_root=tmp_path / "work",
            work_parent=tmp_path / "work_parent",
            lengths_commits=(1024,),
            range_size=1,
            pool=pool,
            manifest=manifest,
            manifest_lock=streaming_conveyor.threading.Lock(),
            resume=True,
            cumulative={"valid": 0},
            dedup_db=db,
            dedup_near=False,
            pr_store=None,
            repo_list=None,
            progress=progress,
            range_target_bytes=0,
            dedup_promote_batch_size=2,
            range_runner_override=runner,
            commit_records_override=records_provider,
        )

    assert (done, failed, all_done) == (3, 0, True)
    range_keys = sorted(key for key in manifest.done if key.startswith("repo::r"))
    assert range_keys == ["repo::r0", "repo::r1", "repo::r2"]
    assert manifest.done["repo::commit_plan"]["n_records"] == 3
    assert manifest.done["repo::commits"] == {
        "source": "commits",
        "repo": "repo",
        "complete": True,
        "completion_proof": "commit_plan_exact_range_coverage",
        "n_records": 3,
        "range_count": 3,
    }
    assert "repo::commits" not in manifest.failed
    assert len(observed_kwargs) == 3
    reader = DedupStore(str(db), near=False, commit_every=1)
    try:
        for tokens in tokens_by_start.values():
            assert reader.seen_exact_tokens(tokens) is True
    finally:
        reader.close()
    done_rows = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "unit_done"
    ]
    batch_sizes = {
        row["stage_timings_s"]["promote_batch_size"]
        for row in done_rows
    }
    assert batch_sizes == {1, 2}
    assert all("dedup_stage" not in info for info in manifest.done.values())


def test_dedup_checkpoint_controller_emits_token_milestone(tmp_path):
    import streaming_conveyor

    db = tmp_path / "dedup.sqlite"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (v INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
    finally:
        conn.close()

    progress_path = tmp_path / "progress.jsonl"
    progress = streaming_conveyor.ProgressWriter(progress_path)
    checkpoints = streaming_conveyor.DedupCheckpointController(
        dedup_db=db,
        interval_tokens=100,
        mode="TRUNCATE",
        busy_timeout_ms=1000,
        progress=progress,
    )

    checkpoints.maybe_checkpoint(99)
    checkpoints.maybe_checkpoint(100)

    rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == ["dedup_checkpoint"]
    assert rows[0]["threshold_tokens"] == 100
    assert rows[0]["cumulative_valid_tokens"] == 100
    assert rows[0]["mode"] == "TRUNCATE"


def test_run_lock_rejects_second_holder(tmp_path):
    import streaming_conveyor

    path = tmp_path / "commits.lock"
    first = streaming_conveyor.RunLock(path)
    second = streaming_conveyor.RunLock(path)
    first.acquire()
    try:
        try:
            second.acquire()
        except RuntimeError as exc:
            assert "already held" in str(exc)
            assert "pid" in path.read_text(encoding="utf-8")
        else:  # pragma: no cover - this is the failure path under pytest.
            raise AssertionError("second lock acquisition unexpectedly succeeded")
    finally:
        first.close()

    second.acquire()
    second.close()


def test_stream_lock_names_for_both_streams():
    import streaming_conveyor

    assert streaming_conveyor.stream_lock_names("both") == ("code", "commits")
    assert streaming_conveyor.stream_lock_names("code") == ("code",)
    assert streaming_conveyor.stream_lock_names("commits") == ("commits",)


def test_default_conveyor_work_parent_lives_under_outputs():
    import streaming_conveyor

    args = streaming_conveyor.parse_args([])

    assert args.work_dir is None
    assert Path(args.work_parent_dir) == streaming_conveyor.DEFAULT_WORK_PARENT
    assert "outputs/conveyor/tmp" in str(streaming_conveyor.DEFAULT_WORK_PARENT)
    assert args.retain_partial_work is False
    assert args.min_free_disk_gb == streaming_conveyor.DEFAULT_MIN_FREE_DISK_GB


def test_parent_work_root_cleanup_honors_retain_partial_work():
    import streaming_conveyor

    assert not streaming_conveyor.should_remove_owned_work_root(
        own_work_root=True,
        keep_temp=False,
        retain_partial_work=True,
    )
    assert not streaming_conveyor.should_remove_owned_work_root(
        own_work_root=True,
        keep_temp=True,
        retain_partial_work=False,
    )
    assert streaming_conveyor.should_remove_owned_work_root(
        own_work_root=True,
        keep_temp=False,
        retain_partial_work=False,
    )


def test_output_root_args_rebase_runtime_paths(tmp_path):
    import streaming_conveyor

    old_code_root = streaming_conveyor.sr.OUTPUT_ROOT
    old_commit_root = streaming_conveyor.src.COMMIT_OUTPUT_ROOT
    old_conveyor_root = streaming_conveyor.CONVEYOR_ROOT
    try:
        args = streaming_conveyor.parse_args(
            [
                "--code-output-root",
                str(tmp_path / "fresh_code"),
                "--commit-output-root",
                str(tmp_path / "fresh_commits"),
                "--conveyor-root",
                str(tmp_path / "fresh_conveyor"),
            ]
        )

        streaming_conveyor.configure_runtime_paths_from_args(args)

        assert streaming_conveyor.sr.OUTPUT_ROOT == tmp_path / "fresh_code"
        assert streaming_conveyor.sr.MANIFEST_PATH == (
            tmp_path / "fresh_code" / "_done.json"
        )
        assert streaming_conveyor.src.COMMIT_OUTPUT_ROOT == (
            tmp_path / "fresh_commits"
        )
        assert streaming_conveyor.src.COMMIT_MANIFEST == (
            tmp_path / "fresh_commits" / "_done.json"
        )
        assert streaming_conveyor.CONVEYOR_ROOT == tmp_path / "fresh_conveyor"
        assert streaming_conveyor.CONVEYOR_MANIFEST == (
            tmp_path / "fresh_conveyor" / "_done.json"
        )
        assert streaming_conveyor.EXTRACT_CACHE_ROOT == (
            tmp_path / "fresh_conveyor" / "extract_cache"
        )
        assert Path(args.progress_jsonl) == tmp_path / "fresh_conveyor" / "progress.jsonl"
        assert Path(args.run_lock_dir) == tmp_path / "fresh_conveyor" / "locks"
        assert Path(args.work_parent_dir) == tmp_path / "fresh_conveyor" / "tmp"
        assert Path(args.reservation_file) == (
            tmp_path / "fresh_conveyor" / "_reservations.json"
        )
    finally:
        streaming_conveyor.configure_output_roots(
            code_output_root=old_code_root,
            commit_output_root=old_commit_root,
            conveyor_root=old_conveyor_root,
        )


def test_min_free_disk_guard_fails_loud_before_staging(tmp_path, monkeypatch):
    from collections import namedtuple

    import pytest

    import streaming_conveyor

    Usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        streaming_conveyor.shutil,
        "disk_usage",
        lambda _path: Usage(total=100 * 1024**3, used=96 * 1024**3, free=4 * 1024**3),
    )

    with pytest.raises(SystemExit) as excinfo:
        streaming_conveyor.ensure_min_free_disk(
            tmp_path,
            10.0,
            context="unit test",
        )

    assert "unsafe conveyor disk state" in str(excinfo.value)
    assert "unit test" in str(excinfo.value)


def test_conveyor_memory_plan_rejects_oversubscribed_parallelism():
    import pytest

    import streaming_conveyor

    with pytest.raises(SystemExit) as excinfo:
        streaming_conveyor.validate_memory_plan(
            streams="both",
            workers=16,
            repo_workers=20,
            memory_limit_gb=10.0,
            memory_budget_gb=64.0,
            allow_oversubscription=False,
        )

    message = str(excinfo.value)
    assert "unsafe conveyor memory plan" in message
    assert "heavy_slots=36" in message
    assert "360.00 GiB" in message


def test_conveyor_memory_plan_allows_safe_parallelism():
    import streaming_conveyor

    plan = streaming_conveyor.validate_memory_plan(
        streams="both",
        workers=3,
        repo_workers=1,
        memory_limit_gb=6.0,
        memory_budget_gb=32.0,
        allow_oversubscription=False,
    )

    assert plan["heavy_slots"] == 4
    assert plan["reserved_gb"] == 24.0


def test_conveyor_memory_plan_accounts_code_and_commit_limits_separately():
    import streaming_conveyor

    plan = streaming_conveyor.validate_memory_plan(
        streams="both",
        workers=20,
        repo_workers=4,
        memory_limit_gb=10.0,
        code_memory_limit_gb=8.0,
        commit_memory_limit_gb=3.0,
        memory_budget_gb=96.0,
        allow_oversubscription=False,
    )

    assert plan["heavy_slots"] == 24
    assert plan["code_reserved_gb"] == 32.0
    assert plan["commit_reserved_gb"] == 60.0
    assert plan["reserved_gb"] == 92.0


def test_should_stage_repo_skips_manifest_proven_complete_both_streams(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "repo::code": {"source": "code"},
            "repo::commit_plan": {"source": "commit_plan", "n_records": 612},
            "repo::r0": {"range": [0, 500], "source": "commits"},
            "repo::r500": {"range": [500, 612], "source": "commits"},
        },
        failed={
            "repo::commits": {
                "stage": "extract_git_history",
                "detail": "stale failure from an earlier attempt",
            }
        },
    )

    assert not streaming_conveyor.should_stage_repo_from_manifest(
        "repo",
        streams="both",
        resume=True,
        manifest=manifest,
        range_size=500,
        only_repos=None,
    )
    assert manifest.done["repo::commits"] == {
        "source": "commits",
        "repo": "repo",
        "complete": True,
        "completion_proof": "commit_plan_exact_range_coverage",
        "n_records": 612,
        "range_count": 2,
    }
    assert "repo::commits" not in manifest.failed
    assert streaming_conveyor.should_stage_repo_from_manifest(
        "repo",
        streams="both",
        resume=False,
        manifest=manifest,
        range_size=500,
        only_repos=None,
    )


def test_should_stage_repo_skips_manifest_known_no_git_for_commit_streams(tmp_path):
    import streaming_conveyor

    manifest = streaming_conveyor.Manifest(
        path=tmp_path / "_done.json",
        done={
            "snapshot-only::no_git": {
                "source": "commits",
                "no_git": True,
                "reason": "missing .git metadata",
            },
        },
        failed={},
    )

    assert not streaming_conveyor.should_stage_repo_from_manifest(
        "snapshot-only",
        streams="commits",
        resume=True,
        manifest=manifest,
        range_size=500,
        only_repos=None,
    )
    assert not streaming_conveyor.should_stage_repo_from_manifest(
        "snapshot-only",
        streams="both",
        resume=True,
        manifest=manifest,
        range_size=500,
        only_repos=None,
    )
    assert streaming_conveyor.should_stage_repo_from_manifest(
        "snapshot-only",
        streams="code",
        resume=True,
        manifest=manifest,
        range_size=500,
        only_repos=None,
    )


def test_conveyor_memory_plan_can_be_explicitly_disabled():
    import streaming_conveyor

    plan = streaming_conveyor.validate_memory_plan(
        streams="both",
        workers=16,
        repo_workers=20,
        memory_limit_gb=10.0,
        memory_budget_gb=0.0,
        allow_oversubscription=False,
    )

    assert plan["reserved_gb"] == 360.0
    assert plan["memory_budget_gb"] == 0.0


def test_unit_reservation_ledger_prevents_duplicate_active_unit(tmp_path):
    import streaming_conveyor

    path = tmp_path / "reservations.json"
    first = streaming_conveyor.UnitReservationLedger(path).acquire(
        "repo::r0",
        stream="commits",
        repo="repo",
    )
    assert first.acquired is True

    second = streaming_conveyor.UnitReservationLedger(path).acquire(
        "repo::r0",
        stream="commits",
        repo="repo",
    )
    assert second.acquired is False
    assert second.holder is not None

    first.release()

    third = streaming_conveyor.UnitReservationLedger(path).acquire(
        "repo::r0",
        stream="commits",
        repo="repo",
    )
    try:
        assert third.acquired is True
    finally:
        third.release()


def test_bounded_future_queue_never_exceeds_submission_window():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    import streaming_conveyor

    active = 0
    max_active = 0
    lock = threading.Lock()
    completed = []

    def work(item):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return item

    with ThreadPoolExecutor(max_workers=8) as pool:
        submitted = streaming_conveyor.run_bounded_future_queue(
            list(range(10)),
            max_pending=3,
            submit=lambda item: (pool.submit(work, item), None),
            handle_done=lambda _item, future, _state: completed.append(future.result()),
        )

    assert submitted == 10
    assert sorted(completed) == list(range(10))
    assert max_active <= 3


def test_commit_stream_finalize_skips_no_git_repo(tmp_path):
    import streaming_reindex_commits as src

    repo_dir = tmp_path / "snapshot-only" / "_src"
    repo_dir.mkdir(parents=True)
    (repo_dir / "main.cpp").write_text("int f() { return 1; }\n", encoding="utf-8")

    assert src._finalize_git_repo_subtree("snapshot-only", repo_dir) is None
    assert not (tmp_path / "snapshot-only").exists()


def test_commit_stream_finalize_reports_no_git_repo(tmp_path):
    import streaming_reindex_commits as src

    repo_dir = tmp_path / "snapshot-only" / "_src"
    repo_dir.mkdir(parents=True)
    (repo_dir / "main.cpp").write_text("int f() { return 1; }\n", encoding="utf-8")

    seen: list[str] = []

    assert (
        src._finalize_git_repo_subtree(
            "snapshot-only",
            repo_dir,
            on_no_git=seen.append,
        )
        is None
    )
    assert seen == ["snapshot-only"]


def test_commit_stream_finalize_keeps_git_repo(tmp_path):
    import streaming_reindex_commits as src

    repo_dir = tmp_path / "real-repo" / "_src"
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / "main.cpp").write_text("int f() { return 1; }\n", encoding="utf-8")

    assert src._finalize_git_repo_subtree("real-repo", repo_dir) == ("real-repo", repo_dir)
    assert repo_dir.exists()


def _stat(rows, valid, target_length):
    """A per-length stat dict in the exact shape produced by
    streaming_reindex._parquet_stats / append_output."""
    capacity = rows * target_length
    pad = capacity - valid
    return {
        "rows": rows,
        "capacity_tokens": capacity,
        "valid_tokens": valid,
        "pad_tokens": pad,
        "pad_frac": (pad / capacity if capacity else 0.0),
    }


def test_length_totals_sums_well_formed_stats():
    import streaming_conveyor

    info = {
        "source": "code",
        "lengths": {
            "1024": _stat(rows=3, valid=2000, target_length=1024),
            "2048": _stat(rows=1, valid=2048, target_length=2048),
        },
    }

    totals = streaming_conveyor._length_totals(info)

    assert totals["rows"] == 4
    assert totals["valid_tokens"] == 2000 + 2048
    assert totals["capacity_tokens"] == 3 * 1024 + 1 * 2048
    assert totals["pad_tokens"] == totals["capacity_tokens"] - totals["valid_tokens"]


def test_primary_bucket_progress_reports_smallest_configured_length():
    import streaming_conveyor

    info = {
        "source": "commits",
        "lengths": {
            "1024": _stat(rows=3, valid=2000, target_length=1024),
            "4096": _stat(rows=1, valid=3000, target_length=4096),
        },
    }

    assert streaming_conveyor._primary_bucket_progress(info, [4096, 1024]) == {
        "primary_bucket_length": 1024,
        "primary_bucket_valid_tokens": 2000,
    }


def test_primary_bucket_progress_is_zero_for_empty_after_dedup():
    import streaming_conveyor

    assert streaming_conveyor._primary_bucket_progress({"lengths": {}}, [1024]) == {
        "primary_bucket_length": 1024,
        "primary_bucket_valid_tokens": 0,
    }


def test_length_totals_raises_on_malformed_stat_dict():
    """RULE #1 regression: a stat dict missing an expected field must crash
    (identifying the offending length bucket + source) instead of silently
    defaulting to 0 and undercounting cumulative valid tokens. The old
    int(st.get(key, 0)) path returned a (wrong) total here without raising."""
    import pytest

    import streaming_conveyor

    good = _stat(rows=2, valid=1500, target_length=1024)
    malformed = _stat(rows=1, valid=2048, target_length=2048)
    del malformed["valid_tokens"]  # simulate a corrupt / truncated stat dict

    info = {
        "source": "commits",
        "lengths": {"1024": good, "2048": malformed},
    }

    with pytest.raises(KeyError) as excinfo:
        streaming_conveyor._length_totals(info)

    message = str(excinfo.value)
    assert "valid_tokens" in message
    assert "2048" in message
    assert "commits" in message
