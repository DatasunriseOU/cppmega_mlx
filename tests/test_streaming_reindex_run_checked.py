from __future__ import annotations

import sys


def test_run_checked_times_out_fail_loud(tmp_path) -> None:
    import streaming_reindex

    log_path = tmp_path / "sleep.log"
    try:
        streaming_reindex.run_checked(
            "repo",
            "index_project",
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            log_path=log_path,
            timeout=1,
        )
    except streaming_reindex.RepoFailure as exc:
        assert exc.repo == "repo"
        assert exc.stage == "index_project"
        assert "timed out after 1s" in exc.detail
    else:  # pragma: no cover
        raise AssertionError("expected run_checked to fail loud on timeout")

    assert "started" in log_path.read_text(encoding="utf-8")


def test_run_checked_stall_watchdog_fail_loud(tmp_path) -> None:
    import streaming_reindex

    log_path = tmp_path / "stall.log"
    try:
        streaming_reindex.run_checked(
            "repo",
            "index_project",
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            log_path=log_path,
            timeout=20,
            stall_timeout=1,
        )
    except streaming_reindex.RepoFailure as exc:
        assert exc.repo == "repo"
        assert exc.stage == "index_project"
        assert "stalled after 1s without log or CPU progress" in exc.detail
    else:  # pragma: no cover
        raise AssertionError("expected run_checked to fail loud on stall")

    assert "started" in log_path.read_text(encoding="utf-8")


def test_run_checked_stall_watchdog_allows_cpu_progress(tmp_path) -> None:
    import streaming_reindex

    log_path = tmp_path / "cpu-progress.log"
    streaming_reindex.run_checked(
        "repo",
        "index_project",
        [
            sys.executable,
            "-c",
            (
                "import time\n"
                "print('started', flush=True)\n"
                "deadline = time.time() + 2.0\n"
                "x = 0\n"
                "while time.time() < deadline:\n"
                "    x += 1\n"
            ),
        ],
        log_path=log_path,
        timeout=10,
        stall_timeout=1,
    )

    assert "started" in log_path.read_text(encoding="utf-8")


def test_parse_ps_time_seconds_variants() -> None:
    import streaming_reindex

    assert streaming_reindex._parse_ps_time_seconds("01:02.50") == 62.5
    assert streaming_reindex._parse_ps_time_seconds("03:01:02.50") == 10862.5
    assert streaming_reindex._parse_ps_time_seconds("2-03:01:02.50") == 183662.5


def test_stream_repo_subtrees_source_cache_only(tmp_path) -> None:
    import streaming_reindex

    cache = tmp_path / "cache"
    repo_dir = cache / "repo" / "src"
    repo_dir.mkdir(parents=True)
    (cache / "repo" / streaming_reindex.SOURCE_CACHE_SENTINEL).write_text(
        "{}\n",
        encoding="utf-8",
    )
    (repo_dir / "x.cc").write_text("int x;\n", encoding="utf-8")

    yielded = list(
        streaming_reindex.stream_repo_subtrees(
            tmp_path / "work",
            lambda repo: repo == "repo",
            source_cache_dir=cache,
            source_cache_only=True,
        )
    )

    assert yielded == [("repo", cache / "repo")]


def test_stream_repo_subtrees_source_cache_skips_bare_git_repos(tmp_path) -> None:
    import streaming_reindex

    cache = tmp_path / "cache"
    bare = cache / "repo.bare"
    bare.mkdir(parents=True)
    (bare / streaming_reindex.SOURCE_CACHE_SENTINEL).write_text(
        "{}\n",
        encoding="utf-8",
    )

    yielded = list(
        streaming_reindex.stream_repo_subtrees(
            tmp_path / "work",
            lambda repo: streaming_reindex.is_code_worktree_repo(repo),
            source_cache_dir=cache,
            source_cache_only=True,
        )
    )

    assert yielded == []


def test_stream_repo_dirs_yields_extracted_src_without_tar(tmp_path) -> None:
    import streaming_reindex

    root = tmp_path / "root"
    (root / "repo" / "_src").mkdir(parents=True)
    (root / "repo" / "_src" / "x.cc").write_text("int x;\n", encoding="utf-8")
    (root / "repo.bare").mkdir(parents=True)

    yielded = list(
        streaming_reindex.stream_repo_dirs(
            [root],
            lambda repo: streaming_reindex.is_code_worktree_repo(repo),
        )
    )

    assert yielded == [("repo", root / "repo" / "_src")]


def test_populate_source_cache_only_materializes_cache(tmp_path, monkeypatch) -> None:
    import streaming_reindex

    cache = tmp_path / "cache"
    calls = []

    def fake_stream(work_root, should_process, *, source_cache_dir, source_cache_only):
        calls.append((work_root, source_cache_dir, source_cache_only))
        for repo in ("repo-a", "repo.bare", "repo-b"):
            if should_process(repo):
                yield repo, source_cache_dir / repo

    monkeypatch.setattr(streaming_reindex, "stream_repo_subtrees", fake_stream)

    ready = []
    report = streaming_reindex.populate_source_cache(
        tmp_path / "work",
        streaming_reindex.is_code_worktree_repo,
        cache,
        max_repos=1,
        on_repo_ready=lambda repo, path, count: ready.append((repo, path, count)),
    )

    assert calls == [(tmp_path / "work", cache, False)]
    assert ready == [("repo-a", cache / "repo-a", 1)]
    assert report == {
        "source_cache_dir": str(cache),
        "repos": [{"repo": "repo-a", "path": str(cache / "repo-a")}],
        "repo_count": 1,
    }


def test_source_cache_populate_rejects_commit_source_before_processing(
    tmp_path,
    monkeypatch,
) -> None:
    import streaming_reindex

    commit_source = tmp_path / "commits.jsonl"
    commit_source.write_text("{}\n", encoding="utf-8")
    calls = []

    def fake_process_one_commit_source(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("commit source should not be processed")

    monkeypatch.setattr(
        streaming_reindex,
        "OUTPUT_ROOT",
        tmp_path / "outputs" / "reindexed",
    )
    monkeypatch.setattr(
        streaming_reindex,
        "MANIFEST_PATH",
        tmp_path / "outputs" / "reindexed" / "_done.json",
    )
    monkeypatch.setattr(
        streaming_reindex,
        "process_one_commit_source",
        fake_process_one_commit_source,
    )

    try:
        streaming_reindex.main(
            [
                "--source-cache-dir",
                str(tmp_path / "cache"),
                "--source-cache-populate-only",
                "--commit-source",
                f"sample={commit_source}",
            ]
        )
    except SystemExit as exc:
        assert "--source-cache-populate-only is code-only" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected source-cache populate validation failure")

    assert calls == []
