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
        assert "stalled after 1s without log progress" in exc.detail
    else:  # pragma: no cover
        raise AssertionError("expected run_checked to fail loud on stall")

    assert "started" in log_path.read_text(encoding="utf-8")


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
