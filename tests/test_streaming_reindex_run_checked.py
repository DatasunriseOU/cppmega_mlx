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
