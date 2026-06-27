from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


MLX_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = MLX_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


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


def test_commit_stream_finalize_skips_no_git_repo(tmp_path):
    import streaming_reindex_commits as src

    repo_dir = tmp_path / "snapshot-only" / "_src"
    repo_dir.mkdir(parents=True)
    (repo_dir / "main.cpp").write_text("int f() { return 1; }\n", encoding="utf-8")

    assert src._finalize_git_repo_subtree("snapshot-only", repo_dir) is None
    assert not (tmp_path / "snapshot-only").exists()


def test_commit_stream_finalize_keeps_git_repo(tmp_path):
    import streaming_reindex_commits as src

    repo_dir = tmp_path / "real-repo" / "_src"
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / "main.cpp").write_text("int f() { return 1; }\n", encoding="utf-8")

    assert src._finalize_git_repo_subtree("real-repo", repo_dir) == ("real-repo", repo_dir)
    assert repo_dir.exists()
