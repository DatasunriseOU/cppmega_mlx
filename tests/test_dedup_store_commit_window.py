"""Regression tests for the cross-process near-dedup commit window.

ONE dedup SQLite DB is shared by several concurrent producers (the code stage,
the commit stage, and --repo-workers>1). Under WAL, a writer's UNCOMMITTED
near-dup reference docs (minhash signature + lsh band rows) are invisible to the
other connections, so the pending-buffer size IS the per-writer cross-process
near-dedup leak window: while a near-dup reference doc sits buffered, a second
writer cannot see it and may accept the SAME near-duplicate. Exact dups have a
backstop (chunk_claims commits immediately, exact is INSERT-OR-IGNORE) but near
dups do NOT, so the default buffer must stay modest.

These tests use real sqlite + real datasketch MinHash (no mocks).
"""
from __future__ import annotations

import sqlite3
import sys
import threading
import time
import inspect
from pathlib import Path


MLX_ROOT = Path(__file__).resolve().parents[1]
CLANG_INDEXER = MLX_ROOT / "tools" / "clang_indexer"
if str(CLANG_INDEXER) not in sys.path:
    sys.path.insert(0, str(CLANG_INDEXER))


def test_default_pending_buffer_is_modest():
    """The default near-dedup leak window must not silently regress (was 1000)."""
    from dedup_store import DedupStore

    # A wide default (e.g. 1000) lets concurrent writers both accept the same
    # near-duplicate for up to that many decisions. Keep it modest.
    assert DedupStore.MAX_PENDING_BEFORE_COMMIT <= 128


def test_near_refs_become_cross_connection_visible_within_the_window(tmp_path):
    """A buffered near-dup reference doc is hidden from a second connection until
    the pending buffer flushes; after exactly MAX_PENDING_BEFORE_COMMIT distinct
    inserts it is committed and visible. This bounds the cross-process leak."""
    from dedup_store import DedupStore

    window = DedupStore.MAX_PENDING_BEFORE_COMMIT
    assert window >= 2  # need a strictly-pre-flush observation point

    db = tmp_path / "dedup.sqlite"
    # commit_every large so commits are driven solely by MAX_PENDING_BEFORE_COMMIT
    # (threshold = min(commit_every, MAX_PENDING_BEFORE_COMMIT)).
    writer = DedupStore(str(db), near=True, commit_every=10_000)
    # A separate connection stands in for a second concurrent producer process.
    reader = sqlite3.connect(str(db))
    try:

        def make_tokens(i: int) -> list[int]:
            # Distinct, non-overlapping token windows so none are near-dups of
            # each other -> every call persists a new reference doc.
            base = (i + 1) * 1000
            return list(range(base, base + 64))

        # Insert window-1 distinct reference docs: still buffered, NOT committed,
        # so the second connection sees zero rows (this is the leak window).
        for i in range(window - 1):
            assert writer.seen_near_tokens(make_tokens(i)) is False
        assert reader.execute("SELECT COUNT(*) FROM minhash").fetchone()[0] == 0

        # The window-th insert trips _mark_pending -> commit; now all are visible.
        assert writer.seen_near_tokens(make_tokens(window - 1)) is False
        assert (
            reader.execute("SELECT COUNT(*) FROM minhash").fetchone()[0] == window
        )
    finally:
        reader.close()
        writer.close()


def test_near_dedup_queries_persisted_sqlite_bands_without_preload(tmp_path):
    """A fresh worker must not rebuild the whole persisted LSH into RAM.

    The global conveyor DB can contain millions of minhash signatures. Startup
    must query the persisted ``lsh`` band index lazily per document, while still
    producing the same near-duplicate decision.
    """
    from dedup_store import DedupStore

    db = tmp_path / "dedup.sqlite"
    base = list(range(10_000, 10_128))
    near_same = [*base[:-1], 999_999]

    first = DedupStore(str(db), near=True, commit_every=1)
    try:
        assert first.seen_near_tokens(base) is False
    finally:
        first.close()

    second = DedupStore(str(db), near=True, commit_every=1)
    try:
        assert second._loaded_count == 0
        assert second.seen_near_tokens(near_same) is True
    finally:
        second.close()


def test_staged_exact_and_chunk_claims_are_invisible_until_promoted(tmp_path):
    """A failed unit must not poison global exact/chunk dedup state."""
    from dedup_store import DedupStore

    db = tmp_path / "dedup.sqlite"
    tokens = [101, 102, 103, 104, 105, 106]

    failed = DedupStore(str(db), near=False, commit_every=1, stage_id="code:repo")
    try:
        assert failed.seen_exact_tokens(tokens) is False
        assert failed.claim_chunk_tokens(tokens, namespace="semantic_chunk:v1") is True
    finally:
        failed.close()

    # Closing a staged writer commits only the staging rows. A later successful
    # unit must still be able to emit the same function/chunk if the first unit
    # never promoted.
    retry = DedupStore(str(db), near=False, commit_every=1, stage_id="code:repo:retry")
    try:
        assert retry.seen_exact_tokens(tokens) is False
        assert retry.claim_chunk_tokens(tokens, namespace="semantic_chunk:v1") is True
        retry.promote_stage()
    finally:
        retry.close()

    committed_reader = DedupStore(str(db), near=False, commit_every=1)
    try:
        assert committed_reader.seen_exact_tokens(tokens) is True
        assert (
            committed_reader.claim_chunk_tokens(tokens, namespace="semantic_chunk:v1")
            is False
        )
    finally:
        committed_reader.close()

    # The abandoned stage can be explicitly discarded after the failed unit is
    # marked failed; discard is idempotent for resume cleanup.
    DedupStore.discard_stage(str(db), "code:repo")
    DedupStore.discard_stage(str(db), "code:repo")


def test_staged_near_refs_are_not_global_until_promoted(tmp_path):
    """Near-dedup references from failed units must not suppress later data."""
    from dedup_store import DedupStore

    db = tmp_path / "dedup.sqlite"
    base = list(range(10_000, 10_128))
    near_same = [*base[:-1], 999_999]

    failed = DedupStore(str(db), near=True, commit_every=1, stage_id="commit:r0")
    try:
        assert failed.seen_near_tokens(base) is False
    finally:
        failed.close()

    # The first stage was not promoted, so another stage must not see its MinHash
    # as global training data.
    retry = DedupStore(str(db), near=True, commit_every=1, stage_id="commit:r0:retry")
    try:
        assert retry.seen_near_tokens(near_same) is False
        retry.promote_stage()
    finally:
        retry.close()

    committed_reader = DedupStore(str(db), near=True, commit_every=1)
    try:
        assert committed_reader.seen_near_tokens(base) is True
    finally:
        committed_reader.close()


def test_staged_near_doc_ids_are_shared_across_handles(tmp_path):
    """One pipeline unit may open multiple DedupStore handles for one stage."""
    from dedup_store import DedupStore

    db = tmp_path / "dedup.sqlite"
    first_tokens = list(range(20_000, 20_128))
    second_tokens = list(range(30_000, 30_128))

    first = DedupStore(str(db), near=True, commit_every=1, stage_id="code:repo")
    try:
        assert first.seen_near_tokens(first_tokens) is False
    finally:
        first.close()

    second = DedupStore(str(db), near=True, commit_every=1, stage_id="code:repo")
    try:
        assert second.seen_near_tokens(second_tokens) is False
        second.promote_stage()
    finally:
        second.close()

    reader = DedupStore(str(db), near=True, commit_every=1)
    try:
        assert reader.seen_near_tokens([*first_tokens[:-1], 999_001]) is True
        assert reader.seen_near_tokens([*second_tokens[:-1], 999_002]) is True
    finally:
        reader.close()


def test_local_stage_db_writes_do_not_block_on_global_writer_lock(tmp_path):
    """Subprocesses stage claims locally; parent promotes after append success."""
    from dedup_store import DedupStore

    global_db = tmp_path / "dedup.sqlite"
    stage_db = tmp_path / "rwork" / "commit-r0.stage.sqlite"
    stage_db.parent.mkdir()
    exact_tokens = [701, 702, 703, 704, 705, 706]
    near_tokens = list(range(40_000, 40_128))
    chunk_tokens = [801, 802, 803, 804, 805, 806]
    stage_id = "commit:r0"

    init = DedupStore(str(global_db), near=True, commit_every=1)
    init.close()

    blocker = sqlite3.connect(str(global_db), timeout=0.1)
    try:
        blocker.execute("BEGIN IMMEDIATE")

        staged = DedupStore(
            str(global_db),
            near=True,
            commit_every=1,
            stage_id=stage_id,
            stage_db_path=str(stage_db),
        )
        try:
            assert staged.seen_exact_tokens(exact_tokens) is False
            assert staged.seen_near_tokens(near_tokens) is False
            assert (
                staged.claim_chunk_tokens(
                    chunk_tokens,
                    namespace="semantic_chunk:v1",
                )
                is True
            )
        finally:
            staged.close()

        global_reader = sqlite3.connect(str(global_db))
        try:
            assert global_reader.execute("SELECT COUNT(*) FROM exact").fetchone()[0] == 0
            assert global_reader.execute("SELECT COUNT(*) FROM minhash").fetchone()[0] == 0
            assert (
                global_reader.execute("SELECT COUNT(*) FROM chunk_claims").fetchone()[0]
                == 0
            )
        finally:
            global_reader.close()

        local_reader = sqlite3.connect(str(stage_db))
        try:
            assert (
                local_reader.execute(
                    "SELECT COUNT(*) FROM exact_stage WHERE stage_id=?",
                    (stage_id,),
                ).fetchone()[0]
                == 1
            )
            assert (
                local_reader.execute(
                    "SELECT COUNT(*) FROM minhash_stage WHERE stage_id=?",
                    (stage_id,),
                ).fetchone()[0]
                == 1
            )
            assert (
                local_reader.execute(
                    "SELECT COUNT(*) FROM chunk_claims_stage WHERE stage_id=?",
                    (stage_id,),
                ).fetchone()[0]
                == 1
            )
        finally:
            local_reader.close()
    finally:
        blocker.rollback()
        blocker.close()

    DedupStore.promote_stage_from_db(str(global_db), str(stage_db), stage_id)

    committed_reader = DedupStore(str(global_db), near=True, commit_every=1)
    try:
        assert committed_reader.seen_exact_tokens(exact_tokens) is True
        assert committed_reader.seen_near_tokens([*near_tokens[:-1], 999_100]) is True
        assert (
            committed_reader.claim_chunk_tokens(
                chunk_tokens,
                namespace="semantic_chunk:v1",
            )
            is False
        )
    finally:
        committed_reader.close()

    stage_reader = sqlite3.connect(str(stage_db))
    try:
        assert (
            stage_reader.execute(
                "SELECT COUNT(*) FROM dedup_stages WHERE stage_id=?",
                (stage_id,),
            ).fetchone()[0]
            == 0
        )
    finally:
        stage_reader.close()


def test_local_stage_db_uses_fast_temp_pragmas_without_weakening_global_db(tmp_path):
    """Local stage DBs are discardable rwork ledgers; global DB stays durable."""
    from dedup_store import DedupStore

    global_db = tmp_path / "dedup.sqlite"
    stage_db = tmp_path / "rwork" / "code-repo.stage.sqlite"
    stage_db.parent.mkdir()

    DedupStore(str(global_db), near=False, commit_every=1).close()

    staged = DedupStore(
        str(global_db),
        near=False,
        commit_every=1,
        stage_id="code:repo",
        stage_db_path=str(stage_db),
    )
    try:
        global_sync = staged.conn.execute("PRAGMA synchronous").fetchone()[0]
        local_sync = staged.stage_conn.execute("PRAGMA synchronous").fetchone()[0]
        local_journal = staged.stage_conn.execute("PRAGMA journal_mode").fetchone()[0]
        global_autockpt = staged.conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
    finally:
        staged.close()

    assert global_sync == 2  # read-only connection reports default FULL
    assert global_autockpt == 1000  # read-only connection default, not tuned here
    assert local_sync == 0
    assert local_journal == "memory"
    assert not Path(f"{stage_db}-wal").exists()


def test_local_stage_promote_uses_set_based_bulk_sql_not_python_row_loop():
    """Local stage promotion is a hot parent path; keep it set-based in SQLite."""
    from dedup_store import DedupStore

    source = inspect.getsource(DedupStore.promote_stage_from_db)

    assert "staged_minhashes = list" not in source
    assert "for stage_doc_id, sig in staged_minhashes" not in source
    assert "ROW_NUMBER() OVER" in source


def test_streaming_reindex_promotes_or_discards_staged_claims(tmp_path):
    """Parent conveyor helpers own commit-on-success semantics."""
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import streaming_reindex
    from dedup_store import DedupStore

    db = tmp_path / "dedup.sqlite"
    promoted_tokens = [401, 402, 403, 404, 405, 406]
    discarded_tokens = [501, 502, 503, 504, 505, 506]

    promoted = DedupStore(str(db), near=False, commit_every=1, stage_id="code:ok")
    try:
        assert promoted.seen_exact_tokens(promoted_tokens) is False
    finally:
        promoted.close()
    streaming_reindex.promote_dedup_stage(db, "code:ok")

    discarded = DedupStore(str(db), near=False, commit_every=1, stage_id="code:bad")
    try:
        assert discarded.seen_exact_tokens(discarded_tokens) is False
    finally:
        discarded.close()
    streaming_reindex.discard_dedup_stage(db, "code:bad")

    reader = DedupStore(str(db), near=False, commit_every=1)
    try:
        assert reader.seen_exact_tokens(promoted_tokens) is True
        assert reader.seen_exact_tokens(discarded_tokens) is False
    finally:
        reader.close()


def test_streaming_reindex_promote_reports_wait_and_duration(tmp_path):
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import streaming_reindex
    from dedup_store import DedupStore

    db = tmp_path / "dedup.sqlite"
    tokens = [601, 602, 603, 604, 605, 606]
    staged = DedupStore(str(db), near=False, commit_every=1, stage_id="code:ok")
    try:
        assert staged.seen_exact_tokens(tokens) is False
    finally:
        staged.close()

    metrics = streaming_reindex.promote_dedup_stage(db, "code:ok")

    assert set(metrics) == {"promote_wait_s", "promote_duration_s"}
    assert metrics["promote_wait_s"] >= 0
    assert metrics["promote_duration_s"] >= 0


def test_streaming_reindex_batch_promotes_local_stage_dbs(tmp_path):
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import streaming_reindex
    from dedup_store import DedupStore

    db = tmp_path / "dedup.sqlite"
    DedupStore(str(db), near=True, commit_every=1).close()

    stages = []
    expected_exact = []
    expected_near = []
    expected_chunks = []
    for idx in range(2):
        stage_id = f"commit:repo:r{idx}"
        stage_db = tmp_path / f"stage_{idx}.sqlite"
        exact_tokens = [10_000 + idx * 100 + n for n in range(6)]
        near_tokens = [20_000 + idx * 1_000 + n for n in range(128)]
        chunk_tokens = [30_000 + idx * 100 + n for n in range(6)]
        staged = DedupStore(
            str(db),
            near=True,
            commit_every=1,
            stage_id=stage_id,
            stage_db_path=str(stage_db),
        )
        try:
            assert staged.seen_exact_tokens(exact_tokens) is False
            assert staged.seen_near_tokens(near_tokens) is False
            assert staged.claim_chunk_tokens(
                chunk_tokens,
                namespace="semantic_chunk:v1",
            ) is True
        finally:
            staged.close()
        stages.append((stage_id, stage_db))
        expected_exact.append(exact_tokens)
        expected_near.append(near_tokens)
        expected_chunks.append(chunk_tokens)

    metrics = streaming_reindex.promote_dedup_stages(db, stages)

    assert metrics["promote_batch_size"] == 2
    assert metrics["promote_wait_s"] >= 0
    assert metrics["promote_duration_s"] >= 0
    reader = DedupStore(str(db), near=True, commit_every=1)
    try:
        for exact_tokens in expected_exact:
            assert reader.seen_exact_tokens(exact_tokens) is True
        for near_tokens in expected_near:
            assert reader.seen_near_tokens([*near_tokens[:-1], 999_999]) is True
        for chunk_tokens in expected_chunks:
            assert reader.claim_chunk_tokens(
                chunk_tokens,
                namespace="semantic_chunk:v1",
            ) is False
    finally:
        reader.close()


def test_streaming_reindex_serializes_global_dedup_promotes(tmp_path):
    """Parent stage promotion must be serialized before touching global SQLite."""
    import fcntl
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import streaming_reindex
    from dedup_store import DedupStore

    db = tmp_path / "dedup.sqlite"
    DedupStore(str(db), near=False, commit_every=1).close()

    lock_path = db.with_name(db.name + ".promote.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    blocker = lock_path.open("a+b")
    fcntl.flock(blocker.fileno(), fcntl.LOCK_EX)

    done = threading.Event()
    errors: list[BaseException] = []

    def promote() -> None:
        try:
            streaming_reindex.promote_dedup_stage(db, "empty-stage")
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=promote)
    thread.start()
    try:
        time.sleep(0.2)
        assert not done.is_set(), "promote bypassed the global promote lock"
    finally:
        fcntl.flock(blocker.fileno(), fcntl.LOCK_UN)
        blocker.close()

    thread.join(5.0)
    assert done.is_set(), "promote did not resume after lock release"
    assert not errors
