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
