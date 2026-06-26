from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


MLX_ROOT = Path(__file__).resolve().parents[1]
CLANG_INDEXER = MLX_ROOT / "tools" / "clang_indexer"
if str(CLANG_INDEXER) not in sys.path:
    sys.path.insert(0, str(CLANG_INDEXER))


def test_chunk_claims_are_wal_backed_and_count_limited(tmp_path):
    from dedup_store import DedupStore

    db = tmp_path / "dedup.sqlite"
    first = DedupStore(str(db), near=False, commit_every=1)
    second = DedupStore(str(db), near=False, commit_every=1)
    try:
        assert first.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

        assert first.claim_chunk_tokens([10, 20, 30], namespace="strict") is True
        assert second.claim_chunk_tokens([10, 20, 30], namespace="strict") is False

        assert first.claim_chunk_tokens([10, 20, 30], namespace="repeat3", max_count=3)
        assert second.claim_chunk_tokens([10, 20, 30], namespace="repeat3", max_count=3)
        assert first.claim_chunk_tokens([10, 20, 30], namespace="repeat3", max_count=3)
        assert not second.claim_chunk_tokens(
            [10, 20, 30],
            namespace="repeat3",
            max_count=3,
        )
    finally:
        first.close()
        second.close()

    conn = sqlite3.connect(db)
    try:
        rows = dict(
            conn.execute(
                "SELECT namespace, claim_count FROM chunk_claims ORDER BY namespace"
            ).fetchall()
        )
    finally:
        conn.close()

    assert rows == {"repeat3": 3, "strict": 1}
