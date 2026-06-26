"""Global, cross-process, resumable exact + near-duplicate store (SQLite-backed).

This module implements ONE shared dedup store used by BOTH the code indexer
(``index_project.py``) and the commit indexer (``process_commits.py``), so that
deduplication is global across every repo and across both the code and commit
streams. The store lives in a single SQLite database file (the ``--dedup-db``
path passed by the drivers).

Design (RULE #1 — fail loud, never silent):
  * SQLite is opened in WAL mode with ``synchronous=NORMAL`` so multiple
    processes can read/write the same file concurrently and safely.
  * Exact dedup: table ``exact(hash BLOB PRIMARY KEY)``. We ``INSERT OR IGNORE``
    the sha1 of the *normalized* body; ``cursor.rowcount == 0`` means the row
    already existed -> the document is an exact duplicate.
  * Near dedup: ``datasketch`` MinHashLSH (threshold=0.7, num_perm=256) over
    5-gram word shingles of the normalized body. The LSH bands are persisted in
    table ``lsh(band_id INT, band_hash BLOB, doc_id INT)`` indexed on
    ``(band_id, band_hash)`` so the index is global and resumable across runs
    and processes. We also persist each accepted document's MinHash signature in
    ``minhash(doc_id INTEGER PRIMARY KEY, sig BLOB)`` so we can verify the TRUE
    Jaccard similarity (>= 0.7) of a candidate against its band neighbours
    BEFORE deciding it is a near-duplicate (LSH gives candidates only).

  If a ``--dedup-db`` path is given but SQLite cannot be opened, or ``datasketch``
  cannot be imported, we RAISE. There is no in-memory / degraded fallback when a
  db path is requested. The legacy in-RAM ``set()`` path lives in the callers and
  is used ONLY when no ``--dedup-db`` is given.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import struct
import time
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Normalization (NO alpha-rename): strip comments, collapse whitespace, trim.
# --------------------------------------------------------------------------- #
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_WS_RUN = re.compile(r"\s+")


def normalize_body(text: str) -> str:
    """Normalize a document body for dedup hashing/shingling.

    Strips ``//`` line comments and ``/* ... */`` block comments, collapses every
    run of whitespace to a single space, and trims. Does NOT rename identifiers
    (no alpha-rename) so semantically-distinct code is not collapsed.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize_body expected str, got {type(text)!r}")
    # Order matters: remove block comments first (may span lines), then line
    # comments, then collapse whitespace.
    s = _BLOCK_COMMENT.sub(" ", text)
    s = _LINE_COMMENT.sub(" ", s)
    s = _WS_RUN.sub(" ", s)
    return s.strip()


def _sha1(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).digest()


def _shingles(normalized: str, k: int = 5) -> list[str]:
    """5-gram word shingles over the normalized body."""
    words = normalized.split(" ")
    if len(words) < k:
        # Whole body is a single shingle when shorter than k words.
        return [" ".join(words)] if words and words[0] else []
    return [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]


def _sha1_tokens(token_ids: Sequence[int]) -> bytes:
    """Stable long hash of a token-id sequence.

    This is the CANONICAL exact-dedup key per the corrected design: the function
    (or commit doc) is hashed AFTER OUR tokenizer, so the whitespace sentinels
    (<SPACE>/<NL>) have already canonicalized formatting. Two functions that
    differ only in formatting tokenize to the SAME id sequence and collapse here.
    """
    if not isinstance(token_ids, (list, tuple)):
        raise TypeError(
            f"_sha1_tokens expected list/tuple of ints, got {type(token_ids)!r}"
        )
    # Pack as little-endian uint32; vocab is 65536 so 4 bytes/id is exact.
    buf = struct.pack(f"<{len(token_ids)}I", *(int(t) & 0xFFFFFFFF for t in token_ids))
    return hashlib.sha1(buf).digest()


def _token_shingles(token_ids: Sequence[int], k: int = 5) -> list[bytes]:
    """k-gram shingles over the token-id sequence (for MinHash near-dup).

    Each shingle is the packed bytes of ``k`` consecutive token ids. Operating on
    token ids (not text words) keeps the near-dup signal aligned with the same
    canonical, format-collapsed representation used by the exact hash.
    """
    ids = [int(t) & 0xFFFFFFFF for t in token_ids]
    if len(ids) < k:
        if not ids:
            return []
        return [struct.pack(f"<{len(ids)}I", *ids)]
    return [
        struct.pack(f"<{k}I", *ids[i:i + k])
        for i in range(len(ids) - k + 1)
    ]


# --------------------------------------------------------------------------- #
# Store.
# --------------------------------------------------------------------------- #
class DedupStore:
    """Cross-process SQLite-backed exact + near-duplicate store.

    Usage (CANONICAL token-id path — hash AFTER OUR tokenizer):
        store = DedupStore(db_path, near=True)
        if store.seen_exact_tokens(token_ids):   # exact dup -> drop
            ...
        elif store.seen_near_tokens(token_ids):  # near dup (Jaccard>=0.7) -> drop
            ...
        store.commit()               # periodically (e.g. every 1000 docs)

    The legacy text path (seen_exact/seen_near over normalize_body) is retained
    for callers that have not yet tokenized; the corrected dedup design uses the
    *_tokens methods so the dedup key is exactly sha1(token_ids).
    """

    NUM_PERM = 256
    THRESHOLD = 0.7
    SHINGLE_K = 5
    SQLITE_TIMEOUT_SECONDS = 300.0
    SQLITE_BUSY_TIMEOUT_MS = int(SQLITE_TIMEOUT_SECONDS * 1000)
    WRITE_RETRY_SLEEP_SECONDS = 0.05
    WRITE_RETRY_MAX_SLEEP_SECONDS = 2.0
    MAX_PENDING_BEFORE_COMMIT = 32

    def __init__(self, db_path: str, *, near: bool = True, commit_every: int = 1000):
        if not db_path:
            raise ValueError("DedupStore requires a non-empty db_path")
        self.db_path = str(db_path)
        self.near_enabled = bool(near)
        self.commit_every = int(commit_every)
        self._pending = 0

        # FAIL LOUD: sqlite open failure raises (no in-memory fallback).
        try:
            self.conn = sqlite3.connect(
                self.db_path,
                timeout=self.SQLITE_TIMEOUT_SECONDS,
            )
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"DedupStore: failed to open sqlite db at {self.db_path!r}: {exc}"
            ) from exc

        # Cross-process safe pragmas.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(f"PRAGMA busy_timeout={self.SQLITE_BUSY_TIMEOUT_MS}")
        self._init_schema()

        # Lazy datasketch objects; built only when near dedup is enabled.
        self._lsh = None
        self._minhash_cls = None
        if self.near_enabled:
            self._init_near()

    # ----------------------------------------------------------------- #
    def _init_schema(self) -> None:
        c = self.conn
        self._execute_write("CREATE TABLE IF NOT EXISTS exact (hash BLOB PRIMARY KEY)")
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS lsh ("
            "band_id INTEGER NOT NULL, band_hash BLOB NOT NULL, doc_id INTEGER NOT NULL)"
        )
        self._execute_write(
            "CREATE INDEX IF NOT EXISTS lsh_band ON lsh (band_id, band_hash)"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS minhash ("
            "doc_id INTEGER PRIMARY KEY, sig BLOB NOT NULL)"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS dedup_meta (key TEXT PRIMARY KEY, val INTEGER)"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS chunk_claims ("
            "namespace TEXT NOT NULL, "
            "hash BLOB NOT NULL, "
            "claim_count INTEGER NOT NULL, "
            "PRIMARY KEY(namespace, hash))"
        )
        # next doc_id counter for near-dup docs.
        self._execute_write(
            "INSERT OR IGNORE INTO dedup_meta (key, val) VALUES ('next_doc_id', 0)"
        )
        c.commit()

    def _execute_write(self, sql: str, params: tuple = ()):
        """Execute a SQLite write with bounded retry on cross-process writer locks.

        The corpus conveyor runs code and commit stages concurrently against one
        global dedup DB. SQLite WAL allows this, but only one writer can commit at
        a time. We keep fail-loud semantics: lock contention is retried for a
        bounded window, then raised with WHERE context instead of falling back.
        """
        deadline = time.monotonic() + self.SQLITE_TIMEOUT_SECONDS
        sleep_s = self.WRITE_RETRY_SLEEP_SECONDS
        while True:
            try:
                return self.conn.execute(sql, params)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "DedupStore: sqlite write remained locked after "
                        f"{self.SQLITE_TIMEOUT_SECONDS:.0f}s at {self.db_path!r}"
                    ) from exc
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 1.5, self.WRITE_RETRY_MAX_SLEEP_SECONDS)

    def _init_near(self) -> None:
        # FAIL LOUD: datasketch import failure raises (no fallback).
        try:
            from datasketch import MinHash, MinHashLSH
        except Exception as exc:  # noqa: BLE001 - re-raise loudly
            raise RuntimeError(
                "DedupStore: datasketch import failed; near-dup requires datasketch "
                f"(pip install datasketch). Underlying error: {exc}"
            ) from exc

        self._minhash_cls = MinHash
        # In-process LSH index. We rebuild it from the persisted lsh/minhash
        # tables so the near index is resumable across runs.
        self._lsh = MinHashLSH(threshold=self.THRESHOLD, num_perm=self.NUM_PERM)
        self._load_persisted_lsh()

    def _load_persisted_lsh(self) -> None:
        """Repopulate the in-process LSH from persisted minhash signatures."""
        assert self._lsh is not None and self._minhash_cls is not None
        cur = self.conn.execute("SELECT doc_id, sig FROM minhash")
        n = 0
        for doc_id, sig in cur:
            mh = self._minhash_from_blob(sig)
            key = str(doc_id)
            # Guard against re-insert if called twice.
            if key not in self._lsh:
                self._lsh.insert(key, mh)
                n += 1
        self._loaded_count = n

    # ----------------------------------------------------------------- #
    # MinHash (de)serialization.
    # ----------------------------------------------------------------- #
    def _build_minhash(self, normalized: str):
        mh = self._minhash_cls(num_perm=self.NUM_PERM)
        for sh in _shingles(normalized, self.SHINGLE_K):
            mh.update(sh.encode("utf-8", errors="replace"))
        return mh

    def _build_minhash_tokens(self, token_ids: Sequence[int]):
        mh = self._minhash_cls(num_perm=self.NUM_PERM)
        for sh in _token_shingles(token_ids, self.SHINGLE_K):
            mh.update(sh)
        return mh

    @staticmethod
    def _minhash_to_blob(mh) -> bytes:
        arr = mh.hashvalues  # numpy uint64 array, length num_perm
        return struct.pack(f"<{len(arr)}Q", *(int(x) for x in arr))

    def _minhash_from_blob(self, blob: bytes):
        n = len(blob) // 8
        vals = struct.unpack(f"<{n}Q", blob)
        import numpy as np
        mh = self._minhash_cls(
            num_perm=self.NUM_PERM,
            hashvalues=np.array(vals, dtype=np.uint64),
        )
        return mh

    # ----------------------------------------------------------------- #
    # Public API.
    # ----------------------------------------------------------------- #
    def seen_exact(self, text: str) -> bool:
        """Return True if this body's normalized sha1 was already seen.

        Inserts the hash atomically (INSERT OR IGNORE); a 0 rowcount means the
        row already existed -> exact duplicate.
        """
        h = _sha1(normalize_body(text))
        cur = self._execute_write(
            "INSERT OR IGNORE INTO exact (hash) VALUES (?)", (h,)
        )
        self._mark_pending()
        return cur.rowcount == 0

    def seen_near(self, text: str) -> bool:
        """Return True if a near-duplicate (true Jaccard >= 0.7) already exists.

        On a non-duplicate, the document's MinHash signature and LSH bands are
        persisted (global + resumable) so future documents dedup against it.
        """
        if not self.near_enabled:
            raise RuntimeError("seen_near called but near dedup is disabled")
        normalized = normalize_body(text)
        mh = self._build_minhash(normalized)

        candidates = self._lsh.query(mh)
        for cand_key in candidates:
            cand_doc_id = int(cand_key)
            cand_mh = self._load_minhash(cand_doc_id)
            if cand_mh is None:
                continue
            # Verify TRUE Jaccard (LSH only gives candidates).
            if mh.jaccard(cand_mh) >= self.THRESHOLD:
                return True

        # Not a near-dup: persist signature + bands so it becomes a reference.
        doc_id = self._next_doc_id()
        sig = self._minhash_to_blob(mh)
        self._execute_write(
            "INSERT INTO minhash (doc_id, sig) VALUES (?, ?)", (doc_id, sig)
        )
        self._persist_bands(doc_id, mh)
        self._lsh.insert(str(doc_id), mh)
        self._mark_pending()
        return False

    # ----------------------------------------------------------------- #
    # Token-id-keyed API (CANONICAL per corrected dedup design).
    #
    # These hash the function/commit-doc AFTER OUR TOKENIZER: the exact key is
    # sha1(token_ids) and the near MinHash shingles are over token-id n-grams.
    # The whitespace sentinels (<SPACE>/<NL>) have already canonicalized format
    # in the id sequence, so format-only differences collapse to one row. Both
    # the exact set and the near LSH/minhash tables are SHARED with the text API
    # (same tables), so the store is global across code (token-id path) and
    # commits (token-id path) within one db file.
    # ----------------------------------------------------------------- #
    def seen_exact_tokens(self, token_ids: Sequence[int]) -> bool:
        """Return True if sha1(token_ids) was already seen (exact duplicate)."""
        h = _sha1_tokens(token_ids)
        cur = self._execute_write(
            "INSERT OR IGNORE INTO exact (hash) VALUES (?)", (h,)
        )
        self._mark_pending()
        return cur.rowcount == 0

    def seen_near_tokens(self, token_ids: Sequence[int]) -> bool:
        """Return True if a near-dup (true Jaccard >= 0.7) over token-id shingles
        already exists. Persists this doc's signature + bands otherwise."""
        if not self.near_enabled:
            raise RuntimeError("seen_near_tokens called but near dedup is disabled")
        mh = self._build_minhash_tokens(token_ids)

        candidates = self._lsh.query(mh)
        for cand_key in candidates:
            cand_doc_id = int(cand_key)
            cand_mh = self._load_minhash(cand_doc_id)
            if cand_mh is None:
                continue
            if mh.jaccard(cand_mh) >= self.THRESHOLD:
                return True

        doc_id = self._next_doc_id()
        sig = self._minhash_to_blob(mh)
        self._execute_write(
            "INSERT INTO minhash (doc_id, sig) VALUES (?, ?)", (doc_id, sig)
        )
        self._persist_bands(doc_id, mh)
        self._lsh.insert(str(doc_id), mh)
        self._mark_pending()
        return False

    def claim_chunk_tokens(
        self,
        token_ids: Sequence[int],
        *,
        namespace: str = "train_chunk",
        max_count: int = 1,
    ) -> bool:
        """Claim one semantic training chunk by its tokenized body.

        This is separate from the ``exact``/``minhash`` dedup tables. Exact/near
        dedup decides whether a function may be emitted as a root document;
        chunk claims decide whether that already-tokenized function/class/type
        body may appear anywhere in the training stream. The unit of ownership is
        still the caller-provided semantic chunk; the hash is only the SQLite key.

        Returns True when this call successfully claimed one slot. Returns False
        when the tokenized chunk has already reached ``max_count`` in ``namespace``.
        The write is committed immediately so concurrent conveyor processes see
        the claim before they assemble overlapping 1k/2k/4k/8k buckets.
        """
        if max_count < 1:
            raise ValueError("claim_chunk_tokens requires max_count >= 1")
        if not namespace:
            raise ValueError("claim_chunk_tokens requires a non-empty namespace")
        h = _sha1_tokens(token_ids)
        cur = self._execute_write(
            "INSERT INTO chunk_claims(namespace, hash, claim_count) "
            "VALUES (?, ?, 1) "
            "ON CONFLICT(namespace, hash) DO UPDATE SET "
            "claim_count = claim_count + 1 "
            "WHERE claim_count < ?",
            (namespace, h, int(max_count)),
        )
        # Claims are coordination records, not buffered dedup rows. Commit them
        # immediately so other WAL connections see the slot before emitting docs.
        self.commit()
        return cur.rowcount == 1

    # ----------------------------------------------------------------- #
    def _load_minhash(self, doc_id: int):
        row = self.conn.execute(
            "SELECT sig FROM minhash WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return self._minhash_from_blob(row[0])

    def _persist_bands(self, doc_id: int, mh) -> None:
        """Persist this doc's LSH band hashes into the lsh table.

        We mirror datasketch's internal banding by reading the in-process LSH
        keys after insert is not portable; instead we re-derive band hashes
        directly from the signature using the LSH's (b, r) configuration.
        """
        b = self._lsh.b
        r = self._lsh.r
        hv = mh.hashvalues
        for band_id in range(b):
            start = band_id * r
            band_slice = hv[start:start + r]
            band_hash = hashlib.sha1(
                struct.pack(f"<{len(band_slice)}Q", *(int(x) for x in band_slice))
            ).digest()
            self._execute_write(
                "INSERT INTO lsh (band_id, band_hash, doc_id) VALUES (?, ?, ?)",
                (band_id, band_hash, doc_id),
            )

    def _next_doc_id(self) -> int:
        cur = self._execute_write(
            "UPDATE dedup_meta SET val = val + 1 WHERE key='next_doc_id'"
        )
        if cur.rowcount != 1:
            raise RuntimeError("DedupStore: failed to advance next_doc_id counter")
        row = self.conn.execute(
            "SELECT val FROM dedup_meta WHERE key='next_doc_id'"
        ).fetchone()
        return int(row[0])

    def _mark_pending(self) -> None:
        self._pending += 1
        threshold = min(self.commit_every, self.MAX_PENDING_BEFORE_COMMIT)
        if self._pending >= threshold:
            self.commit()

    def commit(self) -> None:
        self.conn.commit()
        self._pending = 0

    def close(self) -> None:
        try:
            self.commit()
        finally:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
