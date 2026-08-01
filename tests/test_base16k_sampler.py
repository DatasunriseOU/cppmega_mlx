from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


MLX_ROOT = Path(__file__).resolve().parents[1]
for p in (
    MLX_ROOT,
    MLX_ROOT / "scripts",
    MLX_ROOT / "tools" / "clang_indexer",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


SYMBOL_SCHEMA = """
CREATE TABLE symbols (
    qname        TEXT NOT NULL,
    base_lib     TEXT NOT NULL,
    base_repo    TEXT NOT NULL,
    kind         INTEGER NOT NULL,
    sym_type     TEXT NOT NULL,
    file         TEXT NOT NULL,
    line         INTEGER NOT NULL,
    end_line     INTEGER NOT NULL,
    is_public    INTEGER NOT NULL,
    token_est    INTEGER NOT NULL,
    body_len     INTEGER NOT NULL,
    text         TEXT NOT NULL,
    symbol_key   TEXT NOT NULL,
    symbol_id    TEXT NOT NULL,
    PRIMARY KEY (qname, sym_type)
);
"""


def _make_symbol_index(path: Path, body: str) -> None:
    from cppmega_mlx.data.symbol_identity import compute_symbol_id

    symbol_key = (
        "repo_file_location:project=boost;file=boost/base.hpp;"
        "line=10;column=0;kind=func;qname=boost::base_fn"
    )
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SYMBOL_SCHEMA)
        conn.execute(
            "INSERT INTO symbols VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "boost::base_fn",
                "boost",
                "boost",
                2,
                "func",
                "boost/base.hpp",
                10,
                20,
                1,
                len(body) // 4,
                len(body),
                body,
                symbol_key,
                f"{compute_symbol_id(symbol_key):016x}",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_base16k_sampler_caps_exact_function_form_at_three(tmp_path):
    from dedup_store import DedupStore
    from scripts.crossrepo.export_base16k_sampler import (
        SEMANTIC_CHUNK_NAMESPACE,
        build_repeat_docs,
        load_candidate_symbols,
    )
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    body = """
int base_fn(int x) {
    int y = x + 1;
    y = y * 2;
    y = y - 3;
    y = y + 4;
    y = y / 2;
    return y;
}
""".strip()
    symbol_index = tmp_path / "global_symbols.sqlite"
    dedup_db = tmp_path / "dedup.sqlite"
    tokenizer_path = MLX_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
    _make_symbol_index(symbol_index, body)

    symbols = load_candidate_symbols(
        symbol_index,
        libs=("boost",),
        limit=10,
        offset=0,
        min_body_len=1,
        max_token_est=16_384,
    )
    docs, stats = build_repeat_docs(
        symbols,
        tokenizer_path=tokenizer_path,
        dedup_db=dedup_db,
        repeats=3,
        max_count=3,
    )
    assert len(docs) == 3
    assert stats["docs_emitted"] == 3
    assert all(doc["symbol_identity_schema_version"] == 3 for doc in docs)
    assert all(len(doc["symbol_identities"]) == 1 for doc in docs)
    assert all(
        doc["chunk_boundaries"][0]["symbol_id"]
        == doc["symbol_identities"][0]["symbol_id"]
        for doc in docs
    )
    assert all(
        doc["symbol_ids"]
        == [doc["symbol_identities"][0]["symbol_id"]] * len(doc["text"])
        for doc in docs
    )

    docs_again, stats_again = build_repeat_docs(
        symbols,
        tokenizer_path=tokenizer_path,
        dedup_db=dedup_db,
        repeats=3,
        max_count=3,
    )
    assert docs_again == []
    assert stats_again["claims_rejected"] == 1

    conn = sqlite3.connect(dedup_db)
    try:
        assert conn.execute(
            "SELECT claim_count FROM chunk_claims WHERE namespace=?",
            (SEMANTIC_CHUNK_NAMESPACE,),
        ).fetchone()[0] == 3
    finally:
        conn.close()

    second_db = tmp_path / "dedup_with_existing.sqlite"
    tok = load_cppmega_tokenizer(str(tokenizer_path))
    store = DedupStore(str(second_db), near=False, commit_every=1)
    try:
        assert store.claim_chunk_tokens(
            tok.encode(body),
            namespace=SEMANTIC_CHUNK_NAMESPACE,
            max_count=3,
        )
    finally:
        store.close()

    docs_after_strict_claim, stats_after_strict_claim = build_repeat_docs(
        symbols,
        tokenizer_path=tokenizer_path,
        dedup_db=second_db,
        repeats=3,
        max_count=3,
    )
    assert len(docs_after_strict_claim) == 2
    assert stats_after_strict_claim["docs_emitted"] == 2
