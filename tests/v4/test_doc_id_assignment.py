"""V7-G02: cross-shard doc_id stability."""

from __future__ import annotations

from cppmega_v4.data.doc_id_assignment import (
    assign_stable_doc_ids, manifest_from_assignment,
)


def test_v7_g02_same_text_same_id_in_one_shard():
    rows = [
        {"text": "alpha"}, {"text": "beta"},
        {"text": "alpha"},  # duplicate → same id
    ]
    ids = assign_stable_doc_ids(rows)
    assert ids[0] == ids[2]
    assert ids[0] != ids[1]


def test_v7_g02_continuity_across_two_shards():
    """A doc that appears in shard N and N+1 must keep its id."""
    shard_a = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]
    ids_a = assign_stable_doc_ids(shard_a)
    manifest = manifest_from_assignment(shard_a, ids_a)
    shard_b = [
        {"text": "doc2"},   # repeat — must inherit id
        {"text": "doc4"},   # new
        {"text": "doc1"},   # repeat — must inherit id
    ]
    ids_b = assign_stable_doc_ids(shard_b, seed_ids=manifest)
    # Lookup
    by_text = dict(zip(["doc1", "doc2", "doc3"], ids_a))
    assert ids_b[0] == by_text["doc2"]
    assert ids_b[2] == by_text["doc1"]
    # New doc4 gets a fresh id > any existing.
    assert ids_b[1] not in by_text.values()


def test_v7_g02_continuity_across_three_shards():
    shards = [
        [{"text": "X"}, {"text": "Y"}],
        [{"text": "Y"}, {"text": "Z"}],
        [{"text": "X"}, {"text": "Z"}, {"text": "W"}],
    ]
    manifest: dict[str, int] = {}
    all_ids: list[int] = []
    for shard in shards:
        ids = assign_stable_doc_ids(shard, seed_ids=manifest)
        all_ids.extend(ids)
        # Merge: keep prior entries + add any new ones from this shard.
        merged = dict(manifest)
        merged.update(manifest_from_assignment(shard, ids))
        manifest = merged
    # Map text → seen ids across all shards.
    by_text: dict[str, set[int]] = {}
    flat_rows = [r for s in shards for r in s]
    for r, i in zip(flat_rows, all_ids):
        by_text.setdefault(r["text"], set()).add(i)
    for t, idset in by_text.items():
        assert len(idset) == 1, (
            f"text {t!r} drifted to multiple ids: {idset}")


def test_v7_g02_default_signature_is_text_sha256():
    rows = [{"text": "hello"}, {"foo": 1}]
    ids = assign_stable_doc_ids(rows)
    # No-text row → empty text signature; two distinct rows still
    # get distinct ids (one for "hello", one for "" empty).
    assert ids[0] != ids[1]
