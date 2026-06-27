from __future__ import annotations

import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.fix_packed_parquet_boundaries import (
    TokenPieces,
    _needed_insert_positions,
    _shift_offset,
    _shift_offset_exclusive,
    _shift_span,
    repair_row,
)
from scripts.nanochat_data.pack_enriched_rows import _loss_mask_for_packed_docs
from scripts.render_sidecar_example import get_tokenizer


def _expected_loss_mask(doc_ids_valid: list[int], *, capacity: int) -> list[int]:
    """Producer rule: 1 iff doc_ids[pos]==doc_ids[pos+1] in the valid prefix."""
    keep = []
    for pos in range(len(doc_ids_valid)):
        if pos + 1 >= len(doc_ids_valid):
            keep.append(0)
        else:
            keep.append(1 if doc_ids_valid[pos] == doc_ids_valid[pos + 1] else 0)
    return keep + [0] * (capacity - len(keep))


def _multi_doc_row(segments: list[tuple[str, int]], *, slack: int = 8) -> dict:
    """Build a packed row from >=1 (text, doc_id) docs, mirroring the producer.

    Token-aligned columns are sized to capacity; ``doc_ids`` carries the real
    per-token document id (distinct per segment) and is padded with the
    producer's ``pad_doc_id``; ``loss_mask`` is the producer's inter-document
    mask.  This is the realistic shape the boundary fixer must preserve.
    """
    tok = get_tokenizer()
    ids: list[int] = []
    doc_id_per_token: list[int] = []
    source_doc_ids: list[int] = []
    source_doc_token_lengths: list[int] = []
    for text, doc_id in segments:
        seg_ids = list(tok.encode(text))
        ids.extend(seg_ids)
        doc_id_per_token.extend([int(doc_id)] * len(seg_ids))
        source_doc_ids.append(int(doc_id))
        source_doc_token_lengths.append(len(seg_ids))
    valid = len(ids)
    capacity = valid + slack
    pad_doc_id = max(len(source_doc_ids), max(doc_id_per_token, default=0))
    padded = ids + [0] * slack
    loss_mask = _loss_mask_for_packed_docs(doc_id_per_token, target_length=capacity)
    return {
        "input_ids": padded,
        "target_ids": padded[1:] + [0],
        "loss_mask": loss_mask,
        "doc_ids": doc_id_per_token + [pad_doc_id] * slack,
        "platform_ids": [1] * valid + [0] * slack,
        "token_structure_ids": [6] * valid + [0] * slack,
        "token_dep_levels": [0] * capacity,
        "token_ast_depth": [0] * capacity,
        "token_sibling_index": [0] * capacity,
        "token_ast_node_type": [0] * capacity,
        "token_symbol_ids": [0] * capacity,
        "token_call_targets": [0] * capacity,
        "token_type_refs": [0] * capacity,
        "token_def_use": [0] * capacity,
        "token_change_mask_pre": [0] * capacity,
        "token_change_mask_post": [0] * capacity,
        "hunk_id_per_token": [-1] * valid + [0] * slack,
        "edit_op_per_token": [0] * capacity,
        "token_chunk_starts": [0, max(1, valid // 2)],
        "token_chunk_ends": [max(1, valid // 2), valid],
        "changed_chunk_spans": [{"start": max(1, valid // 2), "end": valid}],
        "source_doc_ids": source_doc_ids,
        "source_doc_token_lengths": source_doc_token_lengths,
        "valid_token_count": valid,
        "trained_token_count": sum(loss_mask),
        "slack_tokens": slack,
    }


def _row_for_text(text: str, *, slack: int = 8) -> dict:
    tok = get_tokenizer()
    ids = list(tok.encode(text))
    valid = len(ids)
    capacity = valid + slack
    padded = ids + [0] * slack
    return {
        "input_ids": padded,
        "target_ids": padded[1:] + [0],
        "loss_mask": [1] * max(valid - 1, 0) + [0] * (capacity - max(valid - 1, 0)),
        "doc_ids": [7] * valid + [0] * slack,
        "platform_ids": [1] * valid + [0] * slack,
        "token_structure_ids": [6] * valid + [0] * slack,
        "token_dep_levels": [0] * capacity,
        "token_ast_depth": [0] * capacity,
        "token_sibling_index": [0] * capacity,
        "token_ast_node_type": [0] * capacity,
        "token_symbol_ids": [0] * capacity,
        "token_call_targets": [0] * capacity,
        "token_type_refs": [0] * capacity,
        "token_def_use": [0] * capacity,
        "token_change_mask_pre": [0] * capacity,
        "token_change_mask_post": [0] * capacity,
        "hunk_id_per_token": [-1] * valid + [0] * slack,
        "edit_op_per_token": [0] * capacity,
        "token_chunk_starts": [0, max(1, valid // 2)],
        "token_chunk_ends": [max(1, valid // 2), valid],
        "changed_chunk_spans": [{"start": max(1, valid // 2), "end": valid}],
        "source_doc_ids": [7],
        "source_doc_token_lengths": [valid],
        "valid_token_count": valid,
        "trained_token_count": max(valid - 1, 0),
        "slack_tokens": slack,
    }


def test_repair_row_inserts_newlines_around_generated_commit_marker() -> None:
    tok = get_tokenizer()
    pieces = TokenPieces(tok)
    row = _row_for_text(
        "/* can't compile; see https://github.com/a/b. */// === PRE-COMMIT ===bool f();\n"
    )

    repaired, info = repair_row(row, pieces)

    text = tok.decode(repaired["input_ids"][: repaired["valid_token_count"]])
    assert info == {"changed": True, "insertions": 2, "overflow": False}
    assert "*/// === PRE-COMMIT" not in text
    assert "*/\n// === PRE-COMMIT ===\nbool f();" in text
    assert repaired["valid_token_count"] == row["valid_token_count"] + 2
    # Single doc: every interior token is same-doc, so the producer rule yields
    # all-ones except the final valid token -> trained == valid - 1 (== num_docs 1).
    assert repaired["trained_token_count"] == sum(repaired["loss_mask"])
    assert repaired["trained_token_count"] == repaired["valid_token_count"] - 1
    assert repaired["target_ids"] == repaired["input_ids"][1:] + [0]
    assert repaired["loss_mask"][repaired["trained_token_count"] - 1] == 1
    assert repaired["loss_mask"][repaired["trained_token_count"]] == 0
    assert repaired["source_doc_token_lengths"] == [row["source_doc_token_lengths"][0] + 2]
    assert repaired["token_chunk_ends"][-1] == row["token_chunk_ends"][-1] + 2
    assert repaired["changed_chunk_spans"][0]["end"] == row["changed_chunk_spans"][0]["end"] + 2


def test_repair_row_preserves_inter_document_loss_masking_multi_doc() -> None:
    """C1 regression: a repaired MULTI-doc row must keep inter-document masking.

    The buggy implementation rebuilt ``loss_mask`` as all-ones over the valid
    prefix, which (a) trained the model to predict document B's first token from
    document A's last token across unrelated packed commits, and (b) inflated
    ``trained_token_count`` by ``num_docs - 1``.  This fixture has two distinct
    docs in one packed row where all-ones != correct, so it FAILS on the old
    code and PASSES once loss_mask is recomputed from the repaired doc_ids.
    """
    tok = get_tokenizer()
    pieces = TokenPieces(tok)
    # Doc A carries the collapsed generated marker (triggers two <NL> inserts);
    # doc B is an unrelated second document concatenated into the same row.
    row = _multi_doc_row(
        [
            ("/* can't compile. */// === PRE-COMMIT ===bool f();\n", 7),
            ("int g(int x) { return x + 1; }\n", 9),
        ],
        slack=16,
    )
    num_docs = len({int(d) for d in row["source_doc_ids"]})
    assert num_docs == 2

    repaired, info = repair_row(row, pieces)

    assert info == {"changed": True, "insertions": 2, "overflow": False}
    new_valid = repaired["valid_token_count"]
    assert new_valid == row["valid_token_count"] + 2
    capacity = len(repaired["input_ids"])

    # The repaired loss_mask must equal the producer rule applied to the
    # REPAIRED doc_ids over the valid prefix (boundary/last/pad all 0).
    valid_doc_ids = [int(d) for d in repaired["doc_ids"][:new_valid]]
    expected_mask = _expected_loss_mask(valid_doc_ids, capacity=capacity)
    assert repaired["loss_mask"] == expected_mask

    # There must be a genuine interior inter-document boundary, and loss_mask
    # MUST be 0 there.  The old all-ones rebuild would put a 1 here.
    boundaries = [
        pos
        for pos in range(new_valid - 1)
        if valid_doc_ids[pos] != valid_doc_ids[pos + 1]
    ]
    assert boundaries, "fixture must contain an interior document boundary"
    for pos in boundaries:
        assert repaired["loss_mask"][pos] == 0

    # trained_token_count == sum(loss_mask) == valid - num_docs (one dropped
    # token per doc: the inter-doc boundary token and the final valid token).
    assert repaired["trained_token_count"] == sum(repaired["loss_mask"])
    assert repaired["trained_token_count"] == new_valid - num_docs

    # Guard the regression explicitly: the old all-ones rebuild would have
    # produced trained == new_valid - 1 and a 1 at the boundary.  Confirm the
    # repaired row is NOT that.
    assert repaired["trained_token_count"] != new_valid - 1
    assert any(repaired["loss_mask"][pos] == 0 for pos in boundaries)
    # Padding is never trained.
    assert all(v == 0 for v in repaired["loss_mask"][new_valid:])


def test_repair_row_refuses_when_no_padding_slack() -> None:
    tok = get_tokenizer()
    pieces = TokenPieces(tok)
    row = _row_for_text(
        "/* can't compile. */// === DIFF ===diff --git a/x b/x\n",
        slack=0,
    )

    repaired, info = repair_row(row, pieces)

    assert repaired is row
    assert info == {"changed": False, "insertions": 2, "overflow": True}


def test_repair_row_ignores_nongenerated_markerlike_comments() -> None:
    """MEDIUM finding: ordinary ``// === ... ===`` comments that merely CONTAIN a
    marker word (DIFF / CONTEXT / PRE-COMMIT / POST-COMMIT) are NOT generated
    markers and must be left untouched.

    The old substring heuristic (``any(name in marker_text ...)``) matched these
    real source comments and silently injected <NL> tokens into training text.
    Each comment below has the exact ``// === ... ===`` structure (so it reaches
    the classification step) but is not a verbatim generated marker.
    """
    tok = get_tokenizer()
    pieces = TokenPieces(tok)
    for comment in (
        "// === DIFF ALGORITHM ===",
        "// === CONTEXT SWITCH ===",
        "// === PRE-COMMIT HOOKS ===",
        "// === POST-COMMIT HOOK SETUP ===",  # no colon -> not the POST-COMMIT form
        "// === NOTES ===",
    ):
        row = _row_for_text(f"int x = 0;{comment}int y = 1;\n")
        before_ids = list(row["input_ids"])
        repaired, info = repair_row(row, pieces)
        assert info == {"changed": False, "insertions": 0, "overflow": False}, comment
        assert repaired is row, comment
        assert repaired["input_ids"] == before_ids, comment


def test_repair_row_still_repairs_context_and_postcommit_markers() -> None:
    """The exact-form anchor must still accept genuine generated markers,
    including CONTEXT and POST-COMMIT (whose subject is variable)."""
    tok = get_tokenizer()
    pieces = TokenPieces(tok)
    for text in (
        "/* can't compile. */// === CONTEXT ===bool f();\n",
        "/* can't compile. */// === POST-COMMIT: fix the parser bug ===bool f();\n",
        # empty subject still starts with the POST-COMMIT prefix.
        "/* can't compile. */// === POST-COMMIT:  ===bool f();\n",
    ):
        row = _row_for_text(text)
        repaired, info = repair_row(row, pieces)
        assert info["changed"] is True, text
        assert info["insertions"] == 2, text
        decoded = tok.decode(repaired["input_ids"][: repaired["valid_token_count"]])
        assert "*/// ===" not in decoded, text


def test_shift_helpers_use_inclusive_start_exclusive_end_semantics() -> None:
    """LOW finding: an insertion landing exactly on an EXCLUSIVE end (one-past-
    last) must not extend the chunk; an INCLUSIVE start at the same index moves
    with its token."""
    positions = [10]
    # Inclusive start at the insertion index moves right (keeps its token).
    assert _shift_offset(10, positions) == 11
    # Exclusive end at the insertion index does NOT move (does not absorb <NL>).
    assert _shift_offset_exclusive(10, positions) == 10
    # Strictly-inside coordinates move for both.
    assert _shift_offset(12, positions) == 13
    assert _shift_offset_exclusive(12, positions) == 13
    # Coordinates before the insertion never move.
    assert _shift_offset(9, positions) == 9
    assert _shift_offset_exclusive(9, positions) == 9
    # Spans: start inclusive, end exclusive.
    span = _shift_span({"start": 10, "end": 10}, positions)
    assert span == {"start": 11, "end": 10}


def test_repair_row_chunk_end_not_overshifted_at_marker_edge() -> None:
    """LOW finding end-to-end: a chunk whose EXCLUSIVE end coincides with a real
    insertion position must not be over-shifted to absorb the inserted <NL>.

    Insertions happen at marker boundaries, which is exactly where chunk edges
    fall, so the off-by-one is systematic, not hypothetical.  Positions are read
    from the real tokenizer-driven detector, then chunk edges are placed on the
    first insertion index; we assert exclusive-end vs inclusive-start divergence.
    """
    tok = get_tokenizer()
    pieces = TokenPieces(tok)
    row = _row_for_text(
        "/* can't compile. */// === PRE-COMMIT ===bool f();\n",
        slack=16,
    )
    positions = _needed_insert_positions(row, pieces)
    assert len(positions) == 2
    edge = positions[0]  # an insertion index that coincides with a chunk edge
    valid = row["valid_token_count"]

    # Place an inclusive start AND an exclusive end exactly at `edge`; the span
    # likewise starts and ends at `edge`.
    row["token_chunk_starts"] = [0, edge]
    row["token_chunk_ends"] = [edge, valid]
    row["changed_chunk_spans"] = [{"start": edge, "end": edge}]

    repaired, info = repair_row(row, pieces)
    assert info["changed"] is True

    before = sum(1 for p in positions if p < edge)   # insertions strictly before
    upto = sum(1 for p in positions if p <= edge)    # insertions up to & incl.
    assert upto == before + 1  # an insertion lands exactly on `edge`

    # Exclusive end shifts only by insertions strictly before it.
    assert repaired["token_chunk_ends"][0] == edge + before
    # Inclusive start shifts by insertions up to and including it.
    assert repaired["token_chunk_starts"][1] == edge + upto
    # Span: start inclusive, end exclusive.
    assert repaired["changed_chunk_spans"][0]["start"] == edge + upto
    assert repaired["changed_chunk_spans"][0]["end"] == edge + before
    # The inserted <NL> at `edge` is NOT absorbed into the prior chunk: its
    # exclusive end stays strictly below the next chunk's (shifted) start.
    assert repaired["token_chunk_ends"][0] < repaired["token_chunk_starts"][1]
    # The trailing exclusive end (strictly inside all insertions) shifts fully.
    assert repaired["token_chunk_ends"][1] == valid + len(positions)


def test_dry_run_fail_on_remaining_rejects_repairable_marker_rows(tmp_path) -> None:
    row = _row_for_text("/* can't compile. */// === PRE-COMMIT ===bool f();\n")
    path = tmp_path / "commits" / "1024" / "bad.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/fix_packed_parquet_boundaries.py",
            str(tmp_path / "commits"),
            "--dry-run",
            "--workers",
            "1",
            "--report",
            str(tmp_path / "report.json"),
            "--fail-on-remaining",
        ],
        check=False,
    )

    assert result.returncode != 0


def test_real_run_fail_on_remaining_accepts_repaired_marker_rows(tmp_path) -> None:
    row = _row_for_text("/* can't compile. */// === PRE-COMMIT ===bool f();\n")
    root = tmp_path / "commits"
    path = root / "1024" / "bad.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), path)

    subprocess.run(
        [
            sys.executable,
            "scripts/fix_packed_parquet_boundaries.py",
            str(root),
            "--workers",
            "1",
            "--report",
            str(tmp_path / "repair_report.json"),
            "--fail-on-remaining",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/fix_packed_parquet_boundaries.py",
            str(root),
            "--dry-run",
            "--workers",
            "1",
            "--report",
            str(tmp_path / "post_report.json"),
            "--fail-on-remaining",
        ],
        check=True,
    )
