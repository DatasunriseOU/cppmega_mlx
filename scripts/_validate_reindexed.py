#!/usr/bin/env python3
"""Validate re-indexed packed parquet outputs (per-channel populated %, edge
block-coordinate correctness, whole-function packing, padding).

Reads outputs/reindexed/{1024,2048,4096,8192,16384}/*.parquet and prints a table.
Run with the mlx venv python and PYTHONPATH=/Volumes/external/sources/cppmega.mlx.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/Volumes/external/sources/cppmega.mlx/outputs/reindexed")
LENGTHS = [1024, 2048, 4096, 8192, 16384]


def col(t, name):
    return t.column(name).to_pylist() if name in t.column_names else None


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def validate_length(tl: int) -> dict:
    d = ROOT / str(tl)
    files = sorted(d.glob("*.parquet"))
    agg = {
        "files": len(files),
        "rows": 0,
        "valid_tokens": 0,
        "capacity": 0,
        "pad_tokens": 0,
        # channel A
        "rows_with_platform": 0,
        # channel B
        "rows_with_chunks": 0,
        "rows_with_structure": 0,
        # channel C
        "rows_with_type_edges": 0,
        "rows_with_symbol": 0,
        "rows_with_def_use": 0,
        "rows_with_type_refs": 0,
        "rows_with_call_edges": 0,
        # channel D
        "rows_with_changed_chunks": 0,
        "rows_with_edit_op": 0,
        # integrity
        "edge_coord_violations": 0,
        "chunk_coord_violations": 0,
        "whole_fn_violations": 0,
        "multi_doc_rows": 0,
    }
    for f in files:
        t = pq.read_table(f)
        n = t.num_rows
        agg["rows"] += n
        vtc = col(t, "valid_token_count") or [0] * n
        input_ids = col(t, "input_ids") or [[]] * n
        num_docs = col(t, "num_docs") or [1] * n
        plat = col(t, "platform_ids")
        tplat = col(t, "token_platform_ids")
        cstarts = col(t, "token_chunk_starts")
        cends = col(t, "token_chunk_ends")
        tstruct = col(t, "token_structure_ids")
        tedges = col(t, "token_type_edges")
        celledges = col(t, "token_call_edges")
        sym = col(t, "token_symbol_ids")
        dfu = col(t, "token_def_use")
        tref = col(t, "token_type_refs")
        cci = col(t, "changed_chunk_ids")
        eop = col(t, "edit_op_per_token")

        for i in range(n):
            row_len = len(input_ids[i]) if input_ids and input_ids[i] is not None else tl
            agg["capacity"] += row_len
            v = vtc[i] or 0
            agg["valid_tokens"] += v
            agg["pad_tokens"] += max(0, row_len - v)
            if num_docs[i] and num_docs[i] > 1:
                agg["multi_doc_rows"] += 1
            # A: row carries a non-empty platform signature
            prow = plat[i] if plat else None
            if prow:
                agg["rows_with_platform"] += 1
            # B: chunk density + per-token structure
            n_chunks = len(cstarts[i]) if cstarts and cstarts[i] is not None else 0
            if n_chunks > 0:
                agg["rows_with_chunks"] += 1
            if tstruct and tstruct[i] and any(x != 0 for x in tstruct[i]):
                agg["rows_with_structure"] += 1
            # C: semantic graph channels
            if tedges and tedges[i]:
                agg["rows_with_type_edges"] += 1
            if celledges and celledges[i]:
                agg["rows_with_call_edges"] += 1
            if sym and sym[i] and any(x != 0 for x in sym[i]):
                agg["rows_with_symbol"] += 1
            if dfu and dfu[i] and any(x != 0 for x in dfu[i]):
                agg["rows_with_def_use"] += 1
            if tref and tref[i] and any(x != 0 for x in tref[i]):
                agg["rows_with_type_refs"] += 1
            # D: commit edit-signal channels
            if cci and cci[i]:
                agg["rows_with_changed_chunks"] += 1
            if eop and eop[i] and any(x != 0 for x in eop[i]):
                agg["rows_with_edit_op"] += 1

            # ----- integrity -----
            # chunk_starts/ends are TOKEN coords in [0, valid_token_count)
            if cstarts and cstarts[i]:
                for s in cstarts[i]:
                    if not (0 <= s <= v):
                        agg["chunk_coord_violations"] += 1
                        break
            if cends and cends[i]:
                for e in cends[i]:
                    if not (0 <= e <= v):
                        agg["chunk_coord_violations"] += 1
                        break
            # edges endpoints + changed_chunk_ids are CHUNK indices in [0, n_chunks)
            def edge_ok(edges):
                for ed in edges:
                    fr = ed["from"] if isinstance(ed, dict) else ed[0]
                    to = ed["to"] if isinstance(ed, dict) else ed[1]
                    if not (0 <= fr < n_chunks and 0 <= to < n_chunks):
                        return False
                return True
            if tedges and tedges[i] and not edge_ok(tedges[i]):
                agg["edge_coord_violations"] += 1
            if celledges and celledges[i] and not edge_ok(celledges[i]):
                agg["edge_coord_violations"] += 1
            if cci and cci[i]:
                if any(not (0 <= c < n_chunks) for c in cci[i]):
                    agg["edge_coord_violations"] += 1

            # whole-function: every doc's tokens are contiguous and the row is
            # not split mid-doc. valid_token_count must be <= row_len and chunks
            # must lie within the valid region (already checked). A split would
            # show as chunk coords exceeding v; counted above. Additionally a
            # single oversized doc must own its row (num_docs==1 and v>tl).
            if v > tl and num_docs[i] != 1:
                agg["whole_fn_violations"] += 1
    return agg


def main() -> int:
    print("Re-indexed packed parquet validation")
    print("=" * 100)
    grand = {}
    for tl in LENGTHS:
        a = validate_length(tl)
        grand[tl] = a
        rows = a["rows"]
        print(f"\n### target_length = {tl}   files={a['files']}  rows={rows}")
        if rows == 0:
            print("  (no rows)")
            continue
        print(f"  tokens: valid={a['valid_tokens']:,}  capacity={a['capacity']:,}  "
              f"pad={a['pad_tokens']:,}  pad_frac={pct(a['pad_tokens'], a['capacity']):.2f}%")
        print(f"  multi-doc rows: {a['multi_doc_rows']} / {rows} "
              f"({pct(a['multi_doc_rows'], rows):.1f}%)")
        print("  --- per-channel populated % (of rows) ---")
        print(f"   A platform_ids:       {pct(a['rows_with_platform'], rows):6.1f}%")
        print(f"   B chunk_boundaries:   {pct(a['rows_with_chunks'], rows):6.1f}%")
        print(f"   B token_structure:    {pct(a['rows_with_structure'], rows):6.1f}%")
        print(f"   C type_edges:         {pct(a['rows_with_type_edges'], rows):6.1f}%")
        print(f"   C call_edges:         {pct(a['rows_with_call_edges'], rows):6.1f}%")
        print(f"   C symbol_ids:         {pct(a['rows_with_symbol'], rows):6.1f}%")
        print(f"   C def_use:            {pct(a['rows_with_def_use'], rows):6.1f}%")
        print(f"   C type_refs:          {pct(a['rows_with_type_refs'], rows):6.1f}%")
        print(f"   D changed_chunk_ids:  {pct(a['rows_with_changed_chunks'], rows):6.1f}%")
        print(f"   D edit_op_per_token:  {pct(a['rows_with_edit_op'], rows):6.1f}%")
        print("  --- integrity (MUST be 0) ---")
        print(f"   chunk-coord violations: {a['chunk_coord_violations']}")
        print(f"   edge-coord violations:  {a['edge_coord_violations']}")
        print(f"   whole-fn violations:    {a['whole_fn_violations']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
