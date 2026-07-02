#!/usr/bin/env python3
"""Render a single parquet row (code or commit) into a human-reviewable form.

Pipeline per row:
  1. DETOKENIZE ``input_ids`` with OUR ``CppMegaTokenizer`` -> source text.
  2. RE-TOKENIZE that text and verify roundtrip. We report THREE roundtrip
     metrics because our tokenizer's ``encode`` deliberately canonicalizes
     whitespace runs to ``<SPACE>``/``<NL>`` sentinels while the stored ids
     preserve raw indentation as repeated literal-space tokens:
       - text_roundtrip   : decode(ids) == decode(encode(decode(ids)))  (byte)
       - reencode_idemp   : encode(t) == encode(decode(encode(t)))      (stable)
       - id_exact         : encode(decode(ids)) == ids                  (literal)
     The honest, load-bearing guarantee is ``reencode_idemp`` (our tokenizer is
     deterministic and self-consistent). ``id_exact`` is expected to FAIL on
     indented code (stored ids keep 4-space runs; encode collapses them) and we
     report exactly where it diverges rather than papering over it.
  3. clang-format the CODE portion (best-effort: if clang-format errors we RAISE
     -- no silent fallback).
  4. Render the SIDECAR as labelled JSON grouped into channel families:
       A platform   : platform_ids (+ token_platform_ids)
       B structure  : token_structure_ids, token_dep_levels, token_ast_depth,
                      token_sibling_index, token_ast_node_type
       C graph-sem  : token_symbol_ids, token_def_use, token_call_targets,
                      token_type_refs, token_call_edges, token_type_edges
       D commit-edit: token_change_mask_pre/post, hunk_id_per_token,
                      edit_op_per_token, changed_chunk_ids, changed_chunk_spans
       + provenance : repo, commit_hash, timestamp, filepath, (@pr from docstring)

Usage:
    python scripts/render_sidecar_example.py --parquet PATH --row N [--md OUT.md]
    python scripts/render_sidecar_example.py --parquet PATH --row N --json   # raw dict
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega_mlx.tokenizer.cpp_tokenizer import (  # noqa: E402
    CppMegaTokenizer,
    load_cppmega_tokenizer,
)
from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS  # noqa: E402

CLANG_FORMAT = "/opt/homebrew/opt/llvm/bin/clang-format"
TOKENIZER_DIR = _REPO_ROOT / "cppmega_mlx" / "tokenizer"

# ---- documented label legends -------------------------------------------------
# edit_op (cppmega_mlx/data/trajectory_packet.py):
EDIT_OP_NAMES = {0: "UNCHANGED", 1: "INSERTED", 2: "MODIFIED", 3: "CONTEXT"}
# structure_id: 9 categories (cppmega_mlx/nn/structure_embedding.py num_categories=9).
# 0 is the "none / whitespace-or-unspecified" sentinel; non-zero ids are the
# syntactic structure class assigned by the indexer (e.g. comment / signature /
# body / declaration). We surface the raw id plus a short generic label so a
# reviewer can see the per-token structure stream without over-claiming names.
STRUCTURE_ID_NAMES = {
    0: "none/ws",
    1: "comment",
    2: "preproc",
    3: "decl/signature",
    4: "body/stmt",
    5: "expr",
    6: "identifier-ctx",
    7: "literal-ctx",
    8: "misc",
}
CHUNK_KIND_NAMES = STRUCTURE_ID_NAMES
DEF_USE_NAMES = {0: "none", 1: "def", 2: "use", 3: "def+use"}
DOMAIN_DELIMITER_NAMES_BY_ID = {
    int(token_id): f"<{role}>"
    for role, token_id in DOMAIN_DELIMITER_TOKEN_IDS.items()
}

# block markers that delimit the commit DOC layout inside decoded text.
# The indexer emits "=== PRE-COMMIT ... ===" / "=== POST-COMMIT: <brief> ===",
# so we match the marker prefix (not the full "===" terminator).
_PRE_MARK = "=== PRE-COMMIT"
_POST_MARK = "=== POST-COMMIT"
_DIFF_MARKS = ("=== DIFF", "=== GIT DIFF", "diff --git")
_DOCSTRING_RE = re.compile(r"/\*\*.*?\*/", re.DOTALL)
_PR_RE = re.compile(r"@pr\s+(\S+)")
_SHA_RE = re.compile(r"@sha\s+(\S+)")
_REPO_RE = re.compile(r"@repo\s+(\S+)")
# @brief value stops at the next " *" doxygen line marker (single-line docstrings)
_BRIEF_RE = re.compile(r"@brief\s+(.+?)(?:\s+\*\s|\s*\*/|$)")
_PAD_RE = re.compile(r"(<PAD>)+")


# -----------------------------------------------------------------------------
@dataclass
class RoundtripResult:
    text_roundtrip: bool
    reencode_idempotent: bool
    id_exact: bool
    id_match_modulo_ws: bool
    first_id_divergence: int | None
    n_stored_ids: int
    n_reencoded_ids: int
    note: str = ""


@dataclass
class RenderResult:
    parquet: str
    row: int
    is_commit: bool
    text: str
    formatted_code: str
    clang_format_ok: bool
    roundtrip: RoundtripResult
    sidecar: dict[str, Any]
    provenance: dict[str, Any]
    docstring: str
    blocks: dict[str, str] = field(default_factory=dict)


_TOK_CACHE: dict[str, CppMegaTokenizer] = {}


def get_tokenizer() -> CppMegaTokenizer:
    key = str(TOKENIZER_DIR)
    if key not in _TOK_CACHE:
        _TOK_CACHE[key] = load_cppmega_tokenizer(TOKENIZER_DIR)
    return _TOK_CACHE[key]


def _canon_ws(ids: list[int], space_raw: int | None, space_sent: int) -> list[int]:
    """Collapse raw-space and <SPACE> runs to a single <SPACE> for fair id compare."""
    out: list[int] = []
    for i in ids:
        j = space_sent if (space_raw is not None and i == space_raw) else i
        if j == space_sent and out and out[-1] == space_sent:
            continue
        out.append(j)
    return out


def verify_roundtrip(ids: list[int], tok: CppMegaTokenizer) -> RoundtripResult:
    text = tok.decode(ids)
    re_ids = tok.encode(text)
    text2 = tok.decode(re_ids)
    re_ids2 = tok.encode(text2)
    space_raw = tok.id_for_token(" ")
    space_sent = tok.space_token_id

    text_rt = text == text2
    idemp = re_ids == re_ids2
    id_exact = re_ids == ids
    id_mod_ws = _canon_ws(ids, space_raw, space_sent) == _canon_ws(
        re_ids, space_raw, space_sent
    )

    first_div: int | None = None
    for k in range(min(len(ids), len(re_ids))):
        if ids[k] != re_ids[k]:
            first_div = k
            break
    if first_div is None and len(ids) != len(re_ids):
        first_div = min(len(ids), len(re_ids))

    note = ""
    if not id_exact:
        note = (
            "id_exact=False is EXPECTED: stored ids preserve raw indentation as "
            "repeated literal-space tokens, while encode() canonicalizes whitespace "
            "runs to a single <SPACE> sentinel. reencode_idempotent is the "
            "load-bearing guarantee (deterministic, self-consistent tokenizer)."
        )
    return RoundtripResult(
        text_roundtrip=text_rt,
        reencode_idempotent=idemp,
        id_exact=id_exact,
        id_match_modulo_ws=id_mod_ws,
        first_id_divergence=first_div,
        n_stored_ids=len(ids),
        n_reencoded_ids=len(re_ids),
        note=note,
    )


def clang_format(code: str) -> tuple[str, bool]:
    """Run clang-format; RAISE on a real failure (no silent fallback)."""
    if not code.strip():
        return code, False
    proc = subprocess.run(
        [CLANG_FORMAT, "--assume-filename=example.cpp", "--style=LLVM"],
        input=code,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"clang-format failed (rc={proc.returncode}) at {CLANG_FORMAT}: "
            f"{proc.stderr.strip()[:400]}"
        )
    return proc.stdout, True


def _after_marker(text: str, start: int) -> int:
    """Return index just past a "=== MARKER ... ===" header, else past the word."""
    nl = text.find("\n", start)
    close = text.find("===", start + 4)  # second '===' that closes the header
    cands = [c for c in (nl, (close + 3 if close != -1 else -1)) if c != -1]
    return min(cands) if cands else start


def _strip_pad(text: str) -> str:
    return _PAD_RE.sub("", text).rstrip()


def _split_commit_blocks(text: str) -> dict[str, str]:
    """Split a commit DOC into docstring / PRE / POST / DIFF best-effort."""
    text = _strip_pad(text)
    blocks: dict[str, str] = {}
    m = _DOCSTRING_RE.search(text)
    if m:
        blocks["docstring"] = m.group(0)
    diff_idx = -1
    for mk in _DIFF_MARKS:
        idx = text.find(mk)
        if idx != -1:
            diff_idx = idx
            break
    pre_idx = text.find(_PRE_MARK)
    post_idx = text.find(_POST_MARK)
    if pre_idx != -1:
        end = post_idx if post_idx != -1 else (diff_idx if diff_idx != -1 else len(text))
        blocks["pre"] = text[_after_marker(text, pre_idx):end].strip()
    if post_idx != -1:
        end = diff_idx if diff_idx != -1 else len(text)
        blocks["post"] = text[_after_marker(text, post_idx):end].strip()
    if diff_idx != -1:
        blocks["diff"] = text[diff_idx:].strip()
    return blocks


def _code_portion(text: str, is_commit: bool, blocks: dict[str, str]) -> str:
    """Pick the C/C++ portion to clang-format (strip diff + metadata header)."""
    if is_commit:
        pieces = [
            ("// ---- PRE-COMMIT ----\n" + blocks["pre"]) if blocks.get("pre") else "",
            ("// ---- POST-COMMIT ----\n" + blocks["post"]) if blocks.get("post") else "",
        ]
        joined = "\n\n".join(p for p in pieces if p)
        if joined.strip():
            return _strip_pad(joined)
    # code parquet: drop the leading // language/platform header comments
    text = _strip_pad(text)
    lines = text.split("\n")
    out: list[str] = []
    skipping = True
    for ln in lines:
        s = ln.strip()
        if skipping and (s.startswith("//") or s == "" or s.startswith("<BOS>")):
            continue
        skipping = False
        out.append(ln)
    return "\n".join(out) if out else text


def _domain_segment_text(
    ids: list[int],
    tok: CppMegaTokenizer,
    *,
    start_role: str,
    end_role: str,
) -> str | None:
    start_id = DOMAIN_DELIMITER_TOKEN_IDS[start_role]
    end_id = DOMAIN_DELIMITER_TOKEN_IDS[end_role]
    try:
        start = ids.index(start_id)
    except ValueError:
        return None
    try:
        end = ids.index(end_id, start + 1)
    except ValueError:
        return None
    if end <= start + 1:
        return ""
    return tok.decode(ids[start + 1:end])


def _token_debug_label(token_id: int, tok: CppMegaTokenizer) -> str | None:
    logical = DOMAIN_DELIMITER_NAMES_BY_ID.get(int(token_id))
    raw = tok.token_for_id(int(token_id))
    if logical is None:
        return raw
    if raw and raw != logical:
        return f"{logical} ({raw})"
    return logical


def _per_token_table(ids: list[int], tok: CppMegaTokenizer, row: dict, start: int,
                     count: int) -> list[dict[str, Any]]:
    """Aligned per-token rows for a window so it's obvious which channel carries what."""
    end = min(start + count, len(ids))
    table: list[dict[str, Any]] = []

    def g(col: str, i: int, default: Any = 0) -> Any:
        v = row.get(col)
        if not v or i >= len(v):
            return default
        return v[i]

    for i in range(start, end):
        tid = ids[i]
        st = int(g("token_structure_ids", i))
        eo = int(g("edit_op_per_token", i))
        du = int(g("token_def_use", i))
        table.append({
            "i": i,
            "tok_id": int(tid),
            "tok": _token_debug_label(int(tid), tok),
            "tok_raw": tok.token_for_id(int(tid)),
            "A_platform": int(g("token_platform_ids", i)),
            "B_structure": f"{st}:{STRUCTURE_ID_NAMES.get(st, '?')}",
            "B_dep_lvl": int(g("token_dep_levels", i)),
            "B_ast_depth": int(g("token_ast_depth", i)),
            "B_sibling": int(g("token_sibling_index", i)),
            "B_ast_node": int(g("token_ast_node_type", i)),
            "C_symbol": int(g("token_symbol_ids", i)),
            "C_def_use": f"{du}:{DEF_USE_NAMES.get(du, '?')}",
            "C_call_tgt": int(g("token_call_targets", i)),
            "C_type_ref": int(g("token_type_refs", i)),
            "D_chg_pre": int(g("token_change_mask_pre", i)),
            "D_chg_post": int(g("token_change_mask_post", i)),
            "D_hunk": int(g("hunk_id_per_token", i, -1)),
            "D_edit_op": f"{eo}:{EDIT_OP_NAMES.get(eo, '?')}",
            "E_domain": int(g("token_domain_ids", i)),
            "E_role": int(g("token_role_ids", i)),
            "E_entity": int(g("token_entity_ids", i)),
            "E_scope": int(g("token_scope_ids", i)),
            "E_source_doc": int(g("token_source_doc_ids", i)),
            "E_confidence": int(g("token_confidence_ids", i)),
        })
    return table


def _struct_list(v: Any) -> list[Any]:
    if v is None:
        return []
    return list(v)


def _edge_records(edges: Any, *, with_kind: bool) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for edge in _struct_list(edges):
        if isinstance(edge, dict):
            if "from" not in edge or "to" not in edge:
                continue
            record = {"from": int(edge["from"]), "to": int(edge["to"])}
            if with_kind and "kind" in edge:
                record["kind"] = int(edge["kind"])
            out.append(record)
            continue
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            record = {"from": int(edge[0]), "to": int(edge[1])}
            if with_kind and len(edge) >= 3:
                record["kind"] = int(edge[2])
            out.append(record)
    return out


def build_sidecar(row: dict, ids: list[int], tok: CppMegaTokenizer, *,
                  window_start: int = 0, window: int = 48) -> dict[str, Any]:
    """Assemble the labelled per-channel sidecar JSON (families A/B/C/D + edges)."""
    # choose a window that actually has changed tokens for commits if possible
    pre = row.get("token_change_mask_pre") or []
    if pre and window_start == 0:
        for i, x in enumerate(pre):
            if x:
                window_start = max(0, i - 4)
                break

    edges_call = _edge_records(row.get("token_call_edges"), with_kind=False)
    edges_type = _edge_records(row.get("token_type_edges"), with_kind=False)
    edges_domain = _edge_records(row.get("token_domain_edges"), with_kind=True)
    edges_build = _edge_records(row.get("token_build_edges"), with_kind=True)
    edges_shell = _edge_records(row.get("token_shell_edges"), with_kind=True)
    edges_diagnostic = _edge_records(row.get("token_diagnostic_edges"), with_kind=True)
    edges_cross_domain = _edge_records(row.get("token_cross_domain_edges"), with_kind=True)
    chunk_ids = [int(x) for x in _struct_list(row.get("changed_chunk_ids"))]
    chunk_spans = [
        {"start": int(s["start"]), "end": int(s["end"])}
        for s in _struct_list(row.get("changed_chunk_spans"))
    ]

    return {
        "_legend": {
            "edit_op": EDIT_OP_NAMES,
            "structure_id": STRUCTURE_ID_NAMES,
            "def_use": DEF_USE_NAMES,
            "families": {
                "A": "platform",
                "B": "structure (syntax+structure)",
                "C": "graph-semantic (symbol/def_use/call/type + edges)",
                "D": "commit-edit (change-mask/hunk/edit-op/changed-chunks)",
                "E": "domain routes (build/shell/diagnostic delimiters, roles, edges)",
            },
            "domain_delimiters_by_id": {
                str(token_id): name
                for token_id, name in sorted(DOMAIN_DELIMITER_NAMES_BY_ID.items())
            },
        },
        "window": {"start": window_start, "count": window},
        "per_token": _per_token_table(ids, tok, row, window_start, window),
        "A_platform": {
            "platform_ids": [int(x) for x in _struct_list(row.get("platform_ids"))],
        },
        "C_graph_edges": {
            "token_call_edges": edges_call[:64],
            "token_call_edges_total": len(edges_call),
            "token_type_edges": edges_type[:64],
            "token_type_edges_total": len(edges_type),
        },
        "D_changed_chunks": {
            "changed_chunk_ids": chunk_ids,
            "changed_chunk_spans": chunk_spans,
        },
        "E_domain_routes": {
            "token_domain_edges": edges_domain[:64],
            "token_domain_edges_total": len(edges_domain),
            "token_build_edges": edges_build[:64],
            "token_build_edges_total": len(edges_build),
            "token_shell_edges": edges_shell[:64],
            "token_shell_edges_total": len(edges_shell),
            "token_diagnostic_edges": edges_diagnostic[:64],
            "token_diagnostic_edges_total": len(edges_diagnostic),
            "token_cross_domain_edges": edges_cross_domain[:64],
            "token_cross_domain_edges_total": len(edges_cross_domain),
        },
    }


def render_row(parquet: str, row_idx: int, *, window: int = 48) -> RenderResult:
    tok = get_tokenizer()
    table = pq.read_table(parquet)
    if row_idx < 0 or row_idx >= table.num_rows:
        raise IndexError(
            f"row {row_idx} out of range for {parquet} (num_rows={table.num_rows})"
        )
    row = table.slice(row_idx, 1).to_pylist()[0]
    ids = list(row["input_ids"])
    text = tok.decode(ids)
    is_commit = bool(row.get("token_change_mask_pre")) and any(
        row.get("token_change_mask_pre") or []
    ) or (_PRE_MARK in text)

    rt = verify_roundtrip(ids, tok)
    blocks = _split_commit_blocks(text) if is_commit else {}
    code = _domain_segment_text(
        ids,
        tok,
        start_role="CPP_CODE_START",
        end_role="CPP_CODE_END",
    )
    if code is None:
        code = _code_portion(text, is_commit, blocks)
    formatted, ok = clang_format(code)

    docstring = blocks.get("docstring", "")
    if not docstring:
        m = _DOCSTRING_RE.search(text)
        docstring = m.group(0) if m else ""

    provenance = {
        "repo": row.get("repo"),
        "filepath": row.get("filepath"),
        "commit_hash": row.get("commit_hash"),
        "timestamp": row.get("timestamp"),
        "pr_number": (_PR_RE.search(docstring).group(1) if _PR_RE.search(docstring) else None),
        "sha_in_doc": (_SHA_RE.search(docstring).group(1) if _SHA_RE.search(docstring) else None),
        "repo_in_doc": (_REPO_RE.search(docstring).group(1) if _REPO_RE.search(docstring) else None),
        "brief": (_BRIEF_RE.search(docstring).group(1).strip() if _BRIEF_RE.search(docstring) else None),
    }

    sidecar = build_sidecar(row, ids, tok, window=window)
    return RenderResult(
        parquet=parquet,
        row=row_idx,
        is_commit=is_commit,
        text=text,
        formatted_code=formatted,
        clang_format_ok=ok,
        roundtrip=rt,
        sidecar=sidecar,
        provenance=provenance,
        docstring=docstring,
        blocks=blocks,
    )


def _rt_md(rt: RoundtripResult) -> str:
    return (
        "| metric | value |\n|---|---|\n"
        f"| text_roundtrip (byte-exact decode) | {rt.text_roundtrip} |\n"
        f"| reencode_idempotent (load-bearing) | {rt.reencode_idempotent} |\n"
        f"| id_exact (literal stored-ids match) | {rt.id_exact} |\n"
        f"| id_match_modulo_ws_collapse | {rt.id_match_modulo_ws} |\n"
        f"| first_id_divergence | {rt.first_id_divergence} |\n"
        f"| n_stored_ids / n_reencoded | {rt.n_stored_ids} / {rt.n_reencoded_ids} |\n"
        + (f"\n> {rt.note}\n" if rt.note else "")
    )


def to_markdown(res: RenderResult, *, title: str | None = None) -> str:
    out: list[str] = []
    kind = "COMMIT" if res.is_commit else "CODE"
    out.append(f"## {title or f'{kind} example'} — `{Path(res.parquet).name}` row {res.row}")
    out.append("")
    out.append("### Provenance")
    out.append("```json")
    out.append(json.dumps(res.provenance, indent=2))
    out.append("```")
    if res.is_commit and res.docstring:
        out.append("### PR / commit docstring (head of the atomic block)")
        out.append("```c")
        out.append(res.docstring.strip())
        out.append("```")
    out.append("### Roundtrip (our cpp_tokenizer: detok -> retok)")
    out.append(_rt_md(res.roundtrip))
    out.append("### clang-format'd CODE portion")
    out.append(f"clang-format ok: **{res.clang_format_ok}**")
    out.append("```cpp")
    out.append(res.formatted_code.strip()[:6000])
    out.append("```")
    if res.is_commit and res.blocks.get("diff"):
        out.append("### Git diff (tail of the atomic block)")
        out.append("```diff")
        out.append(res.blocks["diff"][:4000])
        out.append("```")
    out.append("### Sidecar — per-channel JSON (A platform / B structure / C graph / D edit)")
    out.append("```json")
    out.append(json.dumps(res.sidecar, indent=2)[:14000])
    out.append("```")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--md", type=str, default=None, help="write markdown to path")
    ap.add_argument("--json", action="store_true", help="print raw sidecar json")
    args = ap.parse_args(argv)

    res = render_row(args.parquet, args.row, window=args.window)
    if args.json:
        print(json.dumps(res.sidecar, indent=2))
        return 0
    md = to_markdown(res)
    if args.md:
        Path(args.md).write_text(md)
        print(f"wrote {args.md}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
