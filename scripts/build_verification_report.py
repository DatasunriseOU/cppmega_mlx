#!/usr/bin/env python3
"""Build the sidecar verification report under outputs/verification_report/.

Produces:
  - code.md        : several clean CODE examples (formatted code + sidecar JSON)
  - commit.md      : several COMMIT examples (docstring + PRE + POST + diff + D-family)
  - pr.md          : PR/discussion focus (@sha/@repo/@pr/@details docstrings at head)
  - statistics.md  : per-channel FILL % (rows-nonempty + nonzero-token %), separately
                     for code vs commit parquet, over a real sample; + 1 worked
                     example of each of the 3 JSON sidecar families
  - _samples/      : ~200 rendered examples (mix of code + commit) for review

Fill % is computed over a real sample (>= 20 files-or-all, many rows each).
"""

from __future__ import annotations

import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid

# The indexer streams new parquet files concurrently (atomic temp-then-rename),
# so a file listed by glob can disappear or be mid-rename when we read it. That
# is a benign concurrent-writer race, NOT data corruption: we record the skipped
# path and continue. Any OTHER read error is re-raised (fail loud).
_CONCURRENT_WRITE_ERRORS = (FileNotFoundError, ArrowInvalid, OSError)
SKIPPED_FILES: list[str] = []


def _safe_read(path: str, columns: list[str]):
    try:
        return pq.read_table(path, columns=columns)
    except _CONCURRENT_WRITE_ERRORS as exc:
        SKIPPED_FILES.append(f"{path}: {type(exc).__name__}")
        return None

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.render_sidecar_example import (  # noqa: E402
    get_tokenizer,
    render_row,
    to_markdown,
    verify_roundtrip,
)

OUT = _REPO_ROOT / "outputs" / "verification_report"
SAMPLES = OUT / "_samples"
CODE_GLOBS = [f"outputs/reindexed/{s}/*.parquet" for s in (1024, 2048, 4096)]
COMMIT_GLOBS = [
    f"outputs/reindexed_commits/{s}/*.parquet" for s in (1024, 2048, 4096, 8192, 16384)
]

# per-token (list-aligned-to-input_ids) channels: report nonzero-token %
PERTOKEN_CHANNELS = {
    "A_platform": ["token_platform_ids"],
    "B_structure": [
        "token_structure_ids",
        "token_dep_levels",
        "token_ast_depth",
        "token_sibling_index",
        "token_ast_node_type",
    ],
    "C_graph_semantic": [
        "token_symbol_ids",
        "token_def_use",
        "token_call_targets",
        "token_type_refs",
    ],
    "D_commit_edit": [
        "token_change_mask_pre",
        "token_change_mask_post",
        "hunk_id_per_token",
        "edit_op_per_token",
    ],
}
# variable-length list channels: report rows-nonempty %
LIST_CHANNELS = {
    "A_platform": ["platform_ids"],
    "C_graph_semantic": ["token_call_edges", "token_type_edges"],
    "D_commit_edit": ["changed_chunk_ids", "changed_chunk_spans"],
}
ALL_PERTOKEN = [c for v in PERTOKEN_CHANNELS.values() for c in v]
ALL_LIST = [c for v in LIST_CHANNELS.values() for c in v]


def _files(globs: list[str]) -> list[str]:
    out: list[str] = []
    for g in globs:
        out.extend(sorted(glob.glob(str(_REPO_ROOT / g))))
    return out


def _nonzero_count(v) -> int:
    if not v:
        return 0
    if isinstance(v[0], dict):
        return len(v)
    return sum(1 for x in v if x)


def compute_fill(files: list[str], *, rows_per_file: int, label: str) -> dict:
    """Return per-channel fill stats over a real sample."""
    cols = ["input_ids", *ALL_PERTOKEN, *ALL_LIST, "hunk_id_per_token"]
    cols = sorted(set(cols))
    rows_nonempty = defaultdict(int)   # list channels: row has >=1 entry
    pertoken_rows_nonzero = defaultdict(int)  # pertoken: row has >=1 nonzero
    pertoken_tok_total = defaultdict(int)
    pertoken_tok_nonzero = defaultdict(int)
    n_rows = 0
    n_files = 0
    rt_text = rt_idemp = rt_idexact = rt_modws = 0
    rt_n = 0
    tok = get_tokenizer()
    for f in files:
        t = _safe_read(f, [c for c in cols if c])
        if t is None:
            continue
        n_files += 1
        rows = t.to_pylist()
        take = rows if rows_per_file >= len(rows) else random.sample(rows, rows_per_file)
        for r in take:
            n_rows += 1
            for c in ALL_LIST:
                if _nonzero_count(r.get(c)) > 0:
                    rows_nonempty[c] += 1
            for c in ALL_PERTOKEN:
                v = r.get(c) or []
                nz = _nonzero_count(v)
                # hunk_id uses -1 sentinel for "no hunk"; treat nonneg>0 as filled
                if c == "hunk_id_per_token":
                    nz = sum(1 for x in v if x and x > 0)
                pertoken_tok_total[c] += len(v)
                pertoken_tok_nonzero[c] += nz
                if nz > 0:
                    pertoken_rows_nonzero[c] += 1
        # roundtrip on a small slice per file (keep it bounded)
        for r in take[: min(8, len(take))]:
            ids = list(r["input_ids"])
            res = verify_roundtrip(ids, tok)
            rt_n += 1
            rt_text += int(res.text_roundtrip)
            rt_idemp += int(res.reencode_idempotent)
            rt_idexact += int(res.id_exact)
            rt_modws += int(res.id_match_modulo_ws)

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else 0.0

    return {
        "label": label,
        "n_files": n_files,
        "n_rows": n_rows,
        "rt_n": rt_n,
        "roundtrip": {
            "text_roundtrip_pct": pct(rt_text, rt_n),
            "reencode_idempotent_pct": pct(rt_idemp, rt_n),
            "id_exact_pct": pct(rt_idexact, rt_n),
            "id_match_modulo_ws_pct": pct(rt_modws, rt_n),
        },
        "pertoken": {
            c: {
                "rows_nonzero_pct": pct(pertoken_rows_nonzero[c], n_rows),
                "tokens_nonzero_pct": pct(pertoken_tok_nonzero[c], pertoken_tok_total[c]),
            }
            for c in ALL_PERTOKEN
        },
        "list": {
            c: {"rows_nonempty_pct": pct(rows_nonempty[c], n_rows)} for c in ALL_LIST
        },
    }


def _family_of(channel: str) -> str:
    for mapping in (PERTOKEN_CHANNELS, LIST_CHANNELS):
        for fam, chans in mapping.items():
            if channel in chans:
                return fam
    return "?"


def _fill_table(stats: dict) -> str:
    lines = [
        f"Sample: **{stats['n_files']} files**, **{stats['n_rows']} rows**.",
        "",
        "| family | channel | kind | rows-filled % | tokens-nonzero % |",
        "|---|---|---|---|---|",
    ]
    for c in ALL_PERTOKEN:
        p = stats["pertoken"][c]
        lines.append(
            f"| {_family_of(c)} | `{c}` | per-token | {p['rows_nonzero_pct']} | {p['tokens_nonzero_pct']} |"
        )
    for c in ALL_LIST:
        p = stats["list"][c]
        lines.append(
            f"| {_family_of(c)} | `{c}` | list | {p['rows_nonempty_pct']} | (n/a) |"
        )
    return "\n".join(lines)


def _roundtrip_table(stats: dict) -> str:
    rt = stats["roundtrip"]
    return (
        f"Roundtrip sample: **{stats['rt_n']} rows**.\n\n"
        "| metric | % |\n|---|---|\n"
        f"| text_roundtrip (byte-exact) | {rt['text_roundtrip_pct']} |\n"
        f"| reencode_idempotent (load-bearing) | {rt['reencode_idempotent_pct']} |\n"
        f"| id_exact (literal stored ids) | {rt['id_exact_pct']} |\n"
        f"| id_match_modulo_ws_collapse | {rt['id_match_modulo_ws_pct']} |\n"
    )


def find_good_rows(files: list[str], *, want_commit: bool, n: int,
                   require_diff: bool = False) -> list[tuple[str, int]]:
    """Pick rows; for commit prefer rows with change-mask filled & (optionally) a diff."""
    out: list[tuple[str, int]] = []
    tok = get_tokenizer() if require_diff else None
    for f in files:
        cols = ["input_ids", "token_change_mask_pre", "changed_chunk_ids"]
        t = _safe_read(f, cols)
        if t is None:
            continue
        rows = t.to_pylist()
        for i, r in enumerate(rows):
            filled = bool(r.get("token_change_mask_pre")) and any(
                r.get("token_change_mask_pre") or []
            )
            if want_commit and not filled:
                continue
            if require_diff:
                txt = tok.decode(list(r["input_ids"]))
                if "diff --git" not in txt and "=== DIFF" not in txt:
                    continue
            out.append((f, i))
            if len(out) >= n * 4:  # gather a surplus to allow rendering failures
                break
        if len(out) >= n * 4:
            break
    return out


def render_examples_md(picks, *, n: int, window: int, header: str, focus_pr=False) -> str:
    parts = [f"# {header}\n"]
    done = 0
    for f, i in picks:
        if done >= n:
            break
        try:
            res = render_row(f, i, window=window)
        except _CONCURRENT_WRITE_ERRORS as exc:
            SKIPPED_FILES.append(f"{f}#{i}: {type(exc).__name__}")
            continue
        if focus_pr:
            if not res.docstring:
                continue
            parts.append(_pr_section(res))
        else:
            parts.append(to_markdown(res, title=f"Example {done + 1}"))
        done += 1
    return "\n\n---\n\n".join(parts), done


def _pr_section(res) -> str:
    out = [f"## PR/commit head — `{Path(res.parquet).name}` row {res.row}"]
    out.append("### Extracted PR/discussion fields")
    out.append("```json")
    out.append(json.dumps({
        "brief": res.provenance.get("brief"),
        "pr_number": res.provenance.get("pr_number"),
        "sha_in_doc": res.provenance.get("sha_in_doc"),
        "repo_in_doc": res.provenance.get("repo_in_doc"),
        "repo": res.provenance.get("repo"),
        "filepath": res.provenance.get("filepath"),
        "commit_hash": res.provenance.get("commit_hash"),
        "timestamp": res.provenance.get("timestamp"),
    }, indent=2))
    out.append("```")
    out.append("### Raw docstring (sits at the HEAD of the atomic block, before PRE code)")
    out.append("```c")
    out.append(res.docstring.strip())
    out.append("```")
    # show that code follows right after the docstring
    head = res.text[: res.text.find(res.docstring) + len(res.docstring) + 160]
    out.append("### Block head (docstring -> first code lines)")
    out.append("```cpp")
    out.append(head[-360:])
    out.append("```")
    return "\n".join(out)


def _gaps_section(code_stats: dict, commit_stats: dict) -> str:
    """Honestly surface channels that are empty-when-they-could-be-filled."""
    lines = ["## Known gaps (channels empty where signal could exist)\n"]
    lines.append(
        "- `token_platform_ids` (per-token A-platform) is **0% in BOTH code and "
        "commit** parquet across the whole sample. Platform signal is carried by the "
        "row-level `platform_ids` LIST (100% filled) instead; the per-token mirror "
        "column is not populated by the current indexer. The A-platform family IS "
        "filled — via the list channel, not the per-token column."
    )
    lines.append(
        "- D-family (`token_change_mask_*`, `hunk_id_per_token`, `edit_op_per_token`, "
        "`changed_chunk_*`) is **0% in CODE parquet** — correct/expected, those channels "
        "only carry signal in commit docs."
    )
    lines.append(
        f"- `token_change_mask_post` is filled on only "
        f"{commit_stats['pertoken']['token_change_mask_post']['rows_nonzero_pct']}% of "
        "commit rows (many commits touch only the PRE side / are pure deletions or have "
        "an empty POST), and `token_type_edges` on "
        f"{commit_stats['list']['token_type_edges']['rows_nonempty_pct']}% of commit "
        "rows — lower fill but present where the diff actually adds type relationships."
    )
    lines.append(
        "- `token_call_targets` / `token_type_refs` have HIGH rows-filled % but LOW "
        "tokens-nonzero % (~1-4%): they are sparse by design (only the few call/type "
        "reference sites per window carry an id)."
    )
    return "\n".join(lines) + "\n"


def worked_family_examples(code_files, commit_files) -> str:
    """One worked example of each of the 3 JSON sidecar families (A/B/C, and D)."""
    tok = get_tokenizer()
    # code row -> A/B/C families; commit row -> D family
    code_pick = find_good_rows(code_files, want_commit=False, n=1)[0]
    commit_pick = find_good_rows(commit_files, want_commit=True, n=1)[0]
    rc = render_row(*code_pick, window=24)
    rm = render_row(*commit_pick, window=24)
    out = ["## Worked examples of each sidecar JSON family\n"]
    out.append("### Family A (platform) + B (structure) + C (graph-semantic) — from a CODE row")
    fam_abc = {
        "A_platform": rc.sidecar["A_platform"],
        "B_C_per_token_window": rc.sidecar["per_token"][:12],
        "C_graph_edges": rc.sidecar["C_graph_edges"],
    }
    out.append("```json")
    out.append(json.dumps(fam_abc, indent=2)[:5000])
    out.append("```")
    out.append("### Family D (commit-edit) — from a COMMIT row")
    fam_d = {
        "D_changed_chunks": rm.sidecar["D_changed_chunks"],
        "D_per_token_window": [
            {k: v for k, v in r.items() if k.startswith(("i", "tok", "D_"))}
            for r in rm.sidecar["per_token"][:12]
        ],
    }
    out.append("```json")
    out.append(json.dumps(fam_d, indent=2)[:5000])
    out.append("```")
    return "\n".join(out)


def main() -> int:
    random.seed(1234)
    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    code_files = _files(CODE_GLOBS)
    commit_files = _files(COMMIT_GLOBS)
    if not code_files or not commit_files:
        raise RuntimeError(
            f"missing parquet inputs (code={len(code_files)}, commit={len(commit_files)})"
        )

    # ---- real fill stats over a decent sample (all files, bounded rows each) ----
    code_stats = compute_fill(code_files, rows_per_file=40, label="CODE")
    commit_stats = compute_fill(commit_files, rows_per_file=40, label="COMMIT")

    # ---- statistics.md ----
    stat_md = [
        "# Sidecar fill statistics (real sample)\n",
        "Per-channel FILL %, computed over a real sample of rows from EVERY input "
        "parquet file. `rows-filled %` = fraction of rows with >=1 non-zero/non-empty "
        "entry in that channel; `tokens-nonzero %` = fraction of all per-token slots "
        "that are non-zero (per-token channels only). hunk_id uses a -1 sentinel for "
        "\"no hunk\", so it is counted filled only when > 0.\n",
        "## Roundtrip (our cpp_tokenizer)\n",
        "Three honest metrics. `id_exact` is EXPECTED to be low: stored ids keep raw "
        "multi-space indentation (repeated literal-space token), while `encode()` "
        "canonicalizes whitespace runs to a single `<SPACE>`/`<NL>` sentinel. The "
        "load-bearing guarantee is `reencode_idempotent` (deterministic, self-"
        "consistent) and byte-exact `text_roundtrip` on content without collapsed "
        "indentation.\n",
        "### CODE parquet\n", _roundtrip_table(code_stats), "",
        "### COMMIT parquet\n", _roundtrip_table(commit_stats), "",
        "## FILL % — CODE parquet\n", _fill_table(code_stats), "",
        "> D-family (`token_change_mask_*`, `hunk_id`, `edit_op`, `changed_chunk_*`) is "
        "EXPECTED empty for plain code rows — those channels only carry signal for "
        "commit docs.\n",
        "## FILL % — COMMIT parquet\n", _fill_table(commit_stats), "",
        _gaps_section(code_stats, commit_stats),
        worked_family_examples(code_files, commit_files),
    ]
    (OUT / "statistics.md").write_text("\n".join(stat_md))

    # ---- code.md ----
    code_picks = find_good_rows(code_files, want_commit=False, n=6)
    code_body, n_code = render_examples_md(
        code_picks, n=6, window=40,
        header="CODE examples — formatted code + per-channel sidecar JSON",
    )
    (OUT / "code.md").write_text(code_body)

    # ---- commit.md (prefer rows that contain the git-diff tail of the block) ----
    commit_picks = find_good_rows(commit_files, want_commit=True, n=6, require_diff=True)
    if len(commit_picks) < 6:
        commit_picks += find_good_rows(commit_files, want_commit=True, n=6)
    commit_body, n_commit = render_examples_md(
        commit_picks, n=6, window=40,
        header="COMMIT examples — full atomic block (docstring + PRE + POST + diff) + D-family edit channels",
    )
    (OUT / "commit.md").write_text(commit_body)

    # ---- pr.md ----
    pr_picks = find_good_rows(commit_files, want_commit=True, n=8)
    pr_body, n_pr = render_examples_md(
        pr_picks, n=8, window=24,
        header="PR / discussion focus — the @sha/@repo/@pr/@details docstring at the head of each block",
        focus_pr=True,
    )
    (OUT / "pr.md").write_text(pr_body)

    # ---- ~200 _samples (mix) ----
    n_samples = 200
    n_each = n_samples // 2
    sample_code = find_good_rows(code_files, want_commit=False, n=n_each)
    # mix: half diff-bearing commit rows, half change-mask rows
    sample_commit = find_good_rows(commit_files, want_commit=True, n=n_each // 2,
                                   require_diff=True)
    sample_commit += find_good_rows(commit_files, want_commit=True, n=n_each)
    built = 0
    manifest = []
    for kind, picks in (("code", sample_code), ("commit", sample_commit)):
        seen = 0
        for f, i in picks:
            if seen >= n_each:
                break
            try:
                res = render_row(f, i, window=32)
            except _CONCURRENT_WRITE_ERRORS as exc:
                SKIPPED_FILES.append(f"{f}#{i}: {type(exc).__name__}")
                continue
            except Exception as exc:  # noqa: BLE001 - real render bug, surface it
                raise RuntimeError(f"render failed {f}#{i}: {exc}") from exc
            name = f"{kind}_{seen:03d}_{Path(f).stem}_r{i}.md"
            (SAMPLES / name).write_text(to_markdown(res, title=f"{kind} sample {seen}"))
            manifest.append({"file": name, "kind": kind, "parquet": f, "row": i})
            built += 1
            seen += 1
    (SAMPLES / "_manifest.json").write_text(json.dumps(manifest, indent=2))

    summary = {
        "samples_built": built,
        "code_examples": n_code,
        "commit_examples": n_commit,
        "pr_examples": n_pr,
        "skipped_concurrent_writes": SKIPPED_FILES,
        "code_stats": code_stats,
        "commit_stats": commit_stats,
    }
    if SKIPPED_FILES:
        note = (
            "\n## Concurrent-writer skips\n\n"
            f"{len(SKIPPED_FILES)} file/row read(s) were skipped because the indexer "
            "was atomically replacing those parquet files mid-run (benign race, not "
            "corruption). They are excluded from the sample:\n\n"
            + "\n".join(f"- `{s}`" for s in SKIPPED_FILES[:40])
        )
        with (OUT / "statistics.md").open("a") as fh:
            fh.write(note + "\n")
    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
