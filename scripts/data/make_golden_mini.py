#!/usr/bin/env python3
"""Build a deterministic, tiny GOLDEN fixture for downstream pipeline testing.

This script synthesizes 3 small, self-contained C++ "mini-repos" (fixed source,
no randomness / no timestamps) that exercise:

  * cross-file calls            -> populates call_edges
  * a base class + virtual override, struct + field access, a template,
    and an #include graph -> populates type_edges

It then runs the REAL modern data pipeline end-to-end on them:

  1. tools/clang_indexer/index_project.py  --enriched
        synth repo dir  ->  enriched JSONL (text + structure_ids +
        chunk_boundaries + call_edges + type_edges + char-level metadata)

  2. scripts/nanochat_data/clang_enriched_to_parquet.py
        --input-file <jsonl> --output-file <parquet>
        --materialize-tokenized-enriched   (emits token_ids + token-level
        enriched columns using the canonical 65536-vocab tokenizer)

  3. scripts/nanochat_data/pack_enriched_rows.py
        --input <parquet> --output <packed.parquet> --target-length N
        fixed-length packed code rows (input_ids / target_ids / loss_mask ...)

For the commit path it synthesizes a couple of before/after function pairs and
runs:

  4. tools/clang_indexer/process_commits.py --inputs <raw_commits.jsonl>
        --output <enriched_commits.jsonl>
     then the SAME parquet conversion step (2) to produce commits parquet.

Outputs are written under tests/fixtures/golden_mini/:
    code/code_<repo>.parquet        (packed code rows)
    commits/commits.parquet         (tokenized enriched commit docs)
    README.md                       (synthetic-input + column inventory)

Re-runnable + deterministic: the synthetic source is a fixed Python constant,
the temp working dir is recreated each run, and no Date.now / random is used.

RULE #1 — FAIL FAST, FAIL LOUD: every stage is a single clear subprocess call;
on non-zero exit or empty/missing output we RAISE with WHERE + WHAT. There are
no silent fallbacks: a stage that cannot run aborts the whole build.

Usage:
    /Volumes/external/sources/cppmega.mlx/.venv/bin/python \
        /Volumes/external/sources/cppmega.mlx/scripts/data/make_golden_mini.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed, canonical paths (absolute — this script is location-pinned).
# ---------------------------------------------------------------------------
PY = "/Volumes/external/sources/cppmega.mlx/.venv/bin/python"
CPPMEGA_MLX_ROOT = Path("/Volumes/external/sources/cppmega.mlx")
NANOCHAT_ROOT = Path("/Volumes/external/sources/nanochat")

INDEX_PROJECT = NANOCHAT_ROOT / "tools" / "clang_indexer" / "index_project.py"
PROCESS_COMMITS = NANOCHAT_ROOT / "tools" / "clang_indexer" / "process_commits.py"
CLANG_ENRICHED_TO_PARQUET = (
    CPPMEGA_MLX_ROOT / "scripts" / "nanochat_data" / "clang_enriched_to_parquet.py"
)
PACK_ENRICHED_ROWS = (
    CPPMEGA_MLX_ROOT / "scripts" / "nanochat_data" / "pack_enriched_rows.py"
)
TOKENIZER_JSON = (
    CPPMEGA_MLX_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
)

FIXTURE_DIR = CPPMEGA_MLX_ROOT / "tests" / "fixtures" / "golden_mini"
CODE_OUT_DIR = FIXTURE_DIR / "code"
COMMITS_OUT_DIR = FIXTURE_DIR / "commits"

# Token budget for enriched->parquet conversion (size label) and the packed
# row target length. Kept small but real so the fixture stays tiny.
SIZE_LABEL = "4k"
PACK_TARGET_LENGTH = 4096


# ---------------------------------------------------------------------------
# Synthetic, FIXED mini-repos. Each value maps relative path -> file contents.
# No timestamps, no randomness: byte-identical on every run.
# ---------------------------------------------------------------------------

# Repo A: shapes — base class + virtual override (type_edges via CXX_BASE/override),
# struct + field access, cross-file call (area_report -> Shape::area via vtable,
# and a free function calling across files).
REPO_SHAPES: dict[str, str] = {
    "include/shape.h": """\
#pragma once

struct Point {
    double x;
    double y;
};

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;
    Point centroid() const;
};

class Circle : public Shape {
public:
    explicit Circle(double r) : radius_(r) {}
    double area() const override;
private:
    double radius_;
};
""",
    "src/shape.cpp": """\
#include "shape.h"

static const double kPi = 3.14159265358979323846;

Point Shape::centroid() const {
    Point p;
    p.x = 0.0;
    p.y = 0.0;
    return p;
}

double Circle::area() const {
    return kPi * radius_ * radius_;
}
""",
    "src/report.cpp": """\
#include "shape.h"

double area_report(const Shape& s) {
    Point c = s.centroid();
    double a = s.area();
    return a + c.x + c.y;
}

double circle_report(double r) {
    Circle circle(r);
    return area_report(circle);
}
""",
}

# Repo B: container — a class template (type_edges via TEMPLATE_REF), cross-file
# call from main path into the template-backed stack.
REPO_CONTAINER: dict[str, str] = {
    "include/stack.h": """\
#pragma once

template <typename T>
class Stack {
public:
    void push(const T& value) {
        data_[size_++] = value;
    }
    T pop() {
        return data_[--size_];
    }
    int size() const {
        return size_;
    }
private:
    T data_[64];
    int size_ = 0;
};
""",
    "src/use_stack.cpp": """\
#include "stack.h"

int sum_first_three() {
    Stack<int> s;
    s.push(10);
    s.push(20);
    s.push(30);
    int total = 0;
    total += s.pop();
    total += s.pop();
    total += s.pop();
    return total;
}

int stack_depth_after(int n) {
    Stack<int> s;
    for (int i = 0; i < n; ++i) {
        s.push(i);
    }
    return s.size();
}
""",
}

# Repo C: graph — multi-file #include graph + cross-file calls through 3 layers
# (util -> engine -> api), plus a struct field access.
REPO_GRAPH: dict[str, str] = {
    "include/util.h": """\
#pragma once

struct Config {
    int level;
    int scale;
};

int clamp_level(int level);
""",
    "src/util.cpp": """\
#include "util.h"

int clamp_level(int level) {
    if (level < 0) {
        return 0;
    }
    if (level > 9) {
        return 9;
    }
    return level;
}
""",
    "include/engine.h": """\
#pragma once
#include "util.h"

int run_engine(const Config& cfg);
""",
    "src/engine.cpp": """\
#include "engine.h"

int run_engine(const Config& cfg) {
    int lvl = clamp_level(cfg.level);
    return lvl * cfg.scale;
}
""",
    "src/api.cpp": """\
#include "engine.h"

int public_api(int level, int scale) {
    Config cfg;
    cfg.level = level;
    cfg.scale = scale;
    return run_engine(cfg);
}
""",
}

MINI_REPOS: dict[str, dict[str, str]] = {
    "shapes": REPO_SHAPES,
    "container": REPO_CONTAINER,
    "graph": REPO_GRAPH,
}


# ---------------------------------------------------------------------------
# Synthetic, FIXED commit before/after pairs. The clang commit processor needs
# old_content, new_content, and a unified diff with @@ hunk headers (>=50 bytes
# each, >=100 byte resulting doc). We hand-author small but valid diffs.
# ---------------------------------------------------------------------------

_COMMIT_OLD_1 = """\
#include "shape.h"

double scale_area(const Shape& s, double factor) {
    double a = s.area();
    return a * factor;
}

double total_area(const Shape& s) {
    return scale_area(s, 1.0);
}
"""

_COMMIT_NEW_1 = """\
#include "shape.h"

double scale_area(const Shape& s, double factor) {
    double a = s.area();
    if (factor < 0.0) {
        factor = 0.0;
    }
    return a * factor;
}

double total_area(const Shape& s) {
    return scale_area(s, 1.0);
}
"""

_COMMIT_DIFF_1 = """\
--- a/src/scale.cpp
+++ b/src/scale.cpp
@@ -3,6 +3,9 @@
 double scale_area(const Shape& s, double factor) {
     double a = s.area();
+    if (factor < 0.0) {
+        factor = 0.0;
+    }
     return a * factor;
 }
"""

_COMMIT_OLD_2 = """\
#include "util.h"

int normalize(const Config& cfg) {
    int lvl = clamp_level(cfg.level);
    return lvl;
}

int scaled_level(const Config& cfg) {
    return normalize(cfg);
}
"""

_COMMIT_NEW_2 = """\
#include "util.h"

int normalize(const Config& cfg) {
    int lvl = clamp_level(cfg.level);
    return lvl * cfg.scale;
}

int scaled_level(const Config& cfg) {
    return normalize(cfg);
}
"""

_COMMIT_DIFF_2 = """\
--- a/src/normalize.cpp
+++ b/src/normalize.cpp
@@ -3,5 +3,5 @@
 int normalize(const Config& cfg) {
     int lvl = clamp_level(cfg.level);
-    return lvl;
+    return lvl * cfg.scale;
 }
"""

COMMIT_RECORDS: list[dict[str, str]] = [
    {
        "repo": "golden_mini/shapes",
        "filepath": "src/scale.cpp",
        "commit_hash": "0000000000000000000000000000000000000001",
        "parent_hash": "00000000000000000000000000000000000000a1",
        "old_content": _COMMIT_OLD_1,
        "new_content": _COMMIT_NEW_1,
        "diff": _COMMIT_DIFF_1,
    },
    {
        "repo": "golden_mini/graph",
        "filepath": "src/normalize.cpp",
        "commit_hash": "0000000000000000000000000000000000000002",
        "parent_hash": "00000000000000000000000000000000000000a2",
        "old_content": _COMMIT_OLD_2,
        "new_content": _COMMIT_NEW_2,
        "diff": _COMMIT_DIFF_2,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    stage: str, cmd: list[str], *, pythonpath: Path, cwd: Path | None = None
) -> None:
    """Run a pipeline stage as a subprocess; RAISE loudly on failure.

    `pythonpath` must be EXACTLY the one repo root the stage's imports resolve
    against. The cppmega and nanochat repos BOTH ship top-level `scripts/` and
    `tools/` packages, so mixing both roots on PYTHONPATH makes
    `scripts.nanochat_data` / `tools.clang_indexer` resolve to the wrong repo.
    We therefore pin a single root per stage:
      * nanochat stages (index_project / process_commits) -> NANOCHAT_ROOT
      * cppmega stages (parquet / pack)                    -> CPPMEGA_MLX_ROOT
    """
    print(f"[make_golden_mini] STAGE {stage}: {' '.join(cmd)}", file=sys.stderr)
    env = dict(os.environ)
    # Start from a clean PYTHONPATH so neither repo's packages leak across.
    env["PYTHONPATH"] = str(pythonpath)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"[make_golden_mini] STAGE {stage} FAILED (exit {proc.returncode}).\n"
            f"  WHERE: {' '.join(cmd)}\n"
            f"  STDOUT:\n{proc.stdout}\n"
            f"  STDERR:\n{proc.stderr}"
        )
    if proc.stderr.strip():
        # Surface stage logs (these are informational logs on stderr).
        print(proc.stderr, file=sys.stderr)


def _require_nonempty_parquet(stage: str, path: Path) -> int:
    """Assert a parquet exists and has >0 rows; return the row count."""
    if not path.exists():
        raise RuntimeError(
            f"[make_golden_mini] STAGE {stage}: expected output parquet does not "
            f"exist.\n  WHERE: {path}"
        )
    import pyarrow.parquet as pq  # local import: only the cppmega venv has it

    nrows = pq.ParquetFile(str(path)).metadata.num_rows
    if nrows <= 0:
        raise RuntimeError(
            f"[make_golden_mini] STAGE {stage}: output parquet has 0 rows — a "
            f"pipeline stage produced no documents.\n  WHERE: {path}"
        )
    return nrows


def _materialize_repo(root: Path, files: dict[str, str]) -> None:
    """Write a synthetic mini-repo to disk (deterministic content)."""
    for rel, content in sorted(files.items()):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _column_inventory(path: Path) -> list[tuple[str, str]]:
    """Return [(column_name, arrow_type_str), ...] for a parquet file."""
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(str(path)).schema_arrow
    return [(field.name, str(field.type)) for field in schema]


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts (skipping blank lines)."""
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts to JSONL (one compact object per line)."""
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# Provenance keys the parquet converter reads straight off each enriched JSONL
# record (REPO_COLUMN / FILEPATH_COLUMN / COMMIT_HASH_COLUMN / PARENT_HASHES_COLUMN
# in scripts/nanochat_data/clang_enriched_to_parquet.py). The upstream clang
# indexer / commit processor build per-document text but DO NOT echo these scalar
# provenance fields back into their emitted dict, so they arrive empty. We are the
# only component that knows the real repo/filepath/commit each synthetic document
# came from, so we carry that REAL provenance through here. This is provenance
# pass-through (the values are the actual inputs we fed in), NOT fabricated data:
# we never invent token-aligned signals the pipeline did not compute.

def _merge_code_provenance(
    enriched_jsonl: Path, repo_name: str, files: dict[str, str]
) -> None:
    """Stamp real repo + filepath provenance into a code enriched JSONL in place.

    Every document in ``enriched_<repo>.jsonl`` is the per-project index of a
    single mini-repo, so the repo is known unambiguously. We record the repo as
    ``golden_mini/<repo_name>`` and the constituent source files as ``filepath``
    (newline-joined relative paths actually fed to the indexer). RAISE if the
    file is empty (a stage upstream silently produced nothing).
    """
    records = _read_jsonl(enriched_jsonl)
    if not records:
        raise RuntimeError(
            f"[make_golden_mini] _merge_code_provenance[{repo_name}]: enriched "
            f"JSONL has no records to stamp.\n  WHERE: {enriched_jsonl}"
        )
    repo = f"golden_mini/{repo_name}"
    filepath = "\n".join(sorted(files.keys()))
    for rec in records:
        rec["repo"] = repo
        rec["filepath"] = filepath
    _write_jsonl(enriched_jsonl, records)


def _merge_commit_provenance(enriched_jsonl: Path) -> None:
    """Stamp real repo/filepath/commit_hash/parent_hashes into commit JSONL.

    The commit processor renders a header ``Repository: <repo>`` / ``File:
    <filepath>`` into each document's text from the source record, but drops the
    structured provenance fields from its emitted dict. We map each enriched
    document back to its originating COMMIT_RECORDS entry by (repo, filepath)
    parsed out of that header, then re-attach the REAL commit_hash + a synthetic
    parent chain (the genesis commit's single parent). RAISE if any document
    cannot be matched (that means the header format changed and our mapping is
    no longer trustworthy — fail loud rather than stamp wrong provenance).
    """
    records = _read_jsonl(enriched_jsonl)
    if not records:
        raise RuntimeError(
            "[make_golden_mini] _merge_commit_provenance: enriched commit JSONL "
            f"has no records to stamp.\n  WHERE: {enriched_jsonl}"
        )
    # Index source records by (repo, filepath).
    by_key = {
        (r["repo"], r["filepath"]): r for r in COMMIT_RECORDS
    }
    for rec in records:
        text = rec.get("text", "")
        repo = filepath = None
        for line in text.splitlines():
            s = line.strip().lstrip("*").strip()
            if s.startswith("Repository:"):
                repo = s[len("Repository:"):].strip()
            elif s.startswith("File:"):
                filepath = s[len("File:"):].strip()
            if repo is not None and filepath is not None:
                break
        if repo is None or filepath is None:
            raise RuntimeError(
                "[make_golden_mini] _merge_commit_provenance: could not parse "
                "Repository/File header out of an enriched commit document — the "
                "commit-processor text format changed; refusing to stamp wrong "
                "provenance.\n  TEXT-HEAD:\n" + text[:300]
            )
        src = by_key.get((repo, filepath))
        if src is None:
            raise RuntimeError(
                "[make_golden_mini] _merge_commit_provenance: enriched document "
                f"with provenance ({repo!r}, {filepath!r}) does not match any "
                "synthetic COMMIT_RECORDS entry — mapping is broken."
            )
        rec["repo"] = repo
        rec["filepath"] = filepath
        rec["commit_hash"] = src["commit_hash"]
        # A before/after pair is a child commit on top of one synthetic parent.
        rec["parent_hashes"] = [src["parent_hash"]]
        rec["parent_count"] = 1
        rec["is_merge_commit"] = False
    _write_jsonl(enriched_jsonl, records)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def build_code_fixture(work: Path) -> dict[str, dict]:
    """Run index -> parquet -> pack for each mini-repo. Returns per-repo info."""
    results: dict[str, dict] = {}
    CODE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for repo_name, files in MINI_REPOS.items():
        repo_dir = work / "repos" / repo_name
        _materialize_repo(repo_dir, files)

        enriched_jsonl = work / f"enriched_{repo_name}.jsonl"
        tokenized_parquet = work / f"tokenized_{repo_name}.parquet"
        packed_parquet = CODE_OUT_DIR / f"code_{repo_name}.parquet"

        # Stage 1: clang indexer -> enriched JSONL
        _run(
            f"index_project[{repo_name}]",
            [
                PY,
                str(INDEX_PROJECT),
                "--project-dir",
                str(repo_dir),
                "--output",
                str(enriched_jsonl),
                "--enriched",
                "--max-tokens",
                "4096",
                "--workers",
                "1",
                "--parse-workers",
                "1",
            ],
            pythonpath=NANOCHAT_ROOT,
        )
        if not enriched_jsonl.exists() or enriched_jsonl.stat().st_size == 0:
            raise RuntimeError(
                f"[make_golden_mini] STAGE index_project[{repo_name}]: enriched "
                f"JSONL is missing/empty — clang produced no documents.\n"
                f"  WHERE: {enriched_jsonl}"
            )

        # Sanity: confirm call_edges AND type_edges are non-trivially populated.
        n_call = n_type = n_docs = 0
        with enriched_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_docs += 1
                n_call += len(rec.get("call_edges", []))
                n_type += len(rec.get("type_edges", []))

        # Carry REAL repo + filepath provenance into the enriched JSONL before
        # conversion (the indexer does not echo it; the converter reads it off
        # each record). This is provenance pass-through, not fabrication.
        _merge_code_provenance(enriched_jsonl, repo_name, files)

        # Stage 2: enriched JSONL -> tokenized enriched parquet (token_ids etc.)
        _run(
            f"clang_enriched_to_parquet[{repo_name}]",
            [
                PY,
                str(CLANG_ENRICHED_TO_PARQUET),
                "--size",
                SIZE_LABEL,
                "--input-file",
                str(enriched_jsonl),
                "--output-file",
                str(tokenized_parquet),
                "--tokenizer-path",
                str(TOKENIZER_JSON),
                "--materialize-tokenized-enriched",
                "--overflow-policy",
                "drop",
            ],
            pythonpath=CPPMEGA_MLX_ROOT,
        )
        _require_nonempty_parquet(
            f"clang_enriched_to_parquet[{repo_name}]", tokenized_parquet
        )

        # Stage 3: tokenized parquet -> packed fixed-length rows
        _run(
            f"pack_enriched_rows[{repo_name}]",
            [
                PY,
                str(PACK_ENRICHED_ROWS),
                "--input",
                str(tokenized_parquet),
                "--output",
                str(packed_parquet),
                "--target-length",
                str(PACK_TARGET_LENGTH),
                "--strategy",
                "best_fit",
            ],
            pythonpath=CPPMEGA_MLX_ROOT,
        )
        nrows = _require_nonempty_parquet(
            f"pack_enriched_rows[{repo_name}]", packed_parquet
        )

        results[repo_name] = {
            "enriched_docs": n_docs,
            "call_edges": n_call,
            "type_edges": n_type,
            "packed_rows": nrows,
            "packed_path": packed_parquet,
            "tokenized_path": tokenized_parquet,
        }
        print(
            f"[make_golden_mini] {repo_name}: docs={n_docs} "
            f"call_edges={n_call} type_edges={n_type} packed_rows={nrows}",
            file=sys.stderr,
        )

    # Cross-repo invariant: call_edges AND type_edges must be non-trivial overall.
    total_call = sum(r["call_edges"] for r in results.values())
    total_type = sum(r["type_edges"] for r in results.values())
    if total_call <= 0:
        raise RuntimeError(
            "[make_golden_mini] INVARIANT FAILED: no call_edges across any "
            "mini-repo — the synthetic source or the indexer is not producing "
            "cross-file call edges."
        )
    if total_type <= 0:
        raise RuntimeError(
            "[make_golden_mini] INVARIANT FAILED: no type_edges across any "
            "mini-repo — the synthetic source or the indexer is not producing "
            "type edges (base/override/template/field)."
        )
    return results


def build_commits_fixture(work: Path) -> dict:
    """Run process_commits -> parquet for the synthetic commit pairs."""
    COMMITS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_commits = work / "raw_commits.jsonl"
    with raw_commits.open("w", encoding="utf-8") as fh:
        for rec in COMMIT_RECORDS:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    enriched_commits = work / "enriched_commits.jsonl"
    commits_parquet = COMMITS_OUT_DIR / "commits.parquet"

    # Stage 4a: process_commits -> enriched commit JSONL
    _run(
        "process_commits",
        [
            PY,
            str(PROCESS_COMMITS),
            "--inputs",
            str(raw_commits),
            "--output",
            str(enriched_commits),
            "--max-tokens",
            "4096",
            "--format",
            "both",
            "--tokenizer-path",
            str(TOKENIZER_JSON),
        ],
        pythonpath=NANOCHAT_ROOT,
    )
    if not enriched_commits.exists() or enriched_commits.stat().st_size == 0:
        raise RuntimeError(
            "[make_golden_mini] STAGE process_commits: enriched commit JSONL is "
            "missing/empty — the commit processor produced no documents. Check "
            "the synthetic diffs (need valid @@ hunks and >=50-byte contents).\n"
            f"  WHERE: {enriched_commits}"
        )

    n_docs = 0
    with enriched_commits.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n_docs += 1

    # Carry REAL commit provenance (repo/filepath/commit_hash/parent_hashes) into
    # the enriched JSONL before conversion. The commit processor renders these
    # into the document header text but drops the structured fields; the converter
    # reads them off each record. Provenance pass-through, not fabrication.
    _merge_commit_provenance(enriched_commits)

    # Stage 4b: enriched commit JSONL -> tokenized enriched parquet
    _run(
        "clang_enriched_to_parquet[commits]",
        [
            PY,
            str(CLANG_ENRICHED_TO_PARQUET),
            "--size",
            SIZE_LABEL,
            "--input-file",
            str(enriched_commits),
            "--output-file",
            str(commits_parquet),
            "--tokenizer-path",
            str(TOKENIZER_JSON),
            "--materialize-tokenized-enriched",
            "--overflow-policy",
            "drop",
        ],
        pythonpath=CPPMEGA_MLX_ROOT,
    )
    nrows = _require_nonempty_parquet(
        "clang_enriched_to_parquet[commits]", commits_parquet
    )
    print(
        f"[make_golden_mini] commits: enriched_docs={n_docs} parquet_rows={nrows}",
        file=sys.stderr,
    )
    return {
        "enriched_docs": n_docs,
        "parquet_rows": nrows,
        "parquet_path": commits_parquet,
    }


def write_readme(code_info: dict[str, dict], commit_info: dict) -> None:
    """Write a README documenting synthetic inputs + column inventories."""
    # Pick a representative packed parquet + the commits parquet for inventory.
    a_repo = sorted(code_info)[0]
    code_cols = _column_inventory(code_info[a_repo]["packed_path"])
    commit_cols = _column_inventory(commit_info["parquet_path"])

    def _fmt_cols(cols: list[tuple[str, str]]) -> str:
        return "\n".join(f"  - `{name}`: `{typ}`" for name, typ in cols)

    lines: list[str] = []
    lines.append("# golden_mini fixture")
    lines.append("")
    lines.append(
        "Deterministic, tiny GOLDEN fixture produced by the REAL modern data "
        "pipeline. Regenerate with:"
    )
    lines.append("")
    lines.append("```")
    lines.append(
        f"{PY} {CPPMEGA_MLX_ROOT}/scripts/data/make_golden_mini.py"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "All inputs are fixed Python constants in `make_golden_mini.py` (no "
        "randomness, no timestamps), so the synthetic C++ sources and the "
        "pipeline outputs are byte-stable across runs on the same toolchain."
    )
    lines.append("")
    lines.append("## Pipeline")
    lines.append("")
    lines.append(
        "1. `tools/clang_indexer/index_project.py --enriched` "
        "(synth repo -> enriched JSONL)"
    )
    lines.append(
        "2. `scripts/nanochat_data/clang_enriched_to_parquet.py "
        "--materialize-tokenized-enriched` (JSONL -> tokenized enriched parquet, "
        f"tokenizer={TOKENIZER_JSON.name} vocab=65536, size={SIZE_LABEL})"
    )
    lines.append(
        "3. `scripts/nanochat_data/pack_enriched_rows.py "
        f"--target-length {PACK_TARGET_LENGTH}` (tokenized parquet -> packed rows)"
    )
    lines.append(
        "4. Commits: `tools/clang_indexer/process_commits.py` "
        "(before/after pairs -> enriched commit JSONL) -> step 2 conversion."
    )
    lines.append("")
    lines.append("## Synthetic inputs")
    lines.append("")
    lines.append("### Code mini-repos (`code/`)")
    lines.append("")
    descriptions = {
        "shapes": (
            "base class `Shape` + virtual `area()` override in `Circle`; struct "
            "`Point` field access; cross-file calls (`report.cpp` -> "
            "`Shape::area`/`centroid`, `circle_report` -> `area_report`)."
        ),
        "container": (
            "class template `Stack<T>` (template_ref type edges); cross-file use "
            "in `use_stack.cpp`."
        ),
        "graph": (
            "3-layer #include + call graph `util -> engine -> api`; struct "
            "`Config` field access."
        ),
    }
    for repo in sorted(code_info):
        info = code_info[repo]
        files = sorted(MINI_REPOS[repo].keys())
        lines.append(
            f"- **{repo}** (`code/code_{repo}.parquet`, "
            f"{info['packed_rows']} packed row(s)) — {descriptions[repo]}"
        )
        lines.append(
            f"  - files: {', '.join('`' + f + '`' for f in files)}"
        )
        lines.append(
            f"  - enriched docs: {info['enriched_docs']}, "
            f"call_edges: {info['call_edges']}, type_edges: {info['type_edges']}"
        )
    lines.append("")
    lines.append("### Commit pairs (`commits/`)")
    lines.append("")
    for rec in COMMIT_RECORDS:
        lines.append(
            f"- `{rec['filepath']}` (repo `{rec['repo']}`, commit "
            f"`{rec['commit_hash'][:8]}`) — before/after of one function."
        )
    lines.append(
        f"\nProduced `commits/commits.parquet` with "
        f"{commit_info['parquet_rows']} row(s) from "
        f"{commit_info['enriched_docs']} enriched commit doc(s)."
    )
    lines.append("")
    lines.append("## Column inventory")
    lines.append("")
    lines.append(
        f"### `code/code_{a_repo}.parquet` (packed code rows; "
        "representative — all `code/*.parquet` share this schema)"
    )
    lines.append("")
    lines.append(_fmt_cols(code_cols))
    lines.append("")
    lines.append("### `commits/commits.parquet` (tokenized enriched commit docs)")
    lines.append("")
    lines.append(_fmt_cols(commit_cols))
    lines.append("")

    (FIXTURE_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    # Pre-flight: fail loud if any required tool/input is missing.
    for label, path in [
        ("python", Path(PY)),
        ("index_project.py", INDEX_PROJECT),
        ("process_commits.py", PROCESS_COMMITS),
        ("clang_enriched_to_parquet.py", CLANG_ENRICHED_TO_PARQUET),
        ("pack_enriched_rows.py", PACK_ENRICHED_ROWS),
        ("tokenizer.json", TOKENIZER_JSON),
    ]:
        if not path.exists():
            raise RuntimeError(
                f"[make_golden_mini] PRE-FLIGHT: required {label} not found.\n"
                f"  WHERE: {path}"
            )

    # Recreate fixture output dirs (deterministic, no stale shards).
    if FIXTURE_DIR.exists():
        shutil.rmtree(FIXTURE_DIR)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="golden_mini_") as tmp:
        work = Path(tmp)
        code_info = build_code_fixture(work)
        commit_info = build_commits_fixture(work)
        write_readme(code_info, commit_info)

    print("[make_golden_mini] DONE.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
