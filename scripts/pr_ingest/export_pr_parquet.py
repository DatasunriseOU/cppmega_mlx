#!/usr/bin/env python3
"""Export PR discussions from prs.sqlite into route-by-fit training parquet.

This is the standalone PR stream companion to the commit pipeline. Commit docs
already inject ``record['pr_discussion']`` at the head of PRE/POST/diff samples;
this script emits the same PR discussion content as its own document stream for
curriculum mixes that want explicit ``pr`` batches.

The output is intentionally compatible with the existing conveyor layout:

    outputs/reindexed_pr/{1024,2048,4096,...}/pr_discussions_<offset>.parquet

RULE #1: every stage uses the existing materializer/packer path and raises on
missing input, empty output, or malformed store rows. There is no fallback path.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import streaming_reindex as sr  # noqa: E402
from streaming_reindex_commits import route_by_fit, recompress_zstd_max  # noqa: E402
from scripts.pr_ingest.pr_store import connect, get_by_pr  # noqa: E402
from scripts.pr_ingest.render_discussion import render_discussion  # noqa: E402


DEFAULT_STORE = REPO_ROOT / "outputs" / "pr_ingest" / "prs.sqlite"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "reindexed_pr"
ZSTD_LEVELS = (1024, 2048, 4096, 8192, 16384)


def _comment_safe(text: str) -> str:
    return text.replace("*/", "* /")


def _render_training_doc(rec: dict) -> str:
    discussion = render_discussion(rec)
    if not discussion:
        return ""
    lines = [
        "/*",
        "@doc_type pr_discussion",
        f"@repo {rec['repo']}",
        f"@pr {rec['pr_number']}",
    ]
    sha = rec.get("merge_commit_sha")
    if sha:
        lines.append(f"@merge_commit_sha {sha}")
    lines.extend(["@discussion", _comment_safe(discussion), "*/", ""])
    return "\n".join(lines)


def _iter_pr_keys(
    conn: sqlite3.Connection,
    *,
    repo: str | None,
    offset: int,
    limit: int | None,
):
    sql = "SELECT repo, pr_number FROM prs"
    params: list[object] = []
    if repo:
        sql += " WHERE repo=?"
        params.append(repo)
    sql += " ORDER BY repo, pr_number LIMIT ? OFFSET ?"
    params.append(-1 if limit is None else int(limit))
    params.append(int(offset))
    yield from conn.execute(sql, params)


def _write_pr_jsonl(
    conn: sqlite3.Connection,
    out_jsonl: Path,
    *,
    repo: str | None,
    offset: int,
    limit: int | None,
) -> int:
    n = 0
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in _iter_pr_keys(conn, repo=repo, offset=offset, limit=limit):
            rec = get_by_pr(conn, row["repo"], int(row["pr_number"]))
            if rec is None:
                raise RuntimeError(
                    f"PR key disappeared during export: {row['repo']}#{row['pr_number']}"
                )
            text = _render_training_doc(rec)
            if not text:
                continue
            payload = {
                "text": text,
                "source_text": text,
                "repo": rec["repo"],
                "filepath": f"PR#{rec['pr_number']}",
                "doc_type": "pr_discussion",
                "language": "pr",
                "pr_number": int(rec["pr_number"]),
                "commit_hash": rec.get("merge_commit_sha") or "",
                "constituent_provenance_json": json.dumps(
                    [{
                        "kind": "pr_discussion",
                        "repo": rec["repo"],
                        "pr_number": int(rec["pr_number"]),
                        "merge_commit_sha": rec.get("merge_commit_sha") or "",
                    }],
                    separators=(",", ":"),
                ),
            }
            fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
            n += 1
    if n == 0:
        raise RuntimeError(
            f"PR export produced zero rendered docs (repo={repo!r}, offset={offset}, limit={limit})"
        )
    return n


def _append_output(shard_name: str, packed: Path, target_length: int, output_root: Path) -> dict:
    out_dir = output_root / str(target_length)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{shard_name}.parquet"
    dest.write_bytes(packed.read_bytes())
    recompress_zstd_max(dest)
    return sr._parquet_stats(dest, target_length)


def export_pr_parquet(args: argparse.Namespace) -> dict:
    store = Path(args.store)
    if not store.exists():
        raise FileNotFoundError(f"--store does not exist: {store}")
    output_root = Path(args.output_root)
    lengths = tuple(sorted(int(x) for x in args.target_lengths.split(",") if x.strip()))
    if not lengths:
        raise ValueError("--target-lengths produced no lengths")

    conn = connect(str(store), create=False)
    shard_name = f"pr_discussions_{int(args.offset):08d}"
    with tempfile.TemporaryDirectory(prefix="pr_parquet_export_") as tmp:
        work = Path(tmp)
        jsonl = work / f"{shard_name}.jsonl"
        n_docs = _write_pr_jsonl(
            conn,
            jsonl,
            repo=args.repo,
            offset=int(args.offset),
            limit=args.limit,
        )
        tok = sr.stage_materialize(
            shard_name,
            jsonl,
            work,
            memory_limit_gb=float(args.memory_limit_gb),
        )
        routed = route_by_fit(tok, lengths, work / "routed")
        if not routed:
            raise RuntimeError(f"no PR docs routed for {jsonl}")
        per_length: dict[str, dict] = {}
        for length, route_parquet in sorted(routed.items()):
            packed = sr.stage_pack(shard_name, route_parquet, length, work)
            per_length[str(length)] = _append_output(
                shard_name, packed, length, output_root,
            )
    return {
        "source": "pr",
        "store": str(store),
        "output_root": str(output_root),
        "shard": shard_name,
        "rendered_docs": n_docs,
        "lengths": per_length,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=str(DEFAULT_STORE))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--target-lengths", default=",".join(str(x) for x in ZSTD_LEVELS))
    p.add_argument("--repo", default=None, help="Optional owner/repo filter.")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=10_000)
    p.add_argument("--memory-limit-gb", type=float, default=10.0)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    print(json.dumps(export_pr_parquet(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
