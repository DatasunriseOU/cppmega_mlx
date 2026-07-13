#!/usr/bin/env python3
"""Tier-2 PR ingest, component (3): the local PR store.

A resumable SQLite store of PR discussions keyed by BOTH:
  * (repo, pr_number)
  * (repo, merge_commit_sha)

so the commit pipeline can look up a PR by either the parsed PR number or the
merge-commit SHA (which is what GH Archive's $.pull_request.merge_commit_sha
gives us, and is the authoritative join to our per-commit ``commit_hash``).

Each stored record is:
    {
      "repo": "owner/repo",
      "pr_number": 1234,
      "merge_commit_sha": "abc123..." | None,
      "pr_title": str,
      "pr_body": str,
      "comments": [{"user","body","created_at","path"?}, ...]   # ordered
      "reviews":  [{"user","state","body","created_at"}, ...]   # ordered
      "linked_issues": [{"number","title","body"}, ...]
    }

Population:
  * ingest-gharchive: fold the NDJSON shards exported by gharchive_run.sh into
    the store. MULTIPLE events per PR are deduped/merged (title/body from the
    PR event; comments/reviews appended in created_at order; exact-duplicate
    comment/review bodies collapsed). Idempotent + resumable: re-running folds
    the same data without duplicating (events carry created_at + body, used as
    the dedup key).

Lookup:
  * get_by_pr(repo, pr_number) / get_by_sha(repo, sha) -> assembled record.

RULE #1: no silent fallback. Malformed input rows RAISE with the offending row;
an unknown subcommand RAISES; a missing store on lookup RAISES.

Usage:
  python3 scripts/pr_ingest/pr_store.py ingest-gharchive \
      --store out/pr_store.sqlite --input 'out/pr_discussion_raw-*.json'
  python3 scripts/pr_ingest/pr_store.py get \
      --store out/pr_store.sqlite --repo opencv/opencv --pr 1234
  python3 scripts/pr_ingest/pr_store.py get \
      --store out/pr_store.sqlite --repo opencv/opencv --sha abc123
  python3 scripts/pr_ingest/pr_store.py stats --store out/pr_store.sqlite
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS prs (
    repo             TEXT NOT NULL,
    pr_number        INTEGER NOT NULL,
    merge_commit_sha TEXT,
    pr_title         TEXT NOT NULL DEFAULT '',
    pr_body          TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (repo, pr_number)
);
-- SHA -> (repo, pr_number) index table so we can look up by merge-commit SHA.
CREATE TABLE IF NOT EXISTS pr_by_sha (
    repo             TEXT NOT NULL,
    merge_commit_sha TEXT NOT NULL,
    pr_number        INTEGER NOT NULL,
    PRIMARY KEY (repo, merge_commit_sha)
);
CREATE TABLE IF NOT EXISTS comments (
    repo       TEXT NOT NULL,
    pr_number  INTEGER NOT NULL,
    user       TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL DEFAULT '',
    path       TEXT,
    created_at TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL DEFAULT 'comment',  -- comment | review_comment
    -- dedup key: same PR + body + created_at == same event folded twice.
    UNIQUE (repo, pr_number, kind, created_at, body)
);
CREATE TABLE IF NOT EXISTS reviews (
    repo       TEXT NOT NULL,
    pr_number  INTEGER NOT NULL,
    user       TEXT NOT NULL DEFAULT '',
    state      TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    UNIQUE (repo, pr_number, created_at, body)
);
CREATE TABLE IF NOT EXISTS linked_issues (
    repo      TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    number    INTEGER NOT NULL,
    title     TEXT NOT NULL DEFAULT '',
    body      TEXT NOT NULL DEFAULT '',
    UNIQUE (repo, pr_number, number)
);
"""


def connect(
    store_path: str,
    create: bool = True,
    *,
    readonly: bool = False,
) -> sqlite3.Connection:
    if readonly and create:
        raise ValueError("readonly PR store connections cannot create a database")
    if not create and not os.path.exists(store_path):
        raise SystemExit(f"[pr_store] store does not exist: {store_path}")
    if create:
        d = os.path.dirname(os.path.abspath(store_path))
        os.makedirs(d, exist_ok=True)
    if readonly:
        uri = Path(store_path).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    else:
        conn = sqlite3.connect(store_path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    # Set busy_timeout FIRST. The parallel graphql_pr_stream --workers path opens
    # many writer connections concurrently; with the timeout in place the WAL
    # switch and the CREATE TABLE statements below WAIT for a competing writer
    # instead of erroring out with "database is locked" (fail-loud preserved:
    # SQLite still raises if the lock is not released within the timeout).
    conn.execute("PRAGMA busy_timeout=60000")
    if readonly:
        return conn
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def _int_or_none(v) -> Optional[int]:
    if v in (None, "", "null"):
        return None
    return int(v)


def _upsert_pr_meta(
    conn: sqlite3.Connection, repo: str, pr_number: int,
    merge_commit_sha: Optional[str], title: str, body: str,
) -> None:
    """Insert/merge PR title+body+sha. Non-empty new values win over empty old."""
    cur = conn.execute(
        "SELECT merge_commit_sha, pr_title, pr_body FROM prs WHERE repo=? AND pr_number=?",
        (repo, pr_number),
    )
    row = cur.fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO prs(repo, pr_number, merge_commit_sha, pr_title, pr_body) "
            "VALUES (?,?,?,?,?)",
            (repo, pr_number, merge_commit_sha, title or "", body or ""),
        )
    else:
        new_sha = merge_commit_sha or row["merge_commit_sha"]
        new_title = title or row["pr_title"]
        new_body = body or row["pr_body"]
        conn.execute(
            "UPDATE prs SET merge_commit_sha=?, pr_title=?, pr_body=? "
            "WHERE repo=? AND pr_number=?",
            (new_sha, new_title, new_body, repo, pr_number),
        )
    if merge_commit_sha:
        conn.execute(
            "INSERT OR REPLACE INTO pr_by_sha(repo, merge_commit_sha, pr_number) "
            "VALUES (?,?,?)",
            (repo, merge_commit_sha, pr_number),
        )


def ingest_gharchive_rows(conn: sqlite3.Connection, rows) -> dict:
    """Fold an iterable of GH Archive NDJSON rows into the store.

    Each row is one event (PullRequestEvent / IssueCommentEvent /
    PullRequestReviewCommentEvent / PullRequestReviewEvent) projected by
    gharchive_query.sql. Dedup is via the UNIQUE constraints (INSERT OR IGNORE).
    """
    counts = {"pr": 0, "comments": 0, "review_comments": 0, "reviews": 0, "skipped": 0}
    for i, row in enumerate(rows):
        repo = row.get("repo_name")
        pr_number = _int_or_none(row.get("pr_number"))
        if not repo or pr_number is None:
            counts["skipped"] += 1
            continue
        etype = row.get("event_type") or ""
        sha = row.get("merge_commit_sha") or None
        created_at = row.get("created_at") or ""

        if etype == "PullRequestEvent":
            _upsert_pr_meta(
                conn, repo, pr_number, sha,
                row.get("pr_title") or "", row.get("pr_body") or "",
            )
            counts["pr"] += 1
        elif etype == "IssueCommentEvent":
            # PR comment thread. Ensure a PR row exists (title/body may arrive later).
            _upsert_pr_meta(conn, repo, pr_number, sha, "", "")
            conn.execute(
                "INSERT OR IGNORE INTO comments"
                "(repo, pr_number, user, body, path, created_at, kind) "
                "VALUES (?,?,?,?,?,?, 'comment')",
                (repo, pr_number, row.get("comment_user") or "",
                 row.get("comment_body") or "", None, created_at),
            )
            counts["comments"] += 1
        elif etype == "PullRequestReviewCommentEvent":
            _upsert_pr_meta(conn, repo, pr_number, sha, "", "")
            conn.execute(
                "INSERT OR IGNORE INTO comments"
                "(repo, pr_number, user, body, path, created_at, kind) "
                "VALUES (?,?,?,?,?,?, 'review_comment')",
                (repo, pr_number, row.get("comment_user") or "",
                 row.get("comment_body") or "", row.get("comment_path"), created_at),
            )
            counts["review_comments"] += 1
        elif etype == "PullRequestReviewEvent":
            _upsert_pr_meta(conn, repo, pr_number, sha, "", "")
            conn.execute(
                "INSERT OR IGNORE INTO reviews"
                "(repo, pr_number, user, state, body, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (repo, pr_number, row.get("review_user") or "",
                 row.get("review_state") or "", row.get("review_body") or "", created_at),
            )
            counts["reviews"] += 1
        else:
            raise SystemExit(
                f"[pr_store] row {i}: unknown event_type {etype!r} (corrupt export?)"
            )
    conn.commit()
    return counts


def upsert_record(conn: sqlite3.Connection, rec: dict, *, commit: bool = True) -> None:
    """Insert a fully-assembled PR record (used for tests / GraphQL fallback).

    Keyed by (repo, pr_number); also indexes merge_commit_sha when present.
    Comments/reviews/linked_issues are appended with dedup via UNIQUE.
    """
    repo = rec["repo"]
    pr_number = int(rec["pr_number"])
    _upsert_pr_meta(
        conn, repo, pr_number, rec.get("merge_commit_sha"),
        rec.get("pr_title", ""), rec.get("pr_body", ""),
    )
    for c in rec.get("comments", []):
        conn.execute(
            "INSERT OR IGNORE INTO comments"
            "(repo, pr_number, user, body, path, created_at, kind) "
            "VALUES (?,?,?,?,?,?,?)",
            (repo, pr_number, c.get("user", ""), c.get("body", ""),
             c.get("path"), c.get("created_at", ""), c.get("kind", "comment")),
        )
    for r in rec.get("reviews", []):
        conn.execute(
            "INSERT OR IGNORE INTO reviews"
            "(repo, pr_number, user, state, body, created_at) VALUES (?,?,?,?,?,?)",
            (repo, pr_number, r.get("user", ""), r.get("state", ""),
             r.get("body", ""), r.get("created_at", "")),
        )
    for li in rec.get("linked_issues", []):
        conn.execute(
            "INSERT OR IGNORE INTO linked_issues"
            "(repo, pr_number, number, title, body) VALUES (?,?,?,?,?)",
            (repo, pr_number, int(li["number"]), li.get("title", ""), li.get("body", "")),
        )
    if commit:
        conn.commit()


def _assemble(conn: sqlite3.Connection, repo: str, pr_number: int) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM prs WHERE repo=? AND pr_number=?", (repo, pr_number)
    ).fetchone()
    if row is None:
        return None
    comments = [
        {"user": c["user"], "body": c["body"], "path": c["path"],
         "created_at": c["created_at"], "kind": c["kind"]}
        for c in conn.execute(
            "SELECT * FROM comments WHERE repo=? AND pr_number=? "
            "ORDER BY created_at, rowid", (repo, pr_number)
        )
    ]
    reviews = [
        {"user": r["user"], "state": r["state"], "body": r["body"],
         "created_at": r["created_at"]}
        for r in conn.execute(
            "SELECT * FROM reviews WHERE repo=? AND pr_number=? "
            "ORDER BY created_at, rowid", (repo, pr_number)
        )
    ]
    linked = [
        {"number": li["number"], "title": li["title"], "body": li["body"]}
        for li in conn.execute(
            "SELECT * FROM linked_issues WHERE repo=? AND pr_number=? ORDER BY number",
            (repo, pr_number),
        )
    ]
    return {
        "repo": row["repo"],
        "pr_number": row["pr_number"],
        "merge_commit_sha": row["merge_commit_sha"],
        "pr_title": row["pr_title"],
        "pr_body": row["pr_body"],
        "comments": comments,
        "reviews": reviews,
        "linked_issues": linked,
    }


def get_by_pr(conn: sqlite3.Connection, repo: str, pr_number: int) -> Optional[dict]:
    return _assemble(conn, repo, int(pr_number))


def get_by_sha(conn: sqlite3.Connection, repo: str, sha: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT pr_number FROM pr_by_sha WHERE repo=? AND merge_commit_sha=?",
        (repo, sha),
    ).fetchone()
    if row is None:
        return None
    return _assemble(conn, repo, int(row["pr_number"]))


def _iter_ndjson(paths: list[str]):
    for path in paths:
        with open(path, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(
                        f"[pr_store] {path}:{lineno} not valid JSON: {e}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local PR discussion store.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest-gharchive", help="Fold GH Archive NDJSON shards.")
    p_ing.add_argument("--store", required=True)
    p_ing.add_argument("--input", required=True, help="Glob of NDJSON shards.")

    p_get = sub.add_parser("get", help="Look up a PR by number or merge SHA.")
    p_get.add_argument("--store", required=True)
    p_get.add_argument("--repo", required=True)
    p_get.add_argument("--pr", type=int)
    p_get.add_argument("--sha")

    p_stats = sub.add_parser("stats", help="Print store row counts.")
    p_stats.add_argument("--store", required=True)

    args = parser.parse_args()

    if args.cmd == "ingest-gharchive":
        paths = sorted(glob.glob(args.input))
        if not paths:
            raise SystemExit(f"[pr_store] no files matched --input {args.input!r}")
        conn = connect(args.store, create=True)
        counts = ingest_gharchive_rows(conn, _iter_ndjson(paths))
        print(f"[pr_store] ingested {paths} -> {counts}", file=sys.stderr)
    elif args.cmd == "get":
        conn = connect(args.store, create=False)
        if args.sha:
            rec = get_by_sha(conn, args.repo, args.sha)
        elif args.pr is not None:
            rec = get_by_pr(conn, args.repo, args.pr)
        else:
            raise SystemExit("[pr_store] get requires --pr or --sha")
        if rec is None:
            raise SystemExit(f"[pr_store] MISS: {args.repo} pr={args.pr} sha={args.sha}")
        print(json.dumps(rec, indent=2))
    elif args.cmd == "stats":
        conn = connect(args.store, create=False)
        for t in ("prs", "pr_by_sha", "comments", "reviews", "linked_issues"):
            n = conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            print(f"  {t:16s} {n}")


if __name__ == "__main__":
    main()
