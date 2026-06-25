#!/usr/bin/env python3
"""Tier-2 PR ingest, component (5): GraphQL-PRIMARY whole-repo PR streamer.

The HYBRID strategy the user chose: GraphQL is the PRIMARY (free) path; GH
Archive is only the FALLBACK when we hit a wall on a particular repo (too many
PRs / chronic rate limiting). Where ``github_graphql_fallback.py`` fetches ONE
(repo, pr_number) at a time (gap-filling), THIS module streams EVERY PR of EVERY
repo in ``repo_list.json`` by paginating the ``pullRequests`` connection, and
folds each PR into the same ``pr_store`` (by (repo,pr_number) AND by
(repo, mergeCommit.oid)).

For each repo we page::

  repository(owner,name){
    pullRequests(first:100, after:$cursor, states:[MERGED,CLOSED,OPEN],
                 orderBy:{field:CREATED_AT, direction:ASC}){
      nodes{ number mergeCommit{oid} title body
             comments(first:100){nodes{author{login} body createdAt}}
             reviews(first:50){nodes{author{login} state body submittedAt}}
             closingIssuesReferences(first:20){nodes{number title body}} }
      pageInfo{ hasNextPage endCursor } } }

Token rotation (REUSED from github_graphql_fallback.TokenPool): round-robin over
the 6 tokens (the 5 PATs in ``secrets/gh_tokens.txt`` PLUS the ``gh auth token``
of the logged-in CLI account). On a token's primary limit
(X-RateLimit-Remaining==0), secondary/abuse limit, or 429 we cool that token and
rotate to the next; when EVERY token is cooling we FAIL LOUD (RULE #1) after
recording the (repo, cursor) so a later run resumes mid-repo — we never silently
skip a repo on token exhaustion.

Checkpointing / resume (RULE #1: a crash must not lose progress, and must not
silently drop a repo):
  * A MANIFEST (JSON) records, per repo: status (pending|in_progress|done|
    fallback), the last successfully-committed ``endCursor``, and PR counts.
  * On (re)start we skip ``done`` repos and resume ``in_progress`` repos FROM
    their saved cursor. The manifest is rewritten atomically after each
    committed page, so resume is per-repo AND per-cursor.

GH Archive FALLBACK HOOK (does NOT block the stream): if a repo trips the
fallback heuristic — its total PR count exceeds ``--fallback-pr-threshold`` or it
rate-limits us more than ``--fallback-ratelimit-trips`` times — we mark it
``fallback`` in the manifest AND append it to ``--fallback-list`` (JSONL) for the
gharchive_run.sh path to pick up, then MOVE ON to the next repo. We do not abort
the whole stream for one pathological repo.

Usage:
  python3 scripts/pr_ingest/graphql_pr_stream.py \
      --repo-list outputs/pr_ingest/repo_list.json \
      --store out/pr_store.sqlite \
      --tokens secrets/gh_tokens.txt \
      --manifest out/graphql_stream_manifest.json \
      --fallback-list out/graphql_fallback_repos.jsonl

  # Validate on ONE explicit repo before repo_list.json exists:
  python3 scripts/pr_ingest/graphql_pr_stream.py \
      --repo tilelang/tilelang --store out/pr_store.sqlite \
      --tokens secrets/gh_tokens.txt --manifest out/m.json \
      --fallback-list out/fb.jsonl --max-prs 5
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Local imports of the already-ported Tier-2 toolkit (RULE: REUSE, don't fork).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pr_store  # noqa: E402
from github_graphql_fallback import TokenPool, load_tokens  # noqa: E402

GQL_URL = "https://api.github.com/graphql"

# One page of a repo's pullRequests connection. Each PR's comments/reviews are
# fetched first-page only (100 comments / 50 reviews) inline; the rare PR with
# MORE than that is finished by the per-PR gap-filler in
# github_graphql_fallback.py (we record it for that path). This keeps the stream
# query cheap and bounded so we page repos fast and rate-limit-lightly.
REPO_PR_QUERY = """
query($owner:String!, $name:String!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(first:100, after:$cursor,
                 states:[MERGED,CLOSED,OPEN],
                 orderBy:{field:CREATED_AT, direction:ASC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        body
        mergeCommit { oid }
        comments(first:100) {
          totalCount
          nodes { author { login } body createdAt }
        }
        reviews(first:50) {
          totalCount
          nodes { author { login } state body submittedAt }
        }
        closingIssuesReferences(first:20) {
          nodes { number title body }
        }
      }
    }
  }
}
"""


# --------------------------------------------------------------------------- #
# Token loading: 5 PATs from the file + the gh CLI account token = 6 tokens.   #
# --------------------------------------------------------------------------- #
def load_all_tokens(tokens_file: str, include_gh_cli: bool = True) -> list[str]:
    """Reuse github_graphql_fallback.load_tokens, then append `gh auth token`.

    RULE #1: if --include-gh-cli is on and `gh auth token` cannot be obtained we
    RAISE (we do not silently run with fewer tokens than the user asked for).
    """
    toks = load_tokens(tokens_file)
    if include_gh_cli:
        try:
            out = subprocess.run(
                ["gh", "auth", "token"],
                check=True, capture_output=True, text=True, timeout=30,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise SystemExit(
                f"[graphql-stream] --include-gh-cli set but `gh auth token` "
                f"failed: {e}. Run `gh auth login`, or pass --no-gh-cli."
            )
        gh_tok = (out.stdout or "").strip()
        if not gh_tok:
            raise SystemExit(
                "[graphql-stream] `gh auth token` returned empty; "
                "run `gh auth login` or pass --no-gh-cli."
            )
        if gh_tok not in toks:
            toks.append(gh_tok)
    if not toks:
        raise SystemExit("[graphql-stream] no tokens available (file + gh cli both empty)")
    return toks


# --------------------------------------------------------------------------- #
# Manifest: per-repo resumable checkpoint.                                     #
# --------------------------------------------------------------------------- #
class Manifest:
    """Per-(repo, endCursor) resumable checkpoint, persisted atomically."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict = {"repos": {}}
        if os.path.exists(path):
            with open(path) as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError as e:
                    raise SystemExit(
                        f"[graphql-stream] manifest {path} is corrupt JSON: {e}. "
                        f"Move it aside to start fresh (RULE #1: not silently overwriting)."
                    )
            if "repos" not in self.data:
                raise SystemExit(
                    f"[graphql-stream] manifest {path} missing 'repos' key (wrong file?)"
                )

    def get(self, repo: str) -> dict:
        return self.data["repos"].get(repo, {})

    def status(self, repo: str) -> str:
        return self.get(repo).get("status", "pending")

    def cursor(self, repo: str):
        return self.get(repo).get("cursor")

    def update(self, repo: str, **fields) -> None:
        rec = self.data["repos"].setdefault(repo, {})
        rec.update(fields)
        self._flush()

    def _flush(self) -> None:
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)  # atomic on POSIX


# --------------------------------------------------------------------------- #
# HTTP POST with the same rate-limit semantics as the fallback gap-filler.     #
# --------------------------------------------------------------------------- #
def _post(token: str, variables: dict) -> tuple[int, dict, dict]:
    payload = json.dumps({"query": REPO_PR_QUERY, "variables": variables}).encode()
    req = urllib.request.Request(GQL_URL, data=payload, method="POST")
    req.add_header("Authorization", f"bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "cppmega-pr-ingest")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        headers = dict(e.headers or {})
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {"message": str(e)}
        return e.code, headers, body


class AllTokensExhausted(Exception):
    """Raised when every token is cooling. Carries the soonest reset (s)."""

    def __init__(self, soonest_s: float):
        super().__init__(f"all tokens rate-limited; soonest reset ~{soonest_s:.0f}s")
        self.soonest_s = soonest_s


class RepoRateLimited(Exception):
    """Soft signal: this page kept rate-limiting; caller may route to fallback."""


def _post_with_rotation(pool: TokenPool, variables: dict, owner: str, name: str,
                        max_retries: int) -> dict:
    """POST one page, rotating tokens on rate limits.

    Returns the JSON body on success. RAISES:
      * SystemExit on auth/other-403/GraphQL-errors (fail loud — real bug).
      * AllTokensExhausted when every token is cooling (caller records cursor
        and re-raises as a loud failure).
      * RepoRateLimited when we burn the per-page retry budget on THIS repo
        (caller decides: route to GH Archive fallback, do NOT abort the stream).
    """
    attempts = 0
    while True:
        status, headers, jb = _post(pool.current(), variables)

        if status == 401:
            raise SystemExit(
                f"[graphql-stream] 401 Unauthorized for token #{pool.idx} "
                f"({owner}/{name}); fix the PAT(s)/gh token"
            )

        remaining = headers.get("X-RateLimit-Remaining")
        retry_after = headers.get("Retry-After")
        jb_lower = json.dumps(jb).lower()
        is_secondary = status == 403 and (
            retry_after is not None
            or "secondary rate limit" in jb_lower
            or "abuse" in jb_lower
        )
        if status == 403 and not is_secondary and "rate limit" not in jb_lower:
            raise SystemExit(
                f"[graphql-stream] 403 Forbidden (not a rate limit) for "
                f"{owner}/{name}: {jb.get('message', jb)}"
            )
        if is_secondary or (remaining is not None and remaining == "0") or status == 429:
            attempts += 1
            if attempts > max_retries:
                # Burned the per-page budget on THIS repo -> soft signal up.
                raise RepoRateLimited()
            if retry_after is not None:
                wait = float(retry_after)
            else:
                reset = headers.get("X-RateLimit-Reset")
                wait = max(1.0, float(reset) - time.time()) if reset else 30.0
            pool.cool(wait)
            if not pool.advance():
                soonest = min(pool.cooldown_until) - time.time()
                raise AllTokensExhausted(soonest)
            continue

        if status != 200:
            raise SystemExit(
                f"[graphql-stream] HTTP {status} for {owner}/{name}: {jb}"
            )
        if jb.get("errors"):
            # A NOT_FOUND on the repository is a real, recordable outcome, not a
            # transient. Surface it loudly with the repo so the caller logs it.
            raise SystemExit(
                f"[graphql-stream] GraphQL errors for {owner}/{name}: {jb['errors']}"
            )
        return jb


# --------------------------------------------------------------------------- #
# Map a GraphQL PR node -> the pr_store.upsert_record() record shape.          #
# --------------------------------------------------------------------------- #
def _pr_node_to_record(repo: str, node: dict) -> tuple[dict, bool]:
    """Return (record, truncated). ``truncated`` is True when this PR had MORE
    comments/reviews than one page (the per-PR gap-filler should finish it)."""
    number = node["number"]
    mc = node.get("mergeCommit")
    merge_sha = mc.get("oid") if mc else None

    comments: list[dict] = []
    reviews: list[dict] = []
    linked: list[dict] = []
    truncated = False

    cn = node.get("comments") or {}
    for n in cn.get("nodes", []):
        comments.append({
            "user": (n.get("author") or {}).get("login", "") if n.get("author") else "",
            "body": n.get("body", "") or "",
            "created_at": n.get("createdAt", "") or "",
            "kind": "comment",
        })
    if (cn.get("totalCount") or 0) > len(cn.get("nodes", [])):
        truncated = True

    rv = node.get("reviews") or {}
    for n in rv.get("nodes", []):
        reviews.append({
            "user": (n.get("author") or {}).get("login", "") if n.get("author") else "",
            "state": n.get("state", "") or "",
            "body": n.get("body", "") or "",
            "created_at": n.get("submittedAt", "") or "",
        })
    if (rv.get("totalCount") or 0) > len(rv.get("nodes", [])):
        truncated = True

    for li in (node.get("closingIssuesReferences") or {}).get("nodes", []):
        linked.append({
            "number": li["number"],
            "title": li.get("title", "") or "",
            "body": li.get("body", "") or "",
        })

    comments.sort(key=lambda c: c.get("created_at") or "")
    reviews.sort(key=lambda r: r.get("created_at") or "")
    rec = {
        "repo": repo,
        "pr_number": number,
        "merge_commit_sha": merge_sha,
        "pr_title": node.get("title", "") or "",
        "pr_body": node.get("body", "") or "",
        "comments": comments,
        "reviews": reviews,
        "linked_issues": linked,
    }
    return rec, truncated


# --------------------------------------------------------------------------- #
# Stream ONE repo, paginating + checkpointing per page.                        #
# --------------------------------------------------------------------------- #
def stream_repo(
    pool: TokenPool,
    conn,
    manifest: Manifest,
    repo: str,
    *,
    fallback_pr_threshold: int,
    fallback_ratelimit_trips: int,
    fallback_list_path: str,
    max_prs: int | None = None,
    max_retries: int = 8,
    truncated_targets_path: str | None = None,
) -> dict:
    """Stream all PRs of ``repo``. Resumes from manifest cursor if present.

    Returns a per-repo stats dict. RAISES AllTokensExhausted up to the caller
    (after the manifest already holds the resumable cursor). Routes the repo to
    the GH Archive fallback (and returns) when the fallback heuristic trips.
    """
    if "/" not in repo:
        raise SystemExit(f"[graphql-stream] repo must be 'owner/name', got {repo!r}")
    owner, name = repo.split("/", 1)

    cursor = manifest.cursor(repo)  # None on a fresh repo, else resume point
    manifest.update(repo, status="in_progress")
    stats = {"repo": repo, "prs": 0, "truncated": 0, "pages": 0, "ratelimit_trips": 0}
    total_count = None

    while True:
        variables = {"owner": owner, "name": name, "cursor": cursor}
        try:
            jb = _post_with_rotation(pool, variables, owner, name, max_retries)
        except RepoRateLimited:
            stats["ratelimit_trips"] += 1
            if stats["ratelimit_trips"] >= fallback_ratelimit_trips:
                _route_to_fallback(
                    manifest, repo, fallback_list_path, cursor,
                    reason="ratelimit", stats=stats, total_count=total_count,
                )
                return stats
            # Another wall on the same page but under the trip cap: brief pause,
            # then retry the SAME cursor (no progress lost).
            time.sleep(2.0)
            continue

        prconn = ((jb.get("data") or {}).get("repository") or {}).get("pullRequests")
        if prconn is None:
            # repository present but pullRequests null is anomalous -> fail loud.
            raise SystemExit(
                f"[graphql-stream] {repo}: data.repository.pullRequests is null "
                f"(repo missing or no PR access): {jb}"
            )

        if total_count is None:
            total_count = prconn.get("totalCount")
            # Fallback heuristic on size: a huge repo is routed to GH Archive at
            # the very first page, BEFORE we burn quota paging it here.
            if (total_count or 0) > fallback_pr_threshold:
                _route_to_fallback(
                    manifest, repo, fallback_list_path, cursor,
                    reason="too_many_prs", stats=stats, total_count=total_count,
                )
                return stats

        for node in prconn.get("nodes", []):
            rec, truncated = _pr_node_to_record(repo, node)
            pr_store.upsert_record(conn, rec)  # commits inside upsert_record
            stats["prs"] += 1
            if truncated:
                stats["truncated"] += 1
                if truncated_targets_path:
                    _append_jsonl(
                        truncated_targets_path,
                        {"repo": repo, "pr_number": rec["pr_number"]},
                    )
            if max_prs is not None and stats["prs"] >= max_prs:
                manifest.update(
                    repo, status="in_progress", cursor=cursor,
                    prs=stats["prs"], total_count=total_count,
                    note=f"stopped at --max-prs={max_prs}",
                )
                stats["pages"] += 1
                return stats

        stats["pages"] += 1
        pi = prconn.get("pageInfo") or {}
        end_cursor = pi.get("endCursor")
        has_next = pi.get("hasNextPage")

        # Checkpoint AFTER the page's PRs are committed: resume continues from
        # this exact endCursor (per-repo + per-cursor resumability).
        cursor = end_cursor
        manifest.update(
            repo,
            status="in_progress" if has_next else "done",
            cursor=cursor if has_next else None,
            prs=stats["prs"], truncated=stats["truncated"],
            total_count=total_count,
        )
        if not has_next:
            break

    return stats


def _route_to_fallback(manifest: Manifest, repo: str, fallback_list_path: str,
                       cursor, *, reason: str, stats: dict, total_count) -> None:
    """Mark repo for the GH Archive path WITHOUT aborting the stream."""
    manifest.update(
        repo, status="fallback", cursor=cursor,
        fallback_reason=reason, prs=stats["prs"], total_count=total_count,
    )
    _append_jsonl(fallback_list_path, {
        "repo": repo, "reason": reason, "cursor": cursor,
        "total_count": total_count, "prs_streamed": stats["prs"],
    })
    stats["fallback"] = reason
    sys.stderr.write(
        f"[graphql-stream] {repo}: routed to GH Archive fallback ({reason}; "
        f"total_count={total_count}); continuing stream.\n"
    )
    sys.stderr.flush()


def _append_jsonl(path: str, obj: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# --------------------------------------------------------------------------- #
# Repo list loading.                                                          #
# --------------------------------------------------------------------------- #
def load_repo_list(path: str) -> list[str]:
    """Load owner/repo strings from build_repo_list.py's repo_list.json."""
    if not os.path.exists(path):
        raise SystemExit(
            f"[graphql-stream] --repo-list not found: {path}. "
            f"Pass --repo owner/name to validate on a single repo instead."
        )
    with open(path) as f:
        data = json.load(f)
    repos = data.get("repos")
    if not isinstance(repos, list):
        raise SystemExit(
            f"[graphql-stream] {path} has no 'repos' list (wrong file?)"
        )
    out = []
    for r in repos:
        owner_repo = r.get("owner_repo")
        if not owner_repo:
            raise SystemExit(
                f"[graphql-stream] repo entry missing 'owner_repo': {r}"
            )
        out.append(owner_repo)
    if not out:
        raise SystemExit(f"[graphql-stream] {path} resolved zero repos")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo-list", help="repo_list.json from build_repo_list.py")
    src.add_argument("--repo", help="single owner/name (validate before repo_list exists)")

    ap.add_argument("--store", required=True, help="pr_store SQLite path")
    ap.add_argument("--tokens", required=True, help="File of GitHub PATs, one/line")
    ap.add_argument("--manifest", required=True, help="Per-repo resume manifest JSON")
    ap.add_argument("--fallback-list", required=True,
                    help="JSONL of repos routed to the GH Archive fallback")
    ap.add_argument("--truncated-targets",
                    help="JSONL of (repo,pr_number) whose comments/reviews "
                         "overflowed one page (finish via github_graphql_fallback.py)")
    ap.add_argument("--no-gh-cli", dest="include_gh_cli", action="store_false",
                    help="Do NOT append `gh auth token` as the 6th token")
    ap.add_argument("--fallback-pr-threshold", type=int, default=20000,
                    help="Route a repo to GH Archive if totalCount exceeds this")
    ap.add_argument("--fallback-ratelimit-trips", type=int, default=3,
                    help="Route a repo to GH Archive after this many page-level "
                         "rate-limit walls")
    ap.add_argument("--max-retries", type=int, default=8,
                    help="Per-page rate-limit retry budget before a fallback trip")
    ap.add_argument("--max-prs", type=int,
                    help="Stop each repo after N PRs (validation/smoke)")
    ap.add_argument("--max-repos", type=int,
                    help="Process at most N repos this run (validation/smoke)")
    args = ap.parse_args(argv)

    pool = TokenPool(load_all_tokens(args.tokens, include_gh_cli=args.include_gh_cli))
    sys.stderr.write(f"[graphql-stream] token pool size = {len(pool.tokens)}\n")
    conn = pr_store.connect(args.store, create=True)
    manifest = Manifest(args.manifest)

    repos = [args.repo] if args.repo else load_repo_list(args.repo_list)
    if args.max_repos is not None:
        repos = repos[: args.max_repos]

    totals = {"repos_done": 0, "repos_fallback": 0, "prs": 0, "truncated": 0}
    for repo in repos:
        st = manifest.status(repo)
        if st == "done":
            sys.stderr.write(f"[graphql-stream] {repo}: already done, skipping\n")
            continue
        if st == "fallback":
            sys.stderr.write(f"[graphql-stream] {repo}: already routed to fallback, skipping\n")
            totals["repos_fallback"] += 1
            continue

        sys.stderr.write(
            f"[graphql-stream] streaming {repo}"
            + (f" (resume cursor={manifest.cursor(repo)})" if st == "in_progress" else "")
            + "\n"
        )
        sys.stderr.flush()
        try:
            stats = stream_repo(
                pool, conn, manifest, repo,
                fallback_pr_threshold=args.fallback_pr_threshold,
                fallback_ratelimit_trips=args.fallback_ratelimit_trips,
                fallback_list_path=args.fallback_list,
                max_prs=args.max_prs, max_retries=args.max_retries,
                truncated_targets_path=args.truncated_targets,
            )
        except AllTokensExhausted as e:
            # RULE #1: the manifest already holds this repo's resumable cursor.
            # Do NOT silently skip — crash loud with how to resume.
            cur = manifest.cursor(repo)
            raise SystemExit(
                f"[graphql-stream] ALL {len(pool.tokens)} tokens rate-limited while "
                f"streaming {repo} (soonest reset ~{e.soonest_s:.0f}s). "
                f"Progress saved: manifest={args.manifest} repo={repo} "
                f"cursor={cur!r}. Re-run the SAME command after the reset (or add "
                f"more PATs) to RESUME mid-repo. FAILING LOUD per RULE #1."
            )

        totals["prs"] += stats["prs"]
        totals["truncated"] += stats["truncated"]
        if stats.get("fallback"):
            totals["repos_fallback"] += 1
        elif manifest.status(repo) == "done":
            totals["repos_done"] += 1
        sys.stderr.write(
            f"[graphql-stream] {repo}: prs={stats['prs']} pages={stats['pages']} "
            f"truncated={stats['truncated']} "
            f"{'FALLBACK=' + stats['fallback'] if stats.get('fallback') else 'done'}\n"
        )
        sys.stderr.flush()

    sys.stderr.write(
        f"[graphql-stream] DONE repos_done={totals['repos_done']} "
        f"repos_fallback={totals['repos_fallback']} prs={totals['prs']} "
        f"truncated={totals['truncated']} store={args.store}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
