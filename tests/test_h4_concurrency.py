"""H4 concurrency regression tests (real files / real threads / real processes).

Covers the three H4 concurrency defects on the newly-parallel ingestion paths:

  (1) streaming_conveyor manifest clobber -- concurrent `--streams code` and
      `--streams commits` conveyor PROCESSES share outputs/conveyor/_done.json;
      the base Manifest rewrites the whole file from its stale in-memory snapshot
      so the second writer drops the first writer's keys (lost update).
      ConcurrentManifest must merge under a cross-process flock (no clobber).

  (2) graphql_pr_stream threaded fail-fast HANG -- a fatal worker error must set
      a shared stop flag so siblings abort between pages and the pool tears down
      with shutdown(wait=False, cancel_futures=True) instead of blocking.

  (3) graphql_pr_stream per-worker token pools double-spend the SAME PAT -- ALL
      workers must share ONE thread-safe pool so a cooldown set by any worker is
      honored by every worker.

These use real sqlite/json/flock and a real multiprocessing contention run; the
manifest no-clobber test deterministically FAILS against the old base Manifest
and PASSES with ConcurrentManifest.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing
import sys
import threading
import time
from pathlib import Path

import pytest


MLX_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = MLX_ROOT / "scripts"
PR_INGEST = SCRIPTS / "pr_ingest"
for _p in (SCRIPTS, PR_INGEST):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# --------------------------------------------------------------------------- #
# (1) Manifest atomicity / no-clobber.                                        #
# --------------------------------------------------------------------------- #
def test_base_manifest_clobbers_concurrent_keys(tmp_path):
    """Document the BUG: the base Manifest loses a concurrent writer's key.

    Two independently-loaded manifests (== two processes) write DISJOINT keys.
    Because base ``save()`` blindly rewrites the file from its own snapshot, the
    second writer overwrites the first writer's committed key. This is exactly
    the lost-update the fix prevents; if this assertion ever flips, the base
    class silently gained merge behavior and the fix's rationale changed.
    """
    from streaming_reindex import Manifest

    path = tmp_path / "_done.json"
    a = Manifest.load(path)   # both load empty (file does not exist yet)
    b = Manifest.load(path)

    a.mark_done("repo::code", {"rows": 1})   # writes {repo::code}
    b.mark_done("repo::r0", {"rows": 2})     # rewrites file from b's stale snapshot

    reloaded = Manifest.load(path)
    assert "repo::r0" in reloaded.done                 # last writer survives
    assert "repo::code" not in reloaded.done           # first writer CLOBBERED


def test_concurrent_manifest_no_clobber_sequential(tmp_path):
    """The FIX: ConcurrentManifest re-reads + merges, so neither key is lost."""
    import streaming_conveyor as sc

    path = tmp_path / "_done.json"
    a = sc.ConcurrentManifest.load(path)
    b = sc.ConcurrentManifest.load(path)

    a.mark_done("repo::code", {"rows": 1})
    b.mark_done("repo::r0", {"rows": 2})       # merges with on-disk repo::code
    b.mark_failed("repo::r1", "stage", "boom")

    reloaded = sc.ConcurrentManifest.load(path)
    assert reloaded.is_done("repo::code")
    assert reloaded.is_done("repo::r0")
    assert "repo::r1" in reloaded.failed
    # In-memory state of the last writer reflects the merged union too.
    assert b.is_done("repo::code")


def test_concurrent_manifest_save_is_disabled(tmp_path):
    """A blind full-file save() must RAISE (it would reintroduce the clobber)."""
    import streaming_conveyor as sc

    m = sc.ConcurrentManifest.load(tmp_path / "_done.json")
    with pytest.raises(RuntimeError, match="clobber"):
        m.save()


def test_concurrent_manifest_retry_invalidates_stale_terminal_state(tmp_path):
    """A failed re-run must never leave an older done receipt resumable."""
    import streaming_conveyor as sc

    path = tmp_path / "_done.json"
    manifest = sc.ConcurrentManifest.load(path)
    manifest.mark_done("repo::code", {"generation": "old"})
    manifest.mark_started("repo::code")

    started = sc.ConcurrentManifest.load(path)
    assert "repo::code" not in started.done
    assert "repo::code" not in started.failed

    started.mark_failed("repo::code", "index", "new run failed")
    failed = sc.ConcurrentManifest.load(path)
    assert "repo::code" not in failed.done
    assert failed.failed["repo::code"]["stage"] == "index"


def test_concurrent_manifest_prefix_restart_preserves_other_stream(tmp_path):
    """A commit restart clears its ranges without clobbering code receipts."""
    import streaming_conveyor as sc

    path = tmp_path / "_done.json"
    code_writer = sc.ConcurrentManifest.load(path)
    commit_writer = sc.ConcurrentManifest.load(path)
    code_writer.mark_done("repo::code", {"rows": 1})
    commit_writer.mark_done("repo::r0", {"rows": 2})
    commit_writer.mark_failed("repo::r10", "pack", "old failure")

    commit_writer.mark_started_prefix("repo::r")

    reloaded = sc.ConcurrentManifest.load(path)
    assert reloaded.is_done("repo::code")
    assert "repo::r0" not in reloaded.done
    assert "repo::r10" not in reloaded.failed


def _mp_manifest_writer(manifest_path: str, keys: list[str]) -> None:
    """Top-level worker for the spawn-based contention test."""
    import sys as _sys
    from pathlib import Path as _Path

    scripts = str(_Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in _sys.path:
        _sys.path.insert(0, scripts)
    import streaming_conveyor as sc

    m = sc.ConcurrentManifest.load(_Path(manifest_path))
    for k in keys:
        m.mark_done(k, {"k": k})
        time.sleep(0.0005)  # encourage real OS-level interleaving


def test_concurrent_manifest_no_clobber_under_real_processes(tmp_path):
    """REAL two-process contention: every key from BOTH processes must survive."""
    ctx = multiprocessing.get_context("spawn")
    path = tmp_path / "_done.json"
    keys_a = [f"repoA::r{i}" for i in range(60)]       # commits-stream keyspace
    keys_b = [f"repoB::code{i}" for i in range(60)]    # code-stream keyspace

    pa = ctx.Process(target=_mp_manifest_writer, args=(str(path), keys_a))
    pb = ctx.Process(target=_mp_manifest_writer, args=(str(path), keys_b))
    pa.start()
    pb.start()
    pa.join(120)
    pb.join(120)
    assert pa.exitcode == 0, f"writer A failed: exitcode={pa.exitcode}"
    assert pb.exitcode == 0, f"writer B failed: exitcode={pb.exitcode}"

    from streaming_reindex import Manifest

    final = Manifest.load(path)
    missing = [k for k in (keys_a + keys_b) if k not in final.done]
    assert not missing, f"clobber lost {len(missing)} keys, e.g. {missing[:5]}"
    assert len(final.done) == len(keys_a) + len(keys_b)


# --------------------------------------------------------------------------- #
# (3) Shared, thread-safe token pool (no per-PAT double-spend).               #
# --------------------------------------------------------------------------- #
def test_shared_token_pool_round_robin_and_shared_cooldowns():
    import graphql_pr_stream as g

    pool = g.SharedTokenPool(["t0", "t1", "t2"])
    assert [pool.acquire()[0] for _ in range(6)] == [0, 1, 2, 0, 1, 2]

    # A cooldown set once is honored on every subsequent acquire (shared state).
    pool.cool(0, 3600)
    handed = {pool.acquire()[0] for _ in range(30)}
    assert handed == {1, 2}        # token 0 is never handed out while cooling

    pool.cool(1, 3600)
    pool.cool(2, 3600)
    with pytest.raises(g.AllTokensExhausted):
        pool.acquire()             # every token cooling -> fail loud


def test_shared_token_pool_is_thread_safe():
    import graphql_pr_stream as g

    pool = g.SharedTokenPool([f"t{i}" for i in range(4)])
    seen: list[int] = []
    seen_lock = threading.Lock()
    errors: list[BaseException] = []

    def hammer() -> None:
        try:
            for _ in range(2000):
                idx, _tok = pool.acquire()
                with seen_lock:
                    seen.append(idx)
        except BaseException as exc:  # noqa: BLE001 - record for assertion
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not errors, f"shared pool was not thread-safe: {errors[:3]}"
    assert seen, "no tokens acquired"
    assert set(seen) <= {0, 1, 2, 3}        # never an out-of-range index


# --------------------------------------------------------------------------- #
# (2) Fail-fast stop flag -- abort promptly, never hang / swallow.            #
# --------------------------------------------------------------------------- #
def test_post_with_rotation_aborts_before_http_when_stopped():
    import graphql_pr_stream as g

    class FailIfAcquired:
        tokens = ["t0"]

        def acquire(self):
            raise AssertionError("token acquisition must not happen once stop_event is set")

        def cool(self, _idx, _seconds):
            raise AssertionError("token cooldown must not happen once stop_event is set")

    pool = FailIfAcquired()
    ev = threading.Event()
    ev.set()

    with pytest.raises(g.StreamAborted):
        g._post_with_rotation(
            pool, {"owner": "o", "name": "r", "cursor": None},
            "o", "r", 8, stop_event=ev,
        )


def test_stream_repo_aborts_between_pages_when_stopped(tmp_path):
    import graphql_pr_stream as g
    import pr_store

    db = tmp_path / "prs.sqlite"
    conn = pr_store.connect(str(db), create=True)
    try:
        manifest = g.Manifest(str(tmp_path / "m.json"))
        ev = threading.Event()
        ev.set()
        with pytest.raises(g.StreamAborted):
            g.stream_repo(
                pool=g.SharedTokenPool(["t0"]),
                conn=conn,
                manifest=manifest,
                repo="owner/name",
                fallback_pr_threshold=10**9,
                fallback_ratelimit_trips=99,
                fallback_list_path=str(tmp_path / "fb.jsonl"),
                stop_event=ev,
            )
    finally:
        conn.close()


def test_threaded_pool_aborts_promptly_without_hang():
    """Mirror main()'s threaded loop: a fatal worker must abort siblings fast.

    The OLD code raised SystemExit inside `with ThreadPoolExecutor(...)`, whose
    __exit__ ran shutdown(wait=True): it BLOCKED until every long-running worker
    finished (~the full sleep budget below). The fix sets a shared stop flag and
    uses shutdown(wait=False, cancel_futures=True), so this returns promptly.
    """
    import graphql_pr_stream as g

    stop = threading.Event()
    started = threading.Event()
    per_iter = 0.01
    iters = 1000  # ~10s if a worker is allowed to run to completion

    def long_worker() -> str:
        started.wait(2.0)
        for _ in range(iters):
            if stop.is_set():
                raise g.StreamAborted("aborted by sibling")
            time.sleep(per_iter)
        return "ran-to-completion"  # pragma: no cover - must not happen

    def fatal_worker() -> str:
        started.wait(2.0)
        time.sleep(0.05)
        raise g.AllTokensExhausted(5.0)

    executor = cf.ThreadPoolExecutor(max_workers=4)
    futs = {
        executor.submit(long_worker): "a",
        executor.submit(long_worker): "b",
        executor.submit(long_worker): "c",
        executor.submit(fatal_worker): "fatal",
    }
    started.set()

    t0 = time.time()
    raised_loud = False
    try:
        for fut in cf.as_completed(futs):
            try:
                fut.result()
            except g.StreamAborted:
                continue
            except g.AllTokensExhausted:
                stop.set()
                raised_loud = True
                break
    finally:
        stop.set()
        executor.shutdown(wait=False, cancel_futures=True)
    elapsed = time.time() - t0

    assert raised_loud, "fatal worker error did not surface"
    # Full run-to-completion would be ~iters*per_iter (~10s); prompt abort is far
    # under that. (Old wait=True behavior would block ~10s and fail this bound.)
    assert elapsed < 5.0, f"abort hung for {elapsed:.2f}s"


# --------------------------------------------------------------------------- #
# pr_store concurrency hardening: busy_timeout set so concurrent first-connects #
# wait instead of erroring; WAL still enabled.                                  #
# --------------------------------------------------------------------------- #
def test_pr_store_sets_busy_timeout_and_wal(tmp_path):
    import pr_store

    conn = pr_store.connect(str(tmp_path / "prs.sqlite"), create=True)
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 60000
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_pr_store_concurrent_writers_do_not_error(tmp_path):
    """Many concurrent writer connections (the --workers path) must not raise."""
    import pr_store

    db = tmp_path / "prs.sqlite"
    pr_store.connect(str(db), create=True).close()  # create schema + WAL first
    errors: list[BaseException] = []

    def writer(n: int) -> None:
        try:
            conn = pr_store.connect(str(db), create=True)
            try:
                pr_store.upsert_record(
                    conn,
                    {
                        "repo": f"owner/repo{n}",
                        "pr_number": n,
                        "merge_commit_sha": f"sha{n}",
                        "pr_title": "t",
                        "pr_body": "b",
                        "comments": [],
                        "reviews": [],
                        "linked_issues": [],
                    },
                )
            finally:
                conn.close()
        except BaseException as exc:  # noqa: BLE001 - record for assertion
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not errors, f"concurrent writers errored: {errors[:3]}"
    check = pr_store.connect(str(db), create=True)
    try:
        n = check.execute("SELECT COUNT(*) FROM prs").fetchone()[0]
    finally:
        check.close()
    assert n == 12
