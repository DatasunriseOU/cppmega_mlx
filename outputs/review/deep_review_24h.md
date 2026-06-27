# Deep Review — Last 24h Changes (C++ Foundation-Model Data Pipeline)

**Repo:** `/Volumes/external/sources/cppmega.mlx`
**Range reviewed:** `200afcb~1..HEAD` (2 commits)
- `200afcb` refactor: optimize git history precompute, permit empty build files, thread-safety to PR ingestion
- `4dce873` feat: streaming parquet repair, validation, sidecar management tools
**Scope:** 29 files, +4287/-187. Findings adversarially confirmed against source and the live data store.
**Review standard:** CLAUDE.md RULE #1 — one clear path per op; on failure RAISE with where+what. No silent fallbacks, clamps, zero/garbage returns, gated-off-by-default hard failures, or silent semantic/precision downgrades.

---

## 1. Executive Summary

**Overall health: NOT SAFE to run the corpus build or restart the conveyor as-is.** The new tooling is genuinely useful (streaming repair, sidecar audit gate, S3 upload/download, threaded PR ingest, faster git precompute), but it ships **two training-critical correctness defects** plus a fail-open verification gate that cannot catch them, and several concurrency hazards on exactly the parallel paths these commits newly enabled.

**Top risks (in priority order):**

1. **CRITICAL — Silent loss-target corruption.** `fix_packed_parquet_boundaries.py` rebuilds `loss_mask` as all-ones on every repaired row, destroying inter-document masking on multi-doc packed commit rows. The model is trained to predict document B's first token from document A's last token across unrelated commits, and `trained_token_count` is inflated. The repaired `doc_ids` are correctly preserved, so the right mask is derivable but discarded.
2. **HIGH — The "fail-closed" upload gate cannot detect that corruption.** `audit_sidecar_parquet.py` only length-checks `target_ids`/`loss_mask`/`doc_ids`; lengths are preserved by the repair, so corrupted shards pass clean and upload to Nebius S3. The gate also defaults fail-open and swallows mid-audit crashes.
3. **HIGH — Silent semantic change to training metadata.** Git-history precompute changed `file_local_commit_index` to count diff-filtered commits while the docstring claims "same semantics," producing gapped/inflated indices and corpora that are not byte-comparable across old/new code.
4. **HIGH — Concurrency defects on the newly-parallel paths.** Concurrent code+commits conveyor processes clobber the shared resume manifest (`_done.json`); the threaded PR stream's fail-fast hangs instead of aborting; per-worker token pools double-spend the same PAT rate limit; the destructive row-dropper exits 0 on per-file failure by default.
5. **CRITICAL (PRE-EXISTING, out of diff range) — std cross-link is 0% functional.** The A2 store's `std` index is 100% GCC libiberty/compiler-internal noise (0 actual `std::` symbols). See §3.

**Test posture is misleading:** the boundary-fixer test suite uses a single-doc fixture where the buggy all-ones rebuild coincidentally equals the correct mask, so the corruption ships green and the wrong invariant is enshrined.

---

## 2. Findings by Severity

> Two pairs of confirmed findings describe the same root defect from two angles and are merged below: the loss_mask rebuild (C1) and the git-precompute semantic change (H3).

### CRITICAL

#### C1 — Boundary repair rebuilds `loss_mask` as all-ones, destroying inter-document loss masking
- **File:** `scripts/fix_packed_parquet_boundaries.py:286-288`
- **What:** `repair_row()` unconditionally sets
  `repaired["loss_mask"] = [1]*max(new_valid-1,0) + [0]*(capacity-...)` and
  `repaired["trained_token_count"] = max(new_valid-1,0)`.
  The producer `pack_enriched_rows.py:909-920` (`_loss_mask_for_packed_docs`, trained at 1025-1027) sets `loss_mask[pos]=1` **iff `doc_ids[pos]==doc_ids[pos+1]`** (0 at every inter-doc boundary and last token), with `trained_token_count = sum(loss_mask) = valid - num_docs`.
- **Why it matters:** The default target `outputs/reindexed_commits` packs multiple whole commit docs per fixed-width row (route-by-fit + greedy concat; `num_docs`, `source_doc_token_lengths` confirm multi-doc rows). Live spot-check: sampled multi-doc rows in `outputs/reindexed_commits/1024` have interior `loss_mask==0` exactly at the `doc_id` boundary with `trained == valid - num_docs`. On any repaired multi-doc row, the repair flips every interior boundary 0→1 (cross-document leakage into the loss) and over-reports `trained_token_count` by `num_docs-1`. The consumer `parquet_dataset.py:291-356` feeds the stored `loss_mask` straight into the loss (`dense_cpp_lm.py:644`) — no recompute downstream. This is a silent, training-critical data corruption (RULE #1: silent semantic downgrade producing wrong output instead of failing loud). `doc_ids` is first in `TOKEN_ALIGNED_COLUMNS` and is correctly shifted (`:291-294`), so the correct mask IS recoverable but ignored.
- **Fix:** Recompute `loss_mask` from the **repaired** `doc_ids` using the producer's exact rule (`1 iff doc_ids[pos]==doc_ids[pos+1]` within `new_valid`, last token 0, pad 0), then set `trained_token_count = sum(loss_mask)`. Import/reuse `_loss_mask_for_packed_docs` from `pack_enriched_rows.py` so the two paths cannot drift. Must run after the `TOKEN_ALIGNED_COLUMNS` loop that produces repaired `doc_ids`.
- **Status:** Real, high confidence, deterministically reproduced. **Blocks the corpus run.**

#### C2 — A2 std cross-link store is 100% GCC noise (0 achievable `std::` hit rate)  *(PRE-EXISTING, OUTSIDE DIFF RANGE)*
- **File:** `scripts/crossrepo/build_global_symbol_index.py:141-161` (declared `namespace_prefixes` at 95-108)
- **What:** Each BASE_LIB declares `namespace_prefixes` (e.g. `std`→`["std::"]`), but `is_public_symbol(qname, rel_file, public_only)` never receives or applies it. Verified against live `outputs/crossrepo/global_symbols.sqlite`: `base_lib='std'` = **1711 rows, 0 with a `std::` qname, 0 inline-namespace rows**. All 1711 are GCC compiler internals (`gcc-mirror/libiberty/...`, `cp-demangle.h`, `pex-common.h`, etc.).
- **Why it matters:** The consumer (`index_project.py:370`, `CROSSLINKABLE_NS_PREFIXES=('std::','boost::')`) cross-links only `std::`/`boost::` callees. With 0 stored `std::` qnames, every std lookup misses — the std cross-link feature (the most common base lib in C++ training data) is entirely non-functional, and the store is polluted with mislabeled GCC internals.
- **Fix:** See §3 (A2 verdict). Requires namespace enforcement **plus** correct subtree selection **plus** inline-namespace normalization (H4) — not "looser gating," which would make it worse.
- **Status:** Real, high confidence, reproduced against live store. **NOTE:** This file was last touched in `ea5d3c2`, **outside** the reviewed `200afcb~1..HEAD` range, and the cross-link feature is committed default-off. Route as a pre-existing issue, not a regression of these two commits.

### HIGH

#### H1 — Destructive row-dropper swallows per-file exceptions and exits 0 by default
- **File:** `scripts/drop_invalid_packed_parquet_rows.py:311-317, 406-408`
- **What:** `_process_file` wraps the whole read/validate/in-place-rewrite in `except Exception as exc -> FileResult(error=...)`; `main()` returns 2 only if the opt-in `--fail-on-remaining` flag is set, else `return 0`.
- **Why it matters:** Destructive cleaner for training-critical packed parquet. A read/schema/write failure becomes a non-fatal report entry and the process exits 0 by default; an automated pipeline keyed on exit code treats the corpus as cleaned and proceeds to upload/train with invalid rows still present. RULE #1-forbidden broad-except→continue + gated-off-by-default hard failure. The sibling tool added in the same commit (`fix_packed_parquet_boundaries.py`) does the opposite (bare `future.result()`, `raise SystemExit` on remaining), proving the swallow is an inconsistency, not house style.
- **Fix:** Let unexpected read/schema/write errors propagate (crash loud with where+what). At minimum, make the default exit non-zero whenever `total["error_files"] > 0` without requiring `--fail-on-remaining`. Keep only the existing narrow non-numeric-bucket guard (274-276).

#### H2 — Upload gate validates only LENGTHS of `target_ids`/`loss_mask`/`doc_ids`, not values — cannot catch C1
- **File:** `scripts/audit_sidecar_parquet.py:436-444`
- **What:** The loop over `("target_ids","loss_mask","doc_ids")` calls only `_add_numeric_list_stats(..., expected_lengths=input_lengths, ...)`, which flags rows only on `lengths != expected_lengths`. No check that `target_ids == input_ids` shifted by one; no `loss_mask` consistency check.
- **Why it matters:** The C1 repair preserves every array LENGTH (pads to capacity), so the corrupted all-ones `loss_mask` passes this audit clean (`bad_rows` stays 0) and the shard uploads to Nebius S3. The audit DOES value-check other columns (`input_ids` vocab range 423-434, chunk bounds 481-503, graph edges 319-343), so omitting the core LM loss-target tensors is an asymmetric gap, not by-design. Note `trained_token_count` is bounds-checked (418: `<0` or `>input_length`) but never cross-checked against `sum(loss_mask)`.
- **Fix:** Add value-level validation: assert `target_ids[:valid-1] == input_ids[1:valid]`; assert `loss_mask` is 0 for `pos >= valid`; assert `sum(loss_mask) == trained_token_count`. Confirm the canonical `loss_mask` semantics against the packer before adopting the doc-boundary rule verbatim (some training paths use blanket all-ones). Mark mismatches as `bad_rows` so `--fail-on-bad` blocks the upload.

#### H3 — Git-history precompute changed `file_local_commit_index` semantics despite "Keep the same semantics" claim
- **File:** `scripts/nanochat_data/extract_git_history.py:360-372, 375-402, 549-601`
- **What:** New `precompute_cpp_file_changes` counts indices from `get_commit_cpp_files()` (name-status/extension filters only: `is_cpp_file`/`should_skip_path`/`status=='M'`/`MAX_FILES_PER_COMMIT`). The OLD `compute_file_local_commit_indices` (200afcb~1:284-300) counted via `get_commit_diffs()`, which ALSO drops files on content filters: `diff_len < MIN_DIFF_CHARS(50)` / `> MAX_DIFF_CHARS(50000)`, `len(content) > 200000`, and None blobs. The emit loop (549-555) still applies the full content filter, so the same record SET gets different index VALUES.
- **Why it matters:** `file_local_commit_index` is written into every emitted record (599-601) and packed into sidecar column `source_file_local_commit_indices` (`pack_enriched_rows.py:938/964/1053`). Dropping the content filters from the counting path makes per-file indices systematically higher and gapped (e.g. `0,2,3` instead of `0,1,2`) for files with an interspersed sub-50/over-50000-char or >200KB-content commit. Silent semantic change to training metadata under a docstring asserting identical behavior; corpora built with old vs new code are inconsistent. `test_extract_git_history_precompute.py` only exercises a trivial single small-diff commit, so the divergence is untested.
- **Fix:** Count the index over the actual post-diff-filter set `get_commit_diffs` emits (run the same None/length/content-size acceptance check in precompute before incrementing counters), OR update the docstring/field contract and re-derive the whole corpus consistently. Add a test with an oversized/sub-threshold-diff commit asserting the documented index.

#### H4 — `_RESERVED_LEADING` regex drops EVERY libc++ (`__1`) and all libstdc++ `std::string` (`__cxx11`) symbol  *(PRE-EXISTING, OUTSIDE DIFF RANGE)*
- **File:** `scripts/crossrepo/build_global_symbol_index.py:138, 159-160`
- **What:** `_RESERVED_LEADING = re.compile(r"(^|::)(_[A-Z]|__)")` rejects any qname containing `::__`. libc++ wraps its whole API in `inline namespace std::__1::...`; libstdc++ wraps `std::string` in `std::__cxx11::...`. Both match and are dropped. Live store: 0 rows with `__1::`/`__cxx11`.
- **Why it matters:** Inline namespaces are transparent in real C++ (`std::__1::vector` IS `std::vector`), and the consumer emits these spellings as `baselib_callees`. Even with subtrees fixed, 100% of libc++ and all `std::string` defs are gated out. Compounds C2.
- **Fix:** Normalize transparent inline namespaces at the shared chokepoint `index_project.get_qualified_name` (skip cursors where `Cursor.is_inline_namespace()`, with literal fallback for `__1`/`__cxx11`) so qnames collapse to `std::vector`/`std::basic_string` before the reserved-name and prefix checks — fixing both producer and consumer keys at once. **NOTE:** Pre-existing, outside the reviewed range (last touched `ea5d3c2`).

#### H5 — Concurrent code+commits conveyor processes clobber the shared resume manifest
- **File:** `scripts/streaming_conveyor.py:761` (`Manifest.load`), `287-290` (`stream_lock_names`), `691-704` (per-stream RunLock); `save()` at `scripts/streaming_reindex.py:163-181`
- **What:** RunLock is intentionally per-stream (`stream_lock_names` returns `(streams,)`), so a `--streams code` process and a `--streams commits` process run concurrently. Both load and full-rewrite the single shared `CONVEYOR_MANIFEST = CONVEYOR_ROOT/"_done.json"`. `Manifest.save()` rewrites the whole file from in-memory dict (no disk re-read/merge, no flock); the conveyor's `manifest_lock` is a `threading.Lock`, useless across OS processes. Key spaces are disjoint (`{repo}::code` vs `{repo}::r{start}`), so each process's rewrite drops the other's keys — last-writer-wins.
- **Why it matters:** On crash/restart, one stream's resume entries are lost and it re-processes everything; run-summary token accounting is wrong. This silently defeats the whole point of the per-stream lock. (Scope: progress/accounting only — parquet outputs are per-stream/per-length files and are not corrupted.)
- **Fix:** Per-stream manifest file (`_done.{stream}.json`), or make `save()` re-read on-disk JSON, merge the changed key, and atomically replace under a cross-process `flock`. Per-stream file is simplest and matches the lock model.

#### H6 — Manifest `verified_valid_tokens` overcounts under `--code-commits-only`
- **File:** `scripts/upload_verified_sidecar_to_nebius_s3.py:108-117, 251-264`
- **What:** `_load_verified_token_total` sums `data["total"]["valid_tokens"]` from the single hardcoded all-valid receipt (code+commits+PR) and is invariant to the profile flag. Under `--code-commits-only`, `selections` drops `parquet/pr/*` but the same all-valid total is written verbatim into the manifest alongside `profile="code_commits_only"` and `standalone_pr_included=false`.
- **Why it matters:** From the live receipt: `total.valid_tokens=2,390,823,893` (code 269,131,691 + commits 719,814,003 + pr 1,401,878,199). Under `--code-commits-only` the manifest reports the full 2.39B but uploads only 988,945,694 tokens — overstating by the entire 1.40B PR count (58.6%) in a training-critical accounting artifact while simultaneously asserting PR is excluded. RULE #1 silent integrity downgrade.
- **Fix:** Compute `verified_valid_tokens` for exactly the uploaded buckets (sum the receipt's `by_kind_bucket` for the selected set), or assert receipt coverage matches `selections` and RAISE on mismatch. The receipt already carries `by_kind`/`by_kind_bucket`, so this is feasible with data in hand.

### MEDIUM

#### M1 — "Fail-closed gate" defaults fail-open and swallows mid-audit crashes
- **File:** `scripts/audit_sidecar_parquet.py:730-732, 556-559`
- **What:** Docstring: "the fail-closed gate before uploading parquet shards." But `--fail-on-bad` is `store_true` (default off), so `main()` returns 0 by default even when `bad_files`/`bad_rows>0`. `_audit_file` catches `Exception`, increments `bad_files`, records the traceback, and returns partial stats instead of raising.
- **Why it matters:** A gate whose stated purpose is to block bad uploads passes by default; an operator/CI step is told everything passed. (Partial mitigation: the crash DOES bump `bad_files` into the receipt, which the upload script's `_load_verified_token_total` would catch on its one hardcoded path — limits blast radius, hence medium.)
- **Fix:** Make fail-closed the default (non-zero whenever `bad_files`/`bad_rows>0`; add an explicit `--allow-bad` escape hatch instead). Let audit-logic exceptions propagate (or re-raise after recording) — a crashed audit cannot certify a shard.

#### M2 — Boundary-fixer tests use only a single-document fixture, so C1 ships green
- **File:** `tests/test_fix_packed_parquet_boundaries.py:21-44`
- **What:** Only fixture row is single-doc (`doc_ids=[7]*valid`, `source_doc_ids=[7]`), and tests assert `trained_token_count==valid-1` plus an all-ones mask.
- **Why it matters:** For single-doc rows the buggy all-ones rebuild coincidentally equals the correct mask, so all tests pass while the multi-doc path (the common packed-commit case) is corrupted. The suite encodes the wrong invariant and gives false confidence.
- **Fix:** Add a `≥2` distinct-`doc_ids` fixture (e.g. `[7]*a + [9]*b`) containing a collapsed marker; assert `loss_mask==0` at every inter-doc boundary and `trained_token_count == sum(loss_mask) == valid - num_docs`. This fails against current code and pins C1. Relax the single-doc assertion to `trained == sum(loss_mask)`.

#### M3 — Marker heuristic mutates real source comments that merely contain DIFF/CONTEXT/PRE-COMMIT/POST-COMMIT
- **File:** `scripts/fix_packed_parquet_boundaries.py:132-135`
- **What:** `_generated_marker_end` accepts ANY `// === ... ===` whose decoded text substring-matches `MARKER_NAMES=("PRE-COMMIT","POST-COMMIT","CONTEXT","DIFF")`, then injects `<NL>` (id 47) and shifts all sidecars.
- **Why it matters:** "CONTEXT"/"DIFF" are ordinary words; reproduced over-match: `// === DIFF ALGORITHM ===` / `// === CONTEXT SWITCH ===` get misclassified and have newline tokens silently injected into real training text. Repairs on ambiguous match instead of failing loud; contradicts the file's own "intentionally narrow" docstring (17-20). Blast radius bounded to rows already touched by the newline-drop bug.
- **Fix:** Anchor to exact generated forms: require the decoded marker to START with `// === PRE-COMMIT ===`, `// === CONTEXT ===`, `// === DIFF ===`, or prefix `// === POST-COMMIT:` (the POST-COMMIT form carries a variable subject, see `tools/clang_indexer/process_commits.py:1283`, which is why substring was used). On any ambiguous `// === ... ===`, leave the row untouched (or RAISE).

#### M4 — `MAX_PENDING_BEFORE_COMMIT` raised 32→1000 widens the cross-process near-dedup leak window ~30x
- **File:** `tools/clang_indexer/dedup_store.py:141-143, 472-476`
- **What:** `MAX_PENDING_BEFORE_COMMIT` changed from literal 32 to env-default 1000; `_mark_pending` commits at `min(commit_every, MAX_PENDING_BEFORE_COMMIT)`. Exact/near inserts (minhash+lsh) buffer until this threshold; only `chunk_claims` commit immediately.
- **Why it matters:** One shared dedup DB is threaded into both code and commit subprocesses (and `--repo-workers>1`, new here). SQLite WAL hides a writer's uncommitted near-dup reference docs from other connections, so up to ~1000 (was ~32) near-dup decisions per writer are invisible cross-process — two concurrent stages/repos can both accept the same near-duplicate. `chunk_claims` (exact sha1) backstops exact dups but NOT near-dups. Widens a pre-existing leak rather than introducing one.
- **Fix:** Keep the buffer modest for the shared cross-process DB (restore ~32-128, or scale down when concurrency is enabled / flush more aggressively with multiple writers). At minimum document the cross-process near-dedup trade-off rather than defaulting 1000.

#### M5 — Threaded PR stream defeats fail-fast on token exhaustion (shutdown(wait=True) hangs)
- **File:** `scripts/pr_ingest/graphql_pr_stream.py:669-686`
- **What:** The threaded branch raises `SystemExit` (on `AllTokensExhausted`) INSIDE `with ThreadPoolExecutor(...) as executor:`, so `__exit__` runs `shutdown(wait=True)` with default `cancel_futures=False`.
- **Why it matters:** All repos are submitted upfront; `wait=True` + no cancel blocks until every running AND queued repo finishes streaming before the process exits. "FAILING LOUD per RULE #1" instead hangs for the remainder of in-flight repos. Worse: during the blocking shutdown the main thread stops iterating `as_completed`, so `AllTokensExhausted`/`SystemExit` raised by other workers land in never-`.result()`-ed futures and are silently swallowed (a second RULE #1 violation). The single-threaded branch aborts immediately, confirming the divergence.
- **Fix:** A shared `threading.Event` stop flag that `stream_repo` checks between pages and re-raises on, combined with `shutdown(wait=False, cancel_futures=True)`, then exit. (Note: `cancel_futures=True` alone only cancels queued futures, not the `workers-1` already-running streams.)

#### M6 — Per-worker `TokenPool` cooldowns not shared → concurrent workers double-spend one PAT's rate limit
- **File:** `scripts/pr_ingest/graphql_pr_stream.py:199-204` (`_make_worker_pool`), `263-264`/`294-295` (`cool`/`advance`)
- **What:** `_make_worker_pool` builds a fresh `TokenPool(list(tokens))` per call (once per repo task, line 616); each pool owns a private `cooldown_until`. A 401/rate-limit cooldown mutates only the local pool and is invisible to other concurrently-running workers.
- **Why it matters:** GitHub limits are per-token globally. Independent per-pool rotation + per-repo pool recreation (which also discards rotation/cooldown history across repos) means two workers can hammer the same PAT (secondary-limit/401 storms) and one worker's 24h cooldown is invisible to others → divergent, premature `AllTokensExhausted`, which (via M5) aborts the whole run while tokens are still usable.
- **Fix:** Construct ONE `TokenPool` in `main()` and share it across all workers, guarding `current()/advance()/cool()` with a lock so cooldowns and rotation reflect real per-token global limits.

#### M7 — Audit gate decoupled from actual upload selection (green receipt for a different/partial run passes)
- **File:** `scripts/upload_verified_sidecar_to_nebius_s3.py:104-117, 251-254`
- **What:** `_audit_receipts()` returns a fixed hardcoded path; `_load_verified_token_total` checks only aggregate `total.bad_files`/`bad_rows`. Nothing cross-checks that the receipt's `by_kind_bucket` coverage equals the buckets in `selections`/`sources`.
- **Why it matters:** A stale or narrowly-scoped green receipt (e.g. audited only `--buckets 8192` or only code) passes the gate while ALL buckets upload. Defeats the "verified" guarantee the docstring promises.
- **Fix:** After loading the receipt, require every `(kind,bucket)` in `sources` to appear in `by_kind_bucket` with `bad_files==bad_rows==0`, and RAISE listing any selection bucket not covered.

#### M8 — Download never validates downloaded data against the audit receipt
- **File:** `scripts/download_verified_sidecar_from_nebius_s3.py:200-248`
- **What:** `main()` syncs every selection and prints "downloaded ..."; the receipt is downloaded as just another selection but never read back, and `verified_valid_tokens` is written only by the uploader and read NOWHERE. With `--use-default-manifest` the uploaded manifest is bypassed entirely.
- **Why it matters:** The consume side proves nothing — a missing/incomplete remote prefix syncs successfully, returns 0, and is trusted as the "verified" set. (`aws s3 sync` ETag-checks individual objects, so byte-truncation of a single file is mitigated; the uncaught risk is a missing-shards SET, compounded by no `--delete`.) The unused `verified_valid_tokens` field shows reconciliation was intended but left incomplete.
- **Fix:** After download, RAISE unless the downloaded receipt's `bad_files==bad_rows==0` AND the on-disk token total reconciles with `manifest["verified_valid_tokens"]` (strongest: re-run `audit_sidecar_parquet.py` on the downloaded shards).

#### M9 — Only S3 test is shallow (no parity, creds gate, audit gate, or URI construction)
- **File:** `tests/test_verified_sidecar_manifest_selection.py:26-53`
- **What:** All three tests reduce to set-membership against hardcoded remote keys. No `assert _remotes_from_upload()==_remotes_from_download()` (the round-trip invariant), and nothing exercises `_s3_env()` (creds RAISE / NEBIUS→AWS mapping), `_load_verified_token_total()` (green vs non-green), `_existing_sources()` (missing-input RAISE), or `main(['--dry-run'])` (S3 URI template).
- **Why it matters:** The security-relevant (no-creds RAISE) and integrity-relevant (green-receipt gate, S3 URI) logic of both tools is untested; a typo in the target URI or a regression in the gate ships green. The one invariant a selection test should protect (upload==download) is unprotected.
- **Fix:** Add `_remotes_from_upload(cc)==_remotes_from_download(cc)` for both modes; tests that `_s3_env` raises `SystemExit` with no creds and maps `NEBIUS_*`→`AWS_*`; `_load_verified_token_total` green/non-green behavior; and `main(['--dry-run'])` asserting printed `aws s3 sync` commands and target URIs.

### LOW

#### L1 — Over-length docs dropped from training with only an ephemeral receipt
- **File:** `scripts/streaming_reindex_commits.py:366-388`
- **What:** `route_by_fit` drops docs longer than the largest bucket and writes `dropped_overlong.json` to `out_dir = rwork/"routed"`, which `process_range`'s finally unconditionally `shutil.rmtree`s (line 454, no `--keep-temp` guard). The dropped count is also absent from the returned info/manifest/summary; only the stderr line survives.
- **Why it matters:** The drop itself is a loud, defensible improvement (not a RULE #1 fallback), but corpus-scale drop volume is not durably auditable.
- **Fix:** Aggregate dropped-overlong counts/paths into a persistent run-level report and surface the total in the final summary.

#### L2 — Per-repo extraction failure printed and skipped; run continues, exits 0
- **File:** `scripts/nanochat_data/extract_git_history.py:670-682`
- **What:** `process_repo(...)` is wrapped in `except Exception as e: print(...)`; the loop continues, the JSONL finalizes, and the process exits 0. (Pre-existing pattern, but re-indented into the new `OutputFileLock` try/finally by this diff.)
- **Why it matters:** A partially-extracted corpus (some repos silently missing) looks like a clean run.
- **Fix:** Collect failed repos, exit non-zero (or re-raise), and record failures in a durable manifest rather than stdout-only.

#### L3 — Identical `>=pos` shift on `token_chunk_ends` absorbs an inserted newline into the preceding chunk
- **File:** `scripts/fix_packed_parquet_boundaries.py:220-225`
- **What:** `_shift_offset` shifts when `out >= pos` for both starts and ends; ends are EXCLUSIVE (confirmed `test_token_coordinate_parquet.py:64`, producer `tokenized_enriched.py:350-363`). An exclusive end equal to an insertion position is over-shifted by +1, extending the prior chunk to cover the inserted `<NL>`. Same issue in `changed_chunk_spans["end"]` via `_shift_span`.
- **Why it matters:** Misattributes one whitespace token per coincident insertion; marker boundaries (where insertions occur) are exactly where chunk edges fall, so it can be systematic. Bounded to one token, hence low.
- **Fix:** Shift inclusive starts on `out >= pos` but exclusive ends on `out > pos`; add a multi-chunk test asserting the inserted `<NL>` falls outside both adjacent spans.

#### L4 — `_length_totals` replaces fail-loud key access with silent `.get(key, 0)`
- **File:** `scripts/streaming_conveyor.py:274-278` (used 407, 515)
- **What:** New helper does `totals[key] += int(st.get(key, 0))`; the old path used `st["valid_tokens"]` (KeyError on malformed stat). The same dicts are read fail-loud elsewhere in-file (run report 960-963).
- **Why it matters:** A malformed stat dict silently undercounts cumulative valid tokens; `cumulative["valid"]` drives `--token-budget` stop logic and dedup WAL checkpoint thresholds, so a silent undercount overshoots the budget / skips checkpoints. Control-flow only (not emitted data), hence low. RULE #1 clamp-instead-of-raise.
- **Fix:** Use `st[key]` and raise identifying the offending length bucket.

#### L5 — `cumulative['valid']` read without `manifest_lock` for token-budget checks
- **File:** `scripts/streaming_conveyor.py:877, 911`
- **What:** Writers update under `manifest_lock`; the budget gate reads unlocked at 877 (single-thread path) and 911 (repo_pool drain path). Only 911 is a genuine concurrent unlocked read (877 is sequenced-after a joined pool, no concurrent writer).
- **Why it matters:** GIL-atomic int read, at worst slightly stale → minor budget overshoot. Defensive/future-proofing only; not a RULE #1 violation.
- **Fix:** Snapshot under the lock before comparing.

#### L6 — Unused `import sys` in both S3 scripts
- **File:** `scripts/download_verified_sidecar_from_nebius_s3.py:22`, `scripts/upload_verified_sidecar_to_nebius_s3.py:23`
- **What:** Dead import (both end with `raise SystemExit(main())`).
- **Why it matters:** Harmless, but signals the modules were not lint-checked — the same lapse that left the larger gate/verification gaps unnoticed.
- **Fix:** Remove it from both.

---

## 3. Cross-Link A2 Under-Index Verdict — RE-RUN NEEDED (but only AFTER three root-cause fixes)

**Verdict: YES, a re-run is required, but re-running A2 as-is would reproduce a 0% std hit rate.** The under-index is NOT a gating-looseness problem — the task's "looser gating" hypothesis is wrong and would make it worse by admitting more non-std noise.

**Evidence (live `outputs/crossrepo/global_symbols.sqlite`):** `base_lib='std'` = **1711 rows, 0 with `std::` qname, 0 inline-namespace rows** — 100% GCC libiberty/compiler internals mislabeled as `std`.

**Three independent root causes must all be fixed before re-running:**
1. **Pollution (C2):** `namespace_prefixes` is declared per base-lib but never enforced in `is_public_symbol`, so non-`std::` GCC symbols are admitted under `base_lib='std'`.
2. **Inline-namespace gate (H4):** `_RESERVED_LEADING` drops 100% of libc++ (`__1`) and all `std::string` (`__cxx11`) — so even correctly-extracted std symbols are rejected.
3. **Subtree selection:** libstdc++/libc++ public headers were never indexed; the file cap surfaced gcc's compiler tree instead. (This, not gating, is the primary reason *zero* real `std::` symbols exist.)

**Sequence:** (a) normalize inline namespaces at `index_project.get_qualified_name` so producer and consumer keys match; (b) thread + enforce `namespace_prefixes` (reject qnames not starting with a declared prefix); (c) fix subtree selection so real libstdc++ public headers are indexed; (d) THEN re-run A2 and re-validate `SELECT COUNT(*) WHERE base_lib='std' AND qname LIKE 'std::%'` is non-zero and dominated by real STL symbols.

**SCOPE CAVEAT:** `build_global_symbol_index.py` is **outside** the reviewed `200afcb~1..HEAD` range (last touched `ea5d3c2`) and the cross-link feature is committed **default-off**, so this is a pre-existing defect, not a regression of these two commits. It does not block the corpus run unless the std/boost cross-link signal is enabled — but the std signal it would produce today is entirely fabricated noise, so do not enable cross-link until A2 is re-run green.

---

## 4. Prioritized Suggestions

### MUST FIX before the corpus run
1. **C1** — Recompute `loss_mask` from repaired `doc_ids` (reuse `_loss_mask_for_packed_docs`) and set `trained_token_count = sum(loss_mask)`. Re-repair any already-repaired shards. *(Single most important fix; silently corrupts the loss target.)*
2. **M2** — Add the multi-doc regression test (pins C1, fails against current code). Do not trust green until this exists.
3. **H2** — Add value-level audit of `target_ids`/`loss_mask`/`sum(loss_mask)==trained_token_count`; make those mismatches `bad_rows`. *(This is what should have caught C1 at the gate.)*
4. **H3** — Decide the `file_local_commit_index` contract: either restore the post-diff-filter counting (true "same semantics") or update the docstring AND re-derive the whole commit corpus consistently. Do not mix old/new-code outputs in one corpus.
5. **M3** — Anchor marker detection to exact generated prefixes before re-running the boundary fixer over real data.

### MUST FIX before the conveyor restart
6. **H5** — Per-stream manifest file (or flock+merge `save()`) so concurrent code+commits runs don't clobber resume/accounting.
7. **H1** — Make the row-dropper exit non-zero on `error_files>0` by default; let read/write errors propagate.
8. **M1** — Make `audit_sidecar_parquet.py` fail-closed by default (`--allow-bad` escape hatch); let audit exceptions propagate.
9. **M5 + M6** — Share one locked `TokenPool` across PR workers; replace `with-executor` fail-fast with a shared stop-Event + `cancel_futures=True` so token exhaustion aborts promptly instead of hanging/swallowing.
10. **M4** — Cap `MAX_PENDING_BEFORE_COMMIT` for the shared cross-process dedup DB (or scale it down under concurrency); at minimum document the near-dedup trade-off.

### FIX before relying on S3 publish/consume integrity
11. **H6** — Compute `verified_valid_tokens` for exactly the uploaded buckets (or RAISE on receipt/selection mismatch).
12. **M7** — Cross-check receipt `by_kind_bucket` coverage against the upload selection; RAISE on uncovered buckets.
13. **M8** — Validate downloaded shards against the receipt (re-run audit or reconcile token totals) and RAISE on divergence.
14. **M9** — Add the upload==download parity, creds-gate, receipt-gate, and `--dry-run` URI tests.

### FIX before relying on std/boost cross-link signal (NOT blocking the current corpus run; pre-existing, out of range)
15. **C2 + H4 + subtree selection** — Implement all three root-cause fixes (§3), then re-run A2 and re-validate. Do NOT enable cross-link until the std index contains real `std::` symbols.

### Cleanup / hardening (low urgency)
16. **L1** — Persist dropped-overlong tallies into the run summary.
17. **L2** — Track failed repos in extraction; exit non-zero + durable failure manifest.
18. **L3** — Exclusive-end shift (`out > pos`) for `token_chunk_ends`/spans + test.
19. **L4** — Restore fail-loud key access in `_length_totals`.
20. **L5** — Snapshot `cumulative['valid']` under the lock at line 911.
21. **L6** — Remove dead `import sys`; add the S3 scripts to lint.

---

*Confirmed findings re-verified against source (`200afcb~1..HEAD` diff + full files) and the live `outputs/crossrepo/global_symbols.sqlite` and `outputs/reindexed_commits` data. Two finding pairs (loss_mask rebuild; git-precompute semantics) describe one defect each and were merged.*
