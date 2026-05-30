# CLAUDE.md — cppmega.mlx

## RULE #1 (HIGHEST PRIORITY — overrides everything else) — NO AUTOMATED FALLBACKS. FAIL FAST, FAIL LOUD.

Every operation goes through ONE clear path. **When the clear path does not work as expected,
errors, or misbehaves, the code MUST RAISE IMMEDIATELY** with a precise message saying **WHERE
and WHAT** failed, and crash — so the bug surfaces and we fix the ROOT CAUSE.

**FORBIDDEN** — automated/silent fallbacks that hide a failure:
- `try/except` → degraded/alternate path; `except: pass`; "best-effort"; silent slow-path.
- eager-replaces-fused, MSL-adapter-instead-of-tvm-ffi, gated-off-by-default, watchdog→fallback.
- clamp-instead-of-raise; return zeros/garbage; silent shape/dtype/precision downgrade.
- any code path that papers over a broken clear path instead of surfacing it.

**REQUIRED on failure:** `raise` with where + what → then we debug and fix the bug.
Fail-closed by **raising with a clear error IS correct** ("падать с сообщением"). A SILENT
fallback that produces a degraded/wrong/zero result is what is forbidden.

This is the first and main rule. Existing automated fallbacks must be found and removed; route
everything through the single clear path that fails loud.

## Repos, branches, upstream provenance (David, 2026-05-30)

All our repos live in the **DatasunriseOU** org and we work on the **`main`** branch EVERYWHERE
(local + gb10 + every machine). Do NOT create side/feature branches for our own work — commit to
`main`, push `main`, pull `main`. (Past mistake: scattering fixes across `m4-metal4-keystone` /
`merge/upstream-codegen-reorg` side branches — never do this; consolidate on `main`.)

Our repos (all in DatasunriseOU): `cppmega_mlx`, `tilelang`, `tvm`, `tvm-ffi`, `mlx`, `mlx-lm`,
`cppmega`, `triton-pr9701`. These are OUR working repos (not "forks we don't control") — we commit
to them directly.

**Upstream provenance + how we pull fixes (we cherry-pick needed fixes FROM upstream INTO our main):**
- **tvm** = `apache/tvm` upstream (git remote `origin` = https://github.com/apache/tvm). Our fork =
  `DatasunriseOU/tvm` (remote `datasunrise`). We take tvm from **apache**, NOT from the TileLang/tvm
  fork (`tilelang` remote = TileLang/tvm exists only to cherry-pick specific fixes).
- **tvm-ffi** = `apache/tvm-ffi` upstream (Apache-2.0). Our fork = `DatasunriseOU/tvm-ffi`, vendored
  as the `3rdparty/tvm-ffi` submodule inside our tvm. From **apache**, not tilelang.
- **tilelang** = upstream `TileLang/tilelang`; our fork `DatasunriseOU/tilelang`, work on `main`.
- **mlx / mlx-lm** = upstream `ml-explore/*`; our forks `DatasunriseOU/mlx`, `DatasunriseOU/mlx-lm`.
- The pattern: keep an upstream remote (`origin`=apache for tvm, etc.), pull/cherry-pick the fixes
  WE need from upstream into OUR `main`, and build from our `main`.

## Tensor memory rule (see AGENTS.md)
- Wrappers/adapters must not silently allocate, cast, or copy large tensors to satisfy a
  boundary. If a design seems to require it, the design is wrong — keep looking (this is a
  corollary of Rule #1: no silent papering-over).
