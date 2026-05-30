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

## Tensor memory rule (see AGENTS.md)
- Wrappers/adapters must not silently allocate, cast, or copy large tensors to satisfy a
  boundary. If a design seems to require it, the design is wrong — keep looking (this is a
  corollary of Rule #1: no silent papering-over).
