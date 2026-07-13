# AGENTS.md — cppmega.mlx agent contract

## RULE #1 (HIGHEST PRIORITY) — NO AUTOMATED FALLBACKS. FAIL FAST, FAIL LOUD.

Every op goes through ONE clear path. When the clear path does not work as expected, errors,
or misbehaves, **RAISE IMMEDIATELY** with WHERE + WHAT failed and crash — surface the bug, fix
the root cause. FORBIDDEN: silent/automated fallbacks (try/except→degraded path, except:pass,
eager-replaces-fused, MSL-instead-of-tvm-ffi, gated-off-by-default, clamp-instead-of-raise,
return zeros/garbage, best-effort, silent precision/shape downgrade). Fail-closed by RAISING a
clear error is correct; a silent fallback producing degraded/wrong/zero output is forbidden.
See CLAUDE.md. This overrides every rule below.

## Tensor memory rule

- Wrappers/adapters must not silently allocate or copy large tensors.
- If a design appears to require staging a large tensor, casting a large tensor,
  or copying a large tensor only to satisfy a wrapper boundary, treat that
  design as wrong and keep looking.
- Prefer passing references to existing GPU buffers, fusion, views/broadcasts,
  or IR-level lowering through the TileLang/TVM/tvm-ffi pipeline.
- If zero-copy/fused lowering is not currently possible, FAIL EXPLICITLY (raise with
  where + what). Do NOT silently fall back to another path (Rule #1).
- Dtype/shape mismatches must be solved at the right level:
  - first look higher in the graph and make the producer create the tensor in
    the required dtype/shape from the start;
  - if the current dtype/layout is the right system-level format (for example
    FP8 for the active config), move lower and teach the kernel to consume that
    format directly;
  - only use casts/reshapes as explicit graph decisions, not hidden adapter
    staging.

## Native subagents

- Use Codex native subagents for independent, bounded parallel work when it
  materially improves throughput.
- Every native subagent must use `gpt-5.6-sol` with `ultra` or `max`
  reasoning. Use `ultra` by default.
- Do not use another model or a lower reasoning level unless the user
  explicitly overrides this rule.
