#!/usr/bin/env python3
"""Build per-expert SFT datasets from the agent-trajectory parquet.

Reads the flat transition parquet produced by
:mod:`cppmega_mlx.data.agent_trajectory` (columns: session_id, source, repo,
step_idx, obs_text, action_kind, action_payload, result_text, exit_code,
is_build, is_test, reward, edit_diff) and emits THREE JSONL SFT datasets:

* ``tool_router.jsonl`` -- (obs/state + available typed-action schemas) -> the
  chosen :class:`ToolCall` JSON. EVERY target is round-tripped through
  :meth:`ToolRouter.parse` (schema + typed-action validation); invalid targets
  are DROPPED (never emitted). Includes null-route ``stop`` negatives.

* ``buildops.jsonl`` -- from real-exit-code build transitions ONLY ->
  (build system + failing log span) -> {cause, file:line, fix}. The ``fix`` is
  the next successful edit/diff that preceded a passing build in the SAME session
  where one exists; otherwise the label is cause-classification only (fix=null).
  NO fabricated fixes, NO fabricated rewards (we use only exit_code-bearing rows).

* ``sql.jsonl`` -- embedded SQL extracted from C/C++ obs/edit text (R"(...)" raw
  strings or quoted literals containing SELECT/INSERT/UPDATE/DELETE/CREATE) ->
  (C++ context + SQL + schema-if-any) -> {valid, repaired_sql}. Each candidate is
  validated with the sqlite verifier; unvalidatable candidates are DROPPED.

Per RULE #1 this script FAILS LOUD: a parquet missing required columns, an
unparseable action payload that we cannot route, or a verifier that is
unavailable when SQL rows exist -> RAISE. We never write an unvalidatable target.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from cppmega_mlx.data.agent_trajectory import PARQUET_COLUMNS
from cppmega_mlx.experts.buildops import BUILDOPS_SYSTEM
from cppmega_mlx.experts.sql_expert import SQL_SYSTEM
from cppmega_mlx.experts.tool_router import TOOL_ROUTER_SYSTEM, _schema_hint
from cppmega_mlx.inference.tool_router_schema import ToolRouter
from cppmega_mlx.inference.typed_actions import (
    ActionKind,
    ActionValidationError,
    ToolCall,
)
from cppmega_mlx.runtime.code_verifier import (
    CodeVerifier,
    ToolUnavailableError,
    VerifierError,
)

# --------------------------------------------------------------------------- #
# SQL extraction
# --------------------------------------------------------------------------- #
_SQL_KEYWORD_RE = re.compile(
    r"\b(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|CREATE\s+TABLE|"
    r"CREATE\s+INDEX|REPLACE\s+INTO|WITH)\b",
    re.IGNORECASE,
)
# C++ raw string: R"delim(...)delim"  (delim is optional, no parens/spaces).
_RAW_STR_RE = re.compile(r'R"([^()\\ ]{0,16})\((.*?)\)\1"', re.DOTALL)
# Ordinary double-quoted C string literal.
_DQ_STR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _looks_like_sql(text: str) -> bool:
    return bool(_SQL_KEYWORD_RE.search(text))


def extract_sql_candidates(text: str) -> list[str]:
    """Extract embedded-SQL string literals from C/C++ text (dedup, ordered)."""
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _RAW_STR_RE.finditer(text):
        body = m.group(2).strip()
        if _looks_like_sql(body) and body not in seen:
            seen.add(body)
            out.append(body)
    for m in _DQ_STR_RE.finditer(text):
        body = m.group(1)
        # Unescape common C escapes so the SQL parses.
        body = body.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ").strip()
        if _looks_like_sql(body) and body not in seen:
            seen.add(body)
            out.append(body)
    return out


# --------------------------------------------------------------------------- #
# ToolRouter dataset
# --------------------------------------------------------------------------- #
def _payload_to_toolcall(payload: str, action_kind: str, repo_root: str) -> ToolCall | None:
    """Map a recorded (action_payload, action_kind) to a validated ToolCall.

    Returns ``None`` when the recorded step does not map onto a typed action we
    can express schema-valid (e.g. a raw bash command that is neither build nor
    test, or a path that escapes the root). We DROP those rather than fabricate.
    RAISES only on a genuinely corrupt payload (not valid JSON object).
    """
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"action_payload is not valid JSON: {exc}: {payload!r:.200}")
    if not isinstance(obj, dict):
        raise ValueError(f"action_payload must be a JSON object, got {payload!r:.200}")

    tool = obj.get("tool")
    command = obj.get("command")
    # Map Claude/Codex tool usage onto the frozen typed-action vocabulary.
    kind: ActionKind | None = None
    args: dict[str, Any] = {}
    if tool in ("Read",) and obj.get("file_path"):
        kind = ActionKind.READ_FILE
        args = {"path": obj["file_path"]}
    elif tool in ("Edit", "Write", "MultiEdit") and obj.get("file_path"):
        kind = ActionKind.APPLY_PATCH
        patch = obj.get("new_string") or obj.get("content") or obj.get("patch") or ""
        args = {"path": obj["file_path"], "patch": str(patch)[:4000] or "<edit>"}
    elif command and action_kind == "build":
        kind = ActionKind.RUN_BUILD
        args = {"command": str(command)}
    elif command and action_kind == "test":
        kind = ActionKind.RUN_TEST
        args = {"command": str(command)}
    if kind is None:
        return None
    try:
        return ToolCall.validated(kind, args, repo_root)
    except ActionValidationError:
        # Path escape / disallowed tool / shell smuggling -> drop, never emit.
        return None


def build_tool_router(df: pd.DataFrame, repo_root: str) -> list[dict[str, Any]]:
    router = ToolRouter(repo_root)
    schema_hint = _schema_hint()
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        tc = _payload_to_toolcall(str(r["action_payload"]), str(r["action_kind"]), repo_root)
        if tc is None:
            continue
        target = json.dumps(tc.to_wire(), separators=(",", ":"))
        # HARD GATE: round-trip the exact target string we will train on.
        parsed = router.parse(target)
        if parsed.to_wire() != tc.to_wire():
            raise AssertionError(
                f"ToolRouter round-trip mismatch for {target!r}: {parsed.to_wire()!r}"
            )
        rows.append(
            _chat_example(
                TOOL_ROUTER_SYSTEM,
                f"OBSERVATION:\n{_clip(r['obs_text'])}\n\nSCHEMA:\n{schema_hint}",
                target,
                meta={"expert": "tool_router", "kind": tc.kind.value,
                      "session_id": r.get("session_id")},
            )
        )
    # Null-route negatives: a non-actionable observation -> stop. These teach
    # refusal. The stop target is itself schema-valid (validated below).
    neg = _stop_negatives(df, router)
    rows.extend(neg)
    return rows


def _stop_negatives(df: pd.DataFrame, router: ToolRouter) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stop_target = json.dumps(
        {"kind": "stop", "args": {"reason": "no actionable tool for this observation"}},
        separators=(",", ":"),
    )
    router.parse(stop_target)  # validate the negative target too (fail-loud)
    # Use 'other'/'search'/'read'-only steps that did NOT map to a typed action
    # as refusal exemplars (bounded count to avoid swamping positives).
    cap = max(1, len(df) // 4)
    n = 0
    for _, r in df.iterrows():
        if n >= cap:
            break
        if str(r["action_kind"]) in ("other", "search"):
            out.append(
                _chat_example(
                    TOOL_ROUTER_SYSTEM,
                    f"OBSERVATION:\n{_clip(r['obs_text'])}\n\n"
                    "No file/build/test action applies to this step.",
                    stop_target,
                    meta={"expert": "tool_router", "kind": "stop",
                          "negative": True, "session_id": r.get("session_id")},
                )
            )
            n += 1
    return out


# --------------------------------------------------------------------------- #
# BuildOps dataset
# --------------------------------------------------------------------------- #
_FILE_LINE_RE = re.compile(r"([\w./+\-]+\.(?:c|cc|cpp|cxx|h|hpp|cu)):(\d+)", re.IGNORECASE)


def _classify_cause(log: str) -> str:
    low = log.lower()
    if "undefined reference" in low or "undefined symbol" in low:
        return "linker_undefined_reference"
    if "no such file or directory" in low and "#include" in low:
        return "missing_include_header"
    if "no member named" in low or "no matching function" in low:
        return "type_or_overload_error"
    if "expected" in low and ("';'" in low or "')'" in low or "'}'" in low):
        return "syntax_error"
    if "error:" in low:
        return "compile_error"
    if "cmake error" in low or "cmake warning" in low:
        return "cmake_configuration_error"
    return "build_failure_unclassified"


def _failing_log_span(result_text: str, limit: int = 2000) -> str:
    """Take the span around the first 'error' line (bounded)."""
    if not result_text:
        return ""
    lines = result_text.splitlines()
    idx = next((i for i, ln in enumerate(lines) if "error" in ln.lower()), 0)
    lo = max(0, idx - 3)
    hi = min(len(lines), idx + 20)
    return "\n".join(lines[lo:hi])[:limit]


def build_buildops(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Only REAL-exit, reward-bearing build steps (no fabricated rewards/labels).
    builds = df[(df["is_build"]) & (df["exit_code"].notna())]
    for _, r in builds.iterrows():
        exit_code = int(r["exit_code"])
        if exit_code == 0:
            continue  # a passing build is not a BuildOps repair example
        log = _failing_log_span(str(r["result_text"]))
        if not log.strip():
            continue
        cause = _classify_cause(log)
        m = _FILE_LINE_RE.search(log)
        file_ = m.group(1) if m else None
        line = int(m.group(2)) if m else None
        fix = _next_successful_fix(df, r)  # real edit diff or None (no fabrication)
        target = json.dumps(
            {"cause": cause, "file": file_, "line": line, "fix": fix},
            separators=(",", ":"),
        )
        build_system = _detect_build_system(str(r["action_payload"]), log)
        rows.append(
            _chat_example(
                BUILDOPS_SYSTEM,
                f"BUILD_SYSTEM: {build_system}\nFAILING_LOG:\n{log}",
                target,
                meta={"expert": "buildops", "exit_code": exit_code,
                      "labeled_fix": fix is not None,
                      "session_id": r.get("session_id")},
            )
        )
    return rows


def _detect_build_system(payload: str, log: str) -> str:
    blob = (payload + " " + log).lower()
    for name in ("cmake", "ninja", "bazel", "make", "clang++", "clang", "g++", "gcc"):
        if name in blob:
            return name
    return "unknown"


def _next_successful_fix(df: pd.DataFrame, failing_row: pd.Series) -> str | None:
    """The next edit_diff in the SAME session that precedes a passing build.

    Returns the REAL edit diff if such a sequence exists, else ``None``. Never
    fabricates a fix. Requires a later passing build (exit_code==0) in-session.
    """
    sid = failing_row.get("session_id")
    step = int(failing_row["step_idx"])
    same = df[(df["session_id"] == sid) & (df["step_idx"] > step)].sort_values("step_idx")
    if same.empty:
        return None
    later_pass = same[(same["is_build"]) & (same["exit_code"] == 0)]
    if later_pass.empty:
        return None
    pass_step = int(later_pass["step_idx"].iloc[0])
    edits = same[
        (same["action_kind"] == "edit")
        & (same["step_idx"] < pass_step)
        & (same["edit_diff"].notna())
    ].sort_values("step_idx")
    if edits.empty:
        return None
    diff = edits["edit_diff"].iloc[-1]
    return str(diff)[:4000] if diff else None


# --------------------------------------------------------------------------- #
# SQL dataset
# --------------------------------------------------------------------------- #
def build_sql(df: pd.DataFrame, verifier: CodeVerifier) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, r in df.iterrows():
        for field_name in ("obs_text", "edit_diff", "result_text"):
            text = r.get(field_name)
            if not isinstance(text, str):
                continue
            for sql in extract_sql_candidates(text):
                if sql in seen:
                    continue
                seen.add(sql)
                try:
                    outcome = verifier.validate_sql(sql, dialect="sqlite")
                except ToolUnavailableError:
                    raise  # sqlite missing -> fail loud (do not silently skip)
                except VerifierError:
                    # An internal verifier error is unexpected; surface it.
                    raise
                # repaired_sql is the SQL itself only when it ALREADY validates.
                # We never fabricate a repair offline; unvalidatable -> drop.
                if not outcome.ok:
                    continue
                target = json.dumps(
                    {"valid": True, "repaired_sql": sql}, separators=(",", ":")
                )
                rows.append(
                    _chat_example(
                        SQL_SYSTEM,
                        f"C++_CONTEXT:\n{_clip(text, 800)}\nSQL:\n{sql}\nSCHEMA: (none)",
                        target,
                        meta={"expert": "sql", "validated": True,
                              "session_id": r.get("session_id")},
                    )
                )
    return rows


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _clip(text: Any, limit: int = 2000) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= limit else s[:limit] + "\n...[truncated]"


def _chat_example(
    system: str, user: str, target: str, *, meta: dict[str, Any]
) -> dict[str, Any]:
    """A messages-style SFT example (HF/MLX-LM chat SFT compatible)."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "meta": {k: (None if pd.isna(v) else v) for k, v in meta.items()},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in PARQUET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"trajectory parquet missing required columns {missing}; "
            f"got {list(df.columns)}"
        )


def build_all(
    parquet_path: str, out_dir: str, repo_root: str
) -> dict[str, list[dict[str, Any]]]:
    df = pd.read_parquet(parquet_path)
    _require_columns(df)
    verifier = CodeVerifier(repo_root)
    datasets = {
        "tool_router": build_tool_router(df, repo_root),
        "buildops": build_buildops(df),
        "sql": build_sql(df, verifier),
    }
    out = Path(out_dir)
    for name, rows in datasets.items():
        _write_jsonl(out / f"{name}.jsonl", rows)
    return datasets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True, help="agent-trajectory parquet path")
    ap.add_argument("--out", required=True, help="output dir for *.jsonl datasets")
    ap.add_argument(
        "--repo-root",
        default=str(Path.cwd()),
        help="absolute repo root for typed-action path containment",
    )
    args = ap.parse_args(argv)
    datasets = build_all(args.parquet, args.out, args.repo_root)
    summary = {name: len(rows) for name, rows in datasets.items()}
    print(json.dumps({"out": args.out, "rows": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
