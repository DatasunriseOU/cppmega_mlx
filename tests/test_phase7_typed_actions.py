"""Tests for the Phase-7 base-independent agentic infra.

Verifier tests use REAL clang++ / sqlite3 and skip if absent (fail-loud, never
fake-pass).
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from cppmega_mlx.eval.eval_harness import EvalCase, EvalRunner
from cppmega_mlx.inference.tool_router_schema import (
    TOOL_CALL_GBNF,
    TOOL_CALL_JSON_SCHEMA,
    ToolRouter,
    SchemaValidationError,
)
from cppmega_mlx.inference.typed_actions import (
    ActionKind,
    DisallowedCommandError,
    MissingArgumentError,
    PathEscapeError,
    ToolCall,
    UnknownActionError,
)
from cppmega_mlx.runtime.code_verifier import CodeVerifier, ToolUnavailableError
from cppmega_mlx.runtime.orchestrator import Orchestrator, StubPolicy

HAVE_CLANG = bool(shutil.which("clang++") or shutil.which("clang"))
HAVE_SQLITE = bool(shutil.which("sqlite3"))


# --------------------------------------------------------------------------- #
# 1. typed-action validation RAISES on bad / escape / missing
# --------------------------------------------------------------------------- #
def test_unknown_kind_raises(tmp_path):
    with pytest.raises(UnknownActionError):
        ToolCall.validated("definitely_not_a_kind", {}, str(tmp_path))


def test_missing_required_arg_raises(tmp_path):
    with pytest.raises(MissingArgumentError):
        ToolCall.validated(ActionKind.READ_FILE, {}, str(tmp_path))


def test_path_escape_relative_raises(tmp_path):
    with pytest.raises(PathEscapeError):
        ToolCall.validated(
            ActionKind.READ_FILE, {"path": "../../etc/passwd"}, str(tmp_path)
        )


def test_path_escape_absolute_raises(tmp_path):
    with pytest.raises(PathEscapeError):
        ToolCall.validated(
            ActionKind.READ_FILE, {"path": "/etc/passwd"}, str(tmp_path)
        )


def test_disallowed_build_command_raises(tmp_path):
    with pytest.raises(DisallowedCommandError):
        ToolCall.validated(
            ActionKind.RUN_BUILD, {"command": "rm -rf /"}, str(tmp_path)
        )


def test_shell_metachar_smuggling_raises(tmp_path):
    with pytest.raises(DisallowedCommandError):
        ToolCall.validated(
            ActionKind.RUN_BUILD,
            {"command": "cmake --build . && rm -rf /"},
            str(tmp_path),
        )


def test_valid_call_roundtrips(tmp_path):
    (tmp_path / "a.cpp").write_text("int main(){return 0;}\n")
    call = ToolCall.validated(
        ActionKind.READ_FILE, {"path": "a.cpp"}, str(tmp_path)
    )
    assert call.kind is ActionKind.READ_FILE
    assert call.to_wire() == {"kind": "read_file", "args": {"path": "a.cpp"}}


def test_no_raw_shell_kind_exists():
    names = {k.value for k in ActionKind}
    assert "shell" not in names and "exec" not in names and "bash" not in names


# --------------------------------------------------------------------------- #
# 2. grammar / schema round-trips a sample tool call
# --------------------------------------------------------------------------- #
def test_schema_has_one_branch_per_kind():
    assert len(TOOL_CALL_JSON_SCHEMA["oneOf"]) == len(list(ActionKind))


def test_gbnf_has_rule_per_kind():
    for kind in ActionKind:
        assert f"call-{kind.value.replace('_', '-')}" in TOOL_CALL_GBNF


def test_router_roundtrip_from_model_text(tmp_path):
    (tmp_path / "x.cpp").write_text("int main(){return 0;}\n")
    router = ToolRouter(str(tmp_path))
    text = 'sure, I will: {"kind": "read_file", "args": {"path": "x.cpp"}} done'
    call = router.parse(text)
    assert call.kind is ActionKind.READ_FILE
    assert call.args["path"] == "x.cpp"


def test_router_rejects_unknown_arg(tmp_path):
    router = ToolRouter(str(tmp_path))
    with pytest.raises(SchemaValidationError):
        router.parse('{"kind": "read_file", "args": {"path": "x", "evil": "y"}}')


def test_router_rejects_missing_required(tmp_path):
    router = ToolRouter(str(tmp_path))
    with pytest.raises(SchemaValidationError):
        router.parse('{"kind": "read_file", "args": {}}')


# --------------------------------------------------------------------------- #
# 3. code_verifier.syntax_check PASS on valid, FAIL on broken C++
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_CLANG, reason="clang++ not available")
def test_syntax_check_pass(tmp_path):
    src = tmp_path / "good.cpp"
    src.write_text("int add(int a, int b){ return a + b; }\n")
    v = CodeVerifier(tmp_path)
    out = v.syntax_check(src)
    assert out.ok and out.exit_code == 0


@pytest.mark.skipif(not HAVE_CLANG, reason="clang++ not available")
def test_syntax_check_fail(tmp_path):
    src = tmp_path / "bad.cpp"
    src.write_text("int add(int a, int b){ return a + ; }\n")  # broken
    v = CodeVerifier(tmp_path)
    out = v.syntax_check(src)
    assert not out.ok and out.exit_code != 0 and out.diagnostics


def test_syntax_check_no_clang_raises(tmp_path):
    v = CodeVerifier(tmp_path, clangxx=None)
    v._clangxx = None  # force unavailable
    src = tmp_path / "z.cpp"
    src.write_text("int main(){return 0;}\n")
    with pytest.raises(ToolUnavailableError):
        v.syntax_check(src)


# --------------------------------------------------------------------------- #
# 4. fs_state_diff detects a file change
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_CLANG, reason="needs a runnable build tool path")
def test_fs_state_diff_detects_change(tmp_path):
    # Use a trivial command that mutates the tree: clang++ compiling to an object.
    src = tmp_path / "m.cpp"
    src.write_text("int main(){return 0;}\n")
    v = CodeVerifier(tmp_path)
    clang = shutil.which("clang++") or shutil.which("clang")
    out = v.build([clang, "-c", "m.cpp", "-o", "m.o"], copy_tree=False)
    assert out.ok
    assert out.fs_state_diff.any_change
    assert "m.o" in out.fs_state_diff.added


# --------------------------------------------------------------------------- #
# 5. validate_sql
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_SQLITE, reason="sqlite3 not available")
def test_validate_sql_ok(tmp_path):
    v = CodeVerifier(tmp_path)
    out = v.validate_sql("CREATE TABLE t(a INT); SELECT a FROM t;")
    assert out.ok and out.exit_code == 0


@pytest.mark.skipif(not HAVE_SQLITE, reason="sqlite3 not available")
def test_validate_sql_broken(tmp_path):
    v = CodeVerifier(tmp_path)
    out = v.validate_sql("CREATE TABEL t(a INT);")  # typo
    assert not out.ok


# --------------------------------------------------------------------------- #
# 6. orchestrator demo loop end-to-end with stub policy + real verifier
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_CLANG, reason="clang++ not available")
def test_orchestrator_loop_end_to_end(tmp_path):
    (tmp_path / "ok.cpp").write_text("int add(int a,int b){return a+b;}\n")
    policy = StubPolicy(
        [
            '{"kind": "read_file", "args": {"path": "ok.cpp"}}',
            '{"kind": "stop", "args": {"reason": "done"}}',
        ]
    )
    orch = Orchestrator(tmp_path, policy)
    trace = orch.run()
    assert trace[0].tool_call.kind is ActionKind.READ_FILE
    assert trace[0].observation.outcome["ok"] is True
    assert trace[-1].tool_call.kind is ActionKind.STOP


def test_orchestrator_rejects_escaping_intent(tmp_path):
    policy = StubPolicy(
        ['{"kind": "read_file", "args": {"path": "../../etc/passwd"}}']
    )
    orch = Orchestrator(tmp_path, policy)
    with pytest.raises(PathEscapeError):
        orch.run()


# --------------------------------------------------------------------------- #
# 7. eval_harness one synthetic case through the hard gate
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_CLANG, reason="clang++ not available")
def test_eval_harness_hard_gate_pass(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "good.cpp").write_text("int main(){return 0;}\n")
    case = EvalCase(
        name="syn-ok",
        repo_snapshot=snap,
        task="ensure it compiles",
        oracle_kind="syntax",
        oracle_target="good.cpp",
    )
    res = EvalRunner().run_case(case)
    assert res.passed is True


@pytest.mark.skipif(not HAVE_CLANG, reason="clang++ not available")
def test_eval_harness_hard_gate_fail_and_judge_cannot_override(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "bad.cpp").write_text("int main(){return ;}\n")  # broken
    case = EvalCase(
        name="syn-bad",
        repo_snapshot=snap,
        task="ensure it compiles",
        oracle_kind="syntax",
        oracle_target="bad.cpp",
    )
    # Advisory judge tries to claim success; hard gate must still fail.
    res = EvalRunner().run_case(case, judge=lambda c, o: 1.0)
    assert res.passed is False
    assert res.advisory_score == 1.0  # recorded but powerless


@pytest.mark.skipif(not HAVE_CLANG, reason="clang++ not available")
def test_eval_harness_candidate_fixes_then_passes(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "f.cpp").write_text("int main(){return ;}\n")  # broken in snapshot

    def candidate(work: Path) -> None:
        (work / "f.cpp").write_text("int main(){return 0;}\n")  # fix it

    case = EvalCase(
        name="syn-fix",
        repo_snapshot=snap,
        task="fix the syntax error",
        oracle_kind="syntax",
        oracle_target="f.cpp",
    )
    res = EvalRunner().run_case(case, candidate)
    assert res.passed is True
