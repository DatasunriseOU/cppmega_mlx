"""Tests for the Phase-8 C/C++ agent-trajectory extractor.

Validates against SYNTHETIC-but-faithful fixtures (mirroring the real Claude
.jsonl + Codex rollout schemas) plus a small live local sample. Asserts:

* transitions are extracted in order,
* the C/C++ filter keeps C/C++ sessions and drops non-C/C++,
* Codex 'Process exited with code N' is parsed into a real exit_code,
* Claude Edit structuredPatch is captured as an edit_diff,
* NO fabricated rewards: non-build/non-test steps -> reward is None.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cppmega_mlx.data.agent_trajectory import (
    ACTION_BUILD,
    ACTION_EDIT,
    ACTION_TEST,
    AgentTransition,
    SessionRef,
    classify_cpp,
    enumerate_local_sessions,
    extract_all,
    extract_session,
    walk_claude,
    walk_codex,
)


# --------------------------------------------------------------------------- #
# Fixture builders (faithful to the real schemas)
# --------------------------------------------------------------------------- #
def _claude_cpp_session() -> list[dict]:
    sid = "test-session-cpp"
    cwd = "/Volumes/external/sources/cppmega"
    return [
        {"type": "user", "sessionId": sid, "cwd": cwd, "gitBranch": "main",
         "message": {"role": "user", "content": "fix the build in src/foo.cpp"}},
        {"type": "assistant", "sessionId": sid, "cwd": cwd,
         "message": {"role": "assistant", "content": [
             {"type": "thinking", "thinking": "let me edit foo.cpp"},
             {"type": "tool_use", "id": "tu_edit", "name": "Edit",
              "input": {"file_path": "src/foo.cpp", "old_string": "int x=1;",
                        "new_string": "int x = 2;"}},
         ]}},
        {"type": "user", "sessionId": sid, "cwd": cwd,
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_edit", "content": "ok"}]},
         "toolUseResult": {
             "filePath": "src/foo.cpp", "oldString": "int x=1;",
             "newString": "int x = 2;", "originalFile": None,
             "structuredPatch": [
                 {"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1,
                  "lines": ["-int x=1;", "+int x = 2;"]}],
             "userModified": False, "replaceAll": False}},
        {"type": "assistant", "sessionId": sid, "cwd": cwd,
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "tu_build", "name": "Bash",
              "input": {"command": "cmake --build build"}}]}},
        {"type": "user", "sessionId": sid, "cwd": cwd,
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_build", "content": "done"}]},
         "toolUseResult": {
             "stdout": "[100%] Built target foo", "stderr": "",
             "interrupted": False, "isImage": False,
             "returnCodeInterpretation": "Command completed successfully (exit code 0)",
             "noOutputExpected": False}},
        {"type": "assistant", "sessionId": sid, "cwd": cwd,
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "tu_read", "name": "Read",
              "input": {"file_path": "src/foo.cpp"}}]}},
        {"type": "user", "sessionId": sid, "cwd": cwd,
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_read", "content": "..."}]},
         "toolUseResult": {"type": "text", "filePath": "src/foo.cpp",
                           "content": "int x = 2;", "originalFile": None,
                           "structuredPatch": [], "userModified": False}},
    ]


def _claude_python_session() -> list[dict]:
    sid = "test-session-py"
    cwd = "/Volumes/external/sources/some_python_proj"
    return [
        {"type": "user", "sessionId": sid, "cwd": cwd,
         "message": {"role": "user", "content": "update the django view"}},
        {"type": "assistant", "sessionId": sid, "cwd": cwd,
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "tu1", "name": "Edit",
              "input": {"file_path": "views.py", "old_string": "a", "new_string": "b"}}]}},
        {"type": "user", "sessionId": sid, "cwd": cwd,
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}]},
         "toolUseResult": {"filePath": "views.py", "oldString": "a", "newString": "b",
                           "originalFile": None, "structuredPatch": [], "userModified": False}},
    ]


def _codex_cpp_session() -> list[dict]:
    sid = "019ddead-codex-cpp"
    cwd = "/Volumes/external/sources/tilelang"
    return [
        {"type": "session_meta", "payload": {"id": sid, "cwd": cwd}},
        {"type": "event_msg", "payload": {"type": "user_message",
                                          "text": "build the cuda kernel"}},
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "exec_command",
            "arguments": json.dumps({"cmd": "nvcc -c kernel.cu -o kernel.o",
                                     "workdir": cwd}),
            "call_id": "call_build"}},
        {"type": "response_item", "payload": {
            "type": "function_call_output", "call_id": "call_build",
            "output": "Chunk ID: ab\nWall time: 1.2s\nProcess exited with code 0\n"
                      "Output:\nkernel.o written"}},
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "exec_command",
            "arguments": json.dumps({"cmd": "ctest --output-on-failure",
                                     "workdir": cwd}),
            "call_id": "call_test"}},
        {"type": "response_item", "payload": {
            "type": "function_call_output", "call_id": "call_test",
            "output": "Process exited with code 1\nOutput:\n1 test failed"}},
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "exec_command",
            "arguments": json.dumps({"cmd": "cat kernel.cu", "workdir": cwd}),
            "call_id": "call_cat"}},
        {"type": "response_item", "payload": {
            "type": "function_call_output", "call_id": "call_cat",
            "output": "Process exited with code 0\nOutput:\n__global__ void k(){}"}},
    ]


def _codex_python_session() -> list[dict]:
    sid = "019ddead-codex-py"
    cwd = "/Volumes/external/sources/django_app"
    return [
        {"type": "session_meta", "payload": {"id": sid, "cwd": cwd}},
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "exec_command",
            "arguments": json.dumps({"cmd": "pip install requests", "workdir": cwd}),
            "call_id": "c1"}},
        {"type": "response_item", "payload": {
            "type": "function_call_output", "call_id": "c1",
            "output": "Process exited with code 0\nOutput:\nok"}},
    ]


# --------------------------------------------------------------------------- #
# classify_cpp
# --------------------------------------------------------------------------- #
def test_classify_cpp_positive():
    assert classify_cpp("/path/to/foo.cpp")
    assert classify_cpp("ran cmake --build")
    assert classify_cpp("nvcc kernel.cu")
    assert classify_cpp("include foo.hpp")


def test_classify_cpp_negative():
    assert not classify_cpp("update views.py in django_app")
    assert not classify_cpp("npm install && node index.js")
    assert not classify_cpp("")


# --------------------------------------------------------------------------- #
# Claude walking
# --------------------------------------------------------------------------- #
def test_walk_claude_transitions_and_edit_diff():
    events = _claude_cpp_session()
    trs = list(walk_claude(events, "test-session-cpp", "/Volumes/external/sources/cppmega"))
    # 3 tool_use actions: Edit, Bash(build), Read
    assert len(trs) == 3
    assert [t.step_idx for t in trs] == [0, 1, 2]

    edit, build, read = trs
    # Edit captured as a diff.
    assert edit.action_kind == ACTION_EDIT
    assert edit.edit_diff is not None
    assert "+int x = 2;" in edit.edit_diff
    # Edit is not build/test -> reward None (no fabrication).
    assert edit.reward is None
    assert edit.exit_code is None

    # Build step: parsed exit code 0 from returnCodeInterpretation -> reward 1.0
    assert build.action_kind == ACTION_BUILD
    assert build.is_build is True
    assert build.exit_code == 0
    assert build.reward == 1.0

    # Read step: never reward-bearing.
    assert read.action_kind == "read"
    assert read.reward is None
    assert read.exit_code is None


def test_claude_observation_carries_cwd_and_prompt():
    events = _claude_cpp_session()
    trs = list(walk_claude(events, "test-session-cpp", "/Volumes/external/sources/cppmega"))
    assert "cwd=/Volumes/external/sources/cppmega" in trs[0].obs_text
    assert "fix the build" in trs[0].obs_text


# --------------------------------------------------------------------------- #
# Codex walking
# --------------------------------------------------------------------------- #
def test_walk_codex_exit_codes_and_rewards():
    events = _codex_cpp_session()
    trs = list(walk_codex(events, "019ddead-codex-cpp", "/Volumes/external/sources/tilelang"))
    assert len(trs) == 3
    build, test, cat = trs

    # nvcc build -> exit 0 -> reward 1.0
    assert build.action_kind == ACTION_BUILD
    assert build.exit_code == 0
    assert build.reward == 1.0

    # ctest -> exit 1 -> reward 0.0 (real failing outcome, not fabricated)
    assert test.action_kind == ACTION_TEST
    assert test.is_test is True
    assert test.exit_code == 1
    assert test.reward == 0.0

    # cat -> exit code parsed (0) but NOT build/test -> reward None.
    assert cat.exit_code == 0
    assert cat.reward is None


def test_codex_no_fabricated_reward_on_non_build_test():
    events = _codex_cpp_session()
    trs = list(walk_codex(events, "sid", "/Volumes/external/sources/tilelang"))
    for t in trs:
        if not (t.is_build or t.is_test):
            assert t.reward is None


# --------------------------------------------------------------------------- #
# Filtering via extract_session / extract_all (synthetic temp files)
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def test_extract_filters_non_cpp(tmp_path: Path):
    cpp_claude = tmp_path / "cpp.jsonl"
    py_claude = tmp_path / "py.jsonl"
    cpp_codex = tmp_path / "rollout-cpp.jsonl"
    py_codex = tmp_path / "rollout-py.jsonl"
    _write_jsonl(cpp_claude, _claude_cpp_session())
    _write_jsonl(py_claude, _claude_python_session())
    _write_jsonl(cpp_codex, _codex_cpp_session())
    _write_jsonl(py_codex, _codex_python_session())

    refs = [
        SessionRef(str(cpp_claude), "claude"),
        SessionRef(str(py_claude), "claude"),
        SessionRef(str(cpp_codex), "codex"),
        SessionRef(str(py_codex), "codex"),
    ]
    transitions, stats = extract_all(refs)
    assert stats.sessions_seen == 4
    assert stats.sessions_kept == 2  # 1 cpp claude + 1 cpp codex
    assert stats.sessions_dropped == 2  # python ones
    # build+test exit codes counted only for cpp sessions
    assert stats.build_test_with_exit >= 3  # claude build + codex build + codex test

    # No fabricated rewards anywhere.
    for t in transitions:
        if not (t.is_build or t.is_test):
            assert t.reward is None


def test_python_session_dropped_directly(tmp_path: Path):
    py = tmp_path / "rollout-py.jsonl"
    _write_jsonl(py, _codex_python_session())
    _sid, trs, is_cpp = extract_session(SessionRef(str(py), "codex"))
    assert is_cpp is False
    assert trs == []


# --------------------------------------------------------------------------- #
# AgentTransition fail-loud reward guard
# --------------------------------------------------------------------------- #
def test_agent_transition_rejects_fabricated_reward():
    with pytest.raises(ValueError):
        AgentTransition(
            session_id="s", source="codex", repo="/x", step_idx=0,
            obs_text="o", action_kind="read", action_payload="{}",
            result_text="r", exit_code=0, is_build=False, is_test=False,
            reward=1.0, edit_diff=None,
        )


# --------------------------------------------------------------------------- #
# Live local sample (best-effort; skips if no local C/C++ sessions present)
# --------------------------------------------------------------------------- #
def test_live_local_sample_small():
    refs = enumerate_local_sessions()
    if not refs:
        pytest.skip("no local sessions on this machine")
    # Bias toward known C/C++ project dirs so the 5-session cap finds C/C++.
    cpp_refs = [r for r in refs if "cppmega" in r.path or "tilelang" in r.path]
    sample = (cpp_refs or refs)[:5]
    transitions, stats = extract_all(sample, max_sessions=5)
    assert stats.sessions_seen <= 5
    # Real C/C++ sessions on dev machine should produce transitions.
    if stats.sessions_kept > 0:
        assert stats.transitions > 0
    # RULE #1 holds on real data too.
    for t in transitions:
        if not (t.is_build or t.is_test):
            assert t.reward is None
