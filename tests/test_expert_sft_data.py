"""Tests for the per-expert SFT data builder + train-script scaffolding.

Asserts (RULE #1):
* Every ToolRouter target round-trips through ``ToolRouter.parse`` (schema +
  typed-action validation) -- no unvalidatable target is ever emitted.
* BuildOps labels come ONLY from real-exit-code build transitions, and a labeled
  ``fix`` only ever equals a REAL in-session edit diff (no fabrication).
* SQL targets are validated by the sqlite verifier (validated == True).
* The build + train scripts import and arg-parse; training is a dry-run only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from cppmega_mlx.data.agent_trajectory import PARQUET_COLUMNS
from cppmega_mlx.inference.tool_router_schema import ToolRouter

import scripts.build_expert_sft_data as bd
import scripts.train_expert_lora as tr

REPO_ROOT = "/Volumes/external/sources/cppmega.mlx"
SAMPLE = Path(REPO_ROOT) / "outputs/agent_trajectories/sample_transitions.parquet"

sqlite_missing = shutil.which("sqlite3") is None


def _row(**kw):
    base = {
        "session_id": "s0", "source": "claude", "repo": "cppmega", "step_idx": 0,
        "obs_text": "", "action_kind": "other",
        "action_payload": json.dumps({"tool": "Bash", "command": "echo hi"}),
        "result_text": "", "exit_code": None, "is_build": False, "is_test": False,
        "reward": None, "edit_diff": None,
    }
    base.update(kw)
    return base


def _synthetic_df() -> pd.DataFrame:
    edit_payload = json.dumps({"tool": "Edit", "file_path": "src/a.cpp",
                               "new_string": "int main(){return 0;}"})
    read_payload = json.dumps({"tool": "Read", "file_path": "src/a.cpp"})
    build_payload = json.dumps({"tool": "Bash", "command": "cmake --build build"})
    # Self-contained, standalone-validatable embedded SQL (schema DDL + a
    # tableless SELECT). DML against undeclared tables is intentionally dropped
    # by the builder (the sqlite verifier fails-loud on "no such table").
    sql_obs = ('db.exec(R"sql(CREATE TABLE users(id INTEGER, name TEXT))sql");\n'
               'const char* q = "SELECT 1 AS one";')
    rows = [
        # session sb: failing build -> edit -> passing build (labeled fix)
        _row(session_id="sb", step_idx=0, action_kind="build", is_build=True,
             exit_code=1, reward=0.0, action_payload=build_payload,
             result_text="src/a.cpp:7:3: error: expected ';' before '}' token"),
        _row(session_id="sb", step_idx=1, action_kind="edit",
             action_payload=edit_payload,
             edit_diff="--- a/src/a.cpp\n+++ b/src/a.cpp\n@@\n-x\n+x;"),
        _row(session_id="sb", step_idx=2, action_kind="build", is_build=True,
             exit_code=0, reward=1.0, action_payload=build_payload,
             result_text="[100%] Built target a"),
        # session sf: failing build with NO later pass -> fix must be null
        _row(session_id="sf", step_idx=0, action_kind="build", is_build=True,
             exit_code=2, reward=0.0, action_payload=build_payload,
             result_text="undefined reference to `foo()'\ncollect2: error: ld"),
        # router positives: read + edit
        _row(session_id="sr", step_idx=0, action_kind="read",
             action_payload=read_payload, obs_text="need to read a.cpp"),
        _row(session_id="sr", step_idx=1, action_kind="edit",
             action_payload=edit_payload, obs_text="apply fix"),
        # router negative: a search step (-> stop)
        _row(session_id="sr", step_idx=2, action_kind="search",
             action_payload=json.dumps({"tool": "Grep", "pattern": "foo"}),
             obs_text="grep around"),
        # sql-bearing obs
        _row(session_id="sq", step_idx=0, action_kind="other", obs_text=sql_obs),
    ]
    return pd.DataFrame(rows, columns=PARQUET_COLUMNS)


@pytest.fixture()
def synth_parquet(tmp_path):
    p = tmp_path / "synth.parquet"
    _synthetic_df().to_parquet(p)
    return str(p)


# --------------------------------------------------------------------------- #
def test_tool_router_targets_all_schema_valid(synth_parquet):
    df = pd.read_parquet(synth_parquet)
    rows = bd.build_tool_router(df, REPO_ROOT)
    assert rows, "expected at least one tool_router example"
    router = ToolRouter(REPO_ROOT)
    kinds = set()
    for ex in rows:
        target = ex["messages"][-1]["content"]
        tc = router.parse(target)  # RAISES if not schema/typed-action valid
        kinds.add(tc.kind.value)
    # read + apply_patch positives and a stop negative all present.
    assert {"read_file", "apply_patch", "stop"} <= kinds
    assert any(ex["meta"].get("negative") for ex in rows)


def test_buildops_labels_only_from_real_exit(synth_parquet):
    df = pd.read_parquet(synth_parquet)
    rows = bd.build_buildops(df)
    assert rows, "expected BuildOps examples from failing builds"
    fixes = {}
    for ex in rows:
        tgt = json.loads(ex["messages"][-1]["content"])
        assert tgt["cause"], "cause must be non-empty"
        fixes[ex["meta"]["session_id"]] = tgt["fix"]
    # sb has a real in-session edit before a passing build -> labeled fix.
    assert fixes["sb"] is not None and "a.cpp" in fixes["sb"]
    # sf has no later passing build -> fix MUST be null (no fabrication).
    assert fixes["sf"] is None
    # No example may come from a row without a real exit code.
    builds = df[(df["is_build"]) & (df["exit_code"].notna()) & (df["exit_code"] != 0)]
    assert len(rows) == len(builds)


def test_buildops_skips_rows_without_exit_code():
    df = pd.DataFrame(
        [_row(action_kind="build", is_build=True, exit_code=None,
              result_text="error: something")],
        columns=PARQUET_COLUMNS,
    )
    assert bd.build_buildops(df) == []


@pytest.mark.skipif(sqlite_missing, reason="sqlite3 not on PATH")
def test_sql_targets_validated(synth_parquet):
    from cppmega_mlx.runtime.code_verifier import CodeVerifier

    df = pd.read_parquet(synth_parquet)
    rows = bd.build_sql(df, CodeVerifier(REPO_ROOT))
    assert rows, "expected validated SQL examples"
    for ex in rows:
        tgt = json.loads(ex["messages"][-1]["content"])
        assert tgt["valid"] is True
        assert ex["meta"]["validated"] is True
        # The emitted SQL must itself re-validate (ground truth, not model claim).
        out = CodeVerifier(REPO_ROOT).validate_sql(tgt["repaired_sql"], dialect="sqlite")
        assert out.ok


def test_extract_sql_candidates():
    txt = 'R"(SELECT 1)" and "DELETE FROM t WHERE id=2" and "not sql"'
    cands = bd.extract_sql_candidates(txt)
    assert any("SELECT 1" in c for c in cands)
    assert any("DELETE FROM t" in c for c in cands)
    assert all("not sql" not in c for c in cands)


@pytest.mark.skipif(sqlite_missing, reason="sqlite3 not on PATH")
def test_build_all_writes_three_jsonl(synth_parquet, tmp_path):
    out = tmp_path / "ds"
    datasets = bd.build_all(synth_parquet, str(out), REPO_ROOT)
    for name in ("tool_router", "buildops", "sql"):
        p = out / f"{name}.jsonl"
        assert p.exists()
        n = sum(1 for _ in p.open())
        assert n == len(datasets[name])


def test_build_all_on_real_sample_parquet():
    if not SAMPLE.exists():
        pytest.skip("sample parquet not present")
    df = pd.read_parquet(SAMPLE)
    # Must not raise; targets that survive must be schema-valid.
    rows = bd.build_tool_router(df, REPO_ROOT)
    router = ToolRouter(REPO_ROOT)
    for ex in rows:
        router.parse(ex["messages"][-1]["content"])


def test_missing_columns_raises(tmp_path):
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"obs_text": ["x"]}).to_parquet(p)
    with pytest.raises(ValueError, match="missing required columns"):
        bd.build_all(str(p), str(tmp_path / "o"), REPO_ROOT)


# --------------------------------------------------------------------------- #
# Train script: import + arg-parse + dry-run only (no training).
# --------------------------------------------------------------------------- #
def test_train_parser_and_dry_run(synth_parquet, tmp_path):
    # build a tiny jsonl to point --data at
    out = tmp_path / "ds"
    bd._write_jsonl(out / "tool_router.jsonl",
                    bd.build_tool_router(pd.read_parquet(synth_parquet), REPO_ROOT))
    args = tr.build_parser().parse_args([
        "--expert", "tool_router",
        "--base", "Qwen/Qwen3-4B-Instruct-2507",
        "--data", str(out / "tool_router.jsonl"),
        "--out", str(tmp_path / "adapter"),
    ])
    cfg = tr.TrainConfig(expert=args.expert, base=args.base, data=args.data,
                         out=args.out, backend=args.backend)
    plan = tr.run(cfg, allow_train=False)
    assert plan["dry_run"] is True
    assert plan["expert"] == "tool_router"
    assert plan["examples"] >= 1
    # adapter dir must NOT be created in a dry run (no training side effects).
    assert not Path(args.out).exists()


def test_train_validate_rejects_empty(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    cfg = tr.TrainConfig(expert="sql", base="Qwen/Qwen3-4B-Instruct-2507",
                         data=str(empty), out=str(tmp_path / "a"))
    with pytest.raises(ValueError, match="empty"):
        cfg.validate()


def test_train_validate_rejects_bad_expert(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text('{"messages":[]}\n', encoding="utf-8")
    cfg = tr.TrainConfig(expert="nope", base="b", data=str(f), out=str(tmp_path / "a"))
    with pytest.raises(ValueError, match="--expert"):
        cfg.validate()
