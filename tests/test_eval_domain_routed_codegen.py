from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _test_cpp_compiler() -> str | None:
    apple_clang = Path("/usr/bin/clang++")
    return str(apple_clang) if apple_clang.exists() else shutil.which("clang++")


def _load_eval_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_domain_routed_codegen.py"
    )
    spec = importlib.util.spec_from_file_location(
        "eval_domain_routed_codegen", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_domain_eval_prompt_loader_requires_domains(tmp_path):
    mod = _load_eval_module()
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "p0",
                "task_type": "cpp_docstring_to_code",
                "prompt": "// add\n",
                "required_domains": ["CPP"],
                "expected_sidecars": ["token_domain_ids"],
                "compile_suffix": "\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
                "run_binary": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = mod.load_prompts(path)

    assert len(prompts) == 1
    assert prompts[0].required_domains == ("CPP",)
    assert prompts[0].expected_sidecars == ("token_domain_ids",)


def _case5_ksh_eval_row() -> dict:
    for line in Path("evals/domain_routed_prompts.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        row = json.loads(line)
        if row["id"] == "ksh_build_sidecar_route":
            return row
    raise AssertionError("missing ksh_build_sidecar_route eval fixture")


def test_domain_eval_freezes_structured_ksh_prompt_sidecars(tmp_path: Path) -> None:
    mod = _load_eval_module()
    row = _case5_ksh_eval_row()
    row["compile_suffix"] = "\nint main(){return 0;}\n"
    row["run_binary"] = True
    path = tmp_path / "ksh-prompts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    prompt = mod.load_prompts(path)[0]

    assert prompt.prompt_token_ids == (2, 245, 501, 502, 246, 215, 601, 602, 216, 3)
    assert prompt.prompt_sidecars["token_domain_ids"] == (
        0,
        24,
        24,
        24,
        24,
        43,
        43,
        43,
        43,
        0,
    )
    assert prompt.prompt_sidecars["token_shell_edges"] == (
        {"from": 2, "to": 3, "kind": 40},
    )
    assert prompt.prompt_sidecars["token_diagnostic_edges"] == (
        {"from": 6, "to": 7, "kind": 64},
    )
    assert prompt.prompt_sidecars["token_cross_domain_edges"] == (
        {"from": 3, "to": 6, "kind": 100},
    )

    report = mod.evaluate([prompt], {}, compile=False)
    assert report["rows"][0]["prompt_sidecar_receipt"] == {
        "token_count": 10,
        "columns": sorted(prompt.prompt_sidecars),
        "edge_counts": {
            "token_domain_edges": 0,
            "token_build_edges": 0,
            "token_shell_edges": 1,
            "token_diagnostic_edges": 1,
            "token_cross_domain_edges": 1,
        },
    }


def test_domain_eval_rejects_misaligned_prompt_sidecar_edges(tmp_path: Path) -> None:
    mod = _load_eval_module()
    row = _case5_ksh_eval_row()
    row["compile_suffix"] = "\nint main(){return 0;}\n"
    row["run_binary"] = True
    row["prompt_sidecars"]["token_shell_edges"][0]["to"] = len(
        row["prompt_token_ids"]
    )
    path = tmp_path / "bad-ksh-prompts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside prompt tokens"):
        mod.load_prompts(path)


def test_domain_eval_rejects_shipped_prompt_without_executable_oracle():
    mod = _load_eval_module()
    path = Path("evals/domain_routed_prompts.jsonl")

    with pytest.raises(ValueError, match="executable compile oracle"):
        mod.load_prompts(path)


def test_domain_eval_evaluate_without_completion_fails_closed():
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=("token_domain_ids",),
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )

    report = mod.evaluate([prompt], {}, compile=True)

    assert report["missing_completion"] == 1
    assert report["failed"] == 1
    assert report["passed"] is False
    assert report["rows"][0]["status"] == "missing_completion"


def test_domain_eval_compile_gate_accepts_simple_cpp_completion():
    mod = _load_eval_module()
    compiler = _test_cpp_compiler()
    assert compiler is not None
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=("token_domain_ids",),
        compile_prefix="",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )

    result = mod.compile_cpp_completion(
        prompt,
        "int add(int a, int b){ return a + b; }",
        compiler=compiler,
    )

    assert result["status"] == "compile_passed"


def test_domain_eval_runtime_oracle_rejects_wrong_compilable_completion():
    mod = _load_eval_module()
    compiler = _test_cpp_compiler()
    assert compiler is not None
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=("token_domain_ids",),
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )

    result = mod.compile_cpp_completion(
        prompt,
        "int add(int, int){ return 0; }",
        compiler=compiler,
    )

    assert result["status"] == "runtime_failed"


def test_domain_eval_reports_compile_timeout_instead_of_raising(monkeypatch):
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="timeout",
        task_type="cpp_docstring_to_code",
        prompt="// timeout\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        compile_suffix="\nint main(){return f();}\n",
        run_binary=True,
    )

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["clang++"], timeout=0.01, stderr="busy")

    monkeypatch.setattr(mod.subprocess, "run", timeout)

    result = mod.compile_cpp_completion(
        prompt,
        "int f() { return 0; }",
        compiler="clang++",
        compile_timeout_s=0.01,
    )

    assert result == {
        "status": "compile_timeout",
        "timeout_s": 0.01,
        "stderr": "busy",
    }


def test_domain_eval_marks_no_compile_as_failure():
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )

    report = mod.evaluate(
        [prompt], {"add": "int add(int a,int b){return a+b;}"}, compile=False
    )

    assert report["rows"][0]["status"] == "not_compiled"
    assert report["failed"] == 1
    assert report["passed"] is False


def test_domain_eval_marks_missing_compile_oracle_as_failure():
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="unchecked",
        task_type="cpp_docstring_to_code",
        prompt="// unchecked\n",
        required_domains=("CPP",),
        expected_sidecars=(),
    )

    report = mod.evaluate([prompt], {"unchecked": "int f(){return 0;}"})

    assert report["rows"][0]["status"] == "missing_compile_oracle"
    assert report["failed"] == 1
    assert report["passed"] is False


def test_domain_eval_marks_compiler_unavailable_as_failure(monkeypatch):
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )
    monkeypatch.setattr(mod, "_find_cpp_compiler", lambda: None)

    report = mod.evaluate([prompt], {"add": "int add(int a,int b){return a+b;}"})

    assert report["rows"][0]["status"] == "compile_oracle_unavailable"
    assert report["failed"] == 1
    assert report["passed"] is False


@pytest.mark.parametrize(
    "status",
    ["compile_timeout", "compile_failed", "runtime_timeout", "runtime_failed"],
)
def test_domain_eval_counts_compile_and_runtime_failures(
    monkeypatch,
    status: str,
) -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )
    monkeypatch.setattr(
        mod,
        "compile_cpp_completion",
        lambda *_args, **_kwargs: {"status": status},
    )

    report = mod.evaluate(
        [prompt],
        {"add": "int add(int a,int b){return a+b;}"},
    )

    assert report["status_counts"] == {status: 1}
    assert report["failed"] == 1
    assert report["passed"] is False


def test_domain_eval_cli_exits_nonzero_for_not_compiled(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    completions = tmp_path / "completions.jsonl"
    report_path = tmp_path / "report.json"
    prompts.write_text(
        json.dumps(
            {
                "id": "add",
                "task_type": "cpp_docstring_to_code",
                "prompt": "// add\n",
                "required_domains": ["CPP"],
                "compile_suffix": "\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
                "run_binary": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completions.write_text(
        json.dumps({"id": "add", "completion": "int add(int a,int b){return a+b;}"})
        + "\n",
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_domain_routed_codegen.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--prompts",
            str(prompts),
            "--completions",
            str(completions),
            "--out",
            str(report_path),
            "--no-compile",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert json.loads(report_path.read_text(encoding="utf-8"))["failed"] == 1
