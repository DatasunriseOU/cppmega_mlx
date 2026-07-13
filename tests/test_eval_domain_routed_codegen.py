from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


def _test_cpp_compiler() -> str | None:
    apple_clang = Path("/usr/bin/clang++")
    return str(apple_clang) if apple_clang.exists() else shutil.which("clang++")


def _load_eval_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_domain_routed_codegen.py"
    )
    spec = importlib.util.spec_from_file_location("eval_domain_routed_codegen", module_path)
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
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = mod.load_prompts(path)

    assert len(prompts) == 1
    assert prompts[0].required_domains == ("CPP",)
    assert prompts[0].expected_sidecars == ("token_domain_ids",)


def test_domain_eval_evaluate_without_completion_is_pending(tmp_path):
    mod = _load_eval_module()
    path = Path("evals/domain_routed_prompts.jsonl")

    report = mod.evaluate(mod.load_prompts(path), {}, compile=False)

    assert report["prompts"] >= 4
    assert report["missing_completion"] == report["prompts"]


def test_domain_eval_compile_gate_accepts_simple_cpp_completion():
    mod = _load_eval_module()
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
        compiler=_test_cpp_compiler(),
    )

    assert result["status"] in {"compile_passed", "compile_skipped"}


def test_domain_eval_runtime_oracle_rejects_wrong_compilable_completion():
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

    result = mod.compile_cpp_completion(
        prompt,
        "int add(int, int){ return 0; }",
        compiler=_test_cpp_compiler(),
    )

    assert result["status"] in {"runtime_failed", "compile_skipped"}


def test_domain_eval_reports_compile_timeout_instead_of_raising(monkeypatch):
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="timeout",
        task_type="cpp_docstring_to_code",
        prompt="// timeout\n",
        required_domains=("CPP",),
        expected_sidecars=(),
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
