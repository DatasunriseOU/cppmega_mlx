from __future__ import annotations

import importlib.util
from collections import Counter
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


def _load_generation_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cpp_jsonl_generation_compile_eval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cpp_jsonl_generation_compile_eval_for_domain_tests",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_executable(path: Path, body: str) -> str:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return str(path)


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
                "oracle_kind": "cpp_compile_run",
                "expected_sidecars": ["token_domain_ids", "token_role_ids"],
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
    assert prompts[0].expected_sidecars == ("token_domain_ids", "token_role_ids")


def _case5_ksh_eval_row() -> dict:
    for line in Path("evals/domain_routed_prompts.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        row = json.loads(line)
        if row["id"] == "ksh_build_sidecar_route":
            return row
    raise AssertionError("missing ksh_build_sidecar_route eval fixture")


def _gold_completion_texts(path: Path) -> dict[str, str]:
    """Read gold text only for direct oracle calibration, never for evaluate()."""

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(row.get("completion_source") == "gold_fixture" for row in rows)
    return {str(row["id"]): str(row["completion"]) for row in rows}


def _published_completion(mod, prompt, completion: str):
    generated_token_count = max(1, len(completion.encode("utf-8")))
    generation_receipt = {
        "schema": mod.DOMAIN_GENERATION_RECEIPT_SCHEMA,
        "generated_token_count": generated_token_count,
        "finish_reason": "length",
        "edge_family_route_counts": mod._prompt_edge_counts(prompt),
    }
    row = mod.publish_model_completion(
        prompt,
        completion,
        model_id="test-dense-cpp-lm",
        generation_receipt=generation_receipt,
    )
    return mod.DomainModelCompletion.from_row(row)


def _published_row(mod, prompt, completion: str) -> dict:
    generated_token_count = max(1, len(completion.encode("utf-8")))
    return mod.publish_model_completion(
        prompt,
        completion,
        model_id="test-dense-cpp-lm",
        generation_receipt={
            "schema": mod.DOMAIN_GENERATION_RECEIPT_SCHEMA,
            "generated_token_count": generated_token_count,
            "finish_reason": "length",
            "edge_family_route_counts": mod._prompt_edge_counts(prompt),
        },
    )


def test_domain_eval_freezes_structured_ksh_prompt_sidecars(tmp_path: Path) -> None:
    mod = _load_eval_module()
    row = _case5_ksh_eval_row()
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
    assert report["rows"][0]["static_prompt_sidecar_receipt"] == {
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
    row["prompt_sidecars"]["token_shell_edges"][0]["to"] = len(
        row["prompt_token_ids"]
    )
    path = tmp_path / "bad-ksh-prompts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside prompt tokens"):
        mod.load_prompts(path)


def test_domain_eval_loads_shipped_prompts_with_typed_oracles():
    mod = _load_eval_module()
    path = Path("evals/domain_routed_prompts.jsonl")

    prompts = mod.load_prompts(path)

    assert [prompt.oracle_kind for prompt in prompts] == [
        "cpp_compile_run",
        "cpp_compile_run",
        "build_structure",
        "shell_syntax",
        "sidecar_structure",
    ]


def test_domain_eval_prompt_loader_requires_typed_oracle_contract(tmp_path: Path):
    mod = _load_eval_module()
    row = {
        "id": "p0",
        "task_type": "shell",
        "prompt": "echo ok\n",
        "required_domains": ["SH"],
    }
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="oracle_kind"):
        mod.load_prompts(path)


def test_domain_eval_rejects_oracle_domain_mismatch(tmp_path: Path):
    mod = _load_eval_module()
    row = {
        "id": "p0",
        "task_type": "shell",
        "prompt": "echo ok\n",
        "required_domains": ["SH"],
        "oracle_kind": "build_structure",
        "expected_sidecars": [
            "token_domain_ids",
            "token_role_ids",
            "token_shell_edges",
        ],
    }
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="build_structure oracle requires"):
        mod.load_prompts(path)


def test_domain_eval_evaluate_without_completion_fails_closed():
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=("token_domain_ids",),
        oracle_kind="cpp_compile_run",
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
        oracle_kind="cpp_compile_run",
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
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )

    result = mod.compile_cpp_completion(
        prompt,
        "int add(int, int){ return 0; }",
        compiler=compiler,
    )

    assert result["status"] == "runtime_failed"


def test_domain_eval_reports_compile_timeout_instead_of_raising(tmp_path: Path):
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="timeout",
        task_type="cpp_docstring_to_code",
        prompt="// timeout\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return f();}\n",
        run_binary=True,
    )

    compiler = _write_executable(tmp_path / "slow-compiler", "sleep 1")

    result = mod.compile_cpp_completion(
        prompt,
        "int f() { return 0; }",
        compiler=compiler,
        compile_timeout_s=0.01,
    )

    assert result["status"] == "compile_timeout"
    assert result["timeout_s"] == 0.01


def test_domain_eval_marks_no_compile_as_failure():
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )

    report = mod.evaluate(
        [prompt],
        {"add": _published_completion(mod, prompt, "int add(int a,int b){return a+b;}")},
        compile=False,
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
        oracle_kind="cpp_compile_run",
    )

    report = mod.evaluate(
        [prompt],
        {"unchecked": _published_completion(mod, prompt, "int f(){return 0;}")},
    )

    assert report["rows"][0]["status"] == "missing_compile_oracle"
    assert report["failed"] == 1
    assert report["passed"] is False


def test_domain_eval_marks_compiler_unavailable_as_failure():
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )
    report = mod.evaluate(
        [prompt],
        {
            "add": _published_completion(
                mod,
                prompt,
                "int add(int a,int b){return a+b;",
            )
        },
        tool_overrides={"cpp": None},
    )

    assert report["rows"][0]["status"] == "compile_oracle_unavailable"
    assert report["failed"] == 1
    assert report["passed"] is False


def test_domain_eval_counts_compile_failure() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )
    report = mod.evaluate(
        [prompt],
        {
            "add": _published_completion(
                mod,
                prompt,
                "int add(int a,int b){return ;",
            )
        },
    )

    assert report["status_counts"] == {"compile_failed": 1}
    assert report["failed"] == 1
    assert report["passed"] is False


def test_domain_eval_cli_exits_nonzero_for_not_compiled(
    tmp_path: Path,
) -> None:
    mod = _load_eval_module()
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
                    "oracle_kind": "cpp_compile_run",
                    "expected_sidecars": ["token_domain_ids", "token_role_ids"],
                    "compile_suffix": "\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
                "run_binary": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_for_completion = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=("token_domain_ids", "token_role_ids"),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )
    completions.write_text(
        json.dumps(
            _published_row(
                mod,
                prompt_for_completion,
                "int add(int a,int b){return a+b;}",
            )
        )
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


@pytest.mark.parametrize(
    ("domain", "shell_name", "completion"),
    [
        ("BASH", "bash", "if true; then echo ok; fi\n"),
        ("ZSH", "zsh", "if true; then print ok; fi\n"),
        ("KSH", "ksh", "if true; then print ok; fi\n"),
        ("SH", "sh", "if true; then echo ok; fi\n"),
    ],
)
def test_shell_oracle_checks_declared_dialect(
    domain: str,
    shell_name: str,
    completion: str,
) -> None:
    if shutil.which(shell_name) is None:
        pytest.skip(f"{shell_name} is unavailable")
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id=f"{shell_name}-syntax",
        task_type="shell_syntax",
        prompt=completion,
        required_domains=(domain,),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_shell_edges"),
        oracle_kind="shell_syntax",
    )

    result = mod.shell_syntax_oracle(prompt, completion)

    assert result["status"] == "shell_syntax_passed"
    assert result["shell_kind"] == shell_name


def test_shell_oracle_rejects_invalid_syntax() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="bad-sh",
        task_type="shell_syntax",
        prompt="if true; then echo missing-fi\n",
        required_domains=("SH",),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_shell_edges"),
        oracle_kind="shell_syntax",
    )

    result = mod.shell_syntax_oracle(prompt, "if true; then echo missing-fi\n")

    assert result["status"] in {"shell_parse_failed", "shell_syntax_failed"}


def test_shell_oracle_reports_explicit_unavailability() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="missing-ksh",
        task_type="shell_syntax",
        prompt="print ok\n",
        required_domains=("KSH",),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_shell_edges"),
        oracle_kind="shell_syntax",
    )

    result = mod.shell_syntax_oracle(
        prompt,
        "print ok\n",
        tool_overrides={"ksh": None},
    )

    assert result["status"] == "shell_oracle_unavailable"
    assert result["failed_closed"] is True


@pytest.mark.parametrize(
    ("domain", "completion"),
    [
        ("CMAKE", "add_executable(app main.cpp)\n"),
        ("MAKE", "app: main.o\n\t$(CXX) -o app main.o\n"),
        ("NINJA", "rule cc\n  command = c++ -c $in -o $out\nbuild app.o: cc app.cc\n"),
        ("BAZEL", 'cc_binary(name = "app", srcs = ["main.cc"])\n'),
        ("CONFIGURE", "#!/bin/sh\nCC=cc\nexport CC\n"),
    ],
)
def test_build_structure_oracle_checks_declared_build_domain(
    domain: str,
    completion: str,
) -> None:
    mod = _load_eval_module()
    sidecar = "token_shell_edges" if domain == "CONFIGURE" else "token_build_edges"
    prompt = mod.DomainEvalPrompt(
        id=f"{domain.lower()}-structure",
        task_type="build_structure",
        prompt=completion,
        required_domains=(domain,),
        expected_sidecars=("token_domain_ids", "token_role_ids", sidecar),
        oracle_kind="build_structure",
    )

    result = mod.build_structure_oracle(prompt, completion)

    assert result["status"] == "build_structure_passed"
    assert result["build_kind"] == domain.lower()


def test_build_structure_oracle_rejects_unstructured_text() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="bad-cmake",
        task_type="build_structure",
        prompt="not a cmake command\n",
        required_domains=("CMAKE",),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_build_edges"),
        oracle_kind="build_structure",
    )

    result = mod.build_structure_oracle(prompt, "not a cmake command\n")

    assert result["status"] == "build_structure_failed"


def test_sidecar_structure_oracle_separates_static_prompt_and_completion_checks() -> None:
    mod = _load_eval_module()
    row = _case5_ksh_eval_row()
    path = Path("evals/domain_routed_prompts.jsonl")
    prompt = mod.load_prompts(path)[-1]
    completion = _gold_completion_texts(
        Path("evals/domain_routed_gold_completions.jsonl")
    )[prompt.id]

    result = mod.sidecar_structure_oracle(prompt, completion)

    assert result["status"] == "sidecar_structure_passed"
    assert result["evidence_source"] == "completion_derived"
    assert result["static_prompt_check"]["status"] == (
        "static_prompt_sidecars_passed"
    )
    assert result["static_prompt_check"]["cross_domain_edges"] == 1
    assert result["generated_completion_check"]["status"] == (
        "generated_completion_structure_passed"
    )
    assert result["generated_completion_check"]["generated_cross_domain_edge"][
        "kind"
    ] == "EMBEDDED_DOMAIN"
    assert row["id"] == prompt.id


def test_sidecar_structure_oracle_rejects_garbage_completion() -> None:
    mod = _load_eval_module()
    prompt = mod.load_prompts(Path("evals/domain_routed_prompts.jsonl"))[-1]

    result = mod.sidecar_structure_oracle(prompt, "garbage")

    assert result["status"] == "sidecar_structure_failed"
    assert result["static_prompt_check"]["status"] == (
        "static_prompt_sidecars_passed"
    )
    assert "typed domain blocks" in result["reason"]
    assert "generated_completion_check" not in result


def test_sidecar_structure_oracle_fails_without_frozen_sidecars() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="missing-sidecars",
        task_type="diagnostic",
        prompt="error: bad\n",
        required_domains=("COMPILER_ERROR",),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_diagnostic_edges"),
        oracle_kind="sidecar_structure",
    )

    result = mod.sidecar_structure_oracle(prompt, "add a header")

    assert result["status"] == "missing_sidecar_oracle"
    assert result["failed_closed"] is True


def test_gold_calibration_dispatches_shell_build_and_sidecar_oracles() -> None:
    mod = _load_eval_module()
    prompts = mod.load_prompts(Path("evals/domain_routed_prompts.jsonl"))[2:]
    texts = _gold_completion_texts(
        Path("evals/domain_routed_gold_completions.jsonl")
    )

    results = [
        mod.run_prompt_oracle(prompt, texts[prompt.id]) for prompt in prompts
    ]

    assert [result["status"] for result in results] == [
        "build_structure_passed",
        "shell_syntax_passed",
        "sidecar_structure_passed",
    ]


def test_domain_eval_rejects_gold_completions_as_model_inputs() -> None:
    mod = _load_eval_module()
    with pytest.raises(ValueError, match="gold fixtures are not eval inputs"):
        mod.load_completions(Path("evals/domain_routed_gold_completions.jsonl"))


def test_domain_eval_rejects_unpublished_completion_text() -> None:
    mod = _load_eval_module()
    prompt = mod.load_prompts(Path("evals/domain_routed_prompts.jsonl"))[0]

    report = mod.evaluate([prompt], {prompt.id: "unpublished model text"})

    assert report["rows"][0]["status"] == "completion_contract_failed"
    assert report["rows"][0]["failed_closed"] is True


def _extended_prompts(mod):
    return mod.load_prompts(Path("evals/domain_routed_extended_prompts.jsonl"))


@pytest.mark.parametrize(
    ("prompt_id", "column", "family"),
    [
        ("typed_shell_pipeline", "token_shell_edges", "shell"),
        (
            "typed_compiler_diagnostic",
            "token_diagnostic_edges",
            "diagnostic",
        ),
        (
            "typed_cross_domain_route",
            "token_cross_domain_edges",
            "cross_domain",
        ),
    ],
)
def test_frozen_eval_rejects_each_empty_required_edge_family(
    tmp_path: Path,
    prompt_id: str,
    column: str,
    family: str,
) -> None:
    mod = _load_eval_module()
    rows = [
        json.loads(line)
        for line in Path("evals/domain_routed_extended_prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    row = next(item for item in rows if item["id"] == prompt_id)
    row["prompt_sidecars"][column] = []
    path = tmp_path / f"empty-{family}.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"required frozen edge families are empty.*{family}",
    ):
        mod.load_prompts(path)


def test_extended_gold_fixture_calibrates_four_typed_output_oracles() -> None:
    mod = _load_eval_module()
    prompts = _extended_prompts(mod)
    texts = _gold_completion_texts(
        Path("evals/domain_routed_extended_gold_completions.jsonl")
    )
    statuses = Counter(
        mod.run_prompt_oracle(prompt, texts[prompt.id])["status"]
        for prompt in prompts
    )

    assert dict(statuses) == {
        "build_structure_passed": 1,
        "cross_domain_structure_passed": 1,
        "diagnostic_structure_passed": 1,
        "shell_syntax_passed": 1,
    }


def test_model_completion_receipt_binds_frozen_edge_counts() -> None:
    mod = _load_eval_module()
    prompt = _extended_prompts(mod)[3]
    row = mod.publish_model_completion(
        prompt,
        "model emitted this completion",
        model_id="test-dense-cpp-lm",
        generation_receipt={
            "schema": mod.DOMAIN_GENERATION_RECEIPT_SCHEMA,
            "generated_token_count": 1,
            "finish_reason": "length",
            "edge_family_route_counts": mod._prompt_edge_counts(prompt),
        },
    )
    row["publisher_receipt"]["edge_counts"]["cross_domain"] = 0
    published = mod.DomainModelCompletion.from_row(row)

    report = mod.evaluate([prompt], {prompt.id: published})

    assert report["rows"][0]["status"] == "completion_contract_failed"
    assert report["rows"][0]["failed_closed"] is True
    assert "edge counts" in report["rows"][0]["reason"]


def test_domain_eval_consumes_a_real_dense_cpp_lm_completion() -> None:
    import mlx.core as mx

    from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_eval_module()
    generation = _load_generation_module()
    prompt = _extended_prompts(mod)[3]
    tokenizer = load_cppmega_tokenizer(
        Path("cppmega_mlx/tokenizer/tokenizer.json")
    )
    graph = mod.build_prompt_graph_for_prompt(prompt, tokenizer)
    mx.random.seed(91)
    model = DenseCppLM(
        DenseCppLMConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=16,
            depth=1,
            ffn_hidden_size=32,
            max_seq_length=128,
            num_query_heads=2,
            num_kv_heads=1,
            head_dim=8,
            ngram_hash_enabled=False,
            graph_routes_enabled=True,
            require_graph_routes=True,
            graph_attention_bias_beta=10.0,
        )
    )
    context = generation.GenerationPromptContext(
        token_ids=list(graph.token_ids),
        side_channels={
            name: list(values) for name, values in graph.side_channels.items()
        },
        receipt=dict(graph.receipt),
        graph_artifact=graph,
    )
    (
        completion,
        _prompt_tokens,
        generated_tokens,
        finish_reason,
        generation_receipt,
    ) = generation.generate_completion_from_context(
        model,
        tokenizer,
        context,
        seq_len=128,
        max_new_tokens=2,
        temperature=0.0,
        top_k=None,
        top_p=1.0,
        prompt_graph_mode="repo",
        graph_bias_dtype=mx.float32,
        completion_normalizer=generation.identity_completion,
    )

    assert completion
    assert completion != _gold_completion_texts(
        Path("evals/domain_routed_extended_gold_completions.jsonl")
    )[prompt.id]
    assert generated_tokens > 0
    assert generation_receipt["edge_kind_route_counts"]
    assert {
        family: generation_receipt["edge_family_route_counts"][family]
        for family in ("shell", "diagnostic", "cross_domain")
    } == {
        "shell": 3,
        "diagnostic": 2,
        "cross_domain": 1,
    }
    published = mod.DomainModelCompletion.from_row(
        mod.publish_model_completion(
            prompt,
            completion,
            model_id="tiny-dense-cpp-lm",
            generation_receipt=generation_receipt,
            generated_token_count=generated_tokens,
            finish_reason=finish_reason,
        )
    )
    report = mod.evaluate([prompt], {prompt.id: published})

    assert report["rows"][0]["completion_source"] == "model_generation"
    assert report["rows"][0]["completion_chars"] == len(completion)
    assert report["rows"][0]["status"] in {
        "cross_domain_parse_failed",
        "cross_domain_structure_failed",
    }


def test_extended_prompt_graphs_rebuild_exact_frozen_sidecars() -> None:
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_eval_module()
    tokenizer = load_cppmega_tokenizer(
        Path("cppmega_mlx/tokenizer/tokenizer.json")
    )

    graphs = [
        mod.build_prompt_graph_for_prompt(prompt, tokenizer)
        for prompt in _extended_prompts(mod)
    ]

    assert [graph.receipt["edge_counts"] for graph in graphs] == [
        {"domain": 0, "build": 0, "shell": 3, "diagnostic": 0, "cross_domain": 0},
        {"domain": 0, "build": 0, "shell": 0, "diagnostic": 3, "cross_domain": 0},
        {"domain": 0, "build": 6, "shell": 0, "diagnostic": 0, "cross_domain": 0},
        {"domain": 0, "build": 0, "shell": 3, "diagnostic": 2, "cross_domain": 1},
    ]


def test_shell_oracle_rejects_valid_but_semantically_wrong_command() -> None:
    mod = _load_eval_module()
    prompt = _extended_prompts(mod)[0]

    result = mod.shell_syntax_oracle(prompt, "echo ok\n")

    assert result["status"] == "shell_structure_failed"
    assert "required tokens are absent" in result["reason"]


def test_diagnostic_oracle_rejects_output_without_required_note_edge() -> None:
    mod = _load_eval_module()
    prompt = _extended_prompts(mod)[1]

    result = mod.diagnostic_structure_oracle(
        prompt,
        "src/main.cpp:12:7: error: no matching function for call to 'foo' note\n",
    )

    assert result["status"] == "diagnostic_structure_failed"
    assert "DIAG_NOTE" in result["reason"]


def test_build_oracle_rejects_structured_but_wrong_target() -> None:
    mod = _load_eval_module()
    prompt = _extended_prompts(mod)[2]

    result = mod.build_structure_oracle(
        prompt,
        "add_executable(other other.cpp)\n",
    )

    assert result["status"] == "build_structure_failed"
    assert "required tokens are absent" in result["reason"]


def test_cross_domain_oracle_rejects_missing_diagnostic_block() -> None:
    mod = _load_eval_module()
    prompt = _extended_prompts(mod)[3]

    result = mod.cross_domain_structure_oracle(
        prompt,
        "<KSH_START>print input.txt | tee out.txt\n<KSH_END>",
    )

    assert result["status"] == "cross_domain_structure_failed"
    assert "missing required typed domains" in result["reason"]


def test_cross_domain_oracle_rejects_valid_blocks_without_generated_relation() -> None:
    mod = _load_eval_module()
    prompt = _extended_prompts(mod)[3]

    result = mod.cross_domain_structure_oracle(
        prompt,
        (
            "<KSH_START>print input.txt | tee out.txt\n<KSH_END>"
            "<BUILD_ERROR_START>"
            "ninja: build stopped: subcommand failed with exit code 1\n"
            "<BUILD_ERROR_END>"
        ),
    )

    assert result["status"] == "cross_domain_structure_failed"
    assert "generated cross-domain relation" in result["reason"]
    assert "generated_cross_domain_edge" not in result


def test_extended_domain_eval_cli_rebuilds_graphs_before_scoring_model_output(
    tmp_path: Path,
) -> None:
    mod = _load_eval_module()
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_domain_routed_codegen.py"
    report_path = tmp_path / "extended-report.json"
    completions_path = tmp_path / "model-completions.jsonl"
    prompts = _extended_prompts(mod)
    completions_path.write_text(
        "\n".join(
            json.dumps(
                mod.publish_model_completion(
                    prompt,
                    "model output",
                    model_id="test-dense-cpp-lm",
                    generation_receipt={
                        "schema": mod.DOMAIN_GENERATION_RECEIPT_SCHEMA,
                        "generated_token_count": 1,
                        "finish_reason": "length",
                        "edge_family_route_counts": mod._prompt_edge_counts(prompt),
                    },
                )
            )
            for prompt in prompts
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--prompts",
            "evals/domain_routed_extended_prompts.jsonl",
            "--completions",
            str(completions_path),
            "--out",
            str(report_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["static_prompt_graphs_validated"] == 4
    assert report["oracle_passed"] == 0
    assert report["completion_derived_oracle_passed"] == 0
    assert all(
        row["completion_source"] == "model_generation"
        for row in report["rows"]
    )
    assert report["passed"] is False
