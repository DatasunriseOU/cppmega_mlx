from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cpp_jsonl_generation_compile_eval.py"
CASE3_FIXTURE = ROOT / "tests" / "fixtures" / "case3_prompt_repo"
CASE3_COMPILE_GATE = (
    ROOT.parent
    / "cppmega_case3_prompt"
    / "scripts"
    / "cpp_generation_compile_eval.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("cpp_jsonl_generation_compile_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_text_source_prefix_and_docstring():
    mod = _load_module()
    case = {
        "task_id": "x",
        "prompt": "complete it",
        "source_prefix": "int f(){\n",
    }
    assert mod.prompt_text(case, "source-prefix") == "int f(){\n"
    assert mod.prompt_text(case, "docstring") == "int f(){\n"


def test_default_side_channels_are_token_aligned_zero_arrays():
    mod = _load_module()
    side = mod.default_side_channels(7)
    assert set(side) == set(mod.SIDE_CHANNEL_NAMES)
    for tensor in side.values():
        assert tuple(tensor.shape) == (1, 7)
        assert tensor.dtype == mx.int32
        assert int(mx.sum(tensor).item()) == 0


class _FakeTokenizer:
    code_start_id = 7

    def encode(self, text: str, prepend: int | None = None):
        ids = [101, 102, len(text)]
        return ([prepend] if prepend is not None else []) + ids


def _case3_prompt() -> str:
    row = json.loads((CASE3_FIXTURE / "cases.jsonl").read_text().splitlines()[0])
    return row["source_prefix"]


def _build_case3_index(tmp_path: Path):
    from cppmega_mlx.data.prompt_graph_index import ClangPromptProjectIndexProducer

    return ClangPromptProjectIndexProducer(
        cache_dir=tmp_path / "project-index-cache",
        indexer_root=ROOT,
    ).build(CASE3_FIXTURE, project_id="case3_prompt_repo").index


def test_build_prompt_context_zero_sidecars_align_with_prepend():
    mod = _load_module()
    context = mod.build_prompt_context(
        _FakeTokenizer(),
        "int f() {",
        prompt_graph_mode="off",
        prompt_sidecars="zero",
        prepend_code_start=True,
    )

    assert context.token_ids[0] == _FakeTokenizer.code_start_id
    assert context.receipt == {"prompt_graph_mode": "off", "prompt_sidecars": "zero"}
    assert context.graph_artifact is None
    assert set(context.side_channels) == set(mod.SIDE_CHANNEL_NAMES)
    assert all(
        len(values) == len(context.token_ids)
        for values in context.side_channels.values()
    )
    assert all(sum(values) == 0 for values in context.side_channels.values())


def test_build_model_config_requires_graph_routes_only_in_repo_mode():
    mod = _load_module()
    base = {
        "vocab_size": 128,
        "hidden": 32,
        "depth": 1,
        "ffn": 64,
        "num_query_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 8,
        "seq_len": 64,
        "attention_mode": "dsa",
        "structure_components": "all",
        "structure_num_categories": 9,
        "structure_max_dep_level": 64,
        "structure_bottleneck_dim": 16,
        "disable_ngram": True,
        "ngram_hash_heads": 1,
        "ngram_hash_table_size": 64,
        "ngram_hash_embed_dim": 4,
    }

    graph_cfg = mod.build_model_config(
        SimpleNamespace(**base, prompt_graph_mode="repo"), {}
    )
    off_cfg = mod.build_model_config(
        SimpleNamespace(**base, prompt_graph_mode="off"), {}
    )
    assert graph_cfg.require_graph_routes is True
    assert off_cfg.require_graph_routes is False


def test_shipped_default_cases_exercise_mixed_per_case_graph_contract():
    mod = _load_module()
    args = mod.parse_args(["--checkpoint", "unused.safetensors"])
    cases = mod.load_cases(args.cases)
    modes = [
        mod.effective_case_prompt_graph_mode(case, args.prompt_graph_mode)
        for case in cases
    ]

    assert args.cases == mod.DEFAULT_CASES
    assert set(modes) == {"repo", "off"}
    assert mod.batch_requires_graph_routes(modes) is False
    assert mod.batch_requires_graph_routes(["repo"]) is True
    assert mod.batch_requires_graph_routes(["off"]) is False


def test_build_prompt_context_repo_mode_consumes_project_index(tmp_path: Path):
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)
    project_index = _build_case3_index(tmp_path)
    source = project_index.document_for_path("src/math_prompt.cpp")

    context = mod.build_prompt_context(
        tokenizer,
        _case3_prompt(),
        prompt_graph_mode="repo",
        prompt_sidecars="zero",
        prepend_code_start=False,
        project_index=project_index,
        prompt_document_id=source.id,
        prompt_source_path=source.source_path,
        prompt_source_start=0,
        prompt_graph_cache_dir=tmp_path,
    )
    side, block_bias, window_receipt = mod.prompt_model_inputs(
        context,
        total_token_count=len(context.token_ids) + 1,
        window_start=0,
        window_end=len(context.token_ids) + 1,
    )

    assert context.graph_artifact is not None
    assert context.receipt["edge_counts"]["call"] > 0
    assert context.receipt["edge_counts"]["type"] > 0
    assert context.receipt["edge_counts"]["def_use"] > 0
    assert context.receipt["edge_counts"]["domain"] > 0
    dependency_paths = {
        row["source_path"]
        for row in context.receipt["provenance"]["context_segments"]
        if row["role"] == "dependency"
    }
    assert "include/repo_api.hpp" in dependency_paths
    assert "src/repo_helper.cpp" in dependency_paths
    assert set(context.side_channels) == set(mod.TOKEN_SIDECAR_NAMES)
    assert all(tuple(value.shape) == (1, len(context.token_ids) + 1) for value in side.values())
    assert tuple(block_bias.shape) == (
        1,
        len(context.token_ids) + 1,
        len(context.token_ids) + 1,
    )
    assert float(mx.sum(block_bias).item()) > 0.0
    assert window_receipt["edge_counts"]["call"] > 0
    assert window_receipt["generated_token_policy"] == (
        "generated_continuation_chunk_with_repository_summary_v1"
    )
    assert side["domain_ids"][0, -1].item() == 1
    assert side["confidence_ids"][0, -1].item() == 1
    assert side["source_doc_ids"][0, -1].item() == 0
    assert side["structure_ids"][0, -1].item() > 0
    assert float(mx.sum(block_bias[:, -1, :]).item()) > 0.0


def test_generate_completion_threads_repository_graph_into_model(tmp_path: Path):
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)
    project_index = _build_case3_index(tmp_path)
    source = project_index.document_for_path("src/math_prompt.cpp")

    class CapturingModel:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, **kwargs):
            self.calls.append((input_ids, kwargs))
            logits = mx.zeros(
                (1, input_ids.shape[1], tokenizer.vocab_size),
                dtype=mx.float32,
            )
            return logits, None

    model = CapturingModel()
    _completion, prompt_tokens, generated_tokens, receipt = mod.generate_completion(
        model,
        tokenizer,
        _case3_prompt(),
        seq_len=1024,
        max_new_tokens=1,
        temperature=0.0,
        top_k=None,
        top_p=None,
        prompt_graph_mode="repo",
        prompt_sidecars="zero",
        prepend_code_start=False,
        project_index=project_index,
        prompt_document_id=source.id,
        prompt_source_path=source.source_path,
        prompt_source_start=0,
        prompt_graph_cache_dir=tmp_path,
    )

    assert prompt_tokens > 0
    assert generated_tokens == 1
    assert receipt["edge_counts"]["call"] > 0
    assert len(model.calls) == 1
    input_ids, kwargs = model.calls[0]
    assert tuple(input_ids.shape) == (1, prompt_tokens)
    assert float(mx.sum(kwargs["block_bias"]).item()) > 0.0
    assert int(mx.sum(kwargs["structure_ids"]).item()) > 0
    assert receipt["edge_counts"]["type"] > 0
    assert receipt["edge_counts"]["def_use"] > 0
    assert receipt["edge_counts"]["domain"] > 0


def test_generate_completion_keeps_graph_sensitive_after_token_one(tmp_path: Path):
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)
    project_index = _build_case3_index(tmp_path)
    source = project_index.document_for_path("src/math_prompt.cpp")

    class CapturingModel:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, **kwargs):
            self.calls.append((input_ids, kwargs))
            logits = mx.zeros(
                (1, input_ids.shape[1], tokenizer.vocab_size),
                dtype=mx.float32,
            )
            return logits, None

    model = CapturingModel()
    mod.generate_completion(
        model,
        tokenizer,
        _case3_prompt(),
        seq_len=1024,
        max_new_tokens=3,
        temperature=0.0,
        top_k=None,
        top_p=None,
        prompt_graph_mode="repo",
        prompt_sidecars="zero",
        prepend_code_start=False,
        project_index=project_index,
        prompt_document_id=source.id,
        prompt_source_path=source.source_path,
        prompt_source_start=0,
        prompt_graph_cache_dir=tmp_path / "graph-cache",
    )

    assert len(model.calls) == 3
    for _input_ids, kwargs in model.calls[1:]:
        assert float(mx.sum(kwargs["block_bias"][:, -1, :]).item()) > 0.0
        assert int(kwargs["structure_ids"][0, -1].item()) > 0


def test_generation_e2e_output_is_distinct_from_gold_fixture(tmp_path: Path):
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)
    cases = mod.load_cases(mod.DEFAULT_CASES)

    class OneStepModel:
        def __init__(self):
            self.graph_bias_presence = []

        def __call__(self, input_ids, **kwargs):
            self.graph_bias_presence.append(kwargs["block_bias"] is not None)
            return (
                mx.zeros(
                    (1, input_ids.shape[1], tokenizer.vocab_size),
                    dtype=mx.float32,
                ),
                None,
            )

    model = OneStepModel()
    completions = tmp_path / "generated.jsonl"
    rows = mod.write_completions(
        cases,
        completions,
        model=model,
        tokenizer=tokenizer,
        prompt_mode="source-prefix",
        seq_len=1024,
        max_new_tokens=1,
        temperature=0.0,
        top_k=None,
        top_p=1.0,
        prompt_graph_mode="repo",
        prompt_sidecars="zero",
        prepend_code_start=False,
        cases_dir=mod.DEFAULT_CASES.parent,
        prompt_graph_cache_dir=tmp_path / "graph-cache",
        prompt_index_cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    )

    assert model.graph_bias_presence == [True, False]
    assert [row["prompt_graph_mode"] for row in rows] == ["repo", "off"]
    assert all(row["completion_source"] == "model_generation" for row in rows)
    written = [json.loads(line) for line in completions.read_text().splitlines()]
    assert all(row["completion_source"] == "model_generation" for row in written)


def test_resolve_case_prompt_graph_builds_real_index_when_path_absent(
    tmp_path: Path,
):
    mod = _load_module()
    case = json.loads((CASE3_FIXTURE / "cases.jsonl").read_text().splitlines()[0])

    resolved = mod.resolve_case_prompt_graph(
        case,
        cases_dir=CASE3_FIXTURE,
        mode="repo",
        prompt_index_cache_dir=tmp_path / "index-cache",
        indexer_root=ROOT,
    )

    assert resolved is not None
    assert resolved.index_path.is_file()
    assert resolved.index_receipt["producer"] == "ClangPromptProjectIndexProducer"
    assert resolved.project_index.document_for_path("src/math_prompt.cpp").id == (
        resolved.document_id
    )


def test_generate_completion_refuses_to_discard_repository_graph_window(
    tmp_path: Path,
):
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)
    project_index = _build_case3_index(tmp_path)
    source = project_index.document_for_path("src/math_prompt.cpp")

    class ModelMustNotRun:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("decode started before graph window validation")

    with pytest.raises(ValueError, match="discard.*repository graph"):
        mod.generate_completion(
            ModelMustNotRun(),
            tokenizer,
            _case3_prompt(),
            seq_len=1024,
            max_new_tokens=1024,
            temperature=0.0,
            top_k=None,
            top_p=None,
            prompt_graph_mode="repo",
            prompt_sidecars="zero",
            prepend_code_start=False,
            project_index=project_index,
            prompt_document_id=source.id,
            prompt_source_path=source.source_path,
            prompt_source_start=0,
            prompt_graph_cache_dir=tmp_path,
        )


def test_resolve_case_prompt_graph_fails_closed_when_requested(tmp_path: Path):
    mod = _load_module()
    case = {"task_id": "missing", "source_prefix": "int f() {\n"}

    with pytest.raises(ValueError, match="missing.*prompt_graph_repo"):
        mod.resolve_case_prompt_graph(
            case,
            cases_dir=tmp_path,
            mode="repo",
            prompt_index_cache_dir=tmp_path / "cache",
            indexer_root=ROOT,
        )

    assert mod.resolve_case_prompt_graph(
        case,
        cases_dir=tmp_path,
        mode="off",
        prompt_index_cache_dir=tmp_path / "cache",
        indexer_root=ROOT,
    ) is None


def test_build_prompt_context_rejects_clang_sidecars_with_prepended_code_start():
    mod = _load_module()
    with pytest.raises(ValueError, match="prepend-code-start"):
        mod.build_prompt_context(
            _FakeTokenizer(),
            "int f() {",
            prompt_sidecars="clang",
            prepend_code_start=True,
        )


def test_build_prompt_context_clang_sidecars_are_fail_closed_when_available():
    mod = _load_module()
    tokenizer = pytest.importorskip("cppmega_mlx.tokenizer.cpp_tokenizer")
    adapter = mod.get_builtin_code_metadata_adapter("cpp")
    capabilities = adapter.probe({"language": "cpp"})
    if not capabilities.available:
        pytest.skip(f"clang adapter unavailable: {capabilities.reason}")

    context = mod.build_prompt_context(
        tokenizer.load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER),
        "int f() {\n  return 1;\n}\n",
        prompt_graph_mode="off",
        prompt_sidecars="clang",
        prepend_code_start=False,
    )

    assert context.token_ids
    assert set(context.side_channels) == set(mod.SIDE_CHANNEL_NAMES)
    assert all(
        len(values) == len(context.token_ids)
        for values in context.side_channels.values()
    )
    assert context.receipt.get("adapter") == "cpp:clang-ast-v1"
    assert not any("dropped" in str(value) for value in context.receipt.values())


def test_body_decode_constraints_ban_specials_and_degenerate_token_run():
    mod = _load_module()

    class Tok:
        vocab_size = 64
        bos_token_id = 2
        eos_token_id = 3
        fim_prefix_id = 4
        fim_middle_id = 5
        fim_suffix_id = 6
        code_start_id = 7
        code_end_id = 8
        think_start_id = 9
        think_end_id = 10
        query_tool_id = 11
        tool_result_id = 19

        @staticmethod
        def token_for_id(token_id):
            return "<RESERVED_48>" if token_id == 48 else None

    constraints = mod.BodyDecodeConstraints(Tok(), prompt_len=2, max_token_run=4)
    logits = mx.zeros((1, 64), dtype=mx.float32)
    tokens = mx.array([[100, 101, 42, 42, 42, 42]], dtype=mx.int32)

    masked = constraints(logits, tokens)

    assert float(masked[0, 42].item()) == float("-inf")
    assert float(masked[0, Tok.code_start_id].item()) == float("-inf")
    assert float(masked[0, Tok.fim_prefix_id].item()) == float("-inf")
    assert float(masked[0, 48].item()) == float("-inf")


def test_trim_body_completion_preserves_nested_blocks_and_trailing_statements():
    mod = _load_module()
    completion = """if (value < lo) {
    value = lo;
}
// A brace in a comment does not close the function: }
const char* marker = "}";
return value;
}
int main() { return 0; }
"""

    assert mod.trim_body_completion(completion) == """if (value < lo) {
    value = lo;
}
// A brace in a comment does not close the function: }
const char* marker = "}";
return value;
"""


def test_trim_body_completion_ignores_braces_in_raw_strings():
    mod = _load_module()
    completion = 'auto text = R"tag(})tag";\nreturn text.size();\n}\n'

    assert mod.trim_body_completion(completion) == (
        'auto text = R"tag(})tag";\nreturn text.size();\n'
    )


def test_local_compile_gate_fails_closed_for_failed_candidate(tmp_path: Path):
    mod = _load_module()
    cases = tmp_path / "cases.jsonl"
    completions = tmp_path / "completions.jsonl"
    report = tmp_path / "compile_report.json"
    cases.write_text(
        '{"task_id":"broken","language":"cpp",'
        '"prompt":"return a value",'
        '"source_prefix":"int broken() {\\n",'
        '"source_suffix":"}\\nint main() { return broken(); }\\n"}\n'
    )
    completions.write_text(
        '{"task_id":"broken","completion":"return does_not_exist;"}\n'
    )

    with pytest.raises(subprocess.CalledProcessError):
        mod.run_compile_gate(
            cases=cases,
            completions=completions,
            report=report,
            script=CASE3_COMPILE_GATE,
            keep_workdir=False,
        )

    assert json.loads(report.read_text())["summary"]["passed"] == 0


def test_gold_fixture_is_explicit_and_repository_gate_links_all_sources(
    tmp_path: Path,
):
    mod = _load_module()
    gold = CASE3_FIXTURE / "gold_completions.jsonl"
    gold_row = json.loads(gold.read_text(encoding="utf-8"))
    assert gold_row["completion_source"] == "gold_fixture"

    report = tmp_path / "report.json"
    mod.run_compile_gate(
        cases=CASE3_FIXTURE / "cases.jsonl",
        completions=gold,
        report=report,
        script=CASE3_COMPILE_GATE,
        keep_workdir=False,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["repository_cases"] == 1
    assert payload["results"][0]["linked_sources"] == [
        "src/math_prompt.cpp",
        "src/repo_helper.cpp",
        "src/repo_caller.cpp",
    ]


def test_script_help_bootstraps_repo_root_from_sibling_cwd():
    sibling_cppmega = ROOT.parent / "cppmega"
    if not sibling_cppmega.is_dir():
        pytest.skip("sibling cppmega checkout not present")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=sibling_cppmega,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--prompt-sidecars" in proc.stdout
    assert "--prompt-graph-mode" in proc.stdout


def test_rejects_megatron_distcp_file(tmp_path: Path):
    mod = _load_module()
    ckpt = tmp_path / "__0_0.distcp"
    ckpt.write_bytes(b"not used")
    with pytest.raises(ValueError, match="Megatron torch_dist"):
        mod.reject_unsupported_checkpoint(ckpt)


def test_rejects_megatron_checkpoint_directory(tmp_path: Path):
    mod = _load_module()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("5000\n")
    with pytest.raises(ValueError, match="Megatron torch_dist"):
        mod.reject_unsupported_checkpoint(tmp_path)


def test_checkpoint_model_config_loads_safetensors_sidecar(tmp_path: Path):
    mod = _load_module()
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"not a real checkpoint")
    (tmp_path / "model.json").write_text(
        json.dumps({"config": {"structure_components": "core", "hidden_size": 1280}}),
        encoding="utf-8",
    )

    assert mod.checkpoint_model_config(ckpt) == {
        "structure_components": "core",
        "hidden_size": 1280,
    }


def test_compile_gate_env_prepends_tool_dirs(tmp_path: Path):
    mod = _load_module()
    clang_format = tmp_path / "clang-format"
    clang_format.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    clang_format.chmod(0o755)

    env = mod.compile_gate_env({"PATH": ""}, path_dirs=(tmp_path,))

    assert env["PATH"].split(":")[0] == str(tmp_path)
