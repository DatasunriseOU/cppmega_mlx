from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from cppmega_mlx.data.prompt_graph import PromptProjectIndex


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cpp_jsonl_generation_compile_eval.py"
CASE3_FIXTURE = ROOT / "tests" / "fixtures" / "case3_prompt_repo"
CASE3_COMPILE_GATE = ROOT / "scripts" / "cpp_generation_compile_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cpp_jsonl_generation_compile_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_compile_gate_is_repo_local() -> None:
    mod = _load_module()

    assert mod.DEFAULT_COMPILE_GATE == CASE3_COMPILE_GATE
    assert CASE3_COMPILE_GATE.is_file()


def test_prompt_text_modes_use_comment_instruction_and_source_spans():
    mod = _load_module()
    case = {
        "task_id": "x",
        "prompt": "complete it",
        "source_prefix": "int f(){\n",
        "source_suffix": "}\n",
    }
    assert mod.prompt_text(case, "source-prefix") == "int f(){\n"
    assert mod.prompt_text(case, "docstring") == "// complete it\nint f(){\n"
    assert mod.prompt_text(case, "causal-docstring") == (
        "// complete it\nint f(){\n"
    )
    assert mod.prompt_text(case, "fim") == (
        "<FIM_PREFIX>int f(){\n<FIM_SUFFIX>}\n<FIM_MIDDLE>"
    )
    assert mod.prompt_text(case, "ifim") == (
        "<FIM_INSTRUCTION>// complete it\n"
        "<FIM_PREFIX>int f(){\n<FIM_SUFFIX>}\n<FIM_MIDDLE>"
    )


def test_prompt_modes_have_deterministic_exact_token_ids():
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)
    case = {
        "task_id": "add",
        "prompt": "Return the sum.",
        "source_prefix": "int add(int a, int b) {\n",
        "source_suffix": "}\n",
    }
    prefix_ids = tokenizer.encode(case["source_prefix"])
    suffix_ids = tokenizer.encode(case["source_suffix"])
    instruction_ids = tokenizer.encode("// Return the sum.\n")

    assert mod.evaluation_prompt_token_ids(
        tokenizer, mod.evaluation_prompt(case, "causal-docstring")
    ) == tokenizer.encode("// Return the sum.\n" + case["source_prefix"])
    assert mod.evaluation_prompt_token_ids(
        tokenizer, mod.evaluation_prompt(case, "fim")
    ) == [
        tokenizer.fim_prefix_id,
        *prefix_ids,
        tokenizer.fim_suffix_id,
        *suffix_ids,
        tokenizer.fim_middle_id,
    ]
    assert mod.evaluation_prompt_token_ids(
        tokenizer, mod.evaluation_prompt(case, "ifim")
    ) == [
        tokenizer.fim_instruction_id,
        *instruction_ids,
        tokenizer.fim_prefix_id,
        *prefix_ids,
        tokenizer.fim_suffix_id,
        *suffix_ids,
        tokenizer.fim_middle_id,
    ]
    assert instruction_ids != tokenizer.encode(case["prompt"])


def test_default_side_channels_are_token_aligned_zero_arrays():
    mod = _load_module()
    side = mod.default_side_channels(7)
    assert set(side) == set(mod.SIDE_CHANNEL_NAMES)
    for name, tensor in side.items():
        assert tuple(tensor.shape) == (1, 7)
        expected_dtype = (
            mx.uint64
            if name in mod.OPAQUE_ID_SIDE_CHANNEL_NAMES
            else mx.int32
        )
        assert tensor.dtype == expected_dtype
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
    ).build(CASE3_FIXTURE, project_id="tests/case3-prompt-repo").index


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
    assert context.receipt == {
        "prompt_graph_mode": "off",
        "prompt_sidecars": "zero",
        "prompt_mode": "source-prefix",
    }
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
    assert graph_cfg.graph_routes_enabled is True
    assert off_cfg.require_graph_routes is False
    assert off_cfg.graph_routes_enabled is False


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


def test_cli_exposes_fim_modes_and_rejects_unalignable_options():
    mod = _load_module()
    for mode in ("causal-docstring", "fim", "ifim"):
        assert mode in mod.PROMPT_MODE_CHOICES

    with pytest.raises(ValueError, match="require --prompt-sidecars zero"):
        mod.parse_args(
            [
                "--checkpoint",
                "unused.safetensors",
                "--prompt-mode",
                "fim",
                "--prompt-sidecars",
                "clang",
            ]
        )
    with pytest.raises(ValueError, match="cannot use --prepend-code-start"):
        mod.parse_args(
            [
                "--checkpoint",
                "unused.safetensors",
                "--prompt-mode",
                "ifim",
                "--prepend-code-start",
            ]
        )


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
    side, relation_bias, edge_kind_bias, window_receipt = mod.prompt_model_inputs(
        context,
        total_token_count=len(context.token_ids) + 1,
        window_start=0,
        window_end=len(context.token_ids) + 1,
        bias_dtype=mx.float32,
    )

    assert context.graph_artifact is not None
    assert context.receipt["edge_counts"]["call"] > 0
    assert context.receipt["edge_counts"]["type"] > 0
    assert context.receipt["edge_counts"]["def_use"] > 0
    assert context.receipt["edge_counts"]["domain"] > 0
    assert side["symbol_ids"].dtype == mx.uint64
    assert side["call_targets"].dtype == mx.uint64
    assert side["type_refs"].dtype == mx.uint64
    dependency_paths = {
        row["source_path"]
        for row in context.receipt["provenance"]["context_segments"]
        if row["role"] == "dependency"
    }
    assert "include/repo_api.hpp" in dependency_paths
    assert "src/repo_helper.cpp" in dependency_paths
    assert set(context.side_channels) == set(mod.TOKEN_SIDECAR_NAMES)
    assert all(tuple(value.shape) == (1, len(context.token_ids) + 1) for value in side.values())
    assert tuple(relation_bias.shape) == (
        1,
        len(context.token_ids) + 1,
        len(context.token_ids) + 1,
    )
    assert tuple(edge_kind_bias.shape) == tuple(relation_bias.shape)
    assert relation_bias.dtype == mx.float32
    assert edge_kind_bias.dtype == mx.float32
    assert float(mx.sum(relation_bias).item()) > 0.0
    assert float(mx.sum(edge_kind_bias).item()) > 0.0
    assert window_receipt["edge_counts"]["call"] > 0
    assert window_receipt["generated_token_policy"] == (
        "generated_continuation_chunk_with_repository_summary_v1"
    )
    assert side["domain_ids"][0, -1].item() == 1
    assert side["confidence_ids"][0, -1].item() == 1
    assert side["source_doc_ids"][0, -1].item() == 0
    assert side["structure_ids"][0, -1].item() > 0
    assert float(mx.sum(relation_bias[:, -1, :]).item()) > 0.0
    assert window_receipt["relation_bias_nonzero"] > 0
    assert window_receipt["edge_kind_bias_nonzero"] > 0
    assert window_receipt["edge_kind_route_count"] > 0
    assert {"2", "3", "4"} <= set(window_receipt["edge_kind_route_counts"])
    assert window_receipt["graph_bias_dtype"] == "float32"


def test_repo_ifim_prompt_keeps_prefix_and_suffix_source_alignment(
    tmp_path: Path,
):
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)
    case = json.loads((CASE3_FIXTURE / "cases.jsonl").read_text().splitlines()[0])
    prompt = mod.evaluation_prompt(case, "ifim")
    project_index = _build_case3_index(tmp_path)
    source = project_index.document_for_path("src/math_prompt.cpp")

    context = mod.build_prompt_context(
        tokenizer,
        prompt,
        prompt_graph_mode="repo",
        prompt_sidecars="zero",
        prepend_code_start=False,
        project_index=project_index,
        prompt_document_id=source.id,
        prompt_source_path=source.source_path,
        prompt_source_start=0,
        prompt_graph_cache_dir=tmp_path / "graph-cache",
    )

    target_ids = mod.evaluation_prompt_token_ids(tokenizer, prompt)
    target_start = len(context.token_ids) - len(target_ids)
    instruction_ids = tokenizer.encode(prompt.instruction)
    prefix_ids = tokenizer.encode(prompt.source_prefix)
    suffix_ids = tokenizer.encode(prompt.source_suffix)
    prefix_start = target_start + 1 + len(instruction_ids) + 1
    suffix_marker = prefix_start + len(prefix_ids)
    suffix_start = suffix_marker + 1
    source_doc_ids = context.side_channels["source_doc_ids"]

    assert context.token_ids[target_start:] == target_ids
    assert source_doc_ids[target_start : prefix_start] == [0] * (
        prefix_start - target_start
    )
    assert source_doc_ids[prefix_start:suffix_marker] == [source.id] * len(
        prefix_ids
    )
    assert source_doc_ids[suffix_marker] == 0
    assert source_doc_ids[suffix_start : suffix_start + len(suffix_ids)] == [
        source.id
    ] * len(suffix_ids)
    assert source_doc_ids[-1] == 0
    assert context.receipt["prompt_mode"] == "ifim"
    assert context.receipt["edge_counts"]["call"] > 0

    with pytest.raises(ValueError, match="source_suffix.*ambiguous"):
        mod.build_prompt_context(
            tokenizer,
            mod.EvaluationPrompt("fim", prompt.source_prefix, "}"),
            prompt_graph_mode="repo",
            prompt_sidecars="zero",
            prepend_code_start=False,
            project_index=project_index,
            prompt_document_id=source.id,
            prompt_source_path=source.source_path,
            prompt_source_start=0,
            prompt_graph_cache_dir=tmp_path / "ambiguous-cache",
        )


@pytest.mark.parametrize("mode", ["fim", "ifim"])
def test_fim_prompt_modes_reject_unalignable_options(mode: str):
    mod = _load_module()
    prompt = mod.EvaluationPrompt(
        mode=mode,
        source_prefix="int f() {\n",
        source_suffix="}\n",
        instruction="// Return one.\n" if mode == "ifim" else None,
    )

    with pytest.raises(ValueError, match="require --prompt-sidecars zero"):
        mod.build_prompt_context(
            _FakeTokenizer(),
            prompt,
            prompt_sidecars="clang",
            prepend_code_start=False,
        )
    with pytest.raises(ValueError, match="cannot use --prepend-code-start"):
        mod.build_prompt_context(
            _FakeTokenizer(),
            prompt,
            prompt_sidecars="zero",
            prepend_code_start=True,
        )


def test_ifim_prompt_ids_are_threaded_unchanged_into_model():
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)
    prompt = mod.EvaluationPrompt(
        mode="ifim",
        source_prefix="int answer() {\n",
        source_suffix="}\n",
        instruction="// Return forty two.\n",
    )
    expected_ids = mod.evaluation_prompt_token_ids(tokenizer, prompt)

    class CapturingModel:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, **kwargs):
            self.calls.append((input_ids, kwargs))
            return (
                mx.zeros(
                    (1, input_ids.shape[1], tokenizer.vocab_size),
                    dtype=mx.float32,
                ),
                None,
            )

    model = CapturingModel()
    _completion, prompt_tokens, generated_tokens, finish_reason, receipt = mod.generate_completion(
        model,
        tokenizer,
        prompt,
        seq_len=256,
        max_new_tokens=1,
        temperature=0.0,
        top_k=None,
        top_p=None,
        prompt_graph_mode="off",
        prompt_sidecars="zero",
        prepend_code_start=False,
        project_index=None,
        prompt_document_id=None,
        prompt_source_path=None,
        prompt_source_start=None,
        prompt_graph_cache_dir=None,
        graph_bias_dtype=mx.float32,
    )

    assert prompt_tokens == len(expected_ids)
    assert generated_tokens == 1
    assert finish_reason == "length"
    assert len(model.calls) == 1
    input_ids, kwargs = model.calls[0]
    assert input_ids.tolist() == [expected_ids]
    assert kwargs["block_bias"] is None
    assert kwargs["edge_kind_bias"] is None
    assert receipt["prompt_mode"] == "ifim"


def test_generate_completion_emits_eos_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)

    class EosModel:
        def __call__(self, input_ids, **_kwargs):
            return (
                mx.zeros(
                    (1, input_ids.shape[1], tokenizer.vocab_size),
                    dtype=mx.float32,
                ),
                None,
            )

    monkeypatch.setattr(
        mod,
        "_sample_next",
        lambda *_args, **_kwargs: tokenizer.eos_token_id,
    )
    completion, _prompt_tokens, generated_tokens, finish_reason, _receipt = (
        mod.generate_completion(
            EosModel(),
            tokenizer,
            "int answer() {\n",
            seq_len=64,
            max_new_tokens=8,
            temperature=0.0,
            top_k=None,
            top_p=None,
            prompt_graph_mode="off",
            prompt_sidecars="zero",
            prepend_code_start=False,
            project_index=None,
            prompt_document_id=None,
            prompt_source_path=None,
            prompt_source_start=None,
            prompt_graph_cache_dir=None,
            graph_bias_dtype=mx.float32,
        )
    )

    assert completion == ""
    assert generated_tokens == 0
    assert finish_reason == "eos"


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
    _completion, prompt_tokens, generated_tokens, finish_reason, receipt = mod.generate_completion(
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
        graph_bias_dtype=mx.float32,
    )

    assert prompt_tokens > 0
    assert generated_tokens == 1
    assert finish_reason == "length"
    assert receipt["edge_counts"]["call"] > 0
    assert len(model.calls) == 1
    input_ids, kwargs = model.calls[0]
    assert tuple(input_ids.shape) == (1, prompt_tokens)
    assert float(mx.sum(kwargs["block_bias"]).item()) > 0.0
    assert float(mx.sum(kwargs["edge_kind_bias"]).item()) > 0.0
    assert kwargs["block_bias"].dtype == mx.float32
    assert kwargs["edge_kind_bias"].dtype == mx.float32
    assert int(mx.sum(kwargs["structure_ids"]).item()) > 0
    assert tuple(kwargs["document_ids"].shape) == tuple(input_ids.shape)
    assert int(mx.min(kwargs["document_ids"]).item()) == 1
    assert kwargs["edge_kind_bias"] is not None
    assert tuple(kwargs["domain_ids"].shape) == tuple(input_ids.shape)
    assert tuple(kwargs["role_ids"].shape) == tuple(input_ids.shape)
    assert tuple(kwargs["confidence_ids"].shape) == tuple(input_ids.shape)
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
        graph_bias_dtype=mx.float32,
    )

    assert len(model.calls) == 3
    for _input_ids, kwargs in model.calls[1:]:
        assert float(mx.sum(kwargs["block_bias"][:, -1, :]).item()) > 0.0
        assert float(mx.sum(kwargs["edge_kind_bias"]).item()) > 0.0
        assert int(kwargs["structure_ids"][0, -1].item()) > 0


def test_generation_e2e_output_is_distinct_from_gold_fixture(tmp_path: Path):
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)
    cases = mod.load_cases(mod.DEFAULT_CASES)

    class OneStepModel:
        def __init__(self):
            self.graph_bias_presence = []
            self.graph_kind_bias_presence = []
            self.config = SimpleNamespace(graph_routes_enabled=True)

        def __call__(self, input_ids, **kwargs):
            self.graph_bias_presence.append(kwargs["block_bias"] is not None)
            self.graph_kind_bias_presence.append(kwargs["edge_kind_bias"] is not None)
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
        num_samples=1,
        base_seed=31,
        graph_bias_dtype=mx.float32,
    )

    assert model.graph_bias_presence == [True, True]
    assert model.graph_kind_bias_presence == [True, True]
    assert [row["prompt_graph_mode"] for row in rows] == ["repo", "off"]
    assert [row["sample_index"] for row in rows] == [0, 0]
    assert [row["seed"] for row in rows] == [31, 31]
    assert all(row["finish_reason"] == "length" for row in rows)
    assert all(row["length_truncated"] is True for row in rows)
    assert rows[1]["prompt_graph_receipt"]["graph_bias_ablation"] == (
        "explicit_prompt_graph_mode_off"
    )
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


def test_resolve_case_prompt_graph_requires_stable_project_identity(
    tmp_path: Path,
):
    mod = _load_module()
    case = json.loads((CASE3_FIXTURE / "cases.jsonl").read_text().splitlines()[0])
    case.pop("prompt_graph_project_id")

    with pytest.raises(ValueError, match="prompt_graph_project_id.*owner/repo"):
        mod.resolve_case_prompt_graph(
            case,
            cases_dir=CASE3_FIXTURE,
            mode="repo",
            prompt_index_cache_dir=tmp_path / "index-cache",
            indexer_root=ROOT,
        )


def test_resolve_case_prompt_graph_rejects_synthetic_repository_index(
    tmp_path: Path,
):
    mod = _load_module()
    repo = tmp_path / "repo"
    shutil.copytree(CASE3_FIXTURE, repo)
    real_index = _build_case3_index(tmp_path)
    payload = real_index.to_dict()
    payload["provenance"]["producer"] = "test-fixture"
    index_path = tmp_path / "static-index.json"
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    case = json.loads((CASE3_FIXTURE / "cases.jsonl").read_text().splitlines()[0])
    case["prompt_graph_repo"] = "repo"
    case["prompt_graph_index"] = "static-index.json"

    with pytest.raises(ValueError, match="production repository index"):
        mod.resolve_case_prompt_graph(
            case,
            cases_dir=tmp_path,
            mode="repo",
            prompt_index_cache_dir=tmp_path / "index-cache",
            indexer_root=ROOT,
        )


def test_build_prompt_context_rejects_nonproduction_project_index(tmp_path: Path):
    from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    mod = _load_module()
    tokenizer = load_cppmega_tokenizer(mod.DEFAULT_TOKENIZER)
    real_index = _build_case3_index(tmp_path)
    payload = real_index.to_dict()
    payload["provenance"]["producer"] = "test-fixture"
    project_index = PromptProjectIndex.from_dict(payload)
    source = project_index.document_for_path("src/math_prompt.cpp")

    with pytest.raises(ValueError, match="production repository index"):
        mod.build_prompt_context(
            tokenizer,
            _case3_prompt(),
            prompt_graph_mode="repo",
            prompt_sidecars="zero",
            prepend_code_start=False,
            project_index=project_index,
            prompt_document_id=source.id,
            prompt_source_path=source.source_path,
            prompt_source_start=0,
            prompt_graph_cache_dir=tmp_path / "graph-cache",
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
            graph_bias_dtype=mx.float32,
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


def test_fim_generated_middle_is_composed_between_prefix_and_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_module()
    case = {
        "task_id": "answer",
        "language": "cpp",
        "prompt": "Return forty two.",
        "source_prefix": "int answer() {\n",
        "source_suffix": (
            "}\nint main() { return answer() == 42 ? 0 : 1; }\n"
        ),
        "compile_args": ["-std=c++20", "-O0"],
        "timeout_s": 10,
        "prompt_graph_mode": "off",
    }
    cases = tmp_path / "cases.jsonl"
    completions = tmp_path / "completions.jsonl"
    cases.write_text(json.dumps(case) + "\n", encoding="utf-8")

    def fake_generate(*_args, **_kwargs):
        return "return 42;\n", 9, 2, "eos", {"prompt_mode": "fim"}

    monkeypatch.setattr(mod, "generate_completion", fake_generate)
    rows = mod.write_completions(
        [case],
        completions,
        model=object(),
        tokenizer=object(),
        prompt_mode="fim",
        seq_len=64,
        max_new_tokens=8,
        temperature=0.0,
        top_k=None,
        top_p=None,
        prompt_graph_mode="off",
        prompt_sidecars="zero",
        prepend_code_start=False,
        cases_dir=tmp_path,
        prompt_graph_cache_dir=None,
        prompt_index_cache_dir=None,
        num_samples=1,
        base_seed=7,
        graph_bias_dtype=mx.float32,
    )
    spec = importlib.util.spec_from_file_location(
        "cpp_generation_compile_eval_for_fim_test",
        CASE3_COMPILE_GATE,
    )
    assert spec is not None and spec.loader is not None
    compile_gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compile_gate
    spec.loader.exec_module(compile_gate)
    loaded_case = compile_gate.load_cases(cases)["answer"]
    loaded_completion = compile_gate.load_completions(completions)["answer"]

    assert rows[0]["completion"] == "return 42;\n"
    assert rows[0]["prompt_mode"] == "fim"
    assert rows[0]["sample_index"] == 0
    assert rows[0]["seed"] == 7
    assert rows[0]["finish_reason"] == "eos"
    assert rows[0]["length_truncated"] is False
    assert compile_gate.compose_source(loaded_case, loaded_completion) == (
        case["source_prefix"] + "return 42;\n" + case["source_suffix"]
    )


def test_gold_fixture_is_explicit_and_repository_gate_links_all_sources(
    tmp_path: Path,
):
    mod = _load_module()
    gold = CASE3_FIXTURE / "gold_completions.jsonl"
    gold_row = json.loads(gold.read_text(encoding="utf-8"))
    assert gold_row["completion_source"] == "gold_fixture"

    report = tmp_path / "report.json"
    compile_report = mod.run_compile_gate(
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
    assert compile_report == payload
    receipt = payload["compile_gate_receipt"]
    assert receipt["script"]["path"] == str(CASE3_COMPILE_GATE.resolve())
    assert receipt["script"]["git_commit"]
    assert receipt["script"]["sha256"] == hashlib.sha256(
        CASE3_COMPILE_GATE.read_bytes()
    ).hexdigest()
    assert receipt["inputs"]["cases"]["sha256"] == hashlib.sha256(
        (CASE3_FIXTURE / "cases.jsonl").read_bytes()
    ).hexdigest()


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
    assert "causal-docstring" in proc.stdout
    assert "fim" in proc.stdout
    assert "ifim" in proc.stdout


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


def test_graph_bias_dtype_comes_from_loaded_model_weights() -> None:
    mod = _load_module()
    model = SimpleNamespace(
        token_embedding=SimpleNamespace(
            weight=mx.zeros((2, 2), dtype=mx.bfloat16)
        )
    )

    assert mod.model_graph_bias_dtype(model) == mx.bfloat16


def test_compile_gate_env_prepends_tool_dirs(tmp_path: Path):
    mod = _load_module()
    clang_format = tmp_path / "clang-format"
    clang_format.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    clang_format.chmod(0o755)

    env = mod.compile_gate_env({"PATH": ""}, path_dirs=(tmp_path,))

    assert env["PATH"].split(":")[0] == str(tmp_path)


def test_standard_pass_at_k_estimator_and_unavailable_k() -> None:
    mod = _load_module()

    assert mod.estimate_pass_at_k(10, 0, 1) == 0.0
    assert mod.estimate_pass_at_k(10, 10, 10) == 1.0
    assert mod.estimate_pass_at_k(10, 1, 5) == pytest.approx(0.5)
    assert mod.estimate_pass_at_k(4, 1, 5) is None


def test_compile_gate_evaluates_candidate_identity_and_reports_pass_at_k(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    cases = tmp_path / "cases.jsonl"
    completions = tmp_path / "completions.jsonl"
    report = tmp_path / "compile_report.json"
    cases.write_text(
        '{"task_id":"candidate","language":"cpp",'
        '"prompt":"return zero",'
        '"source_prefix":"int candidate() {\\n",'
        '"source_suffix":"}\\nint main() { return candidate(); }\\n"}\n',
        encoding="utf-8",
    )
    completions.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "task_id": "candidate",
                        "sample_index": 0,
                        "seed": 100,
                        "completion": "return 0;",
                    }
                ),
                json.dumps(
                    {
                        "task_id": "candidate",
                        "sample_index": 1,
                        "seed": 101,
                        "completion": "return does_not_exist;",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    payload = mod.run_compile_gate(
        cases=cases,
        completions=completions,
        report=report,
        script=CASE3_COMPILE_GATE,
        keep_workdir=False,
        fail_on_fail=False,
    )

    assert payload["summary"]["total"] == 2
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["pass@1"] == pytest.approx(0.5)
    assert payload["summary"]["pass@5"] is None
    assert payload["summary"]["pass@10"] is None
    assert payload["summary"]["pass_at_k"] == {
        "1": pytest.approx(0.5),
        "5": None,
        "10": None,
    }
    assert {
        (row["task_id"], row["sample_index"], row["seed"])
        for row in payload["results"]
    } == {("candidate", 0, 100), ("candidate", 1, 101)}


def test_cli_defaults_to_ten_candidates_for_standard_pass_at_k() -> None:
    mod = _load_module()
    args = mod.parse_args(["--checkpoint", "unused.safetensors"])

    assert args.num_samples == 10
    assert args.seed == 0
    assert args.temperature == pytest.approx(0.8)

    with pytest.raises(ValueError, match="temperature > 0"):
        mod.parse_args(
            [
                "--checkpoint",
                "unused.safetensors",
                "--num-samples",
                "10",
                "--temperature",
                "0",
            ]
        )
