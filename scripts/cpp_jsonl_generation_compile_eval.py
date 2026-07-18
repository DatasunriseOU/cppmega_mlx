#!/usr/bin/env python3
"""Generate C/C++ completions from a local MLX checkpoint and run compile gate.

This is the MLX-side counterpart of the Nebius/H200 Megatron generation wrapper:
it consumes the same ``cpp_docstring_compile_cases.jsonl`` shape, writes
``completions.jsonl``, then invokes this repository's compile/run gate.

Only native MLX DenseCppLM checkpoints are supported here. Megatron ``torch_dist``
checkpoints must be converted before this script can evaluate them locally.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

from cppmega_mlx.data.prompt_graph import (  # noqa: E402
    PromptGraphArtifact,
    PromptGraphBuilder,
    PromptGraphContext,
    PromptGraphSegment,
    PromptProjectIndex,
    GENERATED_TOKEN_SIDECAR_DEFAULTS,
    TOKEN_SIDECAR_DEFAULTS,
    TOKEN_SIDECAR_NAMES,
    require_prompt_graph_project_id,
)
from cppmega_mlx.data.prompt_graph_index import ClangPromptProjectIndexProducer  # noqa: E402
from cppmega_mlx.data.fim import FIMSpecialTokenIds  # noqa: E402
from cppmega_mlx.inference.infilling import build_fim_prompt_ids  # noqa: E402
from cppmega_mlx.inference.sampling import sample_next_token  # noqa: E402
from cppmega_mlx.inference.side_channels import (  # noqa: E402
    InferenceSideChannelBuilder,
    get_builtin_code_metadata_adapter,
)
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig  # noqa: E402
from cppmega_mlx.nn.domain_graph_routes import DEFAULT_EDGE_WEIGHTS  # noqa: E402
from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer  # noqa: E402
from cppmega_mlx.training.checkpoint import load_checkpoint  # noqa: E402

PromptMode = Literal[
    "source-prefix",
    "docstring",
    "causal-docstring",
    "fim",
    "ifim",
]
CanonicalPromptMode = Literal["source-prefix", "causal-docstring", "fim", "ifim"]
PromptSidecars = Literal["zero", "clang"]
PromptGraphMode = Literal["off", "repo"]

PROMPT_MODE_CHOICES: tuple[PromptMode, ...] = (
    "source-prefix",
    "docstring",
    "causal-docstring",
    "fim",
    "ifim",
)
FIM_PROMPT_MODES = frozenset({"fim", "ifim"})
FIM_PREFIX_TOKEN = "<FIM_PREFIX>"
FIM_MIDDLE_TOKEN = "<FIM_MIDDLE>"
FIM_SUFFIX_TOKEN = "<FIM_SUFFIX>"
FIM_INSTRUCTION_TOKEN = "<FIM_INSTRUCTION>"

DEFAULT_TOKENIZER = REPO_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
DEFAULT_CASES = REPO_ROOT / "evals" / "cpp_generation_cases.jsonl"
DEFAULT_COMPILE_GATE = REPO_ROOT / "scripts" / "cpp_generation_compile_eval.py"
MODEL_SIDE_CHANNEL_NAMES = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
    "domain_ids",
    "role_ids",
    "confidence_ids",
)
SIDE_CHANNEL_NAMES = TOKEN_SIDECAR_NAMES
OPAQUE_ID_SIDE_CHANNEL_NAMES = frozenset(
    {"symbol_ids", "call_targets", "type_refs"}
)
DEFAULT_COMPILE_PATH_DIRS = (
    Path("/opt/homebrew/opt/llvm/bin"),
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path("/usr/bin"),
    Path("/bin"),
    Path("/usr/sbin"),
    Path("/sbin"),
)
STANDARD_PASS_AT_K = (1, 5, 10)
COMPILE_GATE_RECEIPT_SCHEMA = "cppmega_mlx_compile_gate_receipt_v1"
COMPILE_REPORT_SCHEMA = "cppmega_mlx_candidate_compile_report_v1"


class GenerationPromptContext:
    __slots__ = ("token_ids", "side_channels", "receipt", "graph_artifact")

    def __init__(
        self,
        *,
        token_ids: list[int],
        side_channels: dict[str, list[int]],
        receipt: dict[str, Any],
        graph_artifact: PromptGraphArtifact | None = None,
    ) -> None:
        self.token_ids = token_ids
        self.side_channels = side_channels
        self.receipt = receipt
        self.graph_artifact = graph_artifact


class EvaluationPrompt(NamedTuple):
    mode: CanonicalPromptMode
    source_prefix: str
    source_suffix: str = ""
    instruction: str | None = None

    @property
    def is_fim(self) -> bool:
        return self.mode in FIM_PROMPT_MODES

    def render(self) -> str:
        if self.mode == "source-prefix":
            return self.source_prefix
        if self.mode == "causal-docstring":
            assert self.instruction is not None
            return self.instruction + self.source_prefix
        fim = (
            FIM_PREFIX_TOKEN
            + self.source_prefix
            + FIM_SUFFIX_TOKEN
            + self.source_suffix
            + FIM_MIDDLE_TOKEN
        )
        if self.mode == "ifim":
            assert self.instruction is not None
            return FIM_INSTRUCTION_TOKEN + self.instruction + fim
        return fim


class ResolvedPromptGraph:
    __slots__ = (
        "project_index",
        "document_id",
        "source_path",
        "source_start",
        "repository_root",
        "index_path",
        "index_receipt",
    )

    def __init__(
        self,
        *,
        project_index: PromptProjectIndex,
        document_id: int,
        source_path: str,
        source_start: int,
        repository_root: Path,
        index_path: Path,
        index_receipt: dict[str, Any],
    ) -> None:
        self.project_index = project_index
        self.document_id = document_id
        self.source_path = source_path
        self.source_start = source_start
        self.repository_root = repository_root
        self.index_path = index_path
        self.index_receipt = index_receipt


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            yield row


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = list(iter_jsonl(path))
    if not cases:
        raise ValueError(f"no cases in {path}")
    seen: set[str] = set()
    for row in cases:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{path}: every case needs non-empty task_id")
        if task_id in seen:
            raise ValueError(f"{path}: duplicate task_id {task_id!r}")
        seen.add(task_id)
    return cases


def _is_cpp_comment(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("//"):
        return all(
            not line.strip() or line.lstrip().startswith("//")
            for line in stripped.splitlines()
        )
    return stripped.startswith("/*") and stripped.endswith("*/")


def _require_case_text(case: dict[str, Any], key: str) -> str:
    value = case.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"case {case.get('task_id')!r}: missing non-empty {key}")
    return value


def cpp_comment_instruction(case: dict[str, Any]) -> str:
    """Render task intent as syntax the C/C++ objective actually trains on."""

    intent = " ".join(_require_case_text(case, "prompt").split())
    if not intent:
        raise ValueError(
            f"case {case.get('task_id')!r}: prompt is empty after normalization"
        )
    return f"// {intent}\n"


def evaluation_prompt(
    case: dict[str, Any],
    mode: PromptMode,
) -> EvaluationPrompt:
    source_prefix = _require_case_text(case, "source_prefix")
    source_suffix = case.get("source_suffix", "")
    if not isinstance(source_suffix, str):
        raise ValueError(
            f"case {case.get('task_id')!r}: source_suffix must be a string"
        )
    canonical_mode: CanonicalPromptMode
    if mode == "docstring":
        canonical_mode = "causal-docstring"
    elif mode in {"source-prefix", "causal-docstring", "fim", "ifim"}:
        canonical_mode = mode
    else:
        raise ValueError(f"unsupported prompt mode {mode!r}")
    instruction = (
        cpp_comment_instruction(case)
        if canonical_mode in {"causal-docstring", "ifim"}
        else None
    )
    return EvaluationPrompt(
        canonical_mode,
        source_prefix,
        source_suffix,
        instruction,
    )


def prompt_text(case: dict[str, Any], mode: PromptMode) -> str:
    return evaluation_prompt(case, mode).render()


def _resolve_contained_path(
    root: Path,
    raw_path: Any,
    *,
    where: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{where} must be a non-empty relative path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{where} must be a contained relative path")
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{where} escapes {root}: {raw_path!r}") from exc
    return resolved


def effective_case_prompt_graph_mode(
    case: dict[str, Any],
    global_mode: PromptGraphMode,
) -> PromptGraphMode:
    if global_mode not in {"repo", "off"}:
        raise ValueError(f"unsupported prompt graph mode {global_mode!r}")
    task_id = str(case.get("task_id") or "<unknown>")
    case_mode = case.get("prompt_graph_mode")
    if case_mode not in {"repo", "off"}:
        raise ValueError(
            f"case {task_id!r}: prompt_graph_mode must be explicitly "
            "'repo' or 'off'"
        )
    if global_mode == "off":
        return "off"
    return case_mode


def batch_requires_graph_routes(
    case_modes: Iterable[PromptGraphMode],
) -> bool:
    """Use the model-global assertion only when every case requires routes."""
    modes = tuple(case_modes)
    return bool(modes) and all(mode == "repo" for mode in modes)


def resolve_case_prompt_graph(
    case: dict[str, Any],
    *,
    cases_dir: Path,
    mode: PromptGraphMode,
    prompt_index_cache_dir: Path | None = None,
    indexer_root: Path | None = None,
) -> ResolvedPromptGraph | None:
    if mode == "off":
        return None
    if mode != "repo":
        raise ValueError(f"unsupported prompt graph mode {mode!r}")

    task_id = str(case.get("task_id") or "<unknown>")
    raw_index = case.get("prompt_graph_index")
    raw_repo = case.get("prompt_graph_repo")
    raw_project_id = case.get("prompt_graph_project_id")
    expected_project_id = (
        None
        if raw_project_id is None
        else require_prompt_graph_project_id(
            raw_project_id,
            where=f"case {task_id!r} prompt_graph_project_id",
        )
    )
    if raw_repo is None:
        if isinstance(raw_index, str) and raw_index:
            repository_root = _resolve_contained_path(
                cases_dir,
                str(Path(raw_index).parent or "."),
                where=f"case {task_id!r} prompt graph index repository",
            )
        else:
            raise ValueError(
                f"case {task_id!r}: missing non-empty prompt_graph_repo "
                "while prompt graph mode is repo"
            )
    else:
        repository_root = _resolve_contained_path(
            cases_dir,
            raw_repo,
            where=f"case {task_id!r} prompt_graph_repo",
        )

    if isinstance(raw_index, str) and raw_index:
        index_path = _resolve_contained_path(
            cases_dir,
            raw_index,
            where=f"case {task_id!r} prompt_graph_index",
        )
        if not index_path.is_file():
            raise FileNotFoundError(
                f"case {task_id!r}: prompt graph index not found: {index_path}"
            )
        project_index = PromptProjectIndex.from_json_path(index_path)
        project_index.validate_production_repository_index(
            expected_project_id=expected_project_id,
            expected_indexer_root=indexer_root or REPO_ROOT,
        )
        project_index.verify_repository(repository_root)
        index_receipt = dict(project_index.provenance)
    else:
        if prompt_index_cache_dir is None:
            raise ValueError(
                f"case {task_id!r}: automatic prompt graph index build requires "
                "a deterministic prompt_index_cache_dir"
            )
        if expected_project_id is None:
            raise ValueError(
                f"case {task_id!r} prompt_graph_project_id must be stable owner/repo"
            )
        built = ClangPromptProjectIndexProducer(
            cache_dir=prompt_index_cache_dir,
            indexer_root=indexer_root,
            strict_diagnostics=True,
        ).build(repository_root, project_id=expected_project_id)
        project_index = built.index
        index_path = built.path
        index_receipt = dict(built.receipt)

    raw_source_path = case.get("prompt_source_path")
    if not isinstance(raw_source_path, str) or not raw_source_path:
        raise ValueError(
            f"case {task_id!r}: prompt_source_path must be a non-empty relative path"
        )
    source_document = project_index.document_for_path(raw_source_path)
    source_start = case.get("prompt_source_start")
    if (
        isinstance(source_start, bool)
        or not isinstance(source_start, int)
        or source_start < 0
    ):
        raise ValueError(
            f"case {task_id!r}: prompt_source_start must be a "
            "non-negative integer"
        )
    prompt = prompt_text(case, "source-prefix")
    source_end = source_start + len(prompt)
    if source_document.source[source_start:source_end] != prompt:
        raise ValueError(
            f"case {task_id!r}: source_prefix does not match project index "
            f"document {source_document.source_path!r} span "
            f"[{source_start},{source_end})"
        )
    return ResolvedPromptGraph(
        project_index=project_index,
        document_id=source_document.id,
        source_path=source_document.source_path,
        source_start=source_start,
        repository_root=repository_root,
        index_path=index_path,
        index_receipt=index_receipt,
    )


def default_side_channels(seq_len: int) -> dict[str, mx.array]:
    """Zero/default token sidecars for standalone prompt-only evals."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return {
        name: mx.zeros(
            (1, seq_len),
            dtype=(
                mx.uint64
                if name in OPAQUE_ID_SIDE_CHANNEL_NAMES
                else mx.int32
            ),
        )
        for name in SIDE_CHANNEL_NAMES
    }


def _side_channel_dtype(name: str):
    return (
        mx.uint64
        if name in OPAQUE_ID_SIDE_CHANNEL_NAMES
        else mx.int32
    )


def _row_to_ints(array: mx.array) -> list[int]:
    if array.ndim != 2 or int(array.shape[0]) != 1:
        raise ValueError(f"expected side channel shaped (1, S), got {array.shape}")
    return [int(item) for item in array[0].tolist()]


def _as_evaluation_prompt(prompt: str | EvaluationPrompt) -> EvaluationPrompt:
    if isinstance(prompt, str):
        prompt = EvaluationPrompt("source-prefix", prompt)
    if not isinstance(prompt, EvaluationPrompt):
        raise TypeError(
            "prompt must be a string or EvaluationPrompt, got "
            f"{type(prompt).__name__}"
        )
    if prompt.mode not in {
        "source-prefix",
        "causal-docstring",
        "fim",
        "ifim",
    }:
        raise ValueError(f"unsupported prompt mode {prompt.mode!r}")
    if not prompt.source_prefix:
        raise ValueError("evaluation prompt requires non-empty source_prefix")
    if prompt.is_fim and not prompt.source_suffix:
        raise ValueError(f"{prompt.mode} prompt requires non-empty source_suffix")
    if prompt.mode in {"causal-docstring", "ifim"}:
        if not prompt.instruction or not _is_cpp_comment(prompt.instruction):
            raise ValueError(
                f"{prompt.mode} instruction must be a C/C++ comment or docstring"
            )
    elif prompt.instruction is not None:
        raise ValueError(f"{prompt.mode} prompt must not carry an instruction")
    return prompt


def tokenizer_fim_special_ids(tokenizer: Any) -> FIMSpecialTokenIds:
    return FIMSpecialTokenIds(
        eot=tokenizer.eos_token_id,
        fim_prefix=tokenizer.fim_prefix_id,
        fim_middle=tokenizer.fim_middle_id,
        fim_suffix=tokenizer.fim_suffix_id,
        fim_instruction=tokenizer.fim_instruction_id,
    )


def evaluation_prompt_token_ids(
    tokenizer: Any,
    prompt: str | EvaluationPrompt,
) -> list[int]:
    resolved = _as_evaluation_prompt(prompt)
    if not resolved.is_fim:
        return list(tokenizer.encode(resolved.render()))
    instruction_ids = (
        None
        if resolved.instruction is None
        else list(tokenizer.encode(resolved.instruction))
    )
    return build_fim_prompt_ids(
        tokenizer.encode(resolved.source_prefix),
        tokenizer.encode(resolved.source_suffix),
        mode="psm",
        instruction_token_ids=instruction_ids,
        special_token_ids=tokenizer_fim_special_ids(tokenizer),
    )


def _resolve_source_suffix_start(
    project_index: PromptProjectIndex,
    prompt: EvaluationPrompt,
    *,
    document_id: int,
    source_start: int,
) -> int:
    source = project_index.document_for_id(document_id).source
    minimum_start = source_start + len(prompt.source_prefix)
    match = source.find(prompt.source_suffix, minimum_start)
    repeated = match >= 0 and source.find(prompt.source_suffix, match + 1) >= 0
    if match < 0 or repeated:
        reason = "not found" if match < 0 else "ambiguous"
        raise ValueError(
            "source_suffix must map to exactly one project index span after "
            f"source_prefix; {reason}"
        )
    return match


def _repository_prompt_context(
    project_index: PromptProjectIndex,
    prompt: EvaluationPrompt,
    *,
    document_id: int,
    source_path: str,
    source_start: int,
) -> PromptGraphContext:
    prefix_context = PromptGraphContext.from_repository_prompt(
        project_index,
        prompt.source_prefix,
        document_id=document_id,
        source_path=source_path,
        source_start=source_start,
        language="cpp",
    )
    if prompt.mode == "source-prefix":
        return prefix_context
    if prompt.mode == "causal-docstring":
        assert prompt.instruction is not None
        if prefix_context.segments[-1].role != "target":
            raise ValueError("repository prompt graph target segment is missing")
        return PromptGraphContext(
            segments=(
                *prefix_context.segments[:-1],
                PromptGraphSegment(prompt.instruction, role="instruction"),
                prefix_context.segments[-1],
            ),
            language="cpp",
        )

    suffix_start = _resolve_source_suffix_start(
        project_index,
        prompt,
        document_id=document_id,
        source_start=source_start,
    )
    segments = list(prefix_context.segments[:-1])
    if prompt.mode == "ifim":
        assert prompt.instruction is not None
        segments.extend(
            (
                PromptGraphSegment(
                    FIM_INSTRUCTION_TOKEN,
                    role="fim_instruction_marker",
                ),
                PromptGraphSegment(prompt.instruction, role="instruction"),
            )
        )
    segments.extend(
        (
            PromptGraphSegment(FIM_PREFIX_TOKEN, role="fim_prefix_marker"),
            PromptGraphSegment(
                prompt.source_prefix,
                document_id=document_id,
                source_path=source_path,
                source_start=source_start,
                role="target_prefix",
            ),
            PromptGraphSegment(FIM_SUFFIX_TOKEN, role="fim_suffix_marker"),
            PromptGraphSegment(
                prompt.source_suffix,
                document_id=document_id,
                source_path=source_path,
                source_start=suffix_start,
                role="target_suffix",
            ),
            PromptGraphSegment(FIM_MIDDLE_TOKEN, role="fim_middle_marker"),
        )
    )
    return PromptGraphContext(segments=tuple(segments), language="cpp")


def build_prompt_context(
    tokenizer: Any,
    prompt: str | EvaluationPrompt,
    *,
    prompt_graph_mode: PromptGraphMode = "off",
    prompt_sidecars: PromptSidecars,
    prepend_code_start: bool,
    project_index: PromptProjectIndex | None = None,
    prompt_document_id: int | None = None,
    prompt_source_path: str | None = None,
    prompt_source_start: int | None = None,
    prompt_graph_cache_dir: Path | None = None,
) -> GenerationPromptContext:
    """Encode a prompt and build token-aligned prompt sidecars.

    Repository mode keeps a live generated continuation chunk and routes every
    generated query to a deterministic visible repository summary. Explicit
    graph-off mode still keeps every sidecar shape token-aligned.
    """

    resolved_prompt = _as_evaluation_prompt(prompt)
    if resolved_prompt.is_fim and prompt_sidecars != "zero":
        raise ValueError(
            "FIM/IFIM prompts require --prompt-sidecars zero; clang cannot "
            "align reordered source_prefix/source_suffix tokens"
        )
    if resolved_prompt.is_fim and prepend_code_start:
        raise ValueError(
            "FIM/IFIM prompts cannot use --prepend-code-start because the "
            "special-token contract must start with FIM_PREFIX or FIM_INSTRUCTION"
        )

    if prompt_graph_mode == "repo":
        if prompt_sidecars != "zero":
            raise ValueError(
                "--prompt-graph-mode repo owns prompt sidecars; "
                "--prompt-sidecars must be zero"
            )
        if prepend_code_start:
            raise ValueError(
                "--prepend-code-start is not supported with "
                "--prompt-graph-mode repo because the synthetic token has "
                "no source offset"
            )
        if project_index is None:
            raise ValueError(
                "prompt graph mode repo requires a PromptProjectIndex"
            )
        project_index.validate_production_repository_index(
            expected_indexer_root=REPO_ROOT,
        )
        if prompt_source_start is None:
            raise ValueError(
                "prompt graph mode repo requires prompt_source_start"
            )
        if prompt_document_id is None or not prompt_source_path:
            raise ValueError(
                "prompt graph mode repo requires prompt_document_id and "
                "prompt_source_path"
            )
        if prompt_graph_cache_dir is None:
            raise ValueError(
                "prompt graph mode repo requires a deterministic cache directory"
            )
        artifact = PromptGraphBuilder(
            tokenizer,
            cache_dir=prompt_graph_cache_dir,
        ).build(
            project_index,
            _repository_prompt_context(
                project_index,
                resolved_prompt,
                document_id=prompt_document_id,
                source_path=prompt_source_path,
                source_start=prompt_source_start,
            ),
        )
        if resolved_prompt.is_fim:
            expected = evaluation_prompt_token_ids(tokenizer, resolved_prompt)
            if list(artifact.token_ids[-len(expected) :]) != expected:
                raise ValueError(
                    "repository prompt graph changed the exact FIM/IFIM "
                    "special-token contract"
                )
        side_channels = {
            name: list(artifact.side_channels[name])
            for name in SIDE_CHANNEL_NAMES
        }
        receipt = dict(artifact.receipt)
        receipt["prompt_graph_mode"] = "repo"
        receipt["prompt_mode"] = resolved_prompt.mode
        return GenerationPromptContext(
            token_ids=list(artifact.token_ids),
            side_channels=side_channels,
            receipt=receipt,
            graph_artifact=artifact,
        )

    if prompt_graph_mode != "off":
        raise ValueError(f"unsupported prompt_graph_mode={prompt_graph_mode!r}")

    if prompt_sidecars == "zero":
        ids = evaluation_prompt_token_ids(tokenizer, resolved_prompt)
        if prepend_code_start:
            ids.insert(0, tokenizer.code_start_id)
        side = {name: [0] * len(ids) for name in SIDE_CHANNEL_NAMES}
        return GenerationPromptContext(
            token_ids=ids,
            side_channels=side,
            receipt={
                "prompt_graph_mode": "off",
                "prompt_sidecars": "zero",
                "prompt_mode": resolved_prompt.mode,
            },
        )

    if prompt_sidecars != "clang":
        raise ValueError(f"unsupported prompt_sidecars={prompt_sidecars!r}")
    if prepend_code_start:
        raise ValueError("--prepend-code-start is not supported with --prompt-sidecars clang")

    builder = InferenceSideChannelBuilder(
        tokenizer,
        adapter=get_builtin_code_metadata_adapter("cpp"),
        fail_policy="error",
    )
    result = builder.build(resolved_prompt.render(), language="cpp")
    ids = _row_to_ints(result.prompt_ids)
    side: dict[str, list[int]] = {}
    for name in SIDE_CHANNEL_NAMES:
        value = result.model_kwargs.get(name)
        side[name] = (
            _row_to_ints(value)
            if isinstance(value, mx.array)
            else [int(TOKEN_SIDECAR_DEFAULTS[name])] * len(ids)
        )
    receipt = dict(result.provenance)
    receipt["prompt_graph_mode"] = "off"
    receipt["prompt_mode"] = resolved_prompt.mode
    return GenerationPromptContext(
        token_ids=ids,
        side_channels=side,
        receipt=receipt,
    )


def prompt_model_inputs(
    context: GenerationPromptContext,
    *,
    total_token_count: int,
    window_start: int,
    window_end: int,
    bias_dtype: mx.Dtype,
) -> tuple[
    dict[str, mx.array],
    mx.array | None,
    mx.array | None,
    dict[str, Any],
]:
    if context.graph_artifact is not None:
        graph_inputs = context.graph_artifact.model_inputs(
            total_token_count=total_token_count,
            window_start=window_start,
            window_end=window_end,
        )
        side = {
            name: mx.array(
                [graph_inputs.side_channels[name]],
                dtype=_side_channel_dtype(name),
            )
            for name in SIDE_CHANNEL_NAMES
        }
        relation_values = graph_inputs.dense_relation_attention_bias()
        edge_kind_values = graph_inputs.dense_edge_kind_attention_bias(
            edge_kind_weights=DEFAULT_EDGE_WEIGHTS,
        )
        relation_nonzero = sum(
            value != 0.0 for row in relation_values for value in row
        )
        edge_kind_nonzero = sum(
            value != 0.0 for row in edge_kind_values for value in row
        )
        edge_kind_route_counts = graph_inputs.edge_kind_route_counts()
        if relation_nonzero <= 0:
            raise ValueError(
                "repository prompt graph produced an all-zero relation prior"
            )
        if edge_kind_nonzero <= 0:
            raise ValueError(
                "repository prompt graph produced an all-zero edge-kind prior; "
                "zero kind priors are allowed only in explicit graph-off ablation"
            )
        relation_bias = mx.array([relation_values], dtype=bias_dtype)
        edge_kind_bias = mx.array([edge_kind_values], dtype=bias_dtype)
        receipt = dict(graph_inputs.receipt)
        receipt.update(
            {
                "relation_bias_nonzero": relation_nonzero,
                "edge_kind_bias_nonzero": edge_kind_nonzero,
                "edge_kind_route_count": sum(edge_kind_route_counts.values()),
                "edge_kind_route_counts": {
                    str(kind): count
                    for kind, count in edge_kind_route_counts.items()
                },
                "edge_family_route_counts": {
                    str(family): int(count)
                    for family, count in graph_inputs.receipt[
                        "edge_counts"
                    ].items()
                },
                "graph_bias_dtype": _dtype_name(bias_dtype),
            }
        )
        return side, relation_bias, edge_kind_bias, receipt

    if total_token_count < len(context.token_ids):
        raise ValueError(
            "total_token_count cannot be shorter than the prompt context"
        )
    if (
        window_start < 0
        or window_end <= window_start
        or window_end > total_token_count
    ):
        raise ValueError(
            f"invalid prompt window [{window_start},{window_end}) "
            f"for total {total_token_count}"
        )
    generated = total_token_count - len(context.token_ids)
    if context.receipt.get("prompt_sidecars") == "zero":
        generated_values = {name: 0 for name in SIDE_CHANNEL_NAMES}
        generated_policy = "explicit_graph_off_zero_sidecars_v1"
    else:
        generated_values = {
            name: int(GENERATED_TOKEN_SIDECAR_DEFAULTS[name])
            for name in SIDE_CHANNEL_NAMES
        }
        anchor_candidates = [
            index
            for index, value in enumerate(context.side_channels["structure_ids"])
            if int(value) > 0
        ]
        if anchor_candidates:
            anchor = anchor_candidates[-1]
            for name in MODEL_SIDE_CHANNEL_NAMES:
                generated_values[name] = int(context.side_channels[name][anchor])
        generated_policy = "generated_syntax_continuation_chunk_v1"
    side = {
        name: mx.array(
            [
                (
                    list(context.side_channels[name])
                    + [generated_values[name]] * generated
                )[
                    window_start:window_end
                ]
            ],
            dtype=_side_channel_dtype(name),
        )
        for name in SIDE_CHANNEL_NAMES
    }
    receipt = {
        **context.receipt,
        "window_start": window_start,
        "window_end": window_end,
        "total_token_count": total_token_count,
        "generated_token_policy": generated_policy,
        "generated_token_count": generated,
    }
    return side, None, None, receipt


def _dtype_name(dtype: mx.Dtype) -> str:
    return str(dtype).rsplit(".", maxsplit=1)[-1]


class BodyDecodeConstraints:
    """Small token-level guards for body-generation evals.

    These are not a substitute for grammar decoding.  They keep smoke evals
    from spending all 128 tokens on a single degenerate token run or repeating
    already-seen ngrams, while still leaving C++ syntax choices to the model.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt_len: int,
        max_token_run: int = 8,
        no_repeat_ngram_size: int = 6,
    ) -> None:
        self.prompt_len = int(prompt_len)
        self.max_token_run = int(max_token_run)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self._always_banned = {
            tokenizer.bos_token_id,
            tokenizer.fim_prefix_id,
            tokenizer.fim_middle_id,
            tokenizer.fim_suffix_id,
            tokenizer.think_start_id,
            tokenizer.think_end_id,
            tokenizer.query_tool_id,
            tokenizer.tool_result_id,
            tokenizer.code_start_id,
        }
        fim_instruction_id = getattr(tokenizer, "fim_instruction_id", None)
        if isinstance(fim_instruction_id, int) and not isinstance(
            fim_instruction_id, bool
        ):
            self._always_banned.add(fim_instruction_id)
        token_for_id = getattr(tokenizer, "token_for_id", None)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if callable(token_for_id) and isinstance(vocab_size, int):
            self._always_banned.update(
                token_id
                for token_id in range(vocab_size)
                if str(token_for_id(token_id) or "").startswith("<RESERVED_")
            )

    def __call__(self, logits: mx.array, tokens: mx.array) -> mx.array:
        if logits.ndim != 2 or tokens.ndim != 2:
            raise ValueError("BodyDecodeConstraints expects logits=(B,V), tokens=(B,S)")
        mx.eval(tokens)
        rows = tokens.tolist()
        vocab_ids = mx.arange(int(logits.shape[1]), dtype=mx.int32)
        masks: list[mx.array] = []
        for row in rows:
            generated = [int(token_id) for token_id in row[self.prompt_len :]]
            banned = set(self._always_banned)
            banned.update(self._run_bans(generated))
            banned.update(self._ngram_bans(generated))
            if banned:
                banned_ids = mx.array(sorted(banned), dtype=mx.int32)
                masks.append(mx.any(vocab_ids[:, None] == banned_ids[None, :], axis=1))
            else:
                masks.append(mx.zeros((int(logits.shape[1]),), dtype=mx.bool_))
        mask = mx.stack(masks, axis=0)
        neg_inf = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
        return mx.where(mask, neg_inf, logits)

    def _run_bans(self, generated: list[int]) -> set[int]:
        if self.max_token_run <= 1 or len(generated) < self.max_token_run:
            return set()
        tail = generated[-self.max_token_run :]
        return {tail[-1]} if len(set(tail)) == 1 else set()

    def _ngram_bans(self, generated: list[int]) -> set[int]:
        n = self.no_repeat_ngram_size
        if n <= 1 or len(generated) < n - 1:
            return set()
        prefix = tuple(generated[-(n - 1) :])
        banned: set[int] = set()
        for index in range(0, len(generated) - n + 1):
            if tuple(generated[index : index + n - 1]) == prefix:
                banned.add(generated[index + n - 1])
        return banned


def trim_body_completion(text: str) -> str:
    stripped = text.replace("\r\n", "\n")
    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1]
            first_newline = stripped.find("\n")
            if first_newline != -1 and stripped[:first_newline].strip().isidentifier():
                stripped = stripped[first_newline + 1 :]
    stop_markers = (
        "int main(",
        "#include ",
        "```",
        "<BOS>",
        "<EOS>",
        "<CODE_START>",
        "<CODE_END>",
        "<FIM_",
        "<QUERY_TOOL>",
        "<TOOL_RESULT>",
        "<THINK_",
        "<RESERVED_",
    )
    for marker in stop_markers:
        pos = stripped.find(marker)
        if pos >= 0:
            stripped = stripped[:pos]
    stripped = _trim_at_function_closing_brace(stripped)
    body = stripped.strip()
    return body + ("\n" if body else "")


def _trim_at_function_closing_brace(text: str) -> str:
    """Drop only the brace that closes the prompt's already-open function."""
    depth = 1
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if state == "line-comment":
            if char == "\n":
                state = "code"
            index += 1
            continue
        if state == "block-comment":
            if char == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if state in {"string", "character"}:
            quote = '"' if state == "string" else "'"
            if char == "\\":
                index += 2
            elif char == quote:
                state = "code"
                index += 1
            else:
                index += 1
            continue

        if char == "/" and following == "/":
            state = "line-comment"
            index += 2
            continue
        if char == "/" and following == "*":
            state = "block-comment"
            index += 2
            continue
        if char == "R" and following == '"':
            delimiter_end = text.find("(", index + 2, min(len(text), index + 20))
            if delimiter_end >= 0:
                delimiter = text[index + 2 : delimiter_end]
                if len(delimiter) <= 16 and not any(
                    item.isspace() or item in "()\\" for item in delimiter
                ):
                    terminator = ")" + delimiter + '"'
                    raw_end = text.find(terminator, delimiter_end + 1)
                    if raw_end < 0:
                        return text
                    index = raw_end + len(terminator)
                    continue
        if char == '"':
            state = "string"
            index += 1
            continue
        if char == "'":
            state = "character"
            index += 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[:index]
        index += 1
    return text


def reject_unsupported_checkpoint(path: Path) -> None:
    """Fail closed for Megatron distributed checkpoints.

    This script is intentionally an MLX checkpoint evaluator. Loading
    ``iter_0005000/__0_0.distcp`` directly would silently test a different path
    if we guessed a mapping, so we require an explicit converter first.
    """
    if path.suffix == ".distcp":
        raise ValueError(
            f"{path}: Megatron torch_dist .distcp checkpoint is not an MLX "
            "checkpoint; convert to DenseCppLM safetensors first"
        )
    if path.is_dir() and (path / "latest_checkpointed_iteration.txt").exists():
        raise ValueError(
            f"{path}: Megatron torch_dist checkpoint directory is not supported by "
            "the MLX evaluator; convert to DenseCppLM safetensors first"
        )


def checkpoint_model_config(path: Path) -> dict[str, Any]:
    """Return the converter-emitted DenseCppLM config next to a checkpoint."""

    metadata = path.parent / "model.json" if path.suffix == ".safetensors" else path / "model.json"
    if not metadata.exists():
        return {}
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    config = payload.get("config")
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"{metadata}: config must be an object")
    return dict(config)


def _cfg_value(
    checkpoint_config: dict[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    return checkpoint_config.get(key, fallback)


def build_model_config(
    args: argparse.Namespace,
    checkpoint_config: dict[str, Any],
) -> DenseCppLMConfig:
    return DenseCppLMConfig(
        vocab_size=int(_cfg_value(checkpoint_config, "vocab_size", args.vocab_size)),
        hidden_size=int(_cfg_value(checkpoint_config, "hidden_size", args.hidden)),
        depth=int(_cfg_value(checkpoint_config, "depth", args.depth)),
        ffn_hidden_size=int(_cfg_value(checkpoint_config, "ffn_hidden_size", args.ffn)),
        num_query_heads=int(_cfg_value(checkpoint_config, "num_query_heads", args.num_query_heads)),
        num_kv_heads=int(_cfg_value(checkpoint_config, "num_kv_heads", args.num_kv_heads)),
        head_dim=int(_cfg_value(checkpoint_config, "head_dim", args.head_dim)),
        max_seq_length=int(_cfg_value(checkpoint_config, "max_seq_length", args.seq_len)),
        attention_mode=args.attention_mode,
        structure_components=str(_cfg_value(checkpoint_config, "structure_components", args.structure_components)),
        structure_num_categories=int(
            _cfg_value(
                checkpoint_config,
                "structure_num_categories",
                args.structure_num_categories,
            )
        ),
        structure_max_dep_level=int(
            _cfg_value(
                checkpoint_config,
                "structure_max_dep_level",
                args.structure_max_dep_level,
            )
        ),
        structure_bottleneck_dim=int(
            _cfg_value(
                checkpoint_config,
                "structure_bottleneck_dim",
                args.structure_bottleneck_dim,
            )
        ),
        require_graph_routes=bool(
            getattr(
                args,
                "require_graph_routes",
                args.prompt_graph_mode == "repo",
            )
        ),
        graph_routes_enabled=bool(
            getattr(
                args,
                "graph_routes_enabled",
                args.prompt_graph_mode == "repo",
            )
        ),
        ngram_hash_enabled=not args.disable_ngram,
        ngram_hash_heads=int(_cfg_value(checkpoint_config, "ngram_hash_heads", args.ngram_hash_heads)),
        ngram_hash_table_size=int(_cfg_value(checkpoint_config, "ngram_hash_table_size", args.ngram_hash_table_size)),
        ngram_hash_embed_dim=int(_cfg_value(checkpoint_config, "ngram_hash_embed_dim", args.ngram_hash_embed_dim)),
    )


def build_model(args: argparse.Namespace) -> DenseCppLM:
    dtype = mx.bfloat16 if args.bf16 else None
    checkpoint = Path(args.checkpoint)
    reject_unsupported_checkpoint(checkpoint)
    checkpoint_config = checkpoint_model_config(checkpoint)
    cfg = build_model_config(args, checkpoint_config)
    model = DenseCppLM(cfg, dtype=dtype)
    if checkpoint.is_dir():
        load_checkpoint(model, checkpoint, strict=not args.non_strict)
    elif checkpoint.suffix == ".safetensors":
        model.load_weights(str(checkpoint), strict=not args.non_strict)
        mx.eval(model.parameters())
    else:
        raise ValueError(
            f"{checkpoint}: expected MLX checkpoint dir or .safetensors file"
        )
    return model


def model_graph_bias_dtype(model: DenseCppLM) -> mx.Dtype:
    token_embedding = getattr(model, "token_embedding", None)
    weight = getattr(token_embedding, "weight", None)
    if not isinstance(weight, mx.array):
        raise TypeError(
            "generation model token_embedding.weight must be an MLX array"
        )
    if weight.dtype not in {mx.float16, mx.bfloat16, mx.float32}:
        raise ValueError(
            "generation model requires a floating graph-bias dtype, got "
            f"{weight.dtype}"
        )
    return weight.dtype


def _sample_next(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    tokens: mx.array,
    logits_processors: list[Any] | None = None,
) -> int:
    last = logits[0, -1].astype(mx.float32)
    sampled = sample_next_token(
        last[None, :],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        tokens=tokens,
        logits_processors=logits_processors,
    )
    return int(sampled[0, 0].item())


def identity_completion(text: str) -> str:
    """Preserve non-C++ typed-domain output for the local domain evaluator."""

    return text


def generate_completion_from_context(
    model: DenseCppLM,
    tokenizer: Any,
    prompt_context: GenerationPromptContext,
    *,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    prompt_graph_mode: PromptGraphMode,
    graph_bias_dtype: mx.Dtype,
    completion_normalizer: Callable[[str], str] = trim_body_completion,
    logits_processors: Sequence[Any] | None = None,
) -> tuple[str, int, int, str, dict[str, Any]]:
    if not callable(completion_normalizer):
        raise TypeError("completion_normalizer must be callable")
    prompt_ids = list(prompt_context.token_ids)
    if not prompt_ids:
        raise ValueError("prompt tokenized to zero tokens")
    if len(prompt_ids) >= seq_len:
        raise ValueError(
            f"prompt has {len(prompt_ids)} tokens but seq_len={seq_len}; "
            "refusing to truncate prompt graph coordinates"
        )
    if (
        prompt_context.graph_artifact is not None
        and len(prompt_ids) + max_new_tokens > seq_len
    ):
        raise ValueError(
            "prompt plus max_new_tokens exceeds seq_len; refusing to discard "
            "indexed repository graph tokens during decode"
        )
    token_context = list(prompt_ids)
    generated: list[int] = []
    last_window_receipt: dict[str, Any] | None = None
    finish_reason = "length"
    for _ in range(max_new_tokens):
        window_start = max(0, len(token_context) - seq_len)
        window = token_context[window_start:]
        input_ids = mx.array([window], dtype=mx.int32)
        (
            side,
            block_bias,
            edge_kind_bias,
            last_window_receipt,
        ) = prompt_model_inputs(
            prompt_context,
            total_token_count=len(token_context),
            window_start=window_start,
            window_end=len(token_context),
            bias_dtype=graph_bias_dtype,
        )
        graph_routes_enabled = bool(
            getattr(
                getattr(model, "config", None),
                "graph_routes_enabled",
                block_bias is not None,
            )
        )
        if graph_routes_enabled and block_bias is None:
            if prompt_graph_mode != "off":
                raise RuntimeError(
                    "graph routes are enabled but the prompt graph "
                    "did not produce typed fixed biases"
                )
            block_bias = mx.zeros(
                (1, len(window), len(window)), dtype=graph_bias_dtype
            )
            edge_kind_bias = mx.zeros_like(block_bias)
            last_window_receipt.update(
                {
                    "graph_bias_ablation": "explicit_prompt_graph_mode_off",
                    "relation_bias_nonzero": 0,
                    "edge_kind_bias_nonzero": 0,
                    "edge_kind_route_count": 0,
                    "edge_kind_route_counts": {},
                    "graph_bias_dtype": _dtype_name(graph_bias_dtype),
                }
            )
        elif graph_routes_enabled and edge_kind_bias is None:
            raise RuntimeError(
                "graph routes are enabled but graph_edge_kind_bias is absent; "
                "refusing to fabricate a zero categorical prior"
            )
        elif not graph_routes_enabled and (
            block_bias is not None or edge_kind_bias is not None
        ):
            raise RuntimeError(
                "prompt graph biases were produced for a model with graph "
                "routes disabled"
            )
        logits, _loss = model(
            input_ids,
            block_bias=block_bias,
            edge_kind_bias=edge_kind_bias,
            # The assembled typed prompt is one intentional inference document.
            # Source provenance remains in sidecars; packed document IDs must
            # not split cross-file or cross-domain graph context.
            document_ids=mx.ones_like(input_ids),
            **{name: side[name] for name in MODEL_SIDE_CHANNEL_NAMES},
        )
        next_id = _sample_next(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tokens=input_ids,
            logits_processors=logits_processors,
        )
        mx.eval(logits)
        if next_id in {tokenizer.eos_token_id, tokenizer.code_end_id}:
            finish_reason = "eos"
            break
        generated.append(next_id)
        token_context.append(next_id)
    receipt = dict(prompt_context.receipt)
    receipt["schema"] = "cppmega_mlx_generation_receipt_v1"
    receipt["generated_token_count"] = len(generated)
    receipt["finish_reason"] = finish_reason
    receipt["aligned_side_channels"] = list(SIDE_CHANNEL_NAMES)
    receipt["model_consumed_side_channels"] = list(MODEL_SIDE_CHANNEL_NAMES)
    receipt["model_consumed_runtime_channels"] = [
        "document_ids",
        "graph_attention_bias",
        "graph_edge_kind_bias",
    ]
    if last_window_receipt is not None:
        receipt["last_decode_window"] = last_window_receipt
        for key in (
            "graph_bias_ablation",
            "relation_bias_nonzero",
            "edge_kind_bias_nonzero",
            "edge_kind_route_count",
            "edge_kind_route_counts",
            "edge_family_route_counts",
            "graph_bias_dtype",
        ):
            if key in last_window_receipt:
                receipt[key] = last_window_receipt[key]
    return (
        completion_normalizer(tokenizer.decode(generated)),
        len(prompt_ids),
        len(generated),
        finish_reason,
        receipt,
    )


def generate_completion(
    model: DenseCppLM,
    tokenizer: Any,
    prompt: str | EvaluationPrompt,
    *,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    prompt_graph_mode: PromptGraphMode,
    prompt_sidecars: PromptSidecars,
    prepend_code_start: bool,
    project_index: PromptProjectIndex | None,
    prompt_document_id: int | None,
    prompt_source_path: str | None,
    prompt_source_start: int | None,
    prompt_graph_cache_dir: Path | None,
    graph_bias_dtype: mx.Dtype,
) -> tuple[str, int, int, str, dict[str, Any]]:
    prompt_context = build_prompt_context(
        tokenizer,
        prompt,
        prompt_graph_mode=prompt_graph_mode,
        prompt_sidecars=prompt_sidecars,
        prepend_code_start=prepend_code_start,
        project_index=project_index,
        prompt_document_id=prompt_document_id,
        prompt_source_path=prompt_source_path,
        prompt_source_start=prompt_source_start,
        prompt_graph_cache_dir=prompt_graph_cache_dir,
    )
    return generate_completion_from_context(
        model,
        tokenizer,
        prompt_context,
        seq_len=seq_len,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        prompt_graph_mode=prompt_graph_mode,
        graph_bias_dtype=graph_bias_dtype,
        completion_normalizer=trim_body_completion,
        logits_processors=[
            BodyDecodeConstraints(tokenizer, prompt_len=len(prompt_context.token_ids))
        ],
    )


def write_completions(
    cases: list[dict[str, Any]],
    completions_path: Path,
    *,
    model: DenseCppLM,
    tokenizer: Any,
    prompt_mode: PromptMode,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    prompt_graph_mode: PromptGraphMode,
    prompt_sidecars: PromptSidecars,
    prepend_code_start: bool,
    cases_dir: Path,
    prompt_graph_cache_dir: Path | None,
    prompt_index_cache_dir: Path | None,
    indexer_root: Path | None = None,
    num_samples: int = 1,
    base_seed: int = 0,
    graph_bias_dtype: mx.Dtype = mx.float32,
) -> list[dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")
    if num_samples > 1 and temperature <= 0.0:
        raise ValueError(
            "multiple pass@k candidates require temperature > 0; refusing "
            "duplicate greedy samples"
        )
    completions_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with completions_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            case_prompt = evaluation_prompt(case, prompt_mode)
            case_graph_mode = effective_case_prompt_graph_mode(
                case,
                prompt_graph_mode,
            )
            resolved = resolve_case_prompt_graph(
                case,
                cases_dir=cases_dir,
                mode=case_graph_mode,
                prompt_index_cache_dir=prompt_index_cache_dir,
                indexer_root=indexer_root,
            )
            for sample_index in range(num_samples):
                seed = base_seed + sample_index
                mx.random.seed(seed)
                (
                    completion,
                    prompt_tokens,
                    generated_tokens,
                    finish_reason,
                    side_provenance,
                ) = generate_completion(
                    model,
                    tokenizer,
                    case_prompt,
                    seq_len=seq_len,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    prompt_graph_mode=case_graph_mode,
                    prompt_sidecars=prompt_sidecars,
                    prepend_code_start=prepend_code_start,
                    project_index=(
                        None if resolved is None else resolved.project_index
                    ),
                    prompt_document_id=(
                        None if resolved is None else resolved.document_id
                    ),
                    prompt_source_path=(
                        None if resolved is None else resolved.source_path
                    ),
                    prompt_source_start=(
                        None if resolved is None else resolved.source_start
                    ),
                    prompt_graph_cache_dir=prompt_graph_cache_dir,
                    graph_bias_dtype=graph_bias_dtype,
                )
                row = {
                    "task_id": case["task_id"],
                    "sample_index": sample_index,
                    "seed": seed,
                    "completion": completion,
                    "completion_source": "model_generation",
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "finish_reason": finish_reason,
                    "length_truncated": finish_reason == "length",
                    "prompt_mode": case_prompt.mode,
                    "prompt_graph_mode": case_graph_mode,
                    "prompt_graph_receipt": side_provenance,
                    "prompt_graph_index_receipt": (
                        None if resolved is None else resolved.index_receipt
                    ),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
    return rows


def compile_gate_env(
    base_env: dict[str, str] | None = None,
    *,
    path_dirs: Iterable[Path] = DEFAULT_COMPILE_PATH_DIRS,
) -> dict[str, str]:
    """Return an env that can find local clang/clang-format on macOS.

    The evaluator is commonly run with ``env -i`` to avoid dirty Python virtual
    environments.  That strips Homebrew LLVM from PATH, so the shared compile
    gate cannot find ``clang-format`` even though it is installed.  Make the PATH
    explicit and fail closed if the tool is still absent.
    """

    env = dict(os.environ if base_env is None else base_env)
    existing = [part for part in env.get("PATH", "").split(os.pathsep) if part]
    prepend = [str(path) for path in path_dirs if path.is_dir()]
    path = os.pathsep.join(dict.fromkeys([*prepend, *existing]))
    env["PATH"] = path
    if shutil.which("clang-format", path=path) is None:
        raise FileNotFoundError(
            "clang-format not found for compile gate; checked PATH=" + path
        )
    return env


def _file_receipt(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"receipt input not found: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved.read_bytes()).hexdigest(),
    }


def _git_output(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def build_compile_gate_receipt(
    *,
    script: Path,
    cases: Path,
    completions: Path,
) -> dict[str, Any]:
    script_receipt = _file_receipt(script)
    script_path = Path(script_receipt["path"])
    git_root_text = _git_output(script_path.parent, "rev-parse", "--show-toplevel")
    git_root = Path(git_root_text).resolve()
    try:
        relative_script = script_path.relative_to(git_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"compile gate script {script_path} is outside git root {git_root}"
        ) from exc
    git_commit = _git_output(git_root, "rev-parse", "HEAD")
    last_commit = _git_output(
        git_root,
        "log",
        "-1",
        "--format=%H",
        "--",
        relative_script,
    )
    if not last_commit:
        raise ValueError(
            f"compile gate script is not bound to a git commit: {script_path}"
        )
    worktree_status = _git_output(
        git_root,
        "status",
        "--short",
        "--untracked-files=all",
        "--",
        relative_script,
    )
    branch = _git_output(git_root, "branch", "--show-current")
    return {
        "schema": COMPILE_GATE_RECEIPT_SCHEMA,
        "script": {
            **script_receipt,
            "git_root": str(git_root),
            "git_relative_path": relative_script,
            "git_commit": git_commit,
            "git_last_commit": last_commit,
            "git_branch": branch or None,
            "git_worktree_status": worktree_status or "clean",
        },
        "inputs": {
            "cases": _file_receipt(cases),
            "completions": _file_receipt(completions),
        },
        "python": str(Path(sys.executable).resolve()),
    }


def estimate_pass_at_k(
    num_samples: int,
    num_correct: int,
    k: int,
) -> float | None:
    """Return the standard unbiased pass@k estimator for one task."""

    if isinstance(num_samples, bool) or num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    if (
        isinstance(num_correct, bool)
        or num_correct < 0
        or num_correct > num_samples
    ):
        raise ValueError(
            f"num_correct must be in [0,{num_samples}], got {num_correct}"
        )
    if isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if num_samples < k:
        return None
    return 1.0 - (
        math.comb(num_samples - num_correct, k)
        / math.comb(num_samples, k)
    )


def _candidate_groups(
    cases: Path,
    completions: Path,
) -> tuple[tuple[str, ...], dict[int, dict[str, dict[str, Any]]]]:
    task_ids = tuple(sorted(row["task_id"] for row in load_cases(cases)))
    expected_tasks = set(task_ids)
    groups: dict[int, dict[str, dict[str, Any]]] = {}
    seen_candidates: set[tuple[str, int, int]] = set()
    for raw_row in iter_jsonl(completions):
        row = dict(raw_row)
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("completion candidate needs non-empty task_id")
        has_sample_index = "sample_index" in row
        has_seed = "seed" in row
        if has_sample_index != has_seed:
            raise ValueError(
                f"candidate {task_id!r} must carry both sample_index and seed"
            )
        if not has_sample_index:
            row["sample_index"] = 0
            row["seed"] = 0
        sample_index = row["sample_index"]
        seed = row["seed"]
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
        ):
            raise ValueError(
                f"candidate {task_id!r} sample_index must be non-negative int"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                f"candidate {task_id!r} seed must be non-negative int"
            )
        identity = (task_id, sample_index, seed)
        if identity in seen_candidates:
            raise ValueError(f"duplicate completion candidate {identity!r}")
        seen_candidates.add(identity)
        sample_group = groups.setdefault(sample_index, {})
        if task_id in sample_group:
            raise ValueError(
                f"duplicate task/sample completion for {task_id!r}/{sample_index}"
            )
        sample_group[task_id] = row

    if not groups:
        raise ValueError(f"no completion candidates in {completions}")
    sample_indices = sorted(groups)
    if sample_indices != list(range(len(sample_indices))):
        raise ValueError(
            "candidate sample_index values must be contiguous from zero, got "
            f"{sample_indices}"
        )
    for sample_index, sample_group in groups.items():
        actual_tasks = set(sample_group)
        if actual_tasks != expected_tasks:
            missing = sorted(expected_tasks - actual_tasks)
            extra = sorted(actual_tasks - expected_tasks)
            raise ValueError(
                f"candidate sample {sample_index} task matrix mismatch: "
                f"missing={missing}, extra={extra}"
            )
    return task_ids, groups


def _pass_at_k_summary(
    results: list[dict[str, Any]],
    task_ids: tuple[str, ...],
) -> dict[str, float | None]:
    outcomes = {task_id: [] for task_id in task_ids}
    for result in results:
        task_id = result.get("task_id")
        if task_id not in outcomes:
            raise ValueError(f"compile gate returned unknown task_id {task_id!r}")
        passed = result.get("passed")
        if not isinstance(passed, bool):
            raise ValueError(
                f"compile gate result {task_id!r} has non-boolean passed"
            )
        outcomes[task_id].append(passed)

    metrics: dict[str, float | None] = {}
    for k in STANDARD_PASS_AT_K:
        estimates = [
            estimate_pass_at_k(len(values), sum(values), k)
            for values in outcomes.values()
        ]
        metrics[f"pass@{k}"] = (
            None
            if any(value is None for value in estimates)
            else sum(float(value) for value in estimates) / len(estimates)
        )
    return metrics


def run_compile_gate(
    *,
    cases: Path,
    completions: Path,
    report: Path,
    script: Path,
    keep_workdir: bool,
    fail_on_fail: bool = True,
    expected_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate_receipt = build_compile_gate_receipt(
        script=script,
        cases=cases,
        completions=completions,
    )
    if expected_receipt is not None and gate_receipt != expected_receipt:
        raise RuntimeError(
            "compile gate path/commit/hash changed after receipt binding"
        )
    task_ids, groups = _candidate_groups(cases, completions)
    report.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    gate_env = compile_gate_env()
    with tempfile.TemporaryDirectory(
        prefix="cppmega_mlx_compile_candidates_",
        dir=report.parent,
    ) as temp_dir:
        temp_root = Path(temp_dir)
        for sample_index in sorted(groups):
            sample_completions = temp_root / f"sample_{sample_index:04d}.jsonl"
            sample_report = temp_root / f"sample_{sample_index:04d}.json"
            with sample_completions.open("w", encoding="utf-8") as fh:
                for task_id in task_ids:
                    fh.write(
                        json.dumps(
                            groups[sample_index][task_id],
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            cmd = [
                sys.executable,
                str(Path(script).resolve()),
                "--cases",
                str(Path(cases).resolve()),
                "--completions",
                str(sample_completions),
                "--out",
                str(sample_report),
            ]
            if keep_workdir:
                cmd.append("--keep-workdir")
            subprocess.run(cmd, check=True, env=gate_env)
            sample_payload = json.loads(sample_report.read_text(encoding="utf-8"))
            summary = sample_payload.get("summary")
            results = sample_payload.get("results")
            if not isinstance(summary, dict) or not isinstance(results, list):
                raise ValueError(
                    f"compile gate sample {sample_index} emitted invalid report"
                )
            sample_summaries.append(
                {"sample_index": sample_index, **summary}
            )
            by_task = groups[sample_index]
            for raw_result in results:
                if not isinstance(raw_result, dict):
                    raise ValueError(
                        f"compile gate sample {sample_index} result is not an object"
                    )
                task_id = raw_result.get("task_id")
                candidate = by_task.get(task_id)
                if candidate is None:
                    raise ValueError(
                        f"compile gate sample {sample_index} returned unknown "
                        f"task_id {task_id!r}"
                    )
                all_results.append(
                    {
                        **raw_result,
                        "sample_index": sample_index,
                        "seed": candidate["seed"],
                    }
                )

    first_summary = sample_summaries[0]
    total = len(all_results)
    passed = sum(bool(result["passed"]) for result in all_results)
    pass_at_k = _pass_at_k_summary(all_results, task_ids)
    summary: dict[str, Any] = {
        "total": total,
        "passed": passed,
        "compiled": sum(
            bool(result.get("compile_ok")) for result in all_results
        ),
        "ran": sum(bool(result.get("run_ok")) for result in all_results),
        "clang_format_ok": sum(
            result.get("clang_format") is None
            or bool(result["clang_format"].get("ok"))
            for result in all_results
        ),
        "pass_rate": passed / total if total else 0.0,
        "tasks": len(task_ids),
        "samples_per_task": len(groups),
        "candidate_identity": ["task_id", "sample_index", "seed"],
        "pass_at_k": {
            str(k): pass_at_k[f"pass@{k}"] for k in STANDARD_PASS_AT_K
        },
        **pass_at_k,
    }
    for key in ("cpp_compiler", "c_compiler", "clang_format", "jobs"):
        if key in first_summary:
            summary[key] = first_summary[key]
    if "repository_cases" in first_summary:
        summary["repository_cases"] = first_summary["repository_cases"]
        summary["repository_candidate_evaluations"] = sum(
            int(item.get("repository_cases", 0))
            for item in sample_summaries
        )
    payload = {
        "schema": COMPILE_REPORT_SCHEMA,
        "compile_gate_receipt": gate_receipt,
        "summary": summary,
        "sample_summaries": sample_summaries,
        "results": all_results,
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if fail_on_fail and passed != total:
        raise subprocess.CalledProcessError(
            1,
            [str(Path(script).resolve()), "<candidate-matrix>"],
        )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "mlx_generation_eval")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--compile-gate-script", type=Path, default=DEFAULT_COMPILE_GATE)
    ap.add_argument("--skip-compile-gate", action="store_true")
    ap.add_argument("--keep-workdir", action="store_true")
    ap.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODE_CHOICES,
        default="source-prefix",
        help=(
            "Prompt contract: causal C/C++ comment plus prefix, exact PSM FIM, "
            "or comment-instructed IFIM. 'docstring' is a legacy alias for "
            "'causal-docstring'."
        ),
    )
    ap.add_argument(
        "--prompt-graph-mode",
        choices=("repo", "off"),
        default="repo",
        help="Use a case-linked repository graph, or explicitly ablate graph routes.",
    )
    ap.add_argument(
        "--prompt-graph-cache-dir",
        type=Path,
        default=None,
        help="Hash-addressed graph artifact cache (defaults under --out-dir).",
    )
    ap.add_argument(
        "--prompt-index-cache-dir",
        type=Path,
        default=None,
        help="Hash-addressed clang project-index cache (defaults under --out-dir).",
    )
    ap.add_argument(
        "--clang-indexer-root",
        type=Path,
        default=REPO_ROOT,
        help="Checkout containing tools/clang_indexer/index_project.py.",
    )
    ap.add_argument("--prompt-sidecars", choices=("zero", "clang"), default="zero")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--num-samples",
        "--samples-per-task",
        dest="num_samples",
        type=int,
        default=10,
        help="Candidates per task; 10 enables standard pass@1/5/10.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base MLX sampling seed; candidate i uses seed+i.",
    )
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--prepend-code-start", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--non-strict", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=65536)
    ap.add_argument("--hidden", type=int, default=1280)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--ffn", type=int, default=3456)
    ap.add_argument("--num-query-heads", type=int, default=20)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--attention-mode", choices=("gqa", "full", "mla", "dsa"), default="gqa")
    ap.add_argument("--disable-ngram", action="store_true")
    ap.add_argument("--structure-components", default="all")
    ap.add_argument("--structure-num-categories", type=int, default=9)
    ap.add_argument("--structure-max-dep-level", type=int, default=64)
    ap.add_argument("--structure-bottleneck-dim", type=int, default=128)
    ap.add_argument("--ngram-hash-heads", type=int, default=8)
    ap.add_argument("--ngram-hash-table-size", type=int, default=500_000)
    ap.add_argument("--ngram-hash-embed-dim", type=int, default=16)
    args = ap.parse_args(argv)
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.num_samples > 1 and args.temperature <= 0.0:
        raise ValueError(
            "--num-samples > 1 requires --temperature > 0 for pass@k"
        )
    if args.top_k is not None and args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if args.top_p is not None and not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1]")
    if args.prompt_mode in FIM_PROMPT_MODES and args.prompt_sidecars != "zero":
        raise ValueError(
            "FIM/IFIM prompt modes require --prompt-sidecars zero"
        )
    if args.prompt_mode in FIM_PROMPT_MODES and args.prepend_code_start:
        raise ValueError(
            "FIM/IFIM prompt modes cannot use --prepend-code-start"
        )
    return args


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    cases_dir = args.cases.resolve().parent
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_graph_cache_dir = (
        args.prompt_graph_cache_dir
        if args.prompt_graph_cache_dir is not None
        else out_dir / "prompt_graph_cache"
    )
    prompt_index_cache_dir = (
        args.prompt_index_cache_dir
        if args.prompt_index_cache_dir is not None
        else out_dir / "prompt_index_cache"
    )
    case_graph_modes = [
        effective_case_prompt_graph_mode(case, args.prompt_graph_mode)
        for case in cases
    ]
    has_graph_cases = any(mode == "repo" for mode in case_graph_modes)
    args.require_graph_routes = batch_requires_graph_routes(case_graph_modes)
    args.graph_routes_enabled = has_graph_cases
    if has_graph_cases and args.prompt_sidecars != "zero":
        raise ValueError(
            "repository prompt graph cases cannot be combined with "
            "--prompt-sidecars clang"
        )
    if has_graph_cases and args.prepend_code_start:
        raise ValueError(
            "repository prompt graph cases cannot be combined with "
            "--prepend-code-start"
        )
    if not has_graph_cases:
        prompt_graph_cache_dir = None
        prompt_index_cache_dir = None

    for case, case_graph_mode in zip(cases, case_graph_modes):
        case_prompt = evaluation_prompt(case, args.prompt_mode)
        resolved = resolve_case_prompt_graph(
            case,
            cases_dir=cases_dir,
            mode=case_graph_mode,
            prompt_index_cache_dir=prompt_index_cache_dir,
            indexer_root=args.clang_indexer_root,
        )
        if resolved is not None and case_prompt.is_fim:
            _resolve_source_suffix_start(
                resolved.project_index,
                case_prompt,
                document_id=resolved.document_id,
                source_start=resolved.source_start,
            )

    tokenizer = load_cppmega_tokenizer(args.tokenizer)
    model = build_model(args)
    model_dtype = model_graph_bias_dtype(model)
    if args.bf16 and model_dtype != mx.bfloat16:
        raise RuntimeError(
            "--bf16 requested but loaded model token embeddings have dtype "
            f"{model_dtype}"
        )
    graph_bias_dtype = mx.float32

    completions = out_dir / "completions.jsonl"
    report = out_dir / "compile_report.json"
    rows = write_completions(
        cases,
        completions,
        model=model,
        tokenizer=tokenizer,
        prompt_mode=args.prompt_mode,
        seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        prompt_graph_mode=args.prompt_graph_mode,
        prompt_sidecars=args.prompt_sidecars,
        prepend_code_start=args.prepend_code_start,
        cases_dir=cases_dir,
        prompt_graph_cache_dir=prompt_graph_cache_dir,
        prompt_index_cache_dir=prompt_index_cache_dir,
        indexer_root=args.clang_indexer_root,
        num_samples=args.num_samples,
        base_seed=args.seed,
        graph_bias_dtype=graph_bias_dtype,
    )
    summary = {
        "cases": len(cases),
        "completions": len(rows),
        "candidate_identity": ["task_id", "sample_index", "seed"],
        "num_samples": args.num_samples,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "prompt_mode": args.prompt_mode,
        "seq_len": args.seq_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "prompt_graph_mode": args.prompt_graph_mode,
        "prompt_graph_case_counts": {
            mode: case_graph_modes.count(mode) for mode in ("repo", "off")
        },
        "prompt_graph_cache_dir": (
            None
            if prompt_graph_cache_dir is None
            else str(prompt_graph_cache_dir)
        ),
        "prompt_index_cache_dir": (
            None
            if prompt_index_cache_dir is None
            else str(prompt_index_cache_dir)
        ),
        "prompt_sidecars": args.prompt_sidecars,
        "graph_bias_dtype": _dtype_name(graph_bias_dtype),
        "attention_mask_bias_dtype": _dtype_name(model_dtype),
        "finish_reason_counts": {
            reason: sum(row["finish_reason"] == reason for row in rows)
            for reason in ("eos", "length")
        },
        "length_truncated": sum(row["length_truncated"] for row in rows),
    }
    summary_path = out_dir / "generation_summary.json"
    if args.skip_compile_gate:
        summary["compile_gate"] = {"skipped": True}
    else:
        summary["compile_gate_receipt"] = build_compile_gate_receipt(
            script=args.compile_gate_script,
            cases=args.cases,
            completions=completions,
        )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.skip_compile_gate:
        try:
            compile_payload = run_compile_gate(
                cases=args.cases,
                completions=completions,
                report=report,
                script=args.compile_gate_script,
                keep_workdir=args.keep_workdir,
                expected_receipt=summary["compile_gate_receipt"],
            )
        except subprocess.CalledProcessError:
            if report.is_file():
                compile_payload = json.loads(report.read_text(encoding="utf-8"))
                summary["compile_gate_summary"] = compile_payload["summary"]
                summary_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            raise
        summary["compile_gate_summary"] = compile_payload["summary"]
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
