#!/usr/bin/env python3
"""Generate C/C++ completions from a local MLX checkpoint and run compile gate.

This is the MLX-side counterpart of the Nebius/H200 Megatron generation wrapper:
it consumes the same ``cpp_docstring_compile_cases.jsonl`` shape, writes
``completions.jsonl``, then optionally invokes the shared compile/run gate from
``../cppmega/scripts/cpp_generation_compile_eval.py``.

Only native MLX DenseCppLM checkpoints are supported here. Megatron ``torch_dist``
checkpoints must be converted before this script can evaluate them locally.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
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
DEFAULT_COMPILE_GATE = REPO_ROOT.parent / "cppmega" / "scripts" / "cpp_generation_compile_eval.py"
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
        project_index.verify_repository(repository_root)
        index_receipt = dict(project_index.provenance)
    else:
        if prompt_index_cache_dir is None:
            raise ValueError(
                f"case {task_id!r}: automatic prompt graph index build requires "
                "a deterministic prompt_index_cache_dir"
            )
        project_id = require_prompt_graph_project_id(
            case.get("prompt_graph_project_id"),
            where=f"case {task_id!r} prompt_graph_project_id",
        )
        built = ClangPromptProjectIndexProducer(
            cache_dir=prompt_index_cache_dir,
            indexer_root=indexer_root,
            strict_diagnostics=True,
        ).build(repository_root, project_id=project_id)
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
) -> tuple[dict[str, mx.array], mx.array | None, dict[str, Any]]:
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
        block_bias = mx.array(
            [graph_inputs.dense_attention_bias()],
            dtype=mx.float32,
        )
        return side, block_bias, dict(graph_inputs.receipt)

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
    return side, None, receipt


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
) -> tuple[str, int, int, dict[str, Any]]:
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
    constraints = BodyDecodeConstraints(tokenizer, prompt_len=len(prompt_ids))
    for _ in range(max_new_tokens):
        window_start = max(0, len(token_context) - seq_len)
        window = token_context[window_start:]
        input_ids = mx.array([window], dtype=mx.int32)
        side, block_bias, last_window_receipt = prompt_model_inputs(
            prompt_context,
            total_token_count=len(token_context),
            window_start=window_start,
            window_end=len(token_context),
        )
        graph_routes_enabled = bool(
            getattr(
                getattr(model, "config", None),
                "graph_routes_enabled",
                block_bias is not None,
            )
        )
        if graph_routes_enabled and block_bias is None:
            block_bias = mx.zeros(
                (1, len(window), len(window)), dtype=mx.float32
            )
        edge_kind_bias = (
            None if block_bias is None else mx.zeros_like(block_bias)
        )
        logits, _loss = model(
            input_ids,
            block_bias=block_bias,
            edge_kind_bias=edge_kind_bias,
            # The assembled repository prompt is one intentional inference
            # document. Source-document identity remains in source_doc_ids;
            # packed-document IDs must not split cross-file graph context.
            document_ids=mx.ones_like(input_ids),
            **{name: side[name] for name in MODEL_SIDE_CHANNEL_NAMES},
        )
        next_id = _sample_next(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tokens=input_ids,
            logits_processors=[constraints],
        )
        mx.eval(logits)
        if next_id in {tokenizer.eos_token_id, tokenizer.code_end_id}:
            break
        generated.append(next_id)
        token_context.append(next_id)
    receipt = dict(prompt_context.receipt)
    receipt["aligned_side_channels"] = list(SIDE_CHANNEL_NAMES)
    receipt["model_consumed_side_channels"] = list(MODEL_SIDE_CHANNEL_NAMES)
    receipt["model_consumed_runtime_channels"] = [
        "document_ids",
        "graph_attention_bias",
        "graph_edge_kind_bias",
    ]
    if last_window_receipt is not None:
        receipt["last_decode_window"] = last_window_receipt
    return (
        trim_body_completion(tokenizer.decode(generated)),
        len(prompt_ids),
        len(generated),
        receipt,
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
) -> list[dict[str, Any]]:
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
            completion, prompt_tokens, generated_tokens, side_provenance = generate_completion(
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
                project_index=(None if resolved is None else resolved.project_index),
                prompt_document_id=(None if resolved is None else resolved.document_id),
                prompt_source_path=(None if resolved is None else resolved.source_path),
                prompt_source_start=(None if resolved is None else resolved.source_start),
                prompt_graph_cache_dir=prompt_graph_cache_dir,
            )
            row = {
                "task_id": case["task_id"],
                "completion": completion,
                "completion_source": "model_generation",
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
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


def run_compile_gate(
    *,
    cases: Path,
    completions: Path,
    report: Path,
    script: Path,
    keep_workdir: bool,
) -> None:
    if not script.is_file():
        raise FileNotFoundError(f"compile gate script not found: {script}")
    cmd = [
        sys.executable,
        str(script),
        "--cases",
        str(cases),
        "--completions",
        str(completions),
        "--out",
        str(report),
        "--json",
        "--fail-on-fail",
    ]
    if keep_workdir:
        cmd.append("--keep-workdir")
    subprocess.run(cmd, check=True, env=compile_gate_env())


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
    ap.add_argument("--temperature", type=float, default=0.0)
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
    )
    summary = {
        "cases": len(cases),
        "completions": len(rows),
        "checkpoint": str(args.checkpoint),
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
    }
    (out_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.skip_compile_gate:
        run_compile_gate(
            cases=args.cases,
            completions=completions,
            report=report,
            script=args.compile_gate_script,
            keep_workdir=args.keep_workdir,
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
