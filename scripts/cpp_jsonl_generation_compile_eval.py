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
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx

from cppmega_mlx.data.prompt_graph import (
    PromptGraphArtifact,
    PromptGraphBuilder,
    PromptGraphContext,
    PromptProjectIndex,
)
from cppmega_mlx.inference.sampling import sample_next_token
from cppmega_mlx.inference.side_channels import (
    InferenceSideChannelBuilder,
    get_builtin_code_metadata_adapter,
)
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
from cppmega_mlx.training.checkpoint import load_checkpoint

PromptMode = Literal["source-prefix", "docstring"]
PromptSidecars = Literal["zero", "clang"]
PromptGraphMode = Literal["off", "repo"]

DEFAULT_TOKENIZER = REPO_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
DEFAULT_COMPILE_GATE = REPO_ROOT.parent / "cppmega" / "scripts" / "cpp_generation_compile_eval.py"
SIDE_CHANNEL_NAMES = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
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


def prompt_text(case: dict[str, Any], mode: PromptMode) -> str:
    if mode == "source-prefix":
        value = case.get("source_prefix")
        key = "source_prefix"
    elif mode == "docstring":
        # The model was trained to consume task intent as C/C++ comments around
        # real code, not as a naked English instruction.  Use the case's C++
        # source prefix because it already contains the doc comment and function
        # signature, which also lets clang sidecars align with actual code text.
        value = case.get("source_prefix")
        key = "source_prefix"
    else:
        raise ValueError(f"unsupported prompt mode {mode!r}")
    if not isinstance(value, str) or not value:
        raise ValueError(f"case {case.get('task_id')!r}: missing non-empty {key}")
    return value


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


def resolve_case_prompt_graph(
    case: dict[str, Any],
    *,
    cases_dir: Path,
    mode: PromptGraphMode,
) -> tuple[PromptProjectIndex | None, int | None]:
    if mode == "off":
        return None, None
    if mode != "repo":
        raise ValueError(f"unsupported prompt graph mode {mode!r}")

    task_id = str(case.get("task_id") or "<unknown>")
    raw_index = case.get("prompt_graph_index")
    if not isinstance(raw_index, str) or not raw_index:
        raise ValueError(
            f"case {task_id!r}: missing non-empty prompt_graph_index "
            "while prompt graph mode is repo"
        )
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
    project_index.verify_source_file(index_path.parent)

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
    if project_index.source[source_start:source_end] != prompt:
        raise ValueError(
            f"case {task_id!r}: source_prefix does not match project index "
            f"source span [{source_start},{source_end})"
        )
    return project_index, source_start


def default_side_channels(seq_len: int) -> dict[str, mx.array]:
    """Zero/default token sidecars for standalone prompt-only evals."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return {
        name: mx.zeros((1, seq_len), dtype=mx.int32)
        for name in SIDE_CHANNEL_NAMES
    }


def _row_to_ints(array: mx.array) -> list[int]:
    if array.ndim != 2 or int(array.shape[0]) != 1:
        raise ValueError(f"expected side channel shaped (1, S), got {array.shape}")
    return [int(item) for item in array[0].tolist()]


def build_prompt_context(
    tokenizer: Any,
    prompt: str,
    *,
    prompt_graph_mode: PromptGraphMode = "off",
    prompt_sidecars: PromptSidecars,
    prepend_code_start: bool,
    project_index: PromptProjectIndex | None = None,
    prompt_source_start: int | None = None,
    prompt_graph_cache_dir: Path | None = None,
) -> GenerationPromptContext:
    """Encode a prompt and build token-aligned prompt sidecars.

    Generated suffix tokens get zero/default sidecars until the full candidate is
    reparsed by the compile gate. Prompt tokens, however, should carry the same
    structure channels the model saw during training when requested.
    """

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
        if prompt_graph_cache_dir is None:
            raise ValueError(
                "prompt graph mode repo requires a deterministic cache directory"
            )
        artifact = PromptGraphBuilder(
            tokenizer,
            cache_dir=prompt_graph_cache_dir,
        ).build(
            project_index,
            PromptGraphContext.from_prompt(
                prompt,
                source_start=prompt_source_start,
                language="cpp",
            ),
        )
        side_channels = {
            name: list(artifact.side_channels[name])
            for name in SIDE_CHANNEL_NAMES
        }
        receipt = dict(artifact.receipt)
        receipt["prompt_graph_mode"] = "repo"
        return GenerationPromptContext(
            token_ids=list(artifact.token_ids),
            side_channels=side_channels,
            receipt=receipt,
            graph_artifact=artifact,
        )

    if prompt_graph_mode != "off":
        raise ValueError(f"unsupported prompt_graph_mode={prompt_graph_mode!r}")

    if prompt_sidecars == "zero":
        prepend = tokenizer.code_start_id if prepend_code_start else None
        ids = list(tokenizer.encode(prompt, prepend=prepend))
        side = {name: [0] * len(ids) for name in SIDE_CHANNEL_NAMES}
        return GenerationPromptContext(
            token_ids=ids,
            side_channels=side,
            receipt={
                "prompt_graph_mode": "off",
                "prompt_sidecars": "zero",
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
    result = builder.build(prompt, language="cpp")
    ids = _row_to_ints(result.prompt_ids)
    zero = [0] * len(ids)
    side: dict[str, list[int]] = {}
    for name in SIDE_CHANNEL_NAMES:
        value = result.model_kwargs.get(name)
        side[name] = _row_to_ints(value) if isinstance(value, mx.array) else list(zero)
    receipt = dict(result.provenance)
    receipt["prompt_graph_mode"] = "off"
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
            name: mx.array([graph_inputs.side_channels[name]], dtype=mx.int32)
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
    side = {
        name: mx.array(
            [
                (list(context.side_channels[name]) + [0] * generated)[
                    window_start:window_end
                ]
            ],
            dtype=mx.int32,
        )
        for name in SIDE_CHANNEL_NAMES
    }
    receipt = {
        **context.receipt,
        "window_start": window_start,
        "window_end": window_end,
        "total_token_count": total_token_count,
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
        require_graph_routes=args.prompt_graph_mode == "repo",
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
    prompt: str,
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
    constraints = BodyDecodeConstraints(tokenizer, prompt_len=len(prompt_ids))
    for _ in range(max_new_tokens):
        window_start = max(0, len(token_context) - seq_len)
        window = token_context[window_start:]
        input_ids = mx.array([window], dtype=mx.int32)
        side, block_bias, _window_receipt = prompt_model_inputs(
            prompt_context,
            total_token_count=len(token_context),
            window_start=window_start,
            window_end=len(token_context),
        )
        logits, _loss = model(
            input_ids,
            block_bias=block_bias,
            **side,
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
    return (
        trim_body_completion(tokenizer.decode(generated)),
        len(prompt_ids),
        len(generated),
        dict(prompt_context.receipt),
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
) -> list[dict[str, Any]]:
    completions_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with completions_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            project_index, prompt_source_start = resolve_case_prompt_graph(
                case,
                cases_dir=cases_dir,
                mode=prompt_graph_mode,
            )
            completion, prompt_tokens, generated_tokens, side_provenance = generate_completion(
                model,
                tokenizer,
                prompt_text(case, prompt_mode),
                seq_len=seq_len,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                prompt_graph_mode=prompt_graph_mode,
                prompt_sidecars=prompt_sidecars,
                prepend_code_start=prepend_code_start,
                project_index=project_index,
                prompt_source_start=prompt_source_start,
                prompt_graph_cache_dir=prompt_graph_cache_dir,
            )
            row = {
                "task_id": case["task_id"],
                "completion": completion,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "prompt_graph_receipt": side_provenance,
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
    ]
    if keep_workdir:
        cmd.append("--keep-workdir")
    subprocess.run(cmd, check=True, env=compile_gate_env())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "mlx_generation_eval")
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--compile-gate-script", type=Path, default=DEFAULT_COMPILE_GATE)
    ap.add_argument("--skip-compile-gate", action="store_true")
    ap.add_argument("--keep-workdir", action="store_true")
    ap.add_argument("--prompt-mode", choices=("source-prefix", "docstring"), default="source-prefix")
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
    args = ap.parse_args()
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if args.top_k is not None and args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if args.top_p is not None and not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1]")
    if args.prompt_graph_mode == "repo" and args.prompt_sidecars != "zero":
        raise ValueError(
            "--prompt-graph-mode repo cannot be combined with "
            "--prompt-sidecars clang"
        )
    if args.prompt_graph_mode == "repo" and args.prepend_code_start:
        raise ValueError(
            "--prompt-graph-mode repo cannot be combined with "
            "--prepend-code-start"
        )
    return args


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    cases_dir = args.cases.resolve().parent
    for case in cases:
        resolve_case_prompt_graph(
            case,
            cases_dir=cases_dir,
            mode=args.prompt_graph_mode,
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_graph_cache_dir = (
        args.prompt_graph_cache_dir
        if args.prompt_graph_cache_dir is not None
        else out_dir / "prompt_graph_cache"
    )
    if args.prompt_graph_mode == "off":
        prompt_graph_cache_dir = None

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
        "prompt_graph_cache_dir": (
            None
            if prompt_graph_cache_dir is None
            else str(prompt_graph_cache_dir)
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
