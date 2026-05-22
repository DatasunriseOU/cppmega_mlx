"""Cppmega C++ tool-use prompt templates and loss masks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cppmega_mlx.data.tokenizer_contract import TOOL_USE_SPECIAL_TOKEN_IDS


TOOL_USE_SPECIAL_TOKEN_TEXT: dict[str, str] = {
    "bos": "<BOS>",
    "eos": "<EOS>",
    "code_start": "<CODE_START>",
    "code_end": "<CODE_END>",
    "think_start": "<THINK_START>",
    "think_end": "<THINK_END>",
    "query_tool": "<QUERY_TOOL>",
    "tool_result": "<TOOL_RESULT>",
}

_RESERVED_TOOL_USE_TOKENS = frozenset(TOOL_USE_SPECIAL_TOKEN_TEXT.values())


@dataclass(frozen=True)
class ToolUseSpecialTokenIds:
    """Stable IDs for the deployed cppmega tool-use tokenizer protocol."""

    bos: int = TOOL_USE_SPECIAL_TOKEN_IDS["BOS"]
    eos: int = TOOL_USE_SPECIAL_TOKEN_IDS["EOS"]
    code_start: int = TOOL_USE_SPECIAL_TOKEN_IDS["CODE_START"]
    code_end: int = TOOL_USE_SPECIAL_TOKEN_IDS["CODE_END"]
    think_start: int = TOOL_USE_SPECIAL_TOKEN_IDS["THINK_START"]
    think_end: int = TOOL_USE_SPECIAL_TOKEN_IDS["THINK_END"]
    query_tool: int = TOOL_USE_SPECIAL_TOKEN_IDS["QUERY_TOOL"]
    tool_result: int = TOOL_USE_SPECIAL_TOKEN_IDS["TOOL_RESULT"]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, int]) -> "ToolUseSpecialTokenIds":
        """Build IDs from a mapping using either canonical or lowercase keys."""

        return cls(
            bos=_lookup_id(mapping, "BOS", "bos"),
            eos=_lookup_id(mapping, "EOS", "EOT", "eos"),
            code_start=_lookup_id(mapping, "CODE_START", "code_start"),
            code_end=_lookup_id(mapping, "CODE_END", "code_end"),
            think_start=_lookup_id(mapping, "THINK_START", "think_start"),
            think_end=_lookup_id(mapping, "THINK_END", "think_end"),
            query_tool=_lookup_id(mapping, "QUERY_TOOL", "query_tool"),
            tool_result=_lookup_id(mapping, "TOOL_RESULT", "tool_result"),
        )

    def as_mapping(self) -> dict[str, int]:
        return {
            "bos": self.bos,
            "eos": self.eos,
            "code_start": self.code_start,
            "code_end": self.code_end,
            "think_start": self.think_start,
            "think_end": self.think_end,
            "query_tool": self.query_tool,
            "tool_result": self.tool_result,
        }


@dataclass(frozen=True)
class ToolUseBlock:
    """One ordered cppmega tool-use turn.

    Fields are rendered in protocol order: model thinking, tool query, injected
    tool result, then final or intermediate code.
    """

    think: str | None = None
    query_tool: str | None = None
    tool_result: str | None = None
    code: str | None = None


def render_tool_use_template(
    instruction: str,
    blocks: Sequence[ToolUseBlock] = (),
    *,
    include_bos: bool = True,
    include_eos: bool = True,
    allow_special_tokens: bool = False,
) -> str:
    """Render cppmega's C++ tool-call protocol as plain text."""

    if not allow_special_tokens:
        _reject_reserved_tokens("instruction", instruction)
        for block_index, block in enumerate(blocks):
            _reject_block_reserved_tokens(block_index, block)

    parts: list[str] = []
    if include_bos:
        parts.append(TOOL_USE_SPECIAL_TOKEN_TEXT["bos"])
    if instruction:
        parts.append(instruction.rstrip("\n"))

    for block in blocks:
        _append_block(parts, block)

    if include_eos:
        parts.append(TOOL_USE_SPECIAL_TOKEN_TEXT["eos"])

    return "\n".join(parts)


def encode_tool_use_template(
    tokenizer: Any,
    instruction: str,
    blocks: Sequence[ToolUseBlock] = (),
    *,
    include_bos: bool = True,
    include_eos: bool = True,
    allow_special_tokens: bool = False,
) -> list[int]:
    """Render and encode with a caller-owned tokenizer."""

    rendered = render_tool_use_template(
        instruction,
        blocks,
        include_bos=include_bos,
        include_eos=include_eos,
        allow_special_tokens=allow_special_tokens,
    )
    encoded = tokenizer.encode(rendered)
    if not isinstance(encoded, list) or any(isinstance(item, list) for item in encoded):
        raise TypeError("tokenizer.encode(text) must return a flat list[int]")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in encoded):
        raise TypeError("tokenizer.encode(text) must return integer token IDs")
    return encoded


def compute_tool_use_loss_mask(
    token_ids: Sequence[int],
    special_ids: ToolUseSpecialTokenIds | Mapping[str, int] | None = None,
) -> list[int]:
    """Compute next-token loss mask for cppmega tool-use SFT examples."""

    ids = _normalize_special_ids(special_ids)
    response_start_tokens = {ids.think_start, ids.code_start, ids.query_tool}
    mask = [0] * len(token_ids)

    response_start = len(token_ids)
    for index, token_id in enumerate(token_ids):
        if int(token_id) in response_start_tokens:
            response_start = index
            break

    in_tool_result = False
    for index in range(response_start, len(token_ids)):
        token_id = int(token_ids[index])

        if token_id == ids.bos or token_id == ids.eos:
            mask[index] = 0
            continue

        if token_id == ids.tool_result:
            in_tool_result = True
            mask[index] = 0
            continue

        if in_tool_result:
            if token_id == ids.code_end:
                in_tool_result = False
                mask[index] = 0
            elif token_id in response_start_tokens:
                in_tool_result = False
                mask[index] = 1
            else:
                mask[index] = 0
            continue

        mask[index] = 1

    return mask


def _append_block(parts: list[str], block: ToolUseBlock) -> None:
    if block.think is not None:
        parts.append(TOOL_USE_SPECIAL_TOKEN_TEXT["think_start"])
        if block.think:
            parts.append(block.think.rstrip("\n"))
        parts.append(TOOL_USE_SPECIAL_TOKEN_TEXT["think_end"])

    if block.query_tool is not None:
        query = block.query_tool.strip()
        parts.append(f'{TOOL_USE_SPECIAL_TOKEN_TEXT["query_tool"]} {query} {TOOL_USE_SPECIAL_TOKEN_TEXT["code_end"]}')

    if block.tool_result is not None:
        parts.append(TOOL_USE_SPECIAL_TOKEN_TEXT["tool_result"])
        if block.tool_result:
            parts.append(block.tool_result.rstrip("\n"))
        parts.append(TOOL_USE_SPECIAL_TOKEN_TEXT["code_end"])

    if block.code is not None:
        parts.append(TOOL_USE_SPECIAL_TOKEN_TEXT["code_start"])
        if block.code:
            parts.append(block.code.rstrip("\n"))
        parts.append(TOOL_USE_SPECIAL_TOKEN_TEXT["code_end"])


def _reject_block_reserved_tokens(block_index: int, block: ToolUseBlock) -> None:
    for field_name in ("think", "query_tool", "tool_result", "code"):
        value = getattr(block, field_name)
        if value is not None:
            _reject_reserved_tokens(f"blocks[{block_index}].{field_name}", value)


def _reject_reserved_tokens(field_name: str, value: str) -> None:
    for token in _RESERVED_TOOL_USE_TOKENS:
        if token in value:
            raise ValueError(
                f"{field_name} contains reserved tool-use token {token!r}; "
                "pass allow_special_tokens=True only for trusted preformatted data"
            )


def _normalize_special_ids(
    special_ids: ToolUseSpecialTokenIds | Mapping[str, int] | None,
) -> ToolUseSpecialTokenIds:
    if special_ids is None:
        return ToolUseSpecialTokenIds()
    if isinstance(special_ids, ToolUseSpecialTokenIds):
        return special_ids
    return ToolUseSpecialTokenIds.from_mapping(special_ids)


def _lookup_id(mapping: Mapping[str, int], *keys: str) -> int:
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"special token id {key!r} must be an int")
            return value
    raise ValueError(f"missing tool-use special token id for {keys[0]!r}")


__all__ = [
    "TOOL_USE_SPECIAL_TOKEN_TEXT",
    "ToolUseBlock",
    "ToolUseSpecialTokenIds",
    "compute_tool_use_loss_mask",
    "encode_tool_use_template",
    "render_tool_use_template",
]
