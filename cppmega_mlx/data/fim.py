"""Pure-Python Fill-in-the-Middle token permutations.

This module is a data/preprocessing transform slice only.  It works on token ID
sequences directly and intentionally does not load or vendor tokenizer
artifacts.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias
from typing import Literal

from cppmega_mlx.data.tokenizer_contract import (
    REQUIRED_SPECIAL_TOKEN_IDS,
    SpecialTokenMapping,
    validate_required_special_token_ids,
)

FIM_PREFIX_ID = REQUIRED_SPECIAL_TOKEN_IDS["FIM_PREFIX"]
FIM_MIDDLE_ID = REQUIRED_SPECIAL_TOKEN_IDS["FIM_MIDDLE"]
FIM_SUFFIX_ID = REQUIRED_SPECIAL_TOKEN_IDS["FIM_SUFFIX"]
FIM_INSTRUCTION_ID = REQUIRED_SPECIAL_TOKEN_IDS["FIM_INSTRUCTION"]
EOT_ID = REQUIRED_SPECIAL_TOKEN_IDS["EOT"]

FIMMode = Literal["psm", "spm"]
FIMSpecialTokenInput: TypeAlias = "FIMSpecialTokenIds | SpecialTokenMapping | None"


@dataclass(frozen=True)
class FIMSpecialTokenIds:
    """Fail-closed container for the cppmega FIM/iFIM reserved token IDs."""

    eot: int = EOT_ID
    fim_prefix: int = FIM_PREFIX_ID
    fim_middle: int = FIM_MIDDLE_ID
    fim_suffix: int = FIM_SUFFIX_ID
    fim_instruction: int = FIM_INSTRUCTION_ID

    def __post_init__(self) -> None:
        expected = {
            "EOT": self.eot,
            "FIM_PREFIX": self.fim_prefix,
            "FIM_MIDDLE": self.fim_middle,
            "FIM_SUFFIX": self.fim_suffix,
            "FIM_INSTRUCTION": self.fim_instruction,
        }
        for name, actual_id in expected.items():
            if not isinstance(actual_id, int) or isinstance(actual_id, bool):
                raise ValueError(f"special token {name!r} must use an integer id")
            required_id = REQUIRED_SPECIAL_TOKEN_IDS[name]
            if actual_id != required_id:
                raise ValueError(
                    f"special token {name!r} must use id {required_id}, got {actual_id}"
                )

        seen: dict[int, str] = {}
        for name, token_id in expected.items():
            existing = seen.setdefault(token_id, name)
            if existing != name:
                raise ValueError(
                    f"special token id collision: id {token_id} maps to both "
                    f"{existing!r} and {name!r}"
                )

    @classmethod
    def from_mapping(cls, mapping: SpecialTokenMapping) -> "FIMSpecialTokenIds":
        """Create ids only after validating the full special-token contract."""

        validate_required_special_token_ids(mapping)
        return cls()


FIM_SPECIAL_TOKEN_IDS = FIMSpecialTokenIds()


def apply_fim_permutation(
    token_ids: Sequence[int],
    *,
    span: tuple[int, int],
    mode: FIMMode,
    special_token_ids: FIMSpecialTokenInput = None,
) -> list[int]:
    """Permute ``token_ids`` into PSM or SPM format for an explicit middle span.

    ``span`` is half-open ``[start, end)`` and marks the middle segment to be
    predicted.  Valid spans keep prefix, middle, and suffix non-empty so sampled
    and explicit transforms share the same reference contract.
    """

    ids = _resolve_special_token_ids(special_token_ids)
    _validate_fim_mode(mode)
    start, end = span
    _validate_middle_span(len(token_ids), start, end)

    tokens = list(token_ids)
    prefix = tokens[:start]
    middle = tokens[start:end]
    suffix = tokens[end:]

    if mode == "psm":
        return [
            ids.fim_prefix,
            *prefix,
            ids.fim_suffix,
            *suffix,
            ids.fim_middle,
            *middle,
            ids.eot,
        ]
    return [
        ids.fim_prefix,
        ids.fim_suffix,
        *suffix,
        ids.fim_middle,
        *prefix,
        *middle,
        ids.eot,
    ]


def apply_fim_transform(
    token_ids: Sequence[int],
    *,
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
    seed: int | None = None,
    rng: random.Random | None = None,
    special_token_ids: FIMSpecialTokenInput = None,
) -> list[int]:
    """Apply sampled FIM with deterministic RNG injection.

    Samples shorter than three tokens are returned unchanged because they cannot
    provide non-empty prefix, middle, and suffix segments.
    """

    ids = _resolve_special_token_ids(special_token_ids)
    _validate_rate("fim_rate", fim_rate)
    _validate_rate("spm_rate", spm_rate)
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")

    tokens = list(token_ids)
    if len(tokens) < 3:
        return tokens

    rand = rng if rng is not None else random.Random(seed)
    if rand.random() >= fim_rate:
        return tokens

    start, end = sample_middle_span(len(tokens), rng=rand)
    mode: FIMMode = "spm" if rand.random() < spm_rate else "psm"
    return apply_fim_permutation(
        tokens,
        span=(start, end),
        mode=mode,
        special_token_ids=ids,
    )


def apply_ifim_permutation(
    token_ids: Sequence[int],
    *,
    instruction_token_ids: Sequence[int],
    span: tuple[int, int],
    mode: FIMMode,
    special_token_ids: FIMSpecialTokenInput = None,
) -> list[int]:
    """Permute tokens into instruction-aware FIM format for an explicit span."""

    ids = _resolve_special_token_ids(special_token_ids)
    instruction = _validate_instruction_tokens(instruction_token_ids)
    base = apply_fim_permutation(
        token_ids,
        span=span,
        mode=mode,
        special_token_ids=ids,
    )
    return [ids.fim_instruction, *instruction, *base]


def apply_ifim_transform(
    token_ids: Sequence[int],
    *,
    instruction_token_ids: Sequence[int],
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
    seed: int | None = None,
    rng: random.Random | None = None,
    special_token_ids: FIMSpecialTokenInput = None,
) -> list[int]:
    """Apply sampled instruction-aware FIM with deterministic RNG injection."""

    ids = _resolve_special_token_ids(special_token_ids)
    instruction = _validate_instruction_tokens(instruction_token_ids)
    _validate_rate("fim_rate", fim_rate)
    _validate_rate("spm_rate", spm_rate)
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")

    tokens = list(token_ids)
    if len(tokens) < 3:
        return tokens

    rand = rng if rng is not None else random.Random(seed)
    if rand.random() >= fim_rate:
        return tokens

    start, end = sample_middle_span(len(tokens), rng=rand)
    mode: FIMMode = "spm" if rand.random() < spm_rate else "psm"
    return apply_ifim_permutation(
        tokens,
        instruction_token_ids=instruction,
        span=(start, end),
        mode=mode,
        special_token_ids=ids,
    )


def extract_ifim_instruction_text(source_text: str) -> str | None:
    """Extract a lightweight iFIM instruction from comments or signatures.

    This is intentionally dependency-free and CPU-only.  Tree-sitter/AST-aware
    extraction can be layered on top later without changing token formatting.
    """

    for pattern in (_DOXYGEN_BRIEF_RE, _DOXYGEN_BLOCK_RE, _TRIPLE_DOUBLE_RE):
        match = pattern.search(source_text)
        if match:
            instruction = _clean_instruction(match.group(1))
            if instruction is not None:
                return instruction

    instruction = _extract_leading_comment_instruction(source_text)
    if instruction is not None:
        return instruction

    return _generate_signature_instruction(source_text)


def sample_middle_span(length: int, *, rng: random.Random) -> tuple[int, int]:
    """Sample a half-open middle span with non-empty prefix/middle/suffix."""

    if length < 3:
        raise ValueError("FIM span sampling requires at least 3 tokens")
    start = rng.randint(1, length - 2)
    end = rng.randint(start + 1, length - 1)
    return start, end


def _resolve_special_token_ids(
    special_token_ids: FIMSpecialTokenInput,
) -> FIMSpecialTokenIds:
    if special_token_ids is None:
        return FIM_SPECIAL_TOKEN_IDS
    if isinstance(special_token_ids, FIMSpecialTokenIds):
        return special_token_ids
    return FIMSpecialTokenIds.from_mapping(special_token_ids)


def _validate_instruction_tokens(instruction_token_ids: Sequence[int]) -> list[int]:
    instruction = list(instruction_token_ids)
    if not instruction:
        raise ValueError("iFIM instruction_token_ids must not be empty")
    for token_id in instruction:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError("iFIM instruction_token_ids must be integer token ids")
    return instruction


def _validate_middle_span(length: int, start: int, end: int) -> None:
    if length < 3:
        raise ValueError("FIM permutation requires at least 3 tokens")
    if not 0 < start < end < length:
        raise ValueError(
            "FIM middle span must satisfy 0 < start < end < len(token_ids)"
        )


def _validate_fim_mode(mode: str) -> None:
    if mode not in {"psm", "spm"}:
        raise ValueError("FIM mode must be 'psm' or 'spm'")


def _validate_rate(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


_DOXYGEN_BRIEF_RE = re.compile(
    r"(?:@brief|\\brief)\s+(.+?)(?:\n\s*(?:@|\\\w|$)|\*/)",
    re.DOTALL,
)
_DOXYGEN_BLOCK_RE = re.compile(r"/\*\*\s*(.*?)\*/", re.DOTALL)
_TRIPLE_DOUBLE_RE = re.compile(r'"""(.*?)"""', re.DOTALL)
_LEADING_LINE_COMMENT_RE = re.compile(r"^\s*//\s?(.*)$")
_CPP_FUNC_RE = re.compile(
    r"(?:(?:[\w:*&<>,\s]+)\s+)?([\w:~]+)\s*\(([^)]*)\)",
    re.MULTILINE,
)
_PYTHON_FUNC_RE = re.compile(
    r"def\s+([\w]+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:",
    re.MULTILINE,
)


def _extract_leading_comment_instruction(source_text: str) -> str | None:
    comment_lines: list[str] = []
    for line in source_text.splitlines()[:20]:
        match = _LEADING_LINE_COMMENT_RE.match(line)
        if match is not None:
            text = match.group(1).strip()
            lowered = text.lower()
            if text and not any(
                marker in lowered
                for marker in ("copyright", "license", "all rights reserved")
            ):
                comment_lines.append(text)
            continue
        if comment_lines:
            break

    if not comment_lines:
        return None
    return _clean_instruction(" ".join(comment_lines))


def _generate_signature_instruction(source_text: str) -> str | None:
    match = _PYTHON_FUNC_RE.search(source_text)
    if match is not None:
        instruction = _build_signature_instruction(match.group(1), match.group(2))
        return_type = match.group(3)
        if return_type and return_type.strip() != "None":
            instruction = f"{instruction} and returns {return_type.strip()}"
        return instruction

    match = _CPP_FUNC_RE.search(source_text)
    if match is not None:
        return _build_signature_instruction(match.group(1), match.group(2))
    return None


def _build_signature_instruction(name: str, params: str) -> str:
    clean_name = name.split("::")[-1].lstrip("~")
    words = _split_identifier(clean_name)
    if not words:
        return f"Implement the function {clean_name}"

    base = f"Implement the function {clean_name}"
    if params and len(params) < 100:
        param_names = []
        for param in (part.strip() for part in params.split(",")):
            if not param or param == "self":
                continue
            param = param.split("=")[0].strip()
            if ":" in param:
                name_part = param.split(":", 1)[0].strip()
            else:
                name_part = param.split()[-1].strip("&*")
            name_part = name_part.split("=")[0].strip()
            if name_part and name_part not in {"const", "void"}:
                param_names.append(name_part)
        if param_names and len(param_names) <= 5:
            base = f"{base} taking {', '.join(param_names)}"
    return base


def _split_identifier(name: str) -> list[str]:
    words: list[str] = []
    for part in name.split("_"):
        if not part:
            continue
        matches = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", part)
        words.extend(match.lower() for match in matches)
    return words or [name.lower()]


def _clean_instruction(text: str) -> str | None:
    lines: list[str] = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("*"):
            stripped = stripped[1:].strip()
        if stripped.startswith("@") or stripped.startswith("\\"):
            break
        if stripped:
            lines.append(stripped)

    if not lines:
        return None
    collapsed = re.sub(r"\s+", " ", " ".join(lines)).strip()
    period_idx = collapsed.find(".")
    if 0 < period_idx < len(collapsed) - 1:
        collapsed = collapsed[:period_idx]
    collapsed = collapsed.rstrip(".")
    if len(collapsed) < 3 or len(collapsed) > 300:
        return None
    return collapsed


__all__ = [
    "EOT_ID",
    "FIM_INSTRUCTION_ID",
    "FIMMode",
    "FIM_MIDDLE_ID",
    "FIM_PREFIX_ID",
    "FIM_SPECIAL_TOKEN_IDS",
    "FIMSpecialTokenIds",
    "FIM_SUFFIX_ID",
    "apply_fim_permutation",
    "apply_fim_transform",
    "apply_ifim_permutation",
    "apply_ifim_transform",
    "extract_ifim_instruction_text",
    "sample_middle_span",
]
