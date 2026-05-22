"""Token-id logits processors for constrained local inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx
import numpy as np


TokenIdSet = int | Sequence[int]


class LogitsProcessor(Protocol):
    """Callable hook that can mask or transform next-token logits."""

    def __call__(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Return processed ``(batch, vocab)`` logits for the current prefix."""


@dataclass(frozen=True)
class JsonTokenIds:
    """Token-id categories used by the scoped JSON constrained decoder.

    The processor works in token coordinates. The caller owns tokenizer-specific
    mapping from raw text pieces to these categories; this class deliberately
    does not decode text or claim JSON Schema support.
    """

    object_start: int
    object_end: int
    array_start: int
    array_end: int
    colon: int
    comma: int
    string: TokenIdSet
    number: TokenIdSet
    true_literal: TokenIdSet
    false_literal: TokenIdSet
    null_literal: TokenIdSet
    whitespace: Sequence[int] = ()
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "object_start",
            "object_end",
            "array_start",
            "array_end",
            "colon",
            "comma",
        ):
            _validate_single_token_id(getattr(self, name), name)
        for name in (
            "string",
            "number",
            "true_literal",
            "false_literal",
            "null_literal",
        ):
            _normalize_token_ids(getattr(self, name), name)
        _normalize_token_ids(tuple(self.whitespace), "whitespace", allow_empty=True)
        if self.eos_token_id is not None:
            _validate_single_token_id(self.eos_token_id, "eos_token_id")


class JsonConstrainedLogitsProcessor:
    """Mask logits to token IDs that keep a token-category JSON prefix valid."""

    def __init__(self, token_ids: JsonTokenIds, *, start_position: int = 0) -> None:
        if start_position < 0:
            raise ValueError("start_position must be non-negative")
        self.token_ids = token_ids
        self.start_position = start_position
        self._whitespace = _normalize_token_ids(
            tuple(token_ids.whitespace),
            "whitespace",
            allow_empty=True,
        )
        self._string = _normalize_token_ids(token_ids.string, "string")
        self._number = _normalize_token_ids(token_ids.number, "number")
        self._true = _normalize_token_ids(token_ids.true_literal, "true_literal")
        self._false = _normalize_token_ids(token_ids.false_literal, "false_literal")
        self._null = _normalize_token_ids(token_ids.null_literal, "null_literal")
        self._scalar_values = self._string + self._number + self._true + self._false + self._null
        self._value_starts = (
            token_ids.object_start,
            token_ids.array_start,
            *self._scalar_values,
        )
        self._all_configured_ids = tuple(
            dict.fromkeys(
                (
                    token_ids.object_start,
                    token_ids.object_end,
                    token_ids.array_start,
                    token_ids.array_end,
                    token_ids.colon,
                    token_ids.comma,
                    *self._scalar_values,
                    *self._whitespace,
                    *((token_ids.eos_token_id,) if token_ids.eos_token_id is not None else ()),
                )
            )
        )

    def allowed_token_ids(self, tokens: mx.array) -> tuple[int, ...]:
        """Return valid next token IDs for one batch row prefix."""

        rows = _tokens_to_rows(tokens, start_position=self.start_position)
        if len(rows) != 1:
            raise ValueError("allowed_token_ids expects a single batch row")
        return self._allowed_for_prefix(rows[0])

    def __call__(self, logits: mx.array, tokens: mx.array) -> mx.array:
        if len(logits.shape) != 2:
            raise ValueError("logits must have shape (batch, vocab)")
        rows = _tokens_to_rows(tokens, start_position=self.start_position)
        if len(rows) != int(logits.shape[0]):
            raise ValueError("tokens batch size must match logits batch size")

        vocab_size = int(logits.shape[1])
        for token_id in self._all_configured_ids:
            if token_id >= vocab_size:
                raise ValueError("JSON token id is outside logits vocabulary")

        vocab_ids = mx.arange(vocab_size, dtype=mx.int32)
        row_masks: list[mx.array] = []
        for row in rows:
            allowed = self._allowed_for_prefix(row)
            if not allowed:
                raise ValueError("JSON constraint produced no allowed token ids")
            allowed_ids = mx.array(allowed, dtype=mx.int32)
            row_masks.append(mx.any(vocab_ids[:, None] == allowed_ids[None, :], axis=1))

        mask = mx.stack(row_masks, axis=0)
        neg_inf = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
        return mx.where(mask, logits, neg_inf)

    def _allowed_for_prefix(self, row: Sequence[int]) -> tuple[int, ...]:
        stack, root_complete = self._parse_prefix(row)
        allowed: list[int] = list(self._whitespace)
        if not stack:
            if root_complete:
                if self.token_ids.eos_token_id is not None:
                    allowed.append(self.token_ids.eos_token_id)
            else:
                allowed.extend(self._value_starts)
            return _unique(allowed)

        top = stack[-1]
        if top == "object_key_or_end":
            allowed.extend((self.token_ids.object_end, *self._string))
        elif top == "object_colon":
            allowed.append(self.token_ids.colon)
        elif top == "object_value":
            allowed.extend(self._value_starts)
        elif top == "object_comma_or_end":
            allowed.extend((self.token_ids.comma, self.token_ids.object_end))
        elif top == "array_value_or_end":
            allowed.extend((self.token_ids.array_end, *self._value_starts))
        elif top == "array_comma_or_end":
            allowed.extend((self.token_ids.comma, self.token_ids.array_end))
        else:
            raise ValueError(f"invalid JSON parser state: {top}")
        return _unique(allowed)

    def _parse_prefix(self, row: Sequence[int]) -> tuple[list[str], bool]:
        prefix = tuple(int(token_id) for token_id in row)
        stack: list[str] = []
        root_complete = False

        def mark_value_complete() -> None:
            nonlocal root_complete
            if not stack:
                if root_complete:
                    raise ValueError("invalid JSON prefix: multiple root values")
                root_complete = True
                return
            top = stack[-1]
            if top == "object_value":
                stack[-1] = "object_comma_or_end"
            elif top == "array_value_or_end":
                stack[-1] = "array_comma_or_end"
            else:
                raise ValueError("invalid JSON prefix: value in unexpected position")

        def expect_value() -> bool:
            return (
                not stack
                and not root_complete
                or bool(stack)
                and stack[-1] in {"object_value", "array_value_or_end"}
            )

        for token_id in prefix:
            if token_id in self._whitespace:
                continue
            if token_id == self.token_ids.eos_token_id:
                if stack or not root_complete:
                    raise ValueError("invalid JSON prefix: EOS before complete root")
                root_complete = True
                continue
            if not stack and root_complete:
                raise ValueError("invalid JSON prefix: token after complete root")

            if token_id == self.token_ids.object_start:
                if not expect_value():
                    raise ValueError("invalid JSON prefix: object start in unexpected position")
                stack.append("object_key_or_end")
            elif token_id == self.token_ids.array_start:
                if not expect_value():
                    raise ValueError("invalid JSON prefix: array start in unexpected position")
                stack.append("array_value_or_end")
            elif token_id in self._scalar_values:
                if stack and stack[-1] == "object_key_or_end" and token_id in self._string:
                    stack[-1] = "object_colon"
                else:
                    if not expect_value():
                        raise ValueError("invalid JSON prefix: scalar in unexpected position")
                    mark_value_complete()
            elif token_id == self.token_ids.colon:
                if not stack or stack[-1] != "object_colon":
                    raise ValueError("invalid JSON prefix: colon in unexpected position")
                stack[-1] = "object_value"
            elif token_id == self.token_ids.comma:
                if not stack or stack[-1] not in {"object_comma_or_end", "array_comma_or_end"}:
                    raise ValueError("invalid JSON prefix: comma in unexpected position")
                stack[-1] = (
                    "object_key_or_end"
                    if stack[-1] == "object_comma_or_end"
                    else "array_value_or_end"
                )
            elif token_id == self.token_ids.object_end:
                if not stack or stack[-1] not in {"object_key_or_end", "object_comma_or_end"}:
                    raise ValueError("invalid JSON prefix: object end in unexpected position")
                stack.pop()
                mark_value_complete()
            elif token_id == self.token_ids.array_end:
                if not stack or stack[-1] not in {"array_value_or_end", "array_comma_or_end"}:
                    raise ValueError("invalid JSON prefix: array end in unexpected position")
                stack.pop()
                mark_value_complete()
            else:
                raise ValueError(f"invalid JSON prefix: uncategorized token id {token_id}")

        return stack, root_complete


def _validate_single_token_id(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer token id")


def _normalize_token_ids(
    value: TokenIdSet,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer token id")
    if isinstance(value, int):
        tokens = (value,)
    else:
        tokens = tuple(int(token_id) for token_id in value)
    if not tokens and not allow_empty:
        raise ValueError(f"{name} must contain at least one token id")
    for token_id in tokens:
        _validate_single_token_id(token_id, name)
    return tokens


def _tokens_to_rows(
    tokens: mx.array,
    *,
    start_position: int = 0,
) -> tuple[tuple[int, ...], ...]:
    if len(tokens.shape) != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    mx.eval(tokens)
    values = np.array(tokens[:, start_position:])
    return tuple(tuple(int(token_id) for token_id in row) for row in values)


def _unique(token_ids: Sequence[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(token_ids))
