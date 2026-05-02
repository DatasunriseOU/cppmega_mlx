"""Cppmega tokenizer wrapper with M0.1 contract checks.

The loader intentionally refuses artifacts that do not exactly match the
documented M0.1 tokenizer contract.  This avoids silently training against a
nearby nanochat tokenizer with different reserved IDs.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

EXPECTED_VOCAB_SIZE = 65_536
EXPECTED_SPECIAL_TOKENS: dict[str, int] = {
    "<BOS>": 2,
    "<EOS>": 3,
    "<FIM_PREFIX>": 4,
    "<FIM_MIDDLE>": 5,
    "<FIM_SUFFIX>": 6,
    "<FIM_INSTRUCTION>": 7,
}


class TokenizerContractError(ValueError):
    """Raised when a tokenizer artifact does not satisfy M0.1."""


class CppMegaTokenizer:
    """Thin wrapper around ``tokenizers.Tokenizer`` with stable cppmega APIs."""

    def __init__(self, tokenizer: Any, *, path: Path):
        self._tokenizer = tokenizer
        self.path = path
        self._vocab: dict[str, int] = dict(tokenizer.get_vocab())
        self._id_to_token = {token_id: token for token, token_id in self._vocab.items()}

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size(with_added_tokens=True))

    @property
    def bos_token_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<EOS>"]

    @property
    def fim_prefix_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<FIM_PREFIX>"]

    @property
    def fim_middle_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<FIM_MIDDLE>"]

    @property
    def fim_suffix_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<FIM_SUFFIX>"]

    @property
    def fim_instruction_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<FIM_INSTRUCTION>"]

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def encode(
        self,
        text: str | Sequence[str],
        *,
        prepend: int | str | None = None,
        append: int | str | None = None,
    ) -> list[int] | list[list[int]]:
        prepend_id = self._resolve_optional_token(prepend)
        append_id = self._resolve_optional_token(append)

        if isinstance(text, str):
            ids = list(self._tokenizer.encode(text).ids)
            return self._with_optional_tokens(ids, prepend_id, append_id)
        if isinstance(text, Sequence):
            rows = [list(encoded.ids) for encoded in self._tokenizer.encode_batch(list(text))]
            return [
                self._with_optional_tokens(row, prepend_id, append_id) for row in rows
            ]
        raise TypeError(f"text must be str or sequence[str], got {type(text).__name__}")

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [list(encoded.ids) for encoded in self._tokenizer.encode_batch(list(texts))]

    def decode(self, ids: Iterable[int]) -> str:
        return str(self._tokenizer.decode(list(ids)))

    def token_for_id(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def id_for_token(self, token: str) -> int | None:
        return self._vocab.get(token)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.encode(*args, **kwargs)

    def _resolve_optional_token(self, token: int | str | None) -> int | None:
        if token is None:
            return None
        if isinstance(token, int) and not isinstance(token, bool):
            return token
        if isinstance(token, str):
            token_id = self.id_for_token(token)
            if token_id is None:
                raise TokenizerContractError(f"unknown special token {token!r}")
            return token_id
        raise TypeError(f"token must be int or str, got {type(token).__name__}")

    @staticmethod
    def _with_optional_tokens(
        ids: list[int], prepend_id: int | None, append_id: int | None
    ) -> list[int]:
        if prepend_id is not None:
            ids.insert(0, prepend_id)
        if append_id is not None:
            ids.append(append_id)
        return ids


def load_cppmega_tokenizer(path: str | Path) -> CppMegaTokenizer:
    """Load a tokenizer only if it satisfies the M0.1 contract."""

    tokenizer_path = _resolve_tokenizer_path(path)
    payload = _load_tokenizer_json(tokenizer_path)
    vocab = _extract_vocab(payload, tokenizer_path)
    _validate_vocab_contract(vocab, tokenizer_path)

    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as exc:
        raise TokenizerContractError(
            "tokenizers package is required to load cppmega tokenizer artifacts"
        ) from exc

    return CppMegaTokenizer(Tokenizer.from_file(str(tokenizer_path)), path=tokenizer_path)


def _resolve_tokenizer_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "tokenizer.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"tokenizer artifact not found: {candidate}")
    return candidate


def _load_tokenizer_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise TokenizerContractError(f"{path}: invalid tokenizer JSON") from exc
    if not isinstance(payload, dict):
        raise TokenizerContractError(f"{path}: tokenizer JSON must be an object")
    return payload


def _extract_vocab(payload: dict[str, Any], path: Path) -> dict[str, int]:
    model = payload.get("model")
    if not isinstance(model, dict):
        raise TokenizerContractError(f"{path}: missing tokenizer model")
    raw_vocab = model.get("vocab")
    if not isinstance(raw_vocab, dict):
        raise TokenizerContractError(f"{path}: missing tokenizer model vocab")

    vocab: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in raw_vocab.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise TokenizerContractError(f"{path}: vocab entries must be str->int")
        if token_id in seen_ids:
            raise TokenizerContractError(f"{path}: duplicate vocab id {token_id}")
        seen_ids.add(token_id)
        vocab[token] = token_id
    return vocab


def _validate_vocab_contract(vocab: dict[str, int], path: Path) -> None:
    if len(vocab) != EXPECTED_VOCAB_SIZE:
        raise TokenizerContractError(
            f"{path}: expected vocab size {EXPECTED_VOCAB_SIZE}, got {len(vocab)}"
        )

    id_to_token = {token_id: token for token, token_id in vocab.items()}
    if len(id_to_token) != len(vocab):
        raise TokenizerContractError(f"{path}: vocab ids must be unique")

    for token, expected_id in EXPECTED_SPECIAL_TOKENS.items():
        actual_id = vocab.get(token)
        if actual_id != expected_id:
            raise TokenizerContractError(
                f"{path}: token {token!r} must use id {expected_id}, got {actual_id}"
            )
        actual_token = id_to_token.get(expected_id)
        if actual_token != token:
            raise TokenizerContractError(
                f"{path}: id {expected_id} must map to {token!r}, got {actual_token!r}"
            )


__all__ = [
    "CppMegaTokenizer",
    "EXPECTED_SPECIAL_TOKENS",
    "EXPECTED_VOCAB_SIZE",
    "TokenizerContractError",
    "load_cppmega_tokenizer",
]
