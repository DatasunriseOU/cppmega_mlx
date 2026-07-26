"""Portable token-dataset metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.lib.npyio import NpzFile

from cppmega_mlx.config.model import (
    LOCAL_PROFILE_VOCAB_SIZE,
    MEGACPP_TOKENIZER_VOCAB_SIZE,
)

TokenizerContract = Literal["megacpp", "local_profile", "custom"]
TokenDatasetFormat = Literal["npz", "parquet", "megatron"]


@dataclass(frozen=True)
class TokenDatasetMetadata:
    """Tokenizer/data contract carried with local token shards."""

    vocab_size: int = MEGACPP_TOKENIZER_VOCAB_SIZE
    tokenizer_contract: TokenizerContract = "megacpp"
    local_profile_vocab_size: int = LOCAL_PROFILE_VOCAB_SIZE
    megacpp_tokenizer_vocab_size: int = MEGACPP_TOKENIZER_VOCAB_SIZE
    source_format: str = "npz"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.local_profile_vocab_size <= 0:
            raise ValueError("local_profile_vocab_size must be positive")
        if self.megacpp_tokenizer_vocab_size <= 0:
            raise ValueError("megacpp_tokenizer_vocab_size must be positive")
        if self.tokenizer_contract not in {"megacpp", "local_profile", "custom"}:
            raise ValueError(
                f"unsupported tokenizer_contract={self.tokenizer_contract!r}"
            )

    @classmethod
    def from_npz(cls, data: NpzFile) -> TokenDatasetMetadata:
        """Read optional scalar metadata from an NPZ file."""

        return cls(
            vocab_size=_npz_scalar_int(
                data,
                "vocab_size",
                MEGACPP_TOKENIZER_VOCAB_SIZE,
            ),
            tokenizer_contract=_npz_scalar_str(
                data,
                "tokenizer_contract",
                "megacpp",
            ),
            local_profile_vocab_size=_npz_scalar_int(
                data,
                "local_profile_vocab_size",
                LOCAL_PROFILE_VOCAB_SIZE,
            ),
            megacpp_tokenizer_vocab_size=_npz_scalar_int(
                data,
                "megacpp_tokenizer_vocab_size",
                MEGACPP_TOKENIZER_VOCAB_SIZE,
            ),
            source_format="npz",
        )


def _npz_scalar_int(data: NpzFile, key: str, default: int) -> int:
    if key not in data:
        return default
    return int(np.asarray(data[key]).reshape(()).item())


def _npz_scalar_str(
    data: NpzFile,
    key: str,
    default: TokenizerContract,
) -> TokenizerContract:
    if key not in data:
        return default
    value = str(np.asarray(data[key]).reshape(()).item())
    if value not in {"megacpp", "local_profile", "custom"}:
        raise ValueError(f"unsupported tokenizer_contract={value!r}")
    return cast(TokenizerContract, value)


__all__ = [
    "TokenDatasetFormat",
    "TokenDatasetMetadata",
    "TokenizerContract",
]
