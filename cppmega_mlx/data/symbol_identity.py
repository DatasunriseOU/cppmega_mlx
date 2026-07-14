"""Corpus-wide symbol identity ID contract.

The canonical clang identity key remains the source of truth. Numeric IDs are
opaque unsigned 64-bit projections used by dense training sidecars. Every
serialized corpus row carries the corresponding ID-to-key claims so aggregators
can detect a collision and fail closed instead of aliasing supervision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import re


SYMBOL_IDENTITY_SCHEMA_VERSION = 3
SYMBOL_IDENTITY_SCHEMA_METADATA_KEY = "cppmega.symbol_identity_schema_version"
SYMBOL_IDENTITIES_COLUMN = "symbol_identities"
SYMBOL_ID_MAX = (1 << 64) - 1
_SYMBOL_ID_HASH_DOMAIN = b"cppmega.symbol-id.v3\0"


class SymbolIdentityError(RuntimeError):
    """Raised when a symbol identity claim violates the v3 contract."""


_PROJECT_IDENTITY_RE = re.compile(r"^[^/\s]+/[^/\s]+$")


def require_project_identity(value: object, *, source: str) -> str:
    """Validate and return the canonical ``owner/repo`` project identity."""

    if not isinstance(value, str) or value != value.strip():
        raise SymbolIdentityError(
            f"{source}: project identity must be an exact owner/repo string, got {value!r}"
        )
    if not _PROJECT_IDENTITY_RE.fullmatch(value):
        raise SymbolIdentityError(
            f"{source}: project identity must be stable owner/repo, got {value!r}"
        )
    owner, repo = value.split("/", 1)
    if owner in {".", ".."} or repo in {".", ".."} or repo.endswith(".git"):
        raise SymbolIdentityError(
            f"{source}: project identity must be canonical owner/repo without .git, "
            f"got {value!r}"
        )
    return value


def compute_symbol_id(symbol_key: str) -> int:
    """Return the stable unsigned 64-bit ID for one canonical identity key."""

    if not symbol_key:
        return 0
    digest = hashlib.sha256(
        _SYMBOL_ID_HASH_DOMAIN + symbol_key.encode("utf-8", errors="strict")
    ).digest()
    symbol_id = int.from_bytes(digest[:8], byteorder="big", signed=False)
    if symbol_id == 0:
        raise SymbolIdentityError(
            "canonical symbol key hashed to reserved ID 0; refusing to emit "
            f"semantic supervision for {symbol_key!r}"
        )
    return symbol_id


class SymbolIdentityRegistry:
    """Validate ID/key claims across projects, shards, or a persisted index."""

    def __init__(self) -> None:
        self.keys_by_id: dict[int, str] = {}
        self.ids_by_key: dict[str, int] = {}
        self.sources_by_id: dict[int, str] = {}

    def register(
        self,
        symbol_key: str,
        *,
        symbol_id: int | None = None,
        source: str = "",
    ) -> int:
        if not isinstance(symbol_key, str) or not symbol_key:
            raise SymbolIdentityError(
                f"{source or 'symbol identity'}: symbol_key must be a non-empty string"
            )
        claimed_id = compute_symbol_id(symbol_key) if symbol_id is None else int(symbol_id)
        if not 0 < claimed_id <= SYMBOL_ID_MAX:
            raise SymbolIdentityError(
                f"{source or 'symbol identity'}: symbol_id must be in [1, {SYMBOL_ID_MAX}], "
                f"got {claimed_id}"
            )

        existing_key = self.keys_by_id.get(claimed_id)
        if existing_key is not None and existing_key != symbol_key:
            first_source = self.sources_by_id.get(claimed_id, "unknown source")
            raise SymbolIdentityError(
                "canonical symbol ID collision: "
                f"id={claimed_id} first={existing_key!r} ({first_source}) "
                f"second={symbol_key!r} ({source or 'unknown source'}). "
                "Refusing to emit aliased semantic supervision."
            )

        existing_id = self.ids_by_key.get(symbol_key)
        if existing_id is not None and existing_id != claimed_id:
            raise SymbolIdentityError(
                f"{source or 'symbol identity'}: canonical key {symbol_key!r} was already "
                f"registered as ID {existing_id}, not {claimed_id}"
            )

        expected_id = compute_symbol_id(symbol_key)
        if claimed_id != expected_id:
            raise SymbolIdentityError(
                f"{source or 'symbol identity'}: symbol_id {claimed_id} does not match "
                f"the v{SYMBOL_IDENTITY_SCHEMA_VERSION} canonical ID {expected_id} for "
                f"{symbol_key!r}"
            )

        self.keys_by_id[claimed_id] = symbol_key
        self.ids_by_key[symbol_key] = claimed_id
        self.sources_by_id.setdefault(claimed_id, source or "unknown source")
        return claimed_id

    def register_records(
        self,
        records: object,
        *,
        source: str,
    ) -> list[dict[str, object]]:
        if records is None:
            records = []
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise SymbolIdentityError(
                f"{source}: {SYMBOL_IDENTITIES_COLUMN} must be a list of ID/key records"
            )
        normalized: list[dict[str, object]] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise SymbolIdentityError(
                    f"{source}: {SYMBOL_IDENTITIES_COLUMN}[{index}] must be an object"
                )
            symbol_key = record.get("symbol_key")
            symbol_id = record.get("symbol_id")
            if not isinstance(symbol_key, str) or symbol_id is None:
                raise SymbolIdentityError(
                    f"{source}: {SYMBOL_IDENTITIES_COLUMN}[{index}] requires "
                    "symbol_id and symbol_key"
                )
            registered_id = self.register(
                symbol_key,
                symbol_id=int(symbol_id),
                source=source,
            )
            normalized.append(
                {"symbol_id": registered_id, "symbol_key": symbol_key}
            )
        return normalized

    def require_ids(self, symbol_ids: Iterable[int], *, source: str) -> None:
        missing = sorted(
            {
                int(symbol_id)
                for symbol_id in symbol_ids
                if int(symbol_id) != 0 and int(symbol_id) not in self.keys_by_id
            }
        )
        if missing:
            preview = missing[:8]
            suffix = "..." if len(missing) > len(preview) else ""
            raise SymbolIdentityError(
                f"{source}: semantic symbol IDs are missing canonical identity claims: "
                f"{preview}{suffix}"
            )

    def records(self, symbol_ids: Iterable[int] | None = None) -> list[dict[str, object]]:
        if symbol_ids is None:
            selected = sorted(self.keys_by_id)
        else:
            selected = sorted({int(value) for value in symbol_ids if int(value) != 0})
            self.require_ids(selected, source="symbol identity record export")
        return [
            {"symbol_id": symbol_id, "symbol_key": self.keys_by_id[symbol_id]}
            for symbol_id in selected
        ]


__all__ = [
    "SYMBOL_IDENTITIES_COLUMN",
    "SYMBOL_IDENTITY_SCHEMA_METADATA_KEY",
    "SYMBOL_IDENTITY_SCHEMA_VERSION",
    "SYMBOL_ID_MAX",
    "SymbolIdentityError",
    "SymbolIdentityRegistry",
    "compute_symbol_id",
    "require_project_identity",
]
