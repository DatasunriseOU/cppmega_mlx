"""Shared helpers for deterministic build-domain parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_TOKEN_RE = re.compile(
    r'"[^"\n]*"|\'[^\'\n]*\'|\$\(?[A-Za-z_][A-Za-z0-9_]*\)?|::|&&|\|\||>>|2>|&>|'
    r'[A-Za-z0-9_./@%+$\\<>-]+|[:=(),;\[\]{}|<>]|[^\s]',
    re.MULTILINE,
)


@dataclass(frozen=True)
class LexedToken:
    text: str
    start: int
    end: int
    line: int
    column: int


@dataclass
class ParsedDomainDocument:
    domain: DomainKind
    tokens: list[LexedToken]
    role_ids: list[int]
    entity_ids: list[int]
    scope_ids: list[int]
    confidence_ids: list[int]
    edges: list[tuple[int, int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        domain: DomainKind,
        text: str,
        confidence: ParseConfidence = ParseConfidence.HEURISTIC,
        metadata: dict[str, Any] | None = None,
    ) -> "ParsedDomainDocument":
        tokens = lex_text(text)
        n = len(tokens)
        return cls(
            domain=domain,
            tokens=tokens,
            role_ids=[int(DomainRoleKind.NONE)] * n,
            entity_ids=[0] * n,
            scope_ids=[0] * n,
            confidence_ids=[int(confidence)] * n,
            metadata=metadata or {},
        )

    def set_role(
        self,
        index: int,
        role: DomainRoleKind,
        *,
        entity: int | None = None,
        scope: int | None = None,
        confidence: ParseConfidence | None = None,
    ) -> None:
        if not (0 <= index < len(self.tokens)):
            raise IndexError(f"token index {index} out of range")
        self.role_ids[index] = int(role)
        if entity is not None:
            self.entity_ids[index] = int(entity)
        if scope is not None:
            self.scope_ids[index] = int(scope)
        if confidence is not None:
            self.confidence_ids[index] = int(confidence)

    def add_edge(
        self,
        src: int,
        dst: int,
        kind: DomainEdgeKind,
    ) -> None:
        if not (0 <= src < len(self.tokens)) or not (0 <= dst < len(self.tokens)):
            raise IndexError(f"edge endpoint out of range: {src}->{dst}")
        self.edges.append((int(src), int(dst), int(kind)))

    def token_indices_on_line(self, line_no: int) -> list[int]:
        return [idx for idx, token in enumerate(self.tokens) if token.line == line_no]

    def to_packet(self, token_ids: list[int] | None = None) -> DomainPacket:
        import mlx.core as mx
        import numpy as np

        from cppmega_mlx.data.domain_packet import DomainEdgeIndex, DomainPacket

        if token_ids is None:
            token_ids = [debug_token_id(token.text) for token in self.tokens]
            metadata = dict(self.metadata)
            metadata["debug_token_ids"] = True
        else:
            metadata = dict(self.metadata)
        if len(token_ids) != len(self.tokens):
            raise ValueError(
                f"token_ids length {len(token_ids)} != parsed token count {len(self.tokens)}"
            )
        n = len(token_ids)
        return DomainPacket(
            token_ids=mx.array(np.asarray(token_ids, dtype=np.int32)),
            domain=self.domain,
            domain_ids=mx.array(np.full((n,), int(self.domain), dtype=np.int32)),
            role_ids=mx.array(np.asarray(self.role_ids, dtype=np.int32)),
            entity_ids=mx.array(np.asarray(self.entity_ids, dtype=np.int32)),
            scope_ids=mx.array(np.asarray(self.scope_ids, dtype=np.int32)),
            source_doc_ids=mx.array(np.zeros((n,), dtype=np.int32)),
            confidence_ids=mx.array(np.asarray(self.confidence_ids, dtype=np.int32)),
            graph_edges=DomainEdgeIndex.from_triples(self.edges),
            metadata=metadata,
        )


def lex_text(text: str) -> list[LexedToken]:
    line_starts = [0]
    for match in re.finditer(r"\n", text):
        line_starts.append(match.end())
    tokens: list[LexedToken] = []
    for match in _TOKEN_RE.finditer(text):
        start = match.start()
        line = 0
        # Texts here are small build/script files.  Linear scan keeps the helper
        # dependency-free and deterministic.
        for idx, line_start in enumerate(line_starts):
            if line_start > start:
                break
            line = idx
        tokens.append(
            LexedToken(
                text=match.group(0),
                start=start,
                end=match.end(),
                line=line,
                column=start - line_starts[line],
            )
        )
    return tokens


def debug_token_id(text: str) -> int:
    """Stable non-training token id for parser unit tests."""

    return 1000 + (sum(text.encode("utf-8")) % 30000)


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def is_source_path(value: str) -> bool:
    lower = strip_quotes(value).lower()
    return lower.endswith(
        (
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".h",
            ".hh",
            ".hpp",
            ".hxx",
            ".cu",
            ".cuh",
            ".s",
            ".asm",
        )
    )


def is_option(value: str) -> bool:
    return value.startswith("-") or value.startswith("--")


__all__ = [
    "LexedToken",
    "ParsedDomainDocument",
    "debug_token_id",
    "is_option",
    "is_source_path",
    "lex_text",
    "strip_quotes",
]
