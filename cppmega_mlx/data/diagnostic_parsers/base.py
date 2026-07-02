"""Shared compiler/build diagnostic parser helpers."""

from __future__ import annotations

from cppmega_mlx.data.build_parsers.base import ParsedDomainDocument
from cppmega_mlx.data.domain_schema import (
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


def new_diagnostic_doc(
    text: str,
    *,
    domain: DomainKind,
    tool: str,
    confidence: ParseConfidence = ParseConfidence.HEURISTIC,
) -> ParsedDomainDocument:
    return ParsedDomainDocument.new(
        domain=domain,
        text=text,
        confidence=confidence,
        metadata={"diagnostic_tool": tool},
    )


def mark_message_tail(
    doc: ParsedDomainDocument,
    *,
    line_no: int,
    after_index: int,
) -> None:
    for idx in doc.token_indices_on_line(line_no):
        if idx > after_index and doc.role_ids[idx] == int(DomainRoleKind.NONE):
            doc.set_role(idx, DomainRoleKind.MESSAGE)


__all__ = ["mark_message_tail", "new_diagnostic_doc"]
