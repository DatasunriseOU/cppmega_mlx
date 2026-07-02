"""Linker diagnostic parser."""

from __future__ import annotations

import re

from cppmega_mlx.data.diagnostic_parsers.base import new_diagnostic_doc
from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
)


_UNDEF_RE = re.compile(r"(undefined reference to|Undefined symbols for architecture).*?[`'\"](?P<sym>[^`'\"]+)[`'\"]")


def parse_linker_error(text: str, *, tool: str = "linker") -> object:
    doc = new_diagnostic_doc(text, domain=DomainKind.LINKER_ERROR, tool=tool)
    first_error = 0 if doc.tokens else None
    for idx, token in enumerate(doc.tokens):
        value = token.text.strip("`'\"")
        if value in {"ld", "lld", "link", "link.exe"}:
            doc.set_role(idx, DomainRoleKind.COMMAND)
            if first_error is None:
                first_error = idx
        elif value.lower() in {"error", "undefined"}:
            doc.set_role(idx, DomainRoleKind.SEVERITY)
            if first_error is None:
                first_error = idx
        elif "/" in value or value.endswith((".o", ".obj", ".a", ".lib", ".so", ".dylib")):
            doc.set_role(idx, DomainRoleKind.PATH)

    for match in _UNDEF_RE.finditer(text):
        start, end = match.span("sym")
        first_symbol_idx: int | None = None
        for idx, token in enumerate(doc.tokens):
            if token.start < end and token.end > start:
                doc.set_role(idx, DomainRoleKind.SYMBOL)
                if first_symbol_idx is None:
                    first_symbol_idx = idx
        if first_error is not None and first_symbol_idx is not None:
            doc.add_edge(first_error, first_symbol_idx, DomainEdgeKind.LINK_UNDEFINED_SYMBOL)
    return doc


__all__ = ["parse_linker_error"]
