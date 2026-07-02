"""Autoconf / Automake build-domain parser."""

from __future__ import annotations

from cppmega_mlx.data.build_parsers.base import ParsedDomainDocument
from cppmega_mlx.data.build_parsers.make import parse_automake
from cppmega_mlx.data.domain_schema import (
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_AUTOCONF_MACROS = {
    "AC_INIT",
    "AC_CONFIG_FILES",
    "AC_PROG_CC",
    "AC_PROG_CXX",
    "AM_INIT_AUTOMAKE",
    "PKG_CHECK_MODULES",
}


def parse_autoconf(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.AUTOCONF,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"build_kind": "autoconf"},
    )
    next_entity = 1
    for idx, token in enumerate(doc.tokens):
        value = token.text
        if value in _AUTOCONF_MACROS or value.startswith("AC_") or value.startswith("AM_"):
            doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
            next_entity += 1
        elif value.startswith("$"):
            doc.set_role(idx, DomainRoleKind.VARIABLE)
    return doc


__all__ = ["parse_autoconf", "parse_automake"]
