"""Autotools build-domain parsers."""

from __future__ import annotations

from cppmega_mlx.data.build_parsers.base import ParsedDomainDocument
from cppmega_mlx.data.build_parsers.make import parse_automake
from cppmega_mlx.data.shell_parsers.base import parse_shell
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


def _is_balanced(text: str) -> bool:
    pairs = {"(": ")", "[": "]"}
    stack: list[str] = []
    for ch in text:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack.pop() != ch:
                return False
    return not stack


def parse_autoconf(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.AUTOCONF,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"build_kind": "autoconf", "parser_adapter": "autoconf"},
    )
    if not _is_balanced(text):
        return doc.mark_raw("malformed_autoconf_macro")

    next_entity = 1
    for idx, token in enumerate(doc.tokens):
        value = token.text
        if value in _AUTOCONF_MACROS or value.startswith("AC_") or value.startswith("AM_"):
            doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
            next_entity += 1
        elif value.startswith("$"):
            doc.set_role(idx, DomainRoleKind.VARIABLE)
    return doc


def parse_configure(text: str) -> ParsedDomainDocument:
    doc = parse_shell(
        text,
        domain=DomainKind.CONFIGURE,
        shell_kind="configure",
        malformed_reason="malformed_configure_shell",
    )
    doc.metadata["build_kind"] = "configure"
    return doc


__all__ = ["parse_autoconf", "parse_automake", "parse_configure"]
