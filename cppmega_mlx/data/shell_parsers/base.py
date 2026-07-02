"""Shared shell-domain parser helpers."""

from __future__ import annotations

from cppmega_mlx.data.build_parsers.base import ParsedDomainDocument, is_option
from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_SHELL_KEYWORDS = {"if", "then", "else", "fi", "for", "do", "done", "case", "esac", "while"}
_REDIR_OUT = {">", ">>", "2>", "&>"}
_REDIR_IN = {"<"}


def parse_shell(text: str, *, domain: DomainKind, shell_kind: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=domain,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"shell_kind": shell_kind},
    )
    next_entity = 1
    previous_command: int | None = None
    command_expected = True
    pending_redir: tuple[int, DomainEdgeKind] | None = None

    idx = 0
    while idx < len(doc.tokens):
        token = doc.tokens[idx]
        value = token.text
        if value == "|":
            doc.set_role(idx, DomainRoleKind.PIPE)
            if previous_command is not None:
                pending_pipe_src = previous_command
            else:
                pending_pipe_src = None
            command_expected = True
            pending_redir = None
            # Store temporary source in metadata-like scope via negative-free local.
            previous_command = pending_pipe_src
            idx += 1
            continue
        if value in _REDIR_OUT | _REDIR_IN:
            edge_kind = (
                DomainEdgeKind.SHELL_REDIR_OUT
                if value in _REDIR_OUT
                else DomainEdgeKind.SHELL_REDIR_IN
            )
            doc.set_role(idx, DomainRoleKind.REDIRECT)
            pending_redir = (previous_command if previous_command is not None else idx, edge_kind)
            idx += 1
            continue
        if (
            command_expected
            and value.isidentifier()
            and idx + 2 < len(doc.tokens)
            and doc.tokens[idx + 1].text == "="
        ):
            doc.set_role(idx, DomainRoleKind.ENVIRONMENT, entity=next_entity)
            doc.set_role(idx + 2, DomainRoleKind.STRING, entity=next_entity)
            next_entity += 1
            idx += 3
            continue
        if "=" in value and not value.startswith("=") and value.split("=", 1)[0].isidentifier():
            doc.set_role(idx, DomainRoleKind.ENVIRONMENT, entity=next_entity)
            next_entity += 1
            command_expected = True
            idx += 1
            continue
        if value.startswith("$"):
            doc.set_role(idx, DomainRoleKind.VARIABLE)
            idx += 1
            continue
        if value in _SHELL_KEYWORDS:
            doc.set_role(idx, DomainRoleKind.KEYWORD)
            command_expected = True
            idx += 1
            continue
        if pending_redir is not None:
            src, edge_kind = pending_redir
            doc.set_role(idx, DomainRoleKind.PATH, entity=next_entity)
            next_entity += 1
            doc.add_edge(src, idx, edge_kind)
            pending_redir = None
            idx += 1
            continue
        if command_expected and value not in {";", "&&", "||"}:
            old_command = previous_command
            doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
            previous_command = idx
            next_entity += 1
            if old_command is not None and any(
                doc.tokens[j].text == "|" for j in range(max(0, old_command + 1), idx)
            ):
                doc.add_edge(old_command, idx, DomainEdgeKind.SHELL_PIPE)
            command_expected = False
            idx += 1
            continue
        if is_option(value):
            doc.set_role(idx, DomainRoleKind.OPTION, scope=previous_command or 0)
        elif "/" in value or "." in value:
            doc.set_role(idx, DomainRoleKind.PATH, entity=next_entity, scope=previous_command or 0)
            next_entity += 1
            if previous_command is not None:
                doc.add_edge(previous_command, idx, DomainEdgeKind.SHELL_COMMAND_FILE)

        if value in {";", "&&", "||"}:
            command_expected = True
        idx += 1

    return doc


__all__ = ["parse_shell"]
