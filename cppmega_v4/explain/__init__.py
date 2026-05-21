"""Tooltip catalogue — paper-cited explanations for every option the
Visual Builder exposes (optimizer / activation / norm / schedule /
loss / rewriter / brick)."""

from cppmega_v4.explain.catalog import (
    CATALOG,
    CATEGORIES,
    ExplainEntry,
    get_entry,
    list_options,
)

__all__ = [
    "CATALOG",
    "CATEGORIES",
    "ExplainEntry",
    "get_entry",
    "list_options",
]
