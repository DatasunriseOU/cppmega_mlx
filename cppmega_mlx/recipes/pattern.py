"""NAM-style layer-pattern helpers for the MLX port.

The source cppmega recipe tiles ``AEMEAEMEAEMR`` by depth.  This module keeps
the MLX port fail-closed: only the symbols that have an explicit local meaning
are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NamSymbol = Literal["A", "E", "M", "R"]
LayerRole = Literal["attention", "moe", "mamba3", "m2rnn"]
AttentionRoute = Literal["dsa", "mla"]

SUPPORTED_NAM_SYMBOLS = frozenset({"A", "E", "M", "R"})
ORDERED_NAM_SYMBOLS: tuple[NamSymbol, ...] = ("A", "E", "M", "R")

_ROLE_BY_SYMBOL: dict[NamSymbol, LayerRole] = {
    "A": "attention",
    "E": "moe",
    "M": "mamba3",
    "R": "m2rnn",
}


@dataclass(frozen=True)
class NamLayer:
    """One 1-based layer entry after expanding a tiled NAM pattern."""

    number: int
    symbol: NamSymbol
    role: LayerRole
    # Zero-based index among A layers. This matches cppmega's DSA route tuple.
    a_rank: int | None = None
    attention_route: AttentionRoute | None = None


@dataclass(frozen=True)
class ExpandedNamPattern:
    """Expanded NAM pattern with derived routing lists.

    ``dsa_a_layer_ranks`` are zero-based indices among A layers, not absolute
    layer numbers.  This mirrors cppmega's ``CPPMEGA_DSA_A_LAYER_RANKS``
    contract: source launchers route with ``attn_nums[index]``, and
    ``CppMegaSelectiveAttentionLayer`` uses ``attention_layer_numbers.index``.
    """

    source_pattern: str
    depth: int
    symbols: tuple[NamSymbol, ...]
    layers: tuple[NamLayer, ...]
    dsa_a_layer_ranks: tuple[int, ...] = ()

    @property
    def layer_numbers(self) -> tuple[int, ...]:
        return tuple(layer.number for layer in self.layers)

    @property
    def a_layer_numbers(self) -> tuple[int, ...]:
        return tuple(layer.number for layer in self.layers if layer.symbol == "A")

    @property
    def r_layer_numbers(self) -> tuple[int, ...]:
        return tuple(layer.number for layer in self.layers if layer.symbol == "R")

    @property
    def mamba3_layer_numbers(self) -> tuple[int, ...]:
        return tuple(layer.number for layer in self.layers if layer.symbol == "M")

    @property
    def moe_layer_numbers(self) -> tuple[int, ...]:
        return tuple(layer.number for layer in self.layers if layer.symbol == "E")

    @property
    def dsa_layer_numbers(self) -> tuple[int, ...]:
        return tuple(layer.number for layer in self.layers if layer.attention_route == "dsa")

    @property
    def mla_layer_numbers(self) -> tuple[int, ...]:
        return tuple(layer.number for layer in self.layers if layer.attention_route == "mla")

    @property
    def counts(self) -> dict[NamSymbol, int]:
        return {symbol: self.symbols.count(symbol) for symbol in ORDERED_NAM_SYMBOLS}

    @property
    def role_counts(self) -> dict[LayerRole, int]:
        return {
            "attention": len(self.a_layer_numbers),
            "moe": len(self.moe_layer_numbers),
            "mamba3": len(self.mamba3_layer_numbers),
            "m2rnn": len(self.r_layer_numbers),
        }

    @property
    def layer_numbers_by_role(self) -> dict[LayerRole, tuple[int, ...]]:
        return {
            "attention": self.a_layer_numbers,
            "moe": self.moe_layer_numbers,
            "mamba3": self.mamba3_layer_numbers,
            "m2rnn": self.r_layer_numbers,
        }

    def layer_numbers_for_role(self, role: LayerRole) -> tuple[int, ...]:
        if role == "attention":
            return self.a_layer_numbers
        if role == "moe":
            return self.moe_layer_numbers
        if role == "mamba3":
            return self.mamba3_layer_numbers
        if role == "m2rnn":
            return self.r_layer_numbers
        raise ValueError(f"unsupported NAM layer role: {role!r}")

    def attention_route_for_layer(self, layer_number: int) -> AttentionRoute | None:
        for layer in self.layers:
            if layer.number == layer_number:
                return layer.attention_route
        raise ValueError(f"layer_number={layer_number} is outside expanded depth={self.depth}")


def parse_nam_pattern(pattern: str) -> tuple[NamSymbol, ...]:
    """Parse a NAM source pattern, accepting only A/E/M/R symbols."""

    if not isinstance(pattern, str):
        raise TypeError("pattern must be a string")
    normalized = pattern.strip().upper()
    if not normalized:
        raise ValueError("pattern must be non-empty")
    invalid = sorted({char for char in normalized if char not in SUPPORTED_NAM_SYMBOLS})
    if invalid:
        raise ValueError(
            f"invalid NAM pattern chars {invalid!r}; supported symbols are A, E, M, R"
        )
    return tuple(normalized)  # type: ignore[return-value]


def expand_symbols(pattern: str, depth: int) -> tuple[NamSymbol, ...]:
    """Tile *pattern* to exactly *depth* symbols."""

    if depth <= 0:
        raise ValueError("depth must be positive")
    parsed = parse_nam_pattern(pattern)
    return tuple(parsed[index % len(parsed)] for index in range(depth))


def parse_rank_list(raw: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Parse a comma-separated zero-based integer list or validate a sequence."""

    if isinstance(raw, str):
        if not raw.strip():
            return ()
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    else:
        values = tuple(int(value) for value in raw)
    _validate_non_negative_unique(values, name="rank")
    return values


def expand_nam_pattern(
    pattern: str,
    depth: int,
    *,
    dsa_a_layer_ranks: str | tuple[int, ...] | list[int] = (),
) -> ExpandedNamPattern:
    """Expand a NAM pattern and route A layers to DSA or MLA by zero-based A-rank."""

    symbols = expand_symbols(pattern, depth)
    dsa_ranks = parse_rank_list(dsa_a_layer_ranks)
    a_count = symbols.count("A")
    out_of_range = [rank for rank in dsa_ranks if rank >= a_count]
    if out_of_range:
        raise ValueError(
            f"DSA A-layer ranks {out_of_range!r} exceed zero-based A-layer range 0..{a_count - 1}"
        )

    layers: list[NamLayer] = []
    a_rank = -1
    dsa_rank_set = frozenset(dsa_ranks)
    for index, symbol in enumerate(symbols, start=1):
        if symbol == "A":
            a_rank += 1
            route: AttentionRoute = "dsa" if a_rank in dsa_rank_set else "mla"
            layers.append(
                NamLayer(
                    number=index,
                    symbol=symbol,
                    role=_ROLE_BY_SYMBOL[symbol],
                    a_rank=a_rank,
                    attention_route=route,
                )
            )
        else:
            layers.append(
                NamLayer(
                    number=index,
                    symbol=symbol,
                    role=_ROLE_BY_SYMBOL[symbol],
                )
            )

    return ExpandedNamPattern(
        source_pattern="".join(parse_nam_pattern(pattern)),
        depth=depth,
        symbols=symbols,
        layers=tuple(layers),
        dsa_a_layer_ranks=dsa_ranks,
    )


def layer_numbers_for_symbol(pattern: str, depth: int, symbol: NamSymbol) -> tuple[int, ...]:
    if symbol not in SUPPORTED_NAM_SYMBOLS:
        raise ValueError(f"unsupported NAM symbol: {symbol!r}")
    expanded = expand_symbols(pattern, depth)
    return tuple(index for index, value in enumerate(expanded, start=1) if value == symbol)


def a_layer_numbers(pattern: str, depth: int) -> tuple[int, ...]:
    return layer_numbers_for_symbol(pattern, depth, "A")


def r_layer_numbers(pattern: str, depth: int) -> tuple[int, ...]:
    return layer_numbers_for_symbol(pattern, depth, "R")


def _validate_non_negative_unique(values: tuple[int, ...], *, name: str) -> None:
    seen: set[int] = set()
    duplicates: list[int] = []
    negative: list[int] = []
    for value in values:
        if value < 0:
            negative.append(value)
        elif value in seen:
            duplicates.append(value)
        seen.add(value)
    if negative:
        raise ValueError(f"{name}s must be non-negative, got {negative!r}")
    if duplicates:
        raise ValueError(f"{name}s must be unique, got duplicate values {duplicates!r}")
