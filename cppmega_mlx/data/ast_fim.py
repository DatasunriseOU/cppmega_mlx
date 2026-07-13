"""AST-aware Fill-in-the-Middle span sampling driven by clang chunk boundaries.

This module sits on top of :mod:`cppmega_mlx.data.fim` — it does NOT reimplement
the PSM/SPM permutations.  Its only job is to *choose the middle span* from a
``CodePacket``'s token-aligned structure side-channels, then hand that span to the
existing ``fim.apply_fim_permutation`` emitter.

Span selection (research-grounded AST-FIM):

  * The packet's ``chunk_starts`` / ``chunk_ends`` / ``chunk_kinds`` are produced
    by the clang chunker; each ``[start, end)`` is a COMPLETE syntactic unit
    (expression / statement / block / function-body) over the token axis.  These
    are already token-aligned, so selecting a whole chunk as the FIM middle never
    splits a token.
  * We draw ONE chunk whose span keeps prefix, middle, and suffix non-empty
    (``0 < start < end < len(tokens)``) so the result round-trips through
    ``fim``'s reference contract.

Mix (mid-token robustness):

  * 90% of the time we emit an AST-FIM example (whole-chunk middle).
  * 10% of the time we emit a random-CHAR-FIM example whose middle is a random
    token span (``fim.sample_middle_span``) that may start/end mid-statement —
    this teaches robustness to arbitrary cursor positions.  The fall-back to the
    char path is RECORDED in the returned ``AstFimResult.kind`` ("char_fim"),
    never silent.

RULE #1 (fail fast / fail loud): every required field (``chunk_starts`` /
``chunk_ends``) is validated up-front and a missing one RAISES with WHERE +
WHAT — no silent path.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.fim import (
    FIMMode,
    FIMSpecialTokenIds,
    FIMSpecialTokenInput,
    apply_fim_permutation,
    apply_ifim_permutation,
    sample_middle_span,
)

# Fraction of AST-FIM examples; the remaining slice is random-char-FIM.
DEFAULT_AST_FIM_RATE = 0.9

AstFimKind = Literal["ast_fim", "char_fim", "ast_ifim", "char_ifim"]


class NoEligibleChunkError(ValueError):
    """No clang chunk yields a non-empty prefix/middle/suffix for THIS window.

    This is the ONLY condition that routes to the recorded char-FIM slice; it is
    distinct from a genuinely absent ``chunk_starts``/``chunk_ends`` field, which
    raises a plain ``ValueError`` and propagates (fail-loud).
    """


@dataclass(frozen=True)
class AstFimResult:
    """Result of an AST-aware FIM emission.

    ``kind`` records WHICH path produced the example so the 10% char-FIM fallback
    is observable, not silent.  ``span`` is the half-open ``[start, end)`` middle
    in token space; ``chunk_index`` is the selected chunk (``None`` for char-FIM).
    """

    token_ids: list[int]
    span: tuple[int, int]
    mode: FIMMode
    kind: AstFimKind
    chunk_index: int | None


def _token_count(packet: CodePacket) -> int:
    if packet.token_ids.ndim != 1:
        raise ValueError(
            f"ast_fim requires a single-window CodePacket (1-D token_ids), got "
            f"shape {tuple(packet.token_ids.shape)}"
        )
    return int(packet.token_ids.shape[0])


def _packet_token_list(packet: CodePacket) -> list[int]:
    return [int(x) for x in np.asarray(packet.token_ids).reshape(-1).tolist()]


def _chunk_arrays(packet: CodePacket) -> tuple[list[int], list[int], list[int] | None]:
    if packet.chunk_starts is None or packet.chunk_ends is None:
        raise ValueError(
            "ast_fim span selection requires CodePacket.chunk_starts and "
            "chunk_ends (clang chunk boundaries); both must be present"
        )
    starts = [int(x) for x in np.asarray(packet.chunk_starts).reshape(-1).tolist()]
    ends = [int(x) for x in np.asarray(packet.chunk_ends).reshape(-1).tolist()]
    if len(starts) != len(ends):
        raise ValueError(
            f"ast_fim: chunk_starts ({len(starts)}) and chunk_ends ({len(ends)}) "
            f"length mismatch"
        )
    kinds: list[int] | None = None
    if packet.chunk_kinds is not None:
        kinds = [int(x) for x in np.asarray(packet.chunk_kinds).reshape(-1).tolist()]
        if len(kinds) != len(starts):
            raise ValueError(
                f"ast_fim: chunk_kinds ({len(kinds)}) length != chunk count "
                f"({len(starts)})"
            )
    return starts, ends, kinds


def _eligible_chunks(
    starts: Sequence[int], ends: Sequence[int], length: int
) -> list[int]:
    """Chunk indices whose [start, end) keep prefix/middle/suffix non-empty."""

    eligible: list[int] = []
    for idx, (start, end) in enumerate(zip(starts, ends)):
        if not 0 <= start < end <= length:
            raise ValueError(
                f"ast_fim: chunk {idx} span [{start}, {end}) out of bounds for "
                f"{length} tokens (chunk boundaries must be valid token offsets)"
            )
        # Reference contract: 0 < start < end < length (non-empty pre/mid/suffix).
        if 0 < start and end < length:
            eligible.append(idx)
    return eligible


def select_ast_span(
    packet: CodePacket,
    *,
    rng: random.Random,
) -> tuple[int, int, int]:
    """Select a COMPLETE syntactic unit (a whole clang chunk) as the FIM middle.

    Returns ``(start, end, chunk_index)``.  RAISES if chunk boundaries are absent
    or if no chunk yields a non-empty prefix/middle/suffix (caller decides whether
    to route to the char-FIM slice).
    """

    length = _token_count(packet)
    if length < 3:
        raise ValueError(
            f"ast_fim requires at least 3 tokens to form prefix/middle/suffix, "
            f"got {length}"
        )
    # Absent chunk boundaries are a MISSING REQUIRED FIELD for AST-FIM -> RAISE
    # (this propagates; it is NOT the recorded 10% char fallback).
    starts, ends, _kinds = _chunk_arrays(packet)
    eligible = _eligible_chunks(starts, ends, length)
    if not eligible:
        # Present-but-unusable for THIS window: caller routes to the char slice
        # and RECORDS it (observable, not silent).
        raise NoEligibleChunkError(
            "ast_fim: no clang chunk yields a non-empty prefix/middle/suffix for "
            f"this {length}-token window; route to the char-FIM slice instead"
        )
    chunk_index = eligible[rng.randrange(len(eligible))]
    return starts[chunk_index], ends[chunk_index], chunk_index


def eligible_ast_chunk_indices(packet: CodePacket) -> tuple[int, ...]:
    """Return validated interior clang chunks without drawing randomness."""

    length = _token_count(packet)
    if length < 3:
        return ()
    starts, ends, _kinds = _chunk_arrays(packet)
    return tuple(_eligible_chunks(starts, ends, length))


def apply_ast_fim(
    packet: CodePacket,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
    spm_rate: float = 0.5,
    ast_fim_rate: float = DEFAULT_AST_FIM_RATE,
    special_token_ids: FIMSpecialTokenInput = None,
) -> AstFimResult:
    """Emit one AST-FIM (90%) or random-char-FIM (10%) example from ``packet``.

    Deterministic given ``seed`` (or an injected ``rng``).  The char-FIM fallback
    is recorded in ``AstFimResult.kind``.  Never splits a token on the AST path:
    chunk boundaries are token offsets.  On the 10% char slice a random token span
    is used (which may start/end mid-statement on purpose).
    """

    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")
    if not 0.0 <= ast_fim_rate <= 1.0:
        raise ValueError(f"ast_fim_rate must be in [0, 1], got {ast_fim_rate}")
    if not 0.0 <= spm_rate <= 1.0:
        raise ValueError(f"spm_rate must be in [0, 1], got {spm_rate}")
    rand = rng if rng is not None else random.Random(seed)

    tokens = _packet_token_list(packet)
    length = len(tokens)
    if length < 3:
        raise ValueError(
            f"ast_fim requires at least 3 tokens, got {length}"
        )

    use_ast = rand.random() < ast_fim_rate
    if use_ast:
        try:
            start, end, chunk_index = select_ast_span(packet, rng=rand)
            kind: AstFimKind = "ast_fim"
        except NoEligibleChunkError:
            # No usable chunk for THIS window: fall to the char slice and RECORD
            # it (not a silent degraded path — the kind makes it observable).
            # A genuinely absent chunk field is a plain ValueError and propagates.
            start, end = sample_middle_span(length, rng=rand)
            chunk_index = None
            kind = "char_fim"
    else:
        start, end = sample_middle_span(length, rng=rand)
        chunk_index = None
        kind = "char_fim"

    mode: FIMMode = "spm" if rand.random() < spm_rate else "psm"
    permuted = apply_fim_permutation(
        tokens,
        span=(start, end),
        mode=mode,
        special_token_ids=special_token_ids,
    )
    return AstFimResult(
        token_ids=permuted,
        span=(start, end),
        mode=mode,
        kind=kind,
        chunk_index=chunk_index,
    )


def apply_ast_ifim(
    packet: CodePacket,
    *,
    instruction_token_ids: Sequence[int] | None = None,
    seed: int | None = None,
    rng: random.Random | None = None,
    spm_rate: float = 0.5,
    ast_fim_rate: float = DEFAULT_AST_FIM_RATE,
    special_token_ids: FIMSpecialTokenInput = None,
) -> AstFimResult:
    """Emit typed instruction-aware AST-iFIM with an observable char fallback."""

    if instruction_token_ids is None:
        raise ValueError("ast_ifim requires typed instruction_token_ids")
    instruction = list(instruction_token_ids)
    if not instruction:
        raise ValueError("ast_ifim instruction_token_ids must not be empty")
    if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in instruction):
        raise ValueError("ast_ifim instruction_token_ids must contain integer token ids")
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")
    if not 0.0 <= ast_fim_rate <= 1.0:
        raise ValueError(f"ast_fim_rate must be in [0, 1], got {ast_fim_rate}")
    if not 0.0 <= spm_rate <= 1.0:
        raise ValueError(f"spm_rate must be in [0, 1], got {spm_rate}")
    rand = rng if rng is not None else random.Random(seed)

    tokens = _packet_token_list(packet)
    length = len(tokens)
    if length < 3:
        raise ValueError(f"ast_ifim requires at least 3 tokens, got {length}")

    use_ast = rand.random() < ast_fim_rate
    if use_ast:
        try:
            start, end, chunk_index = select_ast_span(packet, rng=rand)
            kind: AstFimKind = "ast_ifim"
        except NoEligibleChunkError:
            start, end = sample_middle_span(length, rng=rand)
            chunk_index = None
            kind = "char_ifim"
    else:
        start, end = sample_middle_span(length, rng=rand)
        chunk_index = None
        kind = "char_ifim"

    mode: FIMMode = "spm" if rand.random() < spm_rate else "psm"
    permuted = apply_ifim_permutation(
        tokens,
        instruction_token_ids=instruction,
        span=(start, end),
        mode=mode,
        special_token_ids=special_token_ids,
    )
    return AstFimResult(
        token_ids=permuted,
        span=(start, end),
        mode=mode,
        kind=kind,
        chunk_index=chunk_index,
    )


__all__ = [
    "DEFAULT_AST_FIM_RATE",
    "AstFimKind",
    "AstFimResult",
    "NoEligibleChunkError",
    "apply_ast_fim",
    "apply_ast_ifim",
    "eligible_ast_chunk_indices",
    "select_ast_span",
]
