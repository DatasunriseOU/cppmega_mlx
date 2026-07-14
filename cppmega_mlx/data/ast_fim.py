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
    FIMSpecialTokenInput,
    apply_domain_preserving_fim_permutation,
    domain_preserving_fim_region,
    sample_middle_span,
)

# Fraction of AST-FIM examples; the remaining slice is random-char-FIM.
DEFAULT_AST_FIM_RATE = 0.9

AstFimKind = Literal[
    "fim",
    "ifim",
    "ast_fim",
    "char_fim",
    "ast_ifim",
    "char_ifim",
]


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
    source_token_indices: list[int]
    document_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class DomainPreservingDocumentSpan:
    """One logical document and the absolute interior safe to permute."""

    document_start: int
    document_end: int
    document_id: int
    content_start: int
    content_end: int


def _token_count(packet: CodePacket) -> int:
    if packet.token_ids.ndim != 1:
        raise ValueError(
            f"ast_fim requires a single-window CodePacket (1-D token_ids), got "
            f"shape {tuple(packet.token_ids.shape)}"
        )
    return int(packet.token_ids.shape[0])


def _packet_token_list(packet: CodePacket) -> list[int]:
    return [int(x) for x in np.asarray(packet.token_ids).reshape(-1).tolist()]


def logical_document_spans(packet: CodePacket) -> tuple[tuple[int, int, int], ...]:
    """Return contiguous positive ``document_ids`` runs as ``(start,end,id)``."""

    length = _token_count(packet)
    if packet.document_ids is None:
        return ((0, length, 1),)
    document_ids = [
        int(value) for value in np.asarray(packet.document_ids).reshape(-1).tolist()
    ]
    if len(document_ids) != length:
        raise ValueError(
            f"document_ids length {len(document_ids)} != token length {length}"
        )
    if any(document_id <= 0 for document_id in document_ids):
        raise ValueError("document_ids must be positive on every objective token")
    spans: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, length + 1):
        if index == length or document_ids[index] != document_ids[start]:
            spans.append((start, index, document_ids[start]))
            start = index
    return tuple(spans)


def domain_preserving_document_spans(
    packet: CodePacket,
) -> tuple[DomainPreservingDocumentSpan, ...]:
    """Validate every logical document and expose its transformable interior."""

    tokens = _packet_token_list(packet)
    regions: list[DomainPreservingDocumentSpan] = []
    for document_start, document_end, document_id in logical_document_spans(packet):
        region = domain_preserving_fim_region(
            tokens[document_start:document_end],
            where=f"fim logical document [{document_start}, {document_end})",
        )
        regions.append(
            DomainPreservingDocumentSpan(
                document_start=document_start,
                document_end=document_end,
                document_id=document_id,
                content_start=document_start + region.content_start,
                content_end=document_start + region.content_end,
            )
        )
    return tuple(regions)


def _containing_document(
    start: int,
    end: int,
    document_spans: Sequence[tuple[int, int, int]],
    *,
    chunk_index: int,
) -> tuple[int, int, int]:
    for document_start, document_end, document_id in document_spans:
        if document_start <= start < end <= document_end:
            return document_start, document_end, document_id
    raise ValueError(
        f"ast_fim: chunk {chunk_index} span [{start}, {end}) crosses a "
        "document_ids boundary"
    )


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
    starts: Sequence[int],
    ends: Sequence[int],
    length: int,
    document_regions: Sequence[DomainPreservingDocumentSpan],
) -> list[int]:
    """Chunks with non-empty context inside one domain-safe interior."""

    eligible: list[int] = []
    document_spans = [
        (region.document_start, region.document_end, region.document_id)
        for region in document_regions
    ]
    for idx, (start, end) in enumerate(zip(starts, ends)):
        if not 0 <= start < end <= length:
            raise ValueError(
                f"ast_fim: chunk {idx} span [{start}, {end}) out of bounds for "
                f"{length} tokens (chunk boundaries must be valid token offsets)"
            )
        document_start, document_end, document_id = _containing_document(
            start, end, document_spans, chunk_index=idx
        )
        region = next(
            region
            for region in document_regions
            if (
                region.document_start,
                region.document_end,
                region.document_id,
            )
            == (document_start, document_end, document_id)
        )
        if region.content_start < start and end < region.content_end:
            eligible.append(idx)
    return eligible


def _select_ast_span_in_document(
    packet: CodePacket,
    *,
    rng: random.Random,
) -> tuple[int, int, int, int, int, int]:
    """Return document-relative span, chunk index, and absolute document span."""

    length = _token_count(packet)
    if length < 3:
        raise ValueError(
            "ast_fim requires at least 3 tokens to form prefix/middle/suffix, "
            f"got {length}"
        )
    starts, ends, _kinds = _chunk_arrays(packet)
    document_regions = domain_preserving_document_spans(packet)
    eligible = _eligible_chunks(starts, ends, length, document_regions)
    if not eligible:
        raise NoEligibleChunkError(
            "ast_fim: no clang chunk yields non-empty context inside one "
            "logical document"
        )
    chunk_index = eligible[rng.randrange(len(eligible))]
    absolute_start = starts[chunk_index]
    absolute_end = ends[chunk_index]
    region = next(
        region
        for region in document_regions
        if region.document_start <= absolute_start < absolute_end <= region.document_end
    )
    return (
        absolute_start - region.content_start,
        absolute_end - region.content_start,
        chunk_index,
        region.document_start,
        region.document_end,
        region.content_start,
    )


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

    start, end, chunk_index, _document_start, _document_end, content_start = (
        _select_ast_span_in_document(packet, rng=rng)
    )
    return start + content_start, end + content_start, chunk_index


def eligible_ast_chunk_indices(packet: CodePacket) -> tuple[int, ...]:
    """Return validated interior clang chunks without drawing randomness."""

    length = _token_count(packet)
    if length < 3:
        return ()
    starts, ends, _kinds = _chunk_arrays(packet)
    return tuple(
        _eligible_chunks(starts, ends, length, domain_preserving_document_spans(packet))
    )


def _char_document_span(
    packet: CodePacket, *, rng: random.Random
) -> DomainPreservingDocumentSpan:
    eligible = [
        region
        for region in domain_preserving_document_spans(packet)
        if region.content_end - region.content_start >= 3
    ]
    if not eligible:
        raise ValueError(
            "fim requires at least one logical document with at least 3 tokens"
        )
    return eligible[rng.randrange(len(eligible))]


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

    all_tokens = _packet_token_list(packet)
    length = len(all_tokens)
    if length < 3:
        raise ValueError(f"ast_fim requires at least 3 tokens, got {length}")

    use_ast = rand.random() < ast_fim_rate
    if use_ast:
        try:
            start, end, chunk_index, document_start, document_end, _content_start = (
                _select_ast_span_in_document(packet, rng=rand)
            )
            kind: AstFimKind = "ast_fim"
        except NoEligibleChunkError:
            # No usable chunk for THIS window: fall to the char slice and RECORD
            # it (not a silent degraded path — the kind makes it observable).
            # A genuinely absent chunk field is a plain ValueError and propagates.
            region = _char_document_span(packet, rng=rand)
            document_start = region.document_start
            document_end = region.document_end
            start, end = sample_middle_span(
                region.content_end - region.content_start,
                rng=rand,
            )
            chunk_index = None
            kind = "char_fim"
    else:
        region = _char_document_span(packet, rng=rand)
        document_start = region.document_start
        document_end = region.document_end
        start, end = sample_middle_span(
            region.content_end - region.content_start,
            rng=rand,
        )
        chunk_index = None
        kind = "char_fim"

    tokens = all_tokens[document_start:document_end]
    mode: FIMMode = "spm" if rand.random() < spm_rate else "psm"
    permuted = apply_domain_preserving_fim_permutation(
        tokens,
        source_start=document_start,
        span=(start, end),
        mode=mode,
        special_token_ids=special_token_ids,
        where=f"ast_fim logical document [{document_start}, {document_end})",
    )
    return AstFimResult(
        token_ids=permuted.token_ids,
        span=(start, end),
        mode=mode,
        kind=kind,
        chunk_index=chunk_index,
        source_token_indices=permuted.source_token_indices,
        document_span=(document_start, document_end),
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
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in instruction
    ):
        raise ValueError(
            "ast_ifim instruction_token_ids must contain integer token ids"
        )
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")
    if not 0.0 <= ast_fim_rate <= 1.0:
        raise ValueError(f"ast_fim_rate must be in [0, 1], got {ast_fim_rate}")
    if not 0.0 <= spm_rate <= 1.0:
        raise ValueError(f"spm_rate must be in [0, 1], got {spm_rate}")
    rand = rng if rng is not None else random.Random(seed)

    all_tokens = _packet_token_list(packet)
    length = len(all_tokens)
    if length < 3:
        raise ValueError(f"ast_ifim requires at least 3 tokens, got {length}")

    use_ast = rand.random() < ast_fim_rate
    if use_ast:
        try:
            start, end, chunk_index, document_start, document_end, _content_start = (
                _select_ast_span_in_document(packet, rng=rand)
            )
            kind: AstFimKind = "ast_ifim"
        except NoEligibleChunkError:
            region = _char_document_span(packet, rng=rand)
            document_start = region.document_start
            document_end = region.document_end
            start, end = sample_middle_span(
                region.content_end - region.content_start,
                rng=rand,
            )
            chunk_index = None
            kind = "char_ifim"
    else:
        region = _char_document_span(packet, rng=rand)
        document_start = region.document_start
        document_end = region.document_end
        start, end = sample_middle_span(
            region.content_end - region.content_start,
            rng=rand,
        )
        chunk_index = None
        kind = "char_ifim"

    tokens = all_tokens[document_start:document_end]
    mode: FIMMode = "spm" if rand.random() < spm_rate else "psm"
    permuted = apply_domain_preserving_fim_permutation(
        tokens,
        source_start=document_start,
        instruction_token_ids=instruction,
        span=(start, end),
        mode=mode,
        special_token_ids=special_token_ids,
        where=f"ast_ifim logical document [{document_start}, {document_end})",
    )
    return AstFimResult(
        token_ids=permuted.token_ids,
        span=(start, end),
        mode=mode,
        kind=kind,
        chunk_index=chunk_index,
        source_token_indices=permuted.source_token_indices,
        document_span=(document_start, document_end),
    )


__all__ = [
    "DEFAULT_AST_FIM_RATE",
    "AstFimKind",
    "AstFimResult",
    "DomainPreservingDocumentSpan",
    "NoEligibleChunkError",
    "apply_ast_fim",
    "apply_ast_ifim",
    "domain_preserving_document_spans",
    "eligible_ast_chunk_indices",
    "logical_document_spans",
    "select_ast_span",
]
