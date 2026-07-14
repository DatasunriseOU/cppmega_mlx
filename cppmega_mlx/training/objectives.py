"""Stage-1 objective builders over typed CodePacket / CommitPacket inputs.

Each builder returns an ``ObjectiveExample`` carrying aligned
``(input_ids, target_ids, loss_mask)`` arrays where ``target_ids[i]`` is the
next-token label for ``input_ids[i]`` (standard shifted-by-one LM convention) and
``loss_mask[i]`` is ``1`` exactly on the positions whose prediction should be
trained for this objective.  All three arrays share length ``S`` (the number of
INPUT positions; the final position predicts the trailing EOT / sentinel).

The objectives:

  * CAUSAL_LM            — predict every next token (full sequence).
  * FIM / AST_FIM / IFIM — Fill-in-the-Middle (delegates span selection + token
                           permutation to :mod:`cppmega_mlx.data.ast_fim`, which
                           itself reuses :mod:`cppmega_mlx.data.fim`); loss is on
                           the MIDDLE span (after ``FIM_MIDDLE``).
  * COMMIT_DIFF          — predict the unified diff (+ trailing EOT) from the
                           commit message; loss on the diff tokens.
  * PRE_TO_POST          — predict the post-edit file from the pre-edit file
                           (+ commit message context); loss on the post tokens.
  * SYMBOL/TYPE/CALLEE_RECOVERY — mask an identifier / type / callee token span
                           (located via symbol_ids / type_refs / call_targets) and
                           train the model to recover the exact masked tokens.

RULE #1 (fail fast / fail loud): every required field is validated; an absent one
RAISES with WHERE + WHAT.  No silent fallback, no fabricated channels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.ast_fim import (
    DEFAULT_AST_FIM_RATE,
    AstFimResult,
    apply_ast_fim,
    logical_document_spans,
)
from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.fim import (
    FIMMode,
    FIMSpecialTokenIds,
    FIMSpecialTokenInput,
    apply_fim_permutation,
    apply_ifim_permutation,
    sample_middle_span,
)
from cppmega_mlx.data.tokenizer_contract import (
    OBJECTIVE_BOUNDARY_TOKEN_IDS,
    REQUIRED_SPECIAL_TOKEN_IDS,
    TOOL_USE_SPECIAL_TOKEN_IDS,
)

RecoveryKind = Literal["symbol", "type", "callee"]


@dataclass(frozen=True)
class ObjectiveExample:
    """One training example: aligned input / target / loss-mask token arrays.

    ``input_ids`` and ``target_ids`` are int32 ``mx.array`` of identical length
    ``S``; ``loss_mask`` is an int32 ``mx.array`` of the same length with ``1`` on
    trained positions.  ``objective`` records which builder produced it and
    ``metadata`` carries provenance (e.g. FIM ``kind``/``mode``, recovery span).
    """

    input_ids: mx.array
    target_ids: mx.array
    loss_mask: mx.array
    objective: str
    metadata: dict

    def __post_init__(self) -> None:
        n = int(self.input_ids.shape[0])
        for name in ("target_ids", "loss_mask"):
            arr = getattr(self, name)
            if int(arr.shape[0]) != n:
                raise ValueError(
                    f"ObjectiveExample.{name} length {int(arr.shape[0])} != "
                    f"input_ids length {n}"
                )


def _i32(values) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def _token_list(value: mx.array, *, where: str) -> list[int]:
    if value is None:
        raise ValueError(f"{where}: required token field is absent (None)")
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(
            f"{where}: expected 1-D token_ids, got shape {tuple(arr.shape)}"
        )
    return [int(x) for x in arr.reshape(-1).tolist()]


def _shifted(ids: list[int], loss_positions: list[bool]) -> tuple[mx.array, mx.array, mx.array]:
    """Build (input, target, mask) from a full token id list.

    ``input_ids = ids[:-1]``, ``target_ids = ids[1:]``.  ``loss_positions`` is a
    per-TARGET boolean (len == len(ids)-1): True where that next-token prediction
    is trained.
    """

    if len(ids) < 2:
        raise ValueError(
            f"sequence too short to form an LM example: need >=2 tokens, got "
            f"{len(ids)}"
        )
    inputs = ids[:-1]
    targets = ids[1:]
    if len(loss_positions) != len(targets):
        raise ValueError(
            f"loss_positions length {len(loss_positions)} != targets length "
            f"{len(targets)}"
        )
    mask = [1 if flag else 0 for flag in loss_positions]
    return _i32(inputs), _i32(targets), _i32(mask)


# --------------------------------------------------------------------------- #
# CAUSAL_LM
# --------------------------------------------------------------------------- #
def build_causal_lm(packet: CodePacket) -> ObjectiveExample:
    """Predict next tokens without supervising cross-document transitions."""

    ids = _token_list(packet.token_ids, where="causal_lm: CodePacket.token_ids")
    document_spans = logical_document_spans(packet)
    document_ids = [0] * len(ids)
    for start, end, document_id in document_spans:
        document_ids[start:end] = [document_id] * (end - start)
    loss_positions = [
        document_ids[index] == document_ids[index + 1]
        for index in range(len(ids) - 1)
    ]
    if not any(loss_positions):
        raise ValueError(
            "causal_lm requires at least one within-document token transition"
        )
    inputs, targets, mask = _shifted(ids, loss_positions)
    return ObjectiveExample(
        input_ids=inputs,
        target_ids=targets,
        loss_mask=mask,
        objective="causal_lm",
        metadata={
            "length": len(ids),
            "document_spans": tuple((start, end) for start, end, _ in document_spans),
        },
    )


# --------------------------------------------------------------------------- #
# FIM / AST_FIM / IFIM
# --------------------------------------------------------------------------- #
def _resolve_special_ids(special_token_ids: FIMSpecialTokenInput) -> FIMSpecialTokenIds:
    if special_token_ids is None:
        return FIMSpecialTokenIds()
    if isinstance(special_token_ids, FIMSpecialTokenIds):
        return special_token_ids
    return FIMSpecialTokenIds.from_mapping(special_token_ids)


def _fim_loss_positions(
    permuted: list[int], *, fim_middle_id: int, eot_id: int
) -> list[bool]:
    """Trained TARGET positions for a FIM/iFIM permutation.

    The permutation ends with ``... FIM_MIDDLE <middle...> EOT``.  We train every
    next-token prediction whose label falls in the middle span OR is the trailing
    EOT.  Concretely: targets are ``permuted[1:]``; a target position ``j`` (which
    predicts ``permuted[j+1]``) is trained iff ``permuted[j]`` is the FIM_MIDDLE
    marker or lies strictly after it.
    """

    try:
        middle_pos = len(permuted) - 1 - permuted[::-1].index(fim_middle_id)
    except ValueError as exc:
        raise ValueError(
            "fim: permuted sequence has no FIM_MIDDLE marker; cannot locate the "
            "middle span to supervise"
        ) from exc
    if permuted[-1] != eot_id:
        raise ValueError(
            f"fim: permuted sequence must end with EOT id {eot_id}, got "
            f"{permuted[-1]}"
        )
    targets = permuted[1:]
    # target index j supervises permuted[j+1]; train when its SOURCE permuted[j]
    # is at or after the FIM_MIDDLE marker.
    return [(j >= middle_pos) for j in range(len(targets))]


def _example_from_fim(
    result: AstFimResult,
    *,
    objective: str,
    special_token_ids: FIMSpecialTokenInput,
) -> ObjectiveExample:
    ids = _resolve_special_ids(special_token_ids)
    loss_positions = _fim_loss_positions(
        result.token_ids, fim_middle_id=ids.fim_middle, eot_id=ids.eot
    )
    inputs, targets, mask = _shifted(result.token_ids, loss_positions)
    return ObjectiveExample(
        input_ids=inputs,
        target_ids=targets,
        loss_mask=mask,
        objective=objective,
        metadata={
            "fim_kind": result.kind,
            "fim_mode": result.mode,
            "span": result.span,
            "chunk_index": result.chunk_index,
            "source_document_span": result.document_span,
        },
    )


def build_ast_fim(
    packet: CodePacket,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
    spm_rate: float = 0.5,
    ast_fim_rate: float = DEFAULT_AST_FIM_RATE,
    special_token_ids: FIMSpecialTokenInput = None,
) -> ObjectiveExample:
    """AST-aware FIM (90% whole-chunk middle / 10% char-FIM) objective example."""

    result = apply_ast_fim(
        packet,
        seed=seed,
        rng=rng,
        spm_rate=spm_rate,
        ast_fim_rate=ast_fim_rate,
        special_token_ids=special_token_ids,
    )
    return _example_from_fim(
        result,
        objective="ast_fim" if result.kind == "ast_fim" else "fim",
        special_token_ids=special_token_ids,
    )


def _random_fim_result(
    packet: CodePacket,
    *,
    instruction_token_ids: list[int] | None,
    seed: int | None,
    rng: random.Random | None,
    spm_rate: float,
    special_token_ids: FIMSpecialTokenInput,
) -> AstFimResult:
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")
    if not 0.0 <= spm_rate <= 1.0:
        raise ValueError(f"spm_rate must be in [0, 1], got {spm_rate}")
    rand = rng if rng is not None else random.Random(seed)
    all_tokens = _token_list(packet.token_ids, where="fim: CodePacket.token_ids")
    eligible_documents = [
        span
        for span in logical_document_spans(packet)
        if span[1] - span[0] >= 3
    ]
    if not eligible_documents:
        raise ValueError(
            "fim requires at least one logical document with at least 3 tokens"
        )
    document_start, document_end, _document_id = eligible_documents[
        rand.randrange(len(eligible_documents))
    ]
    tokens = all_tokens[document_start:document_end]
    span = sample_middle_span(len(tokens), rng=rand)
    mode: FIMMode = "spm" if rand.random() < spm_rate else "psm"
    if instruction_token_ids is None:
        permuted = apply_fim_permutation(
            tokens,
            span=span,
            mode=mode,
            special_token_ids=special_token_ids,
        )
        kind = "fim"
    else:
        permuted = apply_ifim_permutation(
            tokens,
            instruction_token_ids=instruction_token_ids,
            span=span,
            mode=mode,
            special_token_ids=special_token_ids,
        )
        kind = "ifim"
    return AstFimResult(
        token_ids=permuted,
        span=span,
        mode=mode,
        kind=kind,
        chunk_index=None,
        document_span=(document_start, document_end),
    )


def build_fim(
    packet: CodePacket,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
    spm_rate: float = 0.5,
    special_token_ids: FIMSpecialTokenInput = None,
) -> ObjectiveExample:
    """Plain token-span FIM, distinct from clang chunk-aware AST-FIM."""

    result = _random_fim_result(
        packet,
        instruction_token_ids=None,
        seed=seed,
        rng=rng,
        spm_rate=spm_rate,
        special_token_ids=special_token_ids,
    )
    return _example_from_fim(result, objective="fim", special_token_ids=special_token_ids)


def build_ifim(
    packet: CodePacket,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
    spm_rate: float = 0.5,
    special_token_ids: FIMSpecialTokenInput = None,
) -> ObjectiveExample:
    """Instruction-aware token-span FIM from typed upstream instruction IDs."""

    instruction_ids = _token_list(
        packet.ifim_instruction_token_ids,
        where="ifim: CodePacket.ifim_instruction_token_ids",
    )
    if not instruction_ids:
        raise ValueError(
            "ifim: CodePacket.ifim_instruction_token_ids must not be empty"
        )
    result = _random_fim_result(
        packet,
        instruction_token_ids=instruction_ids,
        seed=seed,
        rng=rng,
        spm_rate=spm_rate,
        special_token_ids=special_token_ids,
    )
    return _example_from_fim(
        result, objective="ifim", special_token_ids=special_token_ids
    )


# --------------------------------------------------------------------------- #
# COMMIT_DIFF / PRE_TO_POST
# --------------------------------------------------------------------------- #
def build_commit_diff(
    packet: CommitPacket,
    *,
    special_token_ids: FIMSpecialTokenInput = None,
) -> ObjectiveExample:
    """Predict the unified diff from the commit message.

    Sequence: ``COMMENT_START <commit_msg> COMMENT_END DIFF_START
    <diff_token_ids> DIFF_END EOT``.  Loss on the diff body, DIFF_END, and EOT.
    RAISES if either ``commit_msg`` or ``diff_token_ids`` is absent.
    """

    ids = _resolve_special_ids(special_token_ids)
    msg = _token_list(packet.commit_msg, where="commit_diff: CommitPacket.commit_msg")
    diff = _token_list(
        packet.diff_token_ids, where="commit_diff: CommitPacket.diff_token_ids"
    )
    comment_start = OBJECTIVE_BOUNDARY_TOKEN_IDS["COMMENT_START"]
    comment_end = OBJECTIVE_BOUNDARY_TOKEN_IDS["COMMENT_END"]
    diff_start = OBJECTIVE_BOUNDARY_TOKEN_IDS["DIFF_START"]
    diff_end = OBJECTIVE_BOUNDARY_TOKEN_IDS["DIFF_END"]
    message_section = [comment_start, *msg, comment_end]
    diff_section = [diff_start, *diff, diff_end]
    full = [*message_section, *diff_section, ids.eot]
    # DIFF_START is context. Supervision begins with the first real diff token.
    diff_content_start = len(message_section) + 1
    targets = full[1:]
    loss_positions = [
        (target_index + 1) >= diff_content_start
        for target_index in range(len(targets))
    ]
    inputs, targets_arr, mask = _shifted(full, loss_positions)
    return ObjectiveExample(
        input_ids=inputs,
        target_ids=targets_arr,
        loss_mask=mask,
        objective="commit_diff",
        metadata={
            "prompt_len": diff_content_start,
            "diff_len": len(diff),
            "section_boundaries": {
                "commit_message": (0, len(message_section)),
                "diff": (len(message_section), len(message_section) + len(diff_section)),
            },
        },
    )


def build_pre_to_post(
    packet: CommitPacket,
    *,
    special_token_ids: FIMSpecialTokenInput = None,
) -> ObjectiveExample:
    """Predict the post-edit file from the pre-edit file (+ commit message).

    Sequence: ``CODE_START <pre> CODE_END COMMENT_START <commit_msg> COMMENT_END
    FILE_SEP CODE_START <post> CODE_END EOT``. Loss is on the post body,
    CODE_END, and EOT. All three typed sections are required.
    """

    ids = _resolve_special_ids(special_token_ids)
    pre = _token_list(packet.pre_token_ids, where="pre_to_post: CommitPacket.pre_token_ids")
    post = _token_list(
        packet.post_token_ids, where="pre_to_post: CommitPacket.post_token_ids"
    )
    msg = _token_list(
        packet.commit_msg, where="pre_to_post: CommitPacket.commit_msg"
    )
    code_start = REQUIRED_SPECIAL_TOKEN_IDS["CODE_START"]
    code_end = TOOL_USE_SPECIAL_TOKEN_IDS["CODE_END"]
    comment_start = OBJECTIVE_BOUNDARY_TOKEN_IDS["COMMENT_START"]
    comment_end = OBJECTIVE_BOUNDARY_TOKEN_IDS["COMMENT_END"]
    file_sep = OBJECTIVE_BOUNDARY_TOKEN_IDS["FILE_SEP"]
    pre_section = [code_start, *pre, code_end]
    message_section = [comment_start, *msg, comment_end]
    post_section = [code_start, *post, code_end]
    post_section_start = len(pre_section) + len(message_section) + 1
    post_content_start = post_section_start + 1
    full = [*pre_section, *message_section, file_sep, *post_section, ids.eot]
    targets = full[1:]
    loss_positions = [
        (target_index + 1) >= post_content_start
        for target_index in range(len(targets))
    ]
    inputs, targets_arr, mask = _shifted(full, loss_positions)
    return ObjectiveExample(
        input_ids=inputs,
        target_ids=targets_arr,
        loss_mask=mask,
        objective="pre_to_post",
        metadata={
            "prompt_len": post_content_start,
            "post_len": len(post),
            "msg_len": len(msg),
            "section_boundaries": {
                "pre": (0, len(pre_section)),
                "commit_message": (
                    len(pre_section),
                    len(pre_section) + len(message_section),
                ),
                "post": (post_section_start, post_section_start + len(post_section)),
            },
        },
    )


# --------------------------------------------------------------------------- #
# SYMBOL / TYPE / CALLEE RECOVERY
# --------------------------------------------------------------------------- #
_RECOVERY_FIELD = {
    "symbol": "symbol_ids",
    "type": "type_refs",
    "callee": "call_targets",
}


def _find_recovery_span(
    marker: list[int], *, rng: random.Random, where: str
) -> tuple[int, int, int]:
    """Pick one contiguous non-zero run in ``marker`` -> (start, end, marker_id).

    A run of equal non-zero ids marks the identifier/type/callee token span to
    recover.  RAISES if no non-zero marker is present (the required channel is
    populated but carries no recoverable span).
    """

    runs: list[tuple[int, int, int]] = []
    i = 0
    n = len(marker)
    while i < n:
        if marker[i] != 0:
            j = i
            while j < n and marker[j] == marker[i]:
                j += 1
            runs.append((i, j, marker[i]))
            i = j
        else:
            i += 1
    if not runs:
        raise ValueError(
            f"{where}: channel has no non-zero recoverable span (all zeros); "
            "cannot build a recovery example"
        )
    start, end, marker_id = runs[rng.randrange(len(runs))]
    return start, end, marker_id


def build_recovery(
    packet: CodePacket,
    *,
    kind: RecoveryKind,
    seed: int | None = None,
    rng: random.Random | None = None,
    special_token_ids: FIMSpecialTokenInput = None,
) -> ObjectiveExample:
    """Remove an identifier/type/callee span and train exact-token recovery.

    The masked span is located from the token-aligned ``symbol_ids`` (symbol),
    ``type_refs`` (type), or ``call_targets`` (callee) channel.  The example is a
    FIM permutation where the answer occurs only after ``FIM_MIDDLE``. The
    objective marker identifies the typed recovery channel, while the source
    answer cannot leak through the causal prefix.

    RAISES if ``kind`` is unknown, the required channel is absent, or the channel
    carries no non-zero span.
    """

    if kind not in _RECOVERY_FIELD:
        raise ValueError(
            f"recovery: kind must be one of {sorted(_RECOVERY_FIELD)}, got {kind!r}"
        )
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")
    rand = rng if rng is not None else random.Random(seed)

    field_name = _RECOVERY_FIELD[kind]
    channel = getattr(packet, field_name)
    if channel is None:
        raise ValueError(
            f"{kind}_recovery: required CodePacket.{field_name} channel is absent "
            "(None); cannot locate a span to recover"
        )
    ids = _token_list(packet.token_ids, where=f"{kind}_recovery: CodePacket.token_ids")
    marker = _token_list(channel, where=f"{kind}_recovery: CodePacket.{field_name}")
    if len(marker) != len(ids):
        raise ValueError(
            f"{kind}_recovery: {field_name} length {len(marker)} != token_ids "
            f"length {len(ids)}"
        )
    start, end, marker_id = _find_recovery_span(
        marker, rng=rand, where=f"{kind}_recovery: CodePacket.{field_name}"
    )
    special = _resolve_special_ids(special_token_ids)
    recovery_marker = {
        "symbol": OBJECTIVE_BOUNDARY_TOKEN_IDS["SYMBOL_REF"],
        "type": OBJECTIVE_BOUNDARY_TOKEN_IDS["TYPE_INFO"],
        "callee": OBJECTIVE_BOUNDARY_TOKEN_IDS["OVERLOAD_SET"],
    }[kind]
    prefix = ids[:start]
    answer = ids[start:end]
    suffix = ids[end:]
    full = [
        special.fim_prefix,
        recovery_marker,
        *prefix,
        special.fim_suffix,
        *suffix,
        special.fim_middle,
        *answer,
        special.eot,
    ]
    answer_start = len(full) - len(answer) - 1
    targets = full[1:]
    loss_positions = [
        (target_index + 1) >= answer_start
        for target_index in range(len(targets))
    ]
    inputs, targets_arr, mask = _shifted(full, loss_positions)
    return ObjectiveExample(
        input_ids=inputs,
        target_ids=targets_arr,
        loss_mask=mask,
        objective=f"{kind}_recovery",
        metadata={
            "span": (start, end),
            "marker_id": marker_id,
            "recovery_marker_id": recovery_marker,
            "answer_start": answer_start,
        },
    )


__all__ = [
    "ObjectiveExample",
    "RecoveryKind",
    "build_ast_fim",
    "build_causal_lm",
    "build_commit_diff",
    "build_fim",
    "build_ifim",
    "build_pre_to_post",
    "build_recovery",
]
