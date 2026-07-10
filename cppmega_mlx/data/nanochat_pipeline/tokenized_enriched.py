"""Helpers for offline token-level enriched parquet materialization."""

from __future__ import annotations

import bisect
import inspect
import json
from typing import Any, Sequence, cast

from cppmega_mlx.tokenizer.cpp_tokenizer import normalize_whitespace_with_offsets
from cppmega_mlx.data.domain_schema import (
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
    delimiter_token_ids,
)

from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    PLATFORM_IDS_COLUMN,
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
    EDIT_OP_PER_TOKEN_COLUMN,
    HUNK_ID_PER_TOKEN_COLUMN,
    TOKEN_AST_DEPTH_COLUMN,
    TOKEN_AST_NODE_TYPE_COLUMN,
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_CALL_TARGETS_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_CONFIDENCE_IDS_COLUMN,
    TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
    TOKEN_DEF_USE_COLUMN,
    TOKEN_DEP_LEVELS_COLUMN,
    TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    TOKEN_DOMAIN_EDGES_COLUMN,
    TOKEN_DOMAIN_IDS_COLUMN,
    TOKEN_BUILD_EDGES_COLUMN,
    TOKEN_ENTITY_IDS_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_ROLE_IDS_COLUMN,
    TOKEN_SCOPE_IDS_COLUMN,
    TOKEN_SIBLING_INDEX_COLUMN,
    TOKEN_SHELL_EDGES_COLUMN,
    TOKEN_SOURCE_DOC_IDS_COLUMN,
    TOKEN_STRUCTURE_IDS_COLUMN,
    TOKEN_SYMBOL_IDS_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
    TOKEN_TYPE_REFS_COLUMN,
    TOKEN_CHANGE_MASK_POST_COLUMN,
    TOKEN_CHANGE_MASK_PRE_COLUMN,
)

_KIND_STR_TO_INT = {
    "other": 0,
    "preamble": 1,
    "func_signature": 2,
    "func_body": 3,
    "class_decl": 4,
    "class_member": 5,
    "comment": 6,
    "typedef": 7,
    "namespace": 8,
}


def _kind_to_int(kind: Any) -> int:
    if isinstance(kind, int):
        return kind
    text = str(kind)
    if text.isdigit():
        return int(text)
    return _KIND_STR_TO_INT.get(text.lower(), 0)


def _normalize_graph_edge_pairs(raw_edges: Any) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for edge in raw_edges or []:
        if isinstance(edge, dict) and "from" in edge and "to" in edge:
            pairs.append((int(edge["from"]), int(edge["to"])))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            pairs.append((int(edge[0]), int(edge[1])))
    return pairs


def _normalize_graph_edge_triples(raw_edges: Any) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for edge in raw_edges or []:
        if isinstance(edge, dict):
            if "from_char" in edge and "to_char" in edge:
                src = edge["from_char"]
                dst = edge["to_char"]
            elif "from" in edge and "to" in edge:
                src = edge["from"]
                dst = edge["to"]
            elif "src" in edge and "dst" in edge:
                src = edge["src"]
                dst = edge["dst"]
            else:
                continue
            triples.append((int(src), int(dst), int(edge.get("kind", 0))))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 3:
            triples.append((int(edge[0]), int(edge[1]), int(edge[2])))
    return triples


def _encode_batch_with_optional_char_spans(
    tokenizer,
    texts: list[str],
    *,
    prepend=None,
    append=None,
    num_threads: int = 8,
) -> tuple[list[list[int]], list[list[tuple[int, int]]] | None]:
    def _call_tokenizer_encode(payload):
        kwargs = {}
        try:
            params = inspect.signature(tokenizer.encode).parameters
        except (TypeError, ValueError):
            params = {}
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_var_kwargs or "prepend" in params:
            kwargs["prepend"] = prepend
        if accepts_var_kwargs or "append" in params:
            kwargs["append"] = append
        if accepts_var_kwargs or "num_threads" in params:
            kwargs["num_threads"] = num_threads
        return tokenizer.encode(payload, **kwargs)

    hf_tok = getattr(tokenizer, "_tokenizer", None) or getattr(
        tokenizer, "tokenizer", None
    )
    if hf_tok is None or not hasattr(hf_tok, "encode_batch"):
        return _call_tokenizer_encode(texts), None

    def _resolve_special_id(value: str | int | None) -> int | None:
        if value is None or isinstance(value, int):
            return value
        if hasattr(tokenizer, "encode_special"):
            return int(tokenizer.encode_special(value))
        raise ValueError(f"Tokenizer cannot resolve special token {value!r}")

    prepend_id = _resolve_special_id(prepend)
    append_id = _resolve_special_id(append)

    # CRITICAL: apply the SAME whitespace-sentinel normalization as
    # ``CppMegaTokenizer.encode`` so the stored ``input_ids`` contain
    # ``<NL>``(47)/``<SPACE>``(46) and newline/indent structure is preserved.
    # Normalization changes char offsets, so we keep a per-character map from
    # the normalized string back onto the ORIGINAL text and translate every
    # per-token span through it. The doc sidecars (structure_ids/ast_depth/
    # change_mask/...) are char-aligned to the ORIGINAL text, so the returned
    # spans MUST be in original-text coordinates to stay aligned (no off-by-N).
    normalized_texts: list[str] = []
    norm_maps: list[list[tuple[int, int]]] = []
    for text in texts:
        normalized, norm_to_orig = normalize_whitespace_with_offsets(text)
        normalized_texts.append(normalized)
        norm_maps.append(norm_to_orig)

    try:
        encodings = hf_tok.encode_batch(normalized_texts, add_special_tokens=False)
    except TypeError:
        encodings = hf_tok.encode_batch(normalized_texts)

    token_lists: list[list[int]] = []
    token_spans: list[list[tuple[int, int]]] = []
    for text, norm_to_orig, enc in zip(texts, norm_maps, encodings):
        ids = list(enc.ids)
        spans = [
            _normalized_span_to_original_span(norm_to_orig, int(start), int(end))
            for start, end in enc.offsets
        ]
        if prepend is not None:
            assert prepend_id is not None
            ids.insert(0, int(prepend_id))
            spans.insert(0, (0, 0))
        if append is not None:
            assert append_id is not None
            ids.append(int(append_id))
            text_len = len(text)
            spans.append((text_len, text_len))
        token_lists.append(ids)
        token_spans.append(spans)
    return token_lists, token_spans


def _normalized_span_to_original_span(
    normalized_to_original: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int]:
    """Translate a [start, end) span in normalized coords to original coords.

    ``normalized_to_original[k]`` is the original ``(start, end)`` span the k-th
    normalized character came from. A token's normalized span maps to the union
    of the original spans of the normalized characters it covers. Empty/zero-
    width tokens map to ``(0, 0)`` (treated as invalid by the char->token
    projection, matching the prior behavior for special-token positions).
    """
    if not normalized_to_original:
        return (0, 0)
    start = min(max(start, 0), len(normalized_to_original))
    end = min(max(end, start), len(normalized_to_original))
    if end <= start:
        return (0, 0)
    covered = normalized_to_original[start:end]
    return min(item[0] for item in covered), max(item[1] for item in covered)


def _extract_token_char_starts_and_valid(
    token_positions: list[int] | list[tuple[int, int]],
) -> tuple[list[int], list[bool]]:
    if not token_positions:
        return [], []
    first = token_positions[0]
    if isinstance(first, tuple) and len(first) == 2:
        spans = cast(list[tuple[int, int]], token_positions)
        span_starts = [int(start) for start, _ in spans]
        span_valid = [int(end) > int(start) for start, end in spans]
        return span_starts, span_valid

    lengths = cast(list[int], token_positions)
    starts: list[int] = []
    valid: list[bool] = []
    running = 0
    for length in lengths:
        n = max(int(length), 0)
        starts.append(running)
        valid.append(n > 0)
        running += n
    return starts, valid


def _chars_to_tokens_structure_ids(
    char_structure_ids: list[int],
    text: str,
    token_lengths: list[int] | list[tuple[int, int]],
) -> list[int]:
    del text
    if not char_structure_ids:
        return [0] * len(token_lengths)
    starts, valid = _extract_token_char_starts_and_valid(token_lengths)
    n_struct = len(char_structure_ids)
    out: list[int] = []
    for start, is_valid in zip(starts, valid):
        if is_valid and start < n_struct:
            out.append(int(char_structure_ids[start]))
        else:
            out.append(0)
    return out


def _chunk_boundaries_to_token_offsets(
    chunk_boundaries: list[dict[str, Any]],
    text: str,
    token_lengths: list[int] | list[tuple[int, int]],
) -> list[dict[str, Any]]:
    del text
    if not chunk_boundaries or not token_lengths:
        return []
    starts, _valid = _extract_token_char_starts_and_valid(token_lengths)
    if not starts:
        return []

    result: list[dict[str, Any]] = []
    for cb in chunk_boundaries:
        char_start = int(cb.get("start", 0))
        tok_idx = bisect.bisect_right(starts, char_start) - 1
        if tok_idx < 0:
            tok_idx = 0
        elif tok_idx >= len(starts):
            tok_idx = len(starts) - 1
        result.append(
            {
                "token_offset": int(tok_idx),
                "end_char": cb.get("end", cb.get("start", 0)),
                "kind": cb.get("kind", 0),
                "name": cb.get("name", ""),
                "dep_level": cb.get("dep_level", 0),
            }
        )
    return result


def _compute_token_dep_levels(
    tok_struct_ids: list[int],
    tok_chunks: list[dict[str, Any]],
    num_tokens: int,
) -> list[int]:
    del tok_struct_ids
    if not tok_chunks:
        return [0] * num_tokens

    sorted_chunks = sorted(tok_chunks, key=lambda c: int(c["token_offset"]))
    out = [0] * num_tokens
    chunk_idx = 0
    current_dep = 0
    for token_idx in range(num_tokens):
        while chunk_idx < len(sorted_chunks) and int(sorted_chunks[chunk_idx]["token_offset"]) <= token_idx:
            current_dep = int(sorted_chunks[chunk_idx].get("dep_level", 0))
            chunk_idx += 1
        out[token_idx] = current_dep
    return out


def _remap_token_edges(
    raw_edges: list[tuple[int, int]],
    index_map: dict[int, int],
) -> list[dict[str, int]]:
    remapped = []
    for src, dst in raw_edges:
        if src in index_map and dst in index_map:
            remapped.append({"from": int(index_map[src]), "to": int(index_map[dst])})
    return remapped


def _char_position_to_token_index(
    token_spans: list[tuple[int, int]],
    char_pos: int,
) -> int | None:
    if not token_spans:
        return None
    char_pos = max(int(char_pos), 0)
    for idx, (start, end) in enumerate(token_spans):
        start_i = int(start)
        end_i = int(end)
        if end_i > start_i and start_i <= char_pos < end_i:
            return idx

    valid_starts = [
        (int(start), idx)
        for idx, (start, end) in enumerate(token_spans)
        if int(end) > int(start)
    ]
    if not valid_starts:
        return None
    starts = [item[0] for item in valid_starts]
    pos = bisect.bisect_right(starts, char_pos) - 1
    if pos < 0:
        return valid_starts[0][1]
    return valid_starts[min(pos, len(valid_starts) - 1)][1]


def _remap_char_edge_triples_to_tokens(
    raw_edges: Any,
    token_spans: list[tuple[int, int]],
) -> list[dict[str, int]]:
    remapped: list[dict[str, int]] = []
    for src_char, dst_char, kind in _normalize_graph_edge_triples(raw_edges):
        src = _char_position_to_token_index(token_spans, src_char)
        dst = _char_position_to_token_index(token_spans, dst_char)
        if src is None or dst is None:
            continue
        remapped.append({"from": int(src), "to": int(dst), "kind": int(kind)})
    return remapped


def _build_token_chunk_layout(
    doc: dict[str, Any],
    tok_chunks: list[dict[str, Any]],
    token_count: int,
) -> dict[str, list[int] | list[dict[str, int]]]:
    if not tok_chunks:
        return {
            TOKEN_CHUNK_STARTS_COLUMN: [],
            TOKEN_CHUNK_ENDS_COLUMN: [],
            TOKEN_CHUNK_KINDS_COLUMN: [],
            TOKEN_CHUNK_DEP_LEVELS_COLUMN: [],
            TOKEN_CALL_EDGES_COLUMN: [],
            TOKEN_TYPE_EDGES_COLUMN: [],
        }

    chunk_entries = list(enumerate(tok_chunks))
    chunk_entries.sort(key=lambda item: int(item[1].get("token_offset", 0)))
    index_map = {orig_idx: new_idx for new_idx, (orig_idx, _) in enumerate(chunk_entries)}

    starts = [int(chunk.get("token_offset", 0)) for _, chunk in chunk_entries]
    ends = [
        starts[i + 1] if i + 1 < len(starts) else int(token_count)
        for i in range(len(starts))
    ]
    kinds = [_kind_to_int(chunk.get("kind", 0)) for _, chunk in chunk_entries]
    dep_levels = [int(chunk.get("dep_level", 0)) for _, chunk in chunk_entries]

    call_edges = _remap_token_edges(
        _normalize_graph_edge_pairs(doc.get("call_edges", [])),
        index_map,
    )
    type_edges = _remap_token_edges(
        _normalize_graph_edge_pairs(doc.get("type_edges", [])),
        index_map,
    )

    return {
        TOKEN_CHUNK_STARTS_COLUMN: starts,
        TOKEN_CHUNK_ENDS_COLUMN: ends,
        TOKEN_CHUNK_KINDS_COLUMN: kinds,
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: dep_levels,
        TOKEN_CALL_EDGES_COLUMN: call_edges,
        TOKEN_TYPE_EDGES_COLUMN: type_edges,
    }


def _tokenize_optional_char_field(
    doc: dict[str, Any],
    keys: tuple[str, ...],
    token_spans: list[tuple[int, int]],
) -> list[int]:
    for key in keys:
        values = doc.get(key, [])
        if values:
            return _chars_to_tokens_structure_ids(values, "", token_spans)
    return []


def _domain_kind_from_doc(doc: dict[str, Any]) -> DomainKind | None:
    raw = doc.get("domain_kind")
    if raw in (None, "", 0, "0"):
        return None
    try:
        if isinstance(raw, str) and not raw.isdigit():
            return DomainKind[raw.upper()]
        return DomainKind(int(raw))
    except (KeyError, TypeError, ValueError):
        return None


def _shift_token_span_values(values: list[int], *, insert_at: int) -> list[int]:
    return [int(value) + 1 if int(value) >= insert_at else int(value) for value in values]


def _shift_token_edge_triples(
    edges: list[dict[str, int]],
    *,
    insert_at: int,
) -> list[dict[str, int]]:
    shifted: list[dict[str, int]] = []
    for edge in edges:
        src = int(edge["from"])
        dst = int(edge["to"])
        shifted.append(
            {
                "from": src + 1 if src >= insert_at else src,
                "to": dst + 1 if dst >= insert_at else dst,
                "kind": int(edge.get("kind", 0)),
            }
        )
    return shifted


def _insert_domain_delimiters(
    row: dict[str, Any],
    *,
    domain: DomainKind,
    insert_at: int = 1,
) -> None:
    if DomainKind(domain) == DomainKind.UNKNOWN:
        return
    try:
        start_id, end_id = delimiter_token_ids(domain)
    except KeyError as exc:
        raise ValueError(f"missing delimiter token contract for domain {domain!r}") from exc
    token_ids = list(row[TOKEN_IDS_COLUMN])
    if insert_at > len(token_ids):
        insert_at = len(token_ids)
    row[TOKEN_IDS_COLUMN] = token_ids[:insert_at] + [int(start_id)] + token_ids[insert_at:] + [int(end_id)]

    token_count_before = len(token_ids)
    token_count_after = token_count_before + 2
    domain_value = int(domain)
    delimiter_role = int(DomainRoleKind.DELIMITER)
    exact_confidence = int(ParseConfidence.EXACT)

    dense_defaults = {
        TOKEN_DOMAIN_IDS_COLUMN: domain_value,
        TOKEN_ROLE_IDS_COLUMN: delimiter_role,
        TOKEN_ENTITY_IDS_COLUMN: 0,
        TOKEN_SCOPE_IDS_COLUMN: 0,
        TOKEN_SOURCE_DOC_IDS_COLUMN: 0,
        TOKEN_CONFIDENCE_IDS_COLUMN: exact_confidence,
    }
    for column, delimiter_value in dense_defaults.items():
        values = list(row.get(column, []))
        if not values:
            values = [0] * token_count_before
        if len(values) != token_count_before:
            raise ValueError(
                f"{column} length {len(values)} does not match token count {token_count_before}"
            )
        row[column] = (
            values[:insert_at]
            + [int(delimiter_value)]
            + values[insert_at:]
            + [int(delimiter_value)]
        )

    for column in (
        TOKEN_STRUCTURE_IDS_COLUMN,
        TOKEN_DEP_LEVELS_COLUMN,
        TOKEN_AST_DEPTH_COLUMN,
        TOKEN_SIBLING_INDEX_COLUMN,
        TOKEN_AST_NODE_TYPE_COLUMN,
        TOKEN_SYMBOL_IDS_COLUMN,
        TOKEN_CALL_TARGETS_COLUMN,
        TOKEN_TYPE_REFS_COLUMN,
        TOKEN_DEF_USE_COLUMN,
        TOKEN_CHANGE_MASK_PRE_COLUMN,
        TOKEN_CHANGE_MASK_POST_COLUMN,
        HUNK_ID_PER_TOKEN_COLUMN,
        EDIT_OP_PER_TOKEN_COLUMN,
    ):
        values = list(row.get(column, []))
        if not values:
            continue
        if len(values) != token_count_before:
            raise ValueError(
                f"{column} length {len(values)} does not match token count {token_count_before}"
            )
        pad_value = -1 if column == HUNK_ID_PER_TOKEN_COLUMN else 0
        row[column] = values[:insert_at] + [pad_value] + values[insert_at:] + [pad_value]

    for column in (TOKEN_CHUNK_STARTS_COLUMN, TOKEN_CHUNK_ENDS_COLUMN):
        row[column] = _shift_token_span_values(list(row.get(column, [])), insert_at=insert_at)

    for column in (
        TOKEN_DOMAIN_EDGES_COLUMN,
        TOKEN_BUILD_EDGES_COLUMN,
        TOKEN_SHELL_EDGES_COLUMN,
        TOKEN_DIAGNOSTIC_EDGES_COLUMN,
        TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
    ):
        row[column] = _shift_token_edge_triples(
            list(row.get(column, [])),
            insert_at=insert_at,
        )

    if len(row[TOKEN_IDS_COLUMN]) != token_count_after:
        raise AssertionError("domain delimiter insertion corrupted token length")


def _changed_chunk_metadata(
    token_chunk_starts: list[int],
    token_chunk_ends: list[int],
    change_masks: tuple[list[int], ...],
) -> tuple[list[int], list[dict[str, int]]]:
    if not token_chunk_starts or not token_chunk_ends:
        return [], []
    changed_ids: list[int] = []
    changed_spans: list[dict[str, int]] = []
    for chunk_idx, (start, end) in enumerate(zip(token_chunk_starts, token_chunk_ends)):
        start_i = max(int(start), 0)
        end_i = max(int(end), start_i)
        changed = False
        for mask in change_masks:
            if not mask:
                continue
            bounded_end = min(end_i, len(mask))
            if start_i < bounded_end and any(int(value) != 0 for value in mask[start_i:bounded_end]):
                changed = True
                break
        if changed:
            changed_ids.append(chunk_idx)
            changed_spans.append({"start": start_i, "end": end_i})
    return changed_ids, changed_spans


def _platform_ids_from_doc(doc: dict[str, Any]) -> list[int]:
    platform_info = doc.get("platform_info")
    if not platform_info:
        return []
    if isinstance(platform_info, str):
        try:
            platform_info = json.loads(platform_info)
        except json.JSONDecodeError:
            return []
    if not isinstance(platform_info, dict):
        return []

    from cppmega_mlx.data.nanochat_pipeline.platform_vocab import platform_info_to_ids

    return platform_info_to_ids(platform_info)


def materialize_tokenized_enriched_batch(
    docs: list[dict[str, Any]],
    tokenizer,
    *,
    num_threads: int = 8,
) -> list[dict[str, Any]]:
    """Convert enriched char-level docs into token-level parquet-ready metadata.

    The returned token arrays align to `token_ids`, which already include the BOS
    token at position 0.
    """
    if not docs:
        return []

    texts = [str(doc.get("text", "")) for doc in docs]
    if hasattr(tokenizer, "get_bos_token_id"):
        bos_token = tokenizer.get_bos_token_id()
    else:
        bos_token = tokenizer.bos_token_id
    token_lists, token_spans_batch = _encode_batch_with_optional_char_spans(
        tokenizer,
        texts,
        prepend=bos_token,
        num_threads=num_threads,
    )
    if token_spans_batch is None:
        raise ValueError(
            "Tokenizer did not expose per-token character spans; "
            "cannot materialize token-level enriched metadata offline."
        )

    out = []
    for doc, token_ids, token_spans in zip(docs, token_lists, token_spans_batch):
        token_ids = [int(tok) for tok in token_ids]
        token_structure_ids = _chars_to_tokens_structure_ids(
            doc.get("structure_ids", []),
            "",
            token_spans,
        )
        tok_chunks = _chunk_boundaries_to_token_offsets(
            doc.get("chunk_boundaries", []),
            "",
            token_spans,
        )
        token_dep_levels = _compute_token_dep_levels(
            token_structure_ids,
            tok_chunks,
            len(token_ids),
        )

        token_change_mask_pre = _tokenize_optional_char_field(
            doc,
            ("change_mask_pre", TOKEN_CHANGE_MASK_PRE_COLUMN),
            token_spans,
        )
        token_change_mask_post = _tokenize_optional_char_field(
            doc,
            ("change_mask_post", "token_change_mask", TOKEN_CHANGE_MASK_POST_COLUMN),
            token_spans,
        )
        hunk_id_per_token = _tokenize_optional_char_field(
            doc,
            ("hunk_id_per_char", HUNK_ID_PER_TOKEN_COLUMN),
            token_spans,
        )
        edit_op_per_token = _tokenize_optional_char_field(
            doc,
            ("edit_op_per_char", "token_edit_op", EDIT_OP_PER_TOKEN_COLUMN),
            token_spans,
        )

        row: dict[str, Any] = {
            TOKEN_IDS_COLUMN: token_ids,
            PLATFORM_IDS_COLUMN: _platform_ids_from_doc(doc),
            TOKEN_STRUCTURE_IDS_COLUMN: token_structure_ids,
            TOKEN_DEP_LEVELS_COLUMN: token_dep_levels,
            TOKEN_AST_DEPTH_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("ast_depth", []),
                "",
                token_spans,
            ),
            TOKEN_SIBLING_INDEX_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("sibling_index", []),
                "",
                token_spans,
            ),
            TOKEN_AST_NODE_TYPE_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("ast_node_type", []),
                "",
                token_spans,
            ),
            TOKEN_SYMBOL_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("symbol_ids", []),
                "",
                token_spans,
            ),
            TOKEN_CALL_TARGETS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("call_targets", []),
                "",
                token_spans,
            ),
            TOKEN_TYPE_REFS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("type_refs", []),
                "",
                token_spans,
            ),
            TOKEN_DEF_USE_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("def_use", []),
                "",
                token_spans,
            ),
            TOKEN_DOMAIN_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_ids", []),
                "",
                token_spans,
            ),
            TOKEN_ROLE_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_role_ids", doc.get("role_ids", [])),
                "",
                token_spans,
            ),
            TOKEN_ENTITY_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_entity_ids", doc.get("entity_ids", [])),
                "",
                token_spans,
            ),
            TOKEN_SCOPE_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_scope_ids", doc.get("scope_ids", [])),
                "",
                token_spans,
            ),
            TOKEN_SOURCE_DOC_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_source_doc_ids", doc.get("source_doc_ids", [])),
                "",
                token_spans,
            ),
            TOKEN_CONFIDENCE_IDS_COLUMN: _chars_to_tokens_structure_ids(
                doc.get("domain_confidence_ids", doc.get("confidence_ids", [])),
                "",
                token_spans,
            ),
            TOKEN_DOMAIN_EDGES_COLUMN: _remap_char_edge_triples_to_tokens(
                doc.get("domain_edges", []),
                token_spans,
            ),
            TOKEN_BUILD_EDGES_COLUMN: _remap_char_edge_triples_to_tokens(
                doc.get("build_edges", []),
                token_spans,
            ),
            TOKEN_SHELL_EDGES_COLUMN: _remap_char_edge_triples_to_tokens(
                doc.get("shell_edges", []),
                token_spans,
            ),
            TOKEN_DIAGNOSTIC_EDGES_COLUMN: _remap_char_edge_triples_to_tokens(
                doc.get("diagnostic_edges", []),
                token_spans,
            ),
            TOKEN_CROSS_DOMAIN_EDGES_COLUMN: _remap_char_edge_triples_to_tokens(
                doc.get("cross_domain_edges", []),
                token_spans,
            ),
            TOKEN_CHANGE_MASK_PRE_COLUMN: token_change_mask_pre,
            TOKEN_CHANGE_MASK_POST_COLUMN: token_change_mask_post,
            HUNK_ID_PER_TOKEN_COLUMN: hunk_id_per_token,
            EDIT_OP_PER_TOKEN_COLUMN: edit_op_per_token,
        }
        row.update(cast(dict[str, list[int]], _build_token_chunk_layout(doc, tok_chunks, len(token_ids))))
        domain = _domain_kind_from_doc(doc)
        if domain is not None:
            _insert_domain_delimiters(row, domain=domain)
        changed_chunk_ids, changed_chunk_spans = _changed_chunk_metadata(
            cast(list[int], row[TOKEN_CHUNK_STARTS_COLUMN]),
            cast(list[int], row[TOKEN_CHUNK_ENDS_COLUMN]),
            (
                cast(list[int], row[TOKEN_CHANGE_MASK_PRE_COLUMN]),
                cast(list[int], row[TOKEN_CHANGE_MASK_POST_COLUMN]),
            ),
        )
        row[CHANGED_CHUNK_IDS_COLUMN] = changed_chunk_ids
        row[CHANGED_CHUNK_SPANS_COLUMN] = changed_chunk_spans
        out.append(row)

    return out
