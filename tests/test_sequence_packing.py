from __future__ import annotations

import numpy as np
import pytest

from cppmega_mlx.data.packing import (
    cumulative_doc_ids_from_eos,
    document_boundary_mask,
    pack_documents_with_eos,
)


def test_pack_documents_with_eos_is_deterministic_and_fixed_length() -> None:
    left = pack_documents_with_eos(
        [[10, 11], np.array([20, 21, 22], dtype=np.int64), [30]],
        seq_len=8,
        eos_token_id=2,
    )
    right = pack_documents_with_eos(
        [[10, 11], np.array([20, 21, 22], dtype=np.int64), [30]],
        seq_len=8,
        eos_token_id=2,
    )

    expected = np.array([[10, 11, 2, 20, 21, 22, 2, 0], [30, 2, 0, 0, 0, 0, 0, 0]])
    expected_mask = np.array(
        [
            [True, True, True, True, True, True, True, False],
            [True, True, False, False, False, False, False, False],
        ]
    )

    np.testing.assert_array_equal(left.tokens, expected)
    np.testing.assert_array_equal(left.token_mask, expected_mask)
    np.testing.assert_array_equal(right.tokens, left.tokens)
    np.testing.assert_array_equal(right.token_mask, left.token_mask)
    assert left.tokens.dtype == np.int32
    assert left.token_mask.dtype == np.bool_


def test_pack_documents_inserts_eos_without_doubling_existing_eos() -> None:
    packed = pack_documents_with_eos(
        [[1, 2], [3, 4]],
        seq_len=6,
        eos_token_id=2,
        pad_token_id=99,
    )

    np.testing.assert_array_equal(
        packed.tokens,
        np.array([[1, 2, 3, 4, 2, 99]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        packed.token_mask,
        np.array([[True, True, True, True, True, False]]),
    )


def test_cumulative_doc_ids_drive_boundary_mask() -> None:
    tokens = np.array([[10, 11, 2, 20, 2, 30, 31, 2]], dtype=np.int32)
    doc_ids = cumulative_doc_ids_from_eos(tokens, eos_token_id=2)
    boundary_mask = document_boundary_mask(doc_ids)

    np.testing.assert_array_equal(
        doc_ids,
        np.array([[0, 0, 0, 1, 1, 2, 2, 2]], dtype=np.int32),
    )
    assert bool(boundary_mask[0, 0, 2])
    assert not bool(boundary_mask[0, 0, 3])
    assert bool(boundary_mask[0, 3, 4])
    assert not bool(boundary_mask[0, 4, 5])
    assert bool(boundary_mask[0, 5, 7])


def test_document_boundary_mask_excludes_padding_and_can_be_causal() -> None:
    packed = pack_documents_with_eos([[5], [6]], seq_len=5, eos_token_id=2)

    np.testing.assert_array_equal(
        packed.doc_ids,
        np.array([[0, 0, 1, 1, -1]], dtype=np.int32),
    )
    assert not bool(packed.boundary_mask[0, 0, 2])
    assert not bool(packed.boundary_mask[0, 0, 4])
    assert not bool(packed.boundary_mask[0, 4, 4])

    causal = document_boundary_mask(
        packed.doc_ids,
        token_mask=packed.token_mask,
        causal=True,
    )
    assert bool(causal[0, 1, 0])
    assert not bool(causal[0, 0, 1])


def test_oversized_samples_are_refused_by_default() -> None:
    with pytest.raises(ValueError, match="sample 0 length 5 exceeds seq_len 4"):
        pack_documents_with_eos([[1, 2, 3, 4]], seq_len=4, eos_token_id=2)


def test_oversized_samples_can_be_truncated_with_final_eos() -> None:
    packed = pack_documents_with_eos(
        [[1, 2, 3, 4]],
        seq_len=4,
        eos_token_id=99,
        oversized="truncate",
    )

    np.testing.assert_array_equal(
        packed.tokens,
        np.array([[1, 2, 3, 99]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        packed.doc_ids,
        np.array([[0, 0, 0, 0]], dtype=np.int32),
    )


def test_pack_documents_validates_shapes_and_token_ids() -> None:
    with pytest.raises(ValueError, match="seq_len must be at least 2"):
        pack_documents_with_eos([[1]], seq_len=1, eos_token_id=2)

    with pytest.raises(ValueError, match="sample 0 must be a 1D token sequence"):
        pack_documents_with_eos([np.array([[1, 2]])], seq_len=4, eos_token_id=2)

    with pytest.raises(ValueError, match="sample 0 token IDs must be non-negative"):
        pack_documents_with_eos([[-1]], seq_len=4, eos_token_id=2)

    with pytest.raises(ValueError, match="tokens must be 2D"):
        cumulative_doc_ids_from_eos(np.array([1, 2]), eos_token_id=2)

    with pytest.raises(ValueError, match="token_mask shape must match tokens"):
        cumulative_doc_ids_from_eos(
            np.array([[1, 2]], dtype=np.int32),
            eos_token_id=2,
            token_mask=np.array([[True, True, False]]),
        )
