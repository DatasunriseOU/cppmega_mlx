from __future__ import annotations

import pytest

from cppmega_mlx.data.tokenizer_contract import (
    REQUIRED_SPECIAL_TOKEN_IDS,
    validate_required_special_token_ids,
)


def test_valid_id_to_token_mapping_passes() -> None:
    id_to_token = {
        token_id: token for token, token_id in REQUIRED_SPECIAL_TOKEN_IDS.items()
    }

    validate_required_special_token_ids(id_to_token)


def test_valid_token_to_id_mapping_passes() -> None:
    validate_required_special_token_ids(REQUIRED_SPECIAL_TOKEN_IDS)


def test_missing_required_special_token_fails() -> None:
    token_to_id = dict(REQUIRED_SPECIAL_TOKEN_IDS)
    token_to_id.pop("FIM_MIDDLE")

    with pytest.raises(ValueError, match="missing required special token 'FIM_MIDDLE'"):
        validate_required_special_token_ids(token_to_id)


def test_missing_required_special_id_fails() -> None:
    id_to_token = {
        token_id: token for token, token_id in REQUIRED_SPECIAL_TOKEN_IDS.items()
    }
    id_to_token.pop(6)

    with pytest.raises(ValueError, match="missing required special token 'FIM_SUFFIX'"):
        validate_required_special_token_ids(id_to_token)


def test_special_token_collision_fails_closed() -> None:
    id_to_token = {
        token_id: token for token, token_id in REQUIRED_SPECIAL_TOKEN_IDS.items()
    }
    id_to_token[8] = "FIM_PREFIX"

    with pytest.raises(ValueError, match="token 'FIM_PREFIX' maps to both 4 and 8"):
        validate_required_special_token_ids(id_to_token)


def test_special_id_collision_fails_closed() -> None:
    token_to_id = dict(REQUIRED_SPECIAL_TOKEN_IDS)
    token_to_id["EXTRA_ALIAS"] = 4

    with pytest.raises(ValueError, match="id 4 maps to both"):
        validate_required_special_token_ids(token_to_id)


def test_wrong_special_token_id_fails() -> None:
    token_to_id = dict(REQUIRED_SPECIAL_TOKEN_IDS)
    token_to_id["EOT"] = 9

    with pytest.raises(ValueError, match="special token 'EOT' must use id 3, got 9"):
        validate_required_special_token_ids(token_to_id)
