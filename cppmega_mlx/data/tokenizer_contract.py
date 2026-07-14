"""Dependency-free tokenizer special-token contract checks."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

REQUIRED_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "BOS": 2,
    "EOT": 3,
    "FIM_PREFIX": 4,
    "FIM_MIDDLE": 5,
    "FIM_SUFFIX": 6,
    "CODE_START": 7,
    "FIM_INSTRUCTION": 45,
    "SPACE": 46,
    "NL": 47,
}

TOOL_USE_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "BOS": 2,
    "EOS": 3,
    "CODE_START": 7,
    "CODE_END": 8,
    "THINK_START": 9,
    "THINK_END": 10,
    "QUERY_TOOL": 11,
    "TOOL_RESULT": 19,
}

# Frozen semantic tokens already present in tokenizer.json. Objective builders
# use these boundaries directly; no new or reassigned vocabulary IDs are allowed.
OBJECTIVE_BOUNDARY_TOKEN_IDS: dict[str, int] = {
    "FILE_SEP": 14,
    "DIFF_START": 15,
    "DIFF_END": 16,
    "COMMENT_START": 17,
    "COMMENT_END": 18,
    "SYMBOL_REF": 38,
    "TYPE_INFO": 39,
    "OVERLOAD_SET": 44,
}

_TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"
TOKENIZER_CONTRACT_PATH = _TOKENIZER_DIR / "tokenizer_contract_v1.json"


def _read_contract_bytes(path: Path) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        contract = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load tokenizer contract {path}: {exc}") from exc
    if not isinstance(contract, dict):
        raise ValueError(f"tokenizer contract {path} must be an object")
    return contract, raw


def _read_contract(path: Path) -> dict:
    contract, _raw = _read_contract_bytes(path)
    return contract


def _derive_domain_delimiter_token_ids(contract: Mapping[str, object]) -> dict[str, int]:
    assignments = contract.get("reserved_role_assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("reserved_role_assignments must be an object")
    starts = {
        str(role).removesuffix("_START")
        for role in assignments
        if isinstance(role, str) and role.endswith("_START")
    }
    ends = {
        str(role).removesuffix("_END")
        for role in assignments
        if isinstance(role, str) and role.endswith("_END")
    }
    if starts != ends:
        missing = sorted(
            [f"{base}_END" for base in starts - ends]
            + [f"{base}_START" for base in ends - starts]
        )
        raise ValueError(f"unpaired domain delimiter roles: {missing}")
    result: dict[str, int] = {}
    seen_ids: dict[int, str] = {}
    for base in sorted(starts):
        for edge in ("START", "END"):
            role = f"{base}_{edge}"
            raw_id = assignments[role]
            if not isinstance(raw_id, int) or isinstance(raw_id, bool):
                raise ValueError(f"domain delimiter {role} id must be int")
            existing = seen_ids.setdefault(raw_id, role)
            if existing != role:
                raise ValueError(
                    f"domain delimiter id {raw_id} maps to both {existing} and {role}"
                )
            result[role] = raw_id
    if not result:
        raise ValueError("tokenizer contract defines no domain delimiter pairs")
    return result


TOKENIZER_CONTRACT, _TOKENIZER_CONTRACT_BYTES = _read_contract_bytes(
    TOKENIZER_CONTRACT_PATH
)
TOKENIZER_CONTRACT_SHA256 = hashlib.sha256(_TOKENIZER_CONTRACT_BYTES).hexdigest()
TOKENIZER_CONTRACT_SHA256_METADATA_KEY = "cppmega.tokenizer_contract_sha256"
DOMAIN_DELIMITER_TOKEN_IDS = _derive_domain_delimiter_token_ids(TOKENIZER_CONTRACT)
DOMAIN_DELIMITER_CONTRACT_METADATA_KEY = "cppmega.domain_delimiter_contract_sha256"
DOMAIN_DELIMITER_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        DOMAIN_DELIMITER_TOKEN_IDS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

SpecialTokenMapping = Mapping[int, str] | Mapping[str, int]


def validate_required_special_token_ids(mapping: SpecialTokenMapping) -> None:
    """Validate the cppmega special-token ids without loading a tokenizer.

    The input may be either id->token or token->id.  Validation fails closed on
    missing entries, duplicate ids/tokens, wrong ids, or ambiguous key/value
    shapes.
    """

    token_to_id = _normalize_special_token_mapping(mapping)
    for token, expected_id in REQUIRED_SPECIAL_TOKEN_IDS.items():
        if token not in token_to_id:
            raise ValueError(f"missing required special token {token!r}")
        actual_id = token_to_id[token]
        if actual_id != expected_id:
            raise ValueError(
                f"special token {token!r} must use id {expected_id}, got {actual_id}"
            )

    seen_ids: dict[int, str] = {}
    for token, token_id in token_to_id.items():
        existing = seen_ids.setdefault(token_id, token)
        if existing != token:
            raise ValueError(
                f"special token id collision: id {token_id} maps to both "
                f"{existing!r} and {token!r}"
            )


def validate_checked_out_tokenizer_contract(
    repo_root: str | Path | None = None,
) -> dict[str, int]:
    """Validate the contract and tokenizer from this checkout, never a sibling."""

    tokenizer_dir = (
        Path(repo_root).resolve() / "cppmega_mlx" / "tokenizer"
        if repo_root is not None
        else _TOKENIZER_DIR
    )
    contract_path = tokenizer_dir / "tokenizer_contract_v1.json"
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    contract = _read_contract(contract_path)
    delimiter_ids = _derive_domain_delimiter_token_ids(contract)
    try:
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load tokenizer artifact {tokenizer_path}: {exc}") from exc
    added_tokens = {
        int(item["id"]): str(item["content"])
        for item in tokenizer.get("added_tokens", [])
        if isinstance(item, dict) and "id" in item and "content" in item
    }
    for role, token_id in delimiter_ids.items():
        expected = f"<RESERVED_{token_id}>"
        if added_tokens.get(token_id) != expected:
            raise ValueError(
                f"{tokenizer_path}: {role} id {token_id} maps to "
                f"{added_tokens.get(token_id)!r}, expected {expected!r}"
            )
    if tokenizer_dir == _TOKENIZER_DIR and delimiter_ids != DOMAIN_DELIMITER_TOKEN_IDS:
        raise ValueError("imported domain delimiter mapping differs from checked-out contract")
    return delimiter_ids


def _normalize_special_token_mapping(mapping: SpecialTokenMapping) -> dict[str, int]:
    if not mapping:
        raise ValueError("special token mapping must not be empty")

    keys_are_ids = all(_is_int_key(key) for key in mapping)
    values_are_tokens = all(isinstance(value, str) for value in mapping.values())
    keys_are_tokens = all(isinstance(key, str) for key in mapping)
    values_are_ids = all(_is_int_key(value) for value in mapping.values())

    if keys_are_ids and values_are_tokens:
        return _invert_id_to_token_mapping(mapping)
    if keys_are_tokens and values_are_ids:
        return {str(token): int(token_id) for token, token_id in mapping.items()}

    raise ValueError(
        "special token mapping must be consistently id->token or token->id"
    )


def _invert_id_to_token_mapping(mapping: SpecialTokenMapping) -> dict[str, int]:
    token_to_id: dict[str, int] = {}
    for raw_id, raw_token in mapping.items():
        token_id = int(raw_id)
        token = str(raw_token)
        existing = token_to_id.setdefault(token, token_id)
        if existing != token_id:
            raise ValueError(
                f"special token collision: token {token!r} maps to both "
                f"{existing} and {token_id}"
            )
    return token_to_id


def _is_int_key(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "DOMAIN_DELIMITER_TOKEN_IDS",
    "OBJECTIVE_BOUNDARY_TOKEN_IDS",
    "DOMAIN_DELIMITER_CONTRACT_METADATA_KEY",
    "DOMAIN_DELIMITER_CONTRACT_SHA256",
    "REQUIRED_SPECIAL_TOKEN_IDS",
    "SpecialTokenMapping",
    "TOKENIZER_CONTRACT",
    "TOKENIZER_CONTRACT_PATH",
    "TOKENIZER_CONTRACT_SHA256",
    "TOKENIZER_CONTRACT_SHA256_METADATA_KEY",
    "TOOL_USE_SPECIAL_TOKEN_IDS",
    "validate_checked_out_tokenizer_contract",
    "validate_required_special_token_ids",
]
