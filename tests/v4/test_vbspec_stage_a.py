"""VBSpec Stage A tests — shape_contract.ShapeExpr + BrickShapeContract."""

from __future__ import annotations

import pytest

from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS
from cppmega_v4.spec import (
    BrickShapeContract,
    ResolveError,
    ShapeExpr,
    contract_for,
    register_contract,
    registered_kinds,
)


# ---------------------------------------------------------------------------
# ShapeExpr unit tests
# ---------------------------------------------------------------------------


def test_shape_expr_rejects_non_tuple():
    with pytest.raises(TypeError, match="must be tuple"):
        ShapeExpr(["B", "S", "H"])  # type: ignore[arg-type]


def test_shape_expr_rejects_empty_or_blank_dim():
    with pytest.raises(ValueError, match="non-empty str"):
        ShapeExpr(("B", "", "H"))
    with pytest.raises(ValueError, match="non-empty str"):
        ShapeExpr(("B", "   ", "H"))


def test_shape_expr_resolve_simple_named_dims():
    e = ShapeExpr(("B", "S", "H"))
    assert e.resolve({"B": 1, "S": 16, "H": 64}) == (1, 16, 64)


def test_shape_expr_resolve_arithmetic_combination():
    e = ShapeExpr(("B", "S", "nh * head_dim"))
    assert e.resolve({"B": 2, "S": 8, "nh": 4, "head_dim": 16}) == (2, 8, 64)


def test_shape_expr_resolve_complex_mla_expression():
    e = ShapeExpr(("B", "S", "nh * (qk_nope_head_dim + qk_rope_head_dim + v_head_dim)"))
    env = {"B": 1, "S": 4, "nh": 8, "qk_nope_head_dim": 16,
           "qk_rope_head_dim": 8, "v_head_dim": 16}
    assert e.resolve(env) == (1, 4, 8 * (16 + 8 + 16))


def test_shape_expr_resolve_missing_dim_raises():
    e = ShapeExpr(("B", "S", "H"))
    with pytest.raises(ResolveError, match="missing dim_env entries"):
        e.resolve({"B": 1, "S": 16})


def test_shape_expr_resolve_non_positive_raises():
    e = ShapeExpr(("B", "S", "H"))
    with pytest.raises(ResolveError, match="non-positive"):
        e.resolve({"B": 0, "S": 16, "H": 64})


def test_shape_expr_free_names_excludes_substrings():
    """Single-letter dim names must not match inside longer identifiers
    like ``head_dim`` or ``num_experts``."""
    e = ShapeExpr(("head_dim",))
    names = e.free_names()
    assert "head_dim" in names
    assert "H" not in names
    assert "S" not in names


def test_shape_expr_free_names_picks_compound_expression():
    e = ShapeExpr(("nh * (qk_nope_head_dim + qk_rope_head_dim)",))
    names = e.free_names()
    assert names == frozenset({"nh", "qk_nope_head_dim", "qk_rope_head_dim"})


def test_shape_expr_rank_property():
    assert ShapeExpr(("B",)).rank == 1
    assert ShapeExpr(("B", "S", "H")).rank == 3


# ---------------------------------------------------------------------------
# BrickShapeContract unit tests
# ---------------------------------------------------------------------------


def _bsh() -> ShapeExpr:
    return ShapeExpr(("B", "S", "H"))


def _scalar(expr: str) -> ShapeExpr:
    return ShapeExpr((expr,))


def test_contract_rejects_empty_inputs_outputs():
    with pytest.raises(ValueError, match="inputs must not be empty"):
        BrickShapeContract(
            inputs={}, outputs={"y": _bsh()},
            params_elems=_scalar("0"), activations_elems=_scalar("0"),
            kv_cache_elems=_scalar("0"),
        )
    with pytest.raises(ValueError, match="outputs must not be empty"):
        BrickShapeContract(
            inputs={"x": _bsh()}, outputs={},
            params_elems=_scalar("0"), activations_elems=_scalar("0"),
            kv_cache_elems=_scalar("0"),
        )


def test_contract_rejects_non_shape_expr_field():
    with pytest.raises(TypeError, match="must be ShapeExpr"):
        BrickShapeContract(
            inputs={"x": (3, 4, 5)},   # type: ignore[dict-item]
            outputs={"y": _bsh()},
            params_elems=_scalar("0"), activations_elems=_scalar("0"),
            kv_cache_elems=_scalar("0"),
        )


def test_contract_byte_fields_must_be_scalar():
    with pytest.raises(ValueError, match="rank-1"):
        BrickShapeContract(
            inputs={"x": _bsh()}, outputs={"y": _bsh()},
            params_elems=_bsh(),  # rank 3 — not allowed
            activations_elems=_scalar("0"), kv_cache_elems=_scalar("0"),
        )


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------


def test_contract_for_unknown_kind_raises():
    with pytest.raises(KeyError, match="no shape contract"):
        contract_for("totally_made_up_kind_xyz")


def test_register_contract_validates_type():
    with pytest.raises(TypeError, match="BrickShapeContract"):
        register_contract("temp_x", "not a contract")  # type: ignore[arg-type]


def test_register_then_recover_contract():
    c = BrickShapeContract(
        inputs={"x": _bsh()}, outputs={"y": _bsh()},
        params_elems=_scalar("H"), activations_elems=_scalar("B * S * H"),
        kv_cache_elems=_scalar("0"),
        description="temp test contract",
    )
    register_contract("temp_test_kind", c)
    assert contract_for("temp_test_kind") is c
    assert "temp_test_kind" in registered_kinds()


@pytest.mark.parametrize("kind", sorted(BLOCK_BUILDERS.keys()))
def test_every_block_builder_kind_has_contract(kind):
    """Every BLOCK_BUILDERS entry must have a shape contract registered —
    this is the CI tripwire from VisualBuilderSpec.md §7 risk #3."""
    c = contract_for(kind)
    assert isinstance(c, BrickShapeContract)
    assert "x" in c.inputs
    assert "y" in c.outputs


# ---------------------------------------------------------------------------
# System: real-world resolves with sensible dim_envs
# ---------------------------------------------------------------------------


_QWEN3_NEXT_ENV = {
    "B": 1, "S": 4096, "H": 4096,
    "nh": 32, "nkv": 4, "head_dim": 128,
    "num_experts": 8, "top_k": 2,
}


_LING26_ENV = {
    "B": 1, "S": 8192, "H": 4096,
    "nh": 32, "nkv": 4, "head_dim": 128,
    "q_lora_rank": 1536, "kv_lora_rank": 512,
    "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
    "num_experts": 16, "top_k": 2,
}


def test_system_gdn_contract_resolves_under_qwen_env():
    c = contract_for("gdn")
    assert c.inputs["x"].resolve(_QWEN3_NEXT_ENV) == (1, 4096, 4096)
    assert c.outputs["y"].resolve(_QWEN3_NEXT_ENV) == (1, 4096, 4096)
    p = c.params_elems.resolve(_QWEN3_NEXT_ENV)[0]
    a = c.activations_elems.resolve(_QWEN3_NEXT_ENV)[0]
    # Sanity ranges (not exact — contract is a model, not the truth).
    assert p > 0
    assert a > 0


def test_system_gated_attention_kv_cache_grows_with_seq():
    c = contract_for("gated_attention")
    env_small = {**_QWEN3_NEXT_ENV, "S": 1024}
    env_large = {**_QWEN3_NEXT_ENV, "S": 8192}
    kv_small = c.kv_cache_elems.resolve(env_small)[0]
    kv_large = c.kv_cache_elems.resolve(env_large)[0]
    assert kv_large == 8 * kv_small


def test_system_mla_contract_resolves_under_ling_env():
    c = contract_for("bailing_mla")
    p = c.params_elems.resolve(_LING26_ENV)[0]
    a = c.activations_elems.resolve(_LING26_ENV)[0]
    kv = c.kv_cache_elems.resolve(_LING26_ENV)[0]
    assert p > 0 and a > 0 and kv > 0
    # MLA's kv-cache is the latent-rank one, NOT 2*S*nkv*head_dim;
    # for our env that's 1*8192*(512+64) ≈ 4.7M elements.
    assert kv == 1 * 8192 * (512 + 64)


def test_system_moe_activation_scales_with_top_k():
    c = contract_for("moe")
    env_1 = {**_QWEN3_NEXT_ENV, "top_k": 1}
    env_2 = {**_QWEN3_NEXT_ENV, "top_k": 2}
    a1 = c.activations_elems.resolve(env_1)[0]
    a2 = c.activations_elems.resolve(env_2)[0]
    assert a2 == 2 * a1


def test_system_opaque_bricks_are_marked():
    """Bricks with data-dependent shape carry opaque_shape=True so the
    resolver (Stage B) knows to skip strict byte-accounting for them."""
    for kind in ("nsa", "csa_hca", "dsv4_attention"):
        assert contract_for(kind).opaque_shape is True
    for kind in ("gdn", "mlp", "gated_attention", "mla"):
        assert contract_for(kind).opaque_shape is False


def test_system_linear_attn_carries_doc_ids_need():
    for kind in ("gdn", "kda", "bailing_linear"):
        assert "doc_ids" in contract_for(kind).needs


def test_system_engram_carries_token_ids_need():
    assert "token_ids" in contract_for("engram").needs
