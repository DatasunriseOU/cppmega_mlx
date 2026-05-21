"""E7-6 tests: norm parameter validation."""

from __future__ import annotations

from cppmega_v4.buildspec.norm_validation import (
    VALID_NORM_KINDS,
    validate_norm_params,
    validate_parallel_block_norms,
)


def test_norm_kinds_lock_three_options():
    assert set(VALID_NORM_KINDS) == {"rmsnorm", "layernorm", "none"}


def test_default_rmsnorm_pre_none_post_no_issues():
    assert validate_norm_params("attn_0") == []


def test_both_none_is_error():
    diags = validate_norm_params("attn_0", pre_norm="none", post_norm="none")
    assert len(diags) == 1
    assert diags[0].severity == "error"
    assert "both be 'none'" in diags[0].message


def test_unknown_pre_norm_is_error():
    diags = validate_norm_params("attn_0", pre_norm="custom_norm",
                                  post_norm="none")
    assert any(d.severity == "error" and "pre_norm" in d.message for d in diags)


def test_unknown_post_norm_is_error():
    diags = validate_norm_params("attn_0", pre_norm="rmsnorm",
                                  post_norm="batchnorm")
    assert any(d.severity == "error" and "post_norm" in d.message for d in diags)


def test_layernorm_pre_rmsnorm_post_warns():
    diags = validate_norm_params("attn_0", pre_norm="layernorm",
                                  post_norm="rmsnorm")
    warnings = [d for d in diags if d.severity == "warning"]
    assert len(warnings) == 1
    assert "unusual" in warnings[0].message


def test_low_eps_warns():
    diags = validate_norm_params("attn_0", pre_norm="rmsnorm",
                                  post_norm="none", eps=1e-12)
    warnings = [d for d in diags if d.severity == "warning"]
    assert any("NaN risk" in w.message for w in warnings)


def test_parallel_block_branch_with_none_pre_is_error():
    diags = validate_parallel_block_norms([
        ("attn", "rmsnorm", "none"),
        ("mlp",  "none",     "rmsnorm"),
    ])
    assert len(diags) == 1
    assert diags[0].severity == "error"
    assert "parallel-block" in diags[0].message


def test_parallel_block_all_pre_set_passes():
    diags = validate_parallel_block_norms([
        ("attn", "rmsnorm", "none"),
        ("mlp",  "rmsnorm", "none"),
    ])
    assert diags == []


def test_all_norms_have_catalog_entry():
    """Every NormKind value must have an explain catalog entry."""
    from cppmega_v4.explain import get_entry
    for kind in VALID_NORM_KINDS:
        assert get_entry("norm", kind) is not None, \
            f"missing catalog entry for norm/{kind}"
