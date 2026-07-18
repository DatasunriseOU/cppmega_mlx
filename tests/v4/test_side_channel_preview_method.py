"""JSON-RPC preview for inference side-channel tensors."""

from __future__ import annotations

from pathlib import Path

from cppmega_v4.jsonrpc import dispatch


TOKENIZER = str(Path("cppmega_mlx/tokenizer/tokenizer.json"))


def test_side_channel_preview_dispatch_uses_real_tokenizer_and_adapter() -> None:
    resp = dispatch(
        {
            "jsonrpc": "2.0",
            "id": "preview",
            "method": "side_channels.preview",
            "params": {
                "tokenizer_source": TOKENIZER,
                "text": "x = 1\n",
                "language": "python",
                "adapter": "python",
                "platform_context": {"os": "linux", "compiler": "gcc"},
                "side_channels": {
                    "mode": "auto",
                    "inference": {
                        "source": "parse_if_possible",
                        "fail_policy": "drop_family",
                        "timeout_ms": 500,
                        "cache_enabled": True,
                    },
                },
            },
        }
    )

    assert resp.error is None
    assert resp.result is not None
    result = resp.result
    assert result["prompt_ids"]["shape"] == [1, result["token_count"]]
    assert result["token_count"] != len("x = 1\n")
    assert result["model_kwargs"]["platform_ids"]["shape"] == [1, 20]
    assert result["model_kwargs"]["structure_ids"]["shape"] == [1, result["token_count"]]
    assert result["side_channels"]["syntax"]["ast_depth_ids"]["shape"] == [
        1,
        result["token_count"],
    ]
    assert result["provenance"]["adapter"] == "python:python-ast-v1"
    assert "os=linux" in result["rendered_platform_context"]


def test_side_channel_preview_source_none_is_text_only() -> None:
    resp = dispatch(
        {
            "jsonrpc": "2.0",
            "id": "preview-none",
            "method": "side_channels.preview",
            "params": {
                "tokenizer_source": TOKENIZER,
                "text": "int x;",
                "platform_context": {"os": "linux"},
                "side_channels": {
                    "mode": "auto",
                    "inference": {
                        "source": "none",
                        "fail_policy": "drop_family",
                        "timeout_ms": 500,
                        "cache_enabled": True,
                    },
                },
            },
        }
    )

    assert resp.error is None
    assert resp.result is not None
    assert resp.result["model_kwargs"] == {}
    assert resp.result["side_channels"] == {}
    assert resp.result["provenance"] == {
        "fallback": "text_only",
        "inference_fail_policy": "text_only",
        "inference_enrichment_status": "explicit_text_only",
        "inference_degraded": "true",
    }
