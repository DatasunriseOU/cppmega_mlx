from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

import mlx.core as mx
import numpy as np
import pytest

import cppmega_mlx.inference as inference
from cppmega_mlx.inference import (
    AdapterCapabilities,
    InferenceSideChannelBuilder,
    TokenMetadata,
)
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM


class TinyTokenizer:
    name_or_path = "tiny-tokenizer"

    def encode(self, text: str) -> list[int]:
        return [max(1, (ord(char) % 31) + 1) for char in text]


class FakeCppAdapter:
    language = "cpp"
    version = "fake-clang-v1"

    def probe(self, context: Mapping[str, Any]) -> AdapterCapabilities:
        return AdapterCapabilities(
            language=self.language,
            version=self.version,
            families=("structure", "syntax"),
        )

    def extract(self, source_or_project: str, options: Mapping[str, Any]) -> str:
        assert options["language"] == "cpp"
        return source_or_project

    def map_to_tokens(
        self,
        metadata: str,
        tokens: Sequence[int],
        tokenizer: Any,
    ) -> TokenMetadata:
        del metadata, tokenizer
        n = len(tokens)
        return TokenMetadata(
            structure_ids=[1] * n,
            dep_levels=list(range(n)),
            ast_depth_ids=[2] * n,
            sibling_index_ids=[0] * n,
            node_type_ids=[3] * n,
            provenance={"adapter": "fake-clang-v1"},
        )


class FailingAdapter(FakeCppAdapter):
    version = "failing-v1"

    def extract(self, source_or_project: str, options: Mapping[str, Any]) -> str:
        del source_or_project, options
        raise ValueError("no compile database")


def _tiny_model() -> HybridTinyLM:
    return HybridTinyLM(
        HybridTinyConfig(
            vocab_size=64,
            hidden_size=16,
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            num_attention_heads=4,
            max_seq_length=16,
            structure_components="all",
            structure_num_categories=16,
        )
    )


def test_inference_side_channel_builder_text_only_prompt() -> None:
    builder = InferenceSideChannelBuilder(TinyTokenizer())

    result = builder.build("int x;")

    assert result.prompt_ids.shape == (1, 6)
    assert result.model_kwargs == {}
    assert result.side_channels == {}
    assert result.cache_components.content_sha256 == sha256(b"int x;").hexdigest()
    assert result.cache_components.tokenizer_id == "tiny-tokenizer"


def test_inference_side_channel_builder_platform_context_smoke() -> None:
    builder = InferenceSideChannelBuilder(TinyTokenizer())

    result = builder.build(
        "int x;",
        platform_context={
            "os": "macos",
            "arch": "arm64",
            "compiler": "clang",
            "accelerator": "metal",
        },
    )
    logits = _tiny_model()(result.prompt_ids, **result.model_kwargs)
    mx.eval(logits)

    platform_ids = result.model_kwargs["platform_ids"]
    assert platform_ids.shape[0] == 1
    assert int(np.count_nonzero(np.array(platform_ids))) > 0
    assert "os=macos" in result.rendered_platform_context
    assert result.side_channels["platform"]["platform_ids"] is platform_ids
    assert logits.shape[:2] == result.prompt_ids.shape


def test_inference_side_channel_builder_adapter_metadata_smoke() -> None:
    builder = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=FakeCppAdapter(),
    )

    result = builder.build("int x;", language="cpp")
    logits = _tiny_model()(result.prompt_ids, **result.model_kwargs)
    mx.eval(logits)

    assert {
        "structure_ids",
        "dep_levels",
        "ast_depth_ids",
        "sibling_index_ids",
        "node_type_ids",
    } <= set(result.model_kwargs)
    assert result.model_kwargs["structure_ids"].shape == result.prompt_ids.shape
    assert result.side_channels["syntax"]["ast_depth_ids"].shape == result.prompt_ids.shape
    assert result.provenance["adapter"] == "fake-clang-v1"
    assert result.cache_components.adapter_language == "cpp"
    assert result.cache_components.adapter_version == "fake-clang-v1"
    assert logits.shape[:2] == result.prompt_ids.shape


def test_inference_side_channel_builder_fallback_policies_are_explicit() -> None:
    platform_context = {"os": "linux", "compiler": "gcc"}
    dropped = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=FailingAdapter(),
        fail_policy="drop_family",
    ).build("int x;", platform_context=platform_context, language="cpp")
    text_only = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=FailingAdapter(),
        fail_policy="text_only",
    ).build("int x;", platform_context=platform_context, language="cpp")

    assert dropped.provenance["adapter"].startswith("dropped:adapter_error")
    assert "platform_ids" in dropped.model_kwargs
    assert text_only.model_kwargs == {}
    assert text_only.side_channels == {}
    assert text_only.provenance == {"fallback": "text_only"}
    with pytest.raises(RuntimeError, match="adapter_error"):
        InferenceSideChannelBuilder(
            TinyTokenizer(),
            adapter=FailingAdapter(),
            fail_policy="error",
        ).build("int x;", platform_context=platform_context, language="cpp")


def test_inference_side_channel_builder_cache_key_uses_required_components() -> None:
    builder = InferenceSideChannelBuilder(
        TinyTokenizer(),
        adapter=FakeCppAdapter(),
    )

    macos = builder.build("int x;", platform_context={"os": "macos"}, language="cpp")
    linux = builder.build("int x;", platform_context={"os": "linux"}, language="cpp")

    assert macos.cache_key != linux.cache_key
    assert macos.cache_components.content_sha256 == linux.cache_components.content_sha256
    assert macos.cache_components.tokenizer_id == "tiny-tokenizer"
    assert macos.cache_components.adapter_language == "cpp"
    assert macos.cache_components.adapter_version == "fake-clang-v1"
    assert "os=macos" in macos.cache_components.platform_context
    assert "os=linux" in linux.cache_components.platform_context


def test_inference_root_exports_side_channel_builder() -> None:
    assert inference.InferenceSideChannelBuilder is InferenceSideChannelBuilder
    assert "InferenceSideChannelBuilder" in inference.__all__
    assert "InferenceFailPolicy" in inference.__all__
