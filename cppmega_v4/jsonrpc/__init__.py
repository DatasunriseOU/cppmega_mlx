"""JSON-RPC 2.0 contract + Python server for the Visual Builder GUI.

See ``VisualBuilderPlan.md`` §3.3 and §5 for the design.

Stage F-A surface (this commit):
  - schema: Pydantic v2 models for all request/response payloads
  - cache: LRU(50) keyed on canonical sha256(spec without layout)
  - methods: pure-Python handlers (verify / suggest_sharding /
    suggest_adapters / build_preset_specs / probe.run)
  - dispatcher: transport-agnostic JSON-RPC 2.0 router
  - server: FastAPI app exposing /rpc + /ws + heartbeat
"""

from __future__ import annotations

from cppmega_v4.jsonrpc.cache import LRUCache, canonical_json, canonical_sha256
from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.schema import (
    EVENT_TAXONOMY,
    METHOD_REGISTRY,
    SCHEMA_VERSION,
    BuildPresetSpecsParams,
    BuildPresetSpecsResult,
    DataMaterializationSpecPayload,
    ErrorCode,
    FamilySpecPayload,
    InferenceEnrichmentSpecPayload,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    PipelineAbortParams,
    PipelineAbortResult,
    ProbeRunParams,
    ProbeRunResult,
    SuggestAdaptersParams,
    SuggestAdaptersResult,
    SuggestShardingParams,
    SuggestShardingResult,
    SideChannelSpecPayload,
    VerifyParams,
    VerifyResult,
)
from cppmega_v4.jsonrpc.server import create_app, serve

__all__ = [
    "BuildPresetSpecsParams",
    "BuildPresetSpecsResult",
    "DataMaterializationSpecPayload",
    "EVENT_TAXONOMY",
    "ErrorCode",
    "FamilySpecPayload",
    "InferenceEnrichmentSpecPayload",
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "LRUCache",
    "METHOD_REGISTRY",
    "PipelineAbortParams",
    "PipelineAbortResult",
    "ProbeRunParams",
    "ProbeRunResult",
    "SCHEMA_VERSION",
    "SuggestAdaptersParams",
    "SuggestAdaptersResult",
    "SuggestShardingParams",
    "SuggestShardingResult",
    "SideChannelSpecPayload",
    "VerifyParams",
    "VerifyResult",
    "canonical_json",
    "canonical_sha256",
    "create_app",
    "dispatch",
    "serve",
]
