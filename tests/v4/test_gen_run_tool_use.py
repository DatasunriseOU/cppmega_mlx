"""V7-H43: gen.run accepts tools=[{name, description, schema}] and
emits a deterministic tool_call JSON block on the smoke path."""

from __future__ import annotations

import json

from cppmega_v4.jsonrpc.dispatcher import dispatch
from cppmega_v4.jsonrpc.gen_run_method import GenRunParams, gen_run


_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Look up the weather for a city.",
    "schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string"},
        },
        "required": ["city"],
    },
}


def test_v7_h43_tools_empty_default_no_tool_call_emitted():
    out = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=99_999, max_new_tokens=2,
    ))
    assert out.tool_call is None


def test_v7_h43_tool_call_block_contains_tool_name_and_schema_args():
    out = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=99_999, max_new_tokens=2,
        seed=11, tools=[_WEATHER_TOOL],
    ))
    tc = out.tool_call
    assert tc is not None
    assert tc["tool"] == "get_weather"
    # Schema's property keys must appear in arguments.
    assert set(tc["arguments"].keys()) == {"city", "units"}
    # raw_json is parseable + matches the structure.
    parsed = json.loads(tc["raw_json"])
    assert parsed["tool"] == "get_weather"
    assert "city" in parsed["arguments"]


def test_v7_h43_tool_call_deterministic_for_same_seed():
    a = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=99_999, max_new_tokens=2,
        seed=42, tools=[_WEATHER_TOOL],
    ))
    b = gen_run(GenRunParams(
        prompt_tokens=[1], eos_token_id=99_999, max_new_tokens=2,
        seed=42, tools=[_WEATHER_TOOL],
    ))
    assert a.tool_call == b.tool_call


def test_v7_h43_dispatcher_route_returns_tool_call():
    resp = dispatch({
        "jsonrpc": "2.0", "id": "T1", "method": "gen.run",
        "params": {
            "prompt_tokens": [1], "eos_token_id": 99_999,
            "max_new_tokens": 2,
            "tools": [_WEATHER_TOOL],
        },
    })
    assert resp.error is None, resp.error
    assert resp.result["tool_call"] is not None
    assert resp.result["tool_call"]["tool"] == "get_weather"
