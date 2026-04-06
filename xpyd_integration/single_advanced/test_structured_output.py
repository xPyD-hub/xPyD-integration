"""Tests for structured output & function calling with xpyd-sim server.

Migrated from xpyd-bench test_structured_output.py.
Only includes tests that exercise the sim server directly.
Validator tests (TestValidateJsonSchema, etc.) remain in bench.
"""

from __future__ import annotations

import json

import pytest
from xpyd_sim.common.tools import generate_dummy_from_schema as _generate_dummy_args
from xpyd_sim.common.tools import build_tool_calls as _build_tool_calls
from xpyd_sim.server import _generate_response_content as _build_json_response


# ---------------------------------------------------------------------------
# Dummy server helpers
# ---------------------------------------------------------------------------

class TestGenerateDummyArgs:
    def test_basic_types(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "active": {"type": "boolean"},
                "items": {"type": "array"},
            },
        }
        result = _generate_dummy_args(schema)
        assert isinstance(result["name"], str)
        assert isinstance(result["count"], int)
        assert isinstance(result["ratio"], float)
        assert isinstance(result["active"], bool)
        assert isinstance(result["items"], list)

    def test_enum(self):
        schema = {
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": ["red", "blue"]},
            },
        }
        result = _generate_dummy_args(schema)
        assert result["color"] == "red"

    def test_empty_schema(self):
        assert _generate_dummy_args({}) == {}
        result = _generate_dummy_args({"type": "string"})
        assert isinstance(result, (str, dict))


# ---------------------------------------------------------------------------
# Dummy server tool call generation
# ---------------------------------------------------------------------------

class TestDummyServerToolCalls:
    def test_build_tool_calls_auto(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        result = _build_tool_calls(tools, tool_choice="auto")
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        args = json.loads(result[0]["function"]["arguments"])
        assert "location" in args

    def test_build_tool_calls_specific(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "fn_a", "parameters": {"type": "object", "properties": {}}},
            },
            {
                "type": "function",
                "function": {"name": "fn_b", "parameters": {"type": "object", "properties": {}}},
            },
        ]
        choice = {"type": "function", "function": {"name": "fn_b"}}
        result = _build_tool_calls(tools, tool_choice=choice)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "fn_b"

    def test_build_tool_calls_parallel(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "fn_a", "parameters": {"type": "object", "properties": {}}},
            },
            {
                "type": "function",
                "function": {"name": "fn_b", "parameters": {"type": "object", "properties": {}}},
            },
        ]
        result = _build_tool_calls(tools, tool_choice="auto", parallel=True)
        assert len(result) == 2

    def test_build_json_response_json_object(self):
        resp = _build_json_response({"type": "json_object"}, 10)
        parsed = json.loads(resp)
        assert isinstance(parsed, dict)

    def test_build_json_response_json_schema(self):
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
        resp = _build_json_response(rf, 10)
        parsed = json.loads(resp)
        assert "city" in parsed


# ---------------------------------------------------------------------------
# Integration: dummy server HTTP
# ---------------------------------------------------------------------------

@pytest.fixture()
def dummy_app():
    from xpyd_sim.server import ServerConfig, create_app

    config = ServerConfig(prefill_delay_ms=0, decode_delay_per_token_ms=0)
    return create_app(config)


@pytest.mark.anyio
async def test_dummy_chat_with_tools(dummy_app):
    from httpx import ASGITransport, AsyncClient

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": tools,
        "tool_choice": "required",
        "max_tokens": 10,
    }
    async with AsyncClient(
        transport=ASGITransport(app=dummy_app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        msg = body["choices"][0]["message"]
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) > 0
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert "location" in args


@pytest.mark.anyio
async def test_dummy_chat_with_response_format_json(dummy_app):
    from httpx import ASGITransport, AsyncClient

    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Give me JSON"}],
        "response_format": {"type": "json_object"},
        "max_tokens": 10,
    }
    async with AsyncClient(
        transport=ASGITransport(app=dummy_app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        assert isinstance(parsed, dict)


@pytest.mark.anyio
async def test_dummy_chat_with_response_format_schema(dummy_app):
    from httpx import ASGITransport, AsyncClient

    rf = {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        },
    }
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Describe a person"}],
        "response_format": rf,
        "max_tokens": 10,
    }
    async with AsyncClient(
        transport=ASGITransport(app=dummy_app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        assert "name" in parsed
        assert "age" in parsed


@pytest.mark.anyio
async def test_dummy_chat_tool_choice_none(dummy_app):
    """tool_choice=none should produce regular text, not tool calls."""
    from httpx import ASGITransport, AsyncClient

    tools = [
        {
            "type": "function",
            "function": {
                "name": "fn",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "tool_choice": "none",
        "max_tokens": 5,
    }
    async with AsyncClient(
        transport=ASGITransport(app=dummy_app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        msg = body["choices"][0]["message"]
        assert not msg.get("tool_calls")
        assert msg["content"] is not None
