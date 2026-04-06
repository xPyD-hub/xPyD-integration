"""Tests for M56: Structured Output & Function Calling Benchmarking."""

from __future__ import annotations

import json

import pytest

from xpyd_bench.bench.structured_output import (
    StructuredOutputResult,
    StructuredOutputSummary,
    _validate_json_schema,
    aggregate_structured_output,
    validate_json_response,
    validate_tool_calls,
)
from xpyd_sim.server import _generate_dummy_args

# ---------------------------------------------------------------------------
# JSON schema validation
# ---------------------------------------------------------------------------

class TestValidateJsonSchema:
    def test_valid_object(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        errors = _validate_json_schema({"name": "Alice", "age": 30}, schema)
        assert errors == []

    def test_missing_required(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        errors = _validate_json_schema({}, schema)
        assert any("name" in e for e in errors)

    def test_wrong_type(self):
        schema = {"type": "string"}
        errors = _validate_json_schema(42, schema)
        assert len(errors) == 1

    def test_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }
            },
        }
        errors = _validate_json_schema({"address": {}}, schema)
        assert any("city" in e for e in errors)

    def test_array_validation(self):
        schema = {"type": "array", "items": {"type": "integer"}}
        assert _validate_json_schema([1, 2, 3], schema) == []
        errors = _validate_json_schema([1, "two", 3], schema)
        assert len(errors) == 1

    def test_boolean(self):
        assert _validate_json_schema(True, {"type": "boolean"}) == []
        assert len(_validate_json_schema("yes", {"type": "boolean"})) == 1

    def test_number(self):
        assert _validate_json_schema(3.14, {"type": "number"}) == []
        assert len(_validate_json_schema("3.14", {"type": "number"})) == 1


# ---------------------------------------------------------------------------
# Tool call validation
# ---------------------------------------------------------------------------

class TestValidateToolCalls:
    def _make_response(self, tool_calls=None, content=None):
        msg = {"role": "assistant", "content": content}
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        return {
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}]
        }

    def _make_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

    def test_valid_tool_call(self):
        tc = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "NYC", "unit": "celsius"}),
                },
            }
        ]
        body = self._make_response(tool_calls=tc)
        result = validate_tool_calls(body, self._make_tools())
        assert result.success
        assert result.tool_calls_found == 1
        assert result.tool_call_results[0].function_name == "get_weather"

    def test_missing_required_arg(self):
        tc = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"unit": "celsius"}),
                },
            }
        ]
        body = self._make_response(tool_calls=tc)
        result = validate_tool_calls(body, self._make_tools())
        assert not result.success
        assert not result.tool_call_results[0].success

    def test_invalid_json_arguments(self):
        tc = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "not json"},
            }
        ]
        body = self._make_response(tool_calls=tc)
        result = validate_tool_calls(body, self._make_tools())
        assert not result.success

    def test_no_tool_calls_when_expected(self):
        body = self._make_response(tool_calls=[], content="I can help with that")
        result = validate_tool_calls(body, self._make_tools())
        assert not result.success
        assert result.tool_calls_found == 0

    def test_no_tools_expected(self):
        body = self._make_response(content="Hello")
        result = validate_tool_calls(body, tools=None)
        assert not result.tool_calls_expected
        assert result.success

    def test_missing_function_name(self):
        tc = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "", "arguments": "{}"},
            }
        ]
        body = self._make_response(tool_calls=tc)
        result = validate_tool_calls(body, self._make_tools())
        assert not result.tool_call_results[0].success


# ---------------------------------------------------------------------------
# JSON response format validation
# ---------------------------------------------------------------------------

class TestValidateJsonResponse:
    def test_json_object_valid(self):
        result = validate_json_response(
            '{"key": "value"}',
            {"type": "json_object"},
        )
        assert result.json_schema_valid is True

    def test_json_object_not_object(self):
        result = validate_json_response(
            '[1, 2, 3]',
            {"type": "json_object"},
        )
        assert result.json_schema_valid is False

    def test_json_object_invalid_json(self):
        result = validate_json_response(
            'not json',
            {"type": "json_object"},
        )
        assert result.json_schema_valid is False

    def test_json_schema_valid(self):
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }
        result = validate_json_response('{"name": "Alice"}', rf)
        assert result.json_schema_valid is True

    def test_json_schema_invalid(self):
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }
        result = validate_json_response('{"age": 30}', rf)
        assert result.json_schema_valid is False

    def test_empty_response(self):
        result = validate_json_response("", {"type": "json_object"})
        assert result.json_schema_valid is False

    def test_no_format(self):
        result = validate_json_response("anything", None)
        assert result.json_schema_valid is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregateStructuredOutput:
    def test_all_success(self):
        results = [
            StructuredOutputResult(
                tool_calls_expected=True,
                tool_calls_found=1,
                tool_call_results=[],
            ),
            StructuredOutputResult(
                tool_calls_expected=True,
                tool_calls_found=1,
                tool_call_results=[],
            ),
        ]
        summary = aggregate_structured_output(results)
        assert summary.tool_call_requests == 2
        assert summary.tool_call_successes == 2
        assert summary.tool_call_success_rate == 100.0

    def test_mixed_results(self):
        from xpyd_bench.bench.structured_output import ToolCallResult

        results = [
            StructuredOutputResult(
                tool_calls_expected=True,
                tool_calls_found=1,
                tool_call_results=[ToolCallResult(success=True)],
            ),
            StructuredOutputResult(
                tool_calls_expected=True,
                tool_calls_found=0,
                tool_call_results=[],
            ),
        ]
        summary = aggregate_structured_output(results)
        assert summary.tool_call_successes == 1
        assert summary.tool_call_failures == 1
        assert summary.tool_call_success_rate == 50.0

    def test_schema_aggregation(self):
        results = [
            StructuredOutputResult(json_schema_valid=True),
            StructuredOutputResult(json_schema_valid=True),
            StructuredOutputResult(json_schema_valid=False, schema_errors=["bad"]),
        ]
        summary = aggregate_structured_output(results)
        assert summary.schema_validations == 3
        assert summary.schema_passes == 2
        assert summary.schema_conformance_rate == pytest.approx(66.67, abs=0.1)

    def test_to_dict(self):
        summary = StructuredOutputSummary(
            total_requests=10,
            tool_call_requests=5,
            tool_call_successes=4,
            tool_call_failures=1,
            total_tool_calls_extracted=5,
            schema_validations=3,
            schema_passes=2,
            schema_failures=1,
        )
        d = summary.to_dict()
        assert d["tool_call_success_rate"] == 80.0
        assert d["schema_conformance_rate"] == pytest.approx(66.67, abs=0.1)
        assert "schema_validations" in d

    def test_empty(self):
        summary = aggregate_structured_output([])
        assert summary.total_requests == 0
        assert summary.tool_call_success_rate == 0.0


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
        # sim generates a value for string type; old dummy returned {}
        result = _generate_dummy_args({"type": "string"})
        assert isinstance(result, (str, dict))


# ---------------------------------------------------------------------------
# Dummy server tool call generation (integration)
# ---------------------------------------------------------------------------

class TestDummyServerToolCalls:
    def test_build_tool_calls_auto(self):
        from xpyd_sim.server import _build_tool_calls

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
        from xpyd_sim.server import _build_tool_calls

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
        from xpyd_sim.server import _build_tool_calls

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
        from xpyd_sim.server import _build_json_response

        resp = _build_json_response({"type": "json_object"}, 10)
        parsed = json.loads(resp)
        assert isinstance(parsed, dict)

    def test_build_json_response_json_schema(self):
        from xpyd_sim.server import _build_json_response

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

    config = ServerConfig(prefill_ms=0, decode_ms=0)
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
        # Validate with our validator
        result = validate_tool_calls(body, tools)
        assert result.success


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
        # Validate with structured output validator
        result = validate_json_response(content, rf)
        assert result.json_schema_valid is True


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
        assert not msg.get("tool_calls")  # None or absent
        assert msg["content"] is not None
