"""Tests for issue #20: Chat-specific OpenAI API parameters."""

from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from xpyd_bench.bench.runner import _build_payload
from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def client():
    config = ServerConfig(prefill_ms=0, decode_ms=0, model_name="test-model", eos_min_ratio=1.0)
    app = create_app(config)
    return TestClient(app)


# ── CLI argument tests ────────────────────────────────────────────────


class TestChatCLIArgs:
    """Verify chat-specific CLI arguments exist and parse correctly."""

    def _get_parser(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        return parser

    def test_chat_group_exists(self):
        parser = self._get_parser()
        group_titles = [g.title for g in parser._action_groups]
        assert "chat-specific parameters" in group_titles

    def test_response_format_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["--response-format", '{"type": "json_object"}'])
        assert args.response_format == '{"type": "json_object"}'

    def test_tools_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["--tools", "/path/to/tools.json"])
        assert args.tools == "/path/to/tools.json"

    def test_tool_choice_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["--tool-choice", "auto"])
        assert args.tool_choice == "auto"

    def test_top_logprobs_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["--top-logprobs", "5"])
        assert args.top_logprobs == 5

    def test_max_completion_tokens_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["--max-completion-tokens", "1024"])
        assert args.max_completion_tokens == 1024

    def test_service_tier_parses(self):
        parser = self._get_parser()
        args = parser.parse_args(["--service-tier", "auto"])
        assert args.service_tier == "auto"


# ── _build_payload tests ─────────────────────────────────────────────


class TestBuildPayloadChatParams:
    """Verify _build_payload includes chat-specific params for chat endpoints."""

    def _make_args(self, **kwargs):
        defaults = {
            "output_len": 128,
            "model": "test",
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "frequency_penalty": None,
            "presence_penalty": None,
            "best_of": None,
            "use_beam_search": False,
            "logprobs": None,
            "ignore_eos": False,
            "stop": None,
            "n": None,
            "api_seed": None,
            "echo": False,
            "suffix": None,
            "logit_bias": None,
            "user": None,
            "stream_options_include_usage": False,
            "response_format": None,
            "tools": None,
            "tool_choice": None,
            "parallel_tool_calls": None,
            "top_logprobs": None,
            "max_completion_tokens": None,
            "service_tier": None,
        }
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_response_format_included(self):
        args = self._make_args(response_format='{"type": "json_object"}')
        payload = _build_payload(args, "hello", is_chat=True)
        assert payload["response_format"] == {"type": "json_object"}

    def test_response_format_excluded_for_completions(self):
        args = self._make_args(response_format='{"type": "json_object"}')
        payload = _build_payload(args, "hello", is_chat=False)
        assert "response_format" not in payload

    def test_tools_from_file(self):
        tools_data = [{"type": "function", "function": {"name": "test_fn"}}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(tools_data, f)
            f.flush()
            args = self._make_args(tools=f.name)
            payload = _build_payload(args, "hello", is_chat=True)
            assert payload["tools"] == tools_data
        Path(f.name).unlink()

    def test_tool_choice_string(self):
        args = self._make_args(tool_choice="auto")
        payload = _build_payload(args, "hello", is_chat=True)
        assert payload["tool_choice"] == "auto"

    def test_tool_choice_json(self):
        args = self._make_args(
            tool_choice='{"type": "function", "function": {"name": "test_fn"}}'
        )
        payload = _build_payload(args, "hello", is_chat=True)
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "test_fn"},
        }

    def test_top_logprobs_included(self):
        args = self._make_args(top_logprobs=5)
        payload = _build_payload(args, "hello", is_chat=True)
        assert payload["top_logprobs"] == 5

    def test_max_completion_tokens_included(self):
        args = self._make_args(max_completion_tokens=1024)
        payload = _build_payload(args, "hello", is_chat=True)
        assert payload["max_completion_tokens"] == 1024

    def test_service_tier_included(self):
        args = self._make_args(service_tier="auto")
        payload = _build_payload(args, "hello", is_chat=True)
        assert payload["service_tier"] == "auto"

    def test_none_params_excluded(self):
        args = self._make_args()
        payload = _build_payload(args, "hello", is_chat=True)
        for key in (
            "response_format",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "top_logprobs",
            "max_completion_tokens",
            "service_tier",
        ):
            assert key not in payload


# ── Dummy server tests ───────────────────────────────────────────────


class TestDummyChatParams:
    """Verify dummy server accepts chat-specific parameters."""

    def test_response_format_accepted(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "response_format": {"type": "json_object"},
            },
        )
        assert resp.status_code == 200

    def test_tools_accepted(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {}},
                    }
                ],
                "tool_choice": "auto",
            },
        )
        assert resp.status_code == 200

    def test_max_completion_tokens_as_fallback(self, client):
        """max_completion_tokens should be used when max_tokens is absent."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_completion_tokens": 3,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        tokens = body["choices"][0]["message"]["content"].split()
        assert len(tokens) == 3

    def test_service_tier_accepted(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "service_tier": "auto",
            },
        )
        assert resp.status_code == 200

    def test_top_logprobs_accepted(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "logprobs": True,
                "top_logprobs": 3,
            },
        )
        assert resp.status_code == 200

    def test_streaming_with_chat_params(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 3,
                "stream": True,
                "response_format": {"type": "json_object"},
                "service_tier": "auto",
            },
        )
        assert resp.status_code == 200
        assert "data: " in resp.text
