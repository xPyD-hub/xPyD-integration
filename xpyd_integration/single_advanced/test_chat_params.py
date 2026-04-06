"""Tests for issue #20: Chat-specific OpenAI API parameters."""

from __future__ import annotations


import pytest
from starlette.testclient import TestClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def client():
    config = ServerConfig(prefill_delay_ms=0, decode_delay_per_token_ms=0, model_name="test-model", eos_min_ratio=1.0)
    app = create_app(config)
    return TestClient(app)


# ── CLI argument tests ────────────────────────────────────────────────


# ── _build_payload tests ─────────────────────────────────────────────


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
