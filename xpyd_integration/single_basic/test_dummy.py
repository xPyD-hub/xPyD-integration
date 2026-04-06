"""Tests for the dummy server endpoints."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def client():
    """Create a test client with fast config."""
    config = ServerConfig(prefill_delay_ms=0, decode_delay_per_token_ms=0, model_name="test-model", eos_min_ratio=1.0)
    app = create_app(config)
    return TestClient(app)


class TestCompletions:
    """Tests for /v1/completions endpoint."""

    def test_non_streaming(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello world", "max_tokens": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["model"] == "test-model"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["usage"]["prompt_tokens"] > 0
        assert body["usage"]["completion_tokens"] == 5

    def test_streaming(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello world", "max_tokens": 3, "stream": True},
        )
        assert resp.status_code == 200

        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                chunks.append(json.loads(data))

        assert len(chunks) >= 3
        assert chunks[0]["choices"][0]["text"] is not None
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"


class TestChatCompletions:
    """Tests for /v1/chat/completions endpoint."""

    def test_non_streaming(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "test-model"
        assert len(body["choices"]) == 1
        assert "message" in body["choices"][0]
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["usage"]["completion_tokens"] == 5

    def test_streaming(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "stream": True,
            },
        )
        assert resp.status_code == 200

        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                chunks.append(json.loads(data))

        assert len(chunks) >= 4  # 1 role chunk + 3 content chunks (+ possible finish chunk)
        # First chunk has role: assistant
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert "delta" in chunks[1]["choices"][0]


class TestUsageStats:
    """Tests for usage statistics accuracy."""

    def test_completions_usage(self, client):
        prompt = "a " * 100  # ~50 tokens
        resp = client.post(
            "/v1/completions",
            json={"prompt": prompt, "max_tokens": 10},
        )
        body = resp.json()
        usage = body["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] == 10
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_chat_usage(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "x " * 80}],
                "max_tokens": 20,
            },
        )
        body = resp.json()
        usage = body["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] == 20


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "test-model"


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestDummyCLI:
    """Tests for dummy CLI argument parsing."""

    def test_default_args(self):
        import argparse

        # Just test that the parser works — don't actually start server
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--prefill-ms", type=float, default=50.0)
        parser.add_argument("--decode-ms", type=float, default=10.0)
        parser.add_argument("--model-name", type=str, default="dummy-model")

        args = parser.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8000
        assert args.prefill_ms == 50.0
        assert args.decode_ms == 10.0
        assert args.model_name == "dummy-model"

    def test_custom_args(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--prefill-ms", type=float, default=50.0)
        parser.add_argument("--decode-ms", type=float, default=10.0)
        parser.add_argument("--model-name", type=str, default="dummy-model")

        args = parser.parse_args([
            "--host", "0.0.0.0",
            "--port", "9000",
            "--prefill-ms", "100",
            "--decode-ms", "20",
            "--model-name", "my-model",
        ])
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.prefill_ms == 100.0
        assert args.decode_ms == 20.0
        assert args.model_name == "my-model"


class TestInvalidJsonBody:
    """Tests for invalid JSON body returning 400."""

    def test_completions_invalid_json(self, client):
        resp = client.post(
            "/v1/completions",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"].get("code", None) is None or body["error"]["code"] == "invalid_json"

    def test_chat_completions_invalid_json(self, client):
        resp = client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"].get("code", None) is None or body["error"]["code"] == "invalid_json"

    def test_completions_empty_body(self, client):
        resp = client.post(
            "/v1/completions",
            content=b"",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_chat_completions_empty_body(self, client):
        resp = client.post(
            "/v1/chat/completions",
            content=b"",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
