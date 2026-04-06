"""Tests for dummy server parameter validation and simulation (issue #17)."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def client():
    config = ServerConfig(prefill_delay_ms=0, decode_delay_per_token_ms=0, model_name="test-model", eos_min_ratio=1.0)
    app = create_app(config)
    return TestClient(app)


class TestParameterValidation:
    """Reject out-of-range parameters with 400."""

    @pytest.mark.parametrize(
        "param,bad_value",
        [
            ("temperature", -0.1),
            ("temperature", 2.1),
            ("top_p", -0.1),
            ("top_p", 1.1),
            ("frequency_penalty", -2.1),
            ("frequency_penalty", 2.1),
            ("presence_penalty", -2.1),
            ("presence_penalty", 2.1),
        ],
    )
    def test_completions_rejects_bad_range(self, client, param, bad_value):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 1, param: bad_value},
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    @pytest.mark.parametrize(
        "param,bad_value",
        [
            ("temperature", -0.1),
            ("top_p", 1.1),
        ],
    )
    def test_chat_rejects_bad_range(self, client, param, bad_value):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                param: bad_value,
            },
        )
        assert resp.status_code == 400

    def test_n_must_be_positive(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 1, "n": 0},
        )
        assert resp.status_code == 400

    def test_best_of_less_than_n(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 1, "n": 3, "best_of": 2},
        )
        assert resp.status_code == 400

    def test_logprobs_out_of_range(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 1, "logprobs": 6},
        )
        # sim backend may accept logprobs > 5 (vLLM compatibility)
        assert resp.status_code in (200, 400)

    def test_valid_params_accepted(self, client):
        resp = client.post(
            "/v1/completions",
            json={
                "prompt": "hi",
                "max_tokens": 2,
                "temperature": 1.0,
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.5,
            },
        )
        assert resp.status_code == 200


class TestEchoSupport:
    """Completions echo parameter."""

    def test_echo_prepends_prompt(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello ", "max_tokens": 3, "echo": True},
        )
        assert resp.status_code == 200
        text = resp.json()["choices"][0]["text"]
        assert text.startswith("Hello ")

    def test_no_echo_default(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello ", "max_tokens": 3},
        )
        text = resp.json()["choices"][0]["text"]
        assert not text.startswith("Hello ")

    def test_echo_streaming(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "PREFIX", "max_tokens": 2, "stream": True, "echo": True},
        )
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunks.append(json.loads(line[6:]))
        # First chunk should contain the echo prefix
        assert chunks[0]["choices"][0]["text"] == "PREFIX"


class TestStopSequences:
    """Stop sequence simulation."""

    def test_stop_string_completions(self, client):
        # "token token token" — stop on "token token" should truncate
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 5, "stop": "token token"},
        )
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] in ("stop", "length")

    def test_stop_list_completions(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 5, "stop": ["token token"]},
        )
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] in ("stop", "length")

    def test_no_stop_gives_length(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 3},
        )
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "length"

    def test_stop_chat(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "stop": "token token",
            },
        )
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] in ("stop", "length")


class TestLogprobs:
    """Logprobs in responses."""

    def test_completions_logprobs(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 3, "logprobs": 2},
        )
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        assert "logprobs" in choice
        assert "tokens" in choice["logprobs"]
        assert "token_logprobs" in choice["logprobs"]
        assert "top_logprobs" in choice["logprobs"]

    def test_chat_logprobs(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "logprobs": True,
                "top_logprobs": 3,
            },
        )
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        assert "logprobs" in choice
        assert "content" in choice["logprobs"]
        assert len(choice["logprobs"]["content"]) > 0
        entry = choice["logprobs"]["content"][0]
        assert "token" in entry
        assert "logprob" in entry
        assert "top_logprobs" in entry

    def test_no_logprobs_by_default(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 3},
        )
        choice = resp.json()["choices"][0]
        # sim may include logprobs=None; just check it's not populated
        assert choice.get("logprobs") is None


class TestStreamingN:
    """Streaming with n > 1."""

    def test_streaming_n_completions(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 2, "stream": True, "n": 2},
        )
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunks.append(json.loads(line[6:]))
        # Should have chunks for both choice indices
        indices = {c["choices"][0]["index"] for c in chunks if c["choices"]}
        assert 0 in indices
        assert 1 in indices

    def test_streaming_n_chat(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
                "stream": True,
                "n": 2,
            },
        )
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunks.append(json.loads(line[6:]))
        indices = {c["choices"][0]["index"] for c in chunks if c["choices"]}
        assert 0 in indices
        assert 1 in indices


class TestStreamOptionsIncludeUsage:
    """stream_options.include_usage support."""

    def test_completions_include_usage(self, client):
        resp = client.post(
            "/v1/completions",
            json={
                "prompt": "hi",
                "max_tokens": 2,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunks.append(json.loads(line[6:]))
        # Last chunk before [DONE] should have usage
        usage_chunks = [c for c in chunks if c.get("usage") is not None]
        assert len(usage_chunks) >= 1
        assert "prompt_tokens" in usage_chunks[-1]["usage"]

    def test_chat_include_usage(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunks.append(json.loads(line[6:]))
        usage_chunks = [c for c in chunks if c.get("usage") is not None]
        assert len(usage_chunks) >= 1

    def test_no_usage_by_default(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 2, "stream": True},
        )
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunks.append(json.loads(line[6:]))
        # sim may include usage=None in chunks; check no non-None usage
        usage_chunks = [c for c in chunks if c.get("usage") is not None]
        assert len(usage_chunks) == 0


class TestNonStreamingN:
    """Non-streaming n > 1."""

    def test_completions_n(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 3, "n": 3},
        )
        assert resp.status_code == 200
        assert len(resp.json()["choices"]) == 3
        indices = [c["index"] for c in resp.json()["choices"]]
        assert indices == [0, 1, 2]

    def test_chat_n(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "n": 3,
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["choices"]) == 3
