"""Tests for issue #85: echo=True completion_tokens consistency."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def client():
    config = ServerConfig(prefill_delay_ms=0, decode_delay_per_token_ms=0, model_name="test-model", eos_min_ratio=1.0)
    app = create_app(config)
    return TestClient(app)


class TestEchoCompletionTokens:
    """Verify completion_tokens is consistent with and without echo."""

    def test_echo_does_not_change_completion_tokens(self, client):
        """completion_tokens must be the same regardless of echo flag."""
        payload = {"prompt": "one two three four five", "max_tokens": 10}

        r1 = client.post("/v1/completions", json=payload).json()
        r2 = client.post("/v1/completions", json={**payload, "echo": True}).json()

        assert r1["usage"]["completion_tokens"] == 10
        assert r2["usage"]["completion_tokens"] == 10

    def test_echo_prepends_prompt_text(self, client):
        """echo=True should prepend the prompt to the generated text."""
        prompt = "hello world"
        r = client.post(
            "/v1/completions",
            json={"prompt": prompt, "max_tokens": 5, "echo": True},
        ).json()

        text = r["choices"][0]["text"]
        assert text.startswith(prompt)

    def test_echo_false_no_prompt_in_text(self, client):
        """echo=False should not include prompt in text."""
        prompt = "hello world"
        r = client.post(
            "/v1/completions",
            json={"prompt": prompt, "max_tokens": 5, "echo": False},
        ).json()

        text = r["choices"][0]["text"]
        assert not text.startswith(prompt)

    def test_echo_with_n_greater_than_1(self, client):
        """completion_tokens consistent with echo and n>1."""
        payload = {"prompt": "a b c", "max_tokens": 5, "n": 2}

        r1 = client.post("/v1/completions", json=payload).json()
        r2 = client.post("/v1/completions", json={**payload, "echo": True}).json()

        assert r1["usage"]["completion_tokens"] == r2["usage"]["completion_tokens"]
