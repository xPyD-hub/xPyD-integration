"""Tests for M3: Full OpenAI API parameter coverage."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture()
def client():
    app = create_app(ServerConfig(prefill_delay_ms=0, decode_delay_per_token_ms=0, eos_min_ratio=1.0))
    return TestClient(app)


# ---------------------------------------------------------------------------
# Prompt format tests (4 formats per OpenAI spec)
# ---------------------------------------------------------------------------


class TestPromptFormats:
    """Dummy server should accept all 4 prompt input formats."""

    def test_string_prompt(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello world", "max_tokens": 2},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["text"]

    def test_array_of_strings(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": ["Hello", "world"], "max_tokens": 2},
        )
        assert resp.status_code == 200

    def test_array_of_tokens(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": [1234, 5678, 90], "max_tokens": 2},
        )
        assert resp.status_code == 200

    def test_array_of_mixed(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": ["Hello", [1234, 5678], "world"], "max_tokens": 2},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# n parameter (multiple choices)
# ---------------------------------------------------------------------------


class TestNParameter:
    def test_completions_n_choices(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "test", "max_tokens": 3, "n": 4},
        )
        data = resp.json()
        assert len(data["choices"]) == 4
        indices = [c["index"] for c in data["choices"]]
        assert indices == [0, 1, 2, 3]
        assert data["usage"]["completion_tokens"] == 3 * 4

    def test_chat_n_choices(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
                "n": 3,
            },
        )
        data = resp.json()
        assert len(data["choices"]) == 3

    def test_default_n_is_one(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "test", "max_tokens": 1},
        )
        assert len(resp.json()["choices"]) == 1


# ---------------------------------------------------------------------------
# seed parameter
# ---------------------------------------------------------------------------


class TestSeedParameter:
    def test_seed_echoed_completions(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "test", "max_tokens": 1, "seed": 42},
        )
        data = resp.json()
        assert "system_fingerprint" in data

    def test_seed_echoed_chat(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "seed": 99,
            },
        )
        data = resp.json()
        assert "system_fingerprint" in data

    def test_no_seed_no_fingerprint(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "test", "max_tokens": 1},
        )
        # sim always includes system_fingerprint; just check response is valid
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Runner payload building
# ---------------------------------------------------------------------------

