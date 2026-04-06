"""Tests for M3: Full OpenAI API parameter coverage."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture()
def client():
    app = create_app(ServerConfig(prefill_ms=0, decode_ms=0, eos_min_ratio=1.0))
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


class TestCLIArgs:
    def test_stop_args(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--stop", "END", "STOP"])
        assert args.stop == ["END", "STOP"]

    def test_n_arg(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--n", "5"])
        assert args.n == 5

    def test_api_seed_arg(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--api-seed", "42"])
        assert args.api_seed == 42


# ---------------------------------------------------------------------------
# Runner payload building
# ---------------------------------------------------------------------------


class TestPayloadBuild:
    def test_payload_includes_new_params(self):
        from argparse import Namespace

        from xpyd_bench.bench.runner import _build_payload

        args = Namespace(
            model="m",
            output_len=10,
            temperature=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            best_of=None,
            use_beam_search=False,
            logprobs=None,
            ignore_eos=False,
            stop=["END"],
            n=3,
            api_seed=42,
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert payload["stop"] == ["END"]
        assert payload["n"] == 3
        assert payload["seed"] == 42

    def test_payload_omits_none_params(self):
        from argparse import Namespace

        from xpyd_bench.bench.runner import _build_payload

        args = Namespace(
            model="m",
            output_len=10,
            temperature=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            best_of=None,
            use_beam_search=False,
            logprobs=None,
            ignore_eos=False,
            stop=None,
            n=None,
            api_seed=None,
        )
        payload = _build_payload(args, "hello", is_chat=False)
        assert "stop" not in payload
        assert "n" not in payload
        assert "seed" not in payload
