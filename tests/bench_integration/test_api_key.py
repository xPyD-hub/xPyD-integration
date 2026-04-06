"""Tests for API key authentication (M11)."""

from __future__ import annotations

import asyncio
import os
from argparse import Namespace
from unittest.mock import patch

import httpx
import pytest
import uvicorn

from xpyd_sim.server import ServerConfig, create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_healthy(base: str, timeout: float = 5.0) -> None:
    async with httpx.AsyncClient() as c:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await c.get(f"{base}/health")
                if r.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.1)
    raise TimeoutError("server did not become healthy")


# ---------------------------------------------------------------------------
# CLI argument tests
# ---------------------------------------------------------------------------


class TestCLIApiKeyArg:
    """Test --api-key CLI argument parsing."""

    def test_api_key_parsed(self) -> None:

        import argparse

        # Just verify the argument is accepted by the parser
        parser = argparse.ArgumentParser()
        from xpyd_bench.cli import _add_vllm_compat_args

        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--api-key", "sk-test123"])
        assert args.api_key == "sk-test123"

    def test_api_key_default_none(self) -> None:
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.api_key is None

    def test_api_key_env_fallback(self) -> None:
        """OPENAI_API_KEY env var is used when --api-key not provided."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])

        # Simulate what bench_main does
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}):
            if args.api_key is None:
                args.api_key = os.environ.get("OPENAI_API_KEY")
        assert args.api_key == "sk-from-env"

    def test_api_key_cli_overrides_env(self) -> None:
        """--api-key takes precedence over env var."""
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--api-key", "sk-cli"])

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}):
            if args.api_key is None:
                args.api_key = os.environ.get("OPENAI_API_KEY")
        assert args.api_key == "sk-cli"


# ---------------------------------------------------------------------------
# Dummy server auth tests
# ---------------------------------------------------------------------------


class TestDummyServerAuth:
    """Test dummy server --require-api-key."""

    @pytest.fixture()
    def _server_with_auth(self):
        """Start a dummy server that requires an API key."""
        port = _find_free_port()
        config = ServerConfig(
            prefill_ms=1.0,
            decode_ms=1.0,
            model_name="test-model",
            require_api_key="sk-secret",
        )
        app = create_app(config)

        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )

        loop = asyncio.new_event_loop()
        thread = None

        import threading

        def run():
            loop.run_until_complete(server.serve())

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        base = f"http://127.0.0.1:{port}"
        asyncio.run(_wait_healthy(base))
        yield base
        server.should_exit = True
        thread.join(timeout=3)

    @pytest.fixture()
    def server_url(self, _server_with_auth):
        return _server_with_auth

    def test_request_without_key_returns_401(self, server_url: str) -> None:
        import httpx as hx

        r = hx.post(
            f"{server_url}/v1/completions",
            json={"prompt": "hello", "max_tokens": 1, "model": "test-model"},
        )
        assert r.status_code == 401
        assert "auth_error" in r.json()["error"]["type"]

    def test_request_with_wrong_key_returns_401(self, server_url: str) -> None:
        import httpx as hx

        r = hx.post(
            f"{server_url}/v1/completions",
            json={"prompt": "hello", "max_tokens": 1, "model": "test-model"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 401

    def test_request_with_correct_key_succeeds(self, server_url: str) -> None:
        import httpx as hx

        r = hx.post(
            f"{server_url}/v1/completions",
            json={"prompt": "hello", "max_tokens": 1, "model": "test-model"},
            headers={"Authorization": "Bearer sk-secret"},
        )
        assert r.status_code == 200

    def test_health_endpoint_no_auth_required(self, server_url: str) -> None:
        import httpx as hx

        r = hx.get(f"{server_url}/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Runner auth header injection test
# ---------------------------------------------------------------------------


class TestRunnerAuthHeaders:
    """Test that runner injects Authorization header."""

    @pytest.fixture()
    def _server_no_auth(self):
        """Start a dummy server without auth requirement."""
        port = _find_free_port()
        config = ServerConfig(
            prefill_ms=1.0,
            decode_ms=1.0,
            model_name="test-model",
        )
        app = create_app(config)
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )
        loop = asyncio.new_event_loop()

        import threading

        def run():
            loop.run_until_complete(server.serve())

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        asyncio.run(_wait_healthy(base))
        yield base
        server.should_exit = True
        thread.join(timeout=3)

    @pytest.fixture()
    def _server_with_key(self):
        """Start a dummy server requiring auth."""
        port = _find_free_port()
        config = ServerConfig(
            prefill_ms=1.0,
            decode_ms=1.0,
            model_name="test-model",
            require_api_key="sk-runner-test",
        )
        app = create_app(config)
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )
        loop = asyncio.new_event_loop()

        import threading

        def run():
            loop.run_until_complete(server.serve())

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        asyncio.run(_wait_healthy(base))
        yield base
        server.should_exit = True
        thread.join(timeout=3)

    async def test_runner_sends_auth_header(self, _server_with_key: str) -> None:
        """Benchmark runner sends Authorization header when api_key is set."""
        from xpyd_bench.bench.runner import run_benchmark

        base_url = _server_with_key
        args = Namespace(
            backend="openai",
            endpoint="/v1/completions",
            model="test-model",
            num_prompts=2,
            request_rate=float("inf"),
            burstiness=1.0,
            max_concurrency=None,
            input_len=8,
            output_len=4,
            seed=42,
            dataset_name="random",
            dataset_path=None,
            disable_tqdm=True,
            save_result=False,
            rich_progress=False,
            warmup=0,
            timeout=10.0,
            retries=0,
            retry_delay=1.0,
            api_key="sk-runner-test",
            # Sampling params
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
            echo=False,
            suffix=None,
            logit_bias=None,
            user=None,
            stream_options_include_usage=False,
        )

        result_dict, bench_result = await run_benchmark(args, base_url)
        assert bench_result.completed == 2
        assert bench_result.failed == 0

    async def test_runner_fails_without_key_when_required(
        self, _server_with_key: str
    ) -> None:
        """Requests fail when server requires key but none provided."""
        from xpyd_bench.bench.runner import run_benchmark

        base_url = _server_with_key
        args = Namespace(
            backend="openai",
            endpoint="/v1/completions",
            model="test-model",
            num_prompts=1,
            request_rate=float("inf"),
            burstiness=1.0,
            max_concurrency=None,
            input_len=8,
            output_len=4,
            seed=42,
            dataset_name="random",
            dataset_path=None,
            disable_tqdm=True,
            save_result=False,
            rich_progress=False,
            warmup=0,
            timeout=10.0,
            retries=0,
            retry_delay=1.0,
            api_key=None,
            # Sampling params
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
            echo=False,
            suffix=None,
            logit_bias=None,
            user=None,
            stream_options_include_usage=False,
        )

        result_dict, bench_result = await run_benchmark(args, base_url)
        # All requests should fail with 401
        assert bench_result.failed == 1
        assert bench_result.completed == 0
