"""Tests for API key authentication (M11)."""

from __future__ import annotations

import asyncio

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
            prefill_delay_ms=1.0,
            decode_delay_per_token_ms=1.0,
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

