"""Tests for node discovery (extracted from test_cli_and_discovery.py)."""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from httpx import ASGITransport, AsyncClient

from xpyd.discovery import DiscoveryTimeout, NodeDiscovery


def _free_port():
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_health_server(port: int):
    app = FastAPI()

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    def _run():
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        uvicorn.Server(config).run()

    threading.Thread(target=_run, daemon=True).start()
    time.sleep(1)


@pytest.mark.anyio
async def test_discovery_finds_healthy_nodes():
    """Discovery should detect healthy nodes and become ready."""
    p_port = _free_port()
    d_port = _free_port()
    _start_health_server(p_port)
    _start_health_server(d_port)

    disc = NodeDiscovery(
        prefill_instances=[f"127.0.0.1:{p_port}"],
        decode_instances=[f"127.0.0.1:{d_port}"],
        probe_interval=0.5,
        wait_timeout=10,
    )
    await disc.start()
    ready = await disc.wait_until_ready()
    await disc.stop()

    assert ready is True
    assert disc.is_ready
    assert f"127.0.0.1:{p_port}" in disc.healthy_prefill
    assert f"127.0.0.1:{d_port}" in disc.healthy_decode


@pytest.mark.anyio
async def test_discovery_timeout_when_no_nodes():
    """Discovery should raise DiscoveryTimeout when nodes are unreachable."""
    disc = NodeDiscovery(
        prefill_instances=["127.0.0.1:1"],
        decode_instances=["127.0.0.1:2"],
        probe_interval=0.2,
        wait_timeout=1.0,
    )
    await disc.start()
    disc._task.remove_done_callback(disc._on_probe_done)

    ready = await disc.wait_until_ready()
    assert ready is False
    assert not disc.is_ready

    with pytest.raises(DiscoveryTimeout):
        await disc._task


@pytest.mark.anyio
async def test_503_before_ready():
    """Proxy should return 503 before discovery reports ready."""
    disc = NodeDiscovery(
        prefill_instances=["127.0.0.1:1"],
        decode_instances=["127.0.0.1:2"],
        probe_interval=60,
        wait_timeout=600,
    )

    app = FastAPI()

    @app.middleware("http")
    async def check_readiness(request, call_next):
        path = request.url.path
        if path in ("/health", "/ping", "/status", "/metrics"):
            return await call_next(request)
        if not disc.is_ready:
            return JSONResponse({"error": "waiting for backend nodes"}, status_code=503)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    @app.post("/v1/completions")
    async def completions():
        return JSONResponse({"choices": []})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200

        resp = await client.post("/v1/completions", json={})
        assert resp.status_code == 503
        assert "waiting for backend nodes" in resp.json()["error"]
