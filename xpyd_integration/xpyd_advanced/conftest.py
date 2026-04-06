"""Shared fixtures for xpyd_advanced tests (3P + 3D + 1 proxy).

Topology: 3 prefill nodes, 3 decode nodes, 1 proxy with round-robin scheduling.
"""

import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from xpyd_sim.server import ServerConfig, create_app
from xpyd.proxy import Proxy, RoundRobinSchedulingPolicy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TOKENIZER_PATH = str(Path(__file__).resolve().parent.parent / "assets" / "tokenizer")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_server(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvicorn.Server(config).run()


def _wait_ready(port, path="/health", timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Server on port {port} not ready after {timeout}s")


def _make_sim_app(mode: str):
    return create_app(ServerConfig(
        mode=mode,
        model_name=_TOKENIZER_PATH,
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        eos_min_ratio=1.0,
        max_model_len=131072,
    ))


# ---------------------------------------------------------------------------
# Module-level: start 3 prefill + 3 decode sim nodes once
# ---------------------------------------------------------------------------

NUM_PREFILL = 3
NUM_DECODE = 3

_prefill_apps = [_make_sim_app("prefill") for _ in range(NUM_PREFILL)]
_decode_apps = [_make_sim_app("decode") for _ in range(NUM_DECODE)]

_PREFILL_PORTS = [_free_port() for _ in range(NUM_PREFILL)]
_DECODE_PORTS = [_free_port() for _ in range(NUM_DECODE)]

for app, port in zip(_prefill_apps, _PREFILL_PORTS):
    threading.Thread(target=_run_server, args=(app, port), daemon=True).start()

for app, port in zip(_decode_apps, _DECODE_PORTS):
    threading.Thread(target=_run_server, args=(app, port), daemon=True).start()

for port in _PREFILL_PORTS + _DECODE_PORTS:
    _wait_ready(port)


# ---------------------------------------------------------------------------
# Proxy app factory
# ---------------------------------------------------------------------------


def _make_proxy_app() -> FastAPI:
    proxy = Proxy(
        prefill_instances=[f"127.0.0.1:{p}" for p in _PREFILL_PORTS],
        decode_instances=[f"127.0.0.1:{p}" for p in _DECODE_PORTS],
        model=_TOKENIZER_PATH,
        scheduling_policy=RoundRobinSchedulingPolicy(),
        generator_on_p_node=False,
    )
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(proxy.router)
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    app = _make_proxy_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli
