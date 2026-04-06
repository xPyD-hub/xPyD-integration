"""Shared test fixtures for 1p1d_basic tests.

Starts sim(prefill) + sim(decode) via xpyd-sim, creates a Proxy app,
and provides an AsyncClient fixture.
"""

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient
from pathlib import Path

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
    """Create a sim app using xpyd-sim."""
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
# Module-level: start sim nodes once
# ---------------------------------------------------------------------------

_prefill_app = _make_sim_app("prefill")
_decode_app = _make_sim_app("decode")

_PREFILL_PORT = _free_port()
_DECODE_PORT = _free_port()

threading.Thread(target=_run_server, args=(_prefill_app, _PREFILL_PORT), daemon=True).start()
threading.Thread(target=_run_server, args=(_decode_app, _DECODE_PORT), daemon=True).start()

_wait_ready(_PREFILL_PORT)
_wait_ready(_DECODE_PORT)


# ---------------------------------------------------------------------------
# Proxy app factory
# ---------------------------------------------------------------------------


def _make_proxy_app(
    prefill_instances: list | None = None,
    decode_instances: list | None = None,
) -> FastAPI:
    proxy = Proxy(
        prefill_instances=prefill_instances or [f"127.0.0.1:{_PREFILL_PORT}"],
        decode_instances=decode_instances or [f"127.0.0.1:{_DECODE_PORT}"],
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
def dummy_ports():
    return _PREFILL_PORT, _DECODE_PORT


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    app = _make_proxy_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cli:
        yield cli


@pytest.fixture
async def prefill_client():
    async with AsyncClient(
        transport=ASGITransport(app=_prefill_app), base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
async def decode_client():
    async with AsyncClient(
        transport=ASGITransport(app=_decode_app), base_url="http://test"
    ) as c:
        yield c
