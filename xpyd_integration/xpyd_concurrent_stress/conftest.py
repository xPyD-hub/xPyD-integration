"""Fixtures for xPyD multi-node concurrent/stress tests (2P+4D+proxy)."""

import socket
import threading
import time
from pathlib import Path

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from xpyd_sim.server import ServerConfig, create_app
from xpyd.proxy import Proxy, RoundRobinSchedulingPolicy

_TOKENIZER = str(Path(__file__).resolve().parent.parent / "assets" / "tokenizer")


def _free_port():
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(app, port):
    uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")).run()


def _wait(port, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


# Start 2P + 4D
_prefill_ports = [_free_port() for _ in range(2)]
_decode_ports = [_free_port() for _ in range(4)]

for p in _prefill_ports:
    app = create_app(ServerConfig(mode="prefill", model_name=_TOKENIZER, prefill_delay_ms=1, kv_transfer_delay_ms=0, decode_delay_per_token_ms=1, eos_min_ratio=1.0, max_model_len=131072))
    threading.Thread(target=_run, args=(app, p), daemon=True).start()

for p in _decode_ports:
    app = create_app(ServerConfig(mode="decode", model_name=_TOKENIZER, prefill_delay_ms=1, kv_transfer_delay_ms=0, decode_delay_per_token_ms=1, eos_min_ratio=1.0, max_model_len=131072))
    threading.Thread(target=_run, args=(app, p), daemon=True).start()

for p in _prefill_ports + _decode_ports:
    _wait(p)


def _make_proxy_app():
    proxy = Proxy(
        prefill_instances=[f"127.0.0.1:{p}" for p in _prefill_ports],
        decode_instances=[f"127.0.0.1:{p}" for p in _decode_ports],
        model=_TOKENIZER,
        scheduling_policy=RoundRobinSchedulingPolicy(),
        generator_on_p_node=False,
    )
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(proxy.router)
    return app


@pytest_asyncio.fixture
async def client():
    app = _make_proxy_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
