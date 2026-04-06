"""Shared fixtures for 1p1d concurrent/stress tests.

Starts sim(prefill) + sim(decode) via xpyd-sim, creates a Proxy app,
and provides AsyncClient + cluster fixtures for both ASGI and subprocess modes.
"""

from __future__ import annotations

import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import httpx
import pytest
import uvicorn
import yaml
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


def _free_port() -> int:
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


def _wait_port(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


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


def _random_content(length: int) -> str:
    """Generate a random string of approximately *length* characters."""
    if length <= 0:
        return ""
    words = ["hello", "world", "bench", "test", "proxy", "stream", "token", "data"]
    pieces: list[str] = []
    cur = 0
    while cur < length:
        w = random.choice(words)
        pieces.append(w)
        cur += len(w) + 1
    return " ".join(pieces)[:length]


# ---------------------------------------------------------------------------
# Module-level: start sim nodes once (for ASGI-based tests)
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
# Common payload
# ---------------------------------------------------------------------------

CHAT_PAYLOAD = {
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5,
    "stream": False,
}


# ---------------------------------------------------------------------------
# Fixtures — ASGI mode (test_concurrent_requests)
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


# ---------------------------------------------------------------------------
# Fixtures — subprocess cluster mode (benchmark tests)
# ---------------------------------------------------------------------------

NUM_PREFILL = 2
NUM_DECODE = 16
MAX_CONCURRENCY = 1_000


@pytest.fixture(scope="module")
def cluster():
    """Spin up dummy sim nodes + proxy as subprocesses, yield connection info."""
    env = os.environ.copy()
    env["PREFILL_DELAY_PER_TOKEN"] = "0"
    env["DECODE_DELAY_PER_TOKEN"] = "0"
    procs: list[subprocess.Popen] = []

    prefill_ports = [_free_port() for _ in range(NUM_PREFILL)]
    decode_ports = [_free_port() for _ in range(NUM_DECODE)]
    proxy_port = _free_port()

    try:
        for port in prefill_ports:
            procs.append(subprocess.Popen(
                [
                    sys.executable, "-c",
                    "from xpyd_sim.cli import main; main(['serve',"
                    f"'--mode','prefill','--model','{_TOKENIZER_PATH}',"
                    f"'--host','127.0.0.1','--port','{port}',"
                    "'--prefill-delay-ms','0','--kv-transfer-delay-ms','0',"
                    "'--decode-delay-per-token-ms','0','--eos-min-ratio','1.0'])",
                ],
                env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))

        for port in decode_ports:
            procs.append(subprocess.Popen(
                [
                    sys.executable, "-c",
                    "from xpyd_sim.cli import main; main(['serve',"
                    f"'--mode','decode','--model','{_TOKENIZER_PATH}',"
                    f"'--host','127.0.0.1','--port','{port}',"
                    "'--prefill-delay-ms','0','--kv-transfer-delay-ms','0',"
                    "'--decode-delay-per-token-ms','0','--eos-min-ratio','1.0'])",
                ],
                env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))

        for port in prefill_ports + decode_ports:
            assert _wait_port(port, timeout=20), f"Node on port {port} failed to start"

        # Write proxy config
        cfg = {
            "model": _TOKENIZER_PATH,
            "prefill": [f"127.0.0.1:{p}" for p in prefill_ports],
            "decode": [f"127.0.0.1:{p}" for p in decode_ports],
            "port": proxy_port,
        }
        cf = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(cfg, cf)
        cf.close()

        procs.append(subprocess.Popen(
            [sys.executable, "-m", "xpyd.proxy", "proxy", "--config", cf.name],
            env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
        assert _wait_port(proxy_port, timeout=30), "Proxy failed to start"

        yield {
            "proxy_port": proxy_port,
            "model": _TOKENIZER_PATH,
            "prefill_ports": prefill_ports,
            "decode_ports": decode_ports,
        }

    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
