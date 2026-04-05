"""End-to-end tests: xpyd-sim (prefill + decode) + xpyd-proxy.

Tests the full PD disaggregation flow:
  client → proxy → sim(prefill) → sim(decode) → client

Validates response FORMAT (not content), both endpoints, streaming + non-streaming.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from xpyd.proxy import Proxy, RoundRobinSchedulingPolicy
from xpyd_sim.server import ServerConfig, create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "tests", "assets", "tokenizer")
_MODEL_NAME = _TOKENIZER_PATH


def _free_port():
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_server(app, port):
    uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")).run()


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


# ---------------------------------------------------------------------------
# Start prefill sim, decode sim, and proxy
# ---------------------------------------------------------------------------

_PREFILL_PORT = _free_port()
_DECODE_PORT = _free_port()
_PROXY_PORT = _free_port()

_prefill_app = create_app(ServerConfig(
    mode="prefill", model_name=_MODEL_NAME, prefill_delay_ms=5,
    kv_transfer_delay_ms=0, decode_delay_per_token_ms=1,
    eos_min_ratio=1.0, max_model_len=4096,
))
_decode_app = create_app(ServerConfig(
    mode="decode", model_name=_MODEL_NAME, prefill_delay_ms=0,
    kv_transfer_delay_ms=2, decode_delay_per_token_ms=3,
    eos_min_ratio=1.0, max_model_len=4096,
))

threading.Thread(target=_run_server, args=(_prefill_app, _PREFILL_PORT), daemon=True).start()
threading.Thread(target=_run_server, args=(_decode_app, _DECODE_PORT), daemon=True).start()
_wait_ready(_PREFILL_PORT)
_wait_ready(_DECODE_PORT)

_proxy = Proxy(
    prefill_instances=[f"127.0.0.1:{_PREFILL_PORT}"],
    decode_instances=[f"127.0.0.1:{_DECODE_PORT}"],
    model=_MODEL_NAME,
    scheduling_policy=RoundRobinSchedulingPolicy(),
    generator_on_p_node=False,
)
_proxy_app = FastAPI()
_proxy_app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)
_proxy_app.include_router(_proxy.router)

threading.Thread(target=_run_server, args=(_proxy_app, _PROXY_PORT), daemon=True).start()
_wait_ready(_PROXY_PORT, path="/status")

_BASE = f"http://127.0.0.1:{_PROXY_PORT}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_proxy_status():
    r = httpx.get(f"{_BASE}/status")
    assert r.status_code == 200


def test_chat_completions_non_streaming():
    r = httpx.post(f"{_BASE}/v1/chat/completions", json={
        "model": _MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) >= 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] in ("stop", "length")
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]


def test_chat_completions_streaming():
    r = httpx.post(f"{_BASE}/v1/chat/completions", json={
        "model": _MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    lines = r.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    assert len(data_lines) >= 2
    assert data_lines[-1] == "data: [DONE]"
    first = json.loads(data_lines[0][6:])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"].get("role") == "assistant"


def test_completions_non_streaming():
    r = httpx.post(f"{_BASE}/v1/completions", json={
        "model": _MODEL_NAME,
        "prompt": "Once upon a time",
        "max_tokens": 10,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) >= 1
    assert data["choices"][0]["finish_reason"] in ("stop", "length")
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]


def test_completions_streaming():
    r = httpx.post(f"{_BASE}/v1/completions", json={
        "model": _MODEL_NAME,
        "prompt": "Once upon a time",
        "max_tokens": 10,
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    lines = r.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    assert len(data_lines) >= 2
    assert data_lines[-1] == "data: [DONE]"


def test_models_endpoint():
    r = httpx.get(f"{_BASE}/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, (dict, list))


def test_ping():
    r = httpx.get(f"{_BASE}/ping", timeout=5)
    assert r.status_code in (200, 404, 405)
