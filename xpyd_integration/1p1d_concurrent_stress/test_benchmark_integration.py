"""Integration benchmark: proxy + dummy nodes end-to-end.

Migrated from xPyD-proxy/tests/stress/test_benchmark_integration.py
Original: 6 tests.

Topology: 2 prefill + 16 decode + 1 proxy (subprocess).
Excluded from CI. Run manually:

    pytest xpyd_integration/1p1d_concurrent_stress/test_benchmark_integration.py -v
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess

import httpx
import pytest

NUM_PREFILL = 2
NUM_DECODE = 16

CHAT_PAYLOAD = {
    "model": "",
    "messages": [{"role": "user", "content": "Hello world"}],
    "max_tokens": 5,
    "stream": False,
}


def test_models_endpoint(cluster):
    """Proxy /v1/models returns OpenAI-compatible model listing."""
    with httpx.Client(
        base_url=f"http://127.0.0.1:{cluster['proxy_port']}",
        timeout=10,
        trust_env=False,
    ) as c:
        r = c.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"


def test_chat_completions(cluster):
    """Non-streaming chat completions through proxy."""
    payload = {**CHAT_PAYLOAD, "model": cluster["model"]}
    with httpx.Client(
        base_url=f"http://127.0.0.1:{cluster['proxy_port']}",
        timeout=30,
        trust_env=False,
    ) as c:
        r = c.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]


def test_chat_completions_streaming(cluster):
    """Streaming chat completions through proxy."""
    payload = {**CHAT_PAYLOAD, "model": cluster["model"], "stream": True}
    with httpx.Client(
        base_url=f"http://127.0.0.1:{cluster['proxy_port']}",
        timeout=30,
        trust_env=False,
    ) as c:
        r = c.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        lines = r.text.strip().split("\n")
        data_lines = [ln for ln in lines if ln.startswith("data: ")]
        assert len(data_lines) >= 2
        assert data_lines[-1] == "data: [DONE]"


def test_status_topology(cluster):
    """Proxy status should reflect correct topology."""
    with httpx.Client(
        base_url=f"http://127.0.0.1:{cluster['proxy_port']}",
        timeout=10,
        trust_env=False,
    ) as c:
        r = c.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["prefill_node_count"] == NUM_PREFILL
        assert data["decode_node_count"] == NUM_DECODE


def test_concurrent_requests(cluster):
    """Multiple concurrent requests should all succeed."""
    payload = {**CHAT_PAYLOAD, "model": cluster["model"]}

    def send_request(idx):
        with httpx.Client(
            base_url=f"http://127.0.0.1:{cluster['proxy_port']}",
            timeout=30,
            trust_env=False,
        ) as c:
            r = c.post("/v1/chat/completions", json=payload)
            return r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(send_request, i) for i in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(code == 200 for code in results), f"Some requests failed: {results}"
