"""Tests for concurrent requests (ASGI transport, 1P1D topology).

Migrated from xPyD-proxy/tests/integration/test_concurrent_requests.py
Original: 3 tests.
"""

import asyncio
import json

import pytest
from httpx import AsyncClient


CHAT_PAYLOAD = {
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5,
    "stream": False,
}


@pytest.mark.anyio
async def test_concurrent_chat_completions(client: AsyncClient):
    """15 concurrent non-streaming requests should all succeed with unique ids."""
    concurrency = 15
    payloads = [
        {**CHAT_PAYLOAD, "messages": [{"role": "user", "content": f"Hello {idx}"}]}
        for idx in range(concurrency)
    ]

    tasks = [client.post("/v1/chat/completions", json=p) for p in payloads]
    responses = await asyncio.gather(*tasks)

    ids = set()
    for resp in responses:
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) >= 1
        ids.add(data["id"])

    assert len(ids) == concurrency


@pytest.mark.anyio
async def test_concurrent_streaming(client: AsyncClient):
    """10 concurrent streaming requests should all produce valid SSE."""
    concurrency = 10
    payload = {**CHAT_PAYLOAD, "stream": True}

    tasks = [
        client.post("/v1/chat/completions", json=payload) for _ in range(concurrency)
    ]
    responses = await asyncio.gather(*tasks)

    for resp in responses:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) >= 2
        assert data_lines[-1] == "data: [DONE]"


@pytest.mark.anyio
async def test_mixed_concurrent_streaming_and_non_streaming(client: AsyncClient):
    """Streaming and non-streaming requests running concurrently."""
    non_stream_tasks = [
        client.post(
            "/v1/chat/completions",
            json={
                **CHAT_PAYLOAD,
                "messages": [{"role": "user", "content": f"ns-{idx}"}],
            },
        )
        for idx in range(8)
    ]
    stream_tasks = [
        client.post(
            "/v1/chat/completions",
            json={
                **CHAT_PAYLOAD,
                "stream": True,
                "messages": [{"role": "user", "content": f"s-{idx}"}],
            },
        )
        for idx in range(8)
    ]

    responses = await asyncio.gather(*non_stream_tasks, *stream_tasks)
    non_stream_responses = responses[:8]
    stream_responses = responses[8:]

    ns_ids = set()
    for resp in non_stream_responses:
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["usage"]["completion_tokens"] == CHAT_PAYLOAD["max_tokens"]
        ns_ids.add(data["id"])

    assert len(ns_ids) == 8, "Non-streaming response ids must be unique"

    for resp in stream_responses:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert data_lines[-1] == "data: [DONE]"
        for line in data_lines[:-1]:
            chunk = json.loads(line.removeprefix("data: "))
            assert chunk["object"] == "chat.completion.chunk"
