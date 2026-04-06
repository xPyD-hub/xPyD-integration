"""Tests for streaming edge cases."""

import json

import pytest
from httpx import AsyncClient


CHAT_PAYLOAD = {
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5,
    "stream": True,
}


@pytest.mark.anyio
async def test_streaming_chunk_structure(client: AsyncClient):
    """Each SSE chunk (except [DONE]) should be valid JSON with expected fields."""
    resp = await client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [
        line for line in lines if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert len(data_lines) >= 1, "Expected at least one data chunk before [DONE]"

    chunk_ids = set()
    for line in data_lines:
        payload = line.removeprefix("data: ")
        chunk = json.loads(payload)
        assert "id" in chunk
        assert chunk["object"] == "chat.completion.chunk"
        assert "choices" in chunk
        assert len(chunk["choices"]) >= 1
        assert "delta" in chunk["choices"][0]
        chunk_ids.add(chunk["id"])

    assert len(chunk_ids) == 1, f"Expected one unique id across chunks, got {chunk_ids}"


@pytest.mark.anyio
async def test_streaming_max_tokens_one(client: AsyncClient):
    """Streaming with max_tokens=1 should produce exactly one content chunk + [DONE]."""
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Say something"}],
        "max_tokens": 1,
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    assert data_lines[-1] == "data: [DONE]"

    content_chunks = [line for line in data_lines if line != "data: [DONE]"]
    assert len(content_chunks) >= 1

    chunk = json.loads(content_chunks[0].removeprefix("data: "))
    assert chunk["object"] == "chat.completion.chunk"
    assert "delta" in chunk["choices"][0]
