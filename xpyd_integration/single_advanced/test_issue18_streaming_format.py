"""Tests for issue #18: Incomplete streaming response format in dummy server.

Verifies:
- Chat streaming first chunk includes role: assistant in delta
- logprobs: null present in all streaming choice objects when not requested
- Completions streaming includes logprobs: null when not requested
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def app():
    config = ServerConfig(
        prefill_delay_ms=0,
        decode_delay_per_token_ms=0,
eos_min_ratio=1.0,
    )
    return create_app(config)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _collect_sse_chunks(resp) -> list[dict]:
    chunks = []
    async for line in resp.aiter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


@pytest.mark.asyncio
async def test_chat_stream_first_chunk_has_role(client: AsyncClient):
    """First chat streaming chunk should have role: assistant in delta."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    chunks = await _collect_sse_chunks(resp)
    assert len(chunks) >= 2  # at least role chunk + content chunks
    first = chunks[0]
    delta = first["choices"][0]["delta"]
    assert delta.get("role") == "assistant"


@pytest.mark.asyncio
async def test_chat_stream_logprobs_null_when_not_requested(client: AsyncClient):
    """All chat streaming choices should have logprobs: null when not requested."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    chunks = await _collect_sse_chunks(resp)
    for chunk in chunks:
        if chunk["choices"]:
            choice = chunk["choices"][0]
            assert "logprobs" in choice, f"Missing logprobs key in chunk: {chunk}"
            assert choice["logprobs"] is None


@pytest.mark.asyncio
async def test_completions_stream_logprobs_null_when_not_requested(
    client: AsyncClient,
):
    """All completions streaming choices should have logprobs: null when not requested."""
    resp = await client.post(
        "/v1/completions",
        json={
            "model": "dummy",
            "prompt": "hello",
            "max_tokens": 3,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    chunks = await _collect_sse_chunks(resp)
    for chunk in chunks:
        if chunk["choices"]:
            choice = chunk["choices"][0]
            assert "logprobs" in choice, f"Missing logprobs key in chunk: {chunk}"
            assert choice["logprobs"] is None


@pytest.mark.asyncio
async def test_chat_stream_logprobs_present_when_requested(client: AsyncClient):
    """Chat streaming should include logprobs data when logprobs=true and top_logprobs set."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "stream": True,
            "logprobs": True,
            "top_logprobs": 3,
        },
    )
    assert resp.status_code == 200
    chunks = await _collect_sse_chunks(resp)
    # Skip the role-only chunk (first), check content chunks
    content_chunks = [
        c for c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")
    ]
    assert len(content_chunks) >= 1
    for chunk in content_chunks:
        lp = chunk["choices"][0]["logprobs"]
        assert lp is not None
        assert "content" in lp
        assert len(lp["content"]) == 1
        assert len(lp["content"][0]["top_logprobs"]) == 3
