"""Tests for issue #23: Realistic EOS behavior with random output length."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def app():
    """Create app with fast timings for testing."""
    config = ServerConfig(prefill_ms=0, decode_ms=0, eos_min_ratio=0.5)
    return create_app(config)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _parse_sse_chunks(text: str) -> list[dict]:
    """Parse SSE response into list of JSON objects."""
    chunks = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


# ── Non-streaming completions ────────────────────────────────────────────


@pytest.mark.anyio
async def test_completions_eos_before_max_tokens(client: AsyncClient):
    """EOS should fire before max_tokens, producing finish_reason='stop'."""
    results = {"stop": 0, "length": 0}
    # Run multiple times to observe randomness
    for _ in range(50):
        resp = await client.post(
            "/v1/completions",
            json={"model": "test", "prompt": "hello", "max_tokens": 20},
        )
        assert resp.status_code == 200
        body = resp.json()
        fr = body["choices"][0]["finish_reason"]
        assert fr in ("stop", "length")
        results[fr] += 1
        tokens = body["choices"][0]["text"].split()
        assert len(tokens) >= 10  # eos_min_ratio=0.5 → at least 10 of 20
        assert len(tokens) <= 20

    # Should see a mix of stop and length over 50 runs
    assert results["stop"] > 0, "Expected at least some EOS stops"


@pytest.mark.anyio
async def test_completions_ignore_eos(client: AsyncClient):
    """ignore_eos=true should always produce max_tokens and finish_reason='length'."""
    for _ in range(10):
        resp = await client.post(
            "/v1/completions",
            json={
                "model": "test",
                "prompt": "hello",
                "max_tokens": 20,
                "ignore_eos": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["finish_reason"] == "length"
        tokens = body["choices"][0]["text"].split()
        assert len(tokens) == 20


# ── Non-streaming chat completions ───────────────────────────────────────


@pytest.mark.anyio
async def test_chat_eos_before_max_tokens(client: AsyncClient):
    """Chat completions should also produce EOS stops."""
    results = {"stop": 0, "length": 0}
    for _ in range(50):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 20,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        fr = body["choices"][0]["finish_reason"]
        assert fr in ("stop", "length")
        results[fr] += 1
        tokens = body["choices"][0]["message"]["content"].split()
        assert len(tokens) >= 10
        assert len(tokens) <= 20

    assert results["stop"] > 0


@pytest.mark.anyio
async def test_chat_ignore_eos(client: AsyncClient):
    """Chat with ignore_eos=true should always produce max_tokens."""
    for _ in range(10):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 20,
                "ignore_eos": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["finish_reason"] == "length"
        tokens = body["choices"][0]["message"]["content"].split()
        assert len(tokens) == 20


# ── Streaming completions ────────────────────────────────────────────────


@pytest.mark.anyio
async def test_stream_completions_eos(client: AsyncClient):
    """Streaming completions should emit EOS with finish_reason='stop'."""
    saw_stop = False
    for _ in range(30):
        resp = await client.post(
            "/v1/completions",
            json={
                "model": "test",
                "prompt": "hello",
                "max_tokens": 20,
                "stream": True,
            },
        )
        chunks = _parse_sse_chunks(resp.text)
        last = chunks[-1]
        fr = last["choices"][0]["finish_reason"]
        assert fr in ("stop", "length")
        if fr == "stop":
            saw_stop = True
            assert len(chunks) <= 21  # Fewer chunks than max_tokens (+ possible finish chunk)

    assert saw_stop, "Expected at least one streaming EOS stop"


@pytest.mark.anyio
async def test_stream_completions_ignore_eos(client: AsyncClient):
    """Streaming completions with ignore_eos should produce all tokens."""
    for _ in range(5):
        resp = await client.post(
            "/v1/completions",
            json={
                "model": "test",
                "prompt": "hello",
                "max_tokens": 10,
                "stream": True,
                "ignore_eos": True,
            },
        )
        chunks = _parse_sse_chunks(resp.text)
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] == "length"
        assert len(chunks) >= 10


# ── Streaming chat completions ───────────────────────────────────────────


@pytest.mark.anyio
async def test_stream_chat_eos(client: AsyncClient):
    """Streaming chat completions should emit EOS stops."""
    saw_stop = False
    for _ in range(30):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 20,
                "stream": True,
            },
        )
        chunks = _parse_sse_chunks(resp.text)
        # First chunk is role chunk, skip it
        content_chunks = [
            c for c in chunks
            if c["choices"][0].get("delta", {}).get("content") is not None
            or c["choices"][0].get("finish_reason")
        ]
        last = content_chunks[-1]
        fr = last["choices"][0]["finish_reason"]
        assert fr in ("stop", "length")
        if fr == "stop":
            saw_stop = True

    assert saw_stop, "Expected at least one streaming chat EOS stop"


@pytest.mark.anyio
async def test_stream_chat_ignore_eos(client: AsyncClient):
    """Streaming chat with ignore_eos should produce all tokens."""
    for _ in range(5):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "stream": True,
                "ignore_eos": True,
            },
        )
        chunks = _parse_sse_chunks(resp.text)
        # First chunk is role-only, skip it; remaining are content chunks
        content_chunks = chunks[1:]
        assert len(content_chunks) >= 10
        last_content = content_chunks[-1]
        assert last_content["choices"][0]["finish_reason"] == "length"


# ── ServerConfig.eos_min_ratio ───────────────────────────────────────────


@pytest.mark.anyio
async def test_eos_min_ratio_respected():
    """eos_min_ratio should control the minimum output length."""
    config = ServerConfig(prefill_ms=0, decode_ms=0, eos_min_ratio=0.8)
    app = create_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(30):
            resp = await client.post(
                "/v1/completions",
                json={"model": "test", "prompt": "hello", "max_tokens": 20},
            )
            body = resp.json()
            tokens = body["choices"][0]["text"].split()
            # With eos_min_ratio=0.8, minimum should be 16 tokens
            assert len(tokens) >= 16
