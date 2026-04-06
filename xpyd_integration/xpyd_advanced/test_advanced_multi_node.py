"""Advanced multi-node tests for xPyD (3P + 3D + 1 proxy).

Tests:
1. test_streaming_through_proxy - streaming correctness across multi-node
2. test_large_prompt_multi_node - long prompt (~500 words) through multi-node
3. test_all_nodes_serving - 30 requests all succeed
4. test_mixed_streaming_nonstreaming - concurrent mixed stream/non-stream
5. test_error_handling_invalid_model - invalid model returns proper error
"""

from __future__ import annotations

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
async def test_streaming_through_proxy(client: AsyncClient):
    """Streaming works correctly through multi-node proxy (3P+3D)."""
    payload = {**CHAT_PAYLOAD, "stream": True, "max_tokens": 10}
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) >= 2, f"Expected at least 2 data lines, got {len(data_lines)}"
    assert data_lines[-1] == "data: [DONE]"

    # Verify first chunk has role
    first = json.loads(data_lines[0].removeprefix("data: "))
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"

    # Verify content chunks exist
    content = ""
    for line in data_lines[1:-1]:
        if line == "data: [DONE]":
            continue
        chunk = json.loads(line.removeprefix("data: "))
        c = chunk["choices"][0]["delta"].get("content", "")
        if c:
            content += c
    assert len(content) > 0, "Expected non-empty streamed content"

    # Verify all chunk IDs are consistent
    ids = set()
    for line in data_lines:
        if line == "data: [DONE]":
            continue
        chunk = json.loads(line.removeprefix("data: "))
        ids.add(chunk["id"])
    assert len(ids) == 1, f"Expected consistent chunk IDs, got {ids}"


@pytest.mark.anyio
async def test_large_prompt_multi_node(client: AsyncClient):
    """A large prompt (~500 words) is handled correctly through 3P+3D."""
    # Generate ~500 words
    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    large_content = " ".join(words * 56)  # 9 * 56 = 504 words

    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": large_content}],
        "max_tokens": 10,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) >= 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0
    assert data["usage"]["prompt_tokens"] > 100, "Large prompt should tokenize to many tokens"
    assert data["usage"]["completion_tokens"] == 10


@pytest.mark.anyio
async def test_all_nodes_serving(client: AsyncClient):
    """Send 30 requests; all must succeed (200)."""
    tasks = []
    for _ in range(30):
        tasks.append(client.post("/v1/chat/completions", json=CHAT_PAYLOAD))

    responses = await asyncio.gather(*tasks)

    for i, resp in enumerate(responses):
        assert resp.status_code == 200, f"Request {i} failed with {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) >= 1


@pytest.mark.anyio
async def test_mixed_streaming_nonstreaming(client: AsyncClient):
    """Concurrent mix of streaming and non-streaming requests all succeed."""
    tasks = []
    for i in range(20):
        stream = i % 2 == 0
        payload = {**CHAT_PAYLOAD, "stream": stream, "max_tokens": 5}
        tasks.append((stream, client.post("/v1/chat/completions", json=payload)))

    results = await asyncio.gather(*[t[1] for t in tasks])

    for i, resp in enumerate(results):
        stream = tasks[i][0]
        assert resp.status_code == 200, f"Request {i} (stream={stream}) failed: {resp.text}"

        if stream:
            lines = resp.text.strip().split("\n")
            data_lines = [l for l in lines if l.startswith("data: ")]
            assert data_lines[-1] == "data: [DONE]"
        else:
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert len(data["choices"]) >= 1


@pytest.mark.anyio
async def test_error_handling_invalid_model(client: AsyncClient):
    """Invalid model name should return an appropriate error (not 500)."""
    payload = {
        "model": "nonexistent-model-that-does-not-exist",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    # Proxy may return 200 (ignoring model) or 4xx error — either is acceptable
    # but it must NOT return 500 (internal server error)
    assert resp.status_code != 500, f"Got 500 error for invalid model: {resp.text}"
    # If it returns 200, it means proxy ignores the model field (valid behavior)
    # If it returns 4xx, verify it has an error structure
    if resp.status_code >= 400:
        data = resp.json()
        assert "error" in data or "detail" in data
