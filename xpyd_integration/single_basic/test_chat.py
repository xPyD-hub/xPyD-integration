"""C1: Basic /v1/chat/completions tests."""

import json
import pytest


@pytest.mark.anyio
async def test_chat_non_streaming(client):
    """Basic non-streaming chat returns valid response."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] in ("stop", "length")


@pytest.mark.anyio
async def test_chat_streaming(client):
    """Basic streaming chat returns role chunk + content chunks + finish."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 3,
            "stream": True,
        },
    )
    assert resp.status_code == 200

    chunks = []
    for line in resp.text.split("\n"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    assert len(chunks) >= 2  # at least role + content

    # First chunk should have role
    first_delta = chunks[0]["choices"][0]["delta"]
    assert first_delta.get("role") == "assistant"

    # Should end with finish_reason
    last_with_choices = [c for c in chunks if c.get("choices")]
    assert any(
        c["choices"][0].get("finish_reason") is not None
        for c in last_with_choices
    )
