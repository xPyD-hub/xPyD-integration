"""C1: Basic /v1/completions tests."""

import json
import pytest


@pytest.mark.anyio
async def test_completions_non_streaming(client):
    """Basic non-streaming completion returns valid response."""
    resp = await client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "Hello", "max_tokens": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["text"]
    assert data["choices"][0]["finish_reason"] in ("stop", "length")
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


@pytest.mark.anyio
async def test_completions_streaming(client):
    """Basic streaming completion returns valid SSE chunks."""
    resp = await client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "Hello", "max_tokens": 3, "stream": True},
    )
    assert resp.status_code == 200

    chunks = []
    for line in resp.text.split("\n"):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[6:]))

    assert len(chunks) >= 1
    # Last content chunk should have finish_reason
    last_with_choices = [c for c in chunks if c.get("choices")]
    assert any(
        c["choices"][0].get("finish_reason") is not None
        for c in last_with_choices
    )
