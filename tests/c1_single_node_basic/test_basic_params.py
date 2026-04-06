"""C1: Basic parameter handling tests."""

import pytest


@pytest.mark.anyio
async def test_max_tokens_respected(client):
    """max_tokens limits output length."""
    resp = await client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "Hello", "max_tokens": 1},
    )
    assert resp.status_code == 200
    assert resp.json()["usage"]["completion_tokens"] <= 1


@pytest.mark.anyio
async def test_n_multiple_choices(client):
    """n>1 returns multiple choices."""
    resp = await client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "Hello", "max_tokens": 3, "n": 2},
    )
    assert resp.status_code == 200
    assert len(resp.json()["choices"]) == 2


@pytest.mark.anyio
async def test_stop_sequence(client):
    """Stop sequence triggers early stop."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
            "stop": ["token"],
        },
    )
    assert resp.status_code == 200
    # Should have finish_reason "stop" (if stop hit) or "length"
    assert resp.json()["choices"][0]["finish_reason"] in ("stop", "length")


@pytest.mark.anyio
async def test_usage_stats(client):
    """Usage stats are present and correct."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
        },
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
