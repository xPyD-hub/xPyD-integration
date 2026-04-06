"""Multi-node (2P+4D+proxy) concurrent stress tests."""

import asyncio
import json

import pytest


@pytest.mark.anyio
async def test_concurrent_50_requests(client):
    """50 concurrent chat requests through multi-node proxy."""
    async def send(i):
        resp = await client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": f"msg {i}"}],
            "max_tokens": 3,
        })
        return resp.status_code

    results = await asyncio.gather(*[send(i) for i in range(50)])
    assert all(r == 200 for r in results)


@pytest.mark.anyio
async def test_burst_requests(client):
    """100 requests fired rapidly, all should succeed."""
    async def send(i):
        resp = await client.post("/v1/completions", json={
            "model": "test-model", "prompt": f"hello {i}", "max_tokens": 2,
        })
        return resp.status_code

    results = await asyncio.gather(*[send(i) for i in range(100)])
    assert all(r == 200 for r in results)
    assert len(results) == 100


@pytest.mark.anyio
async def test_concurrent_streaming_multi_node(client):
    """20 concurrent streaming requests all complete with valid SSE."""
    async def send(i):
        resp = await client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": f"stream {i}"}],
            "max_tokens": 3, "stream": True,
        })
        assert resp.status_code == 200
        has_done = "[DONE]" in resp.text
        return has_done

    results = await asyncio.gather(*[send(i) for i in range(20)])
    assert all(results)


@pytest.mark.anyio
async def test_sustained_load(client):
    """200 sequential requests with no degradation."""
    for i in range(200):
        resp = await client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": f"sustained {i}"}],
            "max_tokens": 2,
        })
        assert resp.status_code == 200


@pytest.mark.anyio
async def test_mixed_endpoint_stress(client):
    """20 concurrent requests per endpoint type (chat + completions + embeddings)."""
    async def chat(i):
        return (await client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": f"c {i}"}],
            "max_tokens": 2,
        })).status_code

    async def completion(i):
        return (await client.post("/v1/completions", json={
            "model": "test-model", "prompt": f"p {i}", "max_tokens": 2,
        })).status_code

    async def embed(i):
        return (await client.post("/v1/embeddings", json={
            "model": "test-model", "input": f"e {i}",
        })).status_code

    tasks = ([chat(i) for i in range(20)]
             + [completion(i) for i in range(20)]
             + [embed(i) for i in range(20)])
    results = await asyncio.gather(*tasks)
    assert all(r == 200 for r in results)
