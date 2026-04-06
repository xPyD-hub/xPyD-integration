"""Single-node concurrent & stress tests."""

import asyncio
import json

import pytest
from httpx import AsyncClient


CHAT_PAYLOAD = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5,
    "stream": False,
}


@pytest.mark.anyio
async def test_concurrent_chat_completions(client: AsyncClient):
    """50 concurrent non-streaming chat requests should all succeed with unique ids."""
    concurrency = 50
    tasks = [
        client.post(
            "/v1/chat/completions",
            json={
                **CHAT_PAYLOAD,
                "messages": [{"role": "user", "content": f"Hello {i}"}],
            },
        )
        for i in range(concurrency)
    ]
    responses = await asyncio.gather(*tasks)

    ids = set()
    for resp in responses:
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"]
        ids.add(data["id"])

    assert len(ids) == concurrency, "All response ids must be unique"


@pytest.mark.anyio
async def test_concurrent_streaming(client: AsyncClient):
    """20 concurrent streaming requests should all produce valid SSE."""
    concurrency = 20
    payload = {**CHAT_PAYLOAD, "stream": True}

    tasks = [
        client.post("/v1/chat/completions", json=payload)
        for _ in range(concurrency)
    ]
    responses = await asyncio.gather(*tasks)

    for resp in responses:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 2, "Should have at least role chunk + DONE"
        assert data_lines[-1] == "data: [DONE]"
        # Verify all non-DONE chunks parse as valid JSON
        for line in data_lines[:-1]:
            chunk = json.loads(line.removeprefix("data: "))
            assert chunk["object"] == "chat.completion.chunk"


@pytest.mark.anyio
async def test_concurrent_mixed_endpoints(client: AsyncClient):
    """Chat + completions + embeddings running concurrently all succeed."""
    chat_tasks = [
        client.post(
            "/v1/chat/completions",
            json={
                **CHAT_PAYLOAD,
                "messages": [{"role": "user", "content": f"chat-{i}"}],
            },
        )
        for i in range(10)
    ]
    completion_tasks = [
        client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": f"complete-{i}", "max_tokens": 5},
        )
        for i in range(10)
    ]
    embedding_tasks = [
        client.post(
            "/v1/embeddings",
            json={"model": "test-model", "input": f"embed-{i}"},
        )
        for i in range(10)
    ]

    responses = await asyncio.gather(
        *chat_tasks, *completion_tasks, *embedding_tasks
    )
    chat_resps = responses[:10]
    completion_resps = responses[10:20]
    embedding_resps = responses[20:]

    for resp in chat_resps:
        assert resp.status_code == 200
        assert resp.json()["object"] == "chat.completion"

    for resp in completion_resps:
        assert resp.status_code == 200
        assert resp.json()["object"] == "text_completion"

    for resp in embedding_resps:
        assert resp.status_code == 200
        assert resp.json()["object"] == "list"
        assert len(resp.json()["data"]) == 1


@pytest.mark.anyio
async def test_rapid_sequential_requests(client: AsyncClient):
    """100 rapid sequential requests complete without state leakage."""
    ids = set()
    for i in range(100):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                **CHAT_PAYLOAD,
                "messages": [{"role": "user", "content": f"seq-{i}"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"]
        ids.add(data["id"])

    assert len(ids) == 100, "All 100 sequential response ids must be unique"


@pytest.mark.anyio
async def test_concurrent_n_gt_1(client: AsyncClient):
    """Concurrent requests with n>1 return correct number of choices."""
    n_values = [2, 3, 4, 5, 2]
    tasks = [
        client.post(
            "/v1/chat/completions",
            json={
                **CHAT_PAYLOAD,
                "n": n,
                "messages": [{"role": "user", "content": f"multi-{i}"}],
            },
        )
        for i, n in enumerate(n_values)
    ]
    responses = await asyncio.gather(*tasks)

    for resp, expected_n in zip(responses, n_values):
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == expected_n, (
            f"Expected {expected_n} choices, got {len(data['choices'])}"
        )
        # Each choice should have a unique index
        indices = [c["index"] for c in data["choices"]]
        assert indices == list(range(expected_n))
