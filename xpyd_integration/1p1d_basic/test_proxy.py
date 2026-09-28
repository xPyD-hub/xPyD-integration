"""Tests for proxy routing, scheduling, streaming."""

import json
from unittest.mock import patch

import pytest
from httpx import AsyncClient

from xpyd.scheduler import (
    Candidate,
    LoadBalancedScheduler,
    RoundRobinSchedulingPolicy,
    Scheduler,
    SchedulingContext,
)


CHAT_PAYLOAD = {
    "model": "dummy",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5,
    "stream": False,
}


@pytest.mark.anyio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) > 0
    for _inst, info in data.items():
        assert info["status"] == 200
        assert info["data"]["status"] == "ok"


@pytest.mark.anyio
async def test_status(client: AsyncClient):
    resp = await client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["prefill_node_count"] == 1
    assert data["decode_node_count"] == 1
    assert len(data["prefill_nodes"]) == 1
    assert len(data["decode_nodes"]) == 1


@pytest.mark.anyio
async def test_non_streaming(client: AsyncClient):
    resp = await client.post("/v1/chat/completions", json=CHAT_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["finish_reason"] in ("stop", "length")
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0

    assert data["usage"]["completion_tokens"] == 5
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + 5


@pytest.mark.anyio
async def test_streaming(client: AsyncClient):
    payload = {**CHAT_PAYLOAD, "stream": True}
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]

    assert len(data_lines) >= 4

    assert data_lines[-1] == "data: [DONE]"

    first = json.loads(data_lines[0].removeprefix("data: "))
    assert first["choices"][0]["delta"]["role"] == "assistant"

    content = ""
    for line in data_lines[1:-2]:
        chunk = json.loads(line.removeprefix("data: "))
        content += chunk["choices"][0]["delta"]["content"]
    assert len(content) > 0


@pytest.mark.anyio
async def test_max_tokens_respected(client: AsyncClient):
    payload = {**CHAT_PAYLOAD, "max_tokens": 3, "stream": False}
    resp = await client.post("/v1/chat/completions", json=payload)
    data = resp.json()
    assert data["usage"]["completion_tokens"] == 3


@pytest.mark.anyio
async def test_streaming_token_count(client: AsyncClient):
    payload = {**CHAT_PAYLOAD, "max_tokens": 7, "stream": True}
    resp = await client.post("/v1/chat/completions", json=payload)

    lines = resp.text.strip().split("\n")
    data_lines = [
        line for line in lines if line.startswith("data: ") and line != "data: [DONE]"
    ]

    content_chunks = 0
    for line in data_lines:
        chunk = json.loads(line.removeprefix("data: "))
        delta = chunk["choices"][0]["delta"]
        if delta.get("content") is not None:
            content_chunks += 1

    assert content_chunks >= 1


def test_round_robin_scheduling():
    policy = RoundRobinSchedulingPolicy()
    instances = ["a:1", "b:2", "c:3"]
    candidates = [Candidate(address) for address in instances]
    context = SchedulingContext(role="decode")
    results = [policy.select_node(context, candidates) for _ in range(6)]
    assert results == ["a:1", "b:2", "c:3", "a:1", "b:2", "c:3"]


def test_round_robin_context_keeps_role_positions_independent():
    policy = RoundRobinSchedulingPolicy()
    instances = ["a:1", "b:2"]
    candidates = [Candidate(address) for address in instances]
    for role in ("prefill", "decode", "aggregated"):
        context = SchedulingContext(role=role, request_len=100, max_tokens=50)
        assert policy.select_node(context, candidates) == "a:1"
        assert policy.select_node(context, candidates) == "b:2"


def test_round_robin_reservation_release_is_idempotent():
    policy = RoundRobinSchedulingPolicy()
    runtime = Scheduler()
    context = SchedulingContext(role="decode", request_len=100)
    lease = runtime.reserve(policy, context, ["a:1"])
    assert lease.address == "a:1"
    assert runtime._active["a:1"] == 1
    lease.release()
    lease.release()
    assert not runtime._active


@patch(
    "xpyd.scheduler.load_balanced.query_instance_model_len",
    return_value=[131072, 131072],
)
def test_load_balanced_scheduling(mock_query):
    prefill = ["p1:1", "p2:2"]
    decode = ["d1:1", "d2:2"]
    policy = LoadBalancedScheduler(prefill, decode)

    runtime = Scheduler()
    p_context = SchedulingContext(role="prefill", request_len=100, max_tokens=50)
    d_context = SchedulingContext(role="decode", request_len=50, max_tokens=50)
    p_leases = [runtime.reserve(policy, p_context, prefill) for _ in range(2)]
    d_leases = [runtime.reserve(policy, d_context, decode) for _ in range(2)]
    assert {lease.address for lease in p_leases} == set(prefill)
    assert {lease.address for lease in d_leases} == set(decode)
    assert policy.prefill_bs_counter == policy.decode_bs_counter == [1, 1]
    for lease in p_leases + d_leases:
        lease.release()
    assert policy.prefill_bs_counter == policy.decode_bs_counter == [0, 0]
    assert policy.prefill_utils_counter == policy.decode_kv_utils_counter == [0, 0]
