"""PD Disaggregation tests (TC8.1-TC8.4) — migrated from xPyD-sim."""

from __future__ import annotations

import time

import httpx
import pytest
from httpx import ASGITransport

from xpyd_sim.server import ServerConfig, create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_body(**overrides):
    body = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
    }
    body.update(overrides)
    return body


def _make_completion_body(**overrides):
    body = {"model": "dummy", "prompt": "hello", "max_tokens": 8}
    body.update(overrides)
    return body


async def _collect_stream(resp: httpx.Response) -> list[str]:
    """Collect SSE data lines from a streaming response."""
    lines = []
    async for line in resp.aiter_lines():
        if line.startswith("data: "):
            lines.append(line[6:])
    return lines


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prefill_config():
    """Config for prefill-only node."""
    return ServerConfig(
        mode="prefill",
        prefill_delay_ms=20,
        kv_transfer_delay_ms=10,
        decode_delay_per_token_ms=5,
        eos_min_ratio=1.0,
    )


@pytest.fixture()
def decode_config():
    """Config for decode-only node."""
    return ServerConfig(
        mode="decode",
        prefill_delay_ms=20,
        kv_transfer_delay_ms=10,
        decode_delay_per_token_ms=5,
        eos_min_ratio=1.0,
    )


# ---------------------------------------------------------------------------
# TC8.1 — Simulated PD disaggregation flow (prefill + decode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc8_1_pd_disaggregation_flow(prefill_config, decode_config):
    """Simulate PD flow: prefill node (max_tokens=1) → decode node (full)."""
    prefill_app = create_app(prefill_config)
    decode_app = create_app(decode_config)
    prefill_transport = ASGITransport(app=prefill_app)
    decode_transport = ASGITransport(app=decode_app)

    max_tokens = 8

    # Step 1: Send to prefill with max_tokens=1
    async with httpx.AsyncClient(
        transport=prefill_transport, base_url="http://prefill"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_make_chat_body(max_tokens=1),
        )
        assert resp.status_code == 200
        prefill_data = resp.json()
        assert prefill_data["usage"]["completion_tokens"] >= 1

    # Step 2: Send full request to decode
    async with httpx.AsyncClient(
        transport=decode_transport, base_url="http://decode"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=_make_chat_body(max_tokens=max_tokens),
        )
        assert resp.status_code == 200
        decode_data = resp.json()
        assert decode_data["usage"]["completion_tokens"] == max_tokens


# ---------------------------------------------------------------------------
# TC8.2 — TTFT validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc8_2_ttft_validation():
    """TTFT ≈ prefill_delay + kv_transfer_delay + first decode token delay."""
    prefill_ms = 30
    decode_ms = 10

    config = ServerConfig(
        mode="dual",
        prefill_delay_ms=prefill_ms,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=decode_ms,
        eos_min_ratio=1.0,
    )
    app = create_app(config)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.monotonic()
        resp = await client.post(
            "/v1/chat/completions",
            json=_make_chat_body(max_tokens=4, stream=True),
        )
        first_line = None
        second_line = None
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                if first_line is None:
                    first_line = line
                elif second_line is None:
                    second_line = line
                    ttft = time.monotonic() - start
                    break

        expected_min = (prefill_ms + decode_ms) / 1000.0
        assert ttft >= expected_min * 0.5, (
            f"TTFT too fast: {ttft:.3f}s, expected >= ~{expected_min:.3f}s"
        )
        assert ttft < expected_min * 3.0, (
            f"TTFT too slow: {ttft:.3f}s, expected ~{expected_min:.3f}s"
        )


# ---------------------------------------------------------------------------
# TC8.3 — TPOT validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc8_3_tpot_validation():
    """TPOT ≈ decode_delay_per_token (measured via total decode time)."""
    decode_ms = 15
    max_tokens = 6
    config = ServerConfig(
        mode="dual",
        prefill_delay_ms=1,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=decode_ms,
        eos_min_ratio=1.0,
    )
    app = create_app(config)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.monotonic()
        resp = await client.post(
            "/v1/chat/completions",
            json=_make_chat_body(max_tokens=max_tokens),
        )
        elapsed = time.monotonic() - start
        assert resp.status_code == 200
        data = resp.json()
        n_tokens = data["usage"]["completion_tokens"]

        expected_decode_s = n_tokens * decode_ms / 1000.0
        assert elapsed >= expected_decode_s * 0.3, (
            f"Total time too fast: {elapsed*1000:.1f}ms for {n_tokens} tokens"
        )


# ---------------------------------------------------------------------------
# TC8.4 — Streaming token intervals match configured delay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc8_4_streaming_token_intervals():
    """Streaming total time should reflect decode delay * token count."""
    decode_ms = 20
    max_tokens = 8
    config = ServerConfig(
        mode="dual",
        prefill_delay_ms=1,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=decode_ms,
        eos_min_ratio=1.0,
    )
    app = create_app(config)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.monotonic()
        resp = await client.post(
            "/v1/completions",
            json=_make_completion_body(max_tokens=max_tokens, stream=True),
        )
        lines = await _collect_stream(resp)
        elapsed = time.monotonic() - start

        assert lines[-1] == "[DONE]"
        expected_min = (max_tokens * decode_ms) / 1000.0
        assert elapsed >= expected_min * 0.3, (
            f"Stream too fast: {elapsed*1000:.1f}ms, expected >= ~{expected_min*1000:.1f}ms"
        )
