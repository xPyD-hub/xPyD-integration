"""End-to-end benchmark: 1000 concurrent clients, 10000 requests, mixed lengths.

Migrated from xPyD-proxy/tests/stress/test_benchmark_e2e.py
Original: 4 tests.

Topology: 2 prefill + 16 decode + 1 proxy.
Excluded from CI. Run manually:

    pytest xpyd_integration/1p1d_concurrent_stress/test_benchmark_e2e.py -v -s
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
import pytest

TOTAL_REQUESTS = 100
MAX_CONCURRENCY = 10


def _random_content(length: int) -> str:
    if length <= 0:
        return ""
    words = ["hello", "world", "bench", "test", "proxy", "stream", "token", "data"]
    pieces: list[str] = []
    cur = 0
    while cur < length:
        w = random.choice(words)
        pieces.append(w)
        cur += len(w) + 1
    return " ".join(pieces)[:length]


def _build_payload(model: str, stream: bool) -> dict[str, Any]:
    """Build a chat completion payload with random prompt length 0-10k chars."""
    prompt_len = random.randint(0, 10_000)
    content = _random_content(prompt_len)
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": random.randint(1, 64),
        "stream": stream,
    }


def _send_non_streaming(base_url: str, payload: dict) -> dict:
    t0 = time.monotonic()
    with httpx.Client(base_url=base_url, timeout=60, trust_env=False) as c:
        r = c.post("/v1/chat/completions", json=payload)
    elapsed = time.monotonic() - t0
    return {"status": r.status_code, "elapsed": elapsed, "stream": False}


def _send_streaming(base_url: str, payload: dict) -> dict:
    t0 = time.monotonic()
    chunks = 0
    status = 0
    with httpx.Client(base_url=base_url, timeout=60, trust_env=False) as c:
        with c.stream("POST", "/v1/chat/completions", json=payload) as r:
            status = r.status_code
            for line in r.iter_lines():
                if line.startswith("data: "):
                    chunks += 1
    elapsed = time.monotonic() - t0
    return {"status": status, "elapsed": elapsed, "stream": True, "chunks": chunks}


def _send_request(base_url: str, model: str, idx: int) -> dict:
    stream = random.choice([True, False])
    payload = _build_payload(model, stream=stream)
    try:
        if stream:
            return _send_streaming(base_url, payload)
        return _send_non_streaming(base_url, payload)
    except Exception as exc:
        return {"status": -1, "error": str(exc), "stream": stream, "elapsed": 0}


@pytest.mark.benchmark
def test_benchmark_10k_mixed(cluster):
    """Fire 10000 mixed requests at 1000 concurrency."""
    base_url = f"http://127.0.0.1:{cluster['proxy_port']}"
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [
            pool.submit(_send_request, base_url, cluster["model"], i)
            for i in range(TOTAL_REQUESTS)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    statuses = [r["status"] for r in results]
    success = statuses.count(200)
    failed = len(statuses) - success
    errors = [r for r in results if r["status"] != 200]

    elapsed_all = sorted(r["elapsed"] for r in results if r["status"] == 200)
    stream_count = sum(1 for r in results if r.get("stream"))
    non_stream_count = len(results) - stream_count

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total requests : {TOTAL_REQUESTS}")
    print(f"Concurrency    : {MAX_CONCURRENCY}")
    print(f"Streaming      : {stream_count}")
    print(f"Non-streaming  : {non_stream_count}")
    print(f"Successful     : {success}")
    print(f"Failed         : {failed}")
    if elapsed_all:
        print(f"Latency p50    : {elapsed_all[len(elapsed_all) // 2]:.3f}s")
        print(f"Latency p90    : {elapsed_all[int(len(elapsed_all) * 0.9)]:.3f}s")
        print(f"Latency p99    : {elapsed_all[int(len(elapsed_all) * 0.99)]:.3f}s")
        print(f"Latency max    : {elapsed_all[-1]:.3f}s")
    print("=" * 60)

    if errors:
        print(f"First {min(5, len(errors))} errors: {errors[:5]}")

    assert failed == 0, f"{failed}/{TOTAL_REQUESTS} requests failed"


@pytest.mark.benchmark
def test_benchmark_streaming_only(cluster):
    """1000 concurrent streaming requests to verify SSE under load."""
    base_url = f"http://127.0.0.1:{cluster['proxy_port']}"
    count = 50

    def send(idx: int) -> dict:
        payload = _build_payload(cluster["model"], stream=True)
        return _send_streaming(base_url, payload)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [pool.submit(send, i) for i in range(count)]
        for f in as_completed(futures):
            results.append(f.result())

    success = sum(1 for r in results if r["status"] == 200)
    has_chunks = sum(1 for r in results if r.get("chunks", 0) >= 2)

    print(f"\nStreaming-only: {success}/{count} OK, {has_chunks} with >=2 chunks")
    assert success == count, f"{count - success} streaming requests failed"
    assert has_chunks == count, "Some streaming responses had fewer than 2 chunks"


@pytest.mark.benchmark
def test_benchmark_burst_short_prompts(cluster):
    """Burst of 5000 short-prompt requests at full concurrency."""
    base_url = f"http://127.0.0.1:{cluster['proxy_port']}"
    count = 50

    def send(idx: int) -> dict:
        payload = {
            "model": cluster["model"],
            "messages": [
                {"role": "user", "content": _random_content(random.randint(0, 100))}
            ],
            "max_tokens": 5,
            "stream": False,
        }
        return _send_non_streaming(base_url, payload)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [pool.submit(send, i) for i in range(count)]
        for f in as_completed(futures):
            results.append(f.result())

    success = sum(1 for r in results if r["status"] == 200)
    elapsed = sorted(r["elapsed"] for r in results if r["status"] == 200)
    if elapsed:
        print(
            f"\nShort burst: {success}/{count} OK, "
            f"p50={elapsed[len(elapsed) // 2]:.3f}s, "
            f"p99={elapsed[int(len(elapsed) * 0.99)]:.3f}s"
        )
    assert success == count, f"{count - success} short-burst requests failed"


@pytest.mark.benchmark
def test_benchmark_long_prompts(cluster):
    """500 requests with long prompts (5k-10k chars) at moderate concurrency."""
    base_url = f"http://127.0.0.1:{cluster['proxy_port']}"
    count = 50
    concurrency = 200

    def send(idx: int) -> dict:
        payload = {
            "model": cluster["model"],
            "messages": [
                {
                    "role": "user",
                    "content": _random_content(random.randint(100, 500)),
                }
            ],
            "max_tokens": 32,
            "stream": random.choice([True, False]),
        }
        if payload["stream"]:
            return _send_streaming(base_url, payload)
        return _send_non_streaming(base_url, payload)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send, i) for i in range(count)]
        for f in as_completed(futures):
            results.append(f.result())

    success = sum(1 for r in results if r["status"] == 200)
    elapsed = sorted(r["elapsed"] for r in results if r["status"] == 200)
    if elapsed:
        print(
            f"\nLong prompts: {success}/{count} OK, "
            f"p50={elapsed[len(elapsed) // 2]:.3f}s, "
            f"p99={elapsed[int(len(elapsed) * 0.99)]:.3f}s"
        )
    assert success == count, f"{count - success} long-prompt requests failed"
