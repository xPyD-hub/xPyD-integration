"""Tests for M42: Request ID Tracking & Correlation."""

from __future__ import annotations

import asyncio
import csv
import io
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from xpyd_bench.bench.debug_log import DebugLogEntry
from xpyd_bench.bench.models import BenchmarkResult, RequestResult
from xpyd_bench.bench.runner import _generate_request_id, _send_request
from xpyd_sim.server import ServerConfig, create_app
from xpyd_bench.reporting.formats import export_per_request_csv

# ---------------------------------------------------------------------------
# Unit tests for _generate_request_id
# ---------------------------------------------------------------------------


class TestGenerateRequestId:
    def test_without_prefix(self):
        rid = _generate_request_id()
        assert len(rid) == 32  # uuid4 hex
        assert rid.isalnum()

    def test_with_prefix(self):
        rid = _generate_request_id("bench-")
        assert rid.startswith("bench-")
        assert len(rid) == 6 + 32  # prefix + uuid hex

    def test_uniqueness(self):
        ids = {_generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_empty_prefix(self):
        rid = _generate_request_id("")
        # Empty prefix treated as no prefix
        assert len(rid) == 32


# ---------------------------------------------------------------------------
# Unit tests for RequestResult.request_id
# ---------------------------------------------------------------------------


class TestRequestResultId:
    def test_default_none(self):
        r = RequestResult()
        assert r.request_id is None

    def test_set_request_id(self):
        r = RequestResult(request_id="test-123")
        assert r.request_id == "test-123"


# ---------------------------------------------------------------------------
# Unit tests for DebugLogEntry with request_id
# ---------------------------------------------------------------------------


class TestDebugLogEntryRequestId:
    def test_request_id_in_dict(self):
        entry = DebugLogEntry(
            timestamp="2025-01-01T00:00:00",
            url="http://example.com",
            payload="{}",
            status="ok",
            latency_ms=100.0,
            success=True,
            request_id="req-abc",
        )
        d = entry.to_dict()
        assert d["request_id"] == "req-abc"

    def test_no_request_id_omitted(self):
        entry = DebugLogEntry(
            timestamp="2025-01-01T00:00:00",
            url="http://example.com",
            payload="{}",
            status="ok",
            latency_ms=100.0,
            success=True,
        )
        d = entry.to_dict()
        assert "request_id" not in d


# ---------------------------------------------------------------------------
# Per-request CSV export includes request_id
# ---------------------------------------------------------------------------


class TestPerRequestCsvExport:
    def test_request_id_column(self, tmp_path):
        result = BenchmarkResult()
        result.requests = [
            RequestResult(
                prompt_tokens=10,
                completion_tokens=5,
                latency_ms=50.0,
                request_id="rid-001",
            ),
            RequestResult(
                prompt_tokens=8,
                completion_tokens=3,
                latency_ms=40.0,
                request_id="rid-002",
            ),
        ]
        path = tmp_path / "requests.csv"
        export_per_request_csv(result, str(path))

        content = path.read_text()
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["request_id"] == "rid-001"
        assert rows[1]["request_id"] == "rid-002"

    def test_missing_request_id(self, tmp_path):
        result = BenchmarkResult()
        result.requests = [RequestResult(latency_ms=10.0)]
        path = tmp_path / "requests.csv"
        export_per_request_csv(result, str(path))

        content = path.read_text()
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        assert rows[0]["request_id"] == ""


# ---------------------------------------------------------------------------
# Integration: dummy server echoes X-Request-ID
# ---------------------------------------------------------------------------


@pytest.fixture()
def dummy_server():
    """Start the dummy server in a background thread."""
    cfg = ServerConfig(prefill_ms=5, decode_ms=1, model_name="test-model")
    app = create_app(cfg)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server_cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(server_cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            httpx.get(f"{base_url}/health", timeout=1.0)
            break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("Dummy server failed to start")

    yield base_url
    server.should_exit = True
    thread.join(timeout=5)


class TestSendRequestWithId:
    def test_request_id_injected(self, dummy_server):
        """_send_request injects X-Request-ID and stores it in result."""
        base_url = dummy_server

        async def _run():
            async with httpx.AsyncClient() as client:
                result = await _send_request(
                    client,
                    f"{base_url}/v1/completions",
                    {"model": "test-model", "prompt": "hello", "max_tokens": 5},
                    is_streaming=False,
                    request_id="myid-42",
                )
                assert result.success is True
                assert result.request_id == "myid-42"

        asyncio.run(_run())

    def test_streaming_request_id(self, dummy_server):
        """_send_request with streaming also carries request_id."""
        base_url = dummy_server

        async def _run():
            async with httpx.AsyncClient() as client:
                result = await _send_request(
                    client,
                    f"{base_url}/v1/completions",
                    {"model": "test-model", "prompt": "hello", "max_tokens": 5},
                    is_streaming=True,
                    request_id="stream-rid",
                )
                assert result.success is True
                assert result.request_id == "stream-rid"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIRequestIdPrefix:
    def test_flag_parsed(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args(["--request-id-prefix", "bench-"])
        assert args.request_id_prefix == "bench-"

    def test_default_none(self):
        import argparse

        from xpyd_bench.cli import _add_vllm_compat_args

        parser = argparse.ArgumentParser()
        _add_vllm_compat_args(parser)
        args = parser.parse_args([])
        assert args.request_id_prefix is None


# ---------------------------------------------------------------------------
# YAML config support
# ---------------------------------------------------------------------------


class TestYamlConfigRequestIdPrefix:
    def test_known_key(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        assert "request_id_prefix" in _KNOWN_KEYS
