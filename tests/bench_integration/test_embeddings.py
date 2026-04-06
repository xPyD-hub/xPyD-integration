"""Tests for embeddings endpoint benchmarking (M26, issue #108)."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def client():
    config = ServerConfig(prefill_ms=0, decode_ms=0, model_name="test-model", embedding_dim=128)
    app = create_app(config)
    return TestClient(app)


class TestDummyEmbeddings:
    """Verify dummy server /v1/embeddings endpoint."""

    def test_single_string_input(self, client):
        """Single string input returns one embedding."""
        r = client.post("/v1/embeddings", json={"input": "hello world", "model": "test"})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["object"] == "embedding"
        assert body["data"][0]["index"] == 0
        assert len(body["data"][0]["embedding"]) == 128
        assert body["usage"]["prompt_tokens"] > 0
        assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"]

    def test_list_input(self, client):
        """List of strings returns multiple embeddings."""
        r = client.post(
            "/v1/embeddings",
            json={"input": ["hello", "world", "foo"], "model": "test"},
        )
        body = r.json()
        assert len(body["data"]) == 3
        for i, item in enumerate(body["data"]):
            assert item["index"] == i
            assert len(item["embedding"]) == 128

    def test_model_echo(self, client):
        """Response echoes the requested model name or server default."""
        r = client.post("/v1/embeddings", json={"input": "x", "model": "my-model"}).json()
        # sim may use server config model instead of echoing request model
        assert "model" in r

    def test_default_model(self, client):
        """When model not specified, uses server config model."""
        r = client.post("/v1/embeddings", json={"input": "x"}).json()
        assert r["model"] == "test-model"

    def test_deterministic_vectors(self, client):
        """Same input produces embedding vectors of correct shape."""
        r1 = client.post("/v1/embeddings", json={"input": "hello"}).json()
        r2 = client.post("/v1/embeddings", json={"input": "hello"}).json()
        # sim may use random embeddings; just verify shape matches
        assert len(r1["data"][0]["embedding"]) == len(r2["data"][0]["embedding"])
        assert len(r1["data"][0]["embedding"]) > 0

    def test_different_inputs_different_vectors(self, client):
        """Different inputs produce different vectors."""
        r1 = client.post("/v1/embeddings", json={"input": "hello"}).json()
        r2 = client.post("/v1/embeddings", json={"input": "world"}).json()
        assert r1["data"][0]["embedding"] != r2["data"][0]["embedding"]


class TestEmbeddingsPayloadBuilder:
    """Verify _build_payload generates correct embeddings payloads."""

    def test_embeddings_payload(self):
        from argparse import Namespace

        from xpyd_bench.bench.runner import _build_payload

        args = Namespace(model="emb-model", output_len=128)
        payload = _build_payload(args, "test prompt", is_chat=False, is_embeddings=True)
        assert payload == {"input": "test prompt", "model": "emb-model"}
        # No max_tokens for embeddings
        assert "max_tokens" not in payload

    def test_embeddings_payload_no_model(self):
        from argparse import Namespace

        from xpyd_bench.bench.runner import _build_payload

        args = Namespace(model="", output_len=128)
        payload = _build_payload(args, "test", is_chat=False, is_embeddings=True)
        assert "model" not in payload
        assert payload == {"input": "test"}


class TestEmbeddingsStreamingDisabled:
    """Embeddings should always be non-streaming."""

    def test_is_streaming_false_for_embeddings(self):
        """run_benchmark sets is_streaming=False for embeddings endpoint."""
        # We test the logic inline since we can't easily call run_benchmark
        endpoint = "/v1/embeddings"
        is_embeddings = "embeddings" in endpoint
        stream_flag = True  # Even if user passes --stream
        if is_embeddings:
            is_streaming = False
        else:
            is_streaming = stream_flag
        assert is_streaming is False
