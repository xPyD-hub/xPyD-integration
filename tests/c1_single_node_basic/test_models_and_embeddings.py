"""C1: Basic /v1/models and /v1/embeddings tests."""

import pytest


@pytest.mark.anyio
async def test_models_endpoint(client):
    """GET /v1/models returns model list."""
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == "test-model"


@pytest.mark.anyio
async def test_embeddings_float(client):
    """POST /v1/embeddings returns float vectors."""
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "test-model", "input": "Hello world"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert isinstance(data["data"][0]["embedding"], list)
    assert len(data["data"][0]["embedding"]) > 0
    assert data["usage"]["prompt_tokens"] > 0
