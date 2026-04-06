"""C2: API key authentication tests (migrated from bench)."""

import pytest


@pytest.mark.anyio
async def test_request_without_key_returns_401(auth_client):
    """Request without API key should return 401."""
    resp = await auth_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_request_with_correct_key_succeeds(auth_client):
    """Request with correct API key should succeed."""
    resp = await auth_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        },
        headers={"Authorization": "Bearer sk-test-secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"]
