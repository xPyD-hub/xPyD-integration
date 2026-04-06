"""Fixtures for C2: single node advanced features."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def auth_config():
    return ServerConfig(
        mode="dual",
        model_name="test-model",
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        eos_min_ratio=1.0,
        max_model_len=4096,
        require_api_key="sk-test-secret",
    )


@pytest.fixture
def auth_app(auth_config):
    return create_app(auth_config)


@pytest_asyncio.fixture
async def auth_client(auth_app):
    transport = ASGITransport(app=auth_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
