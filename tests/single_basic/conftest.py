"""Fixtures for C1: single node (sim dual) + direct client."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from xpyd_sim.server import ServerConfig, create_app


@pytest.fixture
def sim_config():
    """Create a fast sim config for testing."""
    return ServerConfig(
        mode="dual",
        model_name="test-model",
        prefill_delay_ms=0,
        kv_transfer_delay_ms=0,
        decode_delay_per_token_ms=0,
        eos_min_ratio=1.0,
        max_model_len=4096,
    )


@pytest.fixture
def sim_app(sim_config):
    return create_app(sim_config)


@pytest_asyncio.fixture
async def client(sim_app):
    transport = ASGITransport(app=sim_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
