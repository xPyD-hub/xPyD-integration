"""Shared fixtures for 1p1d_advanced tests."""

from pathlib import Path

from xpyd_sim.server import ServerConfig, create_app

_TOKENIZER_PATH = str(
    Path(__file__).resolve().parent.parent / "assets" / "tokenizer"
)


def make_sim_app(model_name=None, mode="dual"):
    """Create a xpyd-sim app, matching proxy repo's sim_adapter.make_sim_app."""
    return create_app(
        ServerConfig(
            mode=mode,
            model_name=model_name or _TOKENIZER_PATH,
            prefill_delay_ms=0,
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
            eos_min_ratio=1.0,
            max_model_len=131072,
        )
    )
