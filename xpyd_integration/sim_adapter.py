"""ASGI applications used by subprocess-based integration tests."""

import os

from xpyd_sim.server import ServerConfig, create_app


def make_sim_app(mode: str):
    return create_app(
        ServerConfig(
            mode=mode,
            model_name=os.environ["model_path"],
            prefill_delay_ms=0,
            kv_transfer_delay_ms=0,
            decode_delay_per_token_ms=0,
            eos_min_ratio=1.0,
            max_model_len=131072,
        )
    )


prefill_app = make_sim_app("prefill")
decode_app = make_sim_app("decode")
