"""Tests for multimodal token estimation in xpyd-sim server.

Migrated from xpyd-bench test_m77_vision.py.
Only includes tests that exercise the sim server directly.
Vision utility tests remain in bench.
"""

from __future__ import annotations

from xpyd_sim.common.helpers import count_prompt_tokens as _estimate_prompt_tokens


class TestDummyServerMultimodalTokens:
    """Test that the sim server handles multimodal content in token estimation."""

    def test_string_content(self):
        result = _estimate_prompt_tokens(None, [{"content": "hello world test"}])
        assert result >= 1

    def test_multimodal_content(self):
        messages = [
            {
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "text", "text": "describe this image"},
                ]
            }
        ]
        result = _estimate_prompt_tokens(None, messages)
        assert result >= 1

    def test_empty_content(self):
        result = _estimate_prompt_tokens(None, [{"content": []}])
        assert result == 1  # max(1, 0//4)
