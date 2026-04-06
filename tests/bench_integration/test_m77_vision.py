"""Tests for M77: Multimodal (Vision) Benchmarking."""

from __future__ import annotations

import base64
from argparse import Namespace

import pytest

from xpyd_bench.bench.vision import (
    build_vision_content,
    build_vision_payload_content,
    encode_image_base64,
    generate_synthetic_image,
    load_image_sources,
)

# ---------------------------------------------------------------------------
# Unit tests for vision.py
# ---------------------------------------------------------------------------


class TestGenerateSyntheticImage:
    """Tests for synthetic image generation."""

    def test_returns_valid_png(self):
        data = generate_synthetic_image(width=8, height=8, seed=42)
        assert isinstance(data, bytes)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature

    def test_deterministic_with_seed(self):
        a = generate_synthetic_image(width=8, height=8, seed=42)
        b = generate_synthetic_image(width=8, height=8, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_synthetic_image(width=8, height=8, seed=1)
        b = generate_synthetic_image(width=8, height=8, seed=2)
        assert a != b


class TestEncodeImageBase64:
    """Tests for base64 encoding."""

    def test_encodes_file(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfakedata")
        result = encode_image_base64(str(img))
        decoded = base64.b64decode(result)
        assert decoded == b"\x89PNG\r\n\x1a\nfakedata"


class TestBuildVisionContent:
    """Tests for building multimodal content arrays."""

    def test_text_only(self):
        parts = build_vision_content("describe this")
        assert len(parts) == 1
        assert parts[0] == {"type": "text", "text": "describe this"}

    def test_with_url(self):
        parts = build_vision_content(
            "describe this",
            image_urls=["https://example.com/img.png"],
        )
        assert len(parts) == 2
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "https://example.com/img.png"
        assert parts[0]["image_url"]["detail"] == "auto"
        assert parts[1]["type"] == "text"

    def test_with_file(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fakeimg")
        parts = build_vision_content(
            "describe",
            image_files=[str(img)],
            image_detail="high",
        )
        assert len(parts) == 2
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert parts[0]["image_url"]["detail"] == "high"

    def test_multiple_images(self):
        parts = build_vision_content(
            "compare",
            image_urls=["https://a.com/1.png", "https://a.com/2.png"],
        )
        assert len(parts) == 3  # 2 images + 1 text


class TestLoadImageSources:
    """Tests for loading image sources."""

    def test_from_url(self):
        sources = load_image_sources(image_url="https://example.com/img.png")
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/img.png"

    def test_from_directory(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"fake-png")
        (tmp_path / "b.jpg").write_bytes(b"fake-jpg")
        (tmp_path / "not_image.txt").write_text("nope")
        sources = load_image_sources(image_dir=str(tmp_path))
        assert len(sources) == 2
        assert all("data_uri" in s for s in sources)

    def test_synthetic(self):
        sources = load_image_sources(synthetic_images=3, seed=42)
        assert len(sources) == 3
        assert all("data_uri" in s for s in sources)
        # Should be valid base64 data URIs
        for s in sources:
            assert s["data_uri"].startswith("data:image/png;base64,")

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image_sources(image_dir="/nonexistent/dir")

    def test_empty_returns_empty(self):
        sources = load_image_sources()
        assert sources == []


class TestBuildVisionPayloadContent:
    """Tests for build_vision_payload_content."""

    def test_with_url_source(self):
        src = {"url": "https://example.com/img.png"}
        parts = build_vision_payload_content("describe", src)
        assert len(parts) == 2
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "https://example.com/img.png"
        assert parts[1] == {"type": "text", "text": "describe"}

    def test_with_data_uri_source(self):
        src = {"data_uri": "data:image/png;base64,abc123"}
        parts = build_vision_payload_content("describe", src, image_detail="low")
        assert parts[0]["image_url"]["url"] == "data:image/png;base64,abc123"
        assert parts[0]["image_url"]["detail"] == "low"


# ---------------------------------------------------------------------------
# Integration: _build_payload with vision sources
# ---------------------------------------------------------------------------


class TestBuildPayloadVision:
    """Test that _build_payload produces multimodal content when vision is enabled."""

    def test_vision_payload_has_multimodal_content(self):
        from xpyd_bench.bench.runner import _build_payload

        args = Namespace(
            model="gpt-4-vision-preview",
            output_len=128,
            temperature=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            best_of=None,
            use_beam_search=False,
            logprobs=None,
            ignore_eos=False,
            stop=None,
            n=None,
            api_seed=None,
            echo=False,
            suffix=None,
            logit_bias=None,
            user=None,
            stream_options_include_usage=False,
            response_format=None,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
            top_logprobs=None,
            max_completion_tokens=None,
            service_tier=None,
            image_detail="auto",
            _vision_sources=[{"url": "https://example.com/img.png"}],
        )
        payload = _build_payload(args, "What is in this image?", is_chat=True)
        messages = payload["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert any(p["type"] == "image_url" for p in content)
        assert any(p["type"] == "text" for p in content)

    def test_no_vision_payload_is_string_content(self):
        from xpyd_bench.bench.runner import _build_payload

        args = Namespace(
            model="gpt-4",
            output_len=128,
            temperature=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            best_of=None,
            use_beam_search=False,
            logprobs=None,
            ignore_eos=False,
            stop=None,
            n=None,
            api_seed=None,
            echo=False,
            suffix=None,
            logit_bias=None,
            user=None,
            stream_options_include_usage=False,
            response_format=None,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
            top_logprobs=None,
            max_completion_tokens=None,
            service_tier=None,
            _vision_sources=None,
        )
        payload = _build_payload(args, "Hello world", is_chat=True)
        messages = payload["messages"]
        assert messages[0]["content"] == "Hello world"


# ---------------------------------------------------------------------------
# Dummy server multimodal token estimation
# ---------------------------------------------------------------------------


class TestDummyServerMultimodalTokens:
    """Test that the dummy server handles multimodal content in token estimation."""

    def test_string_content(self):
        from xpyd_sim.server import _estimate_prompt_tokens

        result = _estimate_prompt_tokens(None, [{"content": "hello world test"}])
        assert result >= 1  # At least 1 token estimated

    def test_multimodal_content(self):
        from xpyd_sim.server import _estimate_prompt_tokens

        messages = [
            {
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "text", "text": "describe this image"},
                ]
            }
        ]
        result = _estimate_prompt_tokens(None, messages)
        assert result >= 1  # At least 1 token estimated

    def test_empty_content(self):
        from xpyd_sim.server import _estimate_prompt_tokens

        result = _estimate_prompt_tokens(None, [{"content": []}])
        assert result == 1  # max(1, 0//4)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _make_parser():
    """Helper to build a parser with vLLM-compat args."""
    import argparse

    from xpyd_bench.cli import _add_vllm_compat_args
    parser = argparse.ArgumentParser()
    _add_vllm_compat_args(parser)
    return parser


class TestVisionCLIArgs:
    """Test CLI argument parsing for vision flags."""

    def test_image_url_flag(self):
        args = _make_parser().parse_args([
            "--base-url", "http://localhost:8000",
            "--image-url", "https://example.com/img.png",
        ])
        assert args.image_url == "https://example.com/img.png"

    def test_image_dir_flag(self):
        args = _make_parser().parse_args([
            "--base-url", "http://localhost:8000",
            "--image-dir", "/tmp/images",
        ])
        assert args.image_dir == "/tmp/images"

    def test_synthetic_images_flag(self):
        args = _make_parser().parse_args([
            "--base-url", "http://localhost:8000",
            "--synthetic-images", "5",
        ])
        assert args.synthetic_images == 5

    def test_image_detail_flag(self):
        args = _make_parser().parse_args([
            "--base-url", "http://localhost:8000",
            "--image-detail", "high",
        ])
        assert args.image_detail == "high"

    def test_synthetic_image_size_flag(self):
        args = _make_parser().parse_args([
            "--base-url", "http://localhost:8000",
            "--synthetic-image-size", "128x128",
        ])
        assert args.synthetic_image_size == "128x128"

    def test_defaults(self):
        args = _make_parser().parse_args(["--base-url", "http://localhost:8000"])
        assert args.image_url is None
        assert args.image_dir is None
        assert args.synthetic_images == 0
        assert args.image_detail == "auto"
        assert args.synthetic_image_size == "64x64"


# ---------------------------------------------------------------------------
# Config known keys
# ---------------------------------------------------------------------------


class TestVisionConfigKeys:
    """Test that vision config keys are recognized."""

    def test_vision_keys_known(self):
        from xpyd_bench.config_cmd import _KNOWN_KEYS

        vision_keys = {
            "image_url", "image_dir", "synthetic_images",
            "synthetic_image_size", "image_detail",
        }
        assert vision_keys.issubset(_KNOWN_KEYS)
