"""
tests/unit/brain/test_gemini_provider.py — Unit tests for GeminiProvider adapter.

Verifies:
  - Capability contract (structured_json=True, supports_json_mode=True, streaming_text=True, cancellation=True)
  - Config generation (response_mime_type="application/json" for structured mode)
  - Valid JSON parsing and pass-through
  - Malformed JSON detection (raising retryable ProviderError)
  - Empty response detection (raising retryable ProviderError)
  - Error translation from google.genai APIError (429, 401, 503)
  - Streaming generation, cancellation, empty stream, timeout, and mid-stream failure
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch
import pytest

from brain.providers.base import (
    GenerationCancelledError,
    GenerationMode,
    ProviderError,
)
from brain.providers.gemini_provider import GeminiProvider
from core.interrupt_controller import CancellationToken
from google.genai import errors as genai_errors
from google.genai import types


@pytest.fixture
def mock_gemini_provider(monkeypatch: pytest.MonkeyPatch) -> GeminiProvider:
    monkeypatch.setattr("config.settings.settings.GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setattr("config.settings.settings.GEMINI_MODEL", "gemini-2.0-flash")
    with patch("google.genai.Client"):
        provider = GeminiProvider()
        provider._client = Mock()
        provider._client.aio = Mock()
        provider._client.aio.models = Mock()
        provider._client.aio.models.generate_content = AsyncMock()
        provider._client.aio.models.generate_content_stream = AsyncMock()
        return provider


class TestGeminiProviderCapabilities:

    def test_capabilities_advertised_accurately(self, mock_gemini_provider: GeminiProvider) -> None:
        caps = mock_gemini_provider.capabilities
        assert caps.supports_json_mode is True
        assert caps.structured_json is True
        assert caps.text is True
        assert caps.streaming is True
        assert caps.cancellation is True
        assert caps.supports_system_prompt is True
        assert caps.max_output_tokens == 8192


@pytest.mark.asyncio
class TestGeminiProviderGenerate:

    async def test_structured_json_mode_sets_mime_type(self, mock_gemini_provider: GeminiProvider) -> None:
        mock_response = Mock()
        mock_response.text = '{"thought": "test", "action": null, "final_answer": "ok"}'
        mock_gemini_provider._client.aio.models.generate_content.return_value = mock_response

        result = await mock_gemini_provider.generate(
            messages=[{"role": "user", "content": "plan"}],
            system_prompt="system",
            generation_mode=GenerationMode.STRUCTURED_JSON,
        )

        assert result == '{"thought": "test", "action": null, "final_answer": "ok"}'
        call_kwargs = mock_gemini_provider._client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert config.response_mime_type == "application/json"

    async def test_text_mode_does_not_set_mime_type(self, mock_gemini_provider: GeminiProvider) -> None:
        mock_response = Mock()
        mock_response.text = "Hello there!"
        mock_gemini_provider._client.aio.models.generate_content.return_value = mock_response

        result = await mock_gemini_provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="system",
            generation_mode=GenerationMode.TEXT,
        )

        assert result == "Hello there!"
        call_kwargs = mock_gemini_provider._client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert getattr(config, "response_mime_type", None) is None

    async def test_empty_response_raises_retryable_provider_error(self, mock_gemini_provider: GeminiProvider) -> None:
        mock_response = Mock()
        mock_response.text = "   "
        mock_gemini_provider._client.aio.models.generate_content.return_value = mock_response

        with pytest.raises(ProviderError) as exc_info:
            await mock_gemini_provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="system",
            )

        assert exc_info.value.retryable is True
        assert exc_info.value.provider == "gemini"
        assert "empty response" in str(exc_info.value).lower()

    async def test_malformed_json_in_structured_mode_raises_retryable_provider_error(
        self, mock_gemini_provider: GeminiProvider
    ) -> None:
        mock_response = Mock()
        mock_response.text = '{"thought": "broken JSON, missing closing brace'
        mock_gemini_provider._client.aio.models.generate_content.return_value = mock_response

        with pytest.raises(ProviderError) as exc_info:
            await mock_gemini_provider.generate(
                messages=[{"role": "user", "content": "plan"}],
                system_prompt="system",
                generation_mode=GenerationMode.STRUCTURED_JSON,
            )

        assert exc_info.value.retryable is True
        assert exc_info.value.provider == "gemini"
        assert "valid json" in str(exc_info.value).lower()

    async def test_api_error_429_translated_to_retryable_provider_error(
        self, mock_gemini_provider: GeminiProvider
    ) -> None:
        api_err = genai_errors.APIError(429, "Resource has been exhausted (e.g. check quota).")
        mock_gemini_provider._client.aio.models.generate_content.side_effect = api_err

        with pytest.raises(ProviderError) as exc_info:
            await mock_gemini_provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="system",
            )

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider == "gemini"


@pytest.mark.asyncio
class TestGeminiProviderStreaming:

    async def test_successful_streaming(self, mock_gemini_provider: GeminiProvider) -> None:
        async def _mock_stream():
            for piece in ["Hello", " ", "from", " ", "Gemini", "!"]:
                c = Mock()
                c.text = piece
                yield c

        mock_gemini_provider._client.aio.models.generate_content_stream.return_value = _mock_stream()
        token = CancellationToken()

        result = await mock_gemini_provider.generate(
            messages=[{"role": "user", "content": "hello"}],
            system_prompt="system",
            cancel_token=token,
        )

        assert result == "Hello from Gemini!"

    async def test_streaming_cancellation(self, mock_gemini_provider: GeminiProvider) -> None:
        token = CancellationToken()

        async def _mock_stream():
            for i, piece in enumerate(["chunk1", "chunk2", "chunk3"]):
                if i == 1:
                    token.cancel("User stop")
                c = Mock()
                c.text = piece
                yield c

        mock_gemini_provider._client.aio.models.generate_content_stream.return_value = _mock_stream()

        with pytest.raises(GenerationCancelledError) as exc_info:
            await mock_gemini_provider.generate(
                messages=[{"role": "user", "content": "hello"}],
                system_prompt="system",
                cancel_token=token,
            )

        assert exc_info.value.partial_chunks_discarded >= 1

    async def test_streaming_empty_raises_retryable_error(self, mock_gemini_provider: GeminiProvider) -> None:
        async def _mock_stream():
            for piece in ["   ", ""]:
                c = Mock()
                c.text = piece
                yield c

        mock_gemini_provider._client.aio.models.generate_content_stream.return_value = _mock_stream()
        token = CancellationToken()

        with pytest.raises(ProviderError) as exc_info:
            await mock_gemini_provider.generate(
                messages=[{"role": "user", "content": "hello"}],
                system_prompt="system",
                cancel_token=token,
            )

        assert exc_info.value.retryable is True
        assert "no content" in str(exc_info.value).lower()

    async def test_streaming_timeout_raises_retryable_error(self, mock_gemini_provider: GeminiProvider) -> None:
        async def _mock_stream():
            raise asyncio.TimeoutError("Streaming connection timed out")
            yield Mock()

        mock_gemini_provider._client.aio.models.generate_content_stream.return_value = _mock_stream()
        token = CancellationToken()

        with pytest.raises(ProviderError) as exc_info:
            await mock_gemini_provider.generate(
                messages=[{"role": "user", "content": "hello"}],
                system_prompt="system",
                cancel_token=token,
            )

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 408
        assert "timed out" in str(exc_info.value).lower()

    async def test_streaming_midstream_api_error(self, mock_gemini_provider: GeminiProvider) -> None:
        async def _mock_stream():
            c = Mock()
            c.text = "partial text"
            yield c
            raise genai_errors.APIError(503, "Service unavailable")

        mock_gemini_provider._client.aio.models.generate_content_stream.return_value = _mock_stream()
        token = CancellationToken()

        with pytest.raises(ProviderError) as exc_info:
            await mock_gemini_provider.generate(
                messages=[{"role": "user", "content": "hello"}],
                system_prompt="system",
                cancel_token=token,
            )

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 503

    async def test_streaming_structured_json_malformed(self, mock_gemini_provider: GeminiProvider) -> None:
        async def _mock_stream():
            for piece in ['{"thought":', ' "incomplete JSON']:
                c = Mock()
                c.text = piece
                yield c

        mock_gemini_provider._client.aio.models.generate_content_stream.return_value = _mock_stream()
        token = CancellationToken()

        with pytest.raises(ProviderError) as exc_info:
            await mock_gemini_provider.generate(
                messages=[{"role": "user", "content": "plan"}],
                system_prompt="system",
                generation_mode=GenerationMode.STRUCTURED_JSON,
                cancel_token=token,
            )

        assert exc_info.value.retryable is True
        assert "valid json" in str(exc_info.value).lower()
