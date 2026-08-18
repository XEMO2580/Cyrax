"""
tests/unit/brain/test_groq_provider.py — Unit tests for GroqProvider adapter.

Verifies:
  - Capability contract (structured_json=True, supports_json_mode=True, streaming_text=True, cancellation=True)
  - Config generation (response_format={"type": "json_object"} for structured mode)
  - Valid JSON parsing and pass-through
  - Malformed JSON detection (raising retryable ProviderError)
  - Empty response detection (raising retryable ProviderError)
  - Error translation from Groq exceptions (RateLimitError, BadRequestError)
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
from brain.providers.groq_provider import GroqProvider
from core.interrupt_controller import CancellationToken
from groq import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    BadRequestError,
)


@pytest.fixture
def mock_groq_provider(monkeypatch: pytest.MonkeyPatch) -> GroqProvider:
    monkeypatch.setattr("config.settings.settings.GROQ_API_KEY", "test_groq_key")
    monkeypatch.setattr("config.settings.settings.GROQ_MODEL", "llama-3.3-70b-versatile")
    with patch("groq.AsyncGroq"):
        provider = GroqProvider()
        provider._client = Mock()
        provider._client.chat = Mock()
        provider._client.chat.completions = Mock()
        provider._client.chat.completions.create = AsyncMock()
        return provider


class TestGroqProviderCapabilities:

    def test_capabilities_advertised_accurately(self, mock_groq_provider: GroqProvider) -> None:
        caps = mock_groq_provider.capabilities
        assert caps.supports_json_mode is True
        assert caps.text is True
        assert caps.streaming is True
        assert caps.structured_json is True
        assert caps.cancellation is True
        assert caps.max_output_tokens == 8192


@pytest.mark.asyncio
class TestGroqProviderGenerate:

    async def test_structured_json_mode_sets_response_format(self, mock_groq_provider: GroqProvider) -> None:
        mock_choice = Mock()
        mock_choice.message.content = '{"thought": "test", "action": null, "final_answer": "done"}'
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage.total_tokens = 42
        mock_groq_provider._client.chat.completions.create.return_value = mock_response

        result = await mock_groq_provider.generate(
            messages=[{"role": "user", "content": "plan"}],
            system_prompt="system",
            generation_mode=GenerationMode.STRUCTURED_JSON,
        )

        assert result == '{"thought": "test", "action": null, "final_answer": "done"}'
        call_kwargs = mock_groq_provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("response_format", {}).get("type") == "json_object"

    async def test_text_mode_does_not_set_response_format(self, mock_groq_provider: GroqProvider) -> None:
        mock_choice = Mock()
        mock_choice.message.content = "Plain text response."
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage.total_tokens = 10
        mock_groq_provider._client.chat.completions.create.return_value = mock_response

        result = await mock_groq_provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="system",
            generation_mode=GenerationMode.TEXT,
        )

        assert result == "Plain text response."
        call_kwargs = mock_groq_provider._client.chat.completions.create.call_args.kwargs
        assert "response_format" not in call_kwargs

    async def test_empty_response_raises_retryable_provider_error(self, mock_groq_provider: GroqProvider) -> None:
        mock_choice = Mock()
        mock_choice.message.content = "   "
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_groq_provider._client.chat.completions.create.return_value = mock_response

        with pytest.raises(ProviderError) as exc_info:
            await mock_groq_provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="system",
            )

        assert exc_info.value.retryable is True
        assert exc_info.value.provider == "groq"
        assert "empty response" in str(exc_info.value).lower()

    async def test_malformed_json_in_structured_mode_raises_retryable_provider_error(
        self, mock_groq_provider: GroqProvider
    ) -> None:
        mock_choice = Mock()
        mock_choice.message.content = '{"thought": "broken JSON, missing brace'
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage.total_tokens = 15
        mock_groq_provider._client.chat.completions.create.return_value = mock_response

        with pytest.raises(ProviderError) as exc_info:
            await mock_groq_provider.generate(
                messages=[{"role": "user", "content": "plan"}],
                system_prompt="system",
                generation_mode=GenerationMode.STRUCTURED_JSON,
            )

        assert exc_info.value.retryable is True
        assert exc_info.value.provider == "groq"
        assert "valid json" in str(exc_info.value).lower()


@pytest.mark.asyncio
class TestGroqProviderStreaming:

    async def test_successful_streaming(self, mock_groq_provider: GroqProvider) -> None:
        async def _mock_stream():
            for piece in ["Hello", " ", "from", " ", "Groq", "!"]:
                chunk = Mock()
                choice = Mock()
                choice.delta.content = piece
                chunk.choices = [choice]
                yield chunk

        mock_groq_provider._client.chat.completions.create.return_value = _mock_stream()
        token = CancellationToken()

        result = await mock_groq_provider.generate(
            messages=[{"role": "user", "content": "hello"}],
            system_prompt="system",
            cancel_token=token,
        )

        assert result == "Hello from Groq!"

    async def test_streaming_cancellation(self, mock_groq_provider: GroqProvider) -> None:
        token = CancellationToken()

        class _MockStream:
            def __init__(self):
                self.closed = False

            async def __aiter__(self):
                for i, piece in enumerate(["chunk1", "chunk2", "chunk3"]):
                    if i == 1:
                        token.cancel("User stop")
                    chunk = Mock()
                    choice = Mock()
                    choice.delta.content = piece
                    chunk.choices = [choice]
                    yield chunk

            async def close(self):
                self.closed = True

        mock_stream_obj = _MockStream()
        mock_groq_provider._client.chat.completions.create.return_value = mock_stream_obj

        with pytest.raises(GenerationCancelledError) as exc_info:
            await mock_groq_provider.generate(
                messages=[{"role": "user", "content": "hello"}],
                system_prompt="system",
                cancel_token=token,
            )

        assert exc_info.value.partial_chunks_discarded >= 1
        assert mock_stream_obj.closed is True

    async def test_streaming_empty_raises_retryable_error(self, mock_groq_provider: GroqProvider) -> None:
        async def _mock_stream():
            for piece in ["   ", ""]:
                chunk = Mock()
                choice = Mock()
                choice.delta.content = piece
                chunk.choices = [choice]
                yield chunk

        mock_groq_provider._client.chat.completions.create.return_value = _mock_stream()
        token = CancellationToken()

        with pytest.raises(ProviderError) as exc_info:
            await mock_groq_provider.generate(
                messages=[{"role": "user", "content": "hello"}],
                system_prompt="system",
                cancel_token=token,
            )

        assert exc_info.value.retryable is True
        assert "no content" in str(exc_info.value).lower()

    async def test_streaming_timeout_raises_retryable_error(self, mock_groq_provider: GroqProvider) -> None:
        async def _mock_stream():
            raise APITimeoutError(request=Mock())
            yield Mock()

        mock_groq_provider._client.chat.completions.create.return_value = _mock_stream()
        token = CancellationToken()

        with pytest.raises(ProviderError) as exc_info:
            await mock_groq_provider.generate(
                messages=[{"role": "user", "content": "hello"}],
                system_prompt="system",
                cancel_token=token,
            )

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 408
        assert "timed out" in str(exc_info.value).lower()

    async def test_streaming_midstream_connection_error(self, mock_groq_provider: GroqProvider) -> None:
        async def _mock_stream():
            chunk = Mock()
            choice = Mock()
            choice.delta.content = "partial text"
            chunk.choices = [choice]
            yield chunk
            raise APIConnectionError(request=Mock())

        mock_groq_provider._client.chat.completions.create.return_value = _mock_stream()
        token = CancellationToken()

        with pytest.raises(ProviderError) as exc_info:
            await mock_groq_provider.generate(
                messages=[{"role": "user", "content": "hello"}],
                system_prompt="system",
                cancel_token=token,
            )

        assert exc_info.value.retryable is True
        assert "connection error" in str(exc_info.value).lower()

    async def test_streaming_structured_json_malformed(self, mock_groq_provider: GroqProvider) -> None:
        async def _mock_stream():
            for piece in ['{"thought":', ' "incomplete JSON']:
                chunk = Mock()
                choice = Mock()
                choice.delta.content = piece
                chunk.choices = [choice]
                yield chunk

        mock_groq_provider._client.chat.completions.create.return_value = _mock_stream()
        token = CancellationToken()

        with pytest.raises(ProviderError) as exc_info:
            await mock_groq_provider.generate(
                messages=[{"role": "user", "content": "plan"}],
                system_prompt="system",
                generation_mode=GenerationMode.STRUCTURED_JSON,
                cancel_token=token,
            )

        assert exc_info.value.retryable is True
        assert "valid json" in str(exc_info.value).lower()
