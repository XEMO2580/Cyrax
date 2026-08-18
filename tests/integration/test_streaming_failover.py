"""
tests/integration/test_streaming_failover.py — MoERouter Streaming Failover and Resiliency Suite.

Verifies:
  1. Groq successful stream through MoERouter
  2. Gemini successful stream through MoERouter
  3. Groq retryable streaming failure -> failover to Gemini
  4. Gemini retryable streaming failure -> failover to Groq
  5. Empty stream from primary -> failover to secondary
  6. Streaming timeout -> failover to secondary
  7. Cancellation mid-stream -> NO failover, immediate GenerationCancelledError propagation
  8. Non-retryable failure (e.g. 401) -> NO failover
  9. All providers exhausted in streaming mode -> graceful user-facing degradation
  10. Regression: plain text chat without cancel token works normally
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock
import pytest

from brain.moe_router import MoERouter
from brain.providers.base import (
    BaseProvider,
    GenerationCancelledError,
    GenerationMode,
    ProviderCapabilities,
    ProviderError,
    ProviderCapabilityError,
)
from core.interrupt_controller import CancellationToken
from core.resource_manager import ResourceManager


def _make_streaming_provider(
    name: str,
    streaming: bool = True,
    cancellation: bool = True,
) -> Mock:
    provider = Mock(spec=BaseProvider)
    provider.provider_name = name
    provider.capabilities = ProviderCapabilities(
        supports_json_mode=True,
        supports_system_prompt=True,
        supports_tool_schemas=True,
        max_output_tokens=8192,
        context_window_tokens=131072,
        text=True,
        streaming=streaming,
        structured_json=False,
        cancellation=cancellation,
    )
    provider.generate = AsyncMock()
    return provider


class TestStreamingFailover:

    @pytest.mark.asyncio
    async def test_groq_successful_stream(self) -> None:
        groq = _make_streaming_provider("groq")
        groq.generate.return_value = "Hello from streaming Groq"

        gemini = _make_streaming_provider("gemini")

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        token = CancellationToken()
        result = await router.chat(
            user_input="hello",
            history=[],
            trace_id="trace-stream-01",
            provider_name="groq",
            cancel_token=token,
        )

        assert result == "Hello from streaming Groq"
        groq.generate.assert_awaited_once()
        gemini.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gemini_successful_stream(self) -> None:
        gemini = _make_streaming_provider("gemini")
        gemini.generate.return_value = "Hello from streaming Gemini"

        groq = _make_streaming_provider("groq")

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        token = CancellationToken()
        result = await router.chat(
            user_input="hello",
            history=[],
            trace_id="trace-stream-02",
            provider_name="gemini",
            cancel_token=token,
        )

        assert result == "Hello from streaming Gemini"
        gemini.generate.assert_awaited_once()
        groq.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_groq_streaming_failure_fails_over_to_gemini(self, caplog) -> None:
        groq = _make_streaming_provider("groq")
        groq.generate.side_effect = ProviderError(
            "Groq rate limit exceeded (429)",
            provider="groq",
            status_code=429,
            retryable=True,
        )

        gemini = _make_streaming_provider("gemini")
        gemini.generate.return_value = "Recovered via Gemini stream"

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        token = CancellationToken()
        with caplog.at_level("WARNING"):
            result = await router.chat(
                user_input="stream test",
                history=[],
                trace_id="trace-stream-03",
                provider_name="groq",
                cancel_token=token,
            )

        assert result == "Recovered via Gemini stream"
        groq.generate.assert_awaited_once()
        gemini.generate.assert_awaited_once()

        failover_logs = [r for r in caplog.records if "[FAILOVER]" in r.message]
        assert len(failover_logs) >= 1
        assert "groq" in failover_logs[0].message
        assert "gemini" in failover_logs[0].message

    @pytest.mark.asyncio
    async def test_gemini_streaming_failure_fails_over_to_groq(self, caplog) -> None:
        gemini = _make_streaming_provider("gemini")
        gemini.generate.side_effect = ProviderError(
            "Gemini quota exhausted (429)",
            provider="gemini",
            status_code=429,
            retryable=True,
        )

        groq = _make_streaming_provider("groq")
        groq.generate.return_value = "Recovered via Groq stream"

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        token = CancellationToken()
        with caplog.at_level("WARNING"):
            result = await router.chat(
                user_input="stream test",
                history=[],
                trace_id="trace-stream-04",
                provider_name="gemini",
                cancel_token=token,
            )

        assert result == "Recovered via Groq stream"
        gemini.generate.assert_awaited_once()
        groq.generate.assert_awaited_once()

        failover_logs = [r for r in caplog.records if "[FAILOVER]" in r.message]
        assert len(failover_logs) >= 1
        assert "gemini" in failover_logs[0].message
        assert "groq" in failover_logs[0].message

    @pytest.mark.asyncio
    async def test_empty_stream_fails_over(self, caplog) -> None:
        groq = _make_streaming_provider("groq")
        groq.generate.side_effect = ProviderError(
            "Groq streaming call returned no content.",
            provider="groq",
            status_code=0,
            retryable=True,
        )

        gemini = _make_streaming_provider("gemini")
        gemini.generate.return_value = "Non-empty content from Gemini"

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        token = CancellationToken()
        with caplog.at_level("WARNING"):
            result = await router.chat(
                user_input="tell me a story",
                history=[],
                trace_id="trace-stream-05",
                provider_name="groq",
                cancel_token=token,
            )

        assert result == "Non-empty content from Gemini"
        groq.generate.assert_awaited_once()
        gemini.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_streaming_timeout_fails_over(self, caplog) -> None:
        groq = _make_streaming_provider("groq")
        groq.generate.side_effect = ProviderError(
            "Groq streaming request timed out",
            provider="groq",
            status_code=408,
            retryable=True,
        )

        gemini = _make_streaming_provider("gemini")
        gemini.generate.return_value = "Fast response from Gemini"

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        token = CancellationToken()
        with caplog.at_level("WARNING"):
            result = await router.chat(
                user_input="quick query",
                history=[],
                trace_id="trace-stream-06",
                provider_name="groq",
                cancel_token=token,
            )

        assert result == "Fast response from Gemini"
        groq.generate.assert_awaited_once()
        gemini.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancellation_does_not_failover(self) -> None:
        groq = _make_streaming_provider("groq")
        groq.generate.side_effect = GenerationCancelledError(partial_chunks_discarded=3)

        gemini = _make_streaming_provider("gemini")
        gemini.generate.return_value = "Should never be called"

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        token = CancellationToken()
        token.cancel("User cancelled")

        with pytest.raises(GenerationCancelledError):
            await router.chat(
                user_input="abort this",
                history=[],
                trace_id="trace-stream-07",
                provider_name="groq",
                cancel_token=token,
            )

        groq.generate.assert_awaited_once()
        gemini.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_retryable_error_does_not_failover(self) -> None:
        groq = _make_streaming_provider("groq")
        groq.generate.side_effect = ProviderError(
            "Groq authentication failed",
            provider="groq",
            status_code=401,
            retryable=False,
        )

        gemini = _make_streaming_provider("gemini")
        gemini.generate.return_value = "Should never be reached"

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        token = CancellationToken()
        result = await router.chat(
            user_input="test auth",
            history=[],
            trace_id="trace-stream-08",
            provider_name="groq",
            cancel_token=token,
        )

        groq.generate.assert_awaited_once()
        gemini.generate.assert_not_awaited()
        assert "configuration issue" in result.lower() or "groq" in result.lower()

    @pytest.mark.asyncio
    async def test_all_streaming_providers_exhausted(self) -> None:
        groq = _make_streaming_provider("groq")
        groq.generate.side_effect = ProviderError("Groq 503", provider="groq", status_code=503, retryable=True)

        gemini = _make_streaming_provider("gemini")
        gemini.generate.side_effect = ProviderError("Gemini 503", provider="gemini", status_code=503, retryable=True)

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        token = CancellationToken()
        with pytest.raises(ProviderCapabilityError) as exc_info:
            await router.chat(
                user_input="hello",
                history=[],
                trace_id="trace-stream-09",
                cancel_token=token,
            )

        groq.generate.assert_awaited_once()
        gemini.generate.assert_awaited_once()
        assert "All AI providers are currently unavailable" in str(exc_info.value)
        assert "Gemini 503" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_normal_text_regression(self) -> None:
        groq = _make_streaming_provider("groq")
        groq.generate.return_value = "Normal text response"

        router = MoERouter(
            providers={"groq": groq},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        result = await router.chat(
            user_input="hello non-streaming",
            history=[],
            trace_id="trace-stream-10",
        )

        assert result == "Normal text response"
        call_kwargs = groq.generate.call_args.kwargs
        assert call_kwargs["cancel_token"] is None
        assert call_kwargs["generation_mode"] == GenerationMode.TEXT
