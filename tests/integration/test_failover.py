"""
tests/integration/test_failover.py — Phase 7.1 MoERouter failover proof.

Proves the [FAILOVER] logging path (Task 2) actually engages end-to-end:
a retryable ProviderError from the requested provider triggers a switch
to the next candidate, and the final response comes from the provider
that actually succeeded — not the one originally requested.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from brain.moe_router import MoERouter
from brain.providers.base import BaseProvider, ProviderCapabilities, ProviderError

pytestmark = pytest.mark.asyncio


def _make_mock_provider(name: str) -> Mock:
    provider = Mock(spec=BaseProvider)
    provider.provider_name = name
    provider.capabilities = ProviderCapabilities(
        supports_json_mode=(name == "groq"),
        supports_system_prompt=True,
        supports_tool_schemas=False,
        max_output_tokens=8192,
        context_window_tokens=131072,
    )
    provider.generate = AsyncMock()
    return provider


class TestMoERouterFailover:

    async def test_gemini_429_fails_over_to_groq(self, caplog) -> None:
        """
        Task 3 mandate: Gemini raises retryable 429 -> Groq is attempted
        next -> final response is Groq's, not Gemini's.
        """
        gemini = _make_mock_provider("gemini")
        gemini.generate.side_effect = ProviderError(
            "Quota exhausted",
            provider="gemini",
            status_code=429,
            retryable=True,
        )

        groq = _make_mock_provider("groq")
        groq.generate.return_value = "Fallback successful"

        router = MoERouter(providers={"gemini": gemini, "groq": groq})

        with caplog.at_level("WARNING"):
            result = await router.chat(
                user_input="write me a poem",
                history=[],
                trace_id="test-trace-001",
                provider_name="gemini",
            )

        gemini.generate.assert_awaited_once()
        groq.generate.assert_awaited_once()

        assert result == "Fallback successful"

        failover_logs = [r for r in caplog.records if "[FAILOVER]" in r.message]
        assert len(failover_logs) == 1
        assert "gemini" in failover_logs[0].message
        assert "429" in failover_logs[0].message
        assert "groq" in failover_logs[0].message

    async def test_requested_provider_succeeds_no_failover_triggered(self) -> None:
        """
        Negative control: when the requested provider succeeds on the
        first attempt, the fallback provider must never be called and
        no [FAILOVER] log should fire.
        """
        gemini = _make_mock_provider("gemini")
        gemini.generate.return_value = "Direct success"

        groq = _make_mock_provider("groq")
        groq.generate.return_value = "Should never be reached"

        router = MoERouter(providers={"gemini": gemini, "groq": groq})

        result = await router.chat(
            user_input="hello",
            history=[],
            trace_id="test-trace-002",
            provider_name="gemini",
        )

        gemini.generate.assert_awaited_once()
        groq.generate.assert_not_awaited()
        assert result == "Direct success"

    async def test_non_retryable_error_does_not_trigger_failover(self, caplog) -> None:
        """
        A hard failure (retryable=False, e.g. 401 auth error) must NOT
        silently fail over to another provider — that could mask a real
        misconfiguration behind an apparently-working response.
        """
        gemini = _make_mock_provider("gemini")
        gemini.generate.side_effect = ProviderError(
            "Invalid API key",
            provider="gemini",
            status_code=401,
            retryable=False,
        )

        groq = _make_mock_provider("groq")
        groq.generate.return_value = "Should never be reached"

        router = MoERouter(providers={"gemini": gemini, "groq": groq})

        result = await router.chat(
            user_input="hello",
            history=[],
            trace_id="test-trace-003",
            provider_name="gemini",
        )

        gemini.generate.assert_awaited_once()
        groq.generate.assert_not_awaited()

        failover_logs = [r for r in caplog.records if "[FAILOVER]" in r.message]
        assert len(failover_logs) == 0
        assert "gemini" in result.lower() or "configuration" in result.lower()

    async def test_all_providers_exhausted_returns_graceful_message(self) -> None:
        """
        Both providers fail with retryable errors — final response must
        be a graceful user-facing message, not an unhandled exception.
        """
        gemini = _make_mock_provider("gemini")
        gemini.generate.side_effect = ProviderError(
            "Quota exhausted", provider="gemini", status_code=429, retryable=True,
        )

        groq = _make_mock_provider("groq")
        groq.generate.side_effect = ProviderError(
            "Service unavailable", provider="groq", status_code=503, retryable=True,
        )

        router = MoERouter(providers={"gemini": gemini, "groq": groq})

        result = await router.chat(
            user_input="hello",
            history=[],
            trace_id="test-trace-004",
            provider_name="gemini",
        )

        gemini.generate.assert_awaited_once()
        groq.generate.assert_awaited_once()
        assert "trouble" in result.lower() or "try again" in result.lower()