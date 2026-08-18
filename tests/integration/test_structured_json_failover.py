"""
tests/integration/test_structured_json_failover.py — Structured JSON failover tests.

Verifies:
  - Groq -> Gemini failover for GenerationMode.STRUCTURED_JSON
  - Gemini -> Groq failover for GenerationMode.STRUCTURED_JSON
  - Both fail -> graceful JSON fallback returned
  - Malformed JSON from first provider triggers failover to second provider
  - MoERouter candidate filtering accurately selects structured-JSON capable providers
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock
import pytest

from brain.moe_router import MoERouter
from brain.providers.base import (
    BaseProvider,
    GenerationMode,
    ProviderCapabilities,
    ProviderError,
    ProviderCapabilityError,
)
from core.resource_manager import ResourceManager


def _make_mock_provider(
    name: str,
    structured_json: bool = True,
    supports_json_mode: bool = True,
) -> Mock:
    provider = Mock(spec=BaseProvider)
    provider.provider_name = name
    provider.capabilities = ProviderCapabilities(
        supports_json_mode=supports_json_mode,
        supports_system_prompt=True,
        supports_tool_schemas=True,
        max_output_tokens=8192,
        context_window_tokens=131072,
        text=True,
        streaming=False,
        structured_json=structured_json,
    )
    provider.generate = AsyncMock()
    return provider


class TestStructuredJsonFailover:

    @pytest.mark.asyncio
    async def test_groq_structured_failure_fails_over_to_gemini(self, caplog) -> None:
        groq = _make_mock_provider("groq", structured_json=True)
        groq.generate.side_effect = ProviderError(
            "Groq rate limit exceeded (429)",
            provider="groq",
            status_code=429,
            retryable=True,
        )

        gemini = _make_mock_provider("gemini", structured_json=True)
        gemini.generate.return_value = '{"thought": "gemini thought", "action": null, "final_answer": "success from gemini"}'

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        with caplog.at_level("WARNING"):
            result = await router.generate(
                messages=[{"role": "user", "content": "plan"}],
                system_prompt="system",
                generation_mode=GenerationMode.STRUCTURED_JSON,
                provider_name="groq",
                trace_id="trace-failover-01",
            )

        groq.generate.assert_awaited_once()
        gemini.generate.assert_awaited_once()
        assert "gemini" in result

        failover_logs = [r for r in caplog.records if "[FAILOVER]" in r.message]
        assert len(failover_logs) >= 1
        assert "groq" in failover_logs[0].message
        assert "gemini" in failover_logs[0].message

    @pytest.mark.asyncio
    async def test_gemini_structured_failure_fails_over_to_groq(self, caplog) -> None:
        gemini = _make_mock_provider("gemini", structured_json=True)
        gemini.generate.side_effect = ProviderError(
            "Gemini quota exhausted (429)",
            provider="gemini",
            status_code=429,
            retryable=True,
        )

        groq = _make_mock_provider("groq", structured_json=True)
        groq.generate.return_value = '{"thought": "groq thought", "action": null, "final_answer": "success from groq"}'

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        with caplog.at_level("WARNING"):
            result = await router.generate(
                messages=[{"role": "user", "content": "plan"}],
                system_prompt="system",
                generation_mode=GenerationMode.STRUCTURED_JSON,
                provider_name="gemini",
                trace_id="trace-failover-02",
            )

        gemini.generate.assert_awaited_once()
        groq.generate.assert_awaited_once()
        assert "groq" in result

        failover_logs = [r for r in caplog.records if "[FAILOVER]" in r.message]
        assert len(failover_logs) >= 1
        assert "gemini" in failover_logs[0].message
        assert "groq" in failover_logs[0].message

    @pytest.mark.asyncio
    async def test_both_fail_in_structured_json_mode_raises_capability_error(self) -> None:
        groq = _make_mock_provider("groq", structured_json=True)
        groq.generate.side_effect = ProviderError("Groq down", provider="groq", status_code=503, retryable=True)

        gemini = _make_mock_provider("gemini", structured_json=True)
        gemini.generate.side_effect = ProviderError("Gemini down", provider="gemini", status_code=503, retryable=True)

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        with pytest.raises(ProviderCapabilityError) as exc_info:
            await router.generate(
                messages=[{"role": "user", "content": "plan"}],
                system_prompt="system",
                generation_mode=GenerationMode.STRUCTURED_JSON,
                provider_name="groq",
                trace_id="trace-failover-03",
            )
            
        assert "All AI providers are currently unavailable" in str(exc_info.value)
        assert "Gemini down" in str(exc_info.value)

        groq.generate.assert_awaited_once()
        gemini.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_malformed_json_triggers_failover(self, caplog) -> None:
        groq = _make_mock_provider("groq", structured_json=True)
        # Groq adapter raises retryable ProviderError when model output is malformed JSON
        groq.generate.side_effect = ProviderError(
            "Groq output is not valid JSON in structured mode",
            provider="groq",
            status_code=0,
            retryable=True,
        )

        gemini = _make_mock_provider("gemini", structured_json=True)
        gemini.generate.return_value = '{"thought": "valid json", "action": null, "final_answer": "ok"}'

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        with caplog.at_level("WARNING"):
            result = await router.generate(
                messages=[{"role": "user", "content": "plan"}],
                system_prompt="system",
                generation_mode=GenerationMode.STRUCTURED_JSON,
                provider_name="groq",
                trace_id="trace-failover-04",
            )

        groq.generate.assert_awaited_once()
        gemini.generate.assert_awaited_once()
        assert "valid json" in result

    def test_candidate_filtering_for_structured_json(self) -> None:
        groq = _make_mock_provider("groq", structured_json=True)
        gemini = _make_mock_provider("gemini", structured_json=True)
        legacy_ollama = _make_mock_provider("ollama", structured_json=False)

        router = MoERouter(
            providers={"groq": groq, "gemini": gemini, "ollama": legacy_ollama},
            metrics_manager=Mock(),
            resource_manager=ResourceManager(),
        )

        # Default order should include groq and gemini, but exclude ollama (which has structured_json=False)
        candidates = router._build_failover_candidates(None, output_format=GenerationMode.STRUCTURED_JSON)
        assert candidates == ["groq", "gemini"]

        # When gemini requested first
        gemini_first = router._build_failover_candidates("gemini", output_format=GenerationMode.STRUCTURED_JSON)
        assert gemini_first == ["gemini", "groq"]

        # Text mode should include all 3 providers
        text_candidates = router._build_failover_candidates(None, output_format=GenerationMode.TEXT)
        assert text_candidates == ["groq", "gemini", "ollama"]
