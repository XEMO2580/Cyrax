# tests/integration/test_adaptive_routing.py

"""
tests/integration/test_adaptive_routing.py — Phase 7.2 APS certification.

Covers the three mandated scenarios: task affinity, circuit breaker
tripping under consecutive failures, and sliding-window recovery.
"""

from __future__ import annotations

import time

import pytest

from brain.provider_events import ProviderExecutionEvent
from brain.provider_metrics import CircuitState, ProviderMetricsManager
from brain.provider_selector import ProviderSelector


def _event(provider: str, success: bool, status_code: int = 200, latency_ms: float = 100.0) -> ProviderExecutionEvent:
    return ProviderExecutionEvent(
        provider=provider,
        intent="CODE_GENERATION",
        latency_ms=latency_ms,
        success=success,
        status_code=status_code,
    )


class TestTaskAffinity:

    def test_gemini_chosen_for_code_when_both_healthy(self) -> None:
        metrics = ProviderMetricsManager()
        selector = ProviderSelector()

        winner, trace = selector.select_provider(
            intent="CODE_GENERATION",
            active_providers=["groq", "gemini"],
            metrics=metrics,
        )

        assert winner == "gemini"
        assert "Gemini selected" in trace

    def test_groq_chosen_for_os_control_when_both_healthy(self) -> None:
        metrics = ProviderMetricsManager()
        selector = ProviderSelector()

        winner, trace = selector.select_provider(
            intent="OS_CONTROL",
            active_providers=["groq", "gemini"],
            metrics=metrics,
        )

        assert winner == "groq"
        assert "Groq selected" in trace


class TestCircuitBreaker:

    def test_consecutive_429s_open_circuit_and_reroute_to_groq(self) -> None:
        metrics = ProviderMetricsManager()
        selector = ProviderSelector()

        for _ in range(3):
            metrics.record_event(_event("gemini", success=False, status_code=429))

        assert metrics.get_circuit_state("gemini") == CircuitState.OPEN
        assert metrics.health_score("gemini") == 0.0

        winner, trace = selector.select_provider(
            intent="CODE_GENERATION",  # Gemini has affinity here, but circuit is OPEN
            active_providers=["groq", "gemini"],
            metrics=metrics,
        )

        assert winner == "groq"
        assert "Circuit OPEN" in trace
        assert "Gemini" in trace

    def test_circuit_reopens_on_half_open_probe_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        metrics = ProviderMetricsManager()

        for _ in range(3):
            metrics.record_event(_event("gemini", success=False, status_code=429))

        # Simulate cooldown elapsed by rewinding the circuit's internal clock.
        circuit = metrics._circuits["gemini"]
        circuit._opened_at = time.monotonic() - 61.0

        assert metrics.get_circuit_state("gemini") == CircuitState.HALF_OPEN

        # Probe fails -> re-opens.
        metrics.record_event(_event("gemini", success=False, status_code=429))
        assert metrics.get_circuit_state("gemini") == CircuitState.OPEN

    def test_circuit_closes_on_half_open_probe_success(self) -> None:
        metrics = ProviderMetricsManager()

        for _ in range(3):
            metrics.record_event(_event("gemini", success=False, status_code=429))

        circuit = metrics._circuits["gemini"]
        circuit._opened_at = time.monotonic() - 61.0
        assert metrics.get_circuit_state("gemini") == CircuitState.HALF_OPEN

        metrics.record_event(_event("gemini", success=True))
        assert metrics.get_circuit_state("gemini") == CircuitState.CLOSED
        assert metrics.health_score("gemini") > 0.0


class TestSlidingWindow:

    def test_100_successes_clear_old_failure_influence(self) -> None:
        metrics = ProviderMetricsManager()

        # Two early failures, not enough to trip the 3-consecutive threshold.
        metrics.record_event(_event("groq", success=False, status_code=500))
        metrics.record_event(_event("groq", success=False, status_code=500))

        low_score = metrics.health_score("groq")
        assert low_score < 1.0

        # 100 successes push the deque(maxlen=100) window to drop both failures.
        for _ in range(100):
            metrics.record_event(_event("groq", success=True, latency_ms=0.0))

        assert metrics.window_size("groq") == 100
        high_score = metrics.health_score("groq")
        assert high_score == 1.0
        assert high_score > low_score

    def test_window_never_exceeds_maxlen(self) -> None:
        metrics = ProviderMetricsManager()
        for _ in range(250):
            metrics.record_event(_event("groq", success=True))
        assert metrics.window_size("groq") == 100