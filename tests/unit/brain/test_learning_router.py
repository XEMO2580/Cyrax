# tests/unit/brain/test_learning_router.py

"""
tests/unit/brain/test_learning_router.py — Phase 7.3 LearningRouter certification.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from brain.learning_router import LearningRouter, _SWITCH_MARGIN

pytestmark = pytest.mark.asyncio


def _make_record(
    provider:     str,
    success:      bool,
    latency_ms:   float = 200.0,
    hours_ago:    float = 0.0,
    fallback_used: bool = False,
) -> dict:
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {
        "provider":      provider,
        "success":       success,
        "latency_ms":    latency_ms,
        "fallback_used": fallback_used,
        "timestamp":     ts.isoformat(),
        "reason":        "",
    }


class TestColdStart:

    async def test_empty_history_defers_to_decision_engine(self) -> None:
        job_store = AsyncMock()
        job_store.get_routing_stats.return_value = []

        router = LearningRouter(job_store=job_store)
        selected, explanation = await router.select_provider(
            intent_family="CODE_GENERATION",
            recommended_provider="gemini",
            trace_id="t1",
        )

        assert selected == "gemini"
        assert "COLD START" in explanation


class TestSuccessBoost:

    async def test_100_successful_code_generation_requests_boost_score(self) -> None:
        job_store = AsyncMock()
        job_store.get_routing_stats.return_value = [
            _make_record("gemini", success=True, latency_ms=100.0, hours_ago=i * 0.01)
            for i in range(100)
        ]

        router = LearningRouter(job_store=job_store)
        comps = router._score_provider(
            "CODE_GENERATION", "gemini", job_store.get_routing_stats.return_value
        )

        assert comps["health"] > 0.95
        assert comps["final_score"] > 0.5  # affinity(1.0) x high health x high learning x recent(1.0)


class TestFailurePenalty:

    async def test_repeated_429s_rapidly_lower_score(self) -> None:
        job_store = AsyncMock()
        job_store.get_routing_stats.return_value = [
            _make_record("gemini", success=False, latency_ms=50.0, hours_ago=i * 0.01)
            for i in range(10)
        ]

        router = LearningRouter(job_store=job_store)
        comps = router._score_provider(
            "CODE_GENERATION", "gemini", job_store.get_routing_stats.return_value
        )

        assert comps["health"] < 0.1
        assert comps["recent_performance"] == 0.2  # most recent record was also a failure
        assert comps["final_score"] < 0.1


class TestAntiOscillation:

    async def test_1_percent_difference_does_not_trigger_switch(self) -> None:
        job_store = AsyncMock()

        # Groq (recommended): slightly lower but nearly tied score.
        # Gemini: marginally higher but within the 5% margin.
        history = (
            [_make_record("groq", success=True, latency_ms=100.0, hours_ago=0.01) for _ in range(20)]
            + [_make_record("gemini", success=True, latency_ms=95.0, hours_ago=0.01) for _ in range(20)]
        )
        job_store.get_routing_stats.return_value = history

        router = LearningRouter(job_store=job_store)
        selected, explanation = await router.select_provider(
            intent_family="OS_CONTROL",  # groq affinity 1.0, gemini affinity 0.8 — groq favored
            recommended_provider="groq",
            trace_id="t2",
        )

        assert selected == "groq"
        assert "KEPT" in explanation

    async def test_greater_than_5_percent_difference_triggers_switch(self) -> None:
        job_store = AsyncMock()

        # Groq (recommended): consistent failures.
        # Gemini: consistent fast successes. Should clear 5% easily.
        history = (
            [_make_record("groq", success=False, latency_ms=4000.0, hours_ago=0.01) for _ in range(20)]
            + [_make_record("gemini", success=True, latency_ms=100.0, hours_ago=0.01) for _ in range(20)]
        )
        job_store.get_routing_stats.return_value = history

        router = LearningRouter(job_store=job_store)
        selected, explanation = await router.select_provider(
            intent_family="CODE_GENERATION",
            recommended_provider="groq",
            trace_id="t3",
        )

        assert selected == "gemini"
        assert "SWITCHED" in explanation


class TestExponentialDecay:

    async def test_recent_successes_outweigh_stale_failures(self) -> None:
        job_store = AsyncMock()

        # Old failures (48h ago, well past the 6h half-life — heavily decayed)
        # Recent successes (minutes ago — near full weight).
        history = (
            [_make_record("gemini", success=False, latency_ms=3000.0, hours_ago=48.0) for _ in range(30)]
            + [_make_record("gemini", success=True, latency_ms=100.0, hours_ago=0.02) for _ in range(10)]
        )
        job_store.get_routing_stats.return_value = history

        router = LearningRouter(job_store=job_store)
        comps = router._score_provider("CODE_GENERATION", "gemini", history)

        # Health should be dominated by the recent successes, not the
        # numerically-more-numerous but heavily-decayed old failures.
        assert comps["health"] > 0.7

    async def test_decay_weight_halves_at_half_life(self) -> None:
        router = LearningRouter(job_store=AsyncMock())
        w0 = router._decay_weight(0.0)
        w_half_life = router._decay_weight(6.0)

        assert w0 == pytest.approx(1.0)
        assert w_half_life == pytest.approx(0.5, abs=0.01)


class TestExplainability:

    async def test_select_provider_returns_nonempty_explanation_with_scores(self) -> None:
        job_store = AsyncMock()
        history = [_make_record("groq", success=True, hours_ago=0.01) for _ in range(5)]
        job_store.get_routing_stats.return_value = history

        router = LearningRouter(job_store=job_store)
        selected, explanation = await router.select_provider(
            intent_family="GENERAL_CHAT",
            recommended_provider="groq",
            trace_id="t4",
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "Affinity" in explanation
        assert "Health" in explanation
        assert "Learning" in explanation
        assert "Recent" in explanation
        assert "t4" in explanation
        assert "GENERAL_CHAT" in explanation