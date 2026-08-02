# brain/learning_router.py

"""
brain/learning_router.py — CYRAX 3.0 Phase 7.3 Adaptive Decision Policy

Sits between DecisionEngine and MoERouter/Planner. Pure mathematical
scoring — no neural nets, no ML libraries (Constraint 4). Reads
persisted routing_history from SQLiteJobStore, applies exponential
time-decay so recent outcomes outweigh stale ones, and enforces an
anti-oscillation margin before ever overriding DecisionEngine's own
recommended_provider.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ── Constraint 2: anti-oscillation margin ─────────────────────────────────
_SWITCH_MARGIN: float = 0.05  # 5%

# ── Exponential decay: half-life in hours. A routing_history row's weight
# halves every _DECAY_HALF_LIFE_HOURS. Chosen so a burst of 429s from
# yesterday stops dominating today's decision but isn't discarded outright.
_DECAY_HALF_LIFE_HOURS: float = 6.0

_HISTORY_WINDOW_SIZE: int = 200


class IntentFamily(str, Enum):
    OS_CONTROL       = "OS_CONTROL"
    CODE_GENERATION  = "CODE_GENERATION"
    GENERAL_CHAT     = "GENERAL_CHAT"
    WEB_RESEARCH     = "WEB_RESEARCH"
    FILE_OPERATIONS  = "FILE_OPERATIONS"
    REASONING        = "REASONING"
    EMAIL            = "EMAIL"
    MEMORY           = "MEMORY"
    MEDIA            = "MEDIA"
    SYSTEM           = "SYSTEM"


# Base Task Affinity — same role as Phase 7.2's static matrix, kept here
# rather than importing brain/provider_selector.py's copy, since this
# taxonomy (IntentFamily) is intentionally coarser than Decision Engine's
# 11-category IntentCategory — LearningRouter operates one layer up.
_BASE_AFFINITY: dict[str, dict[str, float]] = {
    "OS_CONTROL":      {"groq": 1.0, "gemini": 0.8},
    "CODE_GENERATION": {"groq": 0.8, "gemini": 1.0},
    "GENERAL_CHAT":    {"groq": 0.9, "gemini": 0.7},
    "WEB_RESEARCH":    {"groq": 1.0, "gemini": 0.7},
    "FILE_OPERATIONS": {"groq": 1.0, "gemini": 0.7},
    "REASONING":       {"groq": 0.6, "gemini": 1.0},
    "EMAIL":           {"groq": 0.8, "gemini": 0.9},
    "MEMORY":          {"groq": 0.7, "gemini": 0.9},
    "MEDIA":           {"groq": 1.0, "gemini": 0.6},
    "SYSTEM":          {"groq": 1.0, "gemini": 0.6},
}
_DEFAULT_AFFINITY: float = 0.5

_LATENCY_PENALTY_MS: float = 5000.0


class LearningRouter:
    """
    Args:
        job_store: SQLiteJobStore (or anything satisfying its
                   record_routing_outcome/get_routing_stats contract).
                   Injected, per the established DI pattern.
    """

    def __init__(self, job_store: Any) -> None:
        self._job_store = job_store

    async def select_provider(
        self,
        intent_family:       str,
        recommended_provider: str,
        trace_id:             str,
        active_providers:     list[str] | None = None,
    ) -> tuple[str, str]:
        """
        Returns (selected_provider, explanation_log).

        Constraint 1 — Deterministic Cold Start: if get_routing_stats()
        returns no history at all for this intent_family, this method
        does NOT compute any score — it returns recommended_provider
        immediately with an explicit "cold start" explanation, so the
        DecisionEngine's own judgement is the sole authority until real
        data exists.

        Constraint 2 — Anti-Oscillation: even with history present, a
        candidate provider only replaces recommended_provider if its
        final score exceeds recommended_provider's score by at least
        _SWITCH_MARGIN (5%). A smaller edge is treated as noise, not
        signal, and recommended_provider is kept.
        """
        active = active_providers or ["groq", "gemini"]

        history = await self._job_store.get_routing_stats(
            intent_family, limit=_HISTORY_WINDOW_SIZE
        )

        # ── Constraint 1: Deterministic Cold Start ────────────────────────────
        if not history:
            explanation = (
                f"[LEARNING_ROUTER] {trace_id} | intent={intent_family} | "
                f"COLD START (no routing_history) — deferring to "
                f"DecisionEngine recommendation: '{recommended_provider}'."
            )
            logger.info(explanation)
            return recommended_provider, explanation

        # ── Score every active provider ───────────────────────────────────────
        scored: dict[str, dict[str, float]] = {}
        for provider_name in active:
            components = self._score_provider(intent_family, provider_name, history)
            scored[provider_name] = components

        final_scores = {
            name: comps["final_score"] for name, comps in scored.items()
        }

        recommended_score = final_scores.get(recommended_provider, 0.0)
        best_candidate = max(final_scores, key=lambda p: final_scores[p])
        best_score = final_scores[best_candidate]

        # ── Constraint 2: Anti-Oscillation ────────────────────────────────────
        switch_triggered = False
        if best_candidate != recommended_provider:
            if recommended_score <= 0.0:
                # No signal to compare against — allow switch if any real
                # candidate scores positively.
                switch_triggered = best_score > 0.0
            else:
                relative_gain = (best_score - recommended_score) / recommended_score
                switch_triggered = relative_gain >= _SWITCH_MARGIN

        selected = best_candidate if switch_triggered else recommended_provider

        explanation = self._build_explanation(
            trace_id=trace_id,
            intent_family=intent_family,
            recommended_provider=recommended_provider,
            recommended_score=recommended_score,
            best_candidate=best_candidate,
            best_score=best_score,
            switch_triggered=switch_triggered,
            selected=selected,
            scored=scored,
        )

        logger.info(explanation)
        return selected, explanation

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_provider(
        self,
        intent_family: str,
        provider_name: str,
        history:       list[dict],
    ) -> dict[str, float]:
        """
        Final Score = Affinity * Health * LearningScore * RecentPerformance.

        Affinity:          static base matrix (this provider's fit for
                           this intent family).
        Health:             success_rate over provider-specific records
                           in history, exponentially time-decayed.
        LearningScore:      inverse-latency-weighted confidence — rewards
                           consistently fast AND successful providers over
                           ones that are merely occasionally lucky.
        RecentPerformance:  weight of the single most recent record for
                           this provider — a fresh success/failure signal,
                           separate from the decayed aggregate, so one very
                           recent event has an immediate, visible effect
                           on top of the smoothed long-run average.
        """
        affinity = _BASE_AFFINITY.get(intent_family, {}).get(provider_name, _DEFAULT_AFFINITY)

        provider_records = [r for r in history if r["provider"] == provider_name]

        if not provider_records:
            # No data for THIS provider specifically (even though the
            # intent_family has history overall, from other providers).
            # Score at a neutral midpoint rather than 0 — an untried
            # provider shouldn't be scored as "known bad."
            return {
                "affinity":           affinity,
                "health":             0.5,
                "learning_score":     0.5,
                "recent_performance": 0.5,
                "final_score":        affinity * 0.5 * 0.5 * 0.5,
            }

        now = datetime.now(timezone.utc)
        weighted_success_sum = 0.0
        weighted_total = 0.0
        weighted_latency_sum = 0.0

        for record in provider_records:
            record_time = datetime.fromisoformat(record["timestamp"])
            age_hours = max(0.0, (now - record_time).total_seconds() / 3600.0)
            weight = self._decay_weight(age_hours)

            weighted_success_sum += weight * (1.0 if record["success"] else 0.0)
            weighted_total += weight
            weighted_latency_sum += weight * record["latency_ms"]

        health = weighted_success_sum / weighted_total if weighted_total > 0 else 0.5
        avg_weighted_latency = weighted_latency_sum / weighted_total if weighted_total > 0 else 0.0
        latency_factor = max(0.0, 1.0 - (avg_weighted_latency / _LATENCY_PENALTY_MS))

        learning_score = health * latency_factor

        most_recent = provider_records[0]  # get_routing_stats() returns DESC order
        recent_performance = 1.0 if most_recent["success"] else 0.2

        final_score = affinity * health * learning_score * recent_performance

        return {
            "affinity":           round(affinity, 4),
            "health":             round(health, 4),
            "learning_score":     round(learning_score, 4),
            "recent_performance": round(recent_performance, 4),
            "final_score":        round(final_score, 4),
        }

    @staticmethod
    def _decay_weight(age_hours: float) -> float:
        """
        Exponential decay: weight = 0.5 ** (age_hours / half_life).
        At age_hours == 0, weight == 1.0. At age_hours == half_life,
        weight == 0.5. Pure math, no ML — Constraint 4.
        """
        return math.pow(0.5, age_hours / _DECAY_HALF_LIFE_HOURS)

    # ── Constraint 3: Full Explainability ──────────────────────────────────────

    @staticmethod
    def _build_explanation(
        trace_id:              str,
        intent_family:         str,
        recommended_provider:  str,
        recommended_score:     float,
        best_candidate:        str,
        best_score:            float,
        switch_triggered:      bool,
        selected:               str,
        scored:                 dict[str, dict[str, float]],
    ) -> str:
        component_lines = []
        for provider_name, comps in scored.items():
            component_lines.append(
                f"{provider_name}: Affinity({comps['affinity']:.2f}) x "
                f"Health({comps['health']:.2f}) x "
                f"Learning({comps['learning_score']:.2f}) x "
                f"Recent({comps['recent_performance']:.2f}) "
                f"= {comps['final_score']:.4f}"
            )

        decision_line = (
            f"SWITCHED to '{best_candidate}' "
            f"(beat '{recommended_provider}' by "
            f"{((best_score - recommended_score) / recommended_score * 100) if recommended_score > 0 else float('inf'):.1f}%, "
            f">= {_SWITCH_MARGIN * 100:.0f}% margin required)"
            if switch_triggered else
            f"KEPT recommended '{recommended_provider}' "
            f"(candidate '{best_candidate}' did not clear the "
            f"{_SWITCH_MARGIN * 100:.0f}% anti-oscillation margin)"
        )

        return (
            f"[LEARNING_ROUTER] {trace_id} | intent={intent_family} | "
            f"{decision_line} | selected='{selected}' | "
            f"scores: {' | '.join(component_lines)}"
        )