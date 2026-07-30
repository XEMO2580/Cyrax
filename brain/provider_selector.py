# brain/provider_selector.py

"""
brain/provider_selector.py — CYRAX 3.0 Phase 7.2 Adaptive Provider Selection

Replaces the static _PROVIDER_MATRIX. Final score = Task Affinity * Health Score.
Circuit-OPEN providers are excluded from the winner but still shown in the
trace string as skipped, so the decision remains fully auditable from logs.
"""

from __future__ import annotations

from brain.provider_metrics import CircuitState, ProviderMetricsManager

# Static Task Affinity matrix. Values are weights, not probabilities —
# only relative ordering matters since both candidates share the same
# health_score multiplier structure.
_TASK_AFFINITY: dict[str, dict[str, float]] = {
    "OS_CONTROL":       {"groq": 1.0, "gemini": 0.3},
    "FILE_OPERATION":   {"groq": 1.0, "gemini": 0.3},
    "WEB_RESEARCH":     {"groq": 1.0, "gemini": 0.4},
    "LIVE_INFORMATION": {"groq": 0.9, "gemini": 0.4},
    "GENERAL_CHAT":     {"groq": 0.8, "gemini": 0.5},
    "AUTOMATION":       {"groq": 1.0, "gemini": 0.3},
    "CODE_GENERATION":  {"groq": 0.3, "gemini": 1.0},
    "CONTENT_WRITING":  {"groq": 0.4, "gemini": 1.0},
    "REASONING":        {"groq": 0.3, "gemini": 1.0},
    "MEMORY":           {"groq": 0.4, "gemini": 0.9},
    "UNKNOWN":          {"groq": 0.6, "gemini": 0.6},
}

_DEFAULT_AFFINITY: float = 0.5


class ProviderSelector:
    """
    Stateless selection logic. Holds only the static affinity matrix —
    all dynamic state (health, circuit) lives in ProviderMetricsManager,
    injected per call rather than held as an instance dependency, so
    the same selector instance can be reused across requests safely.
    """

    def select_provider(
        self,
        intent:            str,
        active_providers:  list[str],
        metrics:           ProviderMetricsManager,
    ) -> tuple[str, str]:
        """
        Returns (winning_provider, trace_string).

        Final Score = Task Affinity(intent, provider) * Health Score(provider).
        Circuit-OPEN providers score 0.0 (via metrics.health_score's own
        forced-zero rule) and are reported as skipped in the trace, not
        silently omitted.

        If ALL candidates score 0.0 (e.g. every circuit is OPEN), falls
        back to the first entry in active_providers rather than raising —
        a degraded provider is still better than no response at all, and
        the trace string makes this fallback explicit.
        """
        affinity_row = _TASK_AFFINITY.get(intent, {})

        scored: list[tuple[str, float, bool]] = []  # (name, score, circuit_open)

        for provider_name in active_providers:
            affinity = affinity_row.get(provider_name, _DEFAULT_AFFINITY)
            health = metrics.health_score(provider_name)
            circuit_open = metrics.get_circuit_state(provider_name) == CircuitState.OPEN
            final_score = affinity * health
            scored.append((provider_name, final_score, circuit_open))

        scored.sort(key=lambda item: item[1], reverse=True)

        trace_parts: list[str] = []
        for name, score, circuit_open in scored:
            if circuit_open:
                trace_parts.append(f"{name.capitalize()} skipped (Circuit OPEN).")
            else:
                trace_parts.append(f"{name.capitalize()} scored {score:.2f}.")

        winner_name, winner_score, winner_circuit_open = scored[0]

        if winner_score <= 0.0 and winner_circuit_open:
            # Every candidate is circuit-OPEN — degrade rather than fail.
            winner_name = active_providers[0]
            trace = (
                f"All providers circuit-OPEN or zero-scored. "
                f"Degraded fallback to '{winner_name}'. " + " ".join(trace_parts)
            )
        else:
            trace = (
                f"{winner_name.capitalize()} selected (Score: {winner_score:.2f}). "
                + " ".join(p for n, s, c in scored if n != winner_name
                           for p in [f"{n.capitalize()} " + ("skipped (Circuit OPEN)." if c else f"scored {s:.2f}.")])
            )

        return winner_name, trace