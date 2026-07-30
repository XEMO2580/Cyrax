# brain/provider_metrics.py

"""
brain/provider_metrics.py — CYRAX 3.0 Phase 7.2 Provider Metrics & Circuit Breaker

Purely in-memory. collections.deque(maxlen=100) sliding window per provider.
No database, no ML — health scoring is a deterministic formula over the
window's contents.

Circuit Breaker states:
    CLOSED     — healthy, provider is selectable normally.
    OPEN       — tripped, health_score forced to 0.0, provider excluded
                 from selection for the cooldown duration.
    HALF_OPEN  — cooldown elapsed, one probe attempt permitted; a success
                 closes the circuit, a failure re-opens it and resets the
                 cooldown timer.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from enum import Enum, auto

from brain.provider_events import ProviderExecutionEvent

logger = logging.getLogger(__name__)

_WINDOW_SIZE: int = 100
_CONSECUTIVE_FAILURE_THRESHOLD: int = 3
_COOLDOWN_SECONDS: float = 60.0
_LATENCY_PENALTY_MS: float = 5000.0  # latency at/above this floors the latency term to 0


class CircuitState(Enum):
    CLOSED    = auto()
    OPEN      = auto()
    HALF_OPEN = auto()


class _ProviderCircuit:
    """
    Per-provider circuit breaker state. Not exposed outside
    ProviderMetricsManager — internal bookkeeping only.
    """

    def __init__(self) -> None:
        self.state: CircuitState = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._consecutive_failures: int = 0

    def record_success(self) -> None:
        self._consecutive_failures = 0
        if self.state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
            logger.info("[CIRCUIT_BREAKER] Probe succeeded — closing circuit.")
        self.state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1

        if self.state == CircuitState.HALF_OPEN:
            logger.warning(
                "[CIRCUIT_BREAKER] Probe failed during HALF_OPEN — "
                "re-opening circuit, resetting cooldown."
            )
            self._trip()
            return

        if self._consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD:
            self._trip()

    def _trip(self) -> None:
        self.state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        logger.warning(
            f"[CIRCUIT_BREAKER] Circuit OPEN. "
            f"Cooldown: {_COOLDOWN_SECONDS}s."
        )

    def refresh(self) -> CircuitState:
        """
        Call before reading state. Transitions OPEN -> HALF_OPEN once the
        cooldown window has elapsed.
        """
        if self.state == CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= _COOLDOWN_SECONDS:
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    "[CIRCUIT_BREAKER] Cooldown elapsed — HALF_OPEN, "
                    "permitting one probe attempt."
                )
        return self.state


class ProviderMetricsManager:
    """
    Rolling-window metrics and circuit breaker state, per provider.

    Thread/coroutine safety: a single threading.Lock guards all mutation.
    generate() calls happen on the event loop (async), but metrics
    recording is a fast, synchronous, in-memory operation — a plain Lock
    is sufficient and avoids introducing asyncio.Lock coupling into a
    module that has no other async surface.
    """

    def __init__(self) -> None:
        self._windows: dict[str, deque[ProviderExecutionEvent]] = {}
        self._circuits: dict[str, _ProviderCircuit] = {}
        self._lock = threading.Lock()

    def record_event(self, event: ProviderExecutionEvent) -> None:
        with self._lock:
            window = self._windows.setdefault(
                event.provider, deque(maxlen=_WINDOW_SIZE)
            )
            window.append(event)

            circuit = self._circuits.setdefault(event.provider, _ProviderCircuit())
            if event.success:
                circuit.record_success()
            else:
                circuit.record_failure()

        logger.debug(
            f"[METRICS] provider={event.provider} | intent={event.intent} | "
            f"success={event.success} | status={event.status_code} | "
            f"latency={event.latency_ms:.0f}ms"
        )

    def get_circuit_state(self, provider: str) -> CircuitState:
        with self._lock:
            circuit = self._circuits.get(provider)
            if circuit is None:
                return CircuitState.CLOSED
            return circuit.refresh()

    def health_score(self, provider: str) -> float:
        """
        Returns 0.0 if the circuit is OPEN (forced, per Constraint).
        Otherwise: success_rate * latency_factor, both in [0, 1].

        success_rate:   fraction of events in the window with success=True.
                        No events yet -> 1.0 (optimistic default, unproven
                        provider is not penalised before it's ever run).
        latency_factor: 1.0 at 0ms, linearly decaying to 0.0 at
                        _LATENCY_PENALTY_MS, averaged over the window.
        """
        state = self.get_circuit_state(provider)
        if state == CircuitState.OPEN:
            return 0.0

        with self._lock:
            window = self._windows.get(provider)

        if not window:
            return 1.0

        successes = sum(1 for e in window if e.success)
        success_rate = successes / len(window)

        avg_latency = sum(e.latency_ms for e in window) / len(window)
        latency_factor = max(0.0, 1.0 - (avg_latency / _LATENCY_PENALTY_MS))

        score = success_rate * latency_factor
        return round(max(0.0, min(1.0, score)), 4)

    def window_size(self, provider: str) -> int:
        with self._lock:
            window = self._windows.get(provider)
            return len(window) if window else 0