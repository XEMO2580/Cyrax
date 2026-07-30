# brain/provider_events.py

"""
brain/provider_events.py — CYRAX 3.0 Phase 7.2 Provider Execution Events

Pure data. No behavior, no dependencies on ProviderMetricsManager or
ProviderSelector — those consume this type, this type doesn't know about them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class ProviderExecutionEvent:
    """
    One recorded outcome of a single provider.generate() call.

    Attributes:
        provider:     Provider name ("groq", "gemini", etc.).
        intent:       The classified intent this call was serving
                      (from CognitiveRoutingDecision.intent), or "unknown"
                      if the caller has no intent context.
        latency_ms:   Wall-clock duration of the generate() call, in ms.
        success:      True if the call returned normally.
        status_code:  HTTP-style status code. 0 for success/no-code errors,
                      the ProviderError.status_code on failure.
        timestamp:    UTC timestamp of the event.
    """
    provider:    str
    intent:      str
    latency_ms:  float
    success:     bool
    status_code: int
    timestamp:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))