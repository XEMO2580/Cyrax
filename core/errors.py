# core/errors.py — new file: Item 4, sanitized error taxonomy

"""
core/errors.py — CYRAX 3.0 Sanitized Error Mapping

Central mapping from internal exceptions to safe, structured error
payloads. NEVER let a raw exception string (str(exc), traceback, Python
type name) reach an API response body or a client-visible field.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SanitizedError:
    error_code: str
    message:    str


_DEFAULT_ERROR = SanitizedError(
    error_code="INTERNAL_ERROR",
    message="CYRAX couldn't complete that request. Please try again.",
)

_ERROR_MAP: dict[type, SanitizedError] = {}


def register_error_mapping(exc_type: type, error_code: str, message: str) -> None:
    _ERROR_MAP[exc_type] = SanitizedError(error_code=error_code, message=message)


def sanitize_exception(exc: BaseException) -> SanitizedError:
    """
    Never returns exc's raw message, type name, or traceback. Looks up a
    registered mapping by exact type; falls back to _DEFAULT_ERROR for
    anything unmapped (including genuinely unexpected internal bugs —
    those are logged server-side with full detail, never surfaced).
    """
    for exc_type, sanitized in _ERROR_MAP.items():
        if isinstance(exc, exc_type):
            return sanitized
    return _DEFAULT_ERROR


# ── Registered mappings ──────────────────────────────────────────────────────

from brain.providers.base import ProviderError, GenerationCancelledError

register_error_mapping(
    ProviderError, "PROVIDER_EXECUTION_FAILED",
    "CYRAX couldn't reach the AI provider. Please try again.",
)
register_error_mapping(
    GenerationCancelledError, "TASK_CANCELLED",
    "The task was cancelled.",
)
register_error_mapping(
    TimeoutError, "REQUEST_TIMEOUT",
    "The request took too long to complete. Please try again.",
)