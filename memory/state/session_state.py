"""
memory/state/session_state.py — CYRAX 3.0 Session State Store

Ephemeral, in-process key-value store for session-scoped runtime variables.
Not persisted to disk — intentional. Session state is transient by design:
pending auth flows, last opened app, supervisor reasoning traces.

Pre-declared fields document the full set of keys the orchestrator layer
uses. Arbitrary keys are also permitted via get/set for future expansion.

Device-local: one instance per process, scoped to the active session.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PRE-DECLARED STATE KEYS
# ══════════════════════════════════════════════════════════════════════════════

# Centralised key constants prevent typo-driven bugs across dispatcher,
# supervisor, and any future orchestration layer that reads session state.

KEY_PENDING_AUTH             = "pending_auth"
KEY_PENDING_TOOL             = "pending_tool"
KEY_PENDING_ARGS             = "pending_args"
KEY_LAST_OPENED_APP          = "last_opened_app"
KEY_SUPERVISOR_TRACE         = "supervisor_reasoning_trace"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE STORE
# ══════════════════════════════════════════════════════════════════════════════

class SessionStateStore:
    """
    In-process ephemeral key-value store for session-scoped variables.

    Satisfies the 'state' attribute contract of MemoryStackProtocol:
        def get(self, key: str) -> Any
        def set(self, key: str, value: Any) -> None
        def clear(self, key: str) -> None

    Pre-declared fields:
        pending_auth             bool        — True when awaiting a PIN entry
        pending_tool             str | None  — Tool name awaiting auth replay
        pending_args             dict | None — Args for the pending tool replay
        last_opened_app          str | None  — Most recently opened app name
        supervisor_reasoning_trace list      — Supervisor recovery audit trail

    All pre-declared fields are initialised to safe defaults so reads never
    raise KeyError regardless of call order.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {
            KEY_PENDING_AUTH:             False,
            KEY_PENDING_TOOL:             None,
            KEY_PENDING_ARGS:             None,
            KEY_LAST_OPENED_APP:          None,
            KEY_SUPERVISOR_TRACE:         [],
        }
        logger.debug("[SESSION_STATE] Initialised with default fields.")

    # ── Core Interface ────────────────────────────────────────────────────────

    def get(self, key: str) -> Any:
        """
        Returns the value for key, or None if the key has never been set.
        Never raises KeyError.
        """
        value = self._data.get(key)
        logger.debug(f"[SESSION_STATE] get({key!r}) → {value!r}")
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Sets key to value.
        Accepts any key — pre-declared fields and arbitrary runtime keys alike.
        """
        logger.debug(f"[SESSION_STATE] set({key!r}, {value!r})")
        self._data[key] = value

    def clear(self, key: str) -> None:
        """
        Removes key from the store.
        No-op if the key does not exist — never raises.
        """
        removed = self._data.pop(key, None)
        logger.debug(f"[SESSION_STATE] clear({key!r}) — removed={removed!r}")

    # ── Bulk Operations ───────────────────────────────────────────────────────

    def clear_auth_state(self) -> None:
        """
        Atomically clears all auth-flow keys in one call.
        Used by dispatcher after a PIN attempt succeeds or a lockout fires.
        """
        self.clear(KEY_PENDING_AUTH)
        self.clear(KEY_PENDING_TOOL)
        self.clear(KEY_PENDING_ARGS)
        logger.debug("[SESSION_STATE] Auth state cleared.")

    def reset(self) -> None:
        """
        Resets all fields to their initial defaults.
        Called when a session ends or is explicitly invalidated.
        """
        self._data = {
            KEY_PENDING_AUTH:             False,
            KEY_PENDING_TOOL:             None,
            KEY_PENDING_ARGS:             None,
            KEY_LAST_OPENED_APP:          None,
            KEY_SUPERVISOR_TRACE:         [],
        }
        logger.info("[SESSION_STATE] Full reset to defaults.")

    # ── Supervisor Trace Helpers ──────────────────────────────────────────────

    def append_supervisor_trace(self, entry: dict[str, Any]) -> None:
        """
        Appends a reasoning step to the supervisor trace log.
        The trace is a list of dicts recording why the supervisor made
        each recovery decision during a multi-step execution.
        """
        trace: list = self._data.setdefault(KEY_SUPERVISOR_TRACE, [])
        trace.append(entry)
        logger.debug(
            f"[SESSION_STATE] Supervisor trace appended. "
            f"Total entries: {len(trace)}"
        )

    def get_supervisor_trace(self) -> list[dict[str, Any]]:
        """Returns the full supervisor reasoning trace for the current turn."""
        return list(self._data.get(KEY_SUPERVISOR_TRACE, []))

    def clear_supervisor_trace(self) -> None:
        """Clears the supervisor trace at the start of each new user turn."""
        self._data[KEY_SUPERVISOR_TRACE] = []
        logger.debug("[SESSION_STATE] Supervisor trace cleared.")

    # ── Inspection ────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """
        Returns a shallow copy of the current state dict.
        Safe for logging and debugging — does not expose the live dict.
        """
        return dict(self._data)

    def __repr__(self) -> str:
        return f"SessionStateStore({self._data!r})"