"""
security/auth.py — CYRAX 3.0 Security Guard

Implements SecurityGatewayProtocol (defined in core/context.py) and
SecurityLevel tiering used by tools/registry.py.

Design:
  - PIN hash is read from Pydantic settings (settings.CYRAX_PIN_HASH).
    Never from os.environ directly — settings validation already guarantees
    it is a valid bcrypt hash before this module is imported.
  - Lockout state is persisted via LockoutStore (device-global, survives restarts).
  - Session token is in-process only (by design for a local OS assistant).
  - All public methods are synchronous — called from registry.execute_tool()
    which is already running in a thread context.

Satisfies SecurityGatewayProtocol:
    def authorize_action(self, required_level: SecurityLevel) -> bool
"""

from __future__ import annotations

import logging
import secrets
import time
from enum import IntEnum

import bcrypt

from config.settings import settings
from security.lockout_store import LockoutStore

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY LEVEL
# ══════════════════════════════════════════════════════════════════════════════

class SecurityLevel(IntEnum):
    """
    Tiered permission levels for tools.
    IntEnum allows direct comparison: user_level >= required_level.

    UNRESTRICTED  — Safe read-only actions. No auth required.
    USER          — Standard actions (search, open apps). Valid session required.
    ADMIN         — Destructive or OS-level actions. PIN required per session.
    """
    UNRESTRICTED = 0
    USER         = 1
    ADMIN        = 2


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY GUARD
# ══════════════════════════════════════════════════════════════════════════════

class SecurityGuard:
    """
    Device-level authentication and authorisation gate.

    One instance per process. Constructed by bootstrap() and injected into
    ToolRegistry via the SecurityGatewayProtocol interface.

    State split:
        In-process:  Active session token + expiry (lost on restart by design).
        On-disk:     Lockout state via LockoutStore (survives restarts).

    The in-process/on-disk split is intentional:
        - Session tokens are ephemeral credentials — losing them on restart
          is correct behaviour (user re-authenticates after a restart).
        - Lockout state is a security enforcement mechanism — it must survive
          restarts so a reboot cannot be used to bypass brute-force protection.
    """

    def __init__(self, store: LockoutStore | None = None) -> None:
        # Allow store injection for testing; default to the real persistent store.
        self._store: LockoutStore = store or LockoutStore()

        # In-process session state.
        self._active_session_token: str | None = None
        self._session_expiry:       float      = 0.0

        # Load the PIN hash from validated Pydantic settings.
        # settings.CYRAX_PIN_HASH is guaranteed to be a valid bcrypt hash
        # by the field_validator in config/settings.py.
        pin_hash = settings.CYRAX_PIN_HASH
        if not pin_hash:
            # Should never reach here — settings validation enforces presence.
            # Defensive guard in case settings are constructed without validation.
            raise RuntimeError(
                "CYRAX_PIN_HASH is not set. "
                "SecurityGuard cannot be initialised without a PIN hash. "
                "Run bootstrap() after loading the environment."
            )
        self._hashed_pin: bytes = pin_hash.encode("utf-8")

        logger.info(
            f"[AUTH] SecurityGuard initialised. "
            f"Max retries: {settings.MAX_AUTH_RETRIES} | "
            f"Lockout duration: {settings.LOCKOUT_DURATION_SECONDS}s | "
            f"Session duration: {settings.SESSION_DURATION_SECONDS}s"
        )

    # ── SecurityGatewayProtocol ───────────────────────────────────────────────

    def authorize_action(self, required_level: SecurityLevel) -> bool:
        """
        Gate function called by ToolRegistry before every tool execution.

        UNRESTRICTED actions pass immediately.
        USER and ADMIN actions require a valid active session.

        Returns True to permit, False to deny.
        Denial triggers an auth_required response to the user.
        """
        if required_level == SecurityLevel.UNRESTRICTED:
            return True

        if required_level >= SecurityLevel.USER:
            if not self.is_session_valid():
                logger.warning(
                    f"[AUTH] Action blocked — no valid session. "
                    f"Required: {required_level.name}"
                )
                return False
            return True

        return False

    # ── Authentication ────────────────────────────────────────────────────────

    def authenticate(self, raw_pin: str) -> bool:
        """
        Validates a PIN attempt against the stored bcrypt hash.

        On success:
            - Resets lockout state in the persistent store.
            - Issues a new in-process session token.
            - Returns True.

        On failure:
            - Records the attempt in the persistent store.
            - Engages a device-global lockout if MAX_AUTH_RETRIES is exceeded.
            - Returns False.

        Returns False immediately if currently locked out.
        """
        locked, remaining = self.is_locked_out()
        if locked:
            logger.warning(
                f"[AUTH] Authentication blocked — system locked. "
                f"Remaining: {int(remaining)}s"
            )
            return False

        if not raw_pin or not raw_pin.strip():
            logger.warning("[AUTH] Empty PIN attempt rejected.")
            return False

        try:
            matched = bcrypt.checkpw(
                raw_pin.encode("utf-8"),
                self._hashed_pin,
            )
        except ValueError as exc:
            # Invalid hash format — should not happen if settings.py validated it.
            logger.error(f"[AUTH] bcrypt validation error: {exc}")
            return False

        if matched:
            self._store.reset()
            self._issue_session()
            logger.info("[AUTH] Authentication successful. Session issued.")
            return True
        else:
            failed_count = self._store.record_failed_attempt()
            logger.warning(
                f"[AUTH] Failed PIN attempt "
                f"({failed_count}/{settings.MAX_AUTH_RETRIES})."
            )

            if failed_count >= settings.MAX_AUTH_RETRIES:
                self._store.engage_lockout(settings.LOCKOUT_DURATION_SECONDS)
                self.clear_session()
                logger.critical(
                    f"[AUTH] MAX RETRIES EXCEEDED. "
                    f"Device locked for {settings.LOCKOUT_DURATION_SECONDS}s."
                )

            return False

    # ── Session Management ────────────────────────────────────────────────────

    def is_session_valid(self) -> bool:
        """
        Returns True if there is an active, unexpired in-process session.
        Clears the session automatically on expiry.
        """
        if not self._active_session_token:
            return False

        if time.time() > self._session_expiry:
            logger.info("[AUTH] Session expired — cleared.")
            self.clear_session()
            return False

        return True

    def clear_session(self) -> None:
        """
        Voids the current session.
        Called on lockout, explicit logout, or session expiry.
        """
        self._active_session_token = None
        self._session_expiry       = 0.0
        logger.info("[AUTH] Session cleared.")

    # ── Lockout Passthrough ───────────────────────────────────────────────────

    def is_locked_out(self) -> tuple[bool, float]:
        """
        Delegates to LockoutStore — device-global, persistent.

        Returns (locked: bool, seconds_remaining: float).
        Reads from disk on every call; no stale in-process cache.
        """
        return self._store.is_locked_out()

    def get_failed_attempts(self) -> int:
        """Returns the current persistent failed-attempt count."""
        return self._store.get_failed_attempts()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _issue_session(self) -> None:
        """
        Generates a cryptographically secure session token.
        Token and expiry are in-process only — not persisted to disk.
        """
        self._active_session_token = secrets.token_urlsafe(32)
        self._session_expiry       = time.time() + settings.SESSION_DURATION_SECONDS
        logger.debug(
            f"[AUTH] Session issued. "
            f"Expires in {settings.SESSION_DURATION_SECONDS}s."
        )