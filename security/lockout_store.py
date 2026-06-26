"""
security/lockout_store.py — CYRAX 3.0 Device-Global Lockout Persistence

Persists lockout state across process restarts.
Scoped to the DEVICE, not to a session. A lockout triggered on any
session freezes the entire OS execution layer until it expires.

Storage: JSON file written atomically via a temp-file + rename pattern.
Atomic writes prevent a corrupt store from a mid-write crash.

No session_id parameter anywhere in this file — device-global is enforced
by design, not by convention.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Storage path ──────────────────────────────────────────────────────────────
# Resolved relative to the project root so it is stable regardless of the
# working directory the process is launched from.

_MODULE_DIR   = Path(__file__).resolve().parent        # cyrax/security/
_PROJECT_ROOT = _MODULE_DIR.parent                     # cyrax/
_DEFAULT_STORE_PATH = _PROJECT_ROOT / "security" / ".lockout_store.json"

# Keys stored in the JSON file.
_KEY_FAILED_ATTEMPTS = "failed_attempts"
_KEY_LOCKOUT_UNTIL   = "lockout_until"     # Unix timestamp (float). 0.0 = not locked.
_KEY_LAST_UPDATED    = "last_updated"      # ISO timestamp for human inspection.


class LockoutStore:
    """
    Device-global persistent lockout ledger.

    Reads from disk on every query (no in-process cache) so state is
    consistent even if multiple processes access the same store.

    All writes are atomic: data is written to a temp file in the same
    directory, then renamed over the target. On POSIX this is atomic;
    on Windows it requires os.replace() which is also atomic since Python 3.3.
    """

    def __init__(self, store_path: Path = _DEFAULT_STORE_PATH) -> None:
        self._path = store_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if not self._path.exists():
            self._write(_empty_record())
            logger.debug(f"[LOCKOUT_STORE] Initialised at {self._path}")
        else:
            logger.debug(f"[LOCKOUT_STORE] Loaded existing store at {self._path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def is_locked_out(self) -> tuple[bool, float]:
        """
        Returns (locked: bool, seconds_remaining: float).

        Reads from disk on every call — no stale in-process cache.
        If the lockout has expired since the last write, clears it and
        returns (False, 0.0) so subsequent reads don't re-trigger.
        """
        record = self._read()
        lockout_until: float = record.get(_KEY_LOCKOUT_UNTIL, 0.0)

        if lockout_until == 0.0:
            return False, 0.0

        remaining = lockout_until - time.time()
        if remaining <= 0:
            # Lockout has expired — clear it so it doesn't show up again.
            self._clear_lockout(record)
            return False, 0.0

        return True, remaining

    def record_failed_attempt(self) -> int:
        """
        Increments the failed-attempt counter and persists it.
        Returns the new failed-attempt count.
        """
        record = self._read()
        record[_KEY_FAILED_ATTEMPTS] = record.get(_KEY_FAILED_ATTEMPTS, 0) + 1
        record[_KEY_LAST_UPDATED] = _now_iso()
        self._write(record)

        count: int = record[_KEY_FAILED_ATTEMPTS]
        logger.warning(
            f"[LOCKOUT_STORE] Failed attempt recorded. "
            f"Total: {count}"
        )
        return count

    def engage_lockout(self, duration_seconds: int) -> None:
        """
        Sets the lockout expiry timestamp.
        Device-global: no session_id, no tenant scope.

        Args:
            duration_seconds: How long to lock the system.
        """
        record = self._read()
        record[_KEY_LOCKOUT_UNTIL]   = time.time() + duration_seconds
        record[_KEY_FAILED_ATTEMPTS] = 0          # Reset counter after lockout.
        record[_KEY_LAST_UPDATED]    = _now_iso()
        self._write(record)

        logger.critical(
            f"[LOCKOUT_STORE] DEVICE LOCKOUT ENGAGED. "
            f"Duration: {duration_seconds}s. "
            f"Expires: {record[_KEY_LOCKOUT_UNTIL]}"
        )

    def reset(self) -> None:
        """
        Clears failed attempts and any active lockout.
        Called on successful authentication.
        """
        self._write(_empty_record())
        logger.info("[LOCKOUT_STORE] State reset after successful authentication.")

    def get_failed_attempts(self) -> int:
        """Returns the current failed-attempt count without side effects."""
        return self._read().get(_KEY_FAILED_ATTEMPTS, 0)

    # ── Internal I/O ─────────────────────────────────────────────────────────

    def _read(self) -> dict[str, Any]:
        """
        Reads the store from disk.
        Returns an empty record on any read failure to prevent boot crash.
        """
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("Store root is not a dict.")
                return data
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            logger.error(
                f"[LOCKOUT_STORE] Read failed ({exc}). "
                f"Returning empty record — store will be reset on next write."
            )
            return _empty_record()

    def _write(self, record: dict[str, Any]) -> None:
        """
        Atomically writes the record to disk via temp-file + os.replace().
        Prevents a corrupt store from a mid-write crash or power loss.
        """
        try:
            dir_path = self._path.parent
            fd, tmp_path = tempfile.mkstemp(
                dir=dir_path,
                prefix=".lockout_tmp_",
                suffix=".json",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)
                os.replace(tmp_path, self._path)
            except Exception:
                # Clean up the temp file if the write or rename fails.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.error(f"[LOCKOUT_STORE] Atomic write failed: {exc}")

    def _clear_lockout(self, record: dict[str, Any]) -> None:
        """Zeroes the lockout timestamp and persists the cleared record."""
        record[_KEY_LOCKOUT_UNTIL] = 0.0
        record[_KEY_LAST_UPDATED]  = _now_iso()
        self._write(record)
        logger.info("[LOCKOUT_STORE] Expired lockout cleared from disk.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_record() -> dict[str, Any]:
    return {
        _KEY_FAILED_ATTEMPTS: 0,
        _KEY_LOCKOUT_UNTIL:   0.0,
        _KEY_LAST_UPDATED:    _now_iso(),
    }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()