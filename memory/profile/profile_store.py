"""
memory/profile/profile_store.py — CYRAX 3.0 User Profile Store (Patched)

Patch applied vs. v1 — Cross-loop lock corruption fix:

  ROOT CAUSE: asyncio.Lock(), when created, binds internally to whatever
  event loop is running at construction time. The original design called
  asyncio.run(_load_profile(...)) during synchronous bootstrap — this
  creates a temporary event loop, runs load() on it, and immediately
  closes that loop. If self._lock had been instantiated during that
  call (or earlier in __init__ while no loop was running), any later
  asyncio.Lock acquisition on the REAL application event loop would
  raise RuntimeError: Lock is bound to a different event loop.

  FIX:
    1. Disk read happens synchronously in __init__ — no event loop
       involvement at all for the initial load. bootstrap() no longer
       needs asyncio.run() or a _load_profile() helper.
    2. self._lock = None at construction. No Lock object exists until
       the first async method actually needs one.
    3. set(), delete(), and clear_all() lazily instantiate the lock on
       first use: `if self._lock is None: self._lock = asyncio.Lock()`.
       Because these methods are only ever awaited from within the real
       running application event loop (never from bootstrap's synchronous
       context), the lock binds to the correct loop the first time it's
       actually needed.
    4. self._loaded and the async load()/_ensure_loaded() machinery are
       removed entirely — the cache is populated once, synchronously, at
       construction. There is no "not yet loaded" state to guard against.

Storage path:
    CYRAX_WORKSPACE/.cyrax_memory/user_profile.json

Satisfies the 'profile' attribute contract of MemoryStackProtocol:
    async def get(self, key: str) -> Any
    async def set(self, key: str, value: Any) -> None
    async def delete(self, key: str) -> bool
    async def get_all(self) -> dict[str, Any]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Storage location ──────────────────────────────────────────────────────────
_WORKSPACE    = Path.home() / "CYRAX_WORKSPACE"
_MEMORY_DIR   = _WORKSPACE / ".cyrax_memory"
_PROFILE_FILE = _MEMORY_DIR / "user_profile.json"

# Maximum size of the profile file in bytes (1 MB).
_MAX_PROFILE_BYTES: int = 1 * 1024 * 1024


class UserProfileStore:
    """
    Persistent key-value store for long-term user profile memory.

    Construction (synchronous, no event loop required):
        The profile file is read from disk directly inside __init__ via
        self._read_from_disk(). This happens during bootstrap Phase 1,
        before any event loop exists — and that is now exactly correct,
        because no asyncio primitives are touched during construction.

    Async methods (get/set/delete/get_all/clear_all):
        These remain async to satisfy MemoryStackProtocol and to keep
        disk writes off the event loop via asyncio.to_thread(). They are
        only ever called from within the running application event loop
        (via dispatcher → tools → ctx.memory.profile), never from
        bootstrap's synchronous context.

    Lock lifecycle:
        self._lock starts as None. The first call to set(), delete(), or
        clear_all() lazily creates the asyncio.Lock() — at that point a
        real event loop is guaranteed to be running (the one driving the
        coroutine that called the method), so the lock binds correctly
        and stays valid for the lifetime of that loop.

    Corruption recovery:
        If the profile file exists but cannot be parsed as valid JSON,
        the corrupted file is renamed to user_profile.json.corrupt and
        a fresh empty cache is used. Logged at ERROR level; boot proceeds.
    """

    def __init__(self, profile_path: Path = _PROFILE_FILE) -> None:
        self._path = profile_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Patch Directive 3: lock starts unbound. Created lazily on first
        # async use so it binds to the real running event loop, not to
        # whatever transient loop (if any) happens to exist at construction.
        self._lock: Optional[asyncio.Lock] = None

        # Patch Directive 2: synchronous read at construction time.
        # No event loop involvement — this is plain blocking file I/O,
        # which is correct and safe inside bootstrap's synchronous Phase 1.
        self._cache: dict[str, Any] = self._read_from_disk()

        logger.info(
            f"[PROFILE_STORE] Initialised synchronously. "
            f"Path: {self._path} | Loaded {len(self._cache)} entries."
        )

    # ── Public CRUD ───────────────────────────────────────────────────────────

    async def get(self, key: str) -> Any:
        """
        Returns the value for key, or None if the key does not exist.
        Reads from the in-process cache only — no lock needed for reads,
        no disk I/O. Never raises.
        """
        value = self._cache.get(key)
        logger.debug(f"[PROFILE_STORE] get({key!r}) → {str(value)[:80]}")
        return value

    async def set(self, key: str, value: Any) -> None:
        """
        Stores key → value and persists to disk atomically.
        Existing keys are overwritten. New keys are created.

        Lazily instantiates self._lock on first call — see class docstring.
        """
        # Patch Directive 4: lazy lock instantiation, bound to the real
        # running event loop on first actual use.
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            self._cache[key] = value
            await asyncio.to_thread(self._write_to_disk, dict(self._cache))

        logger.info(f"[PROFILE_STORE] set({key!r}, {str(value)[:60]})")

    async def delete(self, key: str) -> bool:
        """
        Removes key from the store and persists to disk.

        Returns:
            True  if the key existed and was deleted.
            False if the key was not found (no-op, no error).
        """
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if key not in self._cache:
                logger.debug(
                    f"[PROFILE_STORE] delete({key!r}) — key not found, no-op."
                )
                return False
            del self._cache[key]
            await asyncio.to_thread(self._write_to_disk, dict(self._cache))

        logger.info(f"[PROFILE_STORE] delete({key!r}) — removed.")
        return True

    async def get_all(self) -> dict[str, Any]:
        """
        Returns a shallow copy of the entire profile dict.
        Read-only — no lock needed, no disk I/O.
        """
        return dict(self._cache)

    async def clear_all(self) -> None:
        """
        Wipes all profile data from memory and disk.
        Intended for testing and explicit user-initiated memory wipes.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            self._cache = {}
            await asyncio.to_thread(self._write_to_disk, {})

        logger.warning("[PROFILE_STORE] All profile data cleared.")

    # ── Internal I/O ─────────────────────────────────────────────────────────

    def _read_from_disk(self) -> dict[str, Any]:
        """
        Synchronous disk read. Called once from __init__ — no thread pool,
        no event loop, plain blocking I/O. This is correct here because
        __init__ itself is a synchronous function called during bootstrap
        Phase 1, before any async machinery exists.

        Never raises — all error paths return {}.
        """
        if not self._path.exists():
            logger.debug("[PROFILE_STORE] No profile file found. Starting fresh.")
            return {}

        file_size = self._path.stat().st_size

        if file_size == 0:
            logger.debug("[PROFILE_STORE] Profile file is empty. Starting fresh.")
            return {}

        if file_size > _MAX_PROFILE_BYTES:
            logger.warning(
                f"[PROFILE_STORE] Profile file exceeds size limit "
                f"({file_size:,} bytes > {_MAX_PROFILE_BYTES:,}). "
                f"Starting fresh to prevent memory exhaustion."
            )
            return {}

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError(
                    f"Profile root is {type(data).__name__}, expected dict."
                )

            return data

        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            logger.error(
                f"[PROFILE_STORE] Corrupted profile file: {exc}. "
                f"Renaming to .corrupt and starting fresh."
            )
            self._quarantine_corrupt_file()
            return {}

        except OSError as exc:
            logger.error(
                f"[PROFILE_STORE] OS error reading profile: {exc}. "
                f"Starting fresh."
            )
            return {}

    def _write_to_disk(self, data: dict[str, Any]) -> None:
        """
        Atomically writes the profile dict to disk via temp-file + os.replace().

        Called from set()/delete()/clear_all() via asyncio.to_thread() — this
        one runs on the application event loop's thread pool, which is correct
        since those methods are invoked while the real loop is running.

        Never raises — errors are logged but do not crash the caller.
        """
        try:
            dir_path = self._path.parent
            fd, tmp_path = tempfile.mkstemp(
                dir    = dir_path,
                prefix = ".profile_tmp_",
                suffix = ".json",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self._path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        except OSError as exc:
            logger.error(
                f"[PROFILE_STORE] Atomic write failed: {exc}. "
                f"In-memory cache is intact but disk was not updated."
            )

    def _quarantine_corrupt_file(self) -> None:
        """Renames the corrupt profile file so it is not silently overwritten."""
        corrupt_path = self._path.with_suffix(".json.corrupt")
        try:
            os.replace(self._path, corrupt_path)
            logger.info(f"[PROFILE_STORE] Corrupt file moved to: {corrupt_path}")
        except OSError as exc:
            logger.warning(
                f"[PROFILE_STORE] Could not quarantine corrupt file: {exc}"
            )