"""
app/sessions/session_manager.py — CYRAX 3.0 Multi-Tenant Session Manager (Phase 9.5B)

Wraps the frozen V1.0 Core OS behind a per-mobile-session boundary. Each
authenticated mobile client gets its own `MobileSession` holding a fresh
`CyraxContext` so conversational state, interrupt state, and task
cancellation never bleed across devices.

Architecture (per the frozen contracts):
  - `docs/api/authentication.md` §1 — each mobile client is a NEW concept
    (`MobileSession`), NOT the CLI's implicit `device_master_001` model.
  - `docs/api/rest.md` §2 — every authenticated request resolves to a
    session-scoped `CyraxContext`; no route handler ever receives a shared,
    cross-session context.
  - `docs/api/authentication.md` §4 — idle sessions (>30 min) and
    absolutely-old sessions (>24 h) are evicted; a valid unexpired JWT
    against an evicted session transparently resumes a fresh context.

Critical isolation contract (per the directive):
  - REUSED singletons from the global boot sequence: `tool_registry`,
    `brain_router`, `decision_engine`, `job_store` (plus stateless
    `dispatcher`, `fallback_policy`, `task_queue`, `notification_center`,
    `resource_manager`, `learning_router`, and the shared `security`
    guard so device-global lockout stays device-global).
  - FRESH per-session instances: a fresh `MemoryStack` (per-session
    ConversationStore + SessionStateStore + shared device-scoped profile)
    and a FRESH `InterruptController` so running-task cancellation does not
    leak across devices.

Cross-loop safety:
  - Follows the codebase convention (see UserProfileStore) that
    `asyncio.Lock()` must NOT be bound at construction time — it is lazily
    instantiated on first async use so it binds to the running event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.context import CyraxContext
from core.interrupt_controller import InterruptController

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Idle timeout before eviction, seconds (frozen contract default: 30 min).
IDLE_SESSION_TIMEOUT_SECONDS: float = 30 * 60.0

# Absolute maximum session lifetime, seconds (frozen contract default: 24 h).
MAX_SESSION_AGE_SECONDS: float = 24 * 60 * 60.0

# How often the eviction loop wakes up to scan for idle/expired sessions.
EVICTION_SWEEP_INTERVAL_SECONDS: float = 60.0


# ══════════════════════════════════════════════════════════════════════════════
# MOBILE SESSION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MobileSession:
    """
    A single authenticated mobile client session.

    Attributes:
        device_id:    Stable client-generated device UUID.
        session_id:   Server-generated UUID, new per login.
        ctx:          The session-scoped CyraxContext (fresh memory +
                      interrupt controller, shared core singletons).
        last_access:  UTC timestamp of the most recent REST/WebSocket activity.
                      Used by the eviction loop for idle-timeout.
        created_at:   UTC timestamp of session creation.
        websocket:    Optional reference to the live WebSocket connection,
                      if the client has one open (Phase 9.5 WebSocket).
    """

    device_id:    str
    session_id:   str
    ctx:          CyraxContext
    last_access:  float = field(default_factory=lambda: time.time())
    created_at:   float = field(default_factory=lambda: time.time())
    websocket:    Any   = None

    def touch(self) -> None:
        """Marks the session as recently accessed (REST/WebSocket activity)."""
        self.last_access = time.time()

    @property
    def idle_seconds(self) -> float:
        """Seconds since the last access."""
        return time.time() - self.last_access

    @property
    def age_seconds(self) -> float:
        """Seconds since the session was created."""
        return time.time() - self.created_at

    def is_idle(self, idle_timeout: float = IDLE_SESSION_TIMEOUT_SECONDS) -> bool:
        """True if the session has been idle longer than `idle_timeout`."""
        return self.idle_seconds > idle_timeout

    def is_expired(self, max_age: float = MAX_SESSION_AGE_SECONDS) -> bool:
        """True if the session has existed longer than `max_age`."""
        return self.age_seconds > max_age


# ══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class SessionManager:
    """
    Multi-tenant registry of active MobileSession instances.

    Owns a process-wide dict of sessions keyed by `session_id`. Constructed
    once from the global boot `CyraxContext` (which supplies the shared core
    singletons) and started as a long-lived background eviction loop.

    Thread-safety: dict mutations are guarded by an asyncio.Lock that is
    lazily instantiated on first async use (codebase convention) so it binds
    to the running event loop and never triggers a cross-loop RuntimeError.
    """

    def __init__(
        self,
        base_ctx: CyraxContext,
        *,
        idle_timeout: float = IDLE_SESSION_TIMEOUT_SECONDS,
        max_age:      float = MAX_SESSION_AGE_SECONDS,
        sweep_interval: float = EVICTION_SWEEP_INTERVAL_SECONDS,
    ) -> None:
        self._base_ctx = base_ctx

        # Eviction policy knobs (overridable for tests).
        self._idle_timeout   = idle_timeout
        self._max_age        = max_age
        self._sweep_interval = sweep_interval

        # Active sessions keyed by session_id.
        self._sessions: dict[str, MobileSession] = {}

        # Lazy asyncio.Lock — created on first async use so it binds to the
        # running event loop (see UserProfileStore convention).
        self._lock: asyncio.Lock | None = None

        # Background eviction task (started via start_eviction_loop()).
        self._eviction_task: asyncio.Task | None = None

        logger.info(
            f"[SESSION_MANAGER] Initialised from base context. "
            f"idle_timeout={idle_timeout}s | max_age={max_age}s | "
            f"sweep_interval={sweep_interval}s"
        )

    # ── Lock helper ───────────────────────────────────────────────────────────

    def _get_lock(self) -> asyncio.Lock:
        """Lazily creates the asyncio.Lock on first use (loop-safe)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ── Session lifecycle ─────────────────────────────────────────────────────

    async def create_session(self, device_id: str) -> MobileSession:
        """
        Creates a new authenticated session for a device.

        Generates a fresh session_id, constructs a brand-new session-scoped
        CyraxContext (reusing core singletons, fresh memory + interrupt
        controller), and stores it in the registry.

        Args:
            device_id: The authenticated client's stable device UUID.

        Returns:
            The newly created MobileSession.
        """
        session_id = str(uuid.uuid4())
        ctx = self._build_context(device_id, session_id)

        session = MobileSession(
            device_id=device_id,
            session_id=session_id,
            ctx=ctx,
        )

        async with self._get_lock():
            self._sessions[session_id] = session

        logger.info(
            f"[SESSION_MANAGER] Created | device_id={device_id} | "
            f"session_id={session_id} | active={len(self._sessions)}"
        )
        return session

    def _build_context(self, device_id: str, session_id: str) -> CyraxContext:
        """
        Constructs a fresh session-scoped CyraxContext.

        CRITICAL isolation contract:
          - REUSE the singleton tool_registry, brain_router, decision_engine,
            job_store (and the shared security guard so lockout stays
            device-global) from the base boot context.
          - FRESH MemoryStack: per-session ConversationStore + SessionStateStore,
            and a device-scoped UserProfileStore (so profile memory is keyed by
            device, not shared across the whole process).
          - FRESH InterruptController: cancellation state is per-session.
        """
        base = self._base_ctx

        # ── Fresh MemoryStack ────────────────────────────────────────────────
        from memory.conversation.conversation_store import ConversationStore
        from memory.profile.profile_store import UserProfileStore
        from memory.state.session_state import SessionStateStore

        state_store        = SessionStateStore()
        conversation_store = ConversationStore(
            session_id=session_id,
            summarise_callback=None,
        )

        # Device-scoped profile: keyed by device_id so each device has its own
        # persistent profile file, isolated from the CLI's profile and from
        # other devices.
        profile_store = UserProfileStore(profile_path=self._device_profile_path(device_id))

        memory = type(base.memory)(
            conversation=conversation_store,
            profile=profile_store,
            state=state_store,
        )

        # ── Fresh InterruptController ────────────────────────────────────────
        interrupt_controller = InterruptController()

        # ── Assemble the session-scoped context ──────────────────────────────
        ctx = CyraxContext(
            session_id=session_id,
            dispatcher=base.dispatcher,
            tool_registry=base.tool_registry,
            brain_router=base.brain_router,
            memory=memory,
            security=base.security,
            fallback_policy=base.fallback_policy,
            decision_engine=base.decision_engine,
            task_queue=base.task_queue,
            notification_center=base.notification_center,
            interrupt_controller=interrupt_controller,
            job_store=base.job_store,
            resource_manager=base.resource_manager,
            learning_router=base.learning_router,
            job_scheduler=base.job_scheduler,
        )

        logger.debug(
            f"[SESSION_MANAGER] Built context | session_id={session_id} | "
            f"device_id={device_id}"
        )
        return ctx

    @staticmethod
    def _device_profile_path(device_id: str) -> Path:
        """
        Resolves the per-device profile file path.

        Mirrors UserProfileStore's default location (`~/CYRAX_WORKSPACE/
        .cyrax_memory/user_profile.json`) but keys it by device_id so each
        mobile device has an isolated persistent profile.
        """
        workspace = Path.home() / "CYRAX_WORKSPACE"
        memory_dir = workspace / ".cyrax_memory" / "mobile"
        return memory_dir / f"profile_{device_id}.json"

    async def resume_session(self, device_id: str, session_id: str) -> MobileSession:
        """
        Resolves a session, transparently reconstructing it if it was evicted.

        Per `docs/api/rest.md` §2 and `docs/api/authentication.md` §4: a valid,
        unexpired JWT against an evicted session results in a fresh CyraxContext
        being constructed transparently (same session_id, new in-memory state) —
        this is a session *resume*, not a re-login.

        Args:
            device_id:  The authenticated device UUID.
            session_id: The token's session UUID.

        Returns:
            The existing (or freshly resume-built) MobileSession.
        """
        async with self._get_lock():
            session = self._sessions.get(session_id)

        if session is not None:
            session.touch()
            return session

        # Evicted or first request after login — rebuild a fresh context.
        logger.info(
            f"[SESSION_MANAGER] Resuming evicted/missing session | "
            f"device_id={device_id} | session_id={session_id}"
        )
        ctx = self._build_context(device_id, session_id)
        session = MobileSession(
            device_id=device_id,
            session_id=session_id,
            ctx=ctx,
        )

        async with self._get_lock():
            self._sessions[session_id] = session

        return session

    async def get_session(self, session_id: str) -> MobileSession | None:
        """Returns the session for `session_id`, or None if it does not exist."""
        async with self._get_lock():
            return self._sessions.get(session_id)

    async def destroy_session(self, session_id: str) -> bool:
        """
        Destroys a session and removes it from the registry.

        Returns True if a session was removed, False if it did not exist.
        """
        async with self._get_lock():
            session = self._sessions.pop(session_id, None)
        if session is None:
            logger.debug(f"[SESSION_MANAGER] destroy_session | unknown '{session_id}'")
            return False

        logger.info(
            f"[SESSION_MANAGER] Destroyed | session_id={session_id} | "
            f"active={len(self._sessions)}"
        )
        return True

    def get_security_guard(self):
        """
        Returns the shared SecurityGuard from the base boot context.

        The security guard is shared across all sessions by design —
        device-global lockout must apply equally to CLI, voice, AND mobile
        (`docs/api/rest.md` §3 POST /auth/login failure semantics).
        """
        return self._base_ctx.security

    def active_count(self) -> int:
        """Returns the number of currently active sessions (best-effort, no lock)."""
        return len(self._sessions)

    def active_session_ids(self) -> list[str]:
        """Returns a snapshot of active session_id values."""
        return list(self._sessions.keys())

    # ── Eviction loop ─────────────────────────────────────────────────────────

    async def evict_idle_sessions(self) -> list[str]:
        """
        Synchronously evicts sessions that are idle or absolutely expired.

        Returns the list of evicted session_ids. Designed to be called by the
        background loop (evict_loop_forever) and callable directly by tests.
        """
        evicted: list[str] = []

        async with self._get_lock():
            for session_id in list(self._sessions.keys()):
                session = self._sessions[session_id]
                if session.is_idle(self._idle_timeout) or session.is_expired(self._max_age):
                    del self._sessions[session_id]
                    evicted.append(session_id)

        if evicted:
            logger.info(
                f"[SESSION_MANAGER] Evicted {len(evicted)} session(s): {evicted}. "
                f"active={len(self._sessions)}"
            )
        return evicted

    async def evict_loop_forever(self) -> None:
        """
        Background async loop that periodically sweeps idle/expired sessions.

        Runs until the task is cancelled. Designed to be started via
        asyncio.create_task() and cancelled on application shutdown.
        """
        logger.info(
            f"[SESSION_MANAGER] Eviction loop started (sweep every "
            f"{self._sweep_interval}s)."
        )
        while True:
            try:
                await asyncio.sleep(self._sweep_interval)
                await self.evict_idle_sessions()
            except asyncio.CancelledError:
                logger.info("[SESSION_MANAGER] Eviction loop cancelled.")
                raise
            except Exception as exc:
                logger.error(f"[SESSION_MANAGER] Eviction loop error: {exc}", exc_info=True)

    def start_eviction_loop(self) -> asyncio.Task:
        """Starts the eviction loop as a background task and returns it."""
        if self._eviction_task is not None and not self._eviction_task.done():
            return self._eviction_task
        self._eviction_task = asyncio.create_task(
            self.evict_loop_forever(),
            name="session-manager-eviction",
        )
        return self._eviction_task

    async def stop_eviction_loop(self) -> None:
        """Cancels and awaits the background eviction loop task."""
        if self._eviction_task is not None:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
            self._eviction_task = None


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL GLOBAL (process-wide singleton)
# ══════════════════════════════════════════════════════════════════════════════

# Populated once by initialise_session_manager() at app startup.
_session_manager: SessionManager | None = None


def initialise_session_manager(
    base_ctx: CyraxContext,
    **kwargs: Any,
) -> SessionManager:
    """
    Creates the process-wide SessionManager from the boot context.

    Idempotent: calling it again returns the existing instance. The base
    context's shared core singletons (tool_registry, brain_router,
    decision_engine, job_store) are captured once and reused by every
    mobile session's context.

    Args:
        base_ctx: The CyraxContext returned by app.bootstrap.bootstrap().
        **kwargs: Optional overrides for idle_timeout/max_age/sweep_interval.

    Returns:
        The process-wide SessionManager instance.
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(base_ctx, **kwargs)
    return _session_manager


def get_session_manager() -> SessionManager:
    """
    Returns the process-wide SessionManager.

    Raises:
        RuntimeError: If initialise_session_manager() has not been called yet.
    """
    if _session_manager is None:
        raise RuntimeError(
            "SessionManager is not initialised. Call "
            "initialise_session_manager(base_ctx) during app startup."
        )
    return _session_manager


def reset_session_manager() -> None:
    """Resets the global SessionManager (primarily for tests)."""
    global _session_manager
    _session_manager = None
