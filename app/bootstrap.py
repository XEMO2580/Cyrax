"""
app/bootstrap.py — CYRAX 3.0 Phase 1 Boot Sequence (Phase 4 Cross-Loop Patch)

Patch applied vs. previous version:
  - Removed asyncio.run(_load_profile(profile_store)) entirely.
  - Removed the _load_profile() async helper function entirely.
  - UserProfileStore() now loads its cache synchronously inside its own
    __init__ — bootstrap() simply instantiates it and reads the resulting
    entry count directly from the synchronous return value.

  This eliminates the temporary event loop that asyncio.run() created,
  which was binding asyncio.Lock (if touched) to a loop that closed
  immediately — the root cause of the cross-loop RuntimeError the
  Systems Reviewer flagged.

Phase 4 changes (unchanged from prior version):
  - UserProfileStore registered in MemoryStack in place of the old stub.
  - CoreMemoryWriteTool and CoreMemoryReadTool registered in the tool registry.
  - _EXPECTED_TOOL_COUNT = 7.

Phase 8.5 changes:
  - Added SQLiteJobStore setup (Step 9a).
  - TaskQueue now requires job_store argument.
  - job_store passed to CyraxContext.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(override=True)

# ── Logger must be first ──────────────────────────────────────────────────────
from logging_.event_logger import initialise_logger
_logger = initialise_logger()
_logger.info("=" * 60)
_logger.info("CYRAX 3.0 — Phase 1 Boot Sequence")
_logger.info("=" * 60)

# ── Core infrastructure ───────────────────────────────────────────────────────
from core.context import (
    CyraxContext,
    DispatcherProtocol,
    ToolRegistryProtocol,
    BrainRouterProtocol,
    MemoryStackProtocol,
    SecurityProtocol,
    FallbackPolicyProtocol,
)
from core.interrupt_controller import InterruptController
from core.job_store import SQLiteJobStore
from core.notification_center import NotificationCenter
from core.resource_manager import ResourceManager
from core.task_queue import TaskQueue
from core.trace import new_trace_id

# ── Settings ──────────────────────────────────────────────────────────────────
from config.settings import settings

# ── Security ──────────────────────────────────────────────────────────────────
from security.lockout_store import LockoutStore
from security.auth import SecurityGuard, SecurityLevel

# ── Tools ─────────────────────────────────────────────────────────────────────
from tools.registry import ToolRegistry, BaseTool

# ── Memory ────────────────────────────────────────────────────────────────────
from memory.state.session_state import SessionStateStore
from memory.conversation.conversation_store import ConversationStore
from memory.profile.profile_store import UserProfileStore   # Patched: synchronous init

# ── Brain ─────────────────────────────────────────────────────────────────────
from brain.providers.base import BaseProvider, ProviderCapabilities, ProviderError
from brain.learning_router import LearningRouter
from brain.moe_router import MoERouter
from brain.provider_metrics import ProviderMetricsManager
from brain.provider_selector import ProviderSelector

# ── Orchestrator ──────────────────────────────────────────────────────────────
from orchestrator.dispatcher import Dispatcher
from orchestrator.fallback_policy import FallbackPolicy

stdlib_logger = logging.getLogger("CYRAX.bootstrap")


# ── Failure helper ────────────────────────────────────────────────────────────

def _hard_fail(message: str) -> None:
    full = f"\n[CYRAX BOOT FAILURE] {message}\n"
    if os.environ.get("CYRAX_TESTING"):
        raise RuntimeError(full)
    print(full, file=sys.stderr)
    sys.exit(1)




# ══════════════════════════════════════════════════════════════════════════════
# MEMORY STACK
# ══════════════════════════════════════════════════════════════════════════════

class MemoryStack:
    """
    Bundles the three memory stores into a single object satisfying
    MemoryStackProtocol. profile is a real, synchronously-initialised
    UserProfileStore — no async load step required at boot.
    """

    def __init__(
        self,
        conversation: ConversationStore,
        profile:      UserProfileStore,
        state:        SessionStateStore,
    ) -> None:
        self.conversation = conversation
        self.profile      = profile
        self.state        = state


# ══════════════════════════════════════════════════════════════════════════════
# EXPECTED TOOL COUNT
# ══════════════════════════════════════════════════════════════════════════════

_EXPECTED_TOOL_COUNT: int = 23  # Update this if you add/remove tools in the registry above.


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap() -> CyraxContext:
    """
    Phase 1 synchronous boot sequence.

    Returns a fully validated, immutable CyraxContext.
    Calls sys.exit(1) on any critical failure (or raises RuntimeError
    in CYRAX_TESTING mode).

    This function remains entirely synchronous end-to-end — no asyncio.run(),
    no event loop creation of any kind. That is the architectural guarantee
    that prevents the cross-loop lock corruption this patch fixes.
    """

    # ── Step 1: Settings validation ───────────────────────────────────────────
    stdlib_logger.info("Step 1/10 — Settings validation...")
    stdlib_logger.info(
        f"ACTIVE_LLM={settings.ACTIVE_LLM.value} | "
        f"BRAIN_TIMEOUT={settings.BRAIN_TIMEOUT_SECONDS}s | "
        f"TOOL_TIMEOUT={settings.TOOL_TIMEOUT_SECONDS}s"
    )
    stdlib_logger.info("Settings: OK ✓")

    # ── Step 2: Lockout store ─────────────────────────────────────────────────
    stdlib_logger.info("Step 2/10 — Lockout store...")
    try:
        lockout_store = LockoutStore()
        locked, remaining = lockout_store.is_locked_out()
        if locked:
            _hard_fail(
                f"System is currently locked out. "
                f"Remaining: {int(remaining)}s. Boot aborted for security."
            )
    except Exception as exc:
        _hard_fail(f"LockoutStore initialisation failed: {exc}")

    stdlib_logger.info("Lockout store: OK ✓")

    # ── Step 3: Security guard ────────────────────────────────────────────────
    stdlib_logger.info("Step 3/10 — Security guard...")
    try:
        security_guard = SecurityGuard(store=lockout_store)
    except Exception as exc:
        _hard_fail(f"SecurityGuard initialisation failed: {exc}")

    stdlib_logger.info("Security guard: OK ✓")

    # ── Step 4: Tool registry ─────────────────────────────────────────────────
    stdlib_logger.info("Step 4/10 — Tool registry...")
    try:
        registry = ToolRegistry(security=security_guard)

        from tools.pc_actions import OpenAppTool, CloseAppTool, TypeTextTool
        from tools.media_power import MediaControlTool, SystemPowerTool
        from tools.web_search import WebSearchTool
        from tools.file_ops import (
            DeleteFileTool, CopyFileTool, MoveFileTool, RenameFileTool,
            ReadFileTool, WriteFileTool, ListDirTool, MakeDirTool,
        )
        from tools.desktop_ops import (
            ClipboardReadTool, ClipboardWriteTool, SendNotificationTool,
        )
        from tools.web_reader import ReadWebpageTool
        from tools.scheduler import ScheduleTaskTool
        from tools.memory_ops import CoreMemoryWriteTool, CoreMemoryReadTool
        from tools.email_ops import SendEmailTool, ReadEmailTool

        registry.register(OpenAppTool())
        registry.register(CloseAppTool())
        registry.register(TypeTextTool())
        registry.register(MediaControlTool())
        registry.register(SystemPowerTool())
        registry.register(WebSearchTool())
        registry.register(DeleteFileTool())
        registry.register(CopyFileTool())
        registry.register(MoveFileTool())
        registry.register(RenameFileTool())
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(ListDirTool())
        registry.register(MakeDirTool())
        registry.register(ClipboardReadTool())
        registry.register(ClipboardWriteTool())
        registry.register(SendNotificationTool())
        registry.register(ReadWebpageTool())
        registry.register(ScheduleTaskTool())
        registry.register(CoreMemoryWriteTool())
        registry.register(CoreMemoryReadTool())
        registry.register(SendEmailTool())
        registry.register(ReadEmailTool())

        registry._lock()

    except Exception as exc:
        _hard_fail(f"Tool registry initialisation failed: {exc}")

    if registry.count() != _EXPECTED_TOOL_COUNT:
        _hard_fail(
            f"Tool registry integrity check FAILED. "
            f"Expected {_EXPECTED_TOOL_COUNT} tools, got {registry.count()}. "
            f"Update _EXPECTED_TOOL_COUNT in bootstrap.py or fix registration."
        )

    stdlib_logger.info(
        f"Tool registry: {registry.count()} tools locked. "
        f"Tools: {registry.get_tool_names()} ✓"
    )

    # ── Step 5: Memory stack ──────────────────────────────────────────────────
    stdlib_logger.info("Step 5/10 — Memory stack...")
    try:
        state_store        = SessionStateStore()
        conversation_store = ConversationStore(
            session_id         = "device_master_001",
            summarise_callback = None,
        )

        # Patch: UserProfileStore loads its cache synchronously inside
        # __init__ — no asyncio.run(), no temporary event loop, no
        # cross-loop lock binding. The entry count is available immediately
        # via the synchronous constructor return.
        profile_store = UserProfileStore()

        memory = MemoryStack(
            conversation = conversation_store,
            profile      = profile_store,
            state        = state_store,
        )

    except Exception as exc:
        _hard_fail(f"Memory stack initialisation failed: {exc}")

    stdlib_logger.info(
        f"Memory stack: conversation + state + persistent profile ready. ✓"
    )

    # ── Step 6: LLM providers ─────────────────────────────────────────────────
    stdlib_logger.info("Step 6/10 — LLM providers...")
    try:
        from brain.providers.groq_provider import GroqProvider
        from brain.providers.gemini_provider import GeminiProvider
        
        providers: dict[str, BaseProvider] = {
            "groq": GroqProvider(),
            "gemini": GeminiProvider(),
        }
    except Exception as exc:
        _hard_fail(f"LLM provider initialisation failed: {exc}")

    stdlib_logger.info(
        f"LLM providers: {list(providers.keys())} registered. ✓"
    )

    # ── Step 6a: Resource manager ─────────────────────────────────────────────
    stdlib_logger.info("Step 6a/10 — Resource manager...")
    try:
        resource_manager = ResourceManager()
    except Exception as exc:
        _hard_fail(f"ResourceManager initialisation failed: {exc}")

    stdlib_logger.info("Resource manager: OK ✓")

    # ── Step 7: Brain router ──────────────────────────────────────────────────
    stdlib_logger.info("Step 7/10 — Brain router (MoERouter)...")
    try:
        metrics_manager = ProviderMetricsManager()
        brain_router = MoERouter(
            providers=providers,
            metrics_manager=metrics_manager,
            resource_manager=resource_manager,
        )
    except Exception as exc:
        _hard_fail(f"MoERouter initialisation failed: {exc}")

    stdlib_logger.info("Brain router: OK ✓")

    # ── Step 7a: Decision Engine ──────────────────────────────────────────────
    stdlib_logger.info("Step 7a/10 — Decision Engine...")
    try:
        from orchestrator.decision_engine import DecisionEngine
        classifier_provider = brain_router._providers.get("groq")
        if classifier_provider is None:
            _hard_fail("Groq provider required for Decision Engine classifier.")
        provider_selector = ProviderSelector()
        decision_engine = DecisionEngine(
            provider=classifier_provider,
            provider_selector=provider_selector,
            metrics_manager=metrics_manager,
            active_providers=['groq', 'gemini'],
        )
    except Exception as exc:
        _hard_fail(f"Decision Engine initialisation failed: {exc}")

    stdlib_logger.info("Decision Engine: OK ✓")

    # ── Step 8: Fallback policy ───────────────────────────────────────────────
    stdlib_logger.info("Step 8/10 — Fallback policy...")
    try:
        fallback_policy = FallbackPolicy()
    except Exception as exc:
        _hard_fail(f"FallbackPolicy initialisation failed: {exc}")

    stdlib_logger.info("Fallback policy: OK ✓")

    # ── Step 9: Dispatcher ────────────────────────────────────────────────────
    stdlib_logger.info("Step 9/10 — Dispatcher...")
    try:
        dispatcher = Dispatcher()
    except Exception as exc:
        _hard_fail(f"Dispatcher initialisation failed: {exc}")

    stdlib_logger.info("Dispatcher: OK ✓")

    # ── Step 9a: SQLite Job Store ──────────────────────────────────────────
    stdlib_logger.info("Step 9a/10 — SQLite job store...")
    try:
        job_store = SQLiteJobStore()
    except Exception as exc:
        _hard_fail(f"SQLiteJobStore initialisation failed: {exc}")

    stdlib_logger.info("SQLite job store: OK ✓")

    # ── Step 9b: Task queue ──────────────────────────────────────────────────
    stdlib_logger.info("Step 9b/10 — Task queue...")
    try:
        task_queue = TaskQueue(job_store=job_store)
    except Exception as exc:
        _hard_fail(f"TaskQueue initialisation failed: {exc}")

    stdlib_logger.info("Task queue: OK ✓")

    # ── Step 9c: Notification Center ─────────────────────────────────────────
    stdlib_logger.info("Step 9c/10 — Notification Center...")
    try:
        notification_center = NotificationCenter()
    except Exception as exc:
        _hard_fail(f"NotificationCenter initialisation failed: {exc}")

    stdlib_logger.info("Notification center: OK ✓")

    # ── Step 9d: Interrupt Controller ────────────────────────────────────────
    stdlib_logger.info("Step 9d/10 — Interrupt Controller...")
    try:
        interrupt_controller = InterruptController()
    except Exception as exc:
        _hard_fail(f"InterruptController initialisation failed: {exc}")

    stdlib_logger.info("Interrupt controller: OK ✓")

    # ── Step 9e: Learning Router ──────────────────────────────────────────────
    stdlib_logger.info("Step 9e/10 — Learning Router...")
    try:
        learning_router = LearningRouter(job_store=job_store)
    except Exception as exc:
        _hard_fail(f"LearningRouter initialisation failed: {exc}")

    stdlib_logger.info("Learning router: OK ✓")

    # ── Step 10: Assemble CyraxContext ────────────────────────────────────────
    stdlib_logger.info("Step 10/10 — Assembling CyraxContext...")
    try:
        ctx = CyraxContext(
            session_id            = "device_master_001",
            dispatcher            = dispatcher,
            tool_registry         = registry,
            brain_router          = brain_router,
            memory                = memory,
            security              = security_guard,
            fallback_policy       = fallback_policy,
            decision_engine       = decision_engine,
            task_queue            = task_queue,
            notification_center   = notification_center,
            interrupt_controller  = interrupt_controller,
            job_store             = job_store,
            resource_manager      = resource_manager,
            learning_router       = learning_router,
        )
    except (TypeError, ValueError) as exc:
        _hard_fail(
            f"CyraxContext construction failed: {exc}\n"
            f"A component does not satisfy its required protocol."
        )

    stdlib_logger.info("CyraxContext: assembled and validated. ✓")
    stdlib_logger.info("=" * 60)
    stdlib_logger.info("Phase 1 boot complete. Handing off to async layer.")
    stdlib_logger.info("=" * 60)

    return ctx
