# """
# app/bootstrap.py — CYRAX 3.0 Phase 1 Synchronous Boot Sequence

# Constructs and returns a fully validated CyraxContext.
# Called once by cli_main.py and api_server.py before the async loop starts.
# No coroutines. No event loop. Pure synchronous initialisation.

# Boot order (dependency-first):
#   1.  Logger
#   2.  Settings (already validated at import by Pydantic)
#   3.  LockoutStore
#   4.  SecurityGuard
#   5.  ToolRegistry (real tools registered + locked)
#   6.  SessionStateStore
#   7.  ConversationStore (wrapped into MemoryStack)
#   8.  Stub LLM Providers  →  MoERouter
#   9.  Dispatcher
#   10. FallbackPolicy
#   11. CyraxContext (validated by __post_init__)

# """

# from __future__ import annotations

# import logging
# import os
# import sys
# from pathlib import Path
# from typing import Any

# from dotenv import load_dotenv

# load_dotenv()

# # ── Logger must be first — everything else may emit log events ────────────────
# from logging_.event_logger import initialise_logger
# _logger = initialise_logger()
# _logger.info("=" * 60)
# _logger.info("CYRAX 3.0 — Phase 1 Boot Sequence")
# _logger.info("=" * 60)

# # ── Core infrastructure ───────────────────────────────────────────────────────
# from core.context import (
#     CyraxContext,
#     DispatcherProtocol,
#     ToolRegistryProtocol,
#     BrainRouterProtocol,
#     MemoryStackProtocol,
#     SecurityProtocol,
#     FallbackPolicyProtocol,
# )
# from core.trace import new_trace_id

# # ── Settings (validated at import — any misconfiguration aborts here) ─────────
# from config.settings import settings

# # ── Security ──────────────────────────────────────────────────────────────────
# from security.lockout_store import LockoutStore
# from security.auth import SecurityGuard, SecurityLevel

# # ── Tools ─────────────────────────────────────────────────────────────────────
# from tools.registry import ToolRegistry, BaseTool
# from pydantic import BaseModel, Field

# # ── Memory ────────────────────────────────────────────────────────────────────
# from memory.state.session_state import SessionStateStore
# from memory.conversation.conversation_store import ConversationStore

# # ── Brain ─────────────────────────────────────────────────────────────────────
# from brain.providers.base import BaseProvider, ProviderCapabilities, ProviderError
# from brain.moe_router import MoERouter
# from brain.providers.groq_provider import GroqProvider
# from brain.providers.gemini_provider import GeminiProvider

# # ── Orchestrator ──────────────────────────────────────────────────────────────
# from orchestrator.dispatcher import Dispatcher
# from orchestrator.fallback_policy import FallbackPolicy

# stdlib_logger = logging.getLogger("CYRAX.bootstrap")

# # ── Failure helper ────────────────────────────────────────────────────────────

# def _hard_fail(message: str) -> None:
#     full = f"\n[CYRAX BOOT FAILURE] {message}\n"
#     if os.environ.get("CYRAX_TESTING"):
#         raise RuntimeError(full)
#     print(full, file=sys.stderr)
#     sys.exit(1)


# # ══════════════════════════════════════════════════════════════════════════════
# # LLM PROVIDERS
# # ══════════════════════════════════════════════════════════════════════════════

# # ── Step 6: LLM providers ─────────────────────────────────────────────────
#     stdlib_logger.info("Step 6/10 — LLM providers...")
#     try:
#         providers: dict[str, BaseProvider] = {
#             "groq":   GroqProvider(),
#             "gemini": GeminiProvider(),
#         }

# # ══════════════════════════════════════════════════════════════════════════════
# # STUB TOOLS
# # Minimal BaseTool subclasses that satisfy the registry contract.
# # Replace with real tool imports as tools/ is migrated.
# # ══════════════════════════════════════════════════════════════════════════════

# class _StubArgs(BaseModel):
#     action: str = Field(default="stub", description="Stub action parameter.")


# class _StubOpenAppArgs(BaseModel):
#     app: str = Field(..., description="Application name to open.")


# class _StubWebSearchArgs(BaseModel):
#     query: str = Field(..., description="Search query.")


# class _StubMediaArgs(BaseModel):
#     action: str = Field(..., description="Media action.")


# class _StubPowerArgs(BaseModel):
#     action: str = Field(..., description="Power action: shutdown | restart | abort.")


# class StubOpenAppTool(BaseTool):
#     name           = "OPEN_APP"
#     description    = "Opens an application. [STUB]"
#     security_level = SecurityLevel.UNRESTRICTED
#     args_schema    = _StubOpenAppArgs

#     def execute(self, **kwargs) -> str:
#         app = kwargs.get("app", "unknown")
#         stdlib_logger.info(f"[STUB TOOL] OPEN_APP | app={app}")
#         return f"Success: {app} opened. [STUB]"


# class StubCloseAppTool(BaseTool):
#     name           = "CLOSE_APP"
#     description    = "Closes an application. [STUB]"
#     security_level = SecurityLevel.ADMIN
#     args_schema    = _StubOpenAppArgs

#     def execute(self, **kwargs) -> str:
#         app = kwargs.get("app", "unknown")
#         stdlib_logger.info(f"[STUB TOOL] CLOSE_APP | app={app}")
#         return f"Success: {app} closed. [STUB]"


# class StubWebSearchTool(BaseTool):
#     name           = "WEB_SEARCH"
#     description    = "Searches the web for a query. [STUB]"
#     security_level = SecurityLevel.USER
#     args_schema    = _StubWebSearchArgs

#     def execute(self, **kwargs) -> str:
#         query = kwargs.get("query", "")
#         stdlib_logger.info(f"[STUB TOOL] WEB_SEARCH | query='{query[:60]}'")
#         return f"Success: Searched for '{query}'. [STUB]"


# class StubMediaControlTool(BaseTool):
#     name           = "MEDIA_CONTROL"
#     description    = "Controls media playback. [STUB]"
#     security_level = SecurityLevel.USER
#     args_schema    = _StubMediaArgs

#     def execute(self, **kwargs) -> str:
#         action = kwargs.get("action", "")
#         stdlib_logger.info(f"[STUB TOOL] MEDIA_CONTROL | action={action}")
#         return f"Success: Media action '{action}' executed. [STUB]"


# class StubSystemPowerTool(BaseTool):
#     name           = "SYSTEM_POWER"
#     description    = "Controls system power state. [STUB]"
#     security_level = SecurityLevel.ADMIN
#     args_schema    = _StubPowerArgs

#     def execute(self, **kwargs) -> str:
#         action = kwargs.get("action", "")
#         stdlib_logger.info(f"[STUB TOOL] SYSTEM_POWER | action={action}")
#         return f"Success: Power action '{action}' executed. [STUB]"


# # ══════════════════════════════════════════════════════════════════════════════
# # MEMORY STACK WRAPPER
# # ══════════════════════════════════════════════════════════════════════════════

# class MemoryStack:
#     """
#     Bundles the three initialised memory stores into a single object
#     that satisfies MemoryStackProtocol.

#     Constructed inside bootstrap() after each store is individually
#     initialised. Passed into CyraxContext as the 'memory' field.
#     """

#     def __init__(
#         self,
#         conversation: ConversationStore,
#         profile:      Any,
#         state:        SessionStateStore,
#     ) -> None:
#         self.conversation = conversation
#         self.profile      = profile
#         self.state        = state


# class _StubUserProfileStore:
#     """Minimal profile store stub. Replace with real ProfileStore when migrated."""

#     def get(self, key: str) -> Any:
#         return None

#     async def set(self, key: str, value: Any) -> None:
#         pass


# # ══════════════════════════════════════════════════════════════════════════════
# # EXPECTED TOOL COUNT
# # Update this constant whenever real tools are added or removed.
# # bootstrap() asserts registry.count() == this value after locking.
# # A mismatch crashes boot — this is the architectural fix for the
# # Round 3 9→8 tool-count drift incident.
# # ══════════════════════════════════════════════════════════════════════════════

# _EXPECTED_TOOL_COUNT: int = 5   # Stub tools registered below.


# # ══════════════════════════════════════════════════════════════════════════════
# # BOOTSTRAP ENTRY POINT
# # ══════════════════════════════════════════════════════════════════════════════

# def bootstrap() -> CyraxContext:
#     """
#     Phase 1 synchronous boot sequence.

#     Returns a fully validated, immutable CyraxContext.
#     Calls sys.exit(1) on any critical failure (or raises RuntimeError
#     in CYRAX_TESTING mode).

#     This function is the only place CyraxContext is constructed.
#     """

#     # ── Step 1: Validate settings ─────────────────────────────────────────────
#     stdlib_logger.info("Step 1/10 — Settings validation...")
#     stdlib_logger.info(
#         f"ACTIVE_LLM={settings.ACTIVE_LLM.value} | "
#         f"BRAIN_TIMEOUT={settings.BRAIN_TIMEOUT_SECONDS}s | "
#         f"TOOL_TIMEOUT={settings.TOOL_TIMEOUT_SECONDS}s"
#     )
#     stdlib_logger.info("Settings: OK ✓")

#     # ── Step 2: Lockout store ─────────────────────────────────────────────────
#     stdlib_logger.info("Step 2/10 — Lockout store...")
#     try:
#         lockout_store = LockoutStore()
#         locked, remaining = lockout_store.is_locked_out()
#         if locked:
#             _hard_fail(
#                 f"System is currently locked out. "
#                 f"Remaining: {int(remaining)}s. "
#                 f"Boot aborted for security."
#             )
#     except Exception as exc:
#         _hard_fail(f"LockoutStore initialisation failed: {exc}")

#     stdlib_logger.info("Lockout store: OK ✓")

#     # ── Step 3: Security guard ────────────────────────────────────────────────
#     stdlib_logger.info("Step 3/10 — Security guard...")
#     try:
#         security_guard = SecurityGuard(store=lockout_store)
#     except Exception as exc:
#         _hard_fail(f"SecurityGuard initialisation failed: {exc}")

#     stdlib_logger.info("Security guard: OK ✓")

#     # ── Step 4: Tool registry ─────────────────────────────────────────────────
#     stdlib_logger.info("Step 4/10 — Tool registry...")
#     try:
#         registry = ToolRegistry(security=security_guard)

#         registry.register(StubOpenAppTool())
#         registry.register(StubCloseAppTool())
#         registry.register(StubWebSearchTool())
#         registry.register(StubMediaControlTool())
#         registry.register(StubSystemPowerTool())

#         # Lock the registry — no further registrations permitted at runtime.
#         registry._lock()

#     except Exception as exc:
#         _hard_fail(f"Tool registry initialisation failed: {exc}")

#     # Integrity check — catches the 9→8 drift class of bug at boot.
#     if registry.count() != _EXPECTED_TOOL_COUNT:
#         _hard_fail(
#             f"Tool registry integrity check FAILED. "
#             f"Expected {_EXPECTED_TOOL_COUNT} tools, "
#             f"got {registry.count()}. "
#             f"Update _EXPECTED_TOOL_COUNT in bootstrap.py or fix registration."
#         )

#     stdlib_logger.info(
#         f"Tool registry: {registry.count()} tools locked. "
#         f"Tools: {registry.get_tool_names()} ✓"
#     )

#     # ── Step 5: Memory stack ──────────────────────────────────────────────────
#     stdlib_logger.info("Step 5/10 — Memory stack...")
#     try:
#         state_store        = SessionStateStore()
#         conversation_store = ConversationStore(
#             session_id="device_master_001",
#             summarise_callback=None,    # Injected when brain/chat.py is migrated.
#         )
#         profile_store      = _StubUserProfileStore()

#         memory = MemoryStack(
#             conversation=conversation_store,
#             profile=profile_store,
#             state=state_store,
#         )
#     except Exception as exc:
#         _hard_fail(f"Memory stack initialisation failed: {exc}")

#     stdlib_logger.info("Memory stack: conversation + profile + state ready. ✓")

#     # ── Step 6: LLM providers ─────────────────────────────────────────────────
#     stdlib_logger.info("Step 6/10 — LLM providers...")
#     try:
#         providers: dict[str, BaseProvider] = {
#             "groq":   GroqProvider(),
#             "gemini": GeminiProvider(),
#         }
#         # When real providers are ready, replace stubs:
#         # providers["groq"]   = GroqProvider(api_key=settings.GROQ_API_KEY, ...)
#         # providers["gemini"] = GeminiProvider(api_key=settings.GEMINI_API_KEY, ...)
#     except Exception as exc:
#         _hard_fail(f"LLM provider initialisation failed: {exc}")

#     stdlib_logger.info(
#         f"LLM providers: {list(providers.keys())} registered. [STUB] ✓"
#     )

#     # ── Step 7: Brain router ──────────────────────────────────────────────────
#     stdlib_logger.info("Step 7/10 — Brain router (MoERouter)...")
#     try:
#         brain_router = MoERouter(providers=providers)
#     except Exception as exc:
#         _hard_fail(f"MoERouter initialisation failed: {exc}")

#     stdlib_logger.info("Brain router: OK ✓")

#     # ── Step 8: Fallback policy ───────────────────────────────────────────────
#     stdlib_logger.info("Step 8/10 — Fallback policy...")
#     try:
#         fallback_policy = FallbackPolicy()
#     except Exception as exc:
#         _hard_fail(f"FallbackPolicy initialisation failed: {exc}")

#     stdlib_logger.info("Fallback policy: OK ✓")

#     # ── Step 9: Dispatcher ────────────────────────────────────────────────────
#     stdlib_logger.info("Step 9/10 — Dispatcher...")
#     try:
#         dispatcher = Dispatcher()
#     except Exception as exc:
#         _hard_fail(f"Dispatcher initialisation failed: {exc}")

#     stdlib_logger.info("Dispatcher: OK ✓")

#     # ── Step 10: Assemble CyraxContext ────────────────────────────────────────
#     stdlib_logger.info("Step 10/10 — Assembling CyraxContext...")
#     try:
#         ctx = CyraxContext(
#             session_id      = "device_master_001",
#             dispatcher      = dispatcher,
#             tool_registry   = registry,
#             brain_router    = brain_router,
#             memory          = memory,
#             security        = security_guard,
#             fallback_policy = fallback_policy,
#         )
#     except (TypeError, ValueError) as exc:
#         _hard_fail(
#             f"CyraxContext construction failed: {exc}\n"
#             f"A component does not satisfy its required protocol."
#         )

#     stdlib_logger.info("CyraxContext: assembled and validated. ✓")
#     stdlib_logger.info("=" * 60)
#     stdlib_logger.info("Phase 1 boot complete. Handing off to async layer.")
#     stdlib_logger.info("=" * 60)

#     return ctx 








"""
app/bootstrap.py — CYRAX 3.0 Phase 1 Synchronous Boot Sequence

Constructs and returns a fully validated CyraxContext.
Called once by cli_main.py and api_server.py before the async loop starts.
No coroutines. No event loop. Pure synchronous initialisation.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Logger must be first — everything else may emit log events ────────────────
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
from core.trace import new_trace_id

# ── Settings (validated at import — any misconfiguration aborts here) ─────────
from config.settings import settings

# ── Security ──────────────────────────────────────────────────────────────────
from security.lockout_store import LockoutStore
from security.auth import SecurityGuard, SecurityLevel

# ── Tools ─────────────────────────────────────────────────────────────────────
from tools.registry import ToolRegistry, BaseTool
from pydantic import BaseModel, Field

# ── Memory ────────────────────────────────────────────────────────────────────
from memory.state.session_state import SessionStateStore
from memory.conversation.conversation_store import ConversationStore

# ── Brain ─────────────────────────────────────────────────────────────────────
from brain.providers.base import BaseProvider, ProviderCapabilities, ProviderError
from brain.providers.groq_provider import GroqProvider
from brain.providers.gemini_provider import GeminiProvider
from brain.moe_router import MoERouter

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
# STUB TOOLS
# Minimal BaseTool subclasses that satisfy the registry contract.
# Replace with real tool imports as tools/ is migrated.
# ══════════════════════════════════════════════════════════════════════════════

class _StubArgs(BaseModel):
    action: str = Field(default="stub", description="Stub action parameter.")

class _StubOpenAppArgs(BaseModel):
    app: str = Field(..., description="Application name to open.")

class _StubWebSearchArgs(BaseModel):
    query: str = Field(..., description="Search query.")

class _StubMediaArgs(BaseModel):
    action: str = Field(..., description="Media action.")

class _StubPowerArgs(BaseModel):
    action: str = Field(..., description="Power action: shutdown | restart | abort.")


class StubOpenAppTool(BaseTool):
    name           = "OPEN_APP"
    description    = "Opens an application. [STUB]"
    security_level = SecurityLevel.UNRESTRICTED
    args_schema    = _StubOpenAppArgs

    def execute(self, **kwargs) -> str:
        app = kwargs.get("app", "unknown")
        stdlib_logger.info(f"[STUB TOOL] OPEN_APP | app={app}")
        return f"Success: {app} opened. [STUB]"

class StubCloseAppTool(BaseTool):
    name           = "CLOSE_APP"
    description    = "Closes an application. [STUB]"
    security_level = SecurityLevel.ADMIN
    args_schema    = _StubOpenAppArgs

    def execute(self, **kwargs) -> str:
        app = kwargs.get("app", "unknown")
        stdlib_logger.info(f"[STUB TOOL] CLOSE_APP | app={app}")
        return f"Success: {app} closed. [STUB]"

class StubWebSearchTool(BaseTool):
    name           = "WEB_SEARCH"
    description    = "Searches the web for a query. [STUB]"
    security_level = SecurityLevel.USER
    args_schema    = _StubWebSearchArgs

    def execute(self, **kwargs) -> str:
        query = kwargs.get("query", "")
        stdlib_logger.info(f"[STUB TOOL] WEB_SEARCH | query='{query[:60]}'")
        return f"Success: Searched for '{query}'. [STUB]"

class StubMediaControlTool(BaseTool):
    name           = "MEDIA_CONTROL"
    description    = "Controls media playback. [STUB]"
    security_level = SecurityLevel.USER
    args_schema    = _StubMediaArgs

    def execute(self, **kwargs) -> str:
        action = kwargs.get("action", "")
        stdlib_logger.info(f"[STUB TOOL] MEDIA_CONTROL | action={action}")
        return f"Success: Media action '{action}' executed. [STUB]"

class StubSystemPowerTool(BaseTool):
    name           = "SYSTEM_POWER"
    description    = "Controls system power state. [STUB]"
    security_level = SecurityLevel.ADMIN
    args_schema    = _StubPowerArgs

    def execute(self, **kwargs) -> str:
        action = kwargs.get("action", "")
        stdlib_logger.info(f"[STUB TOOL] SYSTEM_POWER | action={action}")
        return f"Success: Power action '{action}' executed. [STUB]"


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY STACK WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class MemoryStack:
    def __init__(
        self,
        conversation: ConversationStore,
        profile:      Any,
        state:        SessionStateStore,
    ) -> None:
        self.conversation = conversation
        self.profile      = profile
        self.state        = state


class _StubUserProfileStore:
    def get(self, key: str) -> Any:
        return None
    async def set(self, key: str, value: Any) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# EXPECTED TOOL COUNT
# ══════════════════════════════════════════════════════════════════════════════

_EXPECTED_TOOL_COUNT: int = 5


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap() -> CyraxContext:
    # ── Step 1: Validate settings ─────────────────────────────────────────────
    stdlib_logger.info("Step 1/10 — Settings validation...")
    stdlib_logger.info(
        f"ACTIVE_LLM={settings.ACTIVE_LLM.value} | "
        f"BRAIN_TIMEOUT={settings.BRAIN_TIMEOUT_SECONDS}s | "
        f"TOOL_TIMEOUT={settings.TOOL_TIMEOUT_SECONDS}s"
    )

    # ── Step 2: Lockout store ─────────────────────────────────────────────────
    stdlib_logger.info("Step 2/10 — Lockout store...")
    try:
        lockout_store = LockoutStore()
        locked, remaining = lockout_store.is_locked_out()
        if locked:
            _hard_fail(
                f"System is currently locked out. "
                f"Remaining: {int(remaining)}s. "
                f"Boot aborted for security."
            )
    except Exception as exc:
        _hard_fail(f"LockoutStore initialisation failed: {exc}")

    # ── Step 3: Security guard ────────────────────────────────────────────────
    stdlib_logger.info("Step 3/10 — Security guard...")
    try:
        security_guard = SecurityGuard(store=lockout_store)
    except Exception as exc:
        _hard_fail(f"SecurityGuard initialisation failed: {exc}")

    # ── Step 4: Tool registry ─────────────────────────────────────────────────
    stdlib_logger.info("Step 4/10 — Tool registry...")
    try:
        registry = ToolRegistry(security=security_guard)

        registry.register(StubOpenAppTool())
        registry.register(StubCloseAppTool())
        registry.register(StubWebSearchTool())
        registry.register(StubMediaControlTool())
        registry.register(StubSystemPowerTool())

        registry._lock()
    except Exception as exc:
        _hard_fail(f"Tool registry initialisation failed: {exc}")

    if registry.count() != _EXPECTED_TOOL_COUNT:
        _hard_fail(
            f"Tool registry integrity check FAILED. "
            f"Expected {_EXPECTED_TOOL_COUNT} tools, got {registry.count()}."
        )

    # ── Step 5: Memory stack ──────────────────────────────────────────────────
    stdlib_logger.info("Step 5/10 — Memory stack...")
    try:
        state_store        = SessionStateStore()
        conversation_store = ConversationStore(
            session_id="device_master_001",
            summarise_callback=None,
        )
        profile_store      = _StubUserProfileStore()

        memory = MemoryStack(
            conversation=conversation_store,
            profile=profile_store,
            state=state_store,
        )
    except Exception as exc:
        _hard_fail(f"Memory stack initialisation failed: {exc}")

    # ── Step 6: LLM providers ─────────────────────────────────────────────────
    stdlib_logger.info("Step 6/10 — LLM providers...")
    try:
        # THE REAL PROVIDERS ARE INJECTED HERE
        providers: dict[str, BaseProvider] = {
            "groq":   GroqProvider(),
            "gemini": GeminiProvider(),
        }
    except Exception as exc:
        _hard_fail(f"LLM provider initialisation failed: {exc}")

    stdlib_logger.info(f"LLM providers: {list(providers.keys())} registered. ✓")

    # ── Step 7: Brain router ──────────────────────────────────────────────────
    stdlib_logger.info("Step 7/10 — Brain router (MoERouter)...")
    try:
        brain_router = MoERouter(providers=providers)
    except Exception as exc:
        _hard_fail(f"MoERouter initialisation failed: {exc}")

    # ── Step 8: Fallback policy ───────────────────────────────────────────────
    stdlib_logger.info("Step 8/10 — Fallback policy...")
    try:
        fallback_policy = FallbackPolicy()
    except Exception as exc:
        _hard_fail(f"FallbackPolicy initialisation failed: {exc}")

    # ── Step 9: Dispatcher ────────────────────────────────────────────────────
    stdlib_logger.info("Step 9/10 — Dispatcher...")
    try:
        dispatcher = Dispatcher()
    except Exception as exc:
        _hard_fail(f"Dispatcher initialisation failed: {exc}")

    # ── Step 10: Assemble CyraxContext ────────────────────────────────────────
    stdlib_logger.info("Step 10/10 — Assembling CyraxContext...")
    try:
        ctx = CyraxContext(
            session_id      = "device_master_001",
            dispatcher      = dispatcher,
            tool_registry   = registry,
            brain_router    = brain_router,
            memory          = memory,
            security        = security_guard,
            fallback_policy = fallback_policy,
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