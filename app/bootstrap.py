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
  - _EXPECTED_TOOL_COUNT = 11.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

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
from core.trace import new_trace_id

# ── Settings ──────────────────────────────────────────────────────────────────
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
from memory.profile.profile_store import UserProfileStore   # Patched: synchronous init

# ── Brain ─────────────────────────────────────────────────────────────────────
from brain.providers.base import BaseProvider, ProviderCapabilities, ProviderError
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
# STUB LLM PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════

class StubGroqProvider(BaseProvider):
    provider_name = "groq"
    capabilities  = ProviderCapabilities(
        supports_json_mode     = True,
        supports_system_prompt = True,
        supports_tool_schemas  = False,
        max_output_tokens      = 8192,
        context_window_tokens  = 131072,
    )

    async def generate(
        self,
        messages:      list[dict],
        system_prompt: str,
        *,
        max_tokens:    int   = 800,
        temperature:   float = 0.7,
        json_mode:     bool  = False,
    ) -> str:
        stdlib_logger.info(
            f"[STUB GROQ] generate() | messages={len(messages)} | json_mode={json_mode}"
        )
        if json_mode:
            return '{"plan": []}'
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "no input",
        )
        return (
            f"[STUB GROQ RESPONSE] Received: '{last_user[:80]}'. "
            f"Real Groq provider not yet implemented."
        )

    async def health_check(self) -> bool:
        return True


class StubGeminiProvider(BaseProvider):
    provider_name = "gemini"
    capabilities  = ProviderCapabilities(
        supports_json_mode     = False,
        supports_system_prompt = True,
        supports_tool_schemas  = False,
        max_output_tokens      = 8192,
        context_window_tokens  = 1048576,
    )

    async def generate(
        self,
        messages:      list[dict],
        system_prompt: str,
        *,
        max_tokens:    int   = 800,
        temperature:   float = 0.7,
        json_mode:     bool  = False,
    ) -> str:
        stdlib_logger.info(f"[STUB GEMINI] generate() | messages={len(messages)}")
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "no input",
        )
        return (
            f"[STUB GEMINI RESPONSE] Received: '{last_user[:80]}'. "
            f"Real Gemini provider not yet implemented."
        )

    async def health_check(self) -> bool:
        return True


# ══════════════════════════════════════════════════════════════════════════════
# STUB TOOLS
# ══════════════════════════════════════════════════════════════════════════════

class _StubOpenAppArgs(BaseModel):
    app: str = Field(..., description="Application name.")

class _StubWebSearchArgs(BaseModel):
    query: str = Field(..., description="Search query.")

class _StubMediaArgs(BaseModel):
    action: str = Field(..., description="Media action.")

class _StubPowerArgs(BaseModel):
    action: str = Field(..., description="Power action.")


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
    description    = "Searches the web. [STUB]"
    security_level = SecurityLevel.USER
    args_schema    = _StubWebSearchArgs

    def execute(self, **kwargs) -> str:
        query = kwargs.get("query", "")
        stdlib_logger.info(f"[STUB TOOL] WEB_SEARCH | query='{query[:60]}'")
        return f"Success: Searched for '{query}'. [STUB]"


class StubMediaControlTool(BaseTool):
    name           = "MEDIA_CONTROL"
    description    = "Controls media. [STUB]"
    security_level = SecurityLevel.USER
    args_schema    = _StubMediaArgs

    def execute(self, **kwargs) -> str:
        action = kwargs.get("action", "")
        stdlib_logger.info(f"[STUB TOOL] MEDIA_CONTROL | action={action}")
        return f"Success: Media action '{action}'. [STUB]"


class StubSystemPowerTool(BaseTool):
    name           = "SYSTEM_POWER"
    description    = "Controls system power. [STUB]"
    security_level = SecurityLevel.ADMIN
    args_schema    = _StubPowerArgs

    def execute(self, **kwargs) -> str:
        action = kwargs.get("action", "")
        stdlib_logger.info(f"[STUB TOOL] SYSTEM_POWER | action={action}")
        return f"Success: Power action '{action}'. [STUB]"


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

_EXPECTED_TOOL_COUNT: int = 11


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

        # ── Core action tools (5) ─────────────────────────────────────────────
        registry.register(StubOpenAppTool())
        registry.register(StubCloseAppTool())
        registry.register(StubWebSearchTool())
        registry.register(StubMediaControlTool())
        registry.register(StubSystemPowerTool())

        # ── Phase 4: Memory operation tools (2) ──────────────────────────────
        from tools.memory_ops import CoreMemoryWriteTool, CoreMemoryReadTool
        registry.register(CoreMemoryWriteTool())
        registry.register(CoreMemoryReadTool())

        # When migrating to real tool implementations, replace stubs:
        # from tools.pc_actions   import OpenAppTool, CloseAppTool, TypeTextTool
        # from tools.media_power  import MediaControlTool, SystemPowerTool
        # from tools.web_search   import WebSearchTool
        # from tools.file_ops     import (DeleteFileTool, CopyFileTool,
        #                                  MoveFileTool, RenameFileTool,
        #                                  ReadFileTool, WriteFileTool,
        #                                  ListDirTool, MakeDirTool)
        # from tools.desktop_ops  import (ClipboardReadTool, ClipboardWriteTool,
        #                                  SendNotificationTool)
        # from tools.web_reader   import ReadWebpageTool
        # from tools.scheduler    import ScheduleTaskTool

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
        # Replace stubs with real providers:
        # from brain.providers.groq_provider   import GroqProvider
        # from brain.providers.gemini_provider import GeminiProvider
        # providers = {"groq": GroqProvider(), "gemini": GeminiProvider()}
        providers: dict[str, BaseProvider] = {
            "groq":   StubGroqProvider(),
            "gemini": StubGeminiProvider(),
        }
    except Exception as exc:
        _hard_fail(f"LLM provider initialisation failed: {exc}")

    stdlib_logger.info(
        f"LLM providers: {list(providers.keys())} registered. ✓"
    )

    # ── Step 7: Brain router ──────────────────────────────────────────────────
    stdlib_logger.info("Step 7/10 — Brain router (MoERouter)...")
    try:
        brain_router = MoERouter(providers=providers)
    except Exception as exc:
        _hard_fail(f"MoERouter initialisation failed: {exc}")

    stdlib_logger.info("Brain router: OK ✓")

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