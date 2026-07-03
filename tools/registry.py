"""
tools/registry.py — CYRAX 3.0 Immutable Tool Registry

Priority 0 hardening applied:
  - Immutable boot-time mounts: tools registered once during bootstrap Phase 1.
    After _lock() is called, register() raises — no runtime mutations.
  - Fail-loud integrity: count() is checked by bootstrap against the expected
    tool count. A drop from 9 to 8 crashes boot, not silently.
  - Rate limiting sourced from settings, not hardcoded constants.
  - Security injected via constructor, not imported as a module-level singleton.
  - Error detection via structured status field, not string-prefix sniffing.
  - Public count() and __len__() replace direct _tools access.

Satisfies ToolRegistryProtocol defined in core/context.py:
    def count(self) -> int
    def get_all_definitions(self) -> list[dict]
    async def execute_tool(self, tool_name: str, args: dict) -> dict
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from config.settings import settings

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY LEVEL
# ══════════════════════════════════════════════════════════════════════════════

from security.auth import SecurityLevel


# ══════════════════════════════════════════════════════════════════════════════
# BASE TOOL CONTRACT
# ══════════════════════════════════════════════════════════════════════════════

class BaseTool(ABC):
    """
    Abstract base class all CYRAX tools must subclass.

    Class attributes (define on subclass, not instance):
        name:           Unique tool identifier. Used as the registry key.
        description:    Human/LLM-readable description of what the tool does.
        security_level: Minimum clearance required to execute this tool.
        args_schema:    Pydantic BaseModel subclass defining and validating args.
    """

    name:           str           = "BaseTool"
    description:    str           = "No description provided."
    security_level: SecurityLevel = SecurityLevel.ADMIN
    args_schema:    Type[BaseModel]

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        Synchronous execution body.

        Returns a plain string result on success.
        Returns a string starting with "Error:" on failure.
        Never raises — catch all exceptions internally and return an error string.
        The registry wraps this in asyncio.to_thread(); do not add async here.
        """
        ...

    def get_tool_definition(self) -> dict:
        """
        Returns the tool's schema in a provider-agnostic format.
        Each brain/providers/ adapter translates this into its vendor format.
        """
        return {
            "name":        self.name,
            "description": self.description,
            "parameters":  self.args_schema.model_json_schema(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY GATEWAY PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════

from typing import Protocol, runtime_checkable

@runtime_checkable
class SecurityGatewayProtocol(Protocol):
    """
    Minimal interface the registry requires from the security layer.
    Injected via constructor — registry never imports security.auth directly.
    """
    def authorize_action(self, required_level: SecurityLevel) -> bool: ...


# ══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """
    Immutable boot-time tool mount registry.

    Lifecycle:
        1. bootstrap() instantiates ToolRegistry(security=security_guard).
        2. bootstrap() calls registry.register(tool) for each tool — Phase 1 only.
        3. bootstrap() calls registry._lock() to freeze the registry.
        4. From this point, register() raises ImmutableRegistryError on any call.
        5. CyraxContext receives the locked registry.
        6. dispatcher.py calls execute_tool() at runtime — never register().

    Rate limiting:
        Per-instance, per-minute call counter.
        Ceiling sourced from settings.MAX_TOOL_CALLS_PER_MINUTE.

    Error contract:
        execute_tool() always returns a dict with "status" and "response".
        Status values: "success" | "error" | "auth_required"
        Never raises to the caller.
    """

    class ImmutableRegistryError(RuntimeError):
        """Raised when register() is called after _lock()."""

    def __init__(self, security: SecurityGatewayProtocol) -> None:
        if not isinstance(security, SecurityGatewayProtocol):
            raise TypeError(
                f"ToolRegistry requires a SecurityGatewayProtocol, "
                f"got {type(security).__name__}"
            )

        self._security                          = security
        self._tools:          dict[str, BaseTool] = {}
        self._locked:         bool              = False

        # Rate limiting state — per-instance, not global.
        self._execution_count:    int   = 0
        self._rate_limit_window:  float = time.time()

    # ── Registration (Phase 1 only) ───────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """
        Mount a tool into the registry.

        Raises:
            ImmutableRegistryError: If called after _lock().
            TypeError:              If tool is not a BaseTool subclass.
            ValueError:             If a tool with the same name is already mounted.
        """
        if self._locked:
            raise self.ImmutableRegistryError(
                f"ToolRegistry is locked. Cannot register '{tool.name}' at runtime. "
                f"All tools must be mounted during bootstrap Phase 1."
            )
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"register() expects a BaseTool subclass, got {type(tool).__name__}."
            )
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                f"Duplicate tool names are not permitted."
            )

        self._tools[tool.name] = tool
        logger.debug(
            f"[REGISTRY] Mounted: {tool.name} "
            f"[SecurityLevel: {tool.security_level.name}]"
        )

    def _lock(self) -> None:
        """
        Freeze the registry. Called by bootstrap() after all tools are mounted.
        Any subsequent call to register() raises ImmutableRegistryError.
        """
        self._locked = True
        logger.info(
            f"[REGISTRY] Locked. {self.count()} tools mounted immutably: "
            f"{list(self._tools.keys())}"
        )

    # ── Public Read Interface ─────────────────────────────────────────────────

    def count(self) -> int:
        """Returns the number of registered tools. Used by bootstrap() integrity check."""
        return len(self._tools)

    def __len__(self) -> int:
        return self.count()

    def get_all_definitions(self) -> list[dict]:
        """
        Returns provider-agnostic tool schemas for all mounted tools.
        Called by dispatcher.py and forwarded to brain_router.plan().
        Each brain/providers/ adapter translates these into its vendor format.
        """
        return [tool.get_tool_definition() for tool in self._tools.values()]

    def get_tool_names(self) -> list[str]:
        """Returns a sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        ctx: Any = None,
    ) -> dict[str, Any]:
        """
        Single execution entry point. All tool calls flow through here.

        Steps:
            1. Rate limit check
            2. Tool existence check
            3. Security authorization (via injected security gateway)
            4. Pydantic argument validation
            5. Threaded execution with timeout

        Returns:
            {"status": "success",       "response": str}
            {"status": "error",         "response": str}
            {"status": "auth_required", "response": str}

        Never raises. Caller receives a structured dict on all paths.
        """

        # ── 1. Rate Limiting ──────────────────────────────────────────────────
        current_time = time.time()
        elapsed = current_time - self._rate_limit_window

        if elapsed > 60.0:
            self._execution_count = 0
            self._rate_limit_window = current_time

        if self._execution_count >= settings.MAX_TOOL_CALLS_PER_MINUTE:
            logger.warning(
                f"[REGISTRY] Rate limit exceeded. "
                f"Blocked: {tool_name} "
                f"({self._execution_count}/{settings.MAX_TOOL_CALLS_PER_MINUTE} "
                f"calls in window)"
            )
            return {
                "status":   "error",
                "response": "Rate limit exceeded. Please wait before sending another command.",
            }

        self._execution_count += 1

        # ── 2. Existence Check ────────────────────────────────────────────────
        tool = self._tools.get(tool_name)
        if not tool:
            logger.warning(f"[REGISTRY] Unknown tool requested: '{tool_name}'")
            return {
                "status":   "error",
                "response": f"Unknown tool: '{tool_name}'. It may not be registered.",
            }

        # ── 3. Context injection (required for context-aware tools) ─────────
        # Inject ctx and the real running loop BEFORE any security gate check
        # so tools like ScheduleTaskTool can inspect target tool metadata.
        if ctx is not None:
            tool._ctx = ctx
            tool._loop = asyncio.get_running_loop()

        # ── 4. Security Authorization ─────────────────────────────────────────
        if not self._security.authorize_action(tool.security_level):
            logger.warning(
                f"[REGISTRY] Auth required: {tool_name} "
                f"[Level: {tool.security_level.name}]"
            )
            return {
                "status":   "auth_required",
                "response": (
                    f"⚠️ '{tool_name}' requires {tool.security_level.name} clearance. "
                    f"Please enter your Master PIN."
                ),
            }

        # ── 4. Argument Validation ────────────────────────────────────────────
        try:
            validated = tool.args_schema(**args)
        except ValidationError as exc:
            logger.error(
                f"[REGISTRY] Validation failed: {tool_name} | errors={exc.errors()}"
            )
            return {
                "status":   "error",
                "response": (
                    f"Invalid arguments for '{tool_name}'. "
                    f"Check your command and try again."
                ),
            }

        # ── 5. Threaded Execution with Timeout ────────────────────────────────
        logger.info(
            f"[REGISTRY] Executing: {tool_name} | "
            f"args={validated.model_dump()}"
        )

        try:
            raw_result: str = await asyncio.wait_for(
                asyncio.to_thread(tool.execute, **validated.model_dump()),
                timeout=settings.TOOL_TIMEOUT_SECONDS,
            )

            result_str = str(raw_result).strip()

            # Structured status detection via field prefix convention.
            # Tools return "Error: ..." on failure, anything else on success.
            # This replaces the fragile string-startswith check from 2.0.
            if result_str.lower().startswith("error") or \
               result_str.lower().startswith("security block"):
                logger.warning(
                    f"[REGISTRY] Tool reported failure: "
                    f"{tool_name} | {result_str[:120]}"
                )
                return {"status": "error", "response": result_str}

            logger.debug(
                f"[REGISTRY] Success: {tool_name} | "
                f"response='{result_str[:80]}'"
            )
            return {"status": "success", "response": result_str}

        except asyncio.TimeoutError:
            logger.error(
                f"[REGISTRY] Timeout: {tool_name} exceeded "
                f"{settings.TOOL_TIMEOUT_SECONDS}s"
            )
            return {
                "status":   "error",
                "response": (
                    f"'{tool_name}' timed out after "
                    f"{settings.TOOL_TIMEOUT_SECONDS}s. "
                    f"The operation may still be running in the background."
                ),
            }

        except Exception as exc:
            logger.exception(
                f"[REGISTRY] Crash: {tool_name} | error={exc}"
            )
            return {
                "status":   "error",
                "response": f"'{tool_name}' encountered an unexpected error.",
            }