"""
tools/memory_ops.py — CYRAX 3.0 Memory Operation Tools

Tools:
    CoreMemoryWriteTool — Saves a fact or preference to long-term profile memory
    CoreMemoryReadTool  — Reads from long-term profile memory

Security:
    Both tools — USER
    Profile memory is personal data — requires a valid session but not a PIN.

Context injection:
    Both tools require self._ctx to access ctx.memory.profile.
    self._ctx is injected by ToolRegistry.execute_tool(ctx=ctx) before
    asyncio.to_thread() is called — the same pattern used by SCHEDULE_TASK.

    If self._ctx is missing, tools return a clear Error string rather than
    raising AttributeError into the registry.

execute() constraint:
    BaseTool.execute() must be synchronous. Memory profile methods are async.
    This is resolved by running the async work on the injected event loop:

        asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    self._loop is injected by the registry alongside self._ctx.

Value serialisation:
    Values are stored as Python primitives (str, int, float, bool, list, dict).
    Non-serialisable objects are coerced to str before storage so the profile
    file always remains valid JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from security.auth import SecurityLevel
from tools.registry import BaseTool

logger = logging.getLogger(__name__)

# Maximum key length to prevent abuse.
_MAX_KEY_CHARS:   int = 128
# Maximum value size when serialised to JSON string.
_MAX_VALUE_CHARS: int = 4_000
# Maximum total profile entries to prevent unbounded growth.
_MAX_PROFILE_ENTRIES: int = 500
# Timeout for async profile operations called from sync execute().
_ASYNC_TIMEOUT: float = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: CORE MEMORY WRITE
# ══════════════════════════════════════════════════════════════════════════════

class CoreMemoryWriteSchema(BaseModel):
    key: str = Field(
        ...,
        max_length=_MAX_KEY_CHARS,
        description=(
            "The memory key to write. Use snake_case descriptive names "
            "(e.g. 'user_name', 'preferred_browser', 'favorite_color'). "
            f"Maximum {_MAX_KEY_CHARS} characters."
        ),
    )
    value: str = Field(
        ...,
        max_length=_MAX_VALUE_CHARS,
        description=(
            "The value to store. Can represent any fact, preference, or "
            "learned information about the user. "
            f"Maximum {_MAX_VALUE_CHARS} characters."
        ),
    )


class CoreMemoryWriteTool(BaseTool):
    """
    Saves a fact or user preference to long-term profile memory.

    The LLM calls this when it learns something about the user that should
    persist across sessions — name, preferences, habits, ongoing projects.

    Examples of valid writes:
        key="user_name"              value="Xemo"
        key="preferred_browser"      value="brave"
        key="current_project"        value="CYRAX AI operating system"
        key="prefers_dark_mode"      value="true"

    Overwrite behaviour:
        If the key already exists, its value is replaced. This is intentional —
        the LLM should update facts as they change rather than accumulating
        duplicates.

    Entry limit:
        Writing when the profile already has _MAX_PROFILE_ENTRIES entries
        returns an error rather than silently growing the store unboundedly.
    """

    name           = "CORE_MEMORY_WRITE"
    description    = (
        "Saves a fact, preference, or piece of learned user information to "
        "CYRAX's long-term memory. Use this to remember things about the user "
        "across sessions (e.g. their name, preferences, ongoing tasks). "
        "If the key already exists, its value is updated."
    )
    security_level = SecurityLevel.USER
    args_schema    = CoreMemoryWriteSchema

    def execute(self, key: str, value: str) -> str:  # type: ignore[override]
        # ── Dependency checks ─────────────────────────────────────────────────
        ctx  = getattr(self, "_ctx",  None)
        loop = getattr(self, "_loop", None)

        if ctx is None:
            return (
                "Error: CoreMemoryWriteTool requires a CyraxContext but none "
                "was injected. Ensure ToolRegistry.execute_tool() passes ctx."
            )
        if loop is None:
            return (
                "Error: CoreMemoryWriteTool requires an injected event loop. "
                "Ensure ToolRegistry.execute_tool() sets tool._loop before dispatch."
            )

        # ── Input sanitisation ────────────────────────────────────────────────
        key   = key.strip().lower().replace(" ", "_")
        value = value.strip()

        if not key:
            return "Error: Memory key cannot be empty."
        if not value:
            return "Error: Memory value cannot be empty."

        # Serialise to confirm value is JSON-safe before storing.
        try:
            serialised = json.dumps(value)
        except (TypeError, ValueError):
            value      = str(value)
            serialised = json.dumps(value)

        if len(serialised) > _MAX_VALUE_CHARS:
            return (
                f"Error: Value is too large to store "
                f"({len(serialised):,} chars > {_MAX_VALUE_CHARS:,} limit). "
                f"Summarise the value before storing."
            )

        # ── Entry limit check (async) ─────────────────────────────────────────
        try:
            existing = asyncio.run_coroutine_threadsafe(
                ctx.memory.profile.get_all(),
                loop,
            ).result(timeout=_ASYNC_TIMEOUT)
        except Exception as exc:
            logger.error(f"[MEMORY_WRITE] Failed to read profile for limit check: {exc}")
            return f"Error: Could not access long-term memory — {exc}"

        is_new_key = key not in existing
        if is_new_key and len(existing) >= _MAX_PROFILE_ENTRIES:
            return (
                f"Error: Long-term memory is full "
                f"({_MAX_PROFILE_ENTRIES} entries). "
                f"Delete an existing entry before adding new ones. "
                f"Use CORE_MEMORY_READ to review what is stored."
            )

        # ── Write to profile store ────────────────────────────────────────────
        try:
            asyncio.run_coroutine_threadsafe(
                ctx.memory.profile.set(key, value),
                loop,
            ).result(timeout=_ASYNC_TIMEOUT)
        except Exception as exc:
            logger.error(f"[MEMORY_WRITE] Write failed for key '{key}': {exc}")
            return f"Error: Failed to save to long-term memory — {exc}"

        action = "Updated" if not is_new_key else "Saved"
        logger.info(f"[MEMORY_WRITE] {action}: {key!r} = {str(value)[:60]!r}")
        return (
            f"Success: {action} memory entry '{key}'. "
            f"I will remember this across future sessions."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: CORE MEMORY READ
# ══════════════════════════════════════════════════════════════════════════════

class CoreMemoryReadSchema(BaseModel):
    key: str = Field(
        default="",
        max_length=_MAX_KEY_CHARS,
        description=(
            "The memory key to read. Leave empty to retrieve all stored memories. "
            "Use exact key names as stored (e.g. 'user_name', 'preferred_browser')."
        ),
    )


class CoreMemoryReadTool(BaseTool):
    """
    Reads from CYRAX's long-term profile memory.

    When called with a specific key, returns that entry's value.
    When called with an empty key, returns all stored profile entries
    formatted as a readable list for the LLM to reason over.

    The LLM should call this at the start of a session or when it needs
    to recall something it was told in a previous conversation.

    Output format (all entries):
        Long-term memory (N entries):
          - user_name: Xemo
          - preferred_browser: brave
          - current_project: CYRAX AI operating system
          ...

    Output format (single key):
        Memory 'user_name': Xemo
    """

    name           = "CORE_MEMORY_READ"
    description    = (
        "Reads from CYRAX's long-term memory. "
        "Provide a specific key to retrieve one entry, or leave key empty "
        "to retrieve all stored memories. Use this to recall facts about "
        "the user learned in previous sessions."
    )
    security_level = SecurityLevel.USER
    args_schema    = CoreMemoryReadSchema

    def execute(self, key: str = "") -> str:  # type: ignore[override]
        # ── Dependency checks ─────────────────────────────────────────────────
        ctx  = getattr(self, "_ctx",  None)
        loop = getattr(self, "_loop", None)

        if ctx is None:
            return (
                "Error: CoreMemoryReadTool requires a CyraxContext but none "
                "was injected. Ensure ToolRegistry.execute_tool() passes ctx."
            )
        if loop is None:
            return (
                "Error: CoreMemoryReadTool requires an injected event loop. "
                "Ensure ToolRegistry.execute_tool() sets tool._loop before dispatch."
            )

        key = key.strip().lower()

        # ── Single key read ───────────────────────────────────────────────────
        if key:
            try:
                value = asyncio.run_coroutine_threadsafe(
                    ctx.memory.profile.get(key),
                    loop,
                ).result(timeout=_ASYNC_TIMEOUT)
            except Exception as exc:
                logger.error(f"[MEMORY_READ] Read failed for key '{key}': {exc}")
                return f"Error: Failed to read long-term memory — {exc}"

            if value is None:
                return (
                    f"Memory key '{key}' does not exist. "
                    f"Use CORE_MEMORY_READ with an empty key to list all entries."
                )

            logger.debug(f"[MEMORY_READ] Read key={key!r}")
            return f"Memory '{key}': {value}"

        # ── Full profile read ─────────────────────────────────────────────────
        try:
            all_entries = asyncio.run_coroutine_threadsafe(
                ctx.memory.profile.get_all(),
                loop,
            ).result(timeout=_ASYNC_TIMEOUT)
        except Exception as exc:
            logger.error(f"[MEMORY_READ] get_all() failed: {exc}")
            return f"Error: Failed to retrieve long-term memory — {exc}"

        if not all_entries:
            return (
                "Long-term memory is empty. "
                "Use CORE_MEMORY_WRITE to save facts about the user."
            )

        lines = [f"Long-term memory ({len(all_entries)} entries):"]
        for k, v in sorted(all_entries.items()):
            v_str = str(v)
            if len(v_str) > 120:
                v_str = v_str[:120] + "..."
            lines.append(f"  - {k}: {v_str}")

        logger.debug(f"[MEMORY_READ] Retrieved all {len(all_entries)} entries.")
        return "\n".join(lines)