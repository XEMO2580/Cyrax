"""Compatibility shim.

The test suite (and external callers) import `tools.scheduler`, but this
repo's implementation lives in `tools/schedular.py` (legacy misspelling).
"""

from __future__ import annotations

# This legacy-misspelled module is the real implementation of the
# scheduler tool (the name was fixed later via tools/scheduler.py shim).

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from security.auth import SecurityLevel
from tools.registry import BaseTool

logger = logging.getLogger(__name__)

_MAX_DELAY_SECONDS: int = 3600
_MAX_ARGS_STR_LEN: int = 2000

_SCHEDULABLE_LEVELS: frozenset[SecurityLevel] = frozenset({
    SecurityLevel.UNRESTRICTED,
    SecurityLevel.USER,
})


class ScheduleTaskSchema(BaseModel):
    tool_name: str = Field(..., description="Registered UNRESTRICTED or USER tool name.")
    args: dict[str, Any] = Field(default_factory=dict, description="Args for the target tool.")
    delay_seconds: int = Field(..., ge=1, le=_MAX_DELAY_SECONDS, description="Delay in seconds.")


class ScheduleTaskTool(BaseTool):
    name = "SCHEDULE_TASK"
    description = (
        "Schedules a registered UNRESTRICTED or USER-level tool to execute after a specified delay. "
        "ADMIN-level tools cannot be scheduled."
    )
    security_level = SecurityLevel.USER
    args_schema = ScheduleTaskSchema

    def execute(self, tool_name: str, args: dict[str, Any], delay_seconds: int) -> str:  # type: ignore[override]
        ctx = getattr(self, "_ctx", None)
        if ctx is None:
            return "Error: CyraxContext was not injected into ScheduleTaskTool."

        loop: asyncio.AbstractEventLoop | None = getattr(self, "_loop", None)
        if loop is None:
            return "Error: Event loop was not injected into ScheduleTaskTool."

        if not loop.is_running():
            return "Error: The injected event loop is no longer running."

        tool_names = ctx.tool_registry.get_tool_names()
        if tool_name not in tool_names:
            return f"Error: Tool '{tool_name}' is not registered."

        tool_metadata = ctx.tool_registry.get_tool_metadata(tool_name)
        if tool_metadata is None:
            return f"Error: Could not retrieve metadata for tool '{tool_name}'."

        target_level: SecurityLevel = tool_metadata.get("security_level", SecurityLevel.ADMIN)
        if target_level not in _SCHEDULABLE_LEVELS:
            return (
                "Error: ADMIN-level tools cannot be scheduled for background execution. "
                f"'{tool_name}' requires SecurityLevel.ADMIN."
            )

        try:
            args_str = json.dumps(args)
        except (TypeError, ValueError) as exc:
            return f"Error: Args are not JSON-serialisable — {exc}"

        if len(args_str) > _MAX_ARGS_STR_LEN:
            return "Error: Args payload is too large"

        frozen_args = dict(args)
        self._loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(
                _delayed_dispatch(
                    ctx=ctx,
                    tool_name=tool_name,
                    args=frozen_args,
                    delay_seconds=delay_seconds,
                )
            )
        )

        minutes, seconds = divmod(delay_seconds, 60)
        time_desc = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
        return f"Success: '{tool_name}' scheduled to run in {time_desc}."


async def _delayed_dispatch(ctx: Any, tool_name: str, args: dict[str, Any], delay_seconds: int) -> None:
    try:
        await asyncio.sleep(delay_seconds)
        await ctx.tool_registry.execute_tool(tool_name, args)
    except Exception:
        logger.exception("[SCHEDULER] Scheduled dispatch failed")



