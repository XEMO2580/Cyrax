"""
tests/tools/test_scheduler.py — Certification suite for tools/scheduler.py

TECHNIQUE NOTE (see Phase 4.5A audit, explicitly flagged by task directive):
  1. self._loop must be a REAL running event loop — call_soon_threadsafe
     requires a genuine asyncio.AbstractEventLoop, not a Mock.
  2. asyncio.sleep inside _delayed_dispatch is patched so scheduled tasks
     fire immediately and deterministically — otherwise tests would
     genuinely wait out delay_seconds (up to 3600s per the schema ceiling),
     hanging the suite.

Tests call tool.execute() via asyncio.to_thread() (mirroring
ToolRegistry.execute_tool()'s real threading behaviour), then await a
short synchronization window to let the scheduled coroutine actually run
on the test's event loop before asserting on its side effects.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from security.auth import SecurityLevel
from tools.scheduler import (
    ScheduleTaskTool, ScheduleTaskSchema,
    _MAX_DELAY_SECONDS, _MAX_ARGS_STR_LEN,
)

pytestmark = pytest.mark.asyncio


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_ctx() -> Mock:
    ctx = Mock(name="CyraxContext")
    ctx.tool_registry = Mock(name="ToolRegistry")
    ctx.tool_registry.get_tool_names.return_value = [
        "WEB_SEARCH", "SEND_NOTIFICATION", "OPEN_APP", "SYSTEM_POWER",
    ]

    def _metadata(tool_name: str):
        levels = {
            "WEB_SEARCH":        SecurityLevel.USER,
            "SEND_NOTIFICATION": SecurityLevel.UNRESTRICTED,
            "OPEN_APP":          SecurityLevel.UNRESTRICTED,
            "SYSTEM_POWER":      SecurityLevel.ADMIN,
        }
        if tool_name not in levels:
            return None
        return {"name": tool_name, "security_level": levels[tool_name]}

    ctx.tool_registry.get_tool_metadata.side_effect = _metadata
    ctx.tool_registry.execute_tool = AsyncMock(
        return_value={"status": "success", "response": "Done."}
    )
    return ctx


async def _run_schedule_execute(tool, ctx, loop, **kwargs) -> str:
    """Runs execute() on a worker thread with self._ctx/self._loop injected."""
    tool._ctx  = ctx
    tool._loop = loop
    return await asyncio.wait_for(
        asyncio.to_thread(tool.execute, **kwargs),
        timeout=5.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ScheduleTaskTool
# ══════════════════════════════════════════════════════════════════════════════

class TestScheduleTaskTool:

    async def test_happy_path_immediate_dispatch(self, mock_ctx: Mock) -> None:
        """
        Deterministic scheduling test: asyncio.sleep is patched to a no-op
        so the delayed task fires as soon as it's scheduled, without the
        test actually waiting delay_seconds in real time.
        """
        loop = asyncio.get_running_loop()

        with patch("tools.scheduler.asyncio.sleep", AsyncMock(return_value=None)):
            result = await _run_schedule_execute(
                ScheduleTaskTool(), mock_ctx, loop,
                tool_name="WEB_SEARCH", args={"query": "test"}, delay_seconds=10,
            )
            # Give the event loop one cycle to run the posted coroutine.
            await asyncio.sleep(0)
            await asyncio.sleep(0)  # second yield — ensure_future scheduling settles

        assert result.startswith("Success:")
        assert "WEB_SEARCH" in result
        mock_ctx.tool_registry.execute_tool.assert_awaited_once_with(
            "WEB_SEARCH", {"query": "test"}
        )

    def test_invalid_args_delay_too_large(self) -> None:
        with pytest.raises(ValidationError):
            ScheduleTaskSchema(
                tool_name="WEB_SEARCH", args={}, delay_seconds=_MAX_DELAY_SECONDS + 1
            )

    def test_invalid_args_delay_zero(self) -> None:
        with pytest.raises(ValidationError):
            ScheduleTaskSchema(tool_name="WEB_SEARCH", args={}, delay_seconds=0)

    async def test_missing_context_dependency(self) -> None:
        tool = ScheduleTaskTool()
        result = await asyncio.to_thread(
            tool.execute, tool_name="WEB_SEARCH", args={}, delay_seconds=5
        )
        assert result.startswith("Error:")
        assert "CyraxContext" in result

    async def test_missing_loop_dependency(self, mock_ctx: Mock) -> None:
        tool = ScheduleTaskTool()
        tool._ctx = mock_ctx
        # _loop deliberately NOT set.
        result = await asyncio.to_thread(
            tool.execute, tool_name="WEB_SEARCH", args={}, delay_seconds=5
        )
        assert result.startswith("Error:")
        assert "event loop" in result.lower()

    async def test_malicious_input_admin_tool_hard_rejected(self, mock_ctx: Mock) -> None:
        """
        CRITICAL security regression test: scheduling an ADMIN-level tool
        (SYSTEM_POWER) must be hard-rejected at schedule time, per the
        Phase 3 patch fixing auth-blackholing. Must NOT reach call_soon_threadsafe.
        """
        loop = asyncio.get_running_loop()
        mock_loop = Mock(wraps=loop)  # wrap real loop, spy on call_soon_threadsafe
        mock_loop.is_running.return_value = True

        result = await _run_schedule_execute(
            ScheduleTaskTool(), mock_ctx, mock_loop,
            tool_name="SYSTEM_POWER", args={"action": "shutdown"}, delay_seconds=5,
        )

        assert result.startswith("Error:")
        assert "ADMIN-level tools cannot be scheduled" in result
        mock_loop.call_soon_threadsafe.assert_not_called()
        mock_ctx.tool_registry.execute_tool.assert_not_awaited()

    async def test_unregistered_tool_rejected(self, mock_ctx: Mock) -> None:
        loop = asyncio.get_running_loop()
        result = await _run_schedule_execute(
            ScheduleTaskTool(), mock_ctx, loop,
            tool_name="NONEXISTENT_TOOL", args={}, delay_seconds=5,
        )

        assert result.startswith("Error:")
        assert "not registered" in result.lower()

    async def test_empty_args_dict_accepted(self, mock_ctx: Mock) -> None:
        loop = asyncio.get_running_loop()
        with patch("tools.scheduler.asyncio.sleep", AsyncMock(return_value=None)):
            result = await _run_schedule_execute(
                ScheduleTaskTool(), mock_ctx, loop,
                tool_name="SEND_NOTIFICATION", args={}, delay_seconds=1,
            )
            await asyncio.sleep(0)

        assert result.startswith("Success:")

    async def test_large_args_payload_rejected(self, mock_ctx: Mock) -> None:
        loop = asyncio.get_running_loop()
        oversized_args = {"query": "x" * _MAX_ARGS_STR_LEN}

        result = await _run_schedule_execute(
            ScheduleTaskTool(), mock_ctx, loop,
            tool_name="WEB_SEARCH", args=oversized_args, delay_seconds=5,
        )

        assert result.startswith("Error:")
        assert "too large" in result.lower()
        mock_ctx.tool_registry.execute_tool.assert_not_awaited()

    async def test_non_serialisable_args_rejected(self, mock_ctx: Mock) -> None:
        loop = asyncio.get_running_loop()

        class Unserialisable:
            pass

        result = await _run_schedule_execute(
            ScheduleTaskTool(), mock_ctx, loop,
            tool_name="WEB_SEARCH", args={"bad": Unserialisable()}, delay_seconds=5,
        )

        assert result.startswith("Error:")
        assert "json-serialisable" in result.lower()

    async def test_loop_not_running_rejected(self, mock_ctx: Mock) -> None:
        stopped_loop = Mock()
        stopped_loop.is_running.return_value = False

        tool = ScheduleTaskTool()
        tool._ctx  = mock_ctx
        tool._loop = stopped_loop

        result = await asyncio.to_thread(
            tool.execute, tool_name="WEB_SEARCH", args={}, delay_seconds=5
        )

        assert result.startswith("Error:")
        assert "no longer running" in result.lower()

    async def test_delayed_task_failure_does_not_raise_to_scheduler(
        self, mock_ctx: Mock
    ) -> None:
        """
        If the eventually-executed tool itself fails, _delayed_dispatch
        must log and swallow it — never propagate an unhandled exception
        into the event loop's exception handler.
        """
        loop = asyncio.get_running_loop()
        mock_ctx.tool_registry.execute_tool = AsyncMock(
            side_effect=RuntimeError("downstream tool crashed")
        )

        with patch("tools.scheduler.asyncio.sleep", AsyncMock(return_value=None)):
            result = await _run_schedule_execute(
                ScheduleTaskTool(), mock_ctx, loop,
                tool_name="WEB_SEARCH", args={"query": "x"}, delay_seconds=1,
            )
            # Allow the scheduled coroutine to run and raise internally.
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        # execute() itself still returns success — the failure happens
        # later, inside the background coroutine, and is caught there.
        assert result.startswith("Success:")