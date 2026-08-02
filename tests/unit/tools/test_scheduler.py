"""
tests/unit/tools/test_scheduler.py — Certification suite for tools/scheduler.py

Phase 8.5 refactor: execute() is now async and runs directly on the event
loop via ToolRegistry's async-tool path. There is no _loop injection, no
asyncio.run_coroutine_threadsafe, and no tool_name/args in the schema —
the tool classifies the natural-language request through the Decision
Engine BEFORE writing the SCHEDULED job to the job store.

Tests:
  - Classification runs BEFORE the job is saved (ordering contract).
  - tools_required / provider_name are persisted from the decision so the
    background TaskExecutor routes to the Planner (not chat) when needed.
  - Missing ctx returns a graceful Error string, never raises.
  - DB failure returns a graceful Error string, never raises.
  - Schema validation (delay bounds) is unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from tools.scheduler import (
    ScheduleTaskTool, ScheduleTaskSchema,
    _MAX_DELAY_SECONDS,
)

pytestmark = pytest.mark.asyncio


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_ctx() -> Mock:
    """CyraxContext double with async decision_engine and job_store."""
    ctx = Mock(name="CyraxContext")

    ctx.decision_engine = Mock(name="DecisionEngine")
    ctx.decision_engine.classify = AsyncMock(
        return_value=Mock(
            tools_required=True,
            selected_provider="groq",
        )
    )

    ctx.job_store = Mock(name="SQLiteJobStore")
    ctx.job_store.insert_job = AsyncMock()

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# ScheduleTaskTool
# ══════════════════════════════════════════════════════════════════════════════

class TestScheduleTaskTool:

    async def test_happy_path_schedules_job(self, mock_ctx: Mock) -> None:
        """A valid request is classified, saved as SCHEDULED, and acknowledged."""
        tool = ScheduleTaskTool()
        tool._ctx = mock_ctx

        result = await tool.execute(
            user_input="search the web for RTX 5090 price", delay_seconds=10,
        )

        assert result.startswith("Success: Scheduled.")
        assert "Job ID:" in result
        assert "10s" in result

        mock_ctx.decision_engine.classify.assert_awaited_once_with(
            "search the web for RTX 5090 price"
        )
        mock_ctx.job_store.insert_job.assert_awaited_once()

        saved_job = mock_ctx.job_store.insert_job.await_args.args[0]
        assert saved_job.user_input == "search the web for RTX 5090 price"
        assert saved_job.tools_required is True
        assert saved_job.provider_name == "groq"
        assert saved_job.scheduled_at is not None

    async def test_classification_happens_before_save(self, mock_ctx: Mock) -> None:
        """
        Ordering contract: classify() must be fully awaited before
        insert_job() is ever called — the persisted routing metadata
        comes from that classification.
        """
        call_log: list[str] = []

        async def _classify(user_input: str):
            call_log.append("classify")
            return Mock(tools_required=False, selected_provider="groq")

        async def _insert_job(job) -> None:
            call_log.append("insert")
            assert call_log.count("classify") >= 1

        mock_ctx.decision_engine.classify = AsyncMock(side_effect=_classify)
        mock_ctx.job_store.insert_job = AsyncMock(side_effect=_insert_job)

        tool = ScheduleTaskTool()
        tool._ctx = mock_ctx

        await tool.execute(user_input="hello", delay_seconds=5)

        assert call_log == ["classify", "insert"]

    async def test_routing_metadata_persisted_from_decision(
        self, mock_ctx: Mock
    ) -> None:
        """tools_required / provider_name must come from the decision engine."""
        mock_ctx.decision_engine.classify = AsyncMock(
            return_value=Mock(tools_required=False, selected_provider="gemini")
        )

        tool = ScheduleTaskTool()
        tool._ctx = mock_ctx

        await tool.execute(user_input="tell me a joke", delay_seconds=5)

        saved_job = mock_ctx.job_store.insert_job.await_args.args[0]
        assert saved_job.tools_required is False
        assert saved_job.provider_name == "gemini"

    async def test_missing_context_dependency(self) -> None:
        """No ctx injected — graceful Error string, not an AttributeError."""
        tool = ScheduleTaskTool()

        result = await tool.execute(user_input="do something", delay_seconds=5)

        assert result.startswith("Error:")
        assert "CyraxContext" in result

    async def test_db_failure_returns_error_string(self, mock_ctx: Mock) -> None:
        """A job-store failure is caught and surfaced, never raised."""
        mock_ctx.job_store.insert_job = AsyncMock(
            side_effect=RuntimeError("disk full")
        )

        tool = ScheduleTaskTool()
        tool._ctx = mock_ctx

        result = await tool.execute(user_input="do something", delay_seconds=5)

        assert result.startswith("Error:")
        assert "Could not schedule task" in result
        assert "disk full" in result

    def test_invalid_args_delay_too_large(self) -> None:
        with pytest.raises(ValidationError):
            ScheduleTaskSchema(
                user_input="do something", delay_seconds=_MAX_DELAY_SECONDS + 1
            )

    def test_invalid_args_delay_zero(self) -> None:
        with pytest.raises(ValidationError):
            ScheduleTaskSchema(user_input="do something", delay_seconds=0)

    def test_invalid_args_missing_user_input(self) -> None:
        with pytest.raises(ValidationError):
            ScheduleTaskSchema(delay_seconds=5)

