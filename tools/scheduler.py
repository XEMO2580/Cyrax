"""
tools/scheduler.py — CYRAX 3.0 Phase 8.5 Job Scheduling Tool

Zero-sleep architecture (Constraint 3). execute() writes a SCHEDULED
job to ctx.job_store with a future scheduled_at timestamp and returns
immediately. The polling JobScheduler (core/job_scheduler.py) picks it
up on its next 1-second tick — no per-call asyncio.sleep(delay_seconds)
holding a thread open.

Phase 8.5 fix:
  - execute() is now async, so it runs directly on the event loop via
    ToolRegistry's async-tool path — no threading bridge, no _loop
    injection, no asyncio.run_coroutine_threadsafe().
  - The request is classified through ctx.decision_engine BEFORE the job
    is saved, so the Task carries the correct tools_required and
    provider_name. When the job fires, TaskExecutor routes it to the
    Planner (not plain chat) if tools are needed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from security.auth import SecurityLevel
from tools.registry import BaseTool

logger = logging.getLogger(__name__)

_MAX_DELAY_SECONDS: int = 3600


class ScheduleTaskSchema(BaseModel):
    user_input: str = Field(
        ...,
        description="The natural-language request to execute later (e.g. 'search the web for X').",
    )
    delay_seconds: int = Field(
        ...,
        ge=1,
        le=_MAX_DELAY_SECONDS,
        description=f"Seconds from now to run this. Max {_MAX_DELAY_SECONDS}.",
    )


class ScheduleTaskTool(BaseTool):
    """
    Schedules a future request via the persistent job store.

    Zero sleeping threads: this tool performs one classification + one
    DB write and returns. The JobScheduler's polling loop is the only
    thing that ever waits.

    Requires self._ctx (injected by ToolRegistry.execute_tool(ctx=ctx)),
    same pattern as the Phase-3/Phase-8 context-aware tools. execute()
    is async — the registry detects coroutine functions and awaits them
    directly on the event loop.
    """

    name           = "SCHEDULE_TASK"
    description    = (
        "Schedules a natural-language request to run automatically after "
        "a delay. Returns immediately with a Job ID."
    )
    security_level = SecurityLevel.USER
    args_schema    = ScheduleTaskSchema

    async def execute(self, user_input: str, delay_seconds: int) -> str:  # type: ignore[override]
        ctx = getattr(self, "_ctx", None)
        if ctx is None:
            return (
                "Error: ScheduleTaskTool requires a CyraxContext but none "
                "was injected. Ensure ToolRegistry.execute_tool() passes ctx."
            )

        # Classify the input so the background task knows if it needs tools
        decision = await ctx.decision_engine.classify(user_input)

        future_time = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)

        from core.task import Task, TaskStatus, TaskType
        job = Task(
            user_input=user_input,
            task_type=TaskType.CRON,
            status=TaskStatus.SCHEDULED,
            scheduled_at=future_time,
            tools_required=decision.tools_required,
            provider_name=decision.selected_provider
        )

        try:
            await ctx.job_store.insert_job(job)
        except Exception as exc:
            logger.error(f"[SCHEDULE_TASK] Failed to insert job: {exc}")
            return f"Error: Could not schedule task — {exc}"

        minutes, seconds = divmod(delay_seconds, 60)
        time_desc = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        logger.info(
            f"[SCHEDULE_TASK] Job {job.task_id} scheduled for {future_time.isoformat()} "
            f"(in {time_desc})."
        )

        return (
            f"Success: Scheduled. Job ID: {job.task_id}. "
            f"Will run in {time_desc} (at {future_time.strftime('%H:%M:%S UTC')})."
        )

