# core/job_scheduler.py

"""
core/job_scheduler.py — CYRAX 3.0 Phase 8.5 Tick-Based Job Scheduler

Polls the SQLite job_store every 1.0s for due SCHEDULED jobs and pushes
them into the in-memory TaskQueue. No asyncio.sleep-per-job architecture —
scheduling is a single shared polling loop, not one sleeping coroutine
per scheduled task (Constraint 3, mirrored from tools/scheduler.py's fix).
"""

from __future__ import annotations

import asyncio
import logging

from core.context import CyraxContext
from core.task import Task, TaskStatus

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS: float = 1.0


class JobScheduler:
    """
    Background polling loop. Same start()/stop()/inner-loop shape as
    TaskExecutor, for lifecycle consistency across the codebase's
    background-worker components.
    """

    def __init__(self, ctx: CyraxContext) -> None:
        self._ctx: CyraxContext = ctx
        self._worker_task: asyncio.Task | None = None
        self._running: bool = False

    async def start(self) -> None:
        if self._running:
            logger.warning("[JOB_SCHEDULER] start() called while already running — no-op.")
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._polling_loop(), name="job-scheduler-poll")
        logger.info(f"[JOB_SCHEDULER] Started. Poll interval: {_POLL_INTERVAL_SECONDS}s.")

    async def stop(self) -> None:
        if not self._running:
            logger.debug("[JOB_SCHEDULER] stop() called while not running — no-op.")
            return
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("[JOB_SCHEDULER] Stopped.")

    async def _polling_loop(self) -> None:
        logger.info("[JOB_SCHEDULER] Polling loop started.")
        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            logger.info("[JOB_SCHEDULER] Polling loop cancelled — exiting cleanly.")
            raise

    async def _tick(self) -> None:
        due_jobs: list[Task] = await self._ctx.job_store.get_due_jobs()

        if not due_jobs:
            return

        logger.info(f"[JOB_SCHEDULER] {len(due_jobs)} due job(s) found.")

        for job in due_jobs:
            updated_job = job.model_copy(update={"status": TaskStatus.PENDING})
            await self._ctx.job_store.update_status(updated_job.task_id, TaskStatus.PENDING)
            await self._ctx.task_queue.push_recovered_job(updated_job)
            logger.info(f"[JOB_SCHEDULER] Enqueued due job: {updated_job.task_id}")
