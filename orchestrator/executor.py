"""
orchestrator/executor.py — CYRAX 3.0 Phase 8.3 Background Task Executor

_process_task now performs real work: routes through Planner or
ctx.brain_router.chat() based on task.tools_required/task.provider_name
(both set by Dispatcher at submission time — the executor does not
re-classify). Zero UI logic — no print statements, no stdout writes.
Outcomes are delivered exclusively via ctx.notification_center.publish().
"""

from __future__ import annotations

import asyncio
import logging

from core.context import CyraxContext
from core.notification_center import TaskNotificationEvent
from core.task import Task, TaskStatus
from orchestrator.planner import Planner

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS: float = 0.1


class TaskExecutor:
    """
    Background worker that polls ctx.task_queue and processes tasks.
    Contains ZERO UI logic — all outcomes flow through
    ctx.notification_center.publish(), never print()/stdout.
    """

    def __init__(self, ctx: CyraxContext) -> None:
        self._ctx: CyraxContext = ctx
        self._worker_task: asyncio.Task | None = None
        self._running: bool = False

    async def start(self) -> None:
        if self._running:
            logger.warning("[TASK_EXECUTOR] start() called while already running — no-op.")
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop(), name="task-executor-worker")
        logger.info("[TASK_EXECUTOR] Started.")

    async def stop(self) -> None:
        if not self._running:
            logger.debug("[TASK_EXECUTOR] stop() called while not running — no-op.")
            return
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("[TASK_EXECUTOR] Stopped.")

    async def _worker_loop(self) -> None:
        logger.info("[TASK_EXECUTOR] Worker loop started.")
        try:
            while self._running:
                task: Task | None = await self._ctx.task_queue.get_pending()

                if task is None:
                    await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                    continue

                if task.status != TaskStatus.PENDING:
                    logger.info(
                        f"[TASK_EXECUTOR] Discarding non-PENDING task "
                        f"{task.task_id} (status={task.status.value})."
                    )
                    continue

                await self._ctx.task_queue.update_status(task.task_id, TaskStatus.RUNNING)
                await self._process_task(task)

        except asyncio.CancelledError:
            logger.info("[TASK_EXECUTOR] Worker loop cancelled — exiting cleanly.")
            raise

    async def _process_task(self, task: Task) -> None:
        """
        Routes real LLM work based on the task's carried routing hints
        (set by Dispatcher at submission time — no re-classification here):

            task.tools_required True  -> Planner(provider_name=task.provider_name)
            task.tools_required False -> ctx.brain_router.chat(provider_name=...)

        On completion (success or failure), updates task status in the
        queue AND publishes a TaskNotificationEvent. No print statements —
        delivery to any user-facing surface is entirely the subscriber's
        responsibility, not this method's.
        """
        try:
            logger.info(
                f"[TASK_EXECUTOR] Processing task {task.task_id} | "
                f"tools_required={task.tools_required} | "
                f"provider={task.provider_name}"
            )

            history = self._ctx.memory.conversation.get_history()

            if task.tools_required:
                planner = Planner(
                    ctx=self._ctx,
                    req_id=f"task:{task.task_id}",
                    provider_name=task.provider_name,
                )
                result = await planner.run(user_input=task.user_input, history=history)
                response_text = result.get("response", "")
                task_succeeded = result.get("status") == "success"
            else:
                response_text = await self._ctx.brain_router.chat(
                    user_input=task.user_input,
                    history=history,
                    trace_id=f"task:{task.task_id}",
                    provider_name=task.provider_name,
                )
                task_succeeded = True

            if task_succeeded:
                await self._ctx.task_queue.update_status(
                    task.task_id, TaskStatus.COMPLETED, result=response_text,
                )
                await self._ctx.memory.conversation.add_interaction("user", task.user_input)
                await self._ctx.memory.conversation.add_interaction("assistant", response_text)

                await self._ctx.notification_center.publish(TaskNotificationEvent(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED.value,
                    title="Task Completed",
                    summary=response_text[:150],
                    payload=response_text,
                ))
                logger.info(f"[TASK_EXECUTOR] Task {task.task_id} completed.")

            else:
                await self._ctx.task_queue.update_status(
                    task.task_id, TaskStatus.FAILED, error=response_text,
                )
                await self._ctx.notification_center.publish(TaskNotificationEvent(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED.value,
                    title="Task Failed",
                    summary=response_text[:150],
                    payload=response_text,
                ))
                logger.warning(f"[TASK_EXECUTOR] Task {task.task_id} failed: {response_text[:150]}")

        except Exception as exc:
            logger.exception(f"[TASK_EXECUTOR] Task {task.task_id} crashed: {exc}")
            await self._ctx.task_queue.update_status(
                task.task_id, TaskStatus.FAILED, error=str(exc),
            )
            await self._ctx.notification_center.publish(TaskNotificationEvent(
                task_id=task.task_id,
                status=TaskStatus.FAILED.value,
                title="Task Failed",
                summary=f"Unexpected error: {exc}",
                payload=str(exc),
            ))
