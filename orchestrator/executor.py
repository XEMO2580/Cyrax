"""
orchestrator/executor.py — CYRAX 3.0 Background Task Executor (Phase 8.6)

Phase 8.6 change: _process_task acquires ctx.resource_manager
.background_semaphore BEFORE executing work, releases in finally —
bounds Lane B to MAX_BACKGROUND_WORKERS regardless of how many tasks
the PriorityQueue hands out.

Constraint 1 held: the worker loop itself (get_pending, dequeue) is not
gated — only actual EXECUTION is. This matters because dequeuing doesn't
consume a "worker slot," only running the task's LLM/tool work does.
"""

from __future__ import annotations

import asyncio
import logging

from core.context import CyraxContext
from core.interrupt_controller import CancellationToken
from core.notification_center import TaskNotificationEvent
from core.task import Task, TaskStatus
from orchestrator.planner import Planner

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS: float = 0.1


class TaskExecutor:

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

                # Dequeue and status flip are NOT gated by the background
                # semaphore — only actual execution consumes a worker slot.
                # This means many tasks can be RUNNING-pending-acquisition
                # simultaneously; only MAX_BACKGROUND_WORKERS will be
                # actively executing at once.
                asyncio.create_task(
                    self._process_task_bounded(task), name=f"task-bounded-{task.task_id}"
                )

        except asyncio.CancelledError:
            logger.info("[TASK_EXECUTOR] Worker loop cancelled — exiting cleanly.")
            raise

    async def _process_task_bounded(self, task: Task) -> None:
        """
        Acquires the background semaphore before doing any real work,
        releases in finally. Separated from _process_task so the semaphore
        boundary is unambiguous — everything inside this method's `async
        with` block counts toward the MAX_BACKGROUND_WORKERS cap.
        """
        await self._ctx.task_queue.update_status(task.task_id, TaskStatus.RUNNING)

        async with self._ctx.resource_manager.background_semaphore:
            await self._ctx.resource_manager.mark_worker_started()
            try:
                await self._process_task(task)
            finally:
                await self._ctx.resource_manager.mark_worker_finished()

    async def _process_task(self, task: Task) -> None:
        token = CancellationToken()

        inner_task = asyncio.create_task(
            self._execute_work(task), name=f"task-work-{task.task_id}"
        )

        self._ctx.interrupt_controller.register(
            task_id=task.task_id,
            asyncio_task=inner_task,
            token=token,
            interruptible=task.interruptible,
        )

        try:
            response_text, task_succeeded = await inner_task

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

        except asyncio.CancelledError:
            reason = token.reason or "User requested cancellation."
            await self._ctx.task_queue.update_status(
                task.task_id, TaskStatus.CANCELLED, error=reason,
            )
            await self._ctx.notification_center.publish(TaskNotificationEvent(
                task_id=task.task_id,
                status=TaskStatus.CANCELLED.value,
                title="Task Cancelled",
                summary=f"Task cancelled. Reason: {reason}",
                payload=reason,
            ))
            logger.info(f"[TASK_EXECUTOR] Task {task.task_id} cancelled | reason='{reason}'")

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

        finally:
            self._ctx.interrupt_controller.deregister(task.task_id)

    async def _execute_work(self, task: Task) -> tuple[str, bool]:
        history = self._ctx.memory.conversation.get_history()

        if task.tools_required:
            planner = Planner(
                ctx=self._ctx,
                req_id=f"task:{task.task_id}",
                provider_name=task.provider_name,
            )
            result = await planner.run(user_input=task.user_input, history=history)
            return result.get("response", ""), result.get("status") == "success"

        response = await self._ctx.brain_router.chat(
            user_input=task.user_input,
            history=history,
            trace_id=f"task:{task.task_id}",
            provider_name=task.provider_name,
        )
        return response, True

