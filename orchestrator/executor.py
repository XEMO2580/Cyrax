"""
orchestrator/executor.py — CYRAX 3.0 Phase 8.4 Background Task Executor

_process_task now wraps actual execution in an inner asyncio.Task,
registers it with ctx.interrupt_controller, and handles CancelledError
explicitly with a finally-block deregistration — Constraint 1.
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

                await self._ctx.task_queue.update_status(task.task_id, TaskStatus.RUNNING)
                await self._process_task(task)

        except asyncio.CancelledError:
            logger.info("[TASK_EXECUTOR] Worker loop cancelled — exiting cleanly.")
            raise

    async def _process_task(self, task: Task) -> None:
        """
        Wraps actual execution in an inner asyncio.Task registered with
        InterruptController, so Dispatcher's "stop"/"cancel"/"abort"
        interceptor (via cancel_all) can reach it without TaskExecutor
        and Dispatcher depending on each other directly.

        try/except CancelledError/finally per Constraint 1: cleanup
        (deregistration) always happens, cancellation is never silently
        swallowed, and the task's terminal status always reflects what
        actually happened.
        """
        token = CancellationToken()

        inner_task = asyncio.create_task(
            self._execute_work(task, token), name=f"task-work-{task.task_id}"
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
            # Do NOT re-raise — the worker loop's own while-condition governs
            # its lifecycle; a single task's cancellation must not propagate
            # and kill the loop that's meant to keep processing other tasks.

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

    async def _execute_work(self, task: Task, token: CancellationToken) -> tuple[str, bool]:
        """
        The actual routing logic, isolated into its own coroutine so it
        can be wrapped in an asyncio.Task for InterruptController to hold
        a cancellable handle on. Returns (response_text, succeeded).
        """
        history = self._ctx.memory.conversation.get_history()

        if task.tools_required:
            planner = Planner(
                ctx=self._ctx,
                req_id=f"task:{task.task_id}",
                provider_name=task.provider_name,
            )
            result = await planner.run(
                user_input=task.user_input,
                history=history,
                cancel_token=token,
            )
            return result.get("response", ""), result.get("status") == "success"

        response = await self._ctx.brain_router.chat(
            user_input=task.user_input,
            history=history,
            trace_id=f"task:{task.task_id}",
            provider_name=task.provider_name,
        )
        return response, True
