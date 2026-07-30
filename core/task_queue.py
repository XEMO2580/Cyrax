"""
core/task_queue.py — CYRAX 3.0 Phase 8.1 In-Memory Task Queue (Phase 8.3 Patch)

Phase 8.3 change: submit() now accepts tools_required and provider_name
so the Dispatcher can pass the Decision Engine's routing decision directly,
eliminating the hacky post-submit attribute patching.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from core.task import Task, TaskStatus, TaskType

logger = logging.getLogger(__name__)


class TaskQueue:
    """
    In-memory task manager.

    Design:
        self._tasks holds every task ever submitted, keyed by task_id —
        this is the single source of truth for a task's current state.
        self._pending_ids is an asyncio.Queue carrying only task_ids in
        submission order, used purely to hand out the next PENDING task
        without callers needing to scan the full dict.

        A task removed from _pending_ids via get_pending() is NOT removed
        from _tasks — its status transitions to RUNNING via update_status(),
        called by whatever executor picked it up. get_task() remains valid
        for a task at any point in its lifecycle.
    """

    def __init__(self) -> None:
        self._tasks:       dict[str, Task] = {}
        self._pending_ids: asyncio.Queue[str] = asyncio.Queue()
        self._lock:        asyncio.Lock = asyncio.Lock()

    async def submit(self, user_input: str, task_type: TaskType, tools_required: bool = False, provider_name: str = "groq") -> Task:
        """
        Creates a new Task in PENDING state, stores it, and enqueues its
        id for pickup. Returns the created Task so the caller has the
        task_id immediately without a second lookup.

        tools_required and provider_name are carried from the Decision
        Engine's routing decision so the executor can process the task
        without re-classifying it.
        """
        task = Task(user_input=user_input, task_type=task_type, tools_required=tools_required, provider_name=provider_name)

        async with self._lock:
            self._tasks[task.task_id] = task

        await self._pending_ids.put(task.task_id)

        logger.info(
            f"[TASK_QUEUE] Submitted: {task.task_id} | "
            f"type={task_type.value} | "
            f"tools_required={tools_required} | "
            f"provider={provider_name} | "
            f"input='{user_input[:60]}'"
        )
        return task

    async def get_pending(self) -> Task | None:
        """
        Pops the next PENDING task_id from the queue and returns its
        current Task snapshot.

        Non-blocking: returns None immediately if no task is queued,
        rather than awaiting indefinitely — callers that want to block
        until work arrives should await this in their own polling loop,
        not rely on this method itself blocking.

        Note: a task_id dequeued here might theoretically have been
        cancelled between submission and pickup (a future executor could
        call update_status(..., CANCELLED) concurrently). This method
        returns whatever the task's current status actually is — callers
        must check task.status before treating it as runnable, rather
        than assuming PENDING is guaranteed at the point of return.
        """
        try:
            task_id = self._pending_ids.get_nowait()
        except asyncio.QueueEmpty:
            return None

        async with self._lock:
            task = self._tasks.get(task_id)

        if task is None:
            logger.warning(
                f"[TASK_QUEUE] Dequeued task_id '{task_id}' has no matching "
                f"Task record — dropped."
            )
            return None

        return task

    async def update_status(
        self,
        task_id: str,
        status:  TaskStatus,
        result:  str | None = None,
        error:   str | None = None,
    ) -> Task | None:
        """
        Updates a task's status (and optionally result/error), refreshing
        updated_at. Returns the updated Task, or None if task_id is unknown.

        Pydantic models are immutable-by-convention here (no in-place
        mutation) — a new Task instance replaces the old one in the dict
        so any caller holding a prior reference to the old Task object
        sees a stale-but-valid snapshot, never a half-updated one.
        """
        async with self._lock:
            existing = self._tasks.get(task_id)
            if existing is None:
                logger.warning(
                    f"[TASK_QUEUE] update_status called for unknown "
                    f"task_id '{task_id}' — no-op."
                )
                return None

            updated = existing.model_copy(
                update={
                    "status":     status,
                    "result":     result if result is not None else existing.result,
                    "error":      error if error is not None else existing.error,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._tasks[task_id] = updated

        logger.info(
            f"[TASK_QUEUE] Updated: {task_id} | status={status.value}"
        )
        return updated

    async def get_task(self, task_id: str) -> Task | None:
        """Returns the current snapshot of a task, or None if unknown."""
        async with self._lock:
            return self._tasks.get(task_id)

    async def pending_count(self) -> int:
        """Number of task_ids currently queued for pickup (not yet dequeued)."""
        return self._pending_ids.qsize()
