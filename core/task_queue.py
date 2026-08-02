"""
core/task_queue.py — CYRAX 3.0 In-Memory Task Queue (Phase 8.6)

Phase 8.6 change: asyncio.Queue -> asyncio.PriorityQueue. Items ordered
by (priority, created_at, task_id) — lower priority value first, then
oldest-first for ties, task_id as a final deterministic tiebreaker
(prevents Python from ever attempting to compare Task objects directly,
which would raise since Task doesn't define __lt__).

Phase 8.6 addition (architecture approval): submit() retains
tools_required and provider_name, applied to the Task BEFORE
job_store.insert_job() so the SQLite write-through never persists stale
defaults. dispatcher.py passes these directly into submit().
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.task import Task, TaskStatus, TaskType

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _PriorityItem:
    """
    Sortable wrapper for PriorityQueue. Only priority, created_at, seq,
    and task_id participate in ordering (dataclass order=True generates
    __lt__ over these fields in declaration order).

    priority is the primary key; created_at breaks ties oldest-first.
    seq — a monotonically increasing per-submission counter — guarantees
    strict first-submitted-first-dequeued ordering even when two tasks
    share the same priority AND the same created_at timestamp (which
    happens on coarse-resolution clocks, e.g. Windows). task_id is kept
    as a final deterministic tiebreaker so no two items can ever compare
    equal — PriorityQueue never needs to fall back to comparing Task
    objects directly (Task defines no __lt__).
    """
    priority:    int
    created_at:  datetime
    seq:         int
    task_id:     str


class TaskQueue:
    def __init__(self, job_store: Any) -> None:
        self._job_store    = job_store
        self._tasks:        dict[str, Task] = {}
        self._pending_queue: asyncio.PriorityQueue[_PriorityItem] = asyncio.PriorityQueue()
        self._lock:          asyncio.Lock = asyncio.Lock()
        self._seq_counter:   itertools.count = itertools.count()

    async def submit(
        self,
        user_input:      str,
        task_type:       TaskType,
        tools_required:  bool = False,
        provider_name:   str = "groq",
        priority:        int | None = None,
    ) -> Task:
        """
        Creates a new Task in PENDING state, stores it, enqueues its
        priority-sortable _PriorityItem, and writes through to the
        job_store.

        tools_required/provider_name are applied to the Task BEFORE
        insert_job() so SQLite never persists stale defaults.
        priority defaults to Priority.BACKGROUND (3) unless overridden.
        """
        task = Task(
            user_input=user_input,
            task_type=task_type,
            tools_required=tools_required,
            provider_name=provider_name,
        )
        if priority is not None:
            task.priority = priority

        async with self._lock:
            self._tasks[task.task_id] = task

        await self._job_store.insert_job(task)
        await self._pending_queue.put(
            _PriorityItem(
                priority=task.priority,
                created_at=task.created_at,
                seq=next(self._seq_counter),
                task_id=task.task_id,
            )
        )

        logger.info(
            f"[TASK_QUEUE] Submitted: {task.task_id} | "
            f"type={task_type.value} | priority={task.priority} | "
            f"tools_required={tools_required} | provider={provider_name} | "
            f"input='{user_input[:60]}'"
        )
        return task

    async def get_pending(self) -> Task | None:
        try:
            item = self._pending_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

        async with self._lock:
            task = self._tasks.get(item.task_id)

        if task is None:
            logger.warning(
                f"[TASK_QUEUE] Dequeued task_id '{item.task_id}' has no matching "
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

        await self._job_store.update_status(task_id, status, result=result, error=error)

        logger.info(f"[TASK_QUEUE] Updated: {task_id} | status={status.value}")
        return updated

    async def get_task(self, task_id: str) -> Task | None:
        async with self._lock:
            return self._tasks.get(task_id)

    async def pending_count(self) -> int:
        return self._pending_queue.qsize()

    async def push_recovered_job(self, task: Task) -> None:
        async with self._lock:
            self._tasks[task.task_id] = task

        await self._pending_queue.put(
            _PriorityItem(
                priority=task.priority,
                created_at=task.created_at,
                seq=next(self._seq_counter),
                task_id=task.task_id,
            )
        )
        logger.info(f"[TASK_QUEUE] Recovered/scheduled job pushed to RAM: {task.task_id}")

