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

Phase 9.5D addition: list_tasks() — delegates to job_store.get_jobs()
for the GET /api/v1/jobs endpoint.
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

_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset({
    TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED,
})

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
        task_type:        TaskType,
        priority:          int | None = None,
        tools_required:    bool       = False,
        provider_name:     str        = "groq",
        conversation_id:   str | None = None,
        device_id:         str | None = None,
    ) -> Task:
        """
        FIX: all ownership/routing fields are now set on the Task object
        BEFORE insert_job() persists it — previously chat.py mutated these
        fields on the returned object AFTER submit() had already written
        an incomplete row to SQLite, meaning conversation_id/tools_required/
        provider_name were silently never persisted, ever.
        """
        task = Task(
            user_input=user_input,
            task_type=task_type,
            tools_required=tools_required,
            provider_name=provider_name,
            conversation_id=conversation_id,
            device_id=device_id,
        )
        if priority is not None:
            task.priority = priority

        async with self._lock:
            self._tasks[task.task_id] = task

        await self._job_store.insert_job(task)
        await self._pending_queue.put(
            _PriorityItem(priority=task.priority, seq=next(self._seq_counter), created_at=task.created_at, task_id=task.task_id)
        )

        logger.info(
            f"[TASK_QUEUE] Submitted: {task.task_id} | type={task_type.value} | "
            f"priority={task.priority} | conversation_id={conversation_id} | "
            f"device_id={device_id} | input='{user_input[:60]}'"
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
                logger.warning(f"[TASK_QUEUE] update_status for unknown task_id '{task_id}' — no-op.")
                return None

            # Requirement 20 — race-safety: a terminal task's status is
            # immutable. This is the single choke point that prevents
            # COMPLETED->CANCELLED or CANCELLED->COMPLETED regardless of
            # which write arrives first or last.
            if existing.status in _TERMINAL_STATUSES:
                logger.warning(
                    f"[TASK_QUEUE] REJECTED status update for {task_id}: "
                    f"already terminal ({existing.status.value}), "
                    f"attempted write was {status.value}. Race-safety guard."
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
            if task_id in self._tasks:
                return self._tasks[task_id]

        # Fallback to the database if it was evicted from RAM
        return await self._job_store.get_job(task_id)

    async def list_tasks(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> tuple[list[Task], int]:
        """
        Returns a page of the caller's tasks (most recent first) and the
        total count, backed by the SQLite job store.

        This is the persistence-backed source for `GET /api/v1/jobs`
        (`docs/api/rest.md` §3 GET /jobs). It intentionally delegates to
        `self._job_store.get_jobs(...)` rather than returning the in-memory
        `self._tasks` dict, so historical/terminated jobs that survive a
        process restart are still visible to the API.

        Args:
            limit:  Max rows to return (default 20).
            offset: Row offset for pagination (default 0).
            status: Optional TaskStatus filter. If provided, only jobs in
                    that state are returned.

        Returns:
            A (jobs, total_count) tuple.
        """
        return await self._job_store.get_jobs(
            limit=limit,
            offset=offset,
            status=status,
        )

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