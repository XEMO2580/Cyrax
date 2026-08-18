# tests/stress/test_resource_manager.py

"""
tests/stress/test_resource_manager.py — Phase 8.6 concurrency bound proofs.
"""

from __future__ import annotations

import asyncio

import pytest

from core.resource_manager import Priority, ResourceManager
from core.task import Task, TaskType
from core.task_queue import TaskQueue

pytestmark = pytest.mark.asyncio


class _FakeJobStore:
    async def insert_job(self, task): pass
    async def update_status(self, *a, **kw): pass


class TestBackgroundConcurrencyBound:

    async def test_never_exceeds_4_simultaneous_workers(self) -> None:
        """
        50 concurrent 'jobs' all attempt to acquire background_semaphore
        at once. At every instant, the number of jobs INSIDE the critical
        section must never exceed MAX_BACKGROUND_WORKERS (4).
        """
        rm = ResourceManager()
        NUM_JOBS = 50
        current_concurrent = 0
        max_observed = 0
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal current_concurrent, max_observed
            async with rm.background_semaphore:
                async with lock:
                    current_concurrent += 1
                    max_observed = max(max_observed, current_concurrent)
                await asyncio.sleep(0.02)
                async with lock:
                    current_concurrent -= 1

        await asyncio.gather(*[worker() for _ in range(NUM_JOBS)])

        assert max_observed <= 4
        assert max_observed == 4  # with 50 jobs and a tight cap, should saturate

    async def test_snapshot_reports_running_workers(self) -> None:
        rm = ResourceManager()

        async def worker() -> None:
            async with rm.background_semaphore:
                await rm.mark_worker_started()
                try:
                    await asyncio.sleep(0.1)
                finally:
                    await rm.mark_worker_finished()

        task = asyncio.create_task(worker())
        await asyncio.sleep(0.02)

        snapshot = await rm.get_system_snapshot()
        assert snapshot["running_workers"] >= 1
        assert snapshot["max_workers"] == 4

        await task


class TestPriorityOrdering:

    async def test_immediate_priority_dequeues_ahead_of_50_background(self) -> None:
        """
        50 BACKGROUND-priority tasks submitted first, then one IMMEDIATE
        task submitted last. The IMMEDIATE task must still be the very
        next item dequeued, proving priority — not submission order —
        governs ordering.
        """
        queue = TaskQueue(job_store=_FakeJobStore())

        for i in range(50):
            await queue.submit(f"bg-{i}", TaskType.BACKGROUND, priority=Priority.BACKGROUND.value)

        urgent = await queue.submit("urgent request", TaskType.IMMEDIATE, priority=Priority.IMMEDIATE.value)

        first_dequeued = await queue.get_pending()

        assert first_dequeued is not None
        assert first_dequeued.task_id == urgent.task_id
        assert first_dequeued.priority == Priority.IMMEDIATE.value

    async def test_same_priority_ties_broken_by_created_at_order(self) -> None:
        queue = TaskQueue(job_store=_FakeJobStore())

        first = await queue.submit("first", TaskType.BACKGROUND, priority=Priority.USER.value)
        second = await queue.submit("second", TaskType.BACKGROUND, priority=Priority.USER.value)

        d1 = await queue.get_pending()
        d2 = await queue.get_pending()

        assert d1.task_id == first.task_id
        assert d2.task_id == second.task_id

    async def test_interactive_and_immediate_aliases_and_ordering(self) -> None:
        """
        Verify Priority.INTERACTIVE and Priority.IMMEDIATE share value 0 and both
        dequeue ahead of USER (1), SCHEDULED (2), BACKGROUND (3), and MAINTENANCE (4).
        """
        assert Priority.IMMEDIATE.value == 0
        assert Priority.INTERACTIVE.value == 0
        assert Priority.IMMEDIATE == Priority.INTERACTIVE

        queue = TaskQueue(job_store=_FakeJobStore())
        bg = await queue.submit("bg", TaskType.BACKGROUND, priority=Priority.BACKGROUND.value)
        maint = await queue.submit("maint", TaskType.BACKGROUND, priority=Priority.MAINTENANCE.value)
        sched = await queue.submit("sched", TaskType.BACKGROUND, priority=Priority.SCHEDULED.value)
        user = await queue.submit("user", TaskType.BACKGROUND, priority=Priority.USER.value)
        interactive = await queue.submit("interactive", TaskType.IMMEDIATE, priority=Priority.INTERACTIVE.value)
        immediate = await queue.submit("immediate", TaskType.IMMEDIATE, priority=Priority.IMMEDIATE.value)

        # Expected dequeue order: interactive, immediate (ties broken by FIFO seq/timestamp), user, sched, bg, maint
        t1 = await queue.get_pending()
        t2 = await queue.get_pending()
        t3 = await queue.get_pending()
        t4 = await queue.get_pending()
        t5 = await queue.get_pending()
        t6 = await queue.get_pending()

        assert t1.task_id == interactive.task_id
        assert t2.task_id == immediate.task_id
        assert t3.task_id == user.task_id
        assert t4.task_id == sched.task_id
        assert t5.task_id == bg.task_id
        assert t6.task_id == maint.task_id


class TestProviderSemaphoreLimits:

    async def test_groq_limit_of_2_enforced(self) -> None:
        rm = ResourceManager()
        semaphore = rm.get_provider_semaphore("groq")

        current_concurrent = 0
        max_observed = 0
        lock = asyncio.Lock()

        async def call() -> None:
            nonlocal current_concurrent, max_observed
            async with semaphore:
                async with lock:
                    current_concurrent += 1
                    max_observed = max(max_observed, current_concurrent)
                await asyncio.sleep(0.02)
                async with lock:
                    current_concurrent -= 1

        await asyncio.gather(*[call() for _ in range(10)])

        assert max_observed <= 2
        assert max_observed == 2

    async def test_gemini_limit_of_1_enforced(self) -> None:
        rm = ResourceManager()
        semaphore = rm.get_provider_semaphore("gemini")

        current_concurrent = 0
        max_observed = 0
        lock = asyncio.Lock()

        async def call() -> None:
            nonlocal current_concurrent, max_observed
            async with semaphore:
                async with lock:
                    current_concurrent += 1
                    max_observed = max(max_observed, current_concurrent)
                await asyncio.sleep(0.02)
                async with lock:
                    current_concurrent -= 1

        await asyncio.gather(*[call() for _ in range(10)])

        assert max_observed <= 1
        assert max_observed == 1

    async def test_unlisted_provider_gets_fallback_semaphore(self) -> None:
        rm = ResourceManager()
        semaphore = rm.get_provider_semaphore("ollama")  # not in default map

        assert "ollama" in rm.provider_semaphores
        current_concurrent = 0
        max_observed = 0
        lock = asyncio.Lock()

        async def call() -> None:
            nonlocal current_concurrent, max_observed
            async with semaphore:
                async with lock:
                    current_concurrent += 1
                    max_observed = max(max_observed, current_concurrent)
                await asyncio.sleep(0.02)
                async with lock:
                    current_concurrent -= 1

        await asyncio.gather(*[call() for _ in range(5)])

        assert max_observed <= 1  # fallback limit

    async def test_groq_and_gemini_semaphores_are_independent(self) -> None:
        """A saturated Gemini semaphore must not block Groq calls."""
        rm = ResourceManager()
        gemini_sem = rm.get_provider_semaphore("gemini")
        groq_sem = rm.get_provider_semaphore("groq")

        gemini_started = asyncio.Event()

        async def hold_gemini() -> None:
            async with gemini_sem:
                gemini_started.set()
                await asyncio.sleep(0.3)

        holder = asyncio.create_task(hold_gemini())
        await gemini_started.wait()

        # Groq call must complete quickly despite Gemini being held.
        async def quick_groq_call() -> str:
            async with groq_sem:
                await asyncio.sleep(0.01)
                return "groq done"

        result = await asyncio.wait_for(quick_groq_call(), timeout=1.0)
        assert result == "groq done"

        await holder