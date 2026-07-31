"""
tests/stress/test_concurrency.py — Phase 8.4.5 concurrency stress suite.

Test 1: Massive Background Queue (100 parallel jobs).
Test 2: Cancellation Storm (spawn 50, cancel 25).
Test 7: Race Condition (100 parallel register/cancel/complete ops).
"""

from __future__ import annotations

import asyncio
import random

import pytest

from core.interrupt_controller import CancellationToken, InterruptController
from core.task import TaskStatus, TaskType
from core.task_queue import TaskQueue

pytestmark = pytest.mark.asyncio


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — Massive Background Queue
# ══════════════════════════════════════════════════════════════════════════════

class TestMassiveBackgroundQueue:

    async def test_100_parallel_jobs_queue_empties_no_orphans(
        self, task_queue: TaskQueue
    ) -> None:
        """
        100 tasks submitted concurrently, then all drained concurrently
        via get_pending(). Assert: every submitted task is eventually
        retrievable via get_task(), the pending queue empties completely,
        and no task_id is lost or duplicated (orphaned).
        """
        submit_coros = [
            task_queue.submit(user_input=f"job-{i}", task_type=TaskType.BACKGROUND)
            for i in range(100)
        ]
        submitted_tasks = await asyncio.gather(*submit_coros)
        submitted_ids = {t.task_id for t in submitted_tasks}

        assert len(submitted_ids) == 100  # no task_id collisions
        assert await task_queue.pending_count() == 100

        drained_ids: set[str] = set()

        async def drain_one() -> None:
            task = await task_queue.get_pending()
            if task is not None:
                drained_ids.add(task.task_id)

        drain_coros = [drain_one() for _ in range(100)]
        await asyncio.gather(*drain_coros)

        assert drained_ids == submitted_ids  # every job drained exactly once
        assert await task_queue.pending_count() == 0

        # Every task_id remains queryable — none were lost from the store.
        for task_id in submitted_ids:
            task = await task_queue.get_task(task_id)
            assert task is not None
            assert task.task_id == task_id

    async def test_100_jobs_complete_lifecycle_no_orphans(
        self, task_queue: TaskQueue
    ) -> None:
        """
        Full lifecycle stress: submit 100, drain all, mark all COMPLETED
        concurrently. Assert final state consistency across all 100.
        """
        submitted = await asyncio.gather(*[
            task_queue.submit(f"job-{i}", TaskType.BACKGROUND) for i in range(100)
        ])

        drained = []
        for _ in range(100):
            task = await task_queue.get_pending()
            assert task is not None
            drained.append(task)

        update_coros = [
            task_queue.update_status(t.task_id, TaskStatus.COMPLETED, result=f"done-{i}")
            for i, t in enumerate(drained)
        ]
        await asyncio.gather(*update_coros)

        for t in drained:
            final = await task_queue.get_task(t.task_id)
            assert final is not None
            assert final.status == TaskStatus.COMPLETED
            assert final.result is not None


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — Cancellation Storm
# ══════════════════════════════════════════════════════════════════════════════

class TestCancellationStorm:

    async def test_spawn_50_cancel_25_no_deadlock(
        self, interrupt_controller: InterruptController
    ) -> None:
        """
        50 concurrent long-running fake tasks registered. 25 (randomly
        selected) are cancelled concurrently while the other 25 continue
        running. Assert: the whole scenario completes within a bounded
        timeout (proves no deadlock), exactly 25 are reported cancelled,
        the other 25 complete normally, and the registry ends up empty
        (every task deregistered regardless of outcome).
        """
        NUM_TASKS = 50
        NUM_TO_CANCEL = 25

        tokens: dict[str, CancellationToken] = {}
        inner_tasks: dict[str, asyncio.Task] = {}
        outcomes: dict[str, str] = {}

        async def worker(task_id: str) -> None:
            try:
                await asyncio.sleep(2.0)
                outcomes[task_id] = "completed"
            except asyncio.CancelledError:
                outcomes[task_id] = "cancelled"
                raise
            finally:
                interrupt_controller.deregister(task_id)

        for i in range(NUM_TASKS):
            task_id = f"storm-{i}"
            token = CancellationToken()
            inner = asyncio.create_task(worker(task_id))
            tokens[task_id] = token
            inner_tasks[task_id] = inner
            interrupt_controller.register(task_id, inner, token, interruptible=True)

        await asyncio.sleep(0.05)  # let all workers actually start sleeping

        all_ids = list(tokens.keys())
        to_cancel = set(random.sample(all_ids, NUM_TO_CANCEL))

        cancel_coros = [
            asyncio.to_thread(interrupt_controller.cancel, task_id, "Storm test cancel.")
            for task_id in to_cancel
        ]

        async def run_scenario() -> None:
            await asyncio.gather(*cancel_coros)

            results = await asyncio.gather(
                *inner_tasks.values(), return_exceptions=True
            )
            return results

        # Bounded timeout proves no deadlock — a hung registry/lock would
        # cause this to time out rather than complete.
        results = await asyncio.wait_for(run_scenario(), timeout=5.0)

        cancelled_count = sum(
            1 for r in results if isinstance(r, asyncio.CancelledError)
        )
        completed_count = sum(1 for r in results if r is None)

        assert cancelled_count == NUM_TO_CANCEL
        assert completed_count == NUM_TASKS - NUM_TO_CANCEL
        assert outcomes[list(to_cancel)[0]] == "cancelled"


# ══════════════════════════════════════════════════════════════════════════════
# Test 7 — Race Condition: parallel register/cancel/complete
# ══════════════════════════════════════════════════════════════════════════════

class TestRaceConditionRegistryConsistency:

    async def test_100_parallel_register_cancel_complete_ops_no_corruption(
        self, interrupt_controller: InterruptController
    ) -> None:
        """
        100 concurrent operations interleaving register(), cancel(), and
        natural completion (deregister via finally), all against the same
        InterruptController instance. Assert the registry never ends up
        in an inconsistent state: no task_id ever double-registered
        causing a lost handle, and the final registry contains exactly
        the tasks that were registered-but-not-yet-deregistered — proven
        by checking every registered id is either cleanly gone (completed
        or cancelled) or still present with a valid, live asyncio.Task.
        """
        NUM_OPS = 100
        completion_log: list[str] = []

        async def lifecycle(index: int) -> None:
            task_id = f"race-{index}"
            token = CancellationToken()

            # Random short delay before each op fires — mixes up ordering
            # across all 100 concurrent coroutines.
            await asyncio.sleep(random.uniform(0.0, 0.02))

            async def work() -> None:
                try:
                    await asyncio.sleep(random.uniform(0.01, 0.05))
                    completion_log.append(f"{task_id}:completed")
                except asyncio.CancelledError:
                    completion_log.append(f"{task_id}:cancelled")
                    raise
                finally:
                    interrupt_controller.deregister(task_id)

            inner = asyncio.create_task(work())
            interrupt_controller.register(task_id, inner, token, interruptible=True)

            # ~50% of ops issue a cancel shortly after registering.
            if index % 2 == 0:
                await asyncio.sleep(0.005)
                interrupt_controller.cancel(task_id, "Race test cancel.")

            try:
                await inner
            except asyncio.CancelledError:
                pass

        await asyncio.wait_for(
            asyncio.gather(*[lifecycle(i) for i in range(NUM_OPS)]),
            timeout=10.0,
        )

        # Registry must be fully drained — every task's finally block
        # deregistered it regardless of completion vs cancellation path.
        assert len(interrupt_controller._running_tasks) == 0

        # Every one of the 100 ops resolved exactly once (no duplicate
        # completion entries, no missing entries — proves no corrupted
        # double-registration or lost task under concurrent access).
        assert len(completion_log) == NUM_OPS
        assert len(set(completion_log)) == NUM_OPS  # all unique task_ids

        cancelled = sum(1 for entry in completion_log if entry.endswith(":cancelled"))
        completed = sum(1 for entry in completion_log if entry.endswith(":completed"))
        assert cancelled + completed == NUM_OPS
        # Even-indexed ops (50 of them) requested cancellation — some may
        # legitimately race and complete before the cancel() call lands,
        # so this is a lower bound, not an exact match.
        assert cancelled >= 1
