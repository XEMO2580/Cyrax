# tests/integration/test_persistence.py

"""
tests/integration/test_persistence.py — Phase 8.5 persistence certification.

Uses a real aiosqlite database on a pytest tmp_path — no mocking of the
DB layer itself, since durability/recovery correctness is exactly what
needs proving against a real SQLite file, not a mock standing in for one.

Uses @pytest_asyncio.fixture explicitly to work around pytest-asyncio 1.4.0
STRICT mode handling of async fixtures declared via @pytest.fixture.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pytest_asyncio import fixture as async_fixture

from core.job_store import SQLiteJobStore
from core.task import Task, TaskStatus, TaskType

pytestmark = pytest.mark.asyncio


@async_fixture
async def job_store(tmp_path: Path) -> SQLiteJobStore:
    store = SQLiteJobStore(db_path=tmp_path / "test_jobs.db")
    await store.init()
    yield store
    await store.close()


# ── Boot Recovery Tests ──────────────────────────────────────────────────


async def test_10_running_jobs_revert_to_pending(job_store: SQLiteJobStore) -> None:
    """
    10 jobs inserted directly as RUNNING (simulating a crash mid-execution).
    recover_pending_jobs() must revert all 10 to PENDING and return them.
    """
    inserted_ids: list[str] = []

    for i in range(10):
        task = Task(
            user_input=f"crashed-job-{i}",
            task_type=TaskType.BACKGROUND,
            status=TaskStatus.RUNNING,
        )
        await job_store.insert_job(task)
        inserted_ids.append(task.task_id)

    recovered = await job_store.recover_pending_jobs()
    recovered_ids = {t.task_id for t in recovered}

    assert recovered_ids == set(inserted_ids)
    for task in recovered:
        assert task.status == TaskStatus.PENDING


async def test_already_pending_jobs_included_in_recovery(job_store: SQLiteJobStore) -> None:
    """Jobs already PENDING (never started) must also be returned, unchanged."""
    task = Task(user_input="never started", task_type=TaskType.BACKGROUND, status=TaskStatus.PENDING)
    await job_store.insert_job(task)

    recovered = await job_store.recover_pending_jobs()

    assert len(recovered) == 1
    assert recovered[0].task_id == task.task_id
    assert recovered[0].status == TaskStatus.PENDING


async def test_completed_jobs_excluded_from_recovery(job_store: SQLiteJobStore) -> None:
    """COMPLETED/FAILED/CANCELLED jobs must never be re-surfaced as recoverable work."""
    for status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
        task = Task(user_input=f"finished-{status.value}", task_type=TaskType.BACKGROUND, status=status)
        await job_store.insert_job(task)

    recovered = await job_store.recover_pending_jobs()
    assert recovered == []


async def test_mixed_running_and_pending_both_recovered(job_store: SQLiteJobStore) -> None:
    running_task = Task(user_input="was running", task_type=TaskType.BACKGROUND, status=TaskStatus.RUNNING)
    pending_task = Task(user_input="was pending", task_type=TaskType.BACKGROUND, status=TaskStatus.PENDING)
    await job_store.insert_job(running_task)
    await job_store.insert_job(pending_task)

    recovered = await job_store.recover_pending_jobs()
    recovered_ids = {t.task_id for t in recovered}

    assert recovered_ids == {running_task.task_id, pending_task.task_id}
    assert all(t.status == TaskStatus.PENDING for t in recovered)


# ── Scheduler Polling Tests ──────────────────────────────────────────────


async def test_job_scheduled_slightly_in_past_is_due(job_store: SQLiteJobStore) -> None:
    """
    A job scheduled for 0.1s in the PAST (simulating a tick that just
    missed the exact moment) must be picked up by get_due_jobs().
    """
    past_time = datetime.now(timezone.utc) - timedelta(seconds=0.1)

    task = Task(
        user_input="due job",
        task_type=TaskType.CRON,
        status=TaskStatus.SCHEDULED,
        scheduled_at=past_time,
    )
    await job_store.insert_job(task)

    due = await job_store.get_due_jobs()

    assert len(due) == 1
    assert due[0].task_id == task.task_id


async def test_future_job_not_yet_due(job_store: SQLiteJobStore) -> None:
    future_time = datetime.now(timezone.utc) + timedelta(hours=1)

    task = Task(
        user_input="not due yet",
        task_type=TaskType.CRON,
        status=TaskStatus.SCHEDULED,
        scheduled_at=future_time,
    )
    await job_store.insert_job(task)

    due = await job_store.get_due_jobs()
    assert due == []


async def test_non_scheduled_status_excluded_from_due_jobs(job_store: SQLiteJobStore) -> None:
    """A PENDING job with a past scheduled_at (edge case) must NOT appear —
    only status == SCHEDULED counts, per the method's own contract."""
    past_time = datetime.now(timezone.utc) - timedelta(seconds=1)

    task = Task(
        user_input="wrong status",
        task_type=TaskType.CRON,
        status=TaskStatus.PENDING,  # not SCHEDULED
        scheduled_at=past_time,
    )
    await job_store.insert_job(task)

    due = await job_store.get_due_jobs()
    assert due == []


async def test_update_status_persists_and_is_queryable(job_store: SQLiteJobStore) -> None:
    task = Task(user_input="update test", task_type=TaskType.BACKGROUND)
    await job_store.insert_job(task)

    await job_store.update_status(task.task_id, TaskStatus.COMPLETED, result="done")

    recovered = await job_store.recover_pending_jobs()
    assert task.task_id not in {t.task_id for t in recovered}  # no longer PENDING

