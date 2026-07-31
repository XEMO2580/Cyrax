# tests/integration/test_cancellation.py

"""
tests/integration/test_cancellation.py — Phase 8.4 cancellation proof.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from core.interrupt_controller import CancellationToken, InterruptController

pytestmark = pytest.mark.asyncio


class TestInterruptibleCancellation:

    async def test_long_running_task_cancelled_and_cleaned_up(self) -> None:
        """
        A mocked long-running unit (standing in for Planner.run()) raises
        CancelledError when cancel_all() fires, and the caller's
        try/except/finally performs cleanup (deregistration) correctly.
        """
        controller = InterruptController()
        token = CancellationToken()

        cleanup_called = False

        async def long_running_work() -> str:
            nonlocal cleanup_called
            try:
                await asyncio.sleep(10.0)  # would hang the test if not cancelled
                return "should never reach here"
            except asyncio.CancelledError:
                raise
            finally:
                cleanup_called = True

        inner_task = asyncio.create_task(long_running_work())
        controller.register(
            task_id="task-1", asyncio_task=inner_task, token=token, interruptible=True,
        )

        await asyncio.sleep(0.05)  # let the task actually start sleeping

        cancelled_ids = controller.cancel_all("Test-triggered stop.")

        assert cancelled_ids == ["task-1"]
        assert token.is_cancelled is True
        assert token.reason == "Test-triggered stop."

        with pytest.raises(asyncio.CancelledError):
            await inner_task

        assert cleanup_called is True

        controller.deregister("task-1")
        assert controller.cancel("task-1", "irrelevant") is False  # already gone

    async def test_non_interruptible_task_ignores_cancel_all_and_completes(self) -> None:
        """
        A task with interruptible=False must run to completion even
        when cancel_all() is invoked — Constraint 3.
        """
        controller = InterruptController()
        token = CancellationToken()

        async def protected_work() -> str:
            await asyncio.sleep(0.1)
            return "completed normally"

        inner_task = asyncio.create_task(protected_work())
        controller.register(
            task_id="task-protected", asyncio_task=inner_task, token=token, interruptible=False,
        )

        cancelled_ids = controller.cancel_all("Attempted stop.")

        assert cancelled_ids == []
        assert token.is_cancelled is False
        assert not inner_task.cancelled()

        result = await inner_task
        assert result == "completed normally"

    async def test_cancel_specific_task_by_id(self) -> None:
        controller = InterruptController()
        token = CancellationToken()

        async def work() -> None:
            await asyncio.sleep(5.0)

        inner_task = asyncio.create_task(work())
        controller.register("task-x", inner_task, token, interruptible=True)

        result = controller.cancel("task-x", "Targeted cancel.")
        assert result is True
        assert token.is_cancelled is True

        with pytest.raises(asyncio.CancelledError):
            await inner_task

    async def test_cancel_unknown_task_id_returns_false(self) -> None:
        controller = InterruptController()
        assert controller.cancel("nonexistent", "reason") is False

    async def test_cancel_all_with_mixed_interruptible_tasks(self) -> None:
        """Only interruptible tasks are cancelled; non-interruptible ones survive untouched."""
        controller = InterruptController()

        token_a = CancellationToken()
        token_b = CancellationToken()

        async def work() -> None:
            await asyncio.sleep(5.0)

        task_a = asyncio.create_task(work())
        task_b = asyncio.create_task(work())

        controller.register("a", task_a, token_a, interruptible=True)
        controller.register("b", task_b, token_b, interruptible=False)

        cancelled = controller.cancel_all("Mixed test.")

        assert cancelled == ["a"]
        assert token_a.is_cancelled is True
        assert token_b.is_cancelled is False

        with pytest.raises(asyncio.CancelledError):
            await task_a

        task_b.cancel()  # manual cleanup since test ends before its 5s sleep completes
        try:
            await task_b
        except asyncio.CancelledError:
            pass