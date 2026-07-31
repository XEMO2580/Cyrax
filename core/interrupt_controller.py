# core/interrupt_controller.py

"""
core/interrupt_controller.py — CYRAX 3.0 Phase 8.4 Interrupt Controller

Central registry for cancellable running tasks. Exists specifically to
avoid a circular dependency between Dispatcher (needs to cancel running
work) and TaskExecutor (owns the actual asyncio.Task objects) — both
depend on this instead of on each other.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CancellationToken:
    """
    Cooperative cancellation signal. Checked explicitly by long-running
    code (Planner's ReAct loop) rather than relying solely on
    asyncio.Task.cancel(), since a single CancellationToken can be
    threaded through nested awaits that asyncio.Task.cancel() alone
    would not cleanly propagate a human-readable reason into.
    """

    def __init__(self) -> None:
        self._is_cancelled: bool = False
        self._reason: str = ""

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str) -> None:
        self._is_cancelled = True
        self._reason = reason


@dataclass
class RunningTask:
    """
    A live entry in InterruptController's registry.

    task_id:       Matches core.task.Task.task_id.
    asyncio_task:  The actual asyncio.Task executing the work — cancelled
                   via asyncio.Task.cancel() as the hard-stop mechanism.
    token:         CancellationToken checked cooperatively by Planner's
                   loop, so cancellation is observed at a clean boundary
                   rather than only via an abrupt CancelledError injection.
    interruptible: If False, cancel()/cancel_all() ignore this entry
                   entirely — Constraint 3.
    """
    task_id:       str
    asyncio_task:  asyncio.Task
    token:         CancellationToken
    interruptible: bool


class InterruptController:
    """
    Registry of currently-running cancellable tasks.

    Not itself async-context-managed — register/deregister/cancel are
    synchronous dict operations guarded by a lock only where iteration
    order matters (cancel_all), since dict mutation in CPython under
    asyncio (single-threaded event loop) doesn't need a lock for simple
    get/set/del, but cancel_all() iterates while other coroutines could
    concurrently register/deregister, so it takes a snapshot first.
    """

    def __init__(self) -> None:
        self._running_tasks: dict[str, RunningTask] = {}

    def register(
        self,
        task_id:        str,
        asyncio_task:   asyncio.Task,
        token:          CancellationToken,
        interruptible:  bool,
    ) -> None:
        self._running_tasks[task_id] = RunningTask(
            task_id=task_id,
            asyncio_task=asyncio_task,
            token=token,
            interruptible=interruptible,
        )
        logger.debug(
            f"[INTERRUPT_CONTROLLER] Registered: {task_id} "
            f"(interruptible={interruptible})"
        )

    def deregister(self, task_id: str) -> None:
        removed = self._running_tasks.pop(task_id, None)
        if removed is not None:
            logger.debug(f"[INTERRUPT_CONTROLLER] Deregistered: {task_id}")

    def cancel(self, task_id: str, reason: str) -> bool:
        """
        Cancels a specific task if it is registered and interruptible.

        Returns True if a cancellation was actually issued, False if the
        task was not found or was interruptible=False (Constraint 3 —
        non-cancellable tasks silently ignore the request rather than
        raising, since "task not cancellable" is an expected outcome,
        not an error condition).
        """
        entry = self._running_tasks.get(task_id)
        if entry is None:
            logger.debug(f"[INTERRUPT_CONTROLLER] cancel() — unknown task_id '{task_id}'.")
            return False

        if not entry.interruptible:
            logger.info(
                f"[INTERRUPT_CONTROLLER] cancel() — task '{task_id}' is "
                f"non-interruptible, ignoring cancellation request."
            )
            return False

        entry.token.cancel(reason)
        entry.asyncio_task.cancel()
        logger.info(f"[INTERRUPT_CONTROLLER] Cancelled: {task_id} | reason='{reason}'")
        return True

    def cancel_all(self, reason: str) -> list[str]:
        """
        Cancels every currently-registered interruptible task.
        Returns the list of task_ids that were actually cancelled —
        non-interruptible tasks are skipped and excluded from the result.
        """
        snapshot = list(self._running_tasks.values())
        cancelled_ids: list[str] = []

        for entry in snapshot:
            if not entry.interruptible:
                logger.debug(
                    f"[INTERRUPT_CONTROLLER] cancel_all() — skipping "
                    f"non-interruptible task '{entry.task_id}'."
                )
                continue

            entry.token.cancel(reason)
            entry.asyncio_task.cancel()
            cancelled_ids.append(entry.task_id)

        logger.info(
            f"[INTERRUPT_CONTROLLER] cancel_all() | reason='{reason}' | "
            f"cancelled={len(cancelled_ids)} | skipped={len(snapshot) - len(cancelled_ids)}"
        )
        return cancelled_ids