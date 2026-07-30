# core/notification_center.py

"""
core/notification_center.py — CYRAX 3.0 Phase 8.3 Notification Center

Pub/Sub, transport-agnostic. Zero knowledge of CLI/GUI/Voice — subscribers
(whatever future interface layer registers a callback) decide what to do
with a TaskNotificationEvent. This file never prints, speaks, or renders.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# A listener is any callable — sync or async — accepting one TaskNotificationEvent.
Listener = Callable[["TaskNotificationEvent"], Any]


@dataclass(frozen=True)
class TaskNotificationEvent:
    """
    task_id:    The Task this event concerns.
    status:     String form of the task's TaskStatus at publish time.
    title:      Short human-readable headline (e.g. "Task Completed").
    summary:    One or two sentence summary of the outcome.
    payload:    Full result/error string, or any structured extra data.
    timestamp:  UTC, set at construction.
    """
    task_id:   str
    status:    str
    title:     str
    summary:   str
    payload:   str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class NotificationCenter:
    """
    Async-safe Pub/Sub broker.

    subscribe()/unsubscribe() mutate the listener set under a lock.
    publish() dispatches to a snapshot of listeners taken under the same
    lock, then releases before actually invoking callbacks — so a
    listener that subscribes/unsubscribes from within its own callback
    cannot deadlock against publish()'s own lock acquisition.

    Both sync and async listener callables are supported: async ones are
    awaited, sync ones are called directly. A failing listener is caught
    and logged — one broken subscriber must never break delivery to the
    others or crash the publisher.
    """

    def __init__(self) -> None:
        self._listeners: set[Listener] = set()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def subscribe(self, listener: Listener) -> None:
        async with self._lock:
            self._listeners.add(listener)
        logger.debug(f"[NOTIFICATION_CENTER] Subscribed: {listener!r}")

    async def unsubscribe(self, listener: Listener) -> None:
        async with self._lock:
            self._listeners.discard(listener)
        logger.debug(f"[NOTIFICATION_CENTER] Unsubscribed: {listener!r}")

    async def publish(self, event: TaskNotificationEvent) -> None:
        async with self._lock:
            listeners_snapshot = list(self._listeners)

        logger.info(
            f"[NOTIFICATION_CENTER] Publishing: task_id={event.task_id} | "
            f"status={event.status} | listeners={len(listeners_snapshot)}"
        )

        for listener in listeners_snapshot:
            try:
                result = listener(event)
                if isinstance(result, Awaitable):
                    await result
            except Exception as exc:
                logger.error(
                    f"[NOTIFICATION_CENTER] Listener {listener!r} raised "
                    f"during publish: {exc}",
                    exc_info=True,
                )