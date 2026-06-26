"""
core/events.py — Typed Event Bus for CYRAX 3.0

Wraps asyncio.Queue with a typed, topic-aware API.
cli_main.py and any future entry point interact only with this interface —
the underlying queue implementation is never exposed to callers.

DESIGN DECISIONS:

1. Shutdown via sentinel, not flag-polling.
   When bus.shutdown() is called, a _ShutdownSentinel is enqueued AFTER
   the is_shutdown flag is set. This guarantees:
   - FIFO order is preserved (in-flight events are processed before shutdown)
   - consume() unblocks naturally without a busy-wait polling loop
   - Workers checking `while not bus.is_shutdown` exit on their next iteration

2. Shutdown is idempotent.
   Multiple callers (voice_worker, text_worker, router_brain) may all call
   shutdown(). Only the first call enqueues the sentinel. Subsequent calls
   are no-ops. Prevents multiple sentinels causing multiple consumer exits.

3. task_done() is the caller's responsibility.
   consume() only yields events. It does NOT call task_done(). The consumer
   (router_brain) calls bus.task_done() after each event is fully processed.
   This preserves the join()/task_done() backpressure contract.

4. One consumer model.
   The current implementation uses a single asyncio.Queue, which supports
   exactly one active consumer on consume(). This matches the current
   architecture (one router_brain task). If Priority 2 (Supervisor Agent)
   requires multiple independent consumers on the same event stream, this
   file's internal implementation upgrades to a topic-based fan-out without
   changing the public API.

THREAD-SAFETY NOTE:
   publish() is safe from async contexts only.
   Do NOT call publish() from a synchronous thread (e.g. a pyaudio callback).
   If that becomes necessary, use:
       loop.call_soon_threadsafe(queue.put_nowait, event)
   This constraint will be enforced when voice/wake_word.py is migrated.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator


# ══════════════════════════════════════════════════════════════════════════════
# EVENT TYPES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IntentEvent:
    """
    A user intent — text or voice — ready for orchestration.

    Attributes:
        payload     The raw user utterance or typed command.
        trace_id    Correlation ID stamped at the app/ ingestion boundary.
    """
    payload:    str
    trace_id:   str


@dataclass(frozen=True)
class SystemEvent:
    """
    An internal control signal (e.g. shutdown).

    Attributes:
        payload     The command name. Currently only "shutdown" is defined.
        trace_id    Correlation ID for log tracing.
    """
    payload:    str
    trace_id:   str


# Union type for all events the bus can carry.
# Extend this union as new event types are added (e.g. ScheduledEvent).
BusEvent = IntentEvent | SystemEvent


# ── Sentinel ──────────────────────────────────────────────────────────────────

class _ShutdownSentinel:
    """
    Private sentinel enqueued by shutdown() to unblock consume().
    Never exposed outside this module.
    """
    __slots__ = ()


_SENTINEL = _ShutdownSentinel()


# ══════════════════════════════════════════════════════════════════════════════
# EVENT BUS
# ══════════════════════════════════════════════════════════════════════════════

class EventBus:
    """
    Typed, topic-aware event bus for CYRAX 3.0.

    Wraps asyncio.Queue. Callers never touch the queue directly.

    Usage:
        bus = EventBus()

        # Producer (text_worker / voice_worker):
        await bus.publish(IntentEvent(payload="open chrome", trace_id=tid))

        # Consumer (router_brain):
        async for event in bus.consume():
            process(event)
            bus.task_done()

        # Shutdown (from any worker or router_brain):
        await bus.shutdown()   # idempotent; safe to call multiple times
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[BusEvent | _ShutdownSentinel] = asyncio.Queue()
        self._is_shutdown: bool = False

    # ── Public read-only state ────────────────────────────────────────────────

    @property
    def is_shutdown(self) -> bool:
        """
        True once shutdown() has been called.
        Workers check this flag at the top of their while loops so they
        do not re-enter the loop after the sentinel has been consumed.
        """
        return self._is_shutdown

    # ── Producer API ─────────────────────────────────────────────────────────

    async def publish(self, event: BusEvent) -> None:
        """
        Enqueue a typed event for consumption.

        Raises:
            RuntimeError    If called after shutdown() has been invoked.
                            Publishing after shutdown is a logic error in the
                            caller — events would never be consumed.
        """
        if self._is_shutdown:
            raise RuntimeError(
                f"EventBus.publish() called after shutdown. "
                f"Event dropped: {event!r}"
            )
        await self._queue.put(event)

    # ── Consumer API ─────────────────────────────────────────────────────────

    async def consume(self) -> AsyncIterator[BusEvent]:
        """
        Async generator yielding events until shutdown.

        The consumer is responsible for calling bus.task_done() after each
        yielded event is fully processed. This file never calls task_done().

        Exits cleanly when the shutdown sentinel is dequeued.
        Does NOT call task_done() for the sentinel — the sentinel is not
        a work item and does not increment the queue's unfinished-task count.
        """
        while True:
            item = await self._queue.get()

            if isinstance(item, _ShutdownSentinel):
                # Sentinel is consumed; do NOT call task_done() for it.
                # The queue's internal unfinished-tasks counter was not
                # incremented for the sentinel (put() increments it, but
                # join() should not wait for the sentinel to be "done").
                # However, asyncio.Queue.put() DOES increment the counter
                # unconditionally — so we must call task_done() here to
                # keep the counter balanced, even though the sentinel is
                # not real work.
                self._queue.task_done()
                break

            yield item

    # ── Shutdown API ─────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """
        Broadcast shutdown to all consumers.

        Idempotent: the first call sets the flag and enqueues the sentinel.
        Subsequent calls are no-ops.

        Order of operations:
            1. Set is_shutdown = True   (workers checking the flag exit loops)
            2. Enqueue sentinel         (unblocks consume() if it is waiting)

        This order ensures that a worker which exits consume() via the sentinel
        will see is_shutdown = True if it checks the flag afterward.
        """
        if self._is_shutdown:
            return  # Already shut down — no-op

        self._is_shutdown = True
        await self._queue.put(_SENTINEL)

    # ── Backpressure API ─────────────────────────────────────────────────────

    async def join(self) -> None:
        """
        Block until all published events have been fully processed
        (i.e. task_done() has been called for each one).

        Used by producers (text_worker, voice_worker) to implement
        backpressure: do not accept new input while the brain is mid-execution.

        Delegates directly to asyncio.Queue.join().
        """
        await self._queue.join()

    def task_done(self) -> None:
        """
        Signal that the most recently consumed event has been fully processed.

        Must be called exactly once per event yielded by consume().
        The consumer (router_brain) calls this explicitly after each event.

        Raises:
            ValueError  If called more times than events have been consumed.
                        (Raised by asyncio.Queue — surfaces the bug immediately.)
        """
        self._queue.task_done()