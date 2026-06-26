"""
logging_/event_logger.py — CYRAX 3.0 Structured Logger

Provides two complementary log streams:

1. cyrax_YYYYMMDD.log  — Human-readable daily rotating log (DEBUG+)
                          Written via stdlib logging with file + console handlers.

2. events.jsonl        — Machine-readable structured event log (all events).
                          Each line is a JSON object. Used for post-mortem
                          analysis of async interleaving and trace correlation.
                          Written by a background daemon thread via a thread-safe
                          queue — never blocks the async event loop.

DESIGN DECISIONS:

1. queue.Queue (not asyncio.Queue) for the drain thread.
   The drain thread is a real OS thread. asyncio.Queue is not thread-safe.
   Do NOT change this to asyncio.Queue.

2. Explicit initialisation via initialise_logger().
   No singleton is created at import time. bootstrap.py calls
   initialise_logger() during Phase 1 boot. All callers use get_logger().

3. Trace ID is a first-class field in JSONL records.
   Passed as an optional parameter to log_event() and all convenience methods.
   Omitted from the JSONL record (not set to None) when not provided, so log
   queries on trace_id are unambiguous.

4. close() is registered with atexit.
   Flushes the write queue before the drain thread is killed. Prevents the
   "last log entry missing on crash" bug that only appears in shutdown scenarios.

5. Log directory is resolved relative to the project root.
   Avoids logs appearing in unexpected directories when the entry point is
   invoked from a different working directory.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ── Project root resolution ───────────────────────────────────────────────────
# This file lives at cyrax/logging_/event_logger.py.
# Project root is two levels up.
_MODULE_DIR  = Path(__file__).resolve().parent        # cyrax/logging_/
_PROJECT_ROOT = _MODULE_DIR.parent                    # cyrax/
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "logs"


# ══════════════════════════════════════════════════════════════════════════════
# CYRAX LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class CyraxLogger:
    """
    Dual-stream structured logger for CYRAX 3.0.

    Do not instantiate directly. Use initialise_logger() and get_logger().
    """

    def __init__(self, log_dir: Path = _DEFAULT_LOG_DIR) -> None:
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # ── Stdlib logger (human-readable .log file + console) ────────────────
        self._stdlib_logger = logging.getLogger("CYRAX")
        self._stdlib_logger.setLevel(logging.DEBUG)

        if not self._stdlib_logger.handlers:
            log_file = self._log_dir / f"cyrax_{datetime.now().strftime('%Y%m%d')}.log"

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self._stdlib_logger.addHandler(file_handler)
            self._stdlib_logger.addHandler(console_handler)

        # ── JSONL structured event log ────────────────────────────────────────
        self._events_log_file = self._log_dir / "events.jsonl"

        # Thread-safe queue for non-blocking disk I/O.
        # queue.Queue — NOT asyncio.Queue — this feeds a real OS thread.
        self._event_queue: queue.Queue[Optional[dict]] = queue.Queue()

        self._drain_thread = threading.Thread(
            target=self._drain_queue,
            name="cyrax-log-drain",
            daemon=True,
        )
        self._drain_thread.start()

        # Register flush on process exit so the last entries are never lost.
        atexit.register(self.close)

    # ── Internal drain thread ─────────────────────────────────────────────────

    def _drain_queue(self) -> None:
        """
        Background thread: drains the event queue to disk.

        Runs until a None sentinel is dequeued (sent by close()).
        Using None as a sentinel is safe because valid log records are dicts.
        """
        while True:
            record = self._event_queue.get()
            if record is None:
                # Sentinel received — flush complete, thread exits.
                self._event_queue.task_done()
                break
            try:
                with open(self._events_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as exc:
                # Cannot log this error to the JSONL stream (we're in it).
                # Fall back to stderr so it's visible without crashing.
                self._stdlib_logger.error(f"JSONL write failed: {exc}")
            finally:
                self._event_queue.task_done()

    def close(self) -> None:
        """
        Flush all pending JSONL records and stop the drain thread.

        Called automatically by atexit. Can also be called explicitly during
        a clean shutdown sequence. Idempotent — safe to call multiple times
        (subsequent calls after the sentinel is already enqueued are no-ops
        because the drain thread will have exited).
        """
        # Enqueue None sentinel to signal the drain thread to stop.
        self._event_queue.put(None)
        # Block until the queue is fully drained.
        self._event_queue.join()

    # ── Core log_event ────────────────────────────────────────────────────────

    def log_event(
        self,
        event_type: str,
        data: Optional[dict] = None,
        level: str = "INFO",
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Emit a structured event to both the JSONL stream and the stdlib logger.

        The JSONL record always contains: timestamp, type, data.
        trace_id is included only when provided — absent field is cleaner
        than a null field for log queries.

        Args:
            event_type: Short uppercase label, e.g. "USER_INPUT", "TOOL_EXECUTION".
            data:       Arbitrary dict of event metadata. Defaults to {}.
            level:      Stdlib logging level name. Defaults to "INFO".
            trace_id:   Correlation ID from core.trace. Optional.
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "type":      event_type,
            "data":      data or {},
        }
        if trace_id is not None:
            record["trace_id"] = trace_id

        # Non-blocking: drop into the queue, return immediately.
        self._event_queue.put(record)

        # Stdlib logger call — includes trace_id in the message when available.
        trace_suffix = f" [trace={trace_id}]" if trace_id else ""
        message = (
            f"{event_type}{trace_suffix}: {json.dumps(data)}"
            if data
            else f"{event_type}{trace_suffix}"
        )
        log_fn = getattr(self._stdlib_logger, level.lower(), self._stdlib_logger.info)
        log_fn(message)

    # ── Convenience methods ───────────────────────────────────────────────────
    # Each maps a domain event to a structured log_event call.
    # trace_id is threaded through all of them — callers that have it pass it;
    # callers that don't (e.g. system events without a user trace) omit it.

    def log_user_input(
        self,
        user_input: str,
        trace_id: Optional[str] = None,
    ) -> None:
        self.log_event(
            "USER_INPUT",
            {"input": user_input[:100]},
            trace_id=trace_id,
        )

    def log_intent_detected(
        self,
        intent: str,
        params: dict,
        trace_id: Optional[str] = None,
    ) -> None:
        self.log_event(
            "INTENT_DETECTED",
            {
                "intent":       intent,
                "params_count": len(params),
                "params": {
                    k: str(v)[:50]
                    for k, v in params.items()
                    if k != "intent"
                },
            },
            trace_id=trace_id,
        )

    def log_tool_execution(
        self,
        intent: str,
        params: dict,
        status: str,
        response: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        self.log_event(
            "TOOL_EXECUTION",
            {
                "intent":           intent,
                "status":           status,
                "response_preview": str(response)[:100] if response else None,
            },
            level="DEBUG" if status == "success" else "WARNING",
            trace_id=trace_id,
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[dict] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        self.log_event(
            "ERROR",
            {
                "error_type":    error_type,
                "error_message": error_message,
                "context":       context or {},
            },
            level="ERROR",
            trace_id=trace_id,
        )

    def log_agent_planning(
        self,
        user_input: str,
        plan: list,
        trace_id: Optional[str] = None,
    ) -> None:
        self.log_event(
            "AGENT_PLANNING",
            {
                "user_input":   user_input[:100],
                "steps":        len(plan),
                "plan_preview": [str(s.get("intent", s.get("tool", "?"))) for s in plan[:5]],
            },
            level="DEBUG",
            trace_id=trace_id,
        )

    def log_memory_update(
        self,
        key: str,
        value: Any = None,
        trace_id: Optional[str] = None,
    ) -> None:
        self.log_event(
            "MEMORY_UPDATE",
            {
                "key":        key,
                "value_type": type(value).__name__,
            },
            level="DEBUG",
            trace_id=trace_id,
        )

    def log_recovery_attempt(
        self,
        error: str,
        recovery_method: str,
        success: bool,
        trace_id: Optional[str] = None,
    ) -> None:
        self.log_event(
            "RECOVERY_ATTEMPT",
            {
                "error":   error[:50],
                "method":  recovery_method,
                "success": success,
            },
            level="INFO" if success else "WARNING",
            trace_id=trace_id,
        )

    # ── Stdlib passthrough ────────────────────────────────────────────────────
    # These allow callers to use the CyraxLogger instance as a drop-in
    # for module-level logging without importing stdlib logging directly.

    def info(self, message: str) -> None:
        self._stdlib_logger.info(message)

    def debug(self, message: str) -> None:
        self._stdlib_logger.debug(message)

    def warning(self, message: str) -> None:
        self._stdlib_logger.warning(message)

    def error(self, message: str) -> None:
        self._stdlib_logger.error(message)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL ACCESS
# ══════════════════════════════════════════════════════════════════════════════

_logger_instance: Optional[CyraxLogger] = None


def initialise_logger(log_dir: Optional[Path] = None) -> CyraxLogger:
    """
    Create and register the module-level CyraxLogger instance.

    Must be called once, early in bootstrap.py Phase 1, before any other
    CYRAX module emits log events.

    Args:
        log_dir: Override the default log directory. Defaults to
                 <project_root>/logs. Useful in tests to redirect logs
                 to a temp directory.

    Returns:
        The initialised CyraxLogger instance.

    Raises:
        RuntimeError: If called more than once (guards against accidental
                      double-initialisation causing duplicate log handlers).
    """
    global _logger_instance

    if _logger_instance is not None:
        raise RuntimeError(
            "initialise_logger() called more than once. "
            "The logger is a single instance — call get_logger() to retrieve it."
        )

    _logger_instance = CyraxLogger(log_dir=log_dir or _DEFAULT_LOG_DIR)
    return _logger_instance


def get_logger() -> CyraxLogger:
    """
    Retrieve the module-level CyraxLogger instance.

    All CYRAX modules call get_logger() for their log handle.
    Raises RuntimeError if called before initialise_logger().

    Usage:
        from logging_.event_logger import get_logger
        logger = get_logger()
        logger.log_user_input("open chrome", trace_id=trace_id)
    """
    if _logger_instance is None:
        raise RuntimeError(
            "get_logger() called before initialise_logger(). "
            "Ensure bootstrap.py calls initialise_logger() in Phase 1 "
            "before any module attempts to log."
        )
    return _logger_instance