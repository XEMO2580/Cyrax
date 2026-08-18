# core/resource_manager.py

"""
core/resource_manager.py — CYRAX 3.0 Phase 8.6 Resource Manager

Enforces bounded concurrency (Lane B background workers) and per-provider
API rate protection. Constraint 1: Lane A (interactive CLI/Voice turns
via Dispatcher's synchronous path) never touches background_semaphore —
that gate exists purely for TaskExecutor's background lane.
"""

from __future__ import annotations

import asyncio
import logging
from enum import IntEnum

logger = logging.getLogger(__name__)

MAX_BACKGROUND_WORKERS: int = 4

_DEFAULT_PROVIDER_LIMITS: dict[str, int] = {
    "groq":   2,
    "gemini": 1,
}
_FALLBACK_PROVIDER_LIMIT: int = 1  # for any provider not explicitly listed


class Priority(IntEnum):
    """
    Lower value = higher priority. Used directly as the sort key in
    TaskQueue's PriorityQueue — INTERACTIVE / IMMEDIATE (0) dequeues before
    BACKGROUND (3) without any extra translation.
    """
    INTERACTIVE = 0
    IMMEDIATE   = 0
    USER        = 1
    SCHEDULED   = 2
    BACKGROUND  = 3
    MAINTENANCE = 4


class ResourceManager:
    """
    Concurrency gates, held centrally so TaskExecutor and MoERouter share
    the same semaphore instances rather than each owning private ones.

    background_semaphore:  caps Lane B (background) worker concurrency
                            at MAX_BACKGROUND_WORKERS. Lane A never
                            acquires this — Constraint 1.
    provider_semaphores:    per-provider API concurrency gates, created
                            lazily on first request via
                            get_provider_semaphore() so a provider added
                            later (e.g. "ollama") gets a sane default
                            without a code change here.
    """

    def __init__(self) -> None:
        self.background_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_BACKGROUND_WORKERS)
        self.provider_semaphores: dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(limit)
            for name, limit in _DEFAULT_PROVIDER_LIMITS.items()
        }
        self._running_background_count: int = 0
        self._lock = asyncio.Lock()

    def get_provider_semaphore(self, provider_name: str) -> asyncio.Semaphore:
        """
        Returns the semaphore for provider_name, creating one with the
        fallback limit if this provider wasn't in the default map.
        """
        if provider_name not in self.provider_semaphores:
            self.provider_semaphores[provider_name] = asyncio.Semaphore(_FALLBACK_PROVIDER_LIMIT)
            logger.info(
                f"[RESOURCE_MANAGER] Created semaphore for unlisted provider "
                f"'{provider_name}' with fallback limit {_FALLBACK_PROVIDER_LIMIT}."
            )
        return self.provider_semaphores[provider_name]

    async def mark_worker_started(self) -> None:
        async with self._lock:
            self._running_background_count += 1

    async def mark_worker_finished(self) -> None:
        async with self._lock:
            self._running_background_count = max(0, self._running_background_count - 1)

    async def get_system_snapshot(self) -> dict[str, float | int]:
        """
        Returns current resource metrics. CPU/RAM use psutil if available
        (already an optional dependency elsewhere in this codebase, e.g.
        system_info tooling) — degrades to 0.0 rather than raising if
        psutil is not installed, since this is diagnostic, not load-bearing.
        """
        cpu_percent = 0.0
        ram_percent = 0.0
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=None)
            ram_percent = psutil.virtual_memory().percent
        except ImportError:
            logger.debug("[RESOURCE_MANAGER] psutil not installed — CPU/RAM reported as 0.0.")

        async with self._lock:
            running_workers = self._running_background_count

        return {
            "cpu_percent":        cpu_percent,
            "ram_percent":        ram_percent,
            "running_workers":    running_workers,
            "max_workers":        MAX_BACKGROUND_WORKERS,
            "background_slots_available": self.background_semaphore._value,  # type: ignore[attr-defined]
        }