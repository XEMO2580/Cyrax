"""
tests/stress/conftest.py — Phase 8.4.5 stress-test fixtures.

Fast, in-memory mocks — no real network, no real provider SDKs.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from core.interrupt_controller import InterruptController
from core.notification_center import NotificationCenter
from core.resource_manager import ResourceManager
from core.task_queue import TaskQueue


class FakeMoERouter:
    """
    Minimal stand-in for MoERouter. Both chat() and generate() return
    a fixed successful response by default; tests override via
    router.generate = AsyncMock(...) / router.chat = AsyncMock(...)
    for scenarios needing controlled failure/latency.
    """

    def __init__(self) -> None:
        self.chat     = AsyncMock(return_value="Fake chat response.")
        self.generate = AsyncMock(
            return_value=(
                '{"thought": "done", "action": null, '
                '"action_args": {}, "final_answer": "Fake final answer."}'
            )
        )
        self.route = AsyncMock(return_value="Fake route response.")


class FakeToolRegistry:
    def get_all_definitions(self) -> list[dict]:
        return []

    async def execute_tool(self, tool_name: str, args: dict, ctx=None) -> dict:
        return {"status": "success", "response": "Fake tool result."}


class FakeJobStore:
    """
    Phase 8.5 — minimal in-memory stand-in for SQLiteJobStore so the
    TaskQueue(job_store=...) constructor contract is satisfied in tests.

    Phase 7.3 — also exposes the routing_history surface (record_routing_
    outcome / get_routing_stats) so the dispatcher can record outcomes
    against this fake during stress tests without touching disk.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self.routing_history: list[dict] = []

    async def init(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def insert_job(self, task) -> None:
        self._jobs[task.task_id] = task

    async def update_status(self, job_id, status, result=None, error=None) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].status = status

    async def get_due_jobs(self) -> list:
        return []

    async def recover_pending_jobs(self) -> list:
        return []

    async def record_routing_outcome(
        self, intent_family, provider, success, latency_ms,
        fallback_used=False, reason="",
    ) -> None:
        self.routing_history.append(
            {
                "intent_family": intent_family,
                "provider":      provider,
                "success":       success,
                "latency_ms":    latency_ms,
                "fallback_used": fallback_used,
                "reason":        reason,
            }
        )

    async def get_routing_stats(self, intent_family, limit=200) -> list:
        rows = [
            r for r in self.routing_history
            if r["intent_family"] == intent_family
        ]
        return rows[:limit]


class FakeFallbackPolicy:
    def get_fallback(self, failed_tool, error_status, original_args):
        return None


class FakeMemoryStack:
    def __init__(self) -> None:
        self.conversation = Mock()
        self.conversation.get_history = Mock(return_value=[])
        self.conversation.add_interaction = AsyncMock()
        self.profile = Mock()
        self.state = Mock()
        self.state.get = Mock(return_value=None)
        self.state.set = Mock()
        self.state.clear = Mock()


@pytest.fixture
def fake_moe_router() -> FakeMoERouter:
    return FakeMoERouter()


@pytest.fixture
def mock_ctx(fake_moe_router: FakeMoERouter) -> Mock:
    """
    Fast mock CyraxContext with real TaskQueue, InterruptController, and
    NotificationCenter (these are cheap, in-memory, and their own logic
    is exactly what the stress tests need to actually exercise), plus
    faked brain_router/tool_registry/memory (heavy or network-bound in
    production, irrelevant to concurrency-correctness under test).
    """
    ctx = Mock(name="CyraxContext")
    ctx.session_id = "stress-test-session"

    ctx.brain_router          = fake_moe_router
    ctx.tool_registry         = FakeToolRegistry()
    ctx.fallback_policy       = FakeFallbackPolicy()
    ctx.memory                = FakeMemoryStack()
    ctx.job_store             = FakeJobStore()
    ctx.task_queue            = TaskQueue(job_store=ctx.job_store)
    ctx.interrupt_controller  = InterruptController()
    ctx.notification_center   = NotificationCenter()
    ctx.resource_manager      = ResourceManager()
    ctx.learning_router       = AsyncMock()

    return ctx


@pytest.fixture
def task_queue() -> TaskQueue:
    return TaskQueue(job_store=FakeJobStore())


@pytest.fixture
def interrupt_controller() -> InterruptController:
    return InterruptController()


@pytest.fixture
def notification_center() -> NotificationCenter:
    return NotificationCenter()
