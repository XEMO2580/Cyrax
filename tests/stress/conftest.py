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
    ctx.task_queue            = TaskQueue()
    ctx.interrupt_controller  = InterruptController()
    ctx.notification_center   = NotificationCenter()

    return ctx


@pytest.fixture
def task_queue() -> TaskQueue:
    return TaskQueue()


@pytest.fixture
def interrupt_controller() -> InterruptController:
    return InterruptController()


@pytest.fixture
def notification_center() -> NotificationCenter:
    return NotificationCenter()
