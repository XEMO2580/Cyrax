"""
core/context.py — CYRAX 3.0 Dependency Injection Container (Phase 8.3)

Phase 8.3 change: adds NotificationCenterProtocol and notification_center
field, supporting Pub/Sub delivery of background task outcomes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DispatcherProtocol(Protocol):
    async def handle(
        self,
        user_input: str,
        session_id: str,
        trace_id: str,
        ctx: "CyraxContext",
    ) -> dict[str, Any]: ...


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    def count(self) -> int: ...
    def get_all_definitions(self) -> list[dict]: ...
    async def execute_tool(self, tool_name: str, args: dict) -> dict[str, Any]: ...


@runtime_checkable
class BrainRouterProtocol(Protocol):
    async def route(
        self,
        messages: list[dict],
        trace_id: str,
    ) -> str: ...

    async def chat(
        self,
        user_input: str,
        history: list[dict],
        trace_id: str,
        provider_name: str | None = None,
    ) -> str: ...


@runtime_checkable
class SecurityProtocol(Protocol):
    def authenticate(self, raw_pin: str) -> bool: ...
    def is_locked_out(self) -> tuple[bool, float]: ...
    def is_session_valid(self) -> bool: ...
    def clear_session(self) -> None: ...


@runtime_checkable
class FallbackPolicyProtocol(Protocol):
    def get_fallback(
        self,
        failed_tool: str,
        error_status: str,
        original_args: dict[str, Any],
    ) -> dict[str, Any] | None: ...


@runtime_checkable
class MemoryStackProtocol(Protocol):
    conversation: Any
    profile: Any
    state: Any


@runtime_checkable
class DecisionEngineProtocol(Protocol):
    async def classify(self, user_input: str) -> Any: ...


@runtime_checkable
class TaskQueueProtocol(Protocol):
    async def submit(self, user_input: str, task_type: Any) -> Any: ...
    async def get_pending(self) -> Any: ...
    async def update_status(
        self,
        task_id: str,
        status: Any,
        result: str | None = None,
        error: str | None = None,
    ) -> Any: ...
    async def get_task(self, task_id: str) -> Any: ...


@runtime_checkable
class NotificationCenterProtocol(Protocol):
    """
    Phase 8.3 — Pub/Sub broker interface.
    Concrete implementation: core/notification_center.py's NotificationCenter.
    """
    async def subscribe(self, listener: Any) -> None: ...
    async def unsubscribe(self, listener: Any) -> None: ...
    async def publish(self, event: Any) -> None: ...


@dataclass(frozen=True)
class CyraxContext:
    """
    Immutable DI container. Constructed once by bootstrap().

    Fields:
        session_id:           Device/user session identifier.
        dispatcher:            Orchestrator entry point.
        tool_registry:         Immutable boot-time tool mount.
        brain_router:          Unified LLM interface.
        memory:                Four-tier memory stack.
        security:              Authentication and session management.
        fallback_policy:       Declarative tool failure recovery rules.
        decision_engine:       Phase 7 pre-processor.
        task_queue:            Phase 8.2 — in-memory background task queue.
        notification_center:   Phase 8.3 — Pub/Sub broker for task outcomes.
        cancel_token:          Optional asyncio.Event checked by Planner.
    """
    session_id:          str
    dispatcher:          DispatcherProtocol
    tool_registry:        ToolRegistryProtocol
    brain_router:          BrainRouterProtocol
    memory:                MemoryStackProtocol
    security:              SecurityProtocol
    fallback_policy:       FallbackPolicyProtocol
    decision_engine:       DecisionEngineProtocol
    task_queue:            TaskQueueProtocol
    notification_center:   NotificationCenterProtocol
    cancel_token:          asyncio.Event | None = None

    def __post_init__(self) -> None:
        checks = [
            (self.dispatcher,           DispatcherProtocol,          "dispatcher"),
            (self.tool_registry,        ToolRegistryProtocol,        "tool_registry"),
            (self.brain_router,         BrainRouterProtocol,         "brain_router"),
            (self.memory,               MemoryStackProtocol,         "memory"),
            (self.security,             SecurityProtocol,            "security"),
            (self.fallback_policy,      FallbackPolicyProtocol,      "fallback_policy"),
            (self.decision_engine,      DecisionEngineProtocol,      "decision_engine"),
            (self.task_queue,           TaskQueueProtocol,           "task_queue"),
            (self.notification_center,  NotificationCenterProtocol,  "notification_center"),
        ]
        for instance, protocol, name in checks:
            if not isinstance(instance, protocol):
                raise TypeError(
                    f"CyraxContext: '{name}' does not satisfy "
                    f"{protocol.__name__}. "
                    f"Got: {type(instance).__name__}"
                )
        if not self.session_id or not self.session_id.strip():
            raise ValueError("CyraxContext: 'session_id' cannot be empty.")
        if self.cancel_token is not None and not isinstance(self.cancel_token, asyncio.Event):
            raise TypeError(
                f"CyraxContext: 'cancel_token' must be asyncio.Event or None, "
                f"got {type(self.cancel_token).__name__}"
            )
