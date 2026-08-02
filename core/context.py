"""
core/context.py — CYRAX 3.0 Dependency Injection Container (Phase 8.5)

Phase 8.5 changes:
  - Added JobStoreProtocol — SQLite persistence layer interface.
  - Added JobSchedulerProtocol — tick-based scheduler interface.
  - Added job_store and job_scheduler fields to CyraxContext.
  - job_scheduler is optional (None | JobSchedulerProtocol) so it can
    be injected after context assembly (since JobScheduler needs ctx).
"""

from __future__ import annotations

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
    async def push_recovered_job(self, task: Any) -> None: ...


@runtime_checkable
class NotificationCenterProtocol(Protocol):
    """
    Phase 8.3 — Pub/Sub broker interface.
    Concrete implementation: core/notification_center.py's NotificationCenter.
    """
    async def subscribe(self, listener: Any) -> None: ...
    async def unsubscribe(self, listener: Any) -> None: ...
    async def publish(self, event: Any) -> None: ...


@runtime_checkable
class InterruptControllerProtocol(Protocol):
    """
    Phase 8.4 — central registry for cancellable running tasks.
    Concrete implementation: core/interrupt_controller.py's InterruptController.
    """
    def register(self, task_id: str, asyncio_task: Any, token: Any, interruptible: bool) -> None: ...
    def deregister(self, task_id: str) -> None: ...
    def cancel(self, task_id: str, reason: str) -> bool: ...
    def cancel_all(self, reason: str) -> list[str]: ...


@runtime_checkable
class JobStoreProtocol(Protocol):
    """Phase 8.5 — SQLite persistence layer interface."""
    async def init(self) -> None: ...
    async def close(self) -> None: ...
    async def insert_job(self, task: Any) -> None: ...
    async def update_status(
        self, job_id: str, status: Any, result: str | None = None, error: str | None = None,
    ) -> None: ...
    async def get_due_jobs(self) -> list[Any]: ...
    async def recover_pending_jobs(self) -> list[Any]: ...


@runtime_checkable
class JobSchedulerProtocol(Protocol):
    """Phase 8.5 — tick-based scheduler interface."""
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


@runtime_checkable
class ResourceManagerProtocol(Protocol):
    """Phase 8.6 — bounded concurrency and provider semaphore interface."""
    background_semaphore: Any
    provider_semaphores: dict[str, Any]
    def get_provider_semaphore(self, provider_name: str) -> Any: ...
    async def get_system_snapshot(self) -> dict[str, Any]: ...


@runtime_checkable
class LearningRouterProtocol(Protocol):
    """Phase 7.3 — adaptive decision policy interface."""
    async def select_provider(
        self,
        intent_family:         str,
        recommended_provider:  str,
        trace_id:              str,
        active_providers:      list[str] | None = None,
    ) -> tuple[str, str]: ...


@dataclass(frozen=True)
class CyraxContext:
    """
    Immutable DI container. Constructed once by bootstrap().

    Fields:
        session_id:            Device/user session identifier.
        dispatcher:             Orchestrator entry point.
        tool_registry:          Immutable boot-time tool mount.
        brain_router:           Unified LLM interface.
        memory:                 Four-tier memory stack.
        security:               Authentication and session management.
        fallback_policy:        Declarative tool failure recovery rules.
        decision_engine:        Phase 7 pre-processor.
        task_queue:             Phase 8.2 — in-memory background task queue.
        notification_center:    Phase 8.3 — Pub/Sub broker for task outcomes.
        interrupt_controller:   Phase 8.4 — registry of cancellable running
                                tasks.
        job_store:              Phase 8.5 — SQLite persistence layer.
        job_scheduler:          Phase 8.5 — tick-based scheduler (optional;
                                injected after context assembly via
                                object.__setattr__ since JobScheduler needs ctx).
        resource_manager:       Phase 8.6 — bounded background concurrency
                                (background_semaphore) and per-provider API
                                semaphores (provider_semaphores).
        learning_router:        Phase 7.3 — adaptive decision policy that
                                refines DecisionEngine's provider pick based
                                on persisted routing_history.
    """
    session_id:            str
    dispatcher:            DispatcherProtocol
    tool_registry:          ToolRegistryProtocol
    brain_router:            BrainRouterProtocol
    memory:                  MemoryStackProtocol
    security:                SecurityProtocol
    fallback_policy:         FallbackPolicyProtocol
    decision_engine:         DecisionEngineProtocol
    task_queue:              TaskQueueProtocol
    notification_center:     NotificationCenterProtocol
    interrupt_controller:    InterruptControllerProtocol
    job_store:               JobStoreProtocol
    resource_manager:        ResourceManagerProtocol
    learning_router:         LearningRouterProtocol
    job_scheduler:           JobSchedulerProtocol | None = None

    def __post_init__(self) -> None:
        checks = [
            (self.dispatcher,            DispatcherProtocol,            "dispatcher"),
            (self.tool_registry,         ToolRegistryProtocol,          "tool_registry"),
            (self.brain_router,          BrainRouterProtocol,           "brain_router"),
            (self.memory,                MemoryStackProtocol,           "memory"),
            (self.security,              SecurityProtocol,              "security"),
            (self.fallback_policy,       FallbackPolicyProtocol,        "fallback_policy"),
            (self.decision_engine,       DecisionEngineProtocol,        "decision_engine"),
            (self.task_queue,            TaskQueueProtocol,             "task_queue"),
            (self.notification_center,   NotificationCenterProtocol,    "notification_center"),
            (self.interrupt_controller,  InterruptControllerProtocol,   "interrupt_controller"),
            (self.job_store,             JobStoreProtocol,              "job_store"),
            (self.resource_manager,      ResourceManagerProtocol,       "resource_manager"),
            (self.learning_router,       LearningRouterProtocol,        "learning_router"),
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
