# """
# core/context.py — Dependency Injection container for CYRAX 3.0

# CyraxContext is the single object passed from bootstrap() into the
# orchestration layer. It carries fully-initialised service instances.

# DESIGN CONSTRAINTS:
# - This file must NOT import concrete implementations from brain/, memory/,
#   tools/, or security/. Those imports create circular dependencies.
# - Fields are typed against abstract protocols defined below, or left as
#   Any with a TODO comment pointing to the protocol that must be written.
# - frozen=True prevents accidental reassignment of context fields mid-request.
#   The services themselves manage their own internal state.

# bootstrap.py is responsible for constructing this object correctly.
# dispatcher.py is the primary consumer.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


# # ── Abstract protocols ────────────────────────────────────────────────────────
# # These define the shape that concrete implementations must satisfy.
# # Written here so core/ has no runtime dependency on any domain layer.
# # Concrete classes in brain/, memory/, tools/ satisfy these implicitly
# # (Python structural subtyping — no explicit 'implements' required).

# @runtime_checkable
# class DispatcherProtocol(Protocol):
#     """
#     The orchestrator entry point.
#     dispatcher.py must expose a method matching this signature.
#     """
#     async def handle(
#         self,
#         user_input: str,
#         session_id: str,
#         trace_id: str,
#     ) -> dict[str, Any]:
#         ...


# @runtime_checkable
# class ToolRegistryProtocol(Protocol):
#     """
#     Minimum interface the context requires from the tool registry.
#     registry.py must expose these methods.
#     """
#     def count(self) -> int: ...
#     def get_all_definitions(self) -> list[dict]: ...


# @runtime_checkable
# class BrainRouterProtocol(Protocol):
#     """
#     Minimum interface the context requires from the brain router.
#     moe_router.py must expose this method.
#     """
#     async def route(
#         self,
#         messages: list[dict],
#         trace_id: str,
#     ) -> str:
#         ...


# @runtime_checkable
# class MemoryStackProtocol(Protocol):
#     """
#     Minimum interface the context requires from the memory layer.
#     The concrete object passed here will satisfy this by having the
#     required attributes on its sub-stores, verified at runtime by
#     bootstrap.py before constructing CyraxContext.
#     """
#     # Sub-stores accessed by dispatcher.py
#     conversation: Any   # ConversationStore
#     profile: Any        # UserProfileStore
#     state: Any          # SessionStateStore
#     # semantic: Any     # SemanticStore — added when Priority 1 lands


# # ── DI Container ─────────────────────────────────────────────────────────────

# @dataclass(frozen=True)
# class CyraxContext:
#     """
#     Immutable container for all fully-initialised CYRAX services.

#     Constructed once by bootstrap() at startup.
#     Passed as a single argument into dispatcher.handle().
#     Never reconstructed mid-request.

#     Fields:
#         session_id      Hardware/device session identifier. On desktop this is
#                         'device_master_001'. On a future API deployment this is
#                         the per-request tenant ID from the HTTP layer.
#         dispatcher      The orchestrator entry point. Handles intent routing,
#                         supervisor logic, and tool execution.
#         tool_registry   The immutable tool mount registry.
#         brain_router    The multi-brain routing layer.
#         memory          The memory stack (conversation, profile, state).
#     """

#     session_id:     str
#     dispatcher:     DispatcherProtocol
#     tool_registry:  ToolRegistryProtocol
#     brain_router:   BrainRouterProtocol
#     memory:         MemoryStackProtocol

#     def __post_init__(self) -> None:
#         """
#         Runtime contract validation.
#         Raises TypeError at boot if any field does not satisfy its protocol.
#         Fails loud at startup rather than silently at first request.
#         """
#         if not isinstance(self.dispatcher, DispatcherProtocol):
#             raise TypeError(
#                 f"dispatcher must satisfy DispatcherProtocol, "
#                 f"got {type(self.dispatcher).__name__}"
#             )
#         if not isinstance(self.tool_registry, ToolRegistryProtocol):
#             raise TypeError(
#                 f"tool_registry must satisfy ToolRegistryProtocol, "
#                 f"got {type(self.tool_registry).__name__}"
#             )
#         if not isinstance(self.brain_router, BrainRouterProtocol):
#             raise TypeError(
#                 f"brain_router must satisfy BrainRouterProtocol, "
#                 f"got {type(self.brain_router).__name__}"
#             )
#         if not isinstance(self.memory, MemoryStackProtocol):
#             raise TypeError(
#                 f"memory must satisfy MemoryStackProtocol, "
#                 f"got {type(self.memory).__name__}"
#             )
#         if not self.session_id or not self.session_id.strip():
#             raise ValueError("session_id cannot be empty.")









"""
core/context.py — CYRAX 3.0 Dependency Injection Container (Revised)

Extends the original CyraxContext with two additional injected dependencies:
  - security:        Handles PIN authentication and session management
  - fallback_policy: Declarative recovery rules (no tool names in dispatcher)

All domain logic is injected. The dispatcher never imports from brain/,
security/, or orchestrator/ directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS
# ══════════════════════════════════════════════════════════════════════════════

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
    """
    Unified brain interface. Exposes planning, chat generation, and raw routing.
    Dispatcher calls only these methods — never imports brain modules directly.
    """
    async def route(
        self,
        messages: list[dict],
        trace_id: str,
    ) -> str: ...

    async def plan(
        self,
        user_input: str,
        history: list[dict],
        tool_definitions: list[dict],
        trace_id: str,
    ) -> list[dict]: ...

    async def chat(
        self,
        user_input: str,
        history: list[dict],
        trace_id: str,
    ) -> str: ...


@runtime_checkable
class SecurityProtocol(Protocol):
    """
    Auth and session management interface.
    Dispatcher calls ctx.security.authenticate() — never imports security.auth.
    """
    def authenticate(self, raw_pin: str) -> bool: ...
    def is_locked_out(self) -> tuple[bool, float]: ...
    def is_session_valid(self) -> bool: ...
    def clear_session(self) -> None: ...


@runtime_checkable
class FallbackPolicyProtocol(Protocol):
    """
    Declarative fallback rules.
    Dispatcher queries this instead of hardcoding OPEN_APP -> WEB_SEARCH logic.

    get_fallback returns None when no fallback is defined for the given
    tool/error combination, signalling the dispatcher to abort the plan.
    """
    def get_fallback(
        self,
        failed_tool: str,
        error_status: str,
        original_args: dict[str, Any],
    ) -> dict[str, Any] | None: ...


@runtime_checkable
class MemoryStackProtocol(Protocol):
    conversation: Any   # ConversationStore
    profile: Any        # UserProfileStore
    state: Any          # SessionStateStore


# ══════════════════════════════════════════════════════════════════════════════
# DI CONTAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CyraxContext:
    """
    Immutable DI container. Constructed once by bootstrap().
    Every service the dispatcher needs is injected here.
    No dynamic imports permitted inside any method that receives this context.

    Fields:
        session_id:      Device/user session identifier.
        dispatcher:      Orchestrator entry point.
        tool_registry:   Immutable boot-time tool mount.
        brain_router:    Unified LLM interface (plan + chat + route).
        memory:          Four-tier memory stack.
        security:        Authentication and session management.
        fallback_policy: Declarative tool failure recovery rules.
    """
    session_id:      str
    dispatcher:      DispatcherProtocol
    tool_registry:   ToolRegistryProtocol
    brain_router:    BrainRouterProtocol
    memory:          MemoryStackProtocol
    security:        SecurityProtocol
    fallback_policy: FallbackPolicyProtocol

    def __post_init__(self) -> None:
        checks = [
            (self.dispatcher,      DispatcherProtocol,      "dispatcher"),
            (self.tool_registry,   ToolRegistryProtocol,    "tool_registry"),
            (self.brain_router,    BrainRouterProtocol,     "brain_router"),
            (self.memory,          MemoryStackProtocol,     "memory"),
            (self.security,        SecurityProtocol,        "security"),
            (self.fallback_policy, FallbackPolicyProtocol,  "fallback_policy"),
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