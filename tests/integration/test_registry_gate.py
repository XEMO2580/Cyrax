# ═══════════════════════════════════════════════════════════════════════════
# tests/integration/test_registry_gate.py
# ═══════════════════════════════════════════════════════════════════════════

"""
tests/integration/test_registry_gate.py — ToolRegistry integration contracts.

Uses dummy BaseTool subclasses, not the real production tools — this suite
certifies the REGISTRY's enforcement logic (auth gate, rate limiting,
schema validation, immutability, ctx/loop injection), independent of any
specific tool's business logic, which is already covered by the unit-tier
suites in tests/tools/.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field

from security.auth import SecurityLevel
from tools.registry import BaseTool, ToolRegistry

pytestmark = pytest.mark.asyncio


# ══════════════════════════════════════════════════════════════════════════════
# Dummy tools
# ══════════════════════════════════════════════════════════════════════════════

class _DummyArgs(BaseModel):
    value: str = Field(..., description="Test arg.")


class _DummyUnrestrictedTool(BaseTool):
    name           = "DUMMY_UNRESTRICTED"
    description    = "Dummy unrestricted tool."
    security_level = SecurityLevel.UNRESTRICTED
    args_schema    = _DummyArgs

    def execute(self, value: str) -> str:
        return f"Success: {value}"


class _DummyAdminTool(BaseTool):
    name           = "DUMMY_ADMIN"
    description    = "Dummy admin tool."
    security_level = SecurityLevel.ADMIN
    args_schema    = _DummyArgs

    def execute(self, value: str) -> str:
        return f"Success: {value}"


class _DummyContextAwareTool(BaseTool):
    """Reads self._ctx / self._loop, mirroring scheduler.py / memory_ops.py."""
    name           = "DUMMY_CONTEXT_AWARE"
    description    = "Dummy tool that requires injected context."
    security_level = SecurityLevel.USER
    args_schema    = _DummyArgs

    def execute(self, value: str) -> str:
        ctx  = getattr(self, "_ctx", None)
        loop = getattr(self, "_loop", None)
        if ctx is None or loop is None:
            return "Error: context or loop not injected."
        return f"Success: ctx={ctx is not None} loop={loop is not None}"


class _DummySlowTool(BaseTool):
    name           = "DUMMY_SLOW"
    description    = "Dummy tool used for rate-limit tests."
    security_level = SecurityLevel.UNRESTRICTED
    args_schema    = _DummyArgs

    def execute(self, value: str) -> str:
        return "Success: done"


class _DummyErrorPrefixTool(BaseTool):
    name           = "DUMMY_ERROR_PREFIX"
    description    = "Dummy tool that returns an Error-prefixed string."
    security_level = SecurityLevel.UNRESTRICTED
    args_schema    = _DummyArgs

    def execute(self, value: str) -> str:
        return "Error: intentional failure for testing."


class _DummyCrashingTool(BaseTool):
    name           = "DUMMY_CRASHING"
    description    = "Dummy tool that raises an unhandled exception."
    security_level = SecurityLevel.UNRESTRICTED
    args_schema    = _DummyArgs

    def execute(self, value: str) -> str:
        raise RuntimeError("unhandled crash for testing")


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def security_allow() -> Mock:
    sec = Mock()
    sec.authorize_action.return_value = True
    return sec


@pytest.fixture
def security_deny() -> Mock:
    sec = Mock()
    # It must return True for UNRESTRICTED, but False for ADMIN/USER
    sec.authorize_action.side_effect = lambda level: level == SecurityLevel.UNRESTRICTED
    return sec


@pytest.fixture
def registry_allow(security_allow: Mock) -> ToolRegistry:
    return ToolRegistry(security=security_allow)


@pytest.fixture
def registry_deny(security_deny: Mock) -> ToolRegistry:
    return ToolRegistry(security=security_deny)


# ══════════════════════════════════════════════════════════════════════════════
# Security authorization
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityAuthorization:

    async def test_unrestricted_tool_runs_regardless_of_security_gate(
        self, security_deny: Mock
    ) -> None:
        registry = ToolRegistry(security=security_deny)
        registry.register(_DummyUnrestrictedTool())
        registry._lock()

        result = await registry.execute_tool("DUMMY_UNRESTRICTED", {"value": "x"})

        assert result["status"] == "success"

    async def test_admin_tool_denied_returns_auth_required(
        self, security_deny: Mock
    ) -> None:
        registry = ToolRegistry(security=security_deny)
        registry.register(_DummyAdminTool())
        registry._lock()

        result = await registry.execute_tool("DUMMY_ADMIN", {"value": "x"})

        assert result["status"] == "auth_required"
        security_deny.authorize_action.assert_called_once_with(SecurityLevel.ADMIN)

    async def test_admin_tool_allowed_runs_when_security_grants(
        self, security_allow: Mock
    ) -> None:
        registry = ToolRegistry(security=security_allow)
        registry.register(_DummyAdminTool())
        registry._lock()

        result = await registry.execute_tool("DUMMY_ADMIN", {"value": "x"})

        assert result["status"] == "success"


# ══════════════════════════════════════════════════════════════════════════════
# Rate limiting
# ══════════════════════════════════════════════════════════════════════════════

class TestRateLimiting:

    async def test_third_rapid_call_rejected(
        self, registry_allow: ToolRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from config import settings as settings_module
        monkeypatch.setattr(settings_module.settings, "MAX_TOOL_CALLS_PER_MINUTE", 2)

        registry_allow.register(_DummySlowTool())
        registry_allow._lock()

        r1 = await registry_allow.execute_tool("DUMMY_SLOW", {"value": "1"})
        r2 = await registry_allow.execute_tool("DUMMY_SLOW", {"value": "2"})
        r3 = await registry_allow.execute_tool("DUMMY_SLOW", {"value": "3"})

        assert r1["status"] == "success"
        assert r2["status"] == "success"
        assert r3["status"] == "error"
        assert "rate limit" in r3["response"].lower()

    async def test_rate_limit_window_resets_after_60_seconds(
        self, registry_allow: ToolRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from config import settings as settings_module
        monkeypatch.setattr(settings_module.settings, "MAX_TOOL_CALLS_PER_MINUTE", 1)

        registry_allow.register(_DummySlowTool())
        registry_allow._lock()

        await registry_allow.execute_tool("DUMMY_SLOW", {"value": "1"})
        blocked = await registry_allow.execute_tool("DUMMY_SLOW", {"value": "2"})
        assert blocked["status"] == "error"

        # Simulate window expiry by rewinding the internal timer.
        registry_allow._rate_limit_window -= 61

        recovered = await registry_allow.execute_tool("DUMMY_SLOW", {"value": "3"})
        assert recovered["status"] == "success"


# ══════════════════════════════════════════════════════════════════════════════
# Schema validation at the registry boundary
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaValidationBoundary:

    async def test_invalid_args_returns_structured_error_not_crash(
        self, registry_allow: ToolRegistry
    ) -> None:
        registry_allow.register(_DummyUnrestrictedTool())
        registry_allow._lock()

        result = await registry_allow.execute_tool("DUMMY_UNRESTRICTED", {"wrong_key": 123})

        assert result["status"] == "error"
        assert "validation" in result["response"].lower() or "invalid" in result["response"].lower()

    async def test_missing_required_field_returns_structured_error(
        self, registry_allow: ToolRegistry
    ) -> None:
        registry_allow.register(_DummyUnrestrictedTool())
        registry_allow._lock()

        result = await registry_allow.execute_tool("DUMMY_UNRESTRICTED", {})

        assert result["status"] == "error"

    async def test_unknown_tool_returns_structured_error(
        self, registry_allow: ToolRegistry
    ) -> None:
        registry_allow._lock()

        result = await registry_allow.execute_tool("NONEXISTENT_TOOL", {"value": "x"})

        assert result["status"] == "error"
        assert "not registered" in result["response"].lower() or "unknown" in result["response"].lower()

    async def test_tool_returning_error_prefixed_string_mapped_to_error_status(
        self, registry_allow: ToolRegistry
    ) -> None:
        registry_allow.register(_DummyErrorPrefixTool())
        registry_allow._lock()

        result = await registry_allow.execute_tool("DUMMY_ERROR_PREFIX", {"value": "x"})

        assert result["status"] == "error"

    async def test_tool_raising_exception_caught_not_propagated(
        self, registry_allow: ToolRegistry
    ) -> None:
        registry_allow.register(_DummyCrashingTool())
        registry_allow._lock()

        result = await registry_allow.execute_tool("DUMMY_CRASHING", {"value": "x"})

        assert result["status"] == "error"
        assert "unexpected" in result["response"].lower() or "crashed" in result["response"].lower()

    async def test_tool_timeout_returns_structured_error(
        self, registry_allow: ToolRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time
        from config import settings as settings_module
        monkeypatch.setattr(settings_module.settings, "TOOL_TIMEOUT_SECONDS", 0.1)

        class _DummyHangingTool(BaseTool):
            name           = "DUMMY_HANGING"
            description    = "Blocks past the timeout."
            security_level = SecurityLevel.UNRESTRICTED
            args_schema    = _DummyArgs

            def execute(self, value: str) -> str:
                time.sleep(1.0)
                return "Success: should not reach here"

        registry_allow.register(_DummyHangingTool())
        registry_allow._lock()

        result = await registry_allow.execute_tool("DUMMY_HANGING", {"value": "x"})

        assert result["status"] == "error"
        assert "timed out" in result["response"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# Immutable registry lock
# ══════════════════════════════════════════════════════════════════════════════

class TestImmutableRegistryLock:

    def test_register_after_lock_raises(self, security_allow: Mock) -> None:
        registry = ToolRegistry(security=security_allow)
        registry.register(_DummyUnrestrictedTool())
        registry._lock()

        with pytest.raises(ToolRegistry.ImmutableRegistryError):
            registry.register(_DummyAdminTool())

    def test_register_before_lock_succeeds(self, security_allow: Mock) -> None:
        registry = ToolRegistry(security=security_allow)
        registry.register(_DummyUnrestrictedTool())
        assert registry.count() == 1

    def test_duplicate_tool_name_rejected(self, security_allow: Mock) -> None:
        registry = ToolRegistry(security=security_allow)
        registry.register(_DummyUnrestrictedTool())

        with pytest.raises(ValueError):
            registry.register(_DummyUnrestrictedTool())

    def test_non_basetool_rejected(self, security_allow: Mock) -> None:
        registry = ToolRegistry(security=security_allow)

        with pytest.raises(TypeError):
            registry.register(object())  # type: ignore[arg-type]

    def test_count_reflects_locked_registry(self, security_allow: Mock) -> None:
        registry = ToolRegistry(security=security_allow)
        registry.register(_DummyUnrestrictedTool())
        registry.register(_DummyAdminTool())
        registry._lock()

        assert registry.count() == 2
        assert len(registry) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Context / loop injection
# ══════════════════════════════════════════════════════════════════════════════

class TestContextLoopInjection:

    async def test_ctx_and_loop_injected_before_execution(
        self, registry_allow: ToolRegistry
    ) -> None:
        registry_allow.register(_DummyContextAwareTool())
        registry_allow._lock()

        mock_ctx = Mock(name="CyraxContext")

        result = await registry_allow.execute_tool(
            "DUMMY_CONTEXT_AWARE", {"value": "x"}, ctx=mock_ctx
        )

        assert result["status"] == "success"
        assert "ctx=True" in result["response"]
        assert "loop=True" in result["response"]

    async def test_tool_without_ctx_kwarg_receives_no_injection(
        self, registry_allow: ToolRegistry
    ) -> None:
        registry_allow.register(_DummyContextAwareTool())
        registry_allow._lock()

        result = await registry_allow.execute_tool("DUMMY_CONTEXT_AWARE", {"value": "x"})

        assert result["status"] == "error"
        assert "not injected" in result["response"].lower()

    async def test_injected_loop_is_the_running_loop(
        self, registry_allow: ToolRegistry
    ) -> None:
        captured_loop = {}

        class _LoopCaptureTool(BaseTool):
            name           = "DUMMY_LOOP_CAPTURE"
            description    = "Captures injected loop for identity check."
            security_level = SecurityLevel.UNRESTRICTED
            args_schema    = _DummyArgs

            def execute(self, value: str) -> str:
                captured_loop["loop"] = getattr(self, "_loop", None)
                return "Success: captured"

        registry_allow.register(_LoopCaptureTool())
        registry_allow._lock()

        mock_ctx = Mock(name="CyraxContext")
        await registry_allow.execute_tool("DUMMY_LOOP_CAPTURE", {"value": "x"}, ctx=mock_ctx)

        assert captured_loop["loop"] is asyncio.get_running_loop()