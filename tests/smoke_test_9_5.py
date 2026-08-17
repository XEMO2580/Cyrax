"""
tests/smoke_test_9_5.py — Phase 9.5B/9.5C smoke verification.

Verifies (without requiring live LLM providers):
  1. JWT create/decode/expiry/invalid-signature in security/auth_session.py.
  2. SessionManager create / resume / evict / isolation contract
     (fresh MemoryStack + InterruptController, shared core singletons).
  3. FastAPI login round-trip via TestClient with a mocked base context.

Run: .venv\\Scripts\\python -m pytest tests/smoke_test_9_5.py -v
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# Ensure project root is importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def jwt_settings(monkeypatch):
    """Sets a known JWT secret for the test run."""
    from config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "JWT_SECRET", "test-secret-key")
    monkeypatch.setattr(settings_mod.settings, "JWT_EXPIRY_MINUTES", 60)
    monkeypatch.setattr(settings_mod.settings, "JWT_ALGORITHM", "HS256")
    return settings_mod.settings


def test_jwt_round_trip(jwt_settings):
    """create_token -> decode_token returns the exact claims."""
    from security import auth_session

    token = auth_session.create_token("device-1", "session-1")
    payload = auth_session.decode_token(token)

    assert payload["sub"] == "device-1:session-1"
    assert payload["device_id"] == "device-1"
    assert payload["session_id"] == "session-1"
    assert payload["scope"] == "mobile_client"
    assert payload["iss"] == "cyrax-os"
    assert payload["aud"] == "cyrax-client"
    assert payload["exp"] - payload["iat"] == 60 * 60


def test_jwt_expired(jwt_settings):
    """An expired token raises TokenExpiredError (AUTH_TOKEN_EXPIRED)."""
    from security import auth_session

    token = auth_session.create_token("device-1", "session-1", expires_in_seconds=-10)
    with pytest.raises(auth_session.TokenExpiredError):
        auth_session.decode_token(token)


def test_jwt_invalid_signature(jwt_settings):
    """A token signed with a different secret raises TokenInvalidError."""
    from security import auth_session

    token = auth_session.create_token("device-1", "session-1")
    # Corrupt the signature.
    parts = token.split(".")
    parts[2] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    with pytest.raises(auth_session.TokenInvalidError):
        auth_session.decode_token(".".join(parts))


def test_jwt_missing_secret(monkeypatch):
    """Without a JWT_SECRET, token creation is refused (AUTH_TOKEN_INVALID)."""
    from config import settings as settings_mod
    from security import auth_session

    monkeypatch.setattr(settings_mod.settings, "JWT_SECRET", "")
    with pytest.raises(auth_session.TokenInvalidError):
        auth_session.create_token("device-1", "session-1")


def _make_base_ctx(security=None):
    """Builds a minimal CyraxContext-shaped object for SessionManager tests."""
    from core.context import CyraxContext
    from core.interrupt_controller import InterruptController

    # Local MemoryStack twin (avoids importing app.bootstrap, which has a
    # one-time logger initialisation side effect).
    class MemoryStack:
        def __init__(self, conversation, profile, state):
            self.conversation = conversation
            self.profile = profile
            self.state = state

    memory = MemoryStack(
        conversation=Mock(),
        profile=Mock(),
        state=Mock(),
    )

    base = CyraxContext(
        session_id="device_master_001",
        dispatcher=Mock(),
        tool_registry=Mock(),
        brain_router=Mock(),
        memory=memory,
        security=security or Mock(),
        fallback_policy=Mock(),
        decision_engine=Mock(),
        task_queue=Mock(),
        notification_center=Mock(),
        interrupt_controller=InterruptController(),
        job_store=Mock(),
        resource_manager=Mock(),
        learning_router=Mock(),
    )
    return base


@pytest.mark.asyncio
async def test_session_manager_lifecycle(monkeypatch, tmp_path):
    """create -> count -> evict (idle) -> resume (reconstruct)."""
    from app.sessions.session_manager import SessionManager

    base = _make_base_ctx()
    # Use negative idle/age timeouts so eviction triggers unconditionally.
    mgr = SessionManager(base, idle_timeout=-1.0, max_age=-1.0, sweep_interval=60.0)

    session = await mgr.create_session("device-abc")
    assert session.device_id == "device-abc"
    assert mgr.active_count() == 1

    # Eviction: both idle (0s) and expired (0s) should evict.
    evicted = await mgr.evict_idle_sessions()
    assert session.session_id in evicted
    assert mgr.active_count() == 0

    # Resume: same session_id reconstructs a fresh context.
    resumed = await mgr.resume_session("device-abc", session.session_id)
    assert resumed.session_id == session.session_id
    assert resumed.ctx is not session.ctx  # fresh CyraxContext
    assert mgr.active_count() == 1

    # Destroy.
    assert await mgr.destroy_session(session.session_id) is True
    assert mgr.active_count() == 0


@pytest.mark.asyncio
async def test_session_isolation_contract(monkeypatch, tmp_path):
    """
    Each session gets a FRESH MemoryStack and FRESH InterruptController,
    while sharing the core singletons (tool_registry, brain_router,
    decision_engine, job_store).
    """
    from app.sessions.session_manager import SessionManager

    base = _make_base_ctx()
    mgr = SessionManager(base)

    s1 = await mgr.create_session("device-1")
    s2 = await mgr.create_session("device-2")

    # Shared core singletons.
    assert s1.ctx.tool_registry is base.tool_registry
    assert s2.ctx.tool_registry is base.tool_registry
    assert s1.ctx.brain_router is base.brain_router
    assert s2.ctx.brain_router is base.brain_router
    assert s1.ctx.decision_engine is base.decision_engine
    assert s1.ctx.job_store is base.job_store

    # Fresh per-session memory + interrupt controller.
    assert s1.ctx.memory is not s2.ctx.memory
    assert s1.ctx.memory.conversation is not s2.ctx.memory.conversation
    assert s1.ctx.memory.state is not s2.ctx.memory.state
    assert s1.ctx.interrupt_controller is not s2.ctx.interrupt_controller
    assert s1.ctx.interrupt_controller is not base.interrupt_controller

    # Security shared (device-global lockout).
    assert s1.ctx.security is base.security


def _make_minimal_api_app():
    """
    Builds a minimal FastAPI app with the real auth router + health, WITHOUT
    triggering the full bootstrap() lifespan (which needs live credentials).
    The SessionManager is pre-initialised with a mocked base context instead.
    """
    from fastapi import FastAPI
    from app.api.routes import auth

    app = FastAPI(title="cyrax-test")
    app.include_router(auth.router, prefix="/api/v1")

    @app.get("/api/v1/health")
    async def health() -> dict:
        return {"status": "ok", "version": "3.0"}

    return app


def test_fastapi_login_round_trip(jwt_settings, monkeypatch):
    """POST /api/v1/auth/login returns a JWT + session_id."""
    from fastapi.testclient import TestClient

    from app.sessions.session_manager import (
        initialise_session_manager,
        reset_session_manager,
        get_session_manager,
    )

    reset_session_manager()

    # Mock the SecurityGuard so PIN auth succeeds without bcrypt settings.
    guard = Mock()
    guard.is_locked_out.return_value = (False, 0.0)
    guard.authenticate.return_value = True

    base = _make_base_ctx(security=guard)
    initialise_session_manager(base)

    app = _make_minimal_api_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"device_id": "device-mobile-1", "pin": "147147"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"]
        assert body["token"]
        assert body["expires_in_seconds"] == 60 * 60
        assert body["expires_at"]

        # Decode the returned token to verify binding.
        from security import auth_session

        payload = auth_session.decode_token(body["token"])
        assert payload["device_id"] == "device-mobile-1"
        assert payload["session_id"] == body["session_id"]

        # Health endpoint.
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "version": "3.0"}

    reset_session_manager()


def test_fastapi_login_wrong_pin(jwt_settings, monkeypatch):
    """Bad PIN -> 401 AUTH_INVALID_CREDENTIALS."""
    from fastapi.testclient import TestClient

    from app.sessions.session_manager import (
        initialise_session_manager,
        reset_session_manager,
    )

    reset_session_manager()

    guard = Mock()
    guard.is_locked_out.return_value = (False, 0.0)
    guard.authenticate.return_value = False  # wrong PIN

    base = _make_base_ctx(security=guard)
    initialise_session_manager(base)

    app = _make_minimal_api_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"device_id": "device-mobile-1", "pin": "0000"},
        )
        assert resp.status_code == 401
        assert resp.json()["error_code"] == "AUTH_INVALID_CREDENTIALS"

    reset_session_manager()
