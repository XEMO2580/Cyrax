"""
tests/smoke_test_9_5d.py — Phase 9.5D (REST) + 9.5E (WebSocket) smoke verification.

Verifies (without requiring live LLM providers):
  1. POST /api/v1/chat            — dispatcher bridge, status mapping.
  2. POST /api/v1/tasks           — direct queue submission (202).
  3. GET  /api/v1/tasks/{task_id} — status | 404 TASK_NOT_FOUND.
  4. GET  /api/v1/jobs            — paginated listing.
  5. POST /api/v1/tasks/{task_id}/cancel — 404 unknown / cancel_all.
  6. GET/POST /api/v1/memory/profile     — profile read/write.
  7. Auth-required enforcement on every protected route (401).
  8. WebSocket /api/v1/events?token=<jwt> — connected event; invalid token -> 4401.

Run: .venv\\Scripts\\python -m pytest tests/smoke_test_9_5d.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

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


def _make_security_guard():
    """A SecurityGuard-shaped mock that lets login succeed."""
    guard = Mock()
    guard.is_locked_out.return_value = (False, 0.0)
    guard.authenticate.return_value = True
    return guard


def _make_base_ctx(**overrides):
    """Builds a minimal CyraxContext-shaped object for routed-API tests."""
    from core.context import CyraxContext
    from core.interrupt_controller import InterruptController

    class MemoryStack:
        def __init__(self, conversation, profile, state):
            self.conversation = conversation
            self.profile = profile
            self.state = state

    memory = MemoryStack(
        conversation=Mock(),
        profile=AsyncMock(),
        state=Mock(),
    )

    defaults = dict(
        session_id="device_master_001",
        dispatcher=AsyncMock(),
        tool_registry=Mock(),
        brain_router=AsyncMock(),
        memory=memory,
        security=_make_security_guard(),
        fallback_policy=Mock(),
        decision_engine=AsyncMock(),
        task_queue=AsyncMock(),
        notification_center=AsyncMock(),
        interrupt_controller=InterruptController(),
        job_store=AsyncMock(),
        resource_manager=Mock(),
        learning_router=AsyncMock(),
    )
    defaults.update(overrides)

    return CyraxContext(**defaults)


def _make_api_app(include_routes=(True, True, True), include_ws=True):
    """
    Builds the real routers (auth/chat/jobs/memory + websocket) on a fresh
    FastAPI app, WITHOUT triggering app.bootstrap (logger singleton + live
    credentials). The SessionManager is pre-initialised with a mocked base.
    """
    from fastapi import FastAPI, Request
    from fastapi.exceptions import HTTPException
    from fastapi.responses import JSONResponse
    from app.api.routes import auth, chat, jobs, memory
    from app.api.websocket import router as ws_router

    app = FastAPI(title="cyrax-95d-test")

    @app.exception_handler(HTTPException)
    async def error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Same frozen ErrorResponse unwrapping as app/api/main.py."""
        if isinstance(exc.detail, dict) and "error_code" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail,
                headers=exc.headers,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail if isinstance(exc.detail, (dict, list)) else {"detail": exc.detail},
            headers=exc.headers,
        )

    app.include_router(auth.router, prefix="/api/v1")
    include_chat, include_jobs, include_memory = include_routes
    if include_chat:
        app.include_router(chat.router, prefix="/api/v1")
    if include_jobs:
        app.include_router(jobs.router, prefix="/api/v1")
    if include_memory:
        app.include_router(memory.router, prefix="/api/v1")
    if include_ws:
        app.include_router(ws_router)
    return app


def _make_task(task_id: str | None = None):
    """Creates a real core.task.Task for route responses."""
    from core.task import Task, TaskStatus, TaskType

    return Task(
        task_id=task_id or "task-uuid-0001",
        user_input="open notepad",
        task_type=TaskType.BACKGROUND,
        status=TaskStatus.COMPLETED,
        result="done",
    )


def _setup(base):
    """Initialises a clean SessionManager from a base context."""
    from app.sessions.session_manager import (
        initialise_session_manager,
        reset_session_manager,
    )

    reset_session_manager()
    initialise_session_manager(base)
    return base


def _login(client, device_id: str, pin: str = "147147"):
    """Logs in and returns the JWT."""
    resp = client.post(
        "/api/v1/auth/login",
        json={"device_id": device_id, "pin": pin},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["token"]


# ════════════════════════════════════════════════════════════════════════════
# CHAT
# ════════════════════════════════════════════════════════════════════════════

def test_chat_success(jwt_settings):
    from fastapi.testclient import TestClient

    dispatcher = AsyncMock()
    dispatcher.handle.return_value = {"status": "success", "response": "Hello from Decision Engine."}
    base = _make_base_ctx(dispatcher=dispatcher)
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        token = _login(client, "device-chat-1")
        resp = client.post(
            "/api/v1/chat",
            json={"message": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "success"
        assert body["response"] == "Hello from Decision Engine."
        assert body["trace_id"]


def test_chat_requires_auth(jwt_settings):
    """Protected /chat without a token -> 401 AUTH_TOKEN_INVALID."""
    from fastapi.testclient import TestClient

    _setup(_make_base_ctx())

    app = _make_api_app()
    with TestClient(app) as client:
        resp = client.post("/api/v1/chat", json={"message": "hi"})
        assert resp.status_code == 401
        assert resp.json()["error_code"] == "AUTH_TOKEN_INVALID"


def test_chat_requires_auth_with_bad_token(jwt_settings):
    """Garbage token -> 401 AUTH_TOKEN_INVALID."""
    from fastapi.testclient import TestClient

    _setup(_make_base_ctx())

    app = _make_api_app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat",
            json={"message": "hi"},
            headers={"Authorization": "Bearer not.a.jwt"},
        )
        assert resp.status_code == 401
        assert resp.json()["error_code"] == "AUTH_TOKEN_INVALID"


# ════════════════════════════════════════════════════════════════════════════
# JOBS / TASKS
# ════════════════════════════════════════════════════════════════════════════

def test_tasks_submit(jwt_settings):
    from fastapi.testclient import TestClient

    task = _make_task()
    task_queue = AsyncMock()
    task_queue.submit.return_value = task

    base = _make_base_ctx(task_queue=task_queue)
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        token = _login(client, "device-tasks-1")
        resp = client.post(
            "/api/v1/tasks",
            json={"user_input": "open notepad", "tools_required": True},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["task_id"] == task.task_id
        assert body["status"] == "completed"


def test_tasks_get_status(jwt_settings):
    from fastapi.testclient import TestClient

    task = _make_task()
    task_queue = AsyncMock()
    task_queue.get_task.return_value = task

    base = _make_base_ctx(task_queue=task_queue)
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        token = _login(client, "device-tasks-2")
        resp = client.get(
            f"/api/v1/tasks/{task.task_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["task_id"] == task.task_id
        assert body["status"] == "completed"
        assert body["result"] == "done"


def test_tasks_get_status_404(jwt_settings):
    from fastapi.testclient import TestClient

    task_queue = AsyncMock()
    task_queue.get_task.return_value = None  # unknown/foreign task

    base = _make_base_ctx(task_queue=task_queue)
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        token = _login(client, "device-tasks-3")
        resp = client.get(
            "/api/v1/tasks/does-not-exist",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
        assert resp.json()["error_code"] == "TASK_NOT_FOUND"


def test_jobs_list(jwt_settings):
    from fastapi.testclient import TestClient

    task = _make_task()
    task_queue = AsyncMock()
    task_queue.list_tasks.return_value = ([task], 1)

    base = _make_base_ctx(task_queue=task_queue)
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        token = _login(client, "device-jobs-1")
        resp = client.get(
            "/api/v1/jobs?limit=10&offset=0",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total_count"] == 1
        assert len(body["jobs"]) == 1
        assert body["jobs"][0]["task_id"] == task.task_id
        assert body["limit"] == 10
        assert body["offset"] == 0


def test_tasks_cancel_404(jwt_settings):
    from fastapi.testclient import TestClient

    task_queue = AsyncMock()
    task_queue.get_task.return_value = None

    base = _make_base_ctx(task_queue=task_queue)
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        token = _login(client, "device-cancel-1")
        resp = client.post(
            "/api/v1/tasks/does-not-exist/cancel",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
        assert resp.json()["error_code"] == "TASK_NOT_FOUND"


def test_tasks_cancel_all_empty(jwt_settings):
    from fastapi.testclient import TestClient

    base = _make_base_ctx()
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        token = _login(client, "device-cancel-all-1")
        resp = client.post(
            "/api/v1/tasks/cancel_all",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["count"] == 0
        assert body["cancelled_task_ids"] == []


# ════════════════════════════════════════════════════════════════════════════
# MEMORY PROFILE
# ════════════════════════════════════════════════════════════════════════════

def test_memory_profile_get(jwt_settings):
    from fastapi.testclient import TestClient

    base = _make_base_ctx()
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        token = _login(client, "device-mem-get-1")
        resp = client.get(
            "/api/v1/memory/profile",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        assert isinstance(resp.json()["entries"], dict)


def test_memory_profile_set(jwt_settings):
    from fastapi.testclient import TestClient

    base = _make_base_ctx()
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        token = _login(client, "device-mem-set-1")
        resp = client.post(
            "/api/v1/memory/profile",
            json={"key": "nickname", "value": "cyrax"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"status": "success"}


# ════════════════════════════════════════════════════════════════════════════
# WEBSOCKET (9.5E)
# ════════════════════════════════════════════════════════════════════════════

def test_websocket_connected_event(jwt_settings):
    """Valid token -> 'connected' event with session_id."""
    from fastapi.testclient import TestClient

    base = _make_base_ctx()
    _setup(base)

    from security import auth_session

    app = _make_api_app()
    with TestClient(app) as client:
        token = auth_session.create_token("device-ws-1", "session-ws-1")
        with client.websocket_connect(f"/api/v1/events?token={token}") as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"
            assert data["payload"]["session_id"] == "session-ws-1"
            assert data["timestamp"]


def test_websocket_invalid_token(jwt_settings):
    """Invalid token -> close code 4401."""
    from fastapi.testclient import TestClient

    base = _make_base_ctx()
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/v1/events?token=garbage.token.here"):
                pass
        assert exc_info.value.code == 4401


def test_websocket_missing_token(jwt_settings):
    """No token -> close code 4401."""
    from fastapi.testclient import TestClient

    base = _make_base_ctx()
    _setup(base)

    app = _make_api_app()
    with TestClient(app) as client:
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/v1/events"):
                pass
        assert exc_info.value.code == 4401

