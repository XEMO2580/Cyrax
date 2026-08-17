"""
app/api/main.py — CYRAX 3.0 FastAPI Application (Phase 9.5B/9.5C/9.5D/9.5E)

The FastAPI entry point for the distributed CYRAX API (`/api/v1`).

Delivers:
  - POST /api/v1/auth/login        (Phase 9.5C — JWT authentication)
  - POST /api/v1/chat              (Phase 9.5D — chat pipeline)
  - POST /api/v1/tasks             (Phase 9.5D — explicit background task)
  - GET  /api/v1/tasks/{task_id}   (Phase 9.5D — task status)
  - GET  /api/v1/jobs              (Phase 9.5D — listed jobs)
  - POST /api/v1/tasks/{task_id}/cancel (Phase 9.5D)
  - POST /api/v1/tasks/cancel_all  (Phase 9.5D)
  - GET  /api/v1/memory/profile    (Phase 9.5D)
  - POST /api/v1/memory/profile    (Phase 9.5D)
  - GET  /api/v1/health            (no auth)
  - WS   /api/v1/events            (Phase 9.5E — WebSocket events)

The multi-tenant SessionManager is bootstrapped from the global boot context,
with the background idle-eviction loop started on startup.

The frozen `ErrorResponse` contract (`docs/api/schemas.md` §9) is applied
globally via a custom HTTPException handler: any exception whose `detail` is
a dict containing `error_code`/`message` is serialised verbatim as the
top-level response body (not wrapped in `{"detail": ...}`).

Run (from project root):
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from app.api.routes import auth, chat, jobs, memory
from app.api.websocket import router as websocket_router
from app.bootstrap import bootstrap
from app.sessions.session_manager import (
    initialise_session_manager,
    get_session_manager,
)
from core.job_scheduler import JobScheduler
from orchestrator.executor import TaskExecutor

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: boot the core OS, initialise ALL shared async core
    infrastructure (SQLite job store, JobScheduler, background TaskExecutor),
    build the SessionManager, and start the background session-eviction loop.
    Cleans up every started component in reverse order on shutdown.
    """
    # ── Boot the frozen V1.0 Core OS (synchronous) ───────────────────────────
    base_ctx = bootstrap()
    logger.info("[API] Core OS booted.")

    # ── Initialise the shared async core infrastructure ─────────────────────
    # The API wraps the single global boot context: mobile sessions REUSE its
    # job_store / task_queue / notification_center (see session_manager.py's
    # isolation contract), so these MUST be initialised here exactly like
    # app/cli_main.py does — otherwise background tasks crash with
    # "SQLiteJobStore.init() must be awaited before any other method."
    await base_ctx.job_store.init()
    logger.info("[API] SQLite job store initialised.")

    recovered_jobs = await base_ctx.job_store.recover_pending_jobs()
    for job in recovered_jobs:
        await base_ctx.task_queue.push_recovered_job(job)
    if recovered_jobs:
        logger.info(
            f"[API] Recovered {len(recovered_jobs)} pending job(s) from crash."
        )
    else:
        logger.info("[API] No pending jobs to recover.")

    # ── Inject and start the JobScheduler (tick-based polling loop) ──────────
    # ctx is a frozen dataclass — the CLI does the same object.__setattr__
    # injection because JobScheduler requires ctx at construction time.
    job_scheduler = JobScheduler(base_ctx)
    object.__setattr__(base_ctx, "job_scheduler", job_scheduler)
    await job_scheduler.start()
    logger.info("[API] JobScheduler started.")

    # ── Start the background TaskExecutor (picks up queued tasks) ────────────
    task_executor = TaskExecutor(base_ctx)
    await task_executor.start()
    logger.info("[API] TaskExecutor started.")

    # ── Initialise the process-wide SessionManager ───────────────────────────
    initialise_session_manager(base_ctx)
    session_manager = get_session_manager()

    # ── Start the background idle-eviction loop ──────────────────────────────
    session_manager.start_eviction_loop()
    logger.info("[API] Session eviction loop started.")

    yield

    # ── Shutdown (reverse order) ─────────────────────────────────────────────
    await session_manager.stop_eviction_loop()
    logger.info("[API] Session eviction loop stopped.")

    await task_executor.stop()
    logger.info("[API] TaskExecutor stopped.")

    await job_scheduler.stop()
    logger.info("[API] JobScheduler stopped.")

    await base_ctx.job_store.close()
    logger.info("[API] SQLite job store closed.")


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="CYRAX 3.0 Distributed API",
    version="3.0",
    description=(
        "Distributed API for CYRAX 3.0. JWT-authenticated, session-scoped "
        "control surface wrapping the frozen V1.0 Core OS. Base path /api/v1. "
        "Authentication is JWT-based (Bearer header for REST, `?token=` query "
        "param for WebSocket)."
    ),
    openapi_tags=[
        {
            "name": "auth",
            "description": "Authentication — device login and JWT issuance.",
        },
        {
            "name": "chat",
            "description": "Conversational chat through the Decision Engine pipeline.",
        },
        {
            "name": "jobs",
            "description": "Background task submission, status, listing, and cancellation.",
        },
        {
            "name": "memory",
            "description": "Long-term profile memory read/write.",
        },
        {
            "name": "health",
            "description": "Liveness/readiness probe for infrastructure.",
        },
        {
            "name": "websocket",
            "description": "Real-time event stream (task outcomes, heartbeats).",
        },
    ],
    lifespan=lifespan,
)


# ── Health (no auth, used by infra) ───────────────────────────────────────────

@app.get(
    "/api/v1/health",
    tags=["health"],
    summary="Liveness/readiness probe.",
    description="No auth. Returns service status and version. Used by infra.",
)
async def health() -> dict:
    return {"status": "ok", "version": "3.0"}


# ── Routers ──────────────────────────────────────────────────────────────────

app.include_router(auth.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(memory.router, prefix="/api/v1")
app.include_router(websocket_router)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL EXCEPTION HANDLER (frozen ErrorResponse contract)
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def cyrax_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Catches all HTTPException instances and serialises CYRAX-structured
    detail dicts as the top-level response body per `docs/api/schemas.md` §9.

    Standard FastAPI behaviour wraps `detail` in `{"detail": ...}`. This
    handler detects the CYRAX `error_code` shape and elevates it to the root.
    """
    if isinstance(exc.detail, dict) and "error_code" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers=exc.headers,
        )
    # Fall through to default FastAPI/Starlette behaviour for non-CYRAX errors.
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, (dict, list)) else {"detail": exc.detail},
        headers=exc.headers,
    )
