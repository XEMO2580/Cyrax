"""
app/api/schemas.py — CYRAX 3.0 API Pydantic Models (Phase 9.5)

Implements the frozen request/response schemas from `docs/api/schemas.md`.
These are the Pydantic-equivalent JSON schemas the FastAPI layer (Phase 9.5D)
and the Kotlin client build against.

This module now covers the full Phase 9.5D REST surface:
  - §1. LoginRequest / LoginResponse
  - §2. ChatRequest / ChatResponse
  - §3. TaskSubmitRequest / TaskSubmitResponse
  - §4. TaskStatusResponse
  - §5. JobListResponse
  - §6. CancelResponse
  - §7. CancelAllResponse
  - §8. ProfileGetResponse / ProfileSetRequest
  - §9. ErrorResponse
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ── §1. LoginRequest / LoginResponse ─────────────────────────────────────────


class LoginRequest(BaseModel):
    """POST /api/v1/auth/login request body (schemas.md §1)."""

    device_id: str = Field(
        ...,
        description="Client-generated stable device UUID.",
    )
    pin: str = Field(
        ...,
        min_length=4,
        max_length=8,
        description="Master PIN, 4-8 digits (verified via SecurityGuard).",
    )


class LoginResponse(BaseModel):
    """POST /api/v1/auth/login success response (schemas.md §1)."""

    token: str = Field(..., description="Signed JWT.")
    session_id: str = Field(..., description="Server-generated session UUID.")
    expires_at: str = Field(..., description="ISO 8601 UTC expiry string.")
    expires_in_seconds: int = Field(..., description="JWT lifetime in seconds.")


# ── §2. ChatRequest / ChatResponse ───────────────────────────────────────────


class ChatRequest(BaseModel):
    """POST /api/v1/chat request body (schemas.md §2)."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's utterance, 1-4000 chars.",
    )
    conversation_id:   str   # ADDED — REQUIRED. Android must supply the Room
    

class ChatResponse(BaseModel):
    """POST /api/v1/chat response body (schemas.md §2)."""

    status: str = Field(
        ...,
        description="One of: success | error | auth_required.",
    )
    response: str = Field(..., description="The assistant's reply text.")
    trace_id: str = Field(..., description="Correlation trace id.")

class ChatSubmitResponse(BaseModel):
    task_id:    str
    trace_id:   str
    status:     str = "queued"

# ── §3. TaskSubmitRequest / TaskSubmitResponse ───────────────────────────────


class TaskSubmitRequest(BaseModel):
    """POST /api/v1/tasks request body (schemas.md §3)."""

    user_input: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's input, 1-4000 chars.",
    )
    tools_required: bool = Field(default=False, description="Whether tools are required.")
    provider_name: str | None = Field(
        default=None,
        description="Optional provider: groq | gemini. Defaults to backend routing.",
    )


class TaskSubmitResponse(BaseModel):
    """POST /api/v1/tasks response body (schemas.md §3)."""

    task_id: str = Field(..., description="UUID of the submitted task.")
    status: str = Field(..., description="Initial status, always 'pending'.")


# ── §4. TaskStatusResponse ───────────────────────────────────────────────────


class TaskStatusResponse(BaseModel):
    """Task status snapshot (schemas.md §4) — mirrors core.task.Task."""
    

    task_id: str = Field(..., description="UUID.")
    user_input: str = Field(..., description="The user's input.")
    task_type: str = Field(..., description="immediate | background | cron.")
    status: str = Field(
        ...,
        description="pending | scheduled | running | cancelling | completed | failed | cancelled.",
    )
    priority: int = Field(..., ge=0, le=4, description="0-4 (see Priority enum).")
    conversation_id:   str | None = None   # ADDED
    result: str | None = Field(default=None, description="Result string, if any.")
    error: str | None = Field(default=None, description="Error string, if any.")
    created_at: datetime = Field(..., description="ISO 8601 UTC.")
    updated_at: datetime = Field(..., description="ISO 8601 UTC.")
    scheduled_at: datetime | None = Field(default=None, description="ISO 8601 UTC, if any.")


# ── §5. JobListResponse ──────────────────────────────────────────────────────


class JobListResponse(BaseModel):
    """GET /api/v1/jobs response body (schemas.md §5)."""

    jobs: list[TaskStatusResponse] = Field(..., description="Array of TaskStatusResponse.")
    total_count: int = Field(..., description="Total rows matching the filter.")
    limit: int = Field(..., description="Pagination limit.")
    offset: int = Field(..., description="Pagination offset.")


# ── §6. CancelResponse ───────────────────────────────────────────────────────


class CancelResponse(BaseModel):
    """POST /api/v1/tasks/{task_id}/cancel response body (schemas.md §6)."""

    cancelled: bool = Field(..., description="True if a cancellation was issued.")
    reason: str | None = Field(
        default=None,
        description="Populated only when cancelled=false.",
    )


# ── §7. CancelAllResponse ────────────────────────────────────────────────────


class CancelAllResponse(BaseModel):
    """POST /api/v1/tasks/cancel_all response body (schemas.md §7)."""

    cancelled_task_ids: list[str] = Field(..., description="Ids actually cancelled.")
    count: int = Field(..., description="Number of cancelled tasks.")


# ── §8. ProfileGetResponse / ProfileSetRequest ───────────────────────────────


class ProfileGetResponse(BaseModel):
    """GET /api/v1/memory/profile response body (schemas.md §8)."""

    entries: dict[str, Any] = Field(..., description="Key/value profile entries.")


class ProfileSetRequest(BaseModel):
    """POST /api/v1/memory/profile request body (schemas.md §8)."""

    key: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Profile key, 1-128 chars.",
    )
    value: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Profile value, 1-4000 chars.",
    )


# ── §9. ErrorResponse (all endpoints, on non-2xx) ────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error body for all non-2xx responses (schemas.md §9)."""

    error_code: str = Field(..., description="See docs/api/error_codes.md.")
    message: str = Field(..., description="Human-readable message.")
    trace_id: str | None = Field(default=None, description="Trace id, if any.")
