"""
core/task.py — CYRAX 3.0 Task Data Structures (Phase 8.4)

Phase 8.4 changes:
  - TaskStatus gains CANCELLING (transitional state between a cancel
    request being issued and the executor's finally block confirming
    CANCELLED — lets a get_task() caller distinguish "cancellation in
    flight" from "already fully stopped").
  - Task gains interruptible: bool = True (Constraint 3).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    CANCELLING = "cancelling"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


class TaskType(str, Enum):
    IMMEDIATE  = "immediate"
    BACKGROUND = "background"
    CRON       = "cron"


class Task(BaseModel):
    """
    task_id:         UUID string, generated at creation.
    user_input:       The raw request text this task represents.
    task_type:        IMMEDIATE | BACKGROUND | CRON.
    status:           Current lifecycle state. Starts at PENDING.
    tools_required:   Carried from CognitiveRoutingDecision.
    provider_name:    Carried from CognitiveRoutingDecision.selected_provider.
    interruptible:    If False, this task ignores cancel()/cancel_all()
                       requests entirely — it always runs to completion.
    result:           Populated on COMPLETED. None otherwise.
    error:            Populated on FAILED. None otherwise.
    created_at:       UTC timestamp, set once at construction.
    updated_at:       UTC timestamp, refreshed on every status change.
    """

    task_id:        str        = Field(default_factory=lambda: str(uuid.uuid4()))
    user_input:      str
    task_type:       TaskType
    status:          TaskStatus = TaskStatus.PENDING
    tools_required:  bool       = False
    provider_name:   str        = "groq"
    interruptible:   bool       = True
    result:          str | None = None
    error:           str | None = None
    created_at:      datetime   = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:      datetime   = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"use_enum_values": False}
