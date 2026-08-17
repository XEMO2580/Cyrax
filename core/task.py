"""
core/task.py — CYRAX 3.0 Task Data Structures (Phase 8.6)

Phase 8.6 change: Task gains priority: int, defaulting to
Priority.BACKGROUND.value, consumed by TaskQueue's PriorityQueue ordering.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from core.resource_manager import Priority


class TaskStatus(str, Enum):
    PENDING    = "pending"
    SCHEDULED  = "scheduled"
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
    task_id:        str        = Field(default_factory=lambda: str(uuid.uuid4()))
    user_input:      str
    task_type:       TaskType
    status:          TaskStatus = TaskStatus.PENDING
    tools_required:  bool       = False
    provider_name:   str        = "groq"
    interruptible:   bool       = True
    priority:        int        = Priority.BACKGROUND.value
    conversation_id: str | None = None
    device_id:       str | None = None   # ADDED — Gate C.1 P0.1 ownership
    result:          str | None = None
    error:           str | None = None
    created_at:      datetime   = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:      datetime   = Field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at:    datetime | None = None

    model_config = {"use_enum_values": False}
