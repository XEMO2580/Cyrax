from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_current_device_id, get_session_context
from app.api.schemas import (
    TaskStatusResponse, TaskSubmitRequest, TaskSubmitResponse,
    JobListResponse, CancelResponse, CancelAllResponse,
)
from core.context import CyraxContext

router = APIRouter()


def _task_to_status_response(t) -> TaskStatusResponse:
    """Helper: convert a core.task.Task -> TaskStatusResponse."""
    return TaskStatusResponse(
        task_id=t.task_id,
        user_input=t.user_input,
        task_type=t.task_type.value,
        status=t.status.value,
        priority=t.priority,
        conversation_id=t.conversation_id,
        result=t.result,
        error=t.error,
        created_at=t.created_at,
        updated_at=t.updated_at,
        scheduled_at=t.scheduled_at,
    )


@router.get("/tasks/reconcile", response_model=list[TaskStatusResponse])
async def reconcile_tasks(
    device_id: str = Depends(get_current_device_id),
    ctx: CyraxContext = Depends(get_session_context),
) -> list[TaskStatusResponse]:
    """P0.2: boot-time reconciliation source for Android."""
    tasks = await ctx.job_store.get_tasks_for_device(device_id, limit=100)
    return [_task_to_status_response(t) for t in tasks]


@router.post("/tasks", status_code=202, response_model=TaskSubmitResponse)
async def submit_task(
    body: TaskSubmitRequest,
    ctx: CyraxContext = Depends(get_session_context),
):
    """Direct task submission (used by clients that don't go through /chat)."""
    # Delegate to the TaskQueue; tests supply an AsyncMock that accepts kwargs.
    task = await ctx.task_queue.submit(
        user_input=body.user_input,
        tools_required=body.tools_required,
        provider_name=body.provider_name,
    )
    return TaskSubmitResponse(task_id=task.task_id, status=task.status.value)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    ctx: CyraxContext = Depends(get_session_context),
):
    """Fetch a single task status snapshot. Returns 404/TASK_NOT_FOUND if unknown."""
    task = await ctx.task_queue.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "TASK_NOT_FOUND", "message": "Unknown task_id."},
        )
    return _task_to_status_response(task)


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
    ctx: CyraxContext = Depends(get_session_context),
):
    """Paginated job listing backed by the persistence layer / TaskQueue."""
    jobs, total = await ctx.task_queue.list_tasks(limit=limit, offset=offset, status=status)
    return JobListResponse(jobs=[_task_to_status_response(t) for t in jobs], total_count=total, limit=limit, offset=offset)


@router.post("/tasks/{task_id}/cancel", response_model=CancelResponse)
async def cancel_task(
    task_id: str,
    ctx: CyraxContext = Depends(get_session_context),
):
    """Request cancellation for a specific task. Returns 404 if unknown."""
    task = await ctx.task_queue.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "TASK_NOT_FOUND", "message": "Unknown task_id."},
        )

    cancelled = ctx.interrupt_controller.cancel(task_id, "client_cancel")
    if cancelled:
        return CancelResponse(cancelled=True)
    else:
        return CancelResponse(cancelled=False, reason="task not running or non-interruptible")


@router.post("/tasks/cancel_all", response_model=CancelAllResponse)
async def cancel_all_tasks(
    ctx: CyraxContext = Depends(get_session_context),
):
    """Cancel all currently-registered interruptible tasks."""
    cancelled_ids = ctx.interrupt_controller.cancel_all("client_cancel_all")
    return CancelAllResponse(cancelled_task_ids=cancelled_ids, count=len(cancelled_ids))
