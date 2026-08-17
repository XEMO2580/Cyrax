from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_current_device_id, get_session_context
from app.api.schemas import ChatRequest, ChatSubmitResponse
from core.context import CyraxContext
from core.resource_manager import Priority
from core.task import TaskType
from core.trace import new_trace_id
from orchestrator.intent_parser.fast_path import IntentRouter

router = APIRouter()


@router.post("/chat", status_code=202, response_model=ChatSubmitResponse)
async def submit_chat(
    request: ChatRequest,
    device_id: str = Depends(get_current_device_id),   # ADDED
    ctx: CyraxContext = Depends(get_session_context),
) -> ChatSubmitResponse:
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    trace_id = new_trace_id()
    parsed_intent = IntentRouter.parse(user_text)

    if parsed_intent:
        tools_required = True
        provider_name = "groq"
    else:
        decision = await ctx.decision_engine.classify(user_text)
        tools_required = decision.tools_required
        provider_name = decision.selected_provider

    # FIX: all fields passed directly into submit() — no post-hoc mutation,
    # no lost fields on the first SQLite write.
    task = await ctx.task_queue.submit(
        user_input=user_text,
        task_type=TaskType.IMMEDIATE,
        priority=Priority.INTERACTIVE.value,
        tools_required=tools_required,
        provider_name=provider_name,
        conversation_id=request.conversation_id,
        device_id=device_id,
    )

    return ChatSubmitResponse(task_id=task.task_id, trace_id=trace_id, status="queued")
