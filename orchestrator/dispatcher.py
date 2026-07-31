"""
orchestrator/dispatcher.py — CYRAX 3.0 Orchestration Layer (Phase 8.3)

Phase 8.3 changes:
  - _llm_pipeline() now routes based on execution_mode (background → queue,
    immediate → chat, interactive → Planner).
  - ctx.task_queue.submit() passes tools_required and provider_name directly
    (no more hacky post-submit attribute patching).
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Any

from config.settings import settings
from core.context import CyraxContext
from orchestrator.planner import Planner

logger = logging.getLogger(__name__)


_SHUTDOWN_PHRASES: frozenset[str] = frozenset({
    "shutdown", "exit", "quit", "shutdown cyrax",
})

_PRONOUN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(close|quit|kill|shut)\b.+\b(it|that|this|the app|the application)\b", re.IGNORECASE), "close"),
    (re.compile(r"\b(open|launch|start)\b.+\b(it|that|this|the app|the application)\b", re.IGNORECASE), "open"),
]


class Dispatcher:
    """
    Stateless orchestrator. Pure orchestration only.
    Gates every non-fast-path query through the Decision Engine before
    routing to either direct chat or the ReAct Planner.
    """

    # ── Entry Point ───────────────────────────────────────────────────────────

    async def handle(
        self,
        user_input: str,
        session_id: str,
        trace_id:   str,
        ctx:        CyraxContext,
    ) -> dict[str, Any]:
        req_id = f"{trace_id}:{uuid.uuid4().hex[:4]}"
        logger.info(f"[{req_id}] DISPATCH_START | input='{user_input[:80]}'")

        user_input = user_input.strip()
        if not user_input:
            return {"status": "error", "response": "No input received."}

        if user_input.lower() in _SHUTDOWN_PHRASES:
            return {"status": "shutdown", "response": "Shutting down."}

        if ctx.memory.state.get("pending_auth"):
            return await self._resume_after_auth(user_input, req_id, ctx)

        resolved_input = self._resolve_pronouns(user_input, ctx)

        from orchestrator.intent_parser.fast_path import IntentRouter, ParsedIntent
        parsed: ParsedIntent | None = IntentRouter.parse(resolved_input)

        if parsed:
            logger.debug(f"[{req_id}] FAST_PATH | tool={parsed.tool_name} confidence={parsed.confidence}")
            return await self._execute_intent(
                parsed.tool_name, parsed.parameters, resolved_input, req_id, ctx,
            )

        logger.debug(f"[{req_id}] LLM_PATH")
        return await self._llm_pipeline(resolved_input, req_id, ctx)

    # ── Auth Flow (unchanged) ────────────────────────────────────────────────

    async def _resume_after_auth(self, raw_pin: str, req_id: str, ctx: CyraxContext) -> dict[str, Any]:
        authenticated = ctx.security.authenticate(raw_pin)

        if not authenticated:
            locked, remaining = ctx.security.is_locked_out()
            if locked:
                ctx.memory.state.clear("pending_auth")
                ctx.memory.state.clear("pending_tool")
                ctx.memory.state.clear("pending_args")
                return {
                    "status": "error",
                    "response": f"Too many failed attempts. System locked for {int(remaining)} seconds.",
                }
            return {"status": "auth_required", "response": "Incorrect PIN. Please try again."}

        pending_tool: str | None  = ctx.memory.state.get("pending_tool")
        pending_args: dict | None = ctx.memory.state.get("pending_args")

        ctx.memory.state.clear("pending_auth")
        ctx.memory.state.clear("pending_tool")
        ctx.memory.state.clear("pending_args")

        if not pending_tool:
            return {"status": "success", "response": "Authenticated successfully."}

        logger.info(f"[{req_id}] AUTH_RESUME | replaying tool={pending_tool}")
        return await self._execute_intent(
            pending_tool, pending_args or {}, f"[auth-resume] {pending_tool}", req_id, ctx,
        )

    # ── Pronoun Resolution (unchanged) ───────────────────────────────────────

    def _resolve_pronouns(self, user_input: str, ctx: CyraxContext) -> str:
        last_app: str | None = ctx.memory.state.get("last_opened_app")
        if not last_app:
            return user_input
        for pattern, verb in _PRONOUN_PATTERNS:
            if pattern.search(user_input):
                resolved = f"{verb} {last_app}"
                logger.debug(f"PRONOUN_RESOLVE | '{user_input}' → '{resolved}'")
                return resolved
        return user_input

    # ── Single-Intent Execution (unchanged) ──────────────────────────────────

    async def _execute_intent(
        self, tool_name: str, args: dict[str, Any], user_text: str, req_id: str, ctx: CyraxContext,
    ) -> dict[str, Any]:
        result = await self._execute_step(tool_name, args, req_id, ctx)

        if result["status"] == "auth_required":
            ctx.memory.state.set("pending_auth", True)
            ctx.memory.state.set("pending_tool", tool_name)
            ctx.memory.state.set("pending_args", args)
            await ctx.memory.conversation.add_interaction("user", user_text)
            await ctx.memory.conversation.add_interaction("assistant", result["response"])
            return result

        if result["status"] == "success":
            app_name = args.get("app", "")
            if app_name:
                ctx.memory.state.set("last_opened_app", app_name)

        summary = self._sanitize_output(result["response"])
        await ctx.memory.conversation.add_interaction("user", user_text)
        await ctx.memory.conversation.add_interaction("assistant", summary)
        return {"status": result["status"], "response": summary}

    # ── LLM Pipeline — Phase 8.3: Decision Engine + execution_mode routing ───

    async def _llm_pipeline(self, user_text: str, req_id: str, ctx: CyraxContext) -> dict[str, Any]:
        """
        Phase 8.4: hardcoded system interceptor for stop/cancel/abort
        runs FIRST, before the Decision Engine is ever called — Constraint 2.
        """
        stripped_lower = user_text.strip().lower()

        if stripped_lower in {"stop", "cancel", "abort"}:
            cancelled_ids = ctx.interrupt_controller.cancel_all(
                "User triggered global stop."
            )
            count = len(cancelled_ids)

            if count == 0:
                response = "No running tasks to stop."
            else:
                response = f"✓ Stopped {count} running task(s)."

            await ctx.memory.conversation.add_interaction("user", user_text)
            await ctx.memory.conversation.add_interaction("assistant", response)
            return {"status": "success", "response": response}

        # ── Everything below unchanged from Phase 8.3 ─────────────────────────
        history = ctx.memory.conversation.get_history()

        decision = await ctx.decision_engine.classify(user_text)

        logger.info(
            f"[{req_id}] DECISION_ENGINE | "
            f"intent={decision.intent} | "
            f"tools_required={decision.tools_required} | "
            f"execution_mode={decision.execution_mode.value} | "
            f"provider={decision.selected_provider}"
        )

        # ── Background: queue and return immediately ─────────────────────────
        if decision.execution_mode.value == "background":
            from core.task import TaskType

            task = await ctx.task_queue.submit(
                user_input=user_text,
                task_type=TaskType.BACKGROUND,
                tools_required=decision.tools_required,
                provider_name=decision.selected_provider,
            )

            logger.info(
                f"[{req_id}] BACKGROUND_QUEUED | task_id={task.task_id}"
            )

            ack = (
                f"✓ Task Accepted\n"
                f"Task ID: {task.task_id}\n"
                f"Status: QUEUED. I will notify you when it finishes."
            )
            await ctx.memory.conversation.add_interaction("user", user_text)
            await ctx.memory.conversation.add_interaction("assistant", ack)
            return {"status": "success", "response": ack}

        # ── Immediate: direct chat, no tools ──────────────────────────────────
        if not decision.tools_required:
            logger.debug(f"[{req_id}] CHAT_DIRECT | provider={decision.selected_provider}")
            response = await ctx.brain_router.chat(
                user_input=user_text,
                history=history,
                trace_id=req_id,
                provider_name=decision.selected_provider,
            )
            response = self._sanitize_output(response)
            await ctx.memory.conversation.add_interaction("user", user_text)
            await ctx.memory.conversation.add_interaction("assistant", response)
            return {"status": "success", "response": response}

        # ── Interactive: Planner, forced provider, synchronous ────────────────
        logger.info(
            f"[{req_id}] PLANNER_DELEGATE | "
            f"provider={decision.selected_provider} (forced by Decision Engine)"
        )

        planner = Planner(ctx=ctx, req_id=req_id, provider_name=decision.selected_provider)
        result = await planner.run(user_input=user_text, history=history)

        response_text = self._sanitize_output(result.get("response", ""))
        await ctx.memory.conversation.add_interaction("user", user_text)
        await ctx.memory.conversation.add_interaction("assistant", response_text)

        return {"status": result.get("status", "error"), "response": response_text}

    # ── Tool Execution Step (unchanged) ──────────────────────────────────────

    async def _execute_step(self, tool_name: str, args: dict[str, Any], req_id: str, ctx: CyraxContext) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                ctx.tool_registry.execute_tool(tool_name, args, ctx=ctx),
                timeout=settings.TOOL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return {"status": "error", "response": f"Tool '{tool_name}' timed out after {settings.TOOL_TIMEOUT_SECONDS}s."}
        except Exception as exc:
            logger.exception(f"[{req_id}] EXEC_CRASH | tool={tool_name} error={exc}")
            return {"status": "error", "response": f"Tool '{tool_name}' encountered an unexpected error."}

    # ── Output Sanitizer (unchanged) ─────────────────────────────────────────

    @staticmethod
    def _sanitize_output(text: str) -> str:
        if not text or not text.strip():
            return "I couldn't process that request."
        cleaned = "\n".join(line for line in text.splitlines() if line.strip())
        if len(cleaned) > 1500:
            cleaned = cleaned[:1500] + "\n... [Response truncated]"
        return cleaned
