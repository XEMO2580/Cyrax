"""
orchestrator/dispatcher.py — CYRAX 3.0 Orchestration Layer (DI Compliant)

Architectural contracts enforced:
  - Zero dynamic imports. All domain logic accessed via CyraxContext.
  - No tool names hardcoded. Fallback rules live in fallback_policy.py.
  - Pronoun resolution uses regex boundary detection for voice robustness.
  - Security accessed via ctx.security — never imports security.auth directly.
  - Brain accessed via ctx.brain_router.plan() / .chat() — never imports brain/.

Entry point: Dispatcher.handle()
Called by:   router_brain in app/cli_main.py and app/api_server.py
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Any

from config.settings import settings
from core.context import CyraxContext

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_SHUTDOWN_PHRASES: frozenset[str] = frozenset({
    "shutdown", "exit", "quit", "shutdown cyrax",
})

# Regex patterns for pronoun resolution.
# Matches verb + pronoun anywhere in the utterance, including with polite
# prefixes ("please close it", "can you open that up", "close that now").
_PRONOUN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(close|quit|kill|shut)\b.+\b(it|that|this|the app|the application)\b", re.IGNORECASE), "close"),
    (re.compile(r"\b(open|launch|start)\b.+\b(it|that|this|the app|the application)\b", re.IGNORECASE), "open"),
]


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

class Dispatcher:
    """
    Stateless orchestrator. All state is accessed through CyraxContext.
    No module-level or instance-level state.

    Satisfies DispatcherProtocol from core/context.py:
        async def handle(self, user_input, session_id, trace_id, ctx) -> dict
    """

    # ── Entry Point ───────────────────────────────────────────────────────────

    async def handle(
        self,
        user_input: str,
        session_id: str,
        trace_id: str,
        ctx: CyraxContext,
    ) -> dict[str, Any]:
        """
        Primary dispatch entry point.

        Returns:
            {
                "status":   "success" | "error" | "auth_required" | "shutdown",
                "response": str,
            }
        """
        req_id = f"{trace_id}:{uuid.uuid4().hex[:4]}"
        logger.info(f"[{req_id}] DISPATCH_START | input='{user_input[:80]}'")

        user_input = user_input.strip()
        if not user_input:
            return {"status": "error", "response": "No input received."}

        # ── Shutdown ──────────────────────────────────────────────────────────
        if user_input.lower() in _SHUTDOWN_PHRASES:
            return {"status": "shutdown", "response": "Shutting down."}

        # ── Pending auth interception ─────────────────────────────────────────
        if ctx.memory.state.get("pending_auth"):
            return await self._resume_after_auth(user_input, req_id, ctx)

        # ── Pronoun resolution ────────────────────────────────────────────────
        resolved_input = self._resolve_pronouns(user_input, ctx)

        # ── Fast-path intent resolution ───────────────────────────────────────
        from orchestrator.intent_parser.fast_path import IntentRouter, ParsedIntent

        parsed: ParsedIntent | None = IntentRouter.parse(resolved_input)

        if parsed:
            logger.debug(
                f"[{req_id}] FAST_PATH | "
                f"tool={parsed.tool_name} confidence={parsed.confidence}"
            )
            return await self._execute_intent(
                parsed.tool_name,
                parsed.parameters,
                resolved_input,
                req_id,
                ctx,
            )

        # ── LLM planning path ─────────────────────────────────────────────────
        logger.debug(f"[{req_id}] LLM_PATH")
        return await self._llm_pipeline(resolved_input, req_id, ctx)

    # ── Auth Flow ─────────────────────────────────────────────────────────────

    async def _resume_after_auth(
        self,
        raw_pin: str,
        req_id: str,
        ctx: CyraxContext,
    ) -> dict[str, Any]:
        """
        Treats user input as a PIN attempt.
        On success, replays the pending tool call.
        Accesses security exclusively via ctx.security.
        """
        authenticated = ctx.security.authenticate(raw_pin)

        if not authenticated:
            locked, remaining = ctx.security.is_locked_out()
            if locked:
                ctx.memory.state.clear("pending_auth")
                ctx.memory.state.clear("pending_tool")
                ctx.memory.state.clear("pending_args")
                return {
                    "status": "error",
                    "response": (
                        f"Too many failed attempts. "
                        f"System locked for {int(remaining)} seconds."
                    ),
                }
            return {
                "status": "auth_required",
                "response": "Incorrect PIN. Please try again.",
            }

        pending_tool: str | None = ctx.memory.state.get("pending_tool")
        pending_args: dict | None = ctx.memory.state.get("pending_args")

        ctx.memory.state.clear("pending_auth")
        ctx.memory.state.clear("pending_tool")
        ctx.memory.state.clear("pending_args")

        if not pending_tool:
            return {"status": "success", "response": "Authenticated successfully."}

        logger.info(f"[{req_id}] AUTH_RESUME | replaying tool={pending_tool}")
        return await self._execute_intent(
            pending_tool,
            pending_args or {},
            f"[auth-resume] {pending_tool}",
            req_id,
            ctx,
        )

    # ── Pronoun Resolution ────────────────────────────────────────────────────

    def _resolve_pronouns(
        self,
        user_input: str,
        ctx: CyraxContext,
    ) -> str:
        """
        Replaces app-referencing pronouns with the last opened app name.

        Uses regex boundary detection — catches natural voice utterances:
            "close it"             → "close chrome"
            "please close it"      → "close chrome"
            "can you open that up" → "open chrome"
            "shut that down"       → "close chrome"
        """
        last_app: str | None = ctx.memory.state.get("last_opened_app")
        if not last_app:
            return user_input

        for pattern, verb in _PRONOUN_PATTERNS:
            if pattern.search(user_input):
                resolved = f"{verb} {last_app}"
                logger.debug(
                    f"PRONOUN_RESOLVE | '{user_input}' → '{resolved}'"
                )
                return resolved

        return user_input

    # ── Single-Intent Execution ───────────────────────────────────────────────

    async def _execute_intent(
        self,
        tool_name: str,
        args: dict[str, Any],
        user_text: str,
        req_id: str,
        ctx: CyraxContext,
    ) -> dict[str, Any]:
        """
        Executes a single resolved intent through the registry.
        Handles auth interception, last-opened-app tracking, and memory writes.
        """
        result = await self._execute_step(tool_name, args, req_id, ctx)

        if result["status"] == "auth_required":
            ctx.memory.state.set("pending_auth", True)
            ctx.memory.state.set("pending_tool", tool_name)
            ctx.memory.state.set("pending_args", args)
            await ctx.memory.conversation.add_interaction("user", user_text)
            await ctx.memory.conversation.add_interaction(
                "assistant", result["response"]
            )
            return result

        if result["status"] == "success":
            app_name = args.get("app", "")
            if app_name:
                ctx.memory.state.set("last_opened_app", app_name)

        summary = (
            f"I successfully executed: {tool_name}."
            if result["status"] == "success"
            else result["response"]
        )
        summary = self._sanitize_output(summary)

        await ctx.memory.conversation.add_interaction("user", user_text)
        await ctx.memory.conversation.add_interaction("assistant", summary)

        return {"status": result["status"], "response": summary}

    # ── LLM Planning Pipeline ─────────────────────────────────────────────────

    async def _llm_pipeline(
        self,
        user_text: str,
        req_id: str,
        ctx: CyraxContext,
    ) -> dict[str, Any]:
        """
        Full LLM path. Accessed exclusively via ctx.brain_router:
            ctx.brain_router.plan()  — tool call planning
            ctx.brain_router.chat()  — conversational generation

        Fallback rules accessed exclusively via ctx.fallback_policy:
            ctx.fallback_policy.get_fallback() — no tool names here
        """
        history = ctx.memory.conversation.get_history()
        tool_definitions = ctx.tool_registry.get_all_definitions()

        # ── Planning ──────────────────────────────────────────────────────────
        plan: list[dict] = await ctx.brain_router.plan(
            user_input=user_text,
            history=history,
            tool_definitions=tool_definitions,
            trace_id=req_id,
        )

        logger.debug(
            f"[{req_id}] PLAN | steps={len(plan)} "
            f"preview={[s.get('tool') for s in plan[:3]]}"
        )

        # ── Conversational Fallback ───────────────────────────────────────────
        if not plan:
            logger.debug(f"[{req_id}] CHAT_FALLBACK")
            response = await ctx.brain_router.chat(
                user_input=user_text,
                history=history,
                trace_id=req_id,
            )
            response = self._sanitize_output(response)
            await ctx.memory.conversation.add_interaction("user", user_text)
            await ctx.memory.conversation.add_interaction("assistant", response)
            return {"status": "success", "response": response}

        # ── Plan Execution ────────────────────────────────────────────────────
        if len(plan) > settings.MAX_AGENT_STEPS:
            logger.warning(
                f"[{req_id}] PLAN_TRUNCATED | "
                f"size={len(plan)} → {settings.MAX_AGENT_STEPS}"
            )
            plan = plan[: settings.MAX_AGENT_STEPS]

        results_log: list[str] = []
        successful_tools: list[str] = []

        for step in plan:
            tool_name: str = step.get("tool") or step.get("intent", "")
            args: dict = step.get("args", {})

            if not tool_name:
                logger.warning(f"[{req_id}] PLAN_STEP_SKIP | missing tool name")
                continue

            step_result = await self._execute_step(tool_name, args, req_id, ctx)

            # Auth interception mid-plan
            if step_result["status"] == "auth_required":
                ctx.memory.state.set("pending_auth", True)
                ctx.memory.state.set("pending_tool", tool_name)
                ctx.memory.state.set("pending_args", args)
                await ctx.memory.conversation.add_interaction("user", user_text)
                await ctx.memory.conversation.add_interaction(
                    "assistant", step_result["response"]
                )
                return step_result

            results_log.append(step_result["response"])

            if step_result["status"] == "success":
                successful_tools.append(tool_name)
                app_name = args.get("app", "")
                if app_name:
                    ctx.memory.state.set("last_opened_app", app_name)

            else:
                logger.warning(
                    f"[{req_id}] STEP_FAILED | tool={tool_name} "
                    f"reason={step_result['response'][:80]}"
                )

                # ── Fallback policy query ─────────────────────────────────────
                # Dispatcher does not know which tool falls back to which.
                # That decision lives exclusively in fallback_policy.py.
                fallback = ctx.fallback_policy.get_fallback(
                    failed_tool=tool_name,
                    error_status=step_result["status"],
                    original_args=args,
                )

                if fallback:
                    logger.info(
                        f"[{req_id}] FALLBACK | "
                        f"{tool_name} → {fallback['tool']}"
                    )
                    fb_result = await self._execute_step(
                        fallback["tool"],
                        fallback["args"],
                        req_id,
                        ctx,
                    )
                    results_log.append(fb_result["response"])
                    if fb_result["status"] == "success":
                        successful_tools.append(fallback["tool"])

                # Supervisor hook point:
                # When supervisor.py is wired in (Priority 2), it intercepts
                # here before the break, analyses the failure, and may
                # attempt additional recovery steps.
                break

        # ── Plan Summary ──────────────────────────────────────────────────────
        if successful_tools:
            summary = f"I successfully executed: {', '.join(successful_tools)}."
        else:
            summary = "\n".join(results_log) if results_log else "No actions completed."

        summary = self._sanitize_output(summary)

        await ctx.memory.conversation.add_interaction("user", user_text)
        await ctx.memory.conversation.add_interaction("assistant", summary)

        return {
            "status": "success" if successful_tools else "error",
            "response": summary,
        }

    # ── Tool Execution Step ───────────────────────────────────────────────────

    async def _execute_step(
        self,
        tool_name: str,
        args: dict[str, Any],
        req_id: str,
        ctx: CyraxContext,
    ) -> dict[str, Any]:
        """
        Executes a single tool through the registry with timeout protection.
        Never raises — always returns a structured result dict.
        Timeout sourced from settings, not hardcoded.
        """
        logger.info(f"[{req_id}] EXEC | tool={tool_name} args={args}")

        try:
            result: dict[str, Any] = await asyncio.wait_for(
                ctx.tool_registry.execute_tool(tool_name, args),
                timeout=settings.TOOL_TIMEOUT_SECONDS,
            )
            logger.debug(
                f"[{req_id}] EXEC_DONE | "
                f"tool={tool_name} status={result.get('status')}"
            )
            return result

        except asyncio.TimeoutError:
            logger.error(
                f"[{req_id}] EXEC_TIMEOUT | "
                f"tool={tool_name} after {settings.TOOL_TIMEOUT_SECONDS}s"
            )
            return {
                "status": "error",
                "response": (
                    f"Tool '{tool_name}' timed out after "
                    f"{settings.TOOL_TIMEOUT_SECONDS}s."
                ),
            }

        except Exception as exc:
            logger.exception(
                f"[{req_id}] EXEC_CRASH | tool={tool_name} error={exc}"
            )
            return {
                "status": "error",
                "response": f"Tool '{tool_name}' encountered an unexpected error.",
            }

    # ── Output Sanitizer ──────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_output(text: str) -> str:
        if not text or not text.strip():
            return "I couldn't process that request."
        cleaned = "\n".join(
            line for line in text.splitlines() if line.strip()
        )
        if len(cleaned) > 1500:
            cleaned = cleaned[:1500] + "\n... [Response truncated]"
        return cleaned