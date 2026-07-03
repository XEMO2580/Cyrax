"""
orchestrator/supervisor.py — CYRAX 3.0 Cognitive Supervisor

The AgentSupervisor wraps multi-step plan execution with a reflection-
and-retry loop. When a tool in a plan fails, the Supervisor sends the
failure context to the LLM for reflection, receives a corrected plan,
and retries — up to MAX_RETRIES times before aborting.

Architecture contracts:
  - The Supervisor NEVER executes tools directly.
    All tool calls are routed through ctx.tool_registry.execute_tool()
    so every security gate, rate limit, auth check, and trace log
    remains intact.
  - The Supervisor NEVER calls dispatcher._llm_pipeline() recursively.
    It calls ctx.brain_router.plan() for reflection directly, then
    passes the corrected plan back to its own _execute_plan() method.
  - Supervisor reasoning traces are written to ctx.memory.state after
    every reflection cycle so the Dispatcher can inspect them and the
    trace survives for post-mortem debugging.

Integration:
    Dispatcher._llm_pipeline() creates an AgentSupervisor and delegates
    plan execution to it instead of running the execution loop itself.
    See patch instructions at the bottom of this file.

Cognitive loop:
    1. Receive initial plan from brain_router.plan().
    2. Execute steps sequentially via _execute_plan().
    3. On tool failure:
         a. Write failure to supervisor reasoning trace.
         b. Call brain_router with a reflection prompt.
         c. Parse the corrected plan.
         d. Increment retry counter.
         e. If retries exhausted → abort, return final error.
         f. If corrected plan empty → unrecoverable, return error.
         g. Otherwise → retry from step 2 with corrected plan.
    4. On full success → return combined results.
    5. On auth_required mid-plan → surface immediately (cannot recover).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Hard ceiling on reflection-retry cycles ───────────────────────────────────
MAX_RETRIES: int = 3

# ── Maximum steps in any single plan (initial or reflected) ──────────────────
MAX_PLAN_STEPS: int = 5

# ── Maximum characters of a tool error surfaced to the reflection LLM ────────
_MAX_ERROR_IN_PROMPT: int = 400


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """Result of a single tool execution step."""
    tool_name: str
    args:      dict[str, Any]
    status:    str       # "success" | "error" | "auth_required"
    response:  str


@dataclass
class SupervisorTrace:
    """
    Full reasoning trace for one supervisor invocation.
    Written to ctx.memory.state[KEY_SUPERVISOR_TRACE] after every cycle.
    """
    user_input:    str
    initial_plan:  list[dict]
    cycles:        list[dict] = field(default_factory=list)
    final_status:  str        = "pending"   # "success" | "error" | "aborted"
    final_response: str       = ""

    def add_cycle(
        self,
        attempt:         int,
        plan:            list[dict],
        failed_step:     StepResult | None,
        reflection_plan: list[dict] | None,
    ) -> None:
        self.cycles.append({
            "attempt":         attempt,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "plan":            plan,
            "failed_step":     {
                "tool":     failed_step.tool_name,
                "args":     failed_step.args,
                "status":   failed_step.status,
                "response": failed_step.response[:200],
            } if failed_step else None,
            "reflection_plan": reflection_plan,
        })


# ══════════════════════════════════════════════════════════════════════════════
# REFLECTION PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def _build_reflection_prompt(
    user_input:   str,
    failed_step:  StepResult,
    attempt:      int,
    tool_definitions: list[dict],
) -> str:
    """
    Constructs the system prompt for the reflection phase.

    Instructs the LLM to analyse the failure and produce a corrected
    JSON execution plan, or return [] if recovery is impossible.

    The prompt is intentionally minimal — the reflection call uses
    temperature=0.0 so it needs clear rules, not persuasion.
    """
    import json
    tools_json = json.dumps(tool_definitions, indent=2)

    return f"""You are the Recovery Planner for CYRAX, an AI operating system.
A previous execution attempt failed. Your job is to analyse the failure
and produce a corrected JSON plan to recover, or return [] if unrecoverable.

══════════════════════════════════════════════════════
FAILURE REPORT
══════════════════════════════════════════════════════
Original user request : "{user_input}"
Failed tool           : {failed_step.tool_name}
Tool arguments        : {failed_step.args}
Error received        : {failed_step.response[:_MAX_ERROR_IN_PROMPT]}
Retry attempt         : {attempt} of {MAX_RETRIES}

══════════════════════════════════════════════════════
RECOVERY RULES
══════════════════════════════════════════════════════
1. OUTPUT FORMAT: Return ONLY a raw JSON array. No markdown, no explanation.
2. If you can recover (e.g. try a different tool, different args, fallback):
   Return a new plan array: [{{"tool": "...", "args": {{...}}}}]
3. If the error is unrecoverable (wrong credentials, not found, rate limit
   exhausted, security block, permission denied):
   Return exactly: []
4. Do NOT repeat the exact same tool call with the exact same args that failed.
5. TOOL NAMES must exactly match the AVAILABLE TOOLS list below.
6. Maximum {MAX_PLAN_STEPS} steps in the recovery plan.

══════════════════════════════════════════════════════
AVAILABLE TOOLS
══════════════════════════════════════════════════════
{tools_json}

══════════════════════════════════════════════════════
RECOVERY EXAMPLES
══════════════════════════════════════════════════════
Failure: OPEN_APP with app="spotify" → "Security Block: not on whitelist"
Recovery: [{{"tool": "WEB_SEARCH", "args": {{"query": "spotify"}}}}]
(Reason: Spotify is in the web fallback list — open it via search instead)

Failure: DELETE_FILE filepath="report.txt" → "Error: file does not exist"
Recovery: []
(Reason: The file is gone; deletion is already the desired end state)

Failure: WEB_SEARCH query="..." → "Error: Rate limit exceeded"
Recovery: []
(Reason: Rate limit cannot be resolved by retrying immediately)

Failure: READ_FILE filepath="notes.txt" → "Error: does not exist"
Recovery: [{{"tool": "LIST_DIR", "args": {{"dirpath": "."}}}}]
(Reason: List the workspace to find the correct filename)

══════════════════════════════════════════════════════
When in doubt: return []. Do not guess.
══════════════════════════════════════════════════════"""


# ══════════════════════════════════════════════════════════════════════════════
# AGENT SUPERVISOR
# ══════════════════════════════════════════════════════════════════════════════

class AgentSupervisor:
    """
    Wraps multi-step plan execution with a bounded reflection-and-retry loop.

    Instantiated per-request by Dispatcher._llm_pipeline().
    Stateless across requests — all state lives in SupervisorTrace,
    which is written to ctx.memory.state after every cycle.

    Args:
        ctx:       CyraxContext — provides tool_registry, brain_router, memory.
        req_id:    Trace ID from the dispatcher for log correlation.
    """

    def __init__(self, ctx: Any, req_id: str) -> None:
        self._ctx    = ctx
        self._req_id = req_id

    # ── Public entry point ────────────────────────────────────────────────────

    async def execute_with_reflection(
        self,
        user_input:   str,
        initial_plan: list[dict],
    ) -> dict[str, Any]:
        """
        Executes a plan with supervised reflection-and-retry on failure.

        Args:
            user_input:   Original user utterance (used in reflection prompt).
            initial_plan: Tool execution plan from brain_router.plan().

        Returns:
            {"status": str, "response": str}
            Status values: "success" | "error" | "auth_required"
        """
        trace = SupervisorTrace(
            user_input   = user_input,
            initial_plan = initial_plan,
        )

        current_plan = self._cap_plan(initial_plan)
        attempt      = 0

        while attempt <= MAX_RETRIES:
            attempt += 1

            logger.info(
                f"[{self._req_id}] SUPERVISOR | "
                f"Attempt {attempt}/{MAX_RETRIES + 1} | "
                f"steps={len(current_plan)}"
            )

            result, failed_step = await self._execute_plan(current_plan)

            # ── Auth required mid-plan ────────────────────────────────────────
            # Cannot recover from auth in a background reflection loop.
            # Surface immediately so the dispatcher can prompt the user.
            if result["status"] == "auth_required":
                trace.final_status   = "auth_required"
                trace.final_response = result["response"]
                self._write_trace(trace)
                return result

            # ── Full success ──────────────────────────────────────────────────
            if failed_step is None:
                trace.final_status   = "success"
                trace.final_response = result["response"]
                self._write_trace(trace)
                logger.info(
                    f"[{self._req_id}] SUPERVISOR | "
                    f"Plan succeeded on attempt {attempt}."
                )
                return result

            # ── Step failed — check retry budget ──────────────────────────────
            if attempt > MAX_RETRIES:
                logger.error(
                    f"[{self._req_id}] SUPERVISOR | "
                    f"MAX_RETRIES ({MAX_RETRIES}) exhausted. Aborting."
                )
                trace.add_cycle(
                    attempt         = attempt,
                    plan            = current_plan,
                    failed_step     = failed_step,
                    reflection_plan = None,
                )
                trace.final_status   = "aborted"
                trace.final_response = result["response"]
                self._write_trace(trace)
                return {
                    "status":   "error",
                    "response": (
                        f"I was unable to complete this task after "
                        f"{MAX_RETRIES} attempts. "
                        f"Last error: {failed_step.response}"
                    ),
                }

            # ── Reflection cycle ──────────────────────────────────────────────
            logger.info(
                f"[{self._req_id}] SUPERVISOR | "
                f"Step '{failed_step.tool_name}' failed. "
                f"Requesting reflection (attempt {attempt}/{MAX_RETRIES})."
            )

            reflected_plan = await self._reflect(
                user_input   = user_input,
                failed_step  = failed_step,
                attempt      = attempt,
            )

            trace.add_cycle(
                attempt         = attempt,
                plan            = current_plan,
                failed_step     = failed_step,
                reflection_plan = reflected_plan,
            )
            self._write_trace(trace)

            # ── Reflection returned empty — unrecoverable ─────────────────────
            if not reflected_plan:
                logger.warning(
                    f"[{self._req_id}] SUPERVISOR | "
                    f"Reflection returned [] — unrecoverable failure."
                )
                trace.final_status   = "error"
                trace.final_response = result["response"]
                self._write_trace(trace)
                return {
                    "status":   "error",
                    "response": (
                        f"I attempted to recover but determined the task "
                        f"cannot be completed. "
                        f"Reason: {failed_step.response}"
                    ),
                }

            # ── Use reflected plan for next attempt ───────────────────────────
            current_plan = self._cap_plan(reflected_plan)
            logger.info(
                f"[{self._req_id}] SUPERVISOR | "
                f"Reflected plan accepted: {len(current_plan)} step(s). "
                f"Retrying."
            )

        # Unreachable — loop exits via return in all branches above.
        return {
            "status":   "error",
            "response": "Supervisor loop exited unexpectedly.",
        }

    # ── Plan execution ────────────────────────────────────────────────────────

    async def _execute_plan(
        self,
        plan: list[dict],
    ) -> tuple[dict[str, Any], StepResult | None]:
        """
        Executes steps sequentially through ctx.tool_registry.
        Stops on the first failure and returns (result_dict, failed_StepResult).
        Returns (result_dict, None) on full success.

        The Supervisor NEVER calls tool.execute() directly.
        All calls go through ctx.tool_registry.execute_tool() so that:
          - SecurityLevel gates are enforced
          - Rate limiting is enforced
          - Pydantic validation is enforced
          - Timeouts are enforced
          - All executions appear in event logs with trace IDs
        """
        results_log:       list[str] = []
        successful_tools:  list[str] = []

        for step in plan:
            tool_name: str       = step.get("tool", "")
            args:      dict      = step.get("args", {})

            if not tool_name:
                logger.debug(
                    f"[{self._req_id}] SUPERVISOR | "
                    f"Skipping plan step with no tool name."
                )
                continue

            logger.info(
                f"[{self._req_id}] SUPERVISOR | "
                f"Executing: {tool_name} | args={args}"
            )

            step_result = await self._execute_step(tool_name, args)

            # ── Auth required ─────────────────────────────────────────────────
            if step_result.status == "auth_required":
                # Store pending auth state so dispatcher can resume.
                self._ctx.memory.state.set("pending_auth",  True)
                self._ctx.memory.state.set("pending_tool",  tool_name)
                self._ctx.memory.state.set("pending_args",  args)
                return {
                    "status":   "auth_required",
                    "response": step_result.response,
                }, None

            # ── Tool failed ───────────────────────────────────────────────────
            if step_result.status != "success":
                results_log.append(step_result.response)
                combined = "\n".join(results_log)
                return {
                    "status":   "error",
                    "response": combined or step_result.response,
                }, step_result

            # ── Tool succeeded ────────────────────────────────────────────────
            results_log.append(step_result.response)
            successful_tools.append(tool_name)

            # Track last opened app for pronoun resolution.
            if tool_name == "OPEN_APP":
                app_name = args.get("app", "")
                if app_name:
                    self._ctx.memory.state.set("last_opened_app", app_name)

        # All steps completed without failure.
        if successful_tools:
            summary = f"I successfully executed: {', '.join(successful_tools)}."
        else:
            summary = "\n".join(results_log) if results_log else "No actions taken."

        return {"status": "success", "response": summary}, None

    async def _execute_step(
        self,
        tool_name: str,
        args:      dict[str, Any],
    ) -> StepResult:
        """
        Executes a single tool call through the registry.
        Wraps the result in a StepResult. Never raises.
        """
        import asyncio

        try:
            raw = await asyncio.wait_for(
                self._ctx.tool_registry.execute_tool(tool_name, args),
                timeout=settings.TOOL_TIMEOUT_SECONDS,
            )
            return StepResult(
                tool_name = tool_name,
                args      = args,
                status    = raw.get("status",   "error"),
                response  = raw.get("response", "No response."),
            )
        except asyncio.TimeoutError:
            msg = (
                f"Tool '{tool_name}' timed out after "
                f"{settings.TOOL_TIMEOUT_SECONDS}s."
            )
            logger.error(f"[{self._req_id}] SUPERVISOR | {msg}")
            return StepResult(
                tool_name = tool_name,
                args      = args,
                status    = "error",
                response  = f"Error: {msg}",
            )
        except Exception as exc:
            logger.exception(
                f"[{self._req_id}] SUPERVISOR | "
                f"Unexpected error in {tool_name}: {exc}"
            )
            return StepResult(
                tool_name = tool_name,
                args      = args,
                status    = "error",
                response  = f"Error: '{tool_name}' encountered an unexpected error.",
            )

    # ── Reflection ────────────────────────────────────────────────────────────

    async def _reflect(
        self,
        user_input:  str,
        failed_step: StepResult,
        attempt:     int,
    ) -> list[dict]:
        """
        Sends the failure context to the LLM for reflection.

        Uses brain_router.plan() with a specialised reflection system prompt
        so the response is parsed through the same JSON validation pipeline
        that normal planning uses — no duplicate parsing logic here.

        Returns a validated list[dict] plan or [] if reflection fails.
        """
        tool_definitions = self._ctx.tool_registry.get_all_definitions()
        valid_tool_names = {t["name"] for t in tool_definitions}

        reflection_prompt = _build_reflection_prompt(
            user_input        = user_input,
            failed_step       = failed_step,
            attempt           = attempt,
            tool_definitions  = tool_definitions,
        )

        # We call the provider directly with the reflection prompt as
        # the system context. brain_router.plan() is reused but the
        # user_input here is a structured failure description, not a
        # natural language command — the reflection prompt carries the
        # full context the LLM needs.
        reflection_user_message = (
            f"The tool '{failed_step.tool_name}' failed with: "
            f"{failed_step.response[:_MAX_ERROR_IN_PROMPT]}. "
            f"Provide a recovery plan or []."
        )

        try:
            # Bypass brain_router.plan()'s normal system prompt —
            # use the reflection prompt instead by calling the provider
            # directly through brain_router.route() with constructed messages.
            from brain.moe_router import _parse_plan  # reuse validated parser

            provider = self._ctx.brain_router._get_provider_for_planning()
            raw = await provider.generate(
                messages      = [
                    {"role": "user", "content": reflection_user_message}
                ],
                system_prompt = reflection_prompt,
                max_tokens    = 400,
                temperature   = 0.0,
                json_mode     = provider.capabilities.supports_json_mode,
            )

            plan = _parse_plan(raw, valid_tool_names, self._req_id)
            logger.debug(
                f"[{self._req_id}] REFLECTION | "
                f"Parsed {len(plan)} step(s): "
                f"{[s['tool'] for s in plan]}"
            )
            return plan

        except Exception as exc:
            logger.error(
                f"[{self._req_id}] REFLECTION | "
                f"Reflection call failed: {exc}. "
                f"Treating as unrecoverable."
            )
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cap_plan(self, plan: list[dict]) -> list[dict]:
        """Enforces MAX_PLAN_STEPS on any plan before execution."""
        if len(plan) > MAX_PLAN_STEPS:
            logger.warning(
                f"[{self._req_id}] SUPERVISOR | "
                f"Plan truncated: {len(plan)} → {MAX_PLAN_STEPS} steps."
            )
            return plan[:MAX_PLAN_STEPS]
        return plan

    def _write_trace(self, trace: SupervisorTrace) -> None:
        """
        Persists the supervisor reasoning trace to session state.
        Written after every cycle so it survives even if the process
        is interrupted before the full invocation completes.
        """
        try:
            from memory.state.session_state import KEY_SUPERVISOR_TRACE
            self._ctx.memory.state.set(
                KEY_SUPERVISOR_TRACE,
                {
                    "user_input":     trace.user_input,
                    "initial_plan":   trace.initial_plan,
                    "cycles":         trace.cycles,
                    "final_status":   trace.final_status,
                    "final_response": trace.final_response[:500],
                },
            )
        except Exception as exc:
            logger.warning(
                f"[{self._req_id}] SUPERVISOR | "
                f"Could not write trace to session state: {exc}"
            )