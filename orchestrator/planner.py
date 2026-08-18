"""
orchestrator/planner.py — CYRAX 3.0 Hardened ReAct Planner (Phase 8.4.5 / V12-M2)

V12-M2 Hardening:
  - Strongly typed ReActDecision model with strict mutual exclusivity validation.
  - PlannerState enum for lifecycle transparency.
  - Robust JSON normalization with repair telemetry (fences, prose, normalized keys, unpacked args).
  - Declarative tool argument pre-validation against tool args_schema without retrieving BaseTool instances.
  - Configurable planner bounds (settings.MAX_AGENT_STEPS, PLANNER_MAX_TOKENS, PLANNER_WALL_CLOCK_TIMEOUT_SECONDS).
  - Cancellation-aware bounded schema correction retries with backoff.
  - Safe error boundary handling (ProviderCapabilityError, ProviderError, TimeoutError, CancelledError).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Type

from pydantic import BaseModel, Field, ValidationError, model_validator

from config.settings import settings
from core.interrupt_controller import CancellationToken
from brain.providers.base import (
    GenerationCancelledError,
    GenerationMode,
    ProviderCapabilityError,
    ProviderError,
)

logger = logging.getLogger(__name__)

_MAX_ERROR_IN_OBSERVATION: int = 400

# Simple in-process metrics for parse failures and correction attempts.
import threading as _threading
_parse_metrics_lock = _threading.Lock()
_parse_failures_counter: int = 0
_correction_attempts_counter: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# PLANNER STATE & DECISION MODELS
# ══════════════════════════════════════════════════════════════════════════════

class PlannerState(str, Enum):
    THINKING = "THINKING"
    ACTION = "ACTION"
    FINAL = "FINAL"
    RETRY = "RETRY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ReActDecision(BaseModel):
    thought: str = Field(
        default="",
        description="One sentence reasoning for current step.",
    )
    action: str | None = Field(
        default=None,
        description="Tool name from AVAILABLE TOOLS, or null if you have a final answer.",
    )
    action_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool action.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Final response to the user, or null if not yet done.",
    )

    @model_validator(mode="after")
    def validate_mutual_exclusivity(self) -> "ReActDecision":
        has_action = bool(self.action and self.action.strip())
        has_final = bool(self.final_answer and self.final_answer.strip())

        if has_action and has_final:
            raise ValueError(
                "Exactly one of 'action' or 'final_answer' must be provided, never both."
            )
        if not has_action and not has_final:
            raise ValueError(
                "Exactly one of 'action' or 'final_answer' must be provided, never neither."
            )

        if not has_action:
            self.action = None
        if not has_final:
            self.final_answer = None

        return self


# ══════════════════════════════════════════════════════════════════════════════
# SCRATCHPAD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScratchpadEntry:
    step:        int
    thought:     str
    tool_name:   str | None
    tool_args:   dict[str, Any] | None
    observation: str
    timestamp:   str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Scratchpad:
    def __init__(self, user_input: str) -> None:
        self.user_input = user_input
        self._entries: list[ScratchpadEntry] = []

    def add(
        self,
        step: int,
        thought: str,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
        observation: str,
    ) -> None:
        self._entries.append(
            ScratchpadEntry(
                step=step,
                thought=thought,
                tool_name=tool_name,
                tool_args=tool_args,
                observation=observation,
            )
        )

    def render_for_llm(self) -> str:
        if not self._entries:
            return "(no prior steps yet)"
        lines: list[str] = []
        for entry in self._entries:
            lines.append(f"Step {entry.step}:")
            lines.append(f"  Thought: {entry.thought}")
            if entry.tool_name:
                lines.append(f"  Action: {entry.tool_name}({entry.tool_args})")
            lines.append(f"  Observation: {entry.observation}")
        return "\n".join(lines)

    def step_count(self) -> int:
        return len(self._entries)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_react_system_prompt(tool_definitions: list[dict]) -> str:
    tools_json = json.dumps(tool_definitions, indent=2)
    return f"""You are the ReAct reasoning engine for CYRAX, an AI operating system.
You solve tasks through iterative Thought -> Action -> Observation cycles.

AVAILABLE TOOLS:
{tools_json}

OUTPUT FORMAT — CRITICAL:
Return ONLY a raw JSON object with exactly these keys:
{{
  "thought": "<your reasoning about what to do next, one sentence>",
  "action": "<tool name from AVAILABLE TOOLS, or null if you have a final answer>",
  "action_args": {{...}},
  "final_answer": "<your answer to the user, or null if not yet done>"
}}

RULES:
1. Exactly one of "action" or "final_answer" must be non-null, never both.
2. If you have enough information from prior Observations to answer the
   user, set "action" to null and provide "final_answer".
3. If you need another tool call, set "final_answer" to null and provide
   "action" + "action_args".
4. Never repeat an identical action+args pair that already failed in a
   prior step — the Observation will tell you it failed; try something else.
5. Tool names must exactly match AVAILABLE TOOLS.

No markdown, no code fences, no text outside the JSON object."""


def _build_schema_correction_prompt(bad_output: str, parse_error: str) -> str:
    example = '{"thought": "Decide next step", "action": null, "action_args": {}, "final_answer": "Here is the final answer."}'
    truncated = (bad_output or "").strip()[:800]
    return (
        f"Your previous response could not be parsed as valid JSON.\n\n"
        f"PARSE ERROR:\n{parse_error}\n\n"
        f"YOUR OUTPUT (truncated):\n{truncated}\n\n"
        "Return ONLY a corrected, valid JSON object with the exact keys:\n"
        '"thought", "action", "action_args", "final_answer".\n'
        "No markdown, no code fences, and no surrounding text.\n\n"
        "Example of the exact JSON shape (use this as a template):\n"
        f"{example}\n\n"
        "If you intend to take another action, set \"final_answer\" to null and fill\n"
        "\"action\" with the tool name and \"action_args\" with the args object.\n"
    )


def _format_tool_validation_error(tool_name: str, exc: Exception) -> str:
    """
    Formats Pydantic ValidationErrors or type errors into safe, structured,
    actionable observation messages for the LLM without leaking raw tracebacks
    or sensitive inputs.
    """
    if isinstance(exc, ValidationError):
        error_lines = []
        for err in exc.errors():
            loc = " -> ".join(str(p) for p in err.get("loc", [])) or "root"
            msg = err.get("msg", "invalid")
            error_lines.append(f"{loc}: {msg}.")
        details = "\n".join(error_lines)
        return (
            f"Tool {tool_name} validation failed:\n"
            f"{details}\n"
            f"No execution occurred.\n"
            f"Correct the arguments and retry."
        )
    return (
        f"Tool {tool_name} validation failed:\n"
        f"{str(exc)}\n"
        f"No execution occurred.\n"
        f"Correct the arguments and retry."
    )


def _format_unknown_tool_error(tool_name: str, valid_tools: set[str]) -> str:
    tools_list = ", ".join(sorted(valid_tools))
    return (
        f"Tool '{tool_name}' is not registered.\n"
        f"Available tools: {tools_list}\n"
        f"No execution occurred.\n"
        f"Choose a valid tool name from the AVAILABLE TOOLS list and retry."
    )


# ══════════════════════════════════════════════════════════════════════════════
# JSON NORMALIZATION & PARSER WITH TELEMETRY
# ══════════════════════════════════════════════════════════════════════════════

def _parse_decision_with_repairs(
    raw: str | None,
) -> tuple[ReActDecision | None, list[str]]:
    """
    Robustly parses raw model output into a ReActDecision, recording all repair
    transformations applied.
    """
    if raw is None:
        return None, []

    repairs: list[str] = []
    cleaned = raw if isinstance(raw, str) else json.dumps(raw)
    cleaned = cleaned.strip()

    if not cleaned:
        return None, []

    # 1. Strip Markdown code fences
    if cleaned.startswith("```"):
        repairs.append("MARKDOWN_FENCE_STRIPPED")
        lines = [l for l in cleaned.splitlines() if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    def _try_load(s: str) -> Any:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    parsed = _try_load(cleaned)

    # 2. Extract JSON object from surrounding prose if direct load failed
    if parsed is None:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            substr = cleaned[start : end + 1]
            parsed = _try_load(substr)
            if parsed is not None:
                repairs.append("PROSE_EXTRACTED")

    if parsed is None:
        return None, repairs

    # 3. Unwrap list of length 1
    if isinstance(parsed, list) and parsed:
        if isinstance(parsed[0], dict):
            repairs.append("LIST_UNWRAPPED")
            parsed = parsed[0]
        else:
            return None, repairs

    if not isinstance(parsed, dict):
        return None, repairs

    # 4. Normalize keys
    thought = parsed.get("thought")
    if thought is None and "Thought" in parsed:
        repairs.append("KEY_NORMALIZED")
        thought = parsed.get("Thought")
    thought_str = str(thought) if thought is not None else ""

    action = parsed.get("action")
    if action is None:
        for alt in ("Action", "action_name", "tool", "tool_name"):
            if alt in parsed:
                repairs.append("KEY_NORMALIZED")
                action = parsed.get(alt)
                break

    action_args = parsed.get("action_args")
    if action_args is None:
        for alt in ("actionArgs", "ActionArgs", "args", "parameters", "arguments"):
            if alt in parsed:
                repairs.append("KEY_NORMALIZED")
                action_args = parsed.get(alt)
                break

    if action_args is None:
        action_args = {}
    elif isinstance(action_args, str):
        # 5. Stringified JSON arguments
        try:
            loaded_args = json.loads(action_args)
            if isinstance(loaded_args, dict):
                repairs.append("ARGS_UNPACKED")
                action_args = loaded_args
        except Exception:
            pass

    final_answer = parsed.get("final_answer")
    if final_answer is None:
        for alt in ("finalAnswer", "FinalAnswer", "answer", "response"):
            if alt in parsed:
                repairs.append("KEY_NORMALIZED")
                final_answer = parsed.get(alt)
                break

    try:
        decision = ReActDecision(
            thought=thought_str,
            action=str(action) if action is not None and str(action).strip() else None,
            action_args=action_args if isinstance(action_args, dict) else {},
            final_answer=str(final_answer) if final_answer is not None and str(final_answer).strip() else None,
        )
        return decision, repairs
    except (ValidationError, ValueError):
        return None, repairs


# ══════════════════════════════════════════════════════════════════════════════
# PLANNER
# ══════════════════════════════════════════════════════════════════════════════

class Planner:
    """
    Sole reasoning and orchestration engine for tool-using requests.

    Produces validated actions and delegates execution to the ToolRegistry.
    Never executes tools or checks security clearance directly.
    """

    def __init__(self, ctx: Any, req_id: str, provider_name: str | None = None) -> None:
        self._ctx           = ctx
        self._req_id        = req_id
        self._provider_name = provider_name
        self._last_raw: str | None = None

    async def run(
        self,
        user_input:    str,
        history:       list[dict],
        cancel_token:  CancellationToken | None = None,
    ) -> dict[str, Any]:
        scratchpad = Scratchpad(user_input=user_input)
        tool_definitions = self._ctx.tool_registry.get_all_definitions()
        valid_tool_names = {t["name"] for t in tool_definitions}
        system_prompt = _build_react_system_prompt(tool_definitions)

        max_steps = getattr(settings, "MAX_AGENT_STEPS", 5)
        wall_clock_timeout = getattr(settings, "PLANNER_WALL_CLOCK_TIMEOUT_SECONDS", 60.0)
        start_time = time.monotonic()

        for step in range(1, max_steps + 1):

            # Check wall-clock timeout
            if time.monotonic() - start_time > wall_clock_timeout:
                logger.warning(
                    f"[{self._req_id}] PLANNER | Wall-clock timeout exceeded ({wall_clock_timeout}s)."
                )
                return {
                    "status": "error",
                    "response": f"System Error: Planner exceeded maximum execution time of {wall_clock_timeout}s.",
                    "error_code": "planner_wall_clock_timeout",
                }

            # Check cancellation before step
            if cancel_token is not None and cancel_token.is_cancelled:
                logger.info(
                    f"[{self._req_id}] PLANNER | Cancelled before step {step} | "
                    f"reason='{cancel_token.reason}'"
                )
                raise asyncio.CancelledError(cancel_token.reason)

            try:
                think_result = await self._think(
                    system_prompt=system_prompt,
                    scratchpad=scratchpad,
                    history=history,
                    step=step,
                    cancel_token=cancel_token,
                )
            except (GenerationCancelledError, asyncio.CancelledError):
                logger.info(f"[{self._req_id}] PLANNER | Cancellation propagated during think.")
                raise
            except ProviderCapabilityError as exc:
                logger.error(
                    f"[{self._req_id}] PLANNER | Capability exhaustion at step {step}: {exc}"
                )
                return {
                    "status": "error",
                    "response": "System Error: No eligible AI provider available for planning capabilities.",
                    "error_code": "provider_capability_exhaustion",
                }
            except ProviderError as exc:
                logger.error(
                    f"[{self._req_id}] PLANNER | Provider error at step {step}: {exc}"
                )
                if not exc.retryable:
                    return {
                        "status": "error",
                        "response": f"System Error: AI provider configuration error: {exc}",
                        "error_code": "provider_error",
                    }
                return {
                    "status": "error",
                    "response": "System Error: All AI providers are currently unavailable. Please try again shortly.",
                    "error_code": "provider_exhausted",
                }
            except asyncio.TimeoutError:
                logger.error(f"[{self._req_id}] PLANNER | Think step timed out at step {step}.")
                return {
                    "status": "error",
                    "response": f"System Error: Planning request timed out after {settings.BRAIN_TIMEOUT_SECONDS}s.",
                    "error_code": "planner_timeout",
                }

            if think_result is None:
                payload: dict[str, Any] = {
                    "status": "error",
                    "response": "System Error: Could not parse a valid reasoning step.",
                    "error_code": "schema_parse_failed",
                }
                try:
                    if self._last_raw and getattr(settings, "PLANNER_LOG_RAW_OUTPUT", False):
                        max_chars = getattr(settings, "PLANNER_RAW_OUTPUT_MAX_CHARS", 800)
                        payload["raw_output"] = (self._last_raw or "")[:max_chars]
                except Exception:
                    logger.debug("Failed to attach raw planner output to error payload.")
                return payload

            # Handle direct error payload from think if returned
            if isinstance(think_result, dict) and think_result.get("status") == "error":
                return think_result

            decision: ReActDecision = think_result

            # Final Answer reached
            if decision.final_answer is not None:
                logger.info(f"[{self._req_id}] PLANNER | Final answer at step {step}.")
                return {"status": "success", "response": decision.final_answer}

            tool_name: str = decision.action or ""
            tool_args: dict[str, Any] = decision.action_args or {}
            thought:   str = decision.thought

            # ── 1. Tool existence check ───────────────────────────────────────
            if not tool_name or tool_name not in valid_tool_names:
                observation = _format_unknown_tool_error(tool_name, valid_tool_names)
                scratchpad.add(step, thought, tool_name, tool_args, observation)
                continue

            # ── 2. Declarative Schema validation (WITHOUT BaseTool instance) ──
            args_schema = self._ctx.tool_registry.get_tool_schema(tool_name)
            if args_schema is not None:
                try:
                    if hasattr(args_schema, "model_validate"):
                        args_schema.model_validate(tool_args)
                    else:
                        args_schema(**tool_args)
                except (ValidationError, TypeError, ValueError) as exc:
                    observation = _format_tool_validation_error(tool_name, exc)
                    scratchpad.add(step, thought, tool_name, tool_args, observation)
                    continue

            # ── 3. Cancellation check before tool execution ───────────────────
            if cancel_token is not None and cancel_token.is_cancelled:
                logger.info(
                    f"[{self._req_id}] PLANNER | Cancelled before tool call "
                    f"at step {step} | reason='{cancel_token.reason}'"
                )
                raise asyncio.CancelledError(cancel_token.reason)

            # ── 4. Dispatch to tool execution layer ───────────────────────────
            observation, auth_required_result = await self._act(
                tool_name=tool_name, tool_args=tool_args, step=step,
            )

            if auth_required_result is not None:
                return auth_required_result

            scratchpad.add(step, thought, tool_name, tool_args, observation)

        logger.warning(
            f"[{self._req_id}] PLANNER | MAX_AGENT_STEPS ({max_steps}) exhausted."
        )
        return {
            "status": "error",
            "response": "System Error: Task exceeded maximum reasoning steps.",
            "error_code": "max_steps_exceeded",
        }

    # ── Think phase — Phase 8.4.5: routed through MoERouter.generate() ──────

    async def _think(
        self,
        system_prompt: str,
        scratchpad: Scratchpad,
        history: list[dict],
        step: int,
        cancel_token: CancellationToken | None = None,
    ) -> ReActDecision | dict[str, Any] | None:
        """
        Calls the router in STRUCTURED_JSON mode, parses the decision, and
        optionally attempts bounded schema-correction retries.
        """
        transcript = scratchpad.render_for_llm()
        recent_history = history[-4:]

        user_message = (
            f"User request: {scratchpad.user_input}\n\nPrior steps:\n{transcript}\n\n"
            f"What is your next thought and action?"
        )

        messages = recent_history + [{"role": "user", "content": user_message}]
        max_tokens = getattr(settings, "PLANNER_MAX_TOKENS", 2048)

        raw = await self._ctx.brain_router.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            json_mode=True,
            generation_mode=GenerationMode.STRUCTURED_JSON,
            provider_name=self._provider_name,
            intent="react_think",
            trace_id=self._req_id,
            cancel_token=cancel_token,
        )

        self._last_raw = raw

        decision, repairs = _parse_decision_with_repairs(raw)
        if decision is not None:
            if repairs:
                logger.warning(
                    f"[{self._req_id}] PLANNER | STRUCTURED_JSON repair applied | "
                    f"provider={self._provider_name or 'auto'} | "
                    f"mode=STRUCTURED_JSON | "
                    f"repairs={repairs} | "
                    f"success=True"
                )
            return decision

        if repairs:
            logger.warning(
                f"[{self._req_id}] PLANNER | STRUCTURED_JSON repair attempted but failed | "
                f"provider={self._provider_name or 'auto'} | "
                f"mode=STRUCTURED_JSON | "
                f"repairs={repairs} | "
                f"success=False"
            )

        # Record a parse failure metric before attempting correction retries
        try:
            with _parse_metrics_lock:
                global _parse_failures_counter
                _parse_failures_counter += 1
        except Exception:
            logger.debug("Failed to record parse failure metric.")

        max_retries = getattr(settings, "PLANNER_MAX_SCHEMA_RETRIES", 1)
        backoff_base = getattr(settings, "PLANNER_SCHEMA_RETRY_BACKOFF_SECONDS", 0.5)

        for attempt in range(1, max_retries + 1):
            if cancel_token is not None and cancel_token.is_cancelled:
                logger.info(
                    f"[{self._req_id}] PLANNER | Cancelled during schema retry {attempt} | "
                    f"reason='{cancel_token.reason}'"
                )
                raise asyncio.CancelledError(cancel_token.reason)

            logger.warning(
                f"[{self._req_id}] PLANNER | Schema error at step {step}, retry {attempt}/{max_retries}."
            )

            # Record a correction attempt metric
            try:
                with _parse_metrics_lock:
                    global _correction_attempts_counter
                    _correction_attempts_counter += 1
            except Exception:
                logger.debug("Failed to record correction attempt metric.")

            if getattr(settings, "PLANNER_LOG_RAW_OUTPUT", False):
                try:
                    max_chars = getattr(settings, "PLANNER_RAW_OUTPUT_MAX_CHARS", 800)
                    truncated = (raw or "")[:max_chars]
                    logger.warning(f"[{self._req_id}] PLANNER | Raw LLM output (truncated): {truncated}")
                except Exception:
                    logger.debug("Failed to emit raw planner output for diagnostics.")

            correction_prompt = _build_schema_correction_prompt(
                bad_output=raw, parse_error="Invalid JSON structure or missing required keys."
            )

            raw = await self._ctx.brain_router.generate(
                messages=[{"role": "user", "content": correction_prompt}],
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                json_mode=True,
                generation_mode=GenerationMode.STRUCTURED_JSON,
                provider_name=self._provider_name,
                intent="react_think_correction",
                trace_id=self._req_id,
                cancel_token=cancel_token,
            )

            self._last_raw = raw
            decision, repairs = _parse_decision_with_repairs(raw)
            if decision is not None:
                if repairs:
                    logger.warning(
                        f"[{self._req_id}] PLANNER | STRUCTURED_JSON repair applied during retry {attempt} | "
                        f"provider={self._provider_name or 'auto'} | "
                        f"mode=STRUCTURED_JSON | "
                        f"repairs={repairs} | "
                        f"success=True"
                    )
                return decision

            if repairs:
                logger.warning(
                    f"[{self._req_id}] PLANNER | STRUCTURED_JSON repair attempted during retry {attempt} but failed | "
                    f"provider={self._provider_name or 'auto'} | "
                    f"mode=STRUCTURED_JSON | "
                    f"repairs={repairs} | "
                    f"success=False"
                )

            # Backoff between attempts
            try:
                await asyncio.sleep(backoff_base * attempt)
            except asyncio.CancelledError:
                raise

        return None

    @staticmethod
    def _parse_decision(raw: str) -> dict[str, Any] | None:
        """
        Backwards-compatible parsing helper returning a normalized dict or None.
        """
        decision, _ = _parse_decision_with_repairs(raw)
        if decision is not None:
            return decision.model_dump()
        return None

    # ── Act phase (delegates tool execution) ──────────────────────────────────

    async def _act(
        self, tool_name: str, tool_args: dict[str, Any], step: int,
    ) -> tuple[str, dict[str, Any] | None]:
        result = await self._execute_step(tool_name, tool_args)

        if result["status"] == "auth_required":
            self._ctx.memory.state.set("pending_auth", True)
            self._ctx.memory.state.set("pending_tool", tool_name)
            self._ctx.memory.state.set("pending_args", tool_args)
            return "", {"status": "auth_required", "response": result["response"]}

        if result["status"] == "success":
            if tool_name == "OPEN_APP":
                app_name = tool_args.get("app", "")
                if app_name:
                    self._ctx.memory.state.set("last_opened_app", app_name)
            return result["response"], None

        fallback = self._ctx.fallback_policy.get_fallback(
            failed_tool=tool_name, error_status=result["status"], original_args=tool_args,
        )

        if fallback is not None:
            logger.info(
                f"[{self._req_id}] PLANNER | Tier 1 fallback at step {step}: "
                f"{tool_name} -> {fallback['tool']}"
            )
            fb_result = await self._execute_step(fallback["tool"], fallback["args"])

            if fb_result["status"] == "success":
                observation = (
                    f"Note: '{tool_name}' failed ({result['response'][:100]}). "
                    f"Automatically substituted '{fallback['tool']}' instead, "
                    f"which succeeded: {fb_result['response']}"
                )
            else:
                observation = (
                    f"Error: '{tool_name}' failed ({result['response'][:100]}). "
                    f"Fallback '{fallback['tool']}' also failed: {fb_result['response'][:100]}"
                )
            return observation, None

        observation = f"Error: {result['response'][:_MAX_ERROR_IN_OBSERVATION]}"
        return observation, None

    async def _execute_step(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                self._ctx.tool_registry.execute_tool(tool_name, args, ctx=self._ctx),
                timeout=settings.TOOL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "response": f"Tool '{tool_name}' timed out after {settings.TOOL_TIMEOUT_SECONDS}s.",
            }
        except Exception as exc:
            logger.exception(f"[{self._req_id}] PLANNER | Unexpected error in {tool_name}: {exc}")
            return {"status": "error", "response": f"Tool '{tool_name}' encountered an unexpected error."}
