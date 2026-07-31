"""
orchestrator/planner.py — CYRAX 3.0 Phase 8.4.5 ReAct Planner

Phase 8.4.5 change: _think() no longer calls provider.generate() directly.
It now calls self._ctx.brain_router.generate(...) — MoERouter's failover-
wrapped equivalent — so a 429/retryable error during Planner's Think phase
gets automatic circuit-breaker-aware failover instead of crashing the task.
_select_provider() is removed; provider_name is passed straight through
to brain_router.generate() as the requested candidate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config.settings import settings
from core.interrupt_controller import CancellationToken

logger = logging.getLogger(__name__)

MAX_REACT_STEPS: int = 5
MAX_SCHEMA_RETRIES: int = 1
_MAX_ERROR_IN_OBSERVATION: int = 400


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
        self, step: int, thought: str, tool_name: str | None,
        tool_args: dict[str, Any] | None, observation: str,
    ) -> None:
        self._entries.append(
            ScratchpadEntry(
                step=step, thought=thought, tool_name=tool_name,
                tool_args=tool_args, observation=observation,
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
    return f"""Your previous response could not be parsed as valid JSON.

YOUR OUTPUT:
{bad_output[:500]}

PARSE ERROR:
{parse_error}

Return ONLY a corrected, valid JSON object with keys "thought", "action",
"action_args", "final_answer" — exactly as specified before. No markdown,
no explanation, just the corrected JSON object."""


class Planner:
    """
    Sole execution engine for tool-using requests.

    Args:
        ctx:            CyraxContext.
        req_id:          Trace correlation ID.
        provider_name:   Optional. When set (Decision Engine override),
                         passed through to brain_router.generate() as the
                         requested first candidate — MoERouter's own
                         failover logic decides what happens if that
                         candidate fails, not Planner.
    """

    def __init__(self, ctx: Any, req_id: str, provider_name: str | None = None) -> None:
        self._ctx           = ctx
        self._req_id        = req_id
        self._provider_name = provider_name

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

        for step in range(1, MAX_REACT_STEPS + 1):

            if cancel_token is not None and cancel_token.is_cancelled:
                logger.info(
                    f"[{self._req_id}] PLANNER | Cancelled before step {step} "
                    f"| reason='{cancel_token.reason}'"
                )
                raise asyncio.CancelledError(cancel_token.reason)

            decision = await self._think(
                system_prompt=system_prompt, scratchpad=scratchpad,
                history=history, step=step,
            )

            if decision is None:
                return {
                    "status": "error",
                    "response": "System Error: Could not parse a valid reasoning step.",
                }

            if decision.get("final_answer") is not None:
                logger.info(f"[{self._req_id}] PLANNER | Final answer at step {step}.")
                return {"status": "success", "response": str(decision["final_answer"])}

            tool_name: str = decision.get("action") or ""
            tool_args: dict = decision.get("action_args") or {}
            thought:   str = decision.get("thought", "")

            if not tool_name or tool_name not in valid_tool_names:
                observation = (
                    f"Error: '{tool_name}' is not a valid tool name. "
                    f"Choose from the AVAILABLE TOOLS list."
                )
                scratchpad.add(step, thought, tool_name, tool_args, observation)
                continue

            if cancel_token is not None and cancel_token.is_cancelled:
                logger.info(
                    f"[{self._req_id}] PLANNER | Cancelled before tool call "
                    f"at step {step} | reason='{cancel_token.reason}'"
                )
                raise asyncio.CancelledError(cancel_token.reason)

            observation, auth_required_result = await self._act(
                tool_name=tool_name, tool_args=tool_args, step=step,
            )

            if auth_required_result is not None:
                return auth_required_result

            scratchpad.add(step, thought, tool_name, tool_args, observation)

        logger.warning(f"[{self._req_id}] PLANNER | MAX_REACT_STEPS ({MAX_REACT_STEPS}) exhausted.")
        return {
            "status": "error",
            "response": "System Error: Task exceeded maximum reasoning steps.",
        }

    # ── Think phase — Phase 8.4.5: routed through MoERouter.generate() ──────

    async def _think(
        self, system_prompt: str, scratchpad: Scratchpad, history: list[dict], step: int,
    ) -> dict[str, Any] | None:
        transcript = scratchpad.render_for_llm()
        recent_history = history[-4:]

        user_message = (
            f"User request: {scratchpad.user_input}\n\n"
            f"Prior steps:\n{transcript}\n\n"
            f"What is your next thought and action?"
        )

        messages = recent_history + [{"role": "user", "content": user_message}]

        # Routed through MoERouter.generate() — gains circuit-breaker
        # awareness and automatic failover on a retryable ProviderError,
        # instead of calling a single provider's generate() directly.
        raw = await self._ctx.brain_router.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=400,
            temperature=0.0,
            json_mode=True,
            provider_name=self._provider_name,
            intent="react_think",
            trace_id=self._req_id,
        )

        parsed = self._parse_decision(raw)
        if parsed is not None:
            return parsed

        for attempt in range(1, MAX_SCHEMA_RETRIES + 1):
            logger.warning(
                f"[{self._req_id}] PLANNER | Schema error at step {step}, "
                f"retry {attempt}/{MAX_SCHEMA_RETRIES}."
            )
            correction_prompt = _build_schema_correction_prompt(
                bad_output=raw, parse_error="Invalid JSON structure or missing keys."
            )
            raw = await self._ctx.brain_router.generate(
                messages=[{"role": "user", "content": correction_prompt}],
                system_prompt=system_prompt,
                max_tokens=400,
                temperature=0.0,
                json_mode=True,
                provider_name=self._provider_name,
                intent="react_think_correction",
                trace_id=self._req_id,
            )
            parsed = self._parse_decision(raw)
            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _parse_decision(raw: str) -> dict[str, Any] | None:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = [l for l in cleaned.splitlines() if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        if "action" not in parsed or "final_answer" not in parsed:
            return None
        return parsed

    # ── Act phase (unchanged) ────────────────────────────────────────────────

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
