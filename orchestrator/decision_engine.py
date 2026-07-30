"""
orchestrator/decision_engine.py — CYRAX 3.0 Phase 7 Cognitive Decision Engine

Standalone pre-processor. NOT yet wired into Dispatcher (Phase 7 Step 1
scope — wiring is a later step).

Pipeline (single Groq call, per Phase 7 Constraint):
    1. Intent classification into a fixed 11-category taxonomy.
    2. Hard boolean tool-necessity gate, derived from intent.
    3. Deterministic provider selection via a fixed intent->provider matrix
       (NOT decided by the LLM — computed in Python after classification,
       so a hallucinated provider name is structurally impossible).

Fail-safe direction: on any classification failure (provider error or
unparseable output), defaults to tools_required=True, intent=UNKNOWN,
selected_provider="groq" — fails toward MORE capability (let the
downstream Planner's ReAct loop reason about it) rather than silently
dropping a possibly-tool-needing request into pure chat.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from brain.provider_metrics import ProviderMetricsManager
from brain.provider_selector import ProviderSelector
from brain.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# INTENT TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

IntentCategory = Literal[
    "OS_CONTROL",
    "FILE_OPERATION",
    "WEB_RESEARCH",
    "LIVE_INFORMATION",
    "CODE_GENERATION",
    "CONTENT_WRITING",
    "GENERAL_CHAT",
    "REASONING",
    "MEMORY",
    "AUTOMATION",
    "UNKNOWN",
]

_VALID_INTENTS: frozenset[str] = frozenset(
    {
        "OS_CONTROL", "FILE_OPERATION", "WEB_RESEARCH", "LIVE_INFORMATION",
        "CODE_GENERATION", "CONTENT_WRITING", "GENERAL_CHAT", "REASONING",
        "MEMORY", "AUTOMATION", "UNKNOWN",
    }
)

# Intents that structurally never require an OS/file/tool action.
# Used only as the fail-safe fallback mapping if the LLM's own
# tools_required boolean is missing/malformed — the model's explicit
# boolean is trusted first; this is a backstop, not the primary signal.
_INTENTS_NEVER_NEEDING_TOOLS: frozenset[str] = frozenset(
    {"GENERAL_CHAT", "REASONING", "CONTENT_WRITING"}
)


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION MODE
# ══════════════════════════════════════════════════════════════════════════════

class ExecutionMode(str, Enum):
    IMMEDIATE   = "immediate"    # Simple chat, answer inline
    INTERACTIVE = "interactive"  # Fast OS/tool action, user waits briefly
    BACKGROUND  = "background"   # Long-running, task-queued, notified later


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class CognitiveRoutingDecision(BaseModel):
    """
    Structured output of DecisionEngine.classify().

    intent:            One of the 11 fixed categories. UNKNOWN is a valid,
                        explicit value — not an absence of a value.
    tools_required:     Hard boolean gate. True routes to the ReAct Planner;
                        False routes to direct conversational generation.
    selected_provider:  Provider chosen by ProviderSelector (adaptive,
                        intent-aware, health-weighted).
    execution_mode:     How the request should be executed:
                        immediate | interactive | background.
    confidence:         Model's self-reported confidence in the classification,
                        0.0-1.0. Informational only in this phase; not
                        consulted by any threshold logic yet (that is a
                        wiring-phase decision, out of scope for Step 1).
    reason:             One-sentence explanation for this routing decision.
    """
    intent:            IntentCategory
    tools_required:    bool
    selected_provider: str
    execution_mode:    ExecutionMode
    confidence:        float = Field(default=1.0, ge=0.0, le=1.0)
    reason:            str = Field(
        default="No reason provided.",
        description="One-sentence explanation for this routing decision.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def _build_classification_prompt() -> str:
    return """You are the Cognitive Decision Engine for CYRAX, an AI operating system.
Your ONLY job is to classify a user request into exactly one intent category,
decide whether it requires OS/tool actions, and select an execution mode.

OUTPUT FORMAT — CRITICAL:
Return ONLY a raw JSON object with exactly these keys:
{
  "intent": "<one of the categories below>",
  "tools_required": true or false,
  "execution_mode": "immediate" | "interactive" | "background",
  "confidence": <float 0.0 to 1.0>,
  "reason": "<one sentence explaining this classification>"
}
No markdown, no code fences, no text outside the JSON object.

INTENT CATEGORIES:
- OS_CONTROL:       Opening/closing apps, system power, volume, keyboard input.
- FILE_OPERATION:   Reading, writing, deleting, moving, copying files/directories.
- WEB_RESEARCH:      Searching the web, reading a specific webpage's content.
- LIVE_INFORMATION:  Needs current/real-time facts (weather, news, prices, scores).
- CODE_GENERATION:   Writing, explaining, or debugging code — no file save implied.
- CONTENT_WRITING:   Essays, emails, stories, summaries, creative or long-form text.
- GENERAL_CHAT:      Greetings, identity questions, small talk, opinions.
- REASONING:          Math, logic, multi-step analysis, planning a strategy.
- MEMORY:             Recalling or storing facts about the user long-term.
- AUTOMATION:         Scheduling a delayed task, recurring actions.
- UNKNOWN:             Genuinely ambiguous or doesn't fit any category above.

TOOLS_REQUIRED RULES:
- true  = the request needs an OS/file/web/scheduling ACTION performed.
- false = the request can be fully answered with generated text alone.

EXECUTION_MODE RULES:
- immediate:   Simple conversational answer, no tool call, near-instant.
               e.g. "who are you?", "what's 15% of 340?"
- interactive: A fast, single tool action the user is waiting on.
               e.g. "open notepad", "search the web for X", "close chrome"
- background:  Long-running or multi-step work the user should NOT wait
               synchronously for — heavy code generation, deep web
               crawls/multi-page research, scheduling/automation setup,
               or anything likely to take more than a few seconds.
               e.g. "crawl this site and summarize every page",
               "write a full REST API with tests", "schedule a daily report"

REASON RULE:
- Always provide a concise, single-sentence justification for your
  classification, e.g. "Programming request detected, requires large
  context reasoning." or "User asked to open an application, requires
  OS-level tool execution."

CRITICAL DISTINCTION — read carefully:
"Write a Python script" -> CODE_GENERATION, tools_required: false, execution_mode: immediate
  (short inline answer)
"Write a full REST API with authentication and tests" -> CODE_GENERATION, tools_required: false, execution_mode: background
  (same intent category, but large enough to warrant background execution)
"Write a Python script and save it to disk" -> CODE_GENERATION, tools_required: true, execution_mode: interactive
"Open Notepad" -> OS_CONTROL, tools_required: true, execution_mode: interactive
"Who are you?" -> GENERAL_CHAT, tools_required: false, execution_mode: immediate
"Search the web for RTX 5090 price" -> WEB_RESEARCH, tools_required: true, execution_mode: interactive
"Crawl this website and summarize every page" -> WEB_RESEARCH, tools_required: true, execution_mode: background
"Schedule a reminder in 10 minutes" -> AUTOMATION, tools_required: true, execution_mode: background
"Delete report.txt" -> FILE_OPERATION, tools_required: true, execution_mode: interactive

When genuinely unsure between two categories, prefer the one implying
tools_required: true — it is safer to over-route to the tool-capable
Planner than to silently answer conversationally when an action was wanted."""


# ══════════════════════════════════════════════════════════════════════════════
# DECISION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class DecisionEngine:
    """
    Standalone Phase 7 pre-processor. Not yet invoked by Dispatcher.

    Args:
        provider:          A BaseProvider instance (Groq, per Phase 7 Constraint).
        provider_selector: ProviderSelector for adaptive provider selection.
        metrics_manager:   ProviderMetricsManager for health/circuit state.
        active_providers:  List of provider names eligible for selection.
    """

    def __init__(
        self,
        provider:          BaseProvider,
        provider_selector: ProviderSelector,
        metrics_manager:   ProviderMetricsManager,
        active_providers:  list[str],
    ) -> None:
        self._provider          = provider
        self._selector          = provider_selector
        self._metrics           = metrics_manager
        self._active_providers  = active_providers

    async def classify(self, user_input: str) -> CognitiveRoutingDecision:
        """
        Runs the full classification pipeline in one Groq call.

        Fail-safe on any error (provider failure or unparseable output):
            intent=UNKNOWN, tools_required=True, selected_provider="groq"
            — fails toward more capability, not less. Never raises.
        """
        system_prompt = _build_classification_prompt()

        try:
            raw = await self._provider.generate(
                messages=[{"role": "user", "content": user_input}],
                system_prompt=system_prompt,
                max_tokens=180,
                temperature=0.0,
                json_mode=self._provider.capabilities.supports_json_mode,
            )
        except ProviderError as exc:
            logger.error(
                f"[DECISION_ENGINE] Provider call failed: {exc!r}. "
                f"Falling back to UNKNOWN/tools_required=True."
            )
            return self._fallback_decision()

        parsed = self._parse_and_validate(raw)
        if parsed is None:
            logger.warning(
                f"[DECISION_ENGINE] Unparseable classification output: "
                f"'{raw[:150]}'. Falling back to UNKNOWN/tools_required=True."
            )
            return self._fallback_decision()

        intent, tools_required, confidence, reason, execution_mode = parsed

        selected_provider, selection_trace = self._selector.select_provider(
            intent=intent,
            active_providers=self._active_providers,
            metrics=self._metrics,
        )

        logger.info(
            f"[DECISION_ENGINE] intent={intent} | "
            f"tools_required={tools_required} | "
            f"execution_mode={execution_mode.value} | "
            f"provider={selected_provider} | "
            f"confidence={confidence:.2f} | "
            f"reason='{reason}' | "
            f"selection_trace='{selection_trace}'"
        )

        return CognitiveRoutingDecision(
            intent=intent,
            tools_required=tools_required,
            selected_provider=selected_provider,  # type: ignore[arg-type]
            execution_mode=execution_mode,
            confidence=confidence,
            reason=reason,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_and_validate(
        raw: str,
) -> tuple[IntentCategory, bool, float, str, ExecutionMode] | None:
        """
        Parses and validates the raw LLM output.

Returns (intent, tools_required, confidence, reason, execution_mode)
        on success, None on any parse or validation failure — caller treats
        None as the signal to fall back.
        """
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

        intent_raw = str(parsed.get("intent", "")).strip().upper()
        if intent_raw not in _VALID_INTENTS:
            logger.warning(
                f"[DECISION_ENGINE] Model returned invalid intent "
                f"'{intent_raw}' — not in the fixed taxonomy."
            )
            return None

        tools_required_raw = parsed.get("tools_required")
        if not isinstance(tools_required_raw, bool):
            # Backstop: derive from the intent's structural nature rather
            # than failing the whole classification over a malformed bool.
            tools_required_raw = intent_raw not in _INTENTS_NEVER_NEEDING_TOOLS
            logger.debug(
                f"[DECISION_ENGINE] tools_required missing/malformed — "
                f"derived {tools_required_raw} from intent={intent_raw}."
            )

        confidence_raw = parsed.get("confidence", 1.0)
        try:
            confidence = float(confidence_raw)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 1.0

        reason = str(parsed.get("reason", "")).strip() or "No reason provided."

        execution_mode_raw = str(parsed.get("execution_mode", "")).strip().lower()
        try:
            execution_mode = ExecutionMode(execution_mode_raw)
        except ValueError:
            # Fail-safe: default to interactive (not immediate, not background) —
            # a mid-weight default that neither silently drops a tool action
            # (immediate would) nor queues a possibly-quick request unnecessarily
            # (background would).
            logger.warning(
                f"[DECISION_ENGINE] Invalid or missing execution_mode "
                f"'{execution_mode_raw}' — defaulting to INTERACTIVE."
            )
            execution_mode = ExecutionMode.INTERACTIVE

        return intent_raw, tools_required_raw, confidence, reason, execution_mode  # type: ignore[return-value]

    @staticmethod
    def _fallback_decision() -> CognitiveRoutingDecision:
        """
        Fail-safe default. Asymmetric by design: fails toward tools_required
        =True (more capability), not False (silent conversational answer to
        a possibly action-requiring request).
        """
        return CognitiveRoutingDecision(
            intent="UNKNOWN",
            tools_required=True,
            selected_provider="groq",
            execution_mode=ExecutionMode.INTERACTIVE,
            confidence=0.0,
            reason="Classification failed; defaulting to safe fallback.",
        )
