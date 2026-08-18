"""
brain/moe_router.py — CYRAX 3.0 Mixture-of-Experts Brain Router

Implements BrainRouterProtocol (core/context.py):
    async def route(messages, trace_id) -> str
    async def plan(user_input, history, tool_definitions, trace_id) -> list[dict]
    async def chat(user_input, history, trace_id, provider_name=None) -> str

Routing strategy (settings.ACTIVE_LLM):
    "auto"   — classifier selects groq or gemini based on task complexity.
    "groq"   — always route to Groq.
    "gemini" — always route to Gemini.
    "ollama" — always route to Ollama (local inference).

plan() contract:
    - Returns [] for any conversational, identity, or ambiguous input.
    - Only returns a non-empty list for unambiguous OS action commands.
    - Uses temperature=0.0 and json_mode=True for deterministic output.
    - Validates every step: tool name must exist in registry, args must be dict.

chat() contract:
    - Handles all inputs that plan() returns [] for.
    - Uses persona-injected system prompt.
    - Supports provider_name override (Phase 7 — Decision Engine routing).
    - Built-in failover telemetry with [FAILOVER] log tags.
    - Returns sanitised non-empty string.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from brain.provider_events import ProviderExecutionEvent
from brain.provider_metrics import ProviderMetricsManager
from brain.providers.base import BaseProvider, ProviderError, ProviderCapabilityError, GenerationCancelledError, GenerationMode
from brain.model_registry import registry, ModelMetadata, ModelCapabilities
from config.settings import settings, ActiveLLM

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

def _build_planner_system_prompt(tool_definitions: list[dict]) -> str:
    """
    Constructs the planning system prompt from live tool definitions.
    Forces root-level JSON object {"plan": [...]} to comply with Groq JSON mode.
    """
    tools_json = json.dumps(tool_definitions, indent=2)

    return f"""You are the Action Planner for CYRAX, an AI operating system.
Your ONLY job is to map explicit OS action commands to tools.
You are NOT a conversational assistant. You do NOT answer questions.

AVAILABLE TOOLS:
{tools_json}

══════════════════════════════════════════════════════
CRITICAL RULES — VIOLATING THESE IS A SYSTEM FAILURE
══════════════════════════════════════════════════════

RULE 1 — OUTPUT FORMAT (CRITICAL)
You MUST output a valid JSON object with a single key "plan" containing an array.
No markdown. No explanation.
Correct empty format: {{"plan": []}}
Correct action format: {{"plan": [{{"tool": "...", "args": {{...}}}}]}}

RULE 2 — TOOL NAMES
Every "tool" value MUST exactly match a name from AVAILABLE TOOLS.

RULE 3 — RETURN EMPTY PLAN FOR CONVERSATION
You MUST return {{"plan": []}} when:
  - The input is a greeting: "hello", "hi", "hey"
  - The input is an identity question: "who are you", "what are you"
  - The input is a general question: "what is X", "tell me about"
  - The input is conversational filler or ambiguous.

RULE 4 — ONLY RETURN ACTIONS FOR OS COMMANDS
Only populate the "plan" array if the user explicitly requests an OS action that matches an available tool.

══════════════════════════════════════════════════════
FEW-SHOT EXAMPLES
══════════════════════════════════════════════════════

INPUT: "open chrome"
OUTPUT: {{"plan": [{{"tool": "OPEN_APP", "args": {{"app": "chrome"}}}}]}}

INPUT: "search for RTX 5090 price"
OUTPUT: {{"plan": [{{"tool": "WEB_SEARCH", "args": {{"query": "RTX 5090 price"}}}}]}}

INPUT: "open chrome and search RTX 5090"
OUTPUT: {{"plan": [{{"tool": "OPEN_APP", "args": {{"app": "chrome"}}}}, {{"tool": "WEB_SEARCH", "args": {{"query": "RTX 5090"}}}}]}}

INPUT: "volume up"
OUTPUT: {{"plan": [{{"tool": "MEDIA_CONTROL", "args": {{"action": "volume_up"}}}}]}}

INPUT: "hello"
OUTPUT: {{"plan": []}}

INPUT: "who are you?"
OUTPUT: {{"plan": []}}

INPUT: "what's the weather today?"
OUTPUT: {{"plan": []}}

══════════════════════════════════════════════════════
When in doubt: return {{"plan": []}}. The chat system handles everything else.
══════════════════════════════════════════════════════"""


def _build_chat_system_prompt(active_model: str) -> str:
    """Constructs the conversational system prompt with persona overlay."""
    persona = _load_user_persona()
    return f"""=== CYRAX CORE DIRECTIVE ===
You are CYRAX, an advanced autonomous AI operating system.
You are NOT a generic cloud chatbot. You are the user's direct interface to their machine.

[CORE TRAITS]
1. Tone: Direct, competent, slightly technical. No fluff.
2. Identity: Never say "As an AI" or "I am a language model". You are CYRAX.
3. Honesty: If you lack data, say so. Never hallucinate.
4. Brevity: Respond with the minimum words needed. Be precise.
5. Backend: If asked about your infrastructure, you are routed through '{active_model}'.
{persona}

[CONFLICT RESOLUTION]
If the User Persona conflicts with these Core Traits, the Core Directive overrides it.
Adopt the persona's tone and style while preserving identity and accuracy."""


def _load_user_persona() -> str:
    """Loads user_persona.txt from config/. Returns empty string if missing."""
    from pathlib import Path
    persona_path = (
        Path(__file__).resolve().parent.parent / "config" / "user_persona.txt"
    )
    try:
        content = persona_path.read_text(encoding="utf-8").strip()
        if content:
            return f"\n[USER PERSONA OVERLAY]\n{content}"
    except FileNotFoundError:
        pass
    return ""


def _build_classifier_system_prompt() -> str:
    return """You are a routing classifier for an AI system.
Analyse the user message and decide which provider should handle it.
OUTPUT: a single JSON object — {"model": "groq"} or {"model": "gemini"}

RULES:
- "gemini": code generation, debugging, complex reasoning, math,
  architectural planning, data analysis, long technical writing.
- "groq": greetings, conversational chat, simple questions, general knowledge,
  short tasks, persona interaction, anything not requiring deep reasoning.

Return ONLY the JSON object. Nothing else."""


# ══════════════════════════════════════════════════════════════════════════════
# PLAN PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _parse_plan(
    raw:              str,
    valid_tool_names: set[str],
    trace_id:         str,
) -> list[dict[str, Any]]:
    """
    Safely parses the LLM planner's JSON response into a validated step list.
    Returns [] on any parse or validation failure.
    """
    if not raw or not raw.strip():
        return []

    cleaned = raw.strip()

    # Strip markdown code fences.
    if cleaned.startswith("```"):
        lines = [
            line for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        ]
        cleaned = "\n".join(lines).strip()

    # Attempt JSON parse.
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            f"[{trace_id}] PLAN_PARSE_ERROR | "
            f"JSONDecodeError: {exc} | raw='{raw[:200]}'"
        )
        return []

    # Normalise to list.
    if isinstance(parsed, dict):
        for key in ("plan", "steps", "actions", "tools"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]

    if not isinstance(parsed, list):
        logger.warning(
            f"[{trace_id}] PLAN_PARSE_ERROR | "
            f"Expected list, got {type(parsed).__name__}"
        )
        return []

    # Empty array is valid — means "use chat fallback".
    if len(parsed) == 0:
        return []

    # Validate each step.
    clean_plan: list[dict[str, Any]] = []
    for step in parsed:
        if not isinstance(step, dict):
            continue

        tool_name = (step.get("tool") or step.get("intent", "")).strip()
        args      = step.get("args", {})

        if not tool_name:
            logger.debug(f"[{trace_id}] PLAN_STEP_SKIP | missing tool name")
            continue

        if tool_name not in valid_tool_names:
            logger.warning(
                f"[{trace_id}] PLAN_STEP_SKIP | "
                f"unknown tool '{tool_name}'"
            )
            continue

        if not isinstance(args, dict):
            args = {}

        clean_plan.append({"tool": tool_name, "args": args})

    if len(clean_plan) > settings.MAX_AGENT_STEPS:
        logger.warning(
            f"[{trace_id}] PLAN_TRUNCATED | "
            f"{len(clean_plan)} → {settings.MAX_AGENT_STEPS}"
        )
        clean_plan = clean_plan[: settings.MAX_AGENT_STEPS]

    return clean_plan


# ══════════════════════════════════════════════════════════════════════════════
# MOE ROUTER
# ══════════════════════════════════════════════════════════════════════════════

class MoERouter:
    """
    Mixture-of-Experts brain router.

    Satisfies BrainRouterProtocol (core/context.py):
        async def route(messages, trace_id) -> str
        async def plan(user_input, history, tool_definitions, trace_id) -> list[dict]
        async def chat(user_input, history, trace_id, provider_name=None) -> str

    Injected with pre-initialised provider instances by bootstrap().
    Never imports or instantiates providers directly.
    """

    def __init__(
        self,
        providers:         dict[str, BaseProvider],
        metrics_manager:    ProviderMetricsManager,
        resource_manager:   Any,  # core.resource_manager.ResourceManager
    ) -> None:
        if not providers:
            raise ValueError(
                "MoERouter requires at least one provider. "
                "Received empty providers dict."
            )
        self._providers        = providers
        self._metrics          = metrics_manager
        self._resource_manager = resource_manager
        
        # Populate the model registry
        for name, provider in providers.items():
            p_cap = getattr(provider, "capabilities", None)
            if p_cap:
                m_cap = ModelCapabilities(
                    supports_text=getattr(p_cap, "text", True),
                    supports_structured_json=getattr(p_cap, "structured_json", False),
                    supports_streaming=getattr(p_cap, "streaming", False),
                    supports_cancellation=getattr(p_cap, "cancellation", False),
                    supports_tools=getattr(p_cap, "supports_tool_schemas", False),
                    supports_system_prompt=getattr(p_cap, "supports_system_prompt", True),
                    context_tokens=getattr(p_cap, "context_window_tokens", 8192),
                    output_tokens=getattr(p_cap, "max_output_tokens", 4096),
                )
                meta = ModelMetadata(
                    provider=name,
                    model_id=getattr(provider, "_model", "unknown"),
                    capabilities=m_cap
                )
                registry.register(meta)

        logger.info(
            f"[MOE_ROUTER] Initialised. "
            f"Providers: {list(providers.keys())} | "
            f"ACTIVE_LLM: {settings.ACTIVE_LLM.value}"
        )

    # ── BrainRouterProtocol ───────────────────────────────────────────────────

    async def route(
        self,
        messages:  list[dict],
        trace_id:  str,
    ) -> str:
        """Raw generation. Caller manages the full message list."""
        provider = await self._select_provider(task_hint="chat", trace_id=trace_id)
        system_prompt = _build_chat_system_prompt(provider.provider_name)
        return await self._generate_with_failover(
            provider=provider,
            messages=messages,
            system_prompt=system_prompt,
            trace_id=trace_id,
        )

    async def plan(
        self,
        user_input:       str,
        history:          list[dict],
        tool_definitions: list[dict],
        trace_id:         str,
    ) -> list[dict[str, Any]]:
        """
        Generates a validated tool execution plan.

        Always uses the fastest available provider (Groq) with
        temperature=0.0 for deterministic output.

        Returns [] when:
          - No tool definitions are registered
          - Input is conversational, a greeting, or an identity question
          - The LLM planner explicitly returns []
          - JSON parsing or step validation fails
        """
        if not tool_definitions:
            logger.warning(f"[{trace_id}] PLAN | No tool definitions — returning []")
            return []

        valid_tool_names: set[str] = {t["name"] for t in tool_definitions}
        system_prompt = _build_planner_system_prompt(tool_definitions)

        # Include only recent history — planner needs minimal context.
        messages = history[-4:] + [{"role": "user", "content": user_input}]

        candidates = self._build_failover_candidates(None, output_format=GenerationMode.STRUCTURED_JSON, require_tools=True)
        if not candidates:
            logger.error(f"[{trace_id}] PLAN | Capability exhaustion: No providers support STRUCTURED_JSON + TOOLS.")
            raise ProviderCapabilityError(
                "No eligible providers available for STRUCTURED_JSON + TOOLS planner request."
            )

        logger.info(
            f"[{trace_id}] PLAN | "
            f"candidates={candidates} | "
            f"tools={len(tool_definitions)} | "
            f"input='{user_input[:80]}'"
        )

        raw: str | None = None
        for index, candidate_name in enumerate(candidates):
            provider = self._providers.get(candidate_name)
            if provider is None:
                continue

            try:
                gen_mode = GenerationMode.STRUCTURED_JSON if provider.capabilities.structured_json else GenerationMode.TEXT
                logger.debug(
                    f"[{trace_id}] PLAN | provider={candidate_name} | requested_generation_mode={gen_mode.value} | supports_json_mode={provider.capabilities.supports_json_mode}"
                )
                raw = await provider.generate(
                    messages=messages,
                    system_prompt=system_prompt,
                    max_tokens=600,
                    temperature=0.0,
                    generation_mode=gen_mode,
                )
                try:
                    self._metrics.record_event(ProviderExecutionEvent(
                        provider=candidate_name,
                        intent="planner",
                        latency_ms=0.0,
                        success=True,
                        status_code=200,
                    ))
                except Exception:
                    logger.debug("Failed to record PLAN success metric.")
                break
            except ProviderError as exc:
                logger.error(
                    f"[{trace_id}] PLAN_PROVIDER_ERROR | provider={candidate_name} | {exc!r}"
                )
                try:
                    self._metrics.record_event(ProviderExecutionEvent(
                        provider=candidate_name,
                        intent="planner",
                        latency_ms=0.0,
                        success=False,
                        status_code=getattr(exc, "status_code", 0),
                    ))
                except Exception:
                    logger.debug("Failed to record PLAN_PROVIDER_ERROR metric.")

                if not exc.retryable:
                    logger.error(f"[{trace_id}] PLAN | Non-retryable error from '{candidate_name}'. Halting plan failover.")
                    break
                continue

        if raw is None:
            return []

        plan = _parse_plan(raw, valid_tool_names, trace_id)

        logger.debug(
            f"[{trace_id}] PLAN_RESULT | "
            f"steps={len(plan)} | "
            f"tools={[s['tool'] for s in plan]}"
        )
        return plan

    async def chat(
        self,
        user_input:    str,
        history:       list[dict],
        trace_id:      str,
        provider_name: str | None = None,
        intent:        str = "unknown",
        cancel_token:  "CancellationToken | None" = None,
    ) -> str:
        """
        Generates a conversational response, attempting provider_name first
        (if supplied) and failing over through the remaining providers on
        a retryable ProviderError.

        Failover telemetry: every switch is logged at WARNING with the
        [FAILOVER] tag, the failed provider, its status code, and the
        provider being switched to — all in one grep-able line.

        Non-retryable errors (retryable=False) do NOT trigger failover —
        they are treated as a hard stop and re-raised as a synthesized
        error string, since silently falling back on e.g. an auth failure
        could mask real misconfiguration behind an apparently-working
        response from a different provider.

        Phase 7.2: each generate() call is timed and recorded as a
        ProviderExecutionEvent via ProviderMetricsManager.
        """
        # P3 technical debt: chat historically streamed if cancel_token was present. We preserve this requirement.
        candidates = self._build_failover_candidates(
            provider_name, 
            output_format=GenerationMode.TEXT,
            require_cancellation=(cancel_token is not None),
            require_streaming=(cancel_token is not None) 
        )
        if not candidates:
            logger.error(f"[{trace_id}] MOE_ROUTER | chat(): Capability exhaustion for TEXT + CANCELLATION.")
            raise ProviderCapabilityError("No eligible providers available for chat request.")

        system_prompt = _build_chat_system_prompt(provider_name or "groq")
        messages = history + [{"role": "user", "content": user_input}]

        last_error: ProviderError | None = None

        for index, candidate_name in enumerate(candidates):
            provider = self._providers.get(candidate_name)
            if provider is None:
                logger.warning(
                    f"[{trace_id}] MOE_ROUTER | "
                    f"Candidate provider '{candidate_name}' not registered. Skipping."
                )
                continue

            semaphore = self._resource_manager.get_provider_semaphore(candidate_name)
            start = time.monotonic()

            async with semaphore:
                try:
                    response = await provider.generate(
                        messages=messages,
                        system_prompt=system_prompt,
                        max_tokens=800,
                        temperature=0.7,
                        generation_mode=GenerationMode.TEXT,
                        cancel_token=cancel_token,
                    )
                    latency_ms = (time.monotonic() - start) * 1000

                    self._metrics.record_event(ProviderExecutionEvent(
                        provider=candidate_name, intent=intent, latency_ms=latency_ms,
                        success=True, status_code=200,
                    ))

                    if index > 0:
                        logger.info(
                            f"[{trace_id}] MOE_ROUTER | "
                            f"Failover recovery succeeded via '{candidate_name}' "
                            f"after {index} prior failure(s)."
                        )
                    return _sanitize_output(response)

                except TypeError as exc:
                    # Contract violation: a registered provider does not
                    # accept cancel_token (or another BaseProvider param).
                    # This must never happen post-audit, but is caught
                    # defensively so a future provider addition fails
                    # loud-but-sanitized instead of leaking a raw TypeError
                    # to the API response.
                    logger.error(
                        f"[{trace_id}] MOE_ROUTER | PROVIDER CONTRACT VIOLATION: "
                        f"'{candidate_name}'.generate() raised TypeError: {exc}. "
                        f"This provider does not conform to BaseProvider's signature."
                    )
                    raise ProviderError(
                        f"Provider '{candidate_name}' has an incompatible interface.",
                        provider=candidate_name, status_code=0, retryable=False,
                    ) from exc

                except GenerationCancelledError:
                    # Do NOT failover on cancellation — propagate immediately.
                    logger.info(f"[{trace_id}] MOE_ROUTER | Generation cancelled during chat() on '{candidate_name}'.")
                    raise

                except ProviderError as exc:
                    latency_ms = (time.monotonic() - start) * 1000
                    last_error = exc

                    self._metrics.record_event(ProviderExecutionEvent(
                        provider=candidate_name, intent=intent, latency_ms=latency_ms,
                        success=False, status_code=exc.status_code,
                    ))

                    if not exc.retryable:
                        logger.error(
                            f"[{trace_id}] MOE_ROUTER | "
                            f"Non-retryable error from '{candidate_name}' "
                            f"(status={exc.status_code}): {exc}. Hard stop — no failover."
                        )
                        return (
                            f"I couldn't complete that request due to a configuration "
                            f"issue with the '{candidate_name}' provider ({exc})."
                        )

                    next_candidate = candidates[index + 1] if index + 1 < len(candidates) else None

                    if next_candidate:
                        logger.warning(
                            f"[FAILOVER] [{trace_id}] "
                            f"Provider '{candidate_name}' failed "
                            f"(status={exc.status_code}, retryable=True): {exc}. "
                            f"Switching to '{next_candidate}'."
                        )
                    else:
                        logger.error(
                            f"[FAILOVER] [{trace_id}] "
                            f"Provider '{candidate_name}' failed "
                            f"(status={exc.status_code}, retryable=True): {exc}. "
                            f"No further providers available — failover chain exhausted."
                        )

        logger.error(
            f"[{trace_id}] MOE_ROUTER | "
            f"All providers exhausted. Last error: {last_error!r}"
        )
        raise ProviderCapabilityError(
            f"All AI providers are currently unavailable. Last error: {last_error}"
        )

    # ── Phase 8.4.5: generate() — failover-wrapped provider call ────────────

    async def generate(
        self,
        messages:       list[dict],
        system_prompt:  str,
        *,
        max_tokens:     int = 800,
        temperature:    float = 0.7,
        generation_mode: "GenerationMode | str" = GenerationMode.TEXT,
        json_mode:      bool = False,
        provider_name:  str | None = None,
        intent:         str = "unknown",
        trace_id:       str = "",
        cancel_token:   "CancellationToken | None" = None,
    ) -> str:
        """
        Failover-wrapped equivalent of BaseProvider.generate(). Used by
        Planner instead of calling a provider's generate() directly, so
        Planner's ReAct Think calls get circuit-breaker protection and
        automatic failover on a retryable ProviderError (e.g. 429),
        exactly as chat() already provides for conversational turns.

        Raises:
            Nothing. Same never-raises contract as chat() — on total
            failover exhaustion, returns a plain-string error message
            rather than propagating a ProviderError, since Planner's
            _parse_decision() expects a string it can attempt to parse
            as JSON (a graceful-degradation string will simply fail
            parsing and trigger Planner's existing schema-retry path,
            rather than crashing the task outright).
        """
        # Filter candidates by the requested generation_mode capability so we don't
        # ask providers to do work they don't claim to support.
        try:
            gen_mode = generation_mode if isinstance(generation_mode, GenerationMode) else GenerationMode(generation_mode)
        except Exception:
            gen_mode = GenerationMode.TEXT
        candidates = self._build_failover_candidates(
            provider_name, 
            output_format=gen_mode,
            require_cancellation=(cancel_token is not None)
            # require_streaming is intentionally not tied to cancel_token here
        )
        if not candidates:
            logger.error(
                f"[{trace_id}] MOE_ROUTER | generate(): Capability exhaustion. "
                f"No providers support format={gen_mode.value}, "
                f"cancellation={cancel_token is not None}."
            )
            raise ProviderCapabilityError(
                f"No providers available for capability request (format={gen_mode.value}, "
                f"cancellation={cancel_token is not None})."
            )

        last_error: ProviderError | None = None
        tried_providers: list[str] = []

        for index, candidate_name in enumerate(candidates):
            tried_providers.append(candidate_name)
            provider = self._providers.get(candidate_name)
            if provider is None:
                logger.warning(
                    f"[{trace_id}] MOE_ROUTER | "
                    f"Candidate provider '{candidate_name}' not registered. Skipping."
                )
                continue

            semaphore = self._resource_manager.get_provider_semaphore(candidate_name)
            start = time.monotonic()

            async with semaphore:
                try:
                    response = await provider.generate(
                        messages=messages,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        generation_mode=gen_mode,
                        json_mode=json_mode and provider.capabilities.supports_json_mode,
                        cancel_token=cancel_token,
                    )
                    latency_ms = (time.monotonic() - start) * 1000

                    self._metrics.record_event(ProviderExecutionEvent(
                        provider=candidate_name, intent=intent, latency_ms=latency_ms,
                        success=True, status_code=200,
                    ))

                    if index > 0:
                        logger.info(
                            f"[{trace_id}] MOE_ROUTER | "
                            f"generate() failover recovery succeeded via "
                            f"'{candidate_name}' after {index} prior failure(s)."
                        )
                    return response

                except TypeError as exc:
                    # Contract violation: a registered provider does not
                    # accept cancel_token (or another BaseProvider param).
                    # This must never happen post-audit, but is caught
                    # defensively so a future provider addition fails
                    # loud-but-sanitized instead of leaking a raw TypeError
                    # to the API response.
                    logger.error(
                        f"[{trace_id}] MOE_ROUTER | PROVIDER CONTRACT VIOLATION: "
                        f"'{candidate_name}'.generate() raised TypeError: {exc}. "
                        f"This provider does not conform to BaseProvider's signature."
                    )
                    raise ProviderError(
                        f"Provider '{candidate_name}' has an incompatible interface.",
                        provider=candidate_name, status_code=0, retryable=False,
                    ) from exc

                except GenerationCancelledError:
                    # Do NOT failover on cancellation — propagate immediately.
                    logger.info(f"[{trace_id}] MOE_ROUTER | Generation cancelled during generate() on '{candidate_name}'.")
                    raise

                except ProviderError as exc:
                    latency_ms = (time.monotonic() - start) * 1000
                    last_error = exc

                    self._metrics.record_event(ProviderExecutionEvent(
                        provider=candidate_name, intent=intent, latency_ms=latency_ms,
                        success=False, status_code=exc.status_code,
                    ))

                    if not exc.retryable:
                        logger.error(
                            f"[{trace_id}] MOE_ROUTER | "
                            f"generate(): non-retryable error from '{candidate_name}' "
                            f"(status={exc.status_code}): {exc}. Hard stop — no failover."
                        )
                        return (
                            f'{{"thought": "Provider error", "action": null, '
                            f'"action_args": {{}}, "final_answer": '
                            f'"I hit a configuration issue with the \'{candidate_name}\' provider and cannot continue."}}'
                        )

                    next_candidate = candidates[index + 1] if index + 1 < len(candidates) else None

                    if next_candidate:
                        logger.warning(
                            f"[FAILOVER] [{trace_id}] "
                            f"generate(): provider '{candidate_name}' failed "
                            f"(status={exc.status_code}, retryable=True): {exc}. "
                            f"Switching to '{next_candidate}'."
                        )
                    else:
                        logger.error(
                            f"[FAILOVER] [{trace_id}] "
                            f"generate(): provider '{candidate_name}' failed "
                            f"(status={exc.status_code}, retryable=True): {exc}. "
                            f"No further providers available — failover chain exhausted."
                        )

        logger.error(
            f"[{trace_id}] MOE_ROUTER | "
            f"generate(): all providers exhausted. Last error: {last_error!r}"
        )
        
        raise ProviderCapabilityError(
            f"All AI providers are currently unavailable. Last error: {last_error}"
        )

    def _build_failover_candidates(
        self, 
        provider_name: str | None, 
        output_format: "GenerationMode" = GenerationMode.TEXT,
        require_cancellation: bool = False,
        require_streaming: bool = False,
        require_tools: bool = False
    ) -> list[str]:
        """
        Builds the ordered candidate list: the requested provider first
        (if supplied and registered), followed by the remaining registered
        providers in a stable default order.

        Filters candidates based on exact capability matching via the registry.
        """
        default_order = ["groq", "gemini", "ollama"]

        eligible_from_registry = registry.find_eligible_providers(
            require_text=(output_format == GenerationMode.TEXT),
            require_structured_json=(output_format == GenerationMode.STRUCTURED_JSON),
            require_streaming=require_streaming,
            require_cancellation=require_cancellation,
            require_tools=require_tools,
        )

        # We also intersect with self._providers to ensure the provider is properly loaded/configured
        eligible = [p for p in eligible_from_registry if p in self._providers]
        
        # Build ordered list
        ordered = [p for p in default_order if p in eligible]

        if provider_name and provider_name in ordered:
            rest = [p for p in ordered if p != provider_name]
            return [provider_name] + rest

        return ordered

    # ── Provider Selection ────────────────────────────────────────────────────

    async def _select_provider(
        self,
        task_hint: str,
        trace_id:  str,
    ) -> BaseProvider:
        active = settings.ACTIVE_LLM

        if active == ActiveLLM.GROQ:
            return self._require_provider("groq")
        if active == ActiveLLM.GEMINI:
            return self._require_provider("gemini")
        if active == ActiveLLM.OLLAMA:
            return self._require_provider("ollama")

        return await self._classify_and_select(task_hint, trace_id)

    async def _classify_and_select(
        self,
        task_hint: str,
        trace_id:  str,
    ) -> BaseProvider:
        """
        Uses a lightweight Groq call to decide groq vs gemini for complex tasks.
        Falls back to Groq if the classifier fails.
        """
        groq = self._providers.get("groq")
        if groq is None:
            return next(iter(self._providers.values()))

        try:
            raw = await groq.generate(
                messages=[{"role": "user", "content": f"Task: {task_hint}"}],
                system_prompt=_build_classifier_system_prompt(),
                max_tokens=20,
                temperature=0.0,
                json_mode=groq.capabilities.supports_json_mode,
            )
            decision = json.loads(raw.strip())
            chosen   = decision.get("model", "groq").lower()

            if chosen in self._providers:
                logger.debug(f"[{trace_id}] AUTO_ROUTE → {chosen}")
                return self._providers[chosen]

        except (ProviderError, json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                f"[{trace_id}] AUTO_ROUTE classifier failed ({exc}). "
                f"Defaulting to groq."
            )

        return groq

    def _get_provider_for_planning(self) -> BaseProvider:
        """Groq is always preferred for planning (fast JSON, low latency)."""
        if "groq" in self._providers:
            return self._providers["groq"]
        return next(iter(self._providers.values()))

    def _require_provider(self, name: str) -> BaseProvider:
        provider = self._providers.get(name)
        if provider is None:
            raise RuntimeError(
                f"Provider '{name}' required by ACTIVE_LLM={name} "
                f"but not registered. Available: {list(self._providers.keys())}."
            )
        return provider

    # ── Failover ──────────────────────────────────────────────────────────────

    async def _generate_with_failover(
        self,
        provider:      BaseProvider,
        messages:      list[dict],
        system_prompt: str,
        trace_id:      str,
        max_tokens:    int   = 800,
        temperature:   float = 0.7,
        json_mode:     bool  = False,
    ) -> str:
        """
        Calls generate() with automatic failover on retryable errors.
        Priority order: groq → gemini → ollama.
        Hard failures (401, 400) do not trigger failover.
        Always returns a non-empty string — never raises.
        """
        failover_order = ["groq", "gemini", "ollama"]
        attempted: list[str] = []

        candidates = [provider] + [
            self._providers[name]
            for name in failover_order
            if name in self._providers and name != provider.provider_name
        ]

        last_error = "Unknown error."

        for candidate in candidates:
            if candidate.provider_name in attempted:
                continue
            attempted.append(candidate.provider_name)

            try:
                result = await candidate.generate(
                    messages=messages,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=json_mode and candidate.capabilities.supports_json_mode,
                )

                if result and result.strip():
                    if candidate.provider_name != provider.provider_name:
                        logger.info(
                            f"[{trace_id}] FAILOVER_SUCCESS | "
                            f"{provider.provider_name} → {candidate.provider_name}"
                        )
                    return result

                logger.warning(
                    f"[{trace_id}] EMPTY_RESPONSE | "
                    f"provider={candidate.provider_name}"
                )
                last_error = "Provider returned an empty response."

            except ProviderError as exc:
                logger.error(
                    f"[{trace_id}] PROVIDER_ERROR | "
                    f"provider={candidate.provider_name} | {exc!r}"
                )
                last_error = str(exc)

                if not exc.retryable:
                    logger.error(
                        f"[{trace_id}] HARD_FAILURE | "
                        f"provider={candidate.provider_name} | "
                        f"status={exc.status_code} | No failover."
                    )
                    break

        logger.critical(
            f"[{trace_id}] ALL_PROVIDERS_FAILED | "
            f"attempted={attempted} | last_error={last_error}"
        )
        return (
            f"System Error: All AI providers are currently unavailable. "
            f"Last error: {last_error}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT SANITIZER
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize_output(text: str) -> str:
    if not text or not text.strip():
        return "I couldn't generate a response. Please try again."
    cleaned = "\n".join(
        line for line in text.splitlines() if line.strip()
    )
    if len(cleaned) > 4000:
        cleaned = cleaned[:4000] + "\n... [Response truncated]"
    return cleaned.strip()