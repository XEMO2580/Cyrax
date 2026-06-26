"""
orchestrator/intent_parser/fast_path.py — CYRAX 3.0 Regex Intent Router

Pre-LLM fast path for unambiguous single-action commands.

Compound command guard:
    If the utterance contains a multi-step conjunction ("and", "then",
    "after", "also", "followed by"), parse() returns None immediately.
    The entire string falls through to brain_router.plan() so the LLM
    can produce a multi-step JSON array.

Single-action commands that match a pattern above the confidence
threshold are returned as ParsedIntent and bypass the LLM entirely.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Minimum confidence to accept a fast-path match ───────────────────────────
_CONFIDENCE_THRESHOLD: float = 0.8

# ── Compound command detection ────────────────────────────────────────────────
# If any of these patterns match the utterance, the fast-path returns None
# so the full string is handled by the LLM planner as a multi-step plan.
# Uses word-boundary anchors to avoid false positives:
#   "sandy" should not trip on "and"
#   "then" in "open notepad then search" should trip
#   "after" in "look after" should NOT trip — hence the lookahead requirement

_COMPOUND_PATTERN = re.compile(
    r"""
    \b(
        and\s+then        |   # "open chrome and then search"
        and\s+also        |   # "open chrome and also search"
        and\s+(?:open|close|search|type|launch|start|kill|delete|play|pause|mute|volume|restart|shutdown|find|go|check|run)
                          |   # "open chrome and search X" — verb after "and"
        \bthen\b          |   # "open notepad then search"
        \bafter\s+that\b  |   # "open chrome after that search"
        \bafter\s+which\b |   # "open chrome after which search"
        \bfollowed\s+by\b |   # "open chrome followed by search"
        \bfirst\b.{1,40}\bthen\b  # "first open chrome then search"
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ParsedIntent(BaseModel):
    """Structured output from the IntentRouter fast-path."""
    tool_name:  str
    confidence: float
    parameters: dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN TABLE
# ══════════════════════════════════════════════════════════════════════════════

_I = re.IGNORECASE

_PATTERNS: dict[str, list[tuple[re.Pattern, float, dict | None]]] = {

    "SYSTEM_POWER": [
        (re.compile(r"^\s*(shut\s*down|power\s*off)\s*(?:my\s+pc|the\s+pc|computer)?\s*$", _I),  1.0, {"action": "shutdown"}),
        (re.compile(r"^\s*(restart|reboot)\s*(?:my\s+pc|the\s+pc|computer)?\s*$", _I),            1.0, {"action": "restart"}),
        (re.compile(r"^\s*abort\s*$", _I),                                                         1.0, {"action": "abort"}),
    ],

    "MEDIA_CONTROL": [
        (re.compile(r"^\s*volume\s+up\s*$", _I),                             0.9, {"action": "volume_up"}),
        (re.compile(r"^\s*volume\s+down\s*$", _I),                           0.9, {"action": "volume_down"}),
        (re.compile(r"^\s*volume\s*(max|full)\s*$", _I),                     1.0, {"action": "volume_max"}),
        (re.compile(r"^\s*turn\s+the\s+(full|max)\s+volume\s+up\s*$", _I),  1.0, {"action": "volume_max"}),
        (re.compile(r"^\s*(mute|unmute)\s*$", _I),                           0.9, {"action": "mute"}),
        (re.compile(r"^\s*(play|pause)\s*$", _I),                            0.9, {"action": "play_pause"}),
        (re.compile(r"^\s*(next|skip)\s*(track|song)?\s*$", _I),             0.8, {"action": "next"}),
        (re.compile(r"^\s*previous\s*(track|song)?\s*$", _I),                0.8, {"action": "previous"}),
    ],

    "OPEN_APP": [
        # Anchored: must be the entire utterance — no trailing content.
        # Named group: app
        (re.compile(
            r"^\s*(?:please\s+)?(?:open|launch|start)\s+['\"]?(?P<app>[a-zA-Z0-9_\-\.]+)['\"]?\s*$",
            _I,
        ), 0.9, None),
    ],

    "CLOSE_APP": [
        # Anchored: must be the entire utterance.
        # Named group: app
        (re.compile(
            r"^\s*(?:please\s+)?(?:close|quit|kill)\s+['\"]?(?P<app>[a-zA-Z0-9_\-\.]+)['\"]?\s*$",
            _I,
        ), 0.9, None),
    ],

    "TYPE_TEXT": [
        # Requires quotes — prevents accidental triggers on conversational input.
        # Named group: text
        (re.compile(
            r"^\s*(?:please\s+)?(?:type|write|input)\s+['\"](?P<text>.+?)['\"][\.!]?\s*$",
            _I,
        ), 0.9, None),
    ],

    "WEB_SEARCH": [
        # Anchored: "search X" or "search for X" — full utterance only.
        # Named group: query
        (re.compile(
            r"^\s*(?:please\s+)?(?:search|google|find)\s+(?:for\s+)?['\"]?(?P<query>[^'\"]{3,}?)['\"]?\s*$",
            _I,
        ), 0.8, None),
        (re.compile(
            r"^\s*search\s+the\s+web\s+for\s+['\"]?(?P<query>.+?)['\"]?\s*$",
            _I,
        ), 0.9, None),
    ],

    "DELETE_FILE": [
        # Named group: filepath
        (re.compile(
            r"^\s*(?:delete|remove)\s+(?:file\s+)?['\"]?(?P<filepath>[^\s'\"]+)['\"]?\s*$",
            _I,
        ), 0.8, None),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# INTENT ROUTER
# ══════════════════════════════════════════════════════════════════════════════

class IntentRouter:
    """
    Stateless regex-based pre-LLM intent classifier.

    Returns None for:
      - Compound commands containing multi-step conjunctions
      - Inputs that do not match any anchored pattern
      - Inputs below the confidence threshold

    In all None cases the dispatcher routes to brain_router.plan().
    """

    @classmethod
    def parse(cls, text: str) -> ParsedIntent | None:
        """
        Attempts to match user input against the anchored pattern table.

        Step 1: Compound guard — if the utterance spans multiple actions,
                return None immediately so the LLM planner handles it.
        Step 2: Pattern match — evaluate anchored regexes in table order.
        Step 3: Confidence threshold — discard matches below 0.8.

        Args:
            text: Raw user utterance (post pronoun-resolution).

        Returns:
            ParsedIntent on an unambiguous single-action match.
            None if compound, no match, or below threshold.
        """
        text = text.strip()
        if not text:
            return None

        # ── Compound command guard ────────────────────────────────────────────
        if _COMPOUND_PATTERN.search(text):
            logger.debug(
                f"[FAST_PATH] Compound command detected — "
                f"falling through to LLM planner: '{text[:80]}'"
            )
            return None

        # ── Pattern matching ──────────────────────────────────────────────────
        for tool_name, pattern_list in _PATTERNS.items():
            for compiled_re, confidence, static_params in pattern_list:

                if confidence < _CONFIDENCE_THRESHOLD:
                    continue

                match = compiled_re.search(text)
                if not match:
                    continue

                params = cls._extract_params(
                    tool_name=tool_name,
                    match=match,
                    static_params=static_params,
                )

                logger.debug(
                    f"[FAST_PATH] Match: {tool_name} | "
                    f"confidence={confidence} | "
                    f"params={params}"
                )

                return ParsedIntent(
                    tool_name=tool_name,
                    confidence=confidence,
                    parameters=params,
                )

        logger.debug(f"[FAST_PATH] No match: '{text[:80]}'")
        return None

    @staticmethod
    def _extract_params(
        tool_name:     str,
        match:         re.Match,
        static_params: dict | None,
    ) -> dict[str, Any]:
        """
        Builds the args dict from a regex match.

        Priority:
          1. static_params — regex had no capture groups (SYSTEM_POWER, MEDIA_CONTROL)
          2. Named groups  — preferred over positional groups
          3. Positional groups — fallback
        """
        if static_params is not None:
            return dict(static_params)

        params: dict[str, Any] = {}

        named = {k: v for k, v in match.groupdict().items() if v is not None}

        if named:
            params.update(named)
            if tool_name == "TYPE_TEXT" and "press_enter" not in params:
                params["press_enter"] = False
            return params

        groups = match.groups()
        if not groups:
            return params

        if tool_name in {"OPEN_APP", "CLOSE_APP"}:
            params["app"] = groups[-1].strip()
        elif tool_name == "TYPE_TEXT":
            params["text"]        = groups[-1].strip()
            params["press_enter"] = False
        elif tool_name == "DELETE_FILE":
            params["filepath"] = groups[-1].strip()
        elif tool_name == "WEB_SEARCH":
            params["query"] = groups[-1].strip()

        return params