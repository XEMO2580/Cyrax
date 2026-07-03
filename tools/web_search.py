"""
tools/web_search.py — CYRAX 3.0 Web Search Tool

Uses the duckduckgo-search library for dependency-free web search.
No API key required. No rate-limit billing. Results are parsed into
a clean structured string the LLM can summarise for the user.

Install: pip install duckduckgo-search

Result format returned to dispatcher:
    [1] Title
        URL: https://...
        Snippet: ...

    [2] ...

The LLM receives this string as the tool response and synthesises
a natural language answer for the user.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from security.auth import SecurityLevel
from tools.registry import BaseTool

logger = logging.getLogger(__name__)

# Maximum results to fetch and format.
_MAX_RESULTS: int = 3

# Maximum characters allowed from a single snippet before truncation.
# Prevents one bloated result from consuming the LLM context window.
_MAX_SNIPPET_CHARS: int = 300

# Request timeout in seconds passed to DDGS.
_REQUEST_TIMEOUT: int = 10


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class WebSearchSchema(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The search query to look up on the web.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TOOL
# ══════════════════════════════════════════════════════════════════════════════

class WebSearchTool(BaseTool):
    """
    Performs a web search via DuckDuckGo and returns structured results.

    Requires no API key. Uses the duckduckgo-search Python library.

    Result formatting:
        Results are formatted as a numbered list with Title, URL, and Snippet
        so the LLM in brain/chat.py can easily synthesise a natural response.
        Empty or missing fields are handled gracefully — a partial result is
        better than a crash.

    Error handling:
        - ImportError:   duckduckgo-search not installed.
        - Timeout:       Network request exceeded _REQUEST_TIMEOUT seconds.
        - Empty results: Query returned no matches.
        - Any other exception from DDGS is caught and returned as an Error string.
    """

    name           = "WEB_SEARCH"
    description    = (
        "Searches the web for a query using DuckDuckGo and returns "
        "a structured summary of the top results."
    )
    security_level = SecurityLevel.USER
    args_schema    = WebSearchSchema

    def execute(self, query: str) -> str:  # type: ignore[override]
        # ── Import guard ──────────────────────────────────────────────────────
        try:
            from ddgs import DDGS
        except ImportError:
            logger.error("[WEB_SEARCH] ddgs is not installed.")
            return (
                "Error: The web search library is not installed. "
                "Run: pip install ddgs"
            )

        query = query.strip()
        if not query:
            return "Error: Search query cannot be empty."

        logger.info(f"[WEB_SEARCH] Querying: '{query[:120]}'")

        # ── Search ────────────────────────────────────────────────────────────
        try:
            with DDGS(timeout=_REQUEST_TIMEOUT) as ddgs:
                raw_results: list[dict[str, Any]] = list(
                    ddgs.text(query, max_results=_MAX_RESULTS)
                )
        except Exception as exc:
            error_name = type(exc).__name__
            logger.error(f"[WEB_SEARCH] DDGS error ({error_name}): {exc}")

            # Surface timeout specifically — it is the most common failure
            # and the user benefits from knowing the cause.
            if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                return (
                    f"Error: Web search timed out after {_REQUEST_TIMEOUT}s. "
                    f"Check your internet connection and try again."
                )

            return (
                f"Error: Web search failed ({error_name}). "
                f"Details: {str(exc)[:200]}"
            )

        # ── Empty results ─────────────────────────────────────────────────────
        if not raw_results:
            logger.warning(f"[WEB_SEARCH] No results for: '{query}'")
            return (
                f"Error: No results found for '{query}'. "
                f"Try rephrasing your search."
            )

        # ── Format results ────────────────────────────────────────────────────
        formatted = _format_results(raw_results, query)

        logger.info(
            f"[WEB_SEARCH] Returning {len(raw_results)} result(s) "
            f"for: '{query[:80]}'"
        )
        return formatted


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def _format_results(
    results: list[dict[str, Any]],
    query:   str,
) -> str:
    """
    Formats raw DDGS result dicts into a numbered, LLM-readable string.

    DDGS result dict keys:
        title:  Page title (str or None)
        href:   URL (str or None)
        body:   Snippet / description (str or None)

    Empty or missing fields are replaced with safe fallback strings so
    the formatter never crashes on malformed results.
    """
    lines: list[str] = [
        f"Web search results for: '{query}'",
        "─" * 48,
    ]

    for idx, result in enumerate(results, start=1):
        title   = _safe_field(result, "title",   "No title")
        url     = _safe_field(result, "href",    "No URL")
        snippet = _safe_field(result, "body",    "No description")

        # Truncate long snippets to keep the context window clean.
        if len(snippet) > _MAX_SNIPPET_CHARS:
            snippet = snippet[:_MAX_SNIPPET_CHARS].rstrip() + "..."

        lines.append(f"[{idx}] {title}")
        lines.append(f"     URL: {url}")
        lines.append(f"     {snippet}")

        if idx < len(results):
            lines.append("")  # Blank line between results.

    lines.append("─" * 48)

    return "\n".join(lines)


def _safe_field(
    result:   dict[str, Any],
    key:      str,
    fallback: str,
) -> str:
    """
    Safely extracts a string field from a result dict.
    Returns fallback if the key is missing, None, or not a string.
    """
    value = result.get(key)
    if not value or not isinstance(value, str):
        return fallback
    return value.strip() or fallback