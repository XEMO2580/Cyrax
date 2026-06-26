"""
memory/conversation/conversation_store.py — CYRAX 3.0 Conversation Store

Per-session conversation history with two-tier eviction:
  Tier 1 — Turn count:  Hard cap at settings.CONVERSATION_MAX_TURNS.
            Oldest messages are dropped when the cap is exceeded.
  Tier 2 — Token count: Soft cap at settings.CONVERSATION_MAX_TOKENS.
            When the estimated token count exceeds the threshold, the store
            invokes an optional summarisation callback injected at boot.
            If no callback is provided, oldest messages are truncated instead.

Token estimation:
    Exact tokenisation requires a provider-specific tokeniser (tiktoken for
    OpenAI-compatible, sentencepiece for Gemini). Both are optional dependencies
    not yet in the project. The estimator uses the standard 4-chars ≈ 1-token
    heuristic which is accurate to ±15% for English text — sufficient for
    eviction threshold decisions. Swap out _estimate_tokens() when a real
    tokeniser is available without changing any other code.

Persistence:
    Written to disk after every interaction via asyncio.to_thread to avoid
    blocking the event loop. One file per session_id under memory/users/.

Satisfies the 'conversation' attribute contract of MemoryStackProtocol:
    async def add_interaction(self, role: str, content: str) -> None
    def get_history(self) -> list[dict]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Storage path ──────────────────────────────────────────────────────────────
_MODULE_DIR   = Path(__file__).resolve().parent.parent.parent   # cyrax/
_USERS_DIR    = _MODULE_DIR / "memory" / "users"

# Valid role values accepted by the LLM providers wired into brain/.
_VALID_ROLES: frozenset[str] = frozenset({"user", "assistant", "system", "tool"})

# Type alias for the optional summarisation callback.
# Signature: async (history: list[dict]) -> str
SummarisationCallback = Callable[[list[dict]], Awaitable[str]]


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Estimates total token count for a list of role/content message dicts.

    Heuristic: 1 token ≈ 4 characters (English prose).
    Overhead: 4 tokens per message for role + formatting markers.

    Replace this function with a provider-specific tokeniser when available.
    All callers go through this single function so the swap is one-line.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += max(1, len(str(content)) // 4) + 4
    return total


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION STORE
# ══════════════════════════════════════════════════════════════════════════════

class ConversationStore:
    """
    Per-session conversation history with two-tier token+turn eviction.

    Args:
        session_id:
            Used to isolate the memory file on disk.
            One file per session under memory/users/.

        summarise_callback:
            Optional async function injected by bootstrap() from brain/chat.py.
            Called when the token count exceeds CONVERSATION_MAX_TOKENS.
            Receives the current history, returns a condensed summary string.
            If None, the store falls back to truncation (drops oldest messages).
            Injected — never imported directly — to avoid circular dependencies.
    """

    def __init__(
        self,
        session_id: str,
        summarise_callback: Optional[SummarisationCallback] = None,
    ) -> None:
        self._session_id          = session_id
        self._summarise_callback  = summarise_callback
        self._history: list[dict[str, Any]] = []

        _USERS_DIR.mkdir(parents=True, exist_ok=True)
        self._file_path = _USERS_DIR / f"memory_{session_id}.json"

        logger.debug(
            f"[CONVERSATION] Store initialised. "
            f"Session: {session_id} | Path: {self._file_path}"
        )

    # ── Load / Save ───────────────────────────────────────────────────────────

    async def load(self) -> None:
        """
        Loads persisted history from disk.
        Must be called once by bootstrap() before the first interaction.
        No-op if no file exists yet (fresh session).
        """
        if not self._file_path.exists():
            self._history = []
            logger.debug(
                f"[CONVERSATION] No history file found for '{self._session_id}'. "
                f"Starting fresh."
            )
            return

        try:
            def _read() -> list:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []

            self._history = await asyncio.to_thread(_read)
            logger.info(
                f"[CONVERSATION] Loaded {len(self._history)} messages "
                f"for session '{self._session_id}'."
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                f"[CONVERSATION] Failed to load history for "
                f"'{self._session_id}': {exc}. Starting fresh."
            )
            self._history = []

    async def _save(self) -> None:
        """
        Persists the current history to disk in the background.
        Uses asyncio.to_thread so file I/O never blocks the event loop.
        """
        try:
            def _write() -> None:
                with open(self._file_path, "w", encoding="utf-8") as f:
                    json.dump(self._history, f, indent=2, ensure_ascii=False)

            await asyncio.to_thread(_write)
        except OSError as exc:
            logger.error(
                f"[CONVERSATION] Save failed for '{self._session_id}': {exc}"
            )

    # ── Core Interface ────────────────────────────────────────────────────────

    async def add_interaction(self, role: str, content: str) -> None:
        """
        Appends a message to history and triggers eviction if needed.

        Args:
            role:    Must be one of: "user", "assistant", "system", "tool".
            content: The message text. Long tool outputs are truncated before
                     storage to prevent a single response from bloating memory.

        Eviction order:
            1. Content truncation  — individual message too long
            2. Turn-count eviction — total messages exceed MAX_TURNS
            3. Token eviction      — total tokens exceed MAX_TOKENS
        """
        if role not in _VALID_ROLES:
            logger.warning(
                f"[CONVERSATION] Invalid role '{role}' — "
                f"must be one of {_VALID_ROLES}. Message discarded."
            )
            return

        # Per-message content truncation — prevents a single large tool
        # response from consuming the entire context window on its own.
        _MAX_CONTENT_CHARS = 2000
        if len(content) > _MAX_CONTENT_CHARS:
            content = (
                content[:_MAX_CONTENT_CHARS]
                + "... [truncated for memory efficiency]"
            )
            logger.debug(
                f"[CONVERSATION] Message content truncated to "
                f"{_MAX_CONTENT_CHARS} chars."
            )

        self._history.append({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Apply eviction policies after every append.
        await self._apply_eviction()
        await self._save()

    def get_history(self) -> list[dict[str, Any]]:
        """
        Returns the conversation history formatted for LLM injection.

        Timestamps are stripped — they are for human inspection in the JSON
        file only. LLM providers do not use them and they waste tokens.

        Returns:
            [{"role": str, "content": str}, ...]
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self._history
        ]

    def get_raw_history(self) -> list[dict[str, Any]]:
        """
        Returns the full history including timestamps.
        Used for summarisation callbacks and debugging — not for LLM injection.
        """
        return list(self._history)

    def message_count(self) -> int:
        """Returns the current number of messages in history."""
        return len(self._history)

    def estimated_tokens(self) -> int:
        """Returns the estimated token count of the current history."""
        return _estimate_tokens(self._history)

    def clear(self) -> None:
        """
        Clears the in-process history.
        Does NOT delete the file — call clear() + _save() to wipe disk too.
        """
        self._history = []
        logger.info(
            f"[CONVERSATION] History cleared for session '{self._session_id}'."
        )

    # ── Eviction ──────────────────────────────────────────────────────────────

    async def _apply_eviction(self) -> None:
        """
        Applies turn-count and token-count eviction policies.

        Called after every append. Both checks are applied in sequence:
          1. Turn count: if len > MAX_TURNS, drop oldest messages.
          2. Token count: if estimated_tokens > MAX_TOKENS, summarise or truncate.
        """
        await self._enforce_turn_limit()
        await self._enforce_token_limit()

    async def _enforce_turn_limit(self) -> None:
        """
        Drops oldest messages when the turn count exceeds the hard cap.
        System messages are preserved — only user/assistant/tool turns are counted
        against the limit and candidates for removal.
        """
        max_turns = settings.CONVERSATION_MAX_TURNS

        if len(self._history) <= max_turns:
            return

        excess = len(self._history) - max_turns

        # Preserve system messages at index 0 if present.
        # Drop from the oldest non-system messages first.
        non_system_indices = [
            i for i, m in enumerate(self._history)
            if m["role"] != "system"
        ]

        to_remove = non_system_indices[:excess]
        if to_remove:
            for idx in reversed(to_remove):
                self._history.pop(idx)
            logger.debug(
                f"[CONVERSATION] Turn-limit eviction: "
                f"dropped {len(to_remove)} oldest messages. "
                f"Remaining: {len(self._history)}"
            )

    async def _enforce_token_limit(self) -> None:
        """
        Applies token-count eviction when the estimated total exceeds the threshold.

        Strategy A (callback available): Summarise the oldest half of history
            into a single assistant message, then replace those messages with
            the summary. Preserves conversational context in compressed form.

        Strategy B (no callback): Truncate — drop the oldest quarter of
            non-system messages. Simple, always available, loses context.
        """
        max_tokens = settings.CONVERSATION_MAX_TOKENS
        current_tokens = _estimate_tokens(self._history)

        if current_tokens <= max_tokens:
            return

        logger.warning(
            f"[CONVERSATION] Token threshold exceeded: "
            f"~{current_tokens} tokens > {max_tokens} limit. "
            f"Applying eviction."
        )

        if self._summarise_callback is not None:
            await self._summarise_oldest_half()
        else:
            self._truncate_oldest_quarter()

    async def _summarise_oldest_half(self) -> None:
        """
        Summarises the oldest half of history using the injected callback.
        Replaces those messages with a single compressed assistant message.
        """
        midpoint = len(self._history) // 2
        to_summarise = self._history[:midpoint]
        keep          = self._history[midpoint:]

        logger.info(
            f"[CONVERSATION] Summarising {len(to_summarise)} oldest messages."
        )

        try:
            summary_text = await self._summarise_callback(
                [{"role": m["role"], "content": m["content"]} for m in to_summarise]
            )

            summary_message = {
                "role":      "assistant",
                "content":   f"[Summary of earlier conversation]: {summary_text}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self._history = [summary_message] + keep

            logger.info(
                f"[CONVERSATION] Summarisation complete. "
                f"History compressed from "
                f"{len(to_summarise) + len(keep)} → {len(self._history)} messages."
            )

        except Exception as exc:
            logger.error(
                f"[CONVERSATION] Summarisation callback failed: {exc}. "
                f"Falling back to truncation."
            )
            self._truncate_oldest_quarter()

    def _truncate_oldest_quarter(self) -> None:
        """
        Drops the oldest quarter of non-system messages.
        Fallback when no summarisation callback is available.
        """
        non_system = [
            i for i, m in enumerate(self._history)
            if m["role"] != "system"
        ]
        drop_count = max(1, len(non_system) // 4)
        to_remove  = non_system[:drop_count]

        for idx in reversed(to_remove):
            self._history.pop(idx)

        logger.warning(
            f"[CONVERSATION] Truncation eviction: "
            f"dropped {len(to_remove)} oldest messages (no summariser). "
            f"Remaining: {len(self._history)}"
        )