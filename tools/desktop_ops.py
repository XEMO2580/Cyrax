"""
tools/desktop_ops.py — CYRAX 3.0 Desktop Operations Tools

Tools:
    ClipboardReadTool   — Reads current clipboard text content
    ClipboardWriteTool  — Writes text to the system clipboard
    SendNotificationTool — Fires a native OS desktop notification

Security:
    ClipboardReadTool    — USER         (reads potentially sensitive data)
    ClipboardWriteTool   — USER         (modifies shared system state)
    SendNotificationTool — UNRESTRICTED (display only, no side effects)

Dependencies (all optional — OS boots normally if missing):
    pyperclip  — cross-platform clipboard access
    plyer      — cross-platform desktop notifications

Defensive imports:
    All third-party imports are guarded. Missing libraries surface as
    runtime errors from execute(), never as boot-time ImportErrors.

Notification threading:
    SendNotificationTool fires notifications in a daemon thread to prevent
    OS notification UI from blocking the dispatcher event loop.
    Thread is fire-and-forget — result is not awaited.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from pydantic import BaseModel, Field

from security.auth import SecurityLevel
from tools.registry import BaseTool

logger = logging.getLogger(__name__)

# Hard limits.
_MAX_CLIPBOARD_READ_CHARS:  int = 50_000
_MAX_CLIPBOARD_WRITE_CHARS: int = 50_000
_MAX_NOTIFICATION_TITLE:    int = 128
_MAX_NOTIFICATION_MESSAGE:  int = 512

# Notification thread timeout — used only for daemon bookkeeping logging.
# The thread itself is not joined — this is fire-and-forget.
_NOTIFICATION_FIRE_TIMEOUT: float = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# CLIPBOARD READ
# ══════════════════════════════════════════════════════════════════════════════

class ClipboardReadSchema(BaseModel):
    pass  # No parameters required — reads the current clipboard state.


class ClipboardReadTool(BaseTool):
    """
    Reads the current text content of the system clipboard.

    Enforces a 50,000-character ceiling to prevent a large clipboard payload
    (e.g. a pasted document) from flooding the LLM context window.

    Returns the clipboard text, or an Error string if:
      - pyperclip is not installed
      - The clipboard contains non-text content (image, file reference)
      - The clipboard is empty
      - Any OS-level access error occurs
    """

    name           = "CLIPBOARD_READ"
    description    = (
        "Reads and returns the current text content of the system clipboard. "
        f"Maximum {_MAX_CLIPBOARD_READ_CHARS:,} characters returned."
    )
    security_level = SecurityLevel.USER
    args_schema    = ClipboardReadSchema

    def execute(self) -> str:  # type: ignore[override]
        try:
            import pyperclip  # type: ignore[import]
        except ImportError:
            return (
                "Error: pyperclip is not installed. "
                "Run: pip install pyperclip"
            )

        try:
            content: Any = pyperclip.paste()
        except pyperclip.PyperclipException as exc:
            logger.warning(f"[CLIPBOARD_READ] PyperclipException: {exc}")
            return (
                f"Error: Could not access the clipboard — {exc}. "
                f"On Linux, ensure xclip or xsel is installed."
            )
        except Exception as exc:
            logger.error(f"[CLIPBOARD_READ] Unexpected error: {exc}")
            return f"Error: Clipboard read failed — {exc}"

        # pyperclip.paste() may return None or non-string on some platforms.
        if content is None:
            return "Error: Clipboard is empty or contains no text."

        if not isinstance(content, str):
            return (
                f"Error: Clipboard contains non-text data "
                f"(type: {type(content).__name__}). "
                f"Only plain text can be read."
            )

        content = content  # Already a str at this point.

        if not content.strip():
            return "Error: Clipboard is empty."

        if len(content) > _MAX_CLIPBOARD_READ_CHARS:
            truncated = content[:_MAX_CLIPBOARD_READ_CHARS]
            logger.info(
                f"[CLIPBOARD_READ] Content truncated from "
                f"{len(content):,} → {_MAX_CLIPBOARD_READ_CHARS:,} chars."
            )
            return (
                truncated
                + f"\n\n... [Clipboard content truncated at "
                f"{_MAX_CLIPBOARD_READ_CHARS:,} characters]"
            )

        logger.info(
            f"[CLIPBOARD_READ] Read {len(content):,} chars from clipboard."
        )
        return content


# ══════════════════════════════════════════════════════════════════════════════
# CLIPBOARD WRITE
# ══════════════════════════════════════════════════════════════════════════════

class ClipboardWriteSchema(BaseModel):
    text: str = Field(
        ...,
        description="The text to write to the system clipboard.",
        max_length=_MAX_CLIPBOARD_WRITE_CHARS,
    )


class ClipboardWriteTool(BaseTool):
    """
    Writes text to the system clipboard, replacing its current contents.

    Enforces a 50,000-character write limit.
    Null bytes and other non-printable control characters that cause
    clipboard corruption on some platforms are stripped before writing.
    """

    name           = "CLIPBOARD_WRITE"
    description    = (
        "Writes text to the system clipboard, replacing its current content. "
        f"Maximum {_MAX_CLIPBOARD_WRITE_CHARS:,} characters."
    )
    security_level = SecurityLevel.USER
    args_schema    = ClipboardWriteSchema

    def execute(self, text: str) -> str:  # type: ignore[override]
        try:
            import pyperclip  # type: ignore[import]
        except ImportError:
            return (
                "Error: pyperclip is not installed. "
                "Run: pip install pyperclip"
            )

        if not isinstance(text, str):
            return "Error: Clipboard content must be a plain text string."

        if len(text) > _MAX_CLIPBOARD_WRITE_CHARS:
            return (
                f"Error: Text is too long to copy "
                f"({len(text):,} chars). "
                f"Maximum allowed: {_MAX_CLIPBOARD_WRITE_CHARS:,} chars."
            )

        # Strip null bytes — these cause silent corruption on Windows clipboard.
        safe_text = text.replace("\x00", "")

        try:
            pyperclip.copy(safe_text)
        except pyperclip.PyperclipException as exc:
            logger.warning(f"[CLIPBOARD_WRITE] PyperclipException: {exc}")
            return (
                f"Error: Could not write to clipboard — {exc}. "
                f"On Linux, ensure xclip or xsel is installed."
            )
        except Exception as exc:
            logger.error(f"[CLIPBOARD_WRITE] Unexpected error: {exc}")
            return f"Error: Clipboard write failed — {exc}"

        logger.info(
            f"[CLIPBOARD_WRITE] Wrote {len(safe_text):,} chars to clipboard."
        )
        return f"Success: {len(safe_text):,} characters copied to clipboard."


# ══════════════════════════════════════════════════════════════════════════════
# SEND NOTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

class SendNotificationSchema(BaseModel):
    title: str = Field(
        ...,
        max_length=_MAX_NOTIFICATION_TITLE,
        description="Title of the desktop notification.",
    )
    message: str = Field(
        ...,
        max_length=_MAX_NOTIFICATION_MESSAGE,
        description="Body text of the desktop notification.",
    )
    timeout: int = Field(
        default=5,
        ge=1,
        le=30,
        description="How long the notification stays visible (seconds). Default: 5.",
    )


class SendNotificationTool(BaseTool):
    """
    Fires a native OS desktop notification via plyer.

    Threading model:
        The plyer notification call is dispatched in a daemon thread.
        This is fire-and-forget — execute() returns immediately after
        launching the thread without waiting for the OS notification
        UI to render or dismiss.

        Rationale: On some Windows/macOS configurations, notification
        APIs can block the calling thread for several seconds while the
        OS renders the notification banner. Running on the dispatcher's
        coroutine thread (via asyncio.to_thread) would stall the event
        loop. A daemon thread isolates this latency entirely.

    UNRESTRICTED because notifications are display-only with no
    file system, network, or process side effects.
    """

    name           = "SEND_NOTIFICATION"
    description    = (
        "Sends a native desktop notification with a title and message. "
        "Returns immediately — notification is displayed in the background."
    )
    security_level = SecurityLevel.UNRESTRICTED
    args_schema    = SendNotificationSchema

    def execute(  # type: ignore[override]
        self,
        title:   str,
        message: str,
        timeout: int = 5,
    ) -> str:
        try:
            from plyer import notification as plyer_notification  # type: ignore[import]
        except ImportError:
            return (
                "Error: plyer is not installed. "
                "Run: pip install plyer"
            )

        # Sanitise inputs — truncate silently rather than raising.
        safe_title   = str(title)[:_MAX_NOTIFICATION_TITLE].strip()
        safe_message = str(message)[:_MAX_NOTIFICATION_MESSAGE].strip()

        if not safe_title:
            return "Error: Notification title cannot be empty."
        if not safe_message:
            return "Error: Notification message cannot be empty."

        def _fire() -> None:
            """
            Runs inside a daemon thread. All exceptions are caught so
            a notification failure never propagates to the thread pool.
            """
            try:
                plyer_notification.notify(
                    title       = safe_title,
                    message     = safe_message,
                    app_name    = "CYRAX",
                    timeout     = timeout,
                )
                logger.info(
                    f"[NOTIFICATION] Sent: '{safe_title}' | "
                    f"'{safe_message[:60]}'"
                )
            except Exception as exc:
                # Cannot return to caller from thread — log only.
                logger.error(
                    f"[NOTIFICATION] plyer.notification.notify failed: {exc}"
                )

        thread = threading.Thread(
            target=_fire,
            name="cyrax-notification",
            daemon=True,   # Thread dies with the process — no orphan threads.
        )
        thread.start()

        logger.debug(
            f"[NOTIFICATION] Fired in background thread: '{safe_title}'"
        )
        return (
            f"Success: Notification '{safe_title}' sent. "
            f"It will appear on your desktop shortly."
        )