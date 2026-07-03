"""
tools/pc_actions.py — CYRAX 3.0 PC Action Tools

Tools:
    OpenAppTool     — Opens whitelisted applications or web fallbacks
    CloseAppTool    — Forcefully closes whitelisted applications
    TypeTextTool    — Simulates keyboard typing into the active window

Security:
    OpenAppTool     — UNRESTRICTED (no PIN required, whitelist enforced)
    CloseAppTool    — ADMIN (destructive — kills a running process)
    TypeTextTool    — USER (requires session, injects keystrokes)

Whitelist:
    ALLOWED_APPS and WEB_FALLBACKS are the only permitted execution targets.
    Any app name not in either dict is hard-blocked before subprocess is called.
    No shell interpolation of user input anywhere in this file.

All execute() methods are synchronous, never raise, and return a plain
string. Failures return strings starting with "Error: ".
"""

from __future__ import annotations

import logging
import subprocess
import time
import webbrowser
from enum import Enum

from pydantic import BaseModel, Field

from security.auth import SecurityLevel
from tools.registry import BaseTool

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# WHITELISTS
# ══════════════════════════════════════════════════════════════════════════════

# Executable paths for native OS applications.
# Keys are lowercase canonical names the user/LLM will use.
# Values are the exact executable path or name passed to subprocess.Popen.
ALLOWED_APPS: dict[str, str] = {
    "chrome":      r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "brave":       r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    "notepad":     "notepad.exe",
    "explorer":    "explorer.exe",
    "calculator":  "calc.exe",
    "code":        r"C:\Users\%USERNAME%\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "terminal":    "wt.exe",
    "taskmgr":     "taskmgr.exe",
}

# Web applications opened via the default browser.
# Checked before ALLOWED_APPS — web apps never need subprocess.
WEB_FALLBACKS: dict[str, str] = {
    "youtube":    "https://www.youtube.com",
    "spotify":    "https://open.spotify.com",
    "netflix":    "https://www.netflix.com",
    "github":     "https://github.com",
    "chatgpt":    "https://chatgpt.com",
    "twitter":    "https://twitter.com",
    "x":          "https://twitter.com",
    "instagram":  "https://www.instagram.com",
    "facebook":   "https://www.facebook.com",
    "reddit":     "https://www.reddit.com",
    "gmail":      "https://mail.google.com",
    "drive":      "https://drive.google.com",
    "maps":       "https://maps.google.com",
    "whatsapp":   "https://web.whatsapp.com",
}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: OPEN APP
# ══════════════════════════════════════════════════════════════════════════════

class OpenAppSchema(BaseModel):
    app: str = Field(
        ...,
        description="Name of the application or website to open (e.g. 'chrome', 'youtube').",
    )


class OpenAppTool(BaseTool):
    """
    Opens a whitelisted application or web app.

    Resolution order:
      1. WEB_FALLBACKS  — opens URL in default browser
      2. ALLOWED_APPS   — launches native executable via subprocess.Popen
      3. Hard block     — rejects anything not in either whitelist

    UNRESTRICTED because it is non-destructive and heavily sandboxed.
    """

    name           = "OPEN_APP"
    description    = "Opens a whitelisted application or website."
    security_level = SecurityLevel.UNRESTRICTED
    args_schema    = OpenAppSchema

    def execute(self, app: str) -> str:  # type: ignore[override]
        app_clean = app.lower().strip()

        # 1. Web fallback
        if app_clean in WEB_FALLBACKS:
            try:
                webbrowser.open(WEB_FALLBACKS[app_clean])
                logger.info(f"[OPEN_APP] Web fallback opened: {app_clean}")
                return f"Success: Opened {app_clean} in browser."
            except Exception as exc:
                logger.error(f"[OPEN_APP] Web open failed for {app_clean}: {exc}")
                return f"Error: Failed to open {app_clean} in browser."

        # 2. Native app
        if app_clean in ALLOWED_APPS:
            executable = ALLOWED_APPS[app_clean]
            try:
                subprocess.Popen(
                    executable,
                    shell=False,
                    creationflags=subprocess.DETACHED_PROCESS
                    if hasattr(subprocess, "DETACHED_PROCESS") else 0,
                )
                logger.info(f"[OPEN_APP] Launched: {app_clean}")
                return f"Success: {app_clean} opened."
            except FileNotFoundError:
                logger.warning(f"[OPEN_APP] Executable not found: {executable}")
                return (
                    f"Error: Could not find '{app_clean}'. "
                    f"The application may not be installed at the expected path."
                )
            except Exception as exc:
                logger.error(f"[OPEN_APP] Launch failed for {app_clean}: {exc}")
                return f"Error: Failed to launch {app_clean}."

        # 3. Not whitelisted
        logger.warning(f"[OPEN_APP] Blocked unlisted app: '{app_clean}'")
        return (
            f"Security Block: '{app_clean}' is not on the permitted application list. "
            f"Available apps: {sorted(set(ALLOWED_APPS) | set(WEB_FALLBACKS))}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: CLOSE APP
# ══════════════════════════════════════════════════════════════════════════════

class CloseAppSchema(BaseModel):
    app: str = Field(
        ...,
        description="Name of the application to close (e.g. 'chrome', 'notepad').",
    )


class CloseAppTool(BaseTool):
    """
    Forcefully terminates a whitelisted application process.

    Uses taskkill /F — the process is killed immediately with no save prompt.
    ADMIN level because this is a destructive, irreversible action.

    The process name used with taskkill is derived strictly from the ALLOWED_APPS
    dict value, never from raw user input — prevents process injection attacks.
    """

    name           = "CLOSE_APP"
    description    = "Forcefully closes a whitelisted application."
    security_level = SecurityLevel.ADMIN
    args_schema    = CloseAppSchema

    def execute(self, app: str) -> str:  # type: ignore[override]
        app_clean = app.lower().strip()

        if app_clean not in ALLOWED_APPS:
            logger.warning(f"[CLOSE_APP] Blocked unlisted app: '{app_clean}'")
            return (
                f"Security Block: '{app_clean}' is not on the permitted application list. "
                f"Only whitelisted apps can be closed."
            )

        # Derive the process name from the trusted whitelist path, not user input.
        executable_path = ALLOWED_APPS[app_clean]
        process_name    = executable_path.split("\\")[-1].split("/")[-1]

        try:
            result = subprocess.run(
                ["taskkill", "/IM", process_name, "/F"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )

            if result.returncode == 0:
                logger.info(f"[CLOSE_APP] Terminated: {app_clean} ({process_name})")
                return f"Success: {app_clean} closed."

            # taskkill returns 128 when the process is not found.
            if result.returncode == 128 or "not found" in result.stderr.lower():
                return f"Error: {app_clean} is not currently running."

            logger.warning(
                f"[CLOSE_APP] taskkill non-zero exit for {app_clean}: "
                f"rc={result.returncode} stderr={result.stderr[:100]}"
            )
            return f"Error: Could not close {app_clean}. It may have already exited."

        except subprocess.TimeoutExpired:
            logger.error(f"[CLOSE_APP] taskkill timed out for {app_clean}")
            return f"Error: Timed out trying to close {app_clean}."
        except FileNotFoundError:
            logger.error("[CLOSE_APP] taskkill not found — not running on Windows?")
            return "Error: taskkill is not available on this operating system."
        except Exception as exc:
            logger.exception(f"[CLOSE_APP] Unexpected error for {app_clean}: {exc}")
            return f"Error: Failed to close {app_clean} due to an unexpected error."


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: TYPE TEXT
# ══════════════════════════════════════════════════════════════════════════════

class TypeTextSchema(BaseModel):
    text: str = Field(
        ...,
        description="The exact text string to type into the currently active window.",
    )
    press_enter: bool = Field(
        default=False,
        description="If True, presses Enter after typing the text.",
    )


class TypeTextTool(BaseTool):
    """
    Simulates keyboard input into the currently focused window.

    Security sanitisation:
      - Literal newline characters are stripped from the payload.
        A newline in typed text is equivalent to pressing Enter, which could
        submit forms, execute shell commands, or trigger unintended actions.
        Only the explicit press_enter=True flag may cause an Enter keystroke.
      - Uses pyautogui.write() with a character-level interval rather than
        pyautogui.typewrite() to support a broader Unicode character range.

    USER level — requires a valid session but not a PIN per invocation.
    """

    name           = "TYPE_TEXT"
    description    = "Simulates keyboard typing in the currently active window."
    security_level = SecurityLevel.USER
    args_schema    = TypeTextSchema

    def execute(self, text: str, press_enter: bool = False) -> str:  # type: ignore[override]
        try:
            import pyautogui  # type: ignore[import]
        except ImportError:
            return (
                "Error: pyautogui is not installed. "
                "Run: pip install pyautogui"
            )

        # Strip literal newlines — only press_enter=True may send Enter.
        safe_text = text.replace("\n", "").replace("\r", "").replace("\x00", "")

        if not safe_text:
            return "Error: Text is empty after sanitisation. Nothing typed."

        try:
            time.sleep(0.4)  # Brief focus buffer before typing begins.
            pyautogui.write(safe_text, interval=0.01)

            if press_enter:
                pyautogui.press("enter")
                logger.info(
                    f"[TYPE_TEXT] Typed {len(safe_text)} chars + Enter."
                )
                return "Success: Text typed and Enter pressed."

            logger.info(f"[TYPE_TEXT] Typed {len(safe_text)} chars.")
            return "Success: Text typed."

        except Exception as exc:
            logger.error(f"[TYPE_TEXT] Typing failed: {exc}")
            return f"Error: Keyboard simulation failed — {exc}"