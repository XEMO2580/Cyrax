"""
tools/media_power.py — CYRAX 3.0 Media and Power Tools

Tools:
    MediaControlTool  — Volume, mute, play/pause, next/prev track
    SystemPowerTool   — Shutdown, restart, abort pending shutdown

Security:
    MediaControlTool  — USER  (requires session, non-destructive)
    SystemPowerTool   — ADMIN (destructive, requires PIN)

Platform:
    Windows-primary. pyautogui media keys work on Windows and macOS.
    pycaw (Windows-only) provides precise volume control.
    Imports are defensive — if a library is missing, the tool falls back
    to pyautogui key simulation and logs a warning. Boot never fails due
    to a missing optional audio library.

All execute() methods are synchronous, never raise, and return a plain
string. Failures return strings starting with "Error: ".
"""

from __future__ import annotations

import logging
import subprocess
import sys
from enum import Enum

from pydantic import BaseModel, Field

from security.auth import SecurityLevel
from tools.registry import BaseTool

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DEFENSIVE IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

# pyautogui — used for media keys and fallback volume control.
try:
    import pyautogui          # type: ignore[import]
    _PYAUTOGUI_AVAILABLE = True
except ImportError:
    _PYAUTOGUI_AVAILABLE = False
    logger.warning(
        "[MEDIA] pyautogui not installed. "
        "Media key simulation will be unavailable. "
        "Run: pip install pyautogui"
    )

# pycaw — Windows-only, precise volume control via the Windows Core Audio API.
# Only attempted on Windows to avoid ImportError on other platforms.
_PYCAW_AVAILABLE = False
if sys.platform == "win32":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL              # type: ignore[import]
        from pycaw.pycaw import (                    # type: ignore[import]
            AudioUtilities,
            IAudioEndpointVolume,
        )
        _PYCAW_AVAILABLE = True
    except ImportError:
        logger.warning(
            "[MEDIA] pycaw not installed. "
            "Falling back to pyautogui key simulation for volume. "
            "Run: pip install pycaw comtypes  for precise volume control."
        )
    except Exception as exc:
        logger.warning(f"[MEDIA] pycaw initialisation error: {exc}. Using fallback.")


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_pycaw_volume_interface():
    """
    Returns the Windows IAudioEndpointVolume COM interface, or None on failure.
    Called lazily — not at import time — so COM errors don't crash the module.
    """
    if not _PYCAW_AVAILABLE:
        return None
    try:
        from ctypes import cast, POINTER
        import comtypes                                # <--- Ensure this is imported
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        comtypes.CoInitialize()

        devices   = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as exc:
        logger.warning(f"[MEDIA] pycaw volume interface error: {exc}")
        return None


def _set_volume_pycaw(level: float) -> bool:
    """
    Sets master volume to an absolute scalar (0.0–1.0) via pycaw.
    Returns True on success, False on failure.
    """
    volume = _get_pycaw_volume_interface()
    if volume is None:
        return False
    try:
        volume.SetMasterVolumeLevelScalar(
            max(0.0, min(1.0, level)), None
        )
        return True
    except Exception as exc:
        logger.warning(f"[MEDIA] pycaw SetMasterVolumeLevelScalar failed: {exc}")
        return False


def _get_volume_pycaw() -> float | None:
    """Returns current master volume scalar (0.0–1.0), or None on failure."""
    volume = _get_pycaw_volume_interface()
    if volume is None:
        return None
    try:
        return float(volume.GetMasterVolumeLevelScalar())
    except Exception as exc:
        logger.warning(f"[MEDIA] pycaw GetMasterVolumeLevelScalar failed: {exc}")
        return None


def _set_mute_pycaw(muted: bool) -> bool:
    """Sets master mute state via pycaw. Returns True on success."""
    volume = _get_pycaw_volume_interface()
    if volume is None:
        return False
    try:
        volume.SetMute(int(muted), None)
        return True
    except Exception as exc:
        logger.warning(f"[MEDIA] pycaw SetMute failed: {exc}")
        return False


def _press_key(key: str, presses: int = 1) -> bool:
    """
    Sends a media/volume key via pyautogui.
    Returns True on success, False if pyautogui is unavailable or errors.
    """
    if not _PYAUTOGUI_AVAILABLE:
        return False
    try:
        if presses == 1:
            pyautogui.press(key)
        else:
            pyautogui.press(key, presses=presses)
        return True
    except Exception as exc:
        logger.error(f"[MEDIA] pyautogui.press({key!r}) failed: {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: MEDIA CONTROL
# ══════════════════════════════════════════════════════════════════════════════

class MediaAction(str, Enum):
    PLAY_PAUSE = "play_pause"
    NEXT       = "next"
    PREVIOUS   = "previous"
    MUTE       = "mute"
    UNMUTE     = "unmute"
    VOLUME_UP  = "volume_up"
    VOLUME_DOWN= "volume_down"
    VOLUME_MAX = "volume_max"


class MediaControlSchema(BaseModel):
    action: MediaAction = Field(
        ...,
        description=(
            "The media action to perform. "
            "Valid values: play_pause, next, previous, mute, unmute, "
            "volume_up, volume_down, volume_max."
        ),
    )


class MediaControlTool(BaseTool):
    """
    Controls system media playback and volume.

    Volume strategy:
        volume_up / volume_down: Uses pycaw for ±10% precise steps on Windows.
                                  Falls back to 10× pyautogui key presses (~20%).
        volume_max:              Sets scalar to 1.0 via pycaw, or 50× key presses.
        mute / unmute:           Toggle via pycaw COM interface or volumemute key.

    Playback strategy:
        play_pause / next / previous: pyautogui media keys (cross-platform).
    """

    name           = "MEDIA_CONTROL"
    description    = "Controls system volume and media playback (play, pause, next, previous, mute, volume)."
    security_level = SecurityLevel.USER
    args_schema    = MediaControlSchema

    # Volume step for pycaw (10% per up/down command).
    _VOLUME_STEP: float = 0.10

    def execute(self, action: MediaAction) -> str:  # type: ignore[override]
        handler = {
            MediaAction.PLAY_PAUSE:  self._play_pause,
            MediaAction.NEXT:        self._next,
            MediaAction.PREVIOUS:    self._previous,
            MediaAction.MUTE:        self._mute,
            MediaAction.UNMUTE:      self._unmute,
            MediaAction.VOLUME_UP:   self._volume_up,
            MediaAction.VOLUME_DOWN: self._volume_down,
            MediaAction.VOLUME_MAX:  self._volume_max,
        }.get(action)

        if handler is None:
            return f"Error: Unknown media action '{action}'."

        return handler()

    # ── Playback ──────────────────────────────────────────────────────────────

    def _play_pause(self) -> str:
        ok = _press_key("playpause")
        return "Success: Play/pause toggled." if ok else _no_pyautogui()

    def _next(self) -> str:
        ok = _press_key("nexttrack")
        return "Success: Skipped to next track." if ok else _no_pyautogui()

    def _previous(self) -> str:
        ok = _press_key("prevtrack")
        return "Success: Went to previous track." if ok else _no_pyautogui()

    # ── Mute ──────────────────────────────────────────────────────────────────

    def _mute(self) -> str:
        if _PYCAW_AVAILABLE:
            ok = _set_mute_pycaw(True)
            if ok:
                logger.info("[MEDIA] Muted via pycaw.")
                return "Success: System audio muted."
        ok = _press_key("volumemute")
        return "Success: Mute toggled." if ok else _no_pyautogui()

    def _unmute(self) -> str:
        if _PYCAW_AVAILABLE:
            ok = _set_mute_pycaw(False)
            if ok:
                logger.info("[MEDIA] Unmuted via pycaw.")
                return "Success: System audio unmuted."
        ok = _press_key("volumemute")
        return "Success: Mute toggled." if ok else _no_pyautogui()

    # ── Volume ────────────────────────────────────────────────────────────────

    def _volume_up(self) -> str:
        if _PYCAW_AVAILABLE:
            current = _get_volume_pycaw()
            if current is not None:
                new_level = min(1.0, current + self._VOLUME_STEP)
                ok = _set_volume_pycaw(new_level)
                if ok:
                    logger.info(
                        f"[MEDIA] Volume up: {current:.0%} → {new_level:.0%}"
                    )
                    return f"Success: Volume increased to {new_level:.0%}."

        ok = _press_key("volumeup", presses=10)
        return "Success: Volume increased." if ok else _no_pyautogui()

    def _volume_down(self) -> str:
        if _PYCAW_AVAILABLE:
            current = _get_volume_pycaw()
            if current is not None:
                new_level = max(0.0, current - self._VOLUME_STEP)
                ok = _set_volume_pycaw(new_level)
                if ok:
                    logger.info(
                        f"[MEDIA] Volume down: {current:.0%} → {new_level:.0%}"
                    )
                    return f"Success: Volume decreased to {new_level:.0%}."

        ok = _press_key("volumedown", presses=10)
        return "Success: Volume decreased." if ok else _no_pyautogui()

    def _volume_max(self) -> str:
        if _PYCAW_AVAILABLE:
            ok = _set_volume_pycaw(1.0)
            if ok:
                logger.info("[MEDIA] Volume set to maximum via pycaw.")
                return "Success: Volume set to maximum."

        ok = _press_key("volumeup", presses=50)
        return "Success: Volume maximised." if ok else _no_pyautogui()


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: SYSTEM POWER
# ══════════════════════════════════════════════════════════════════════════════

class PowerAction(str, Enum):
    SHUTDOWN = "shutdown"
    RESTART  = "restart"
    ABORT    = "abort"


class SystemPowerSchema(BaseModel):
    action: PowerAction = Field(
        ...,
        description=(
            "Power action to perform. "
            "shutdown — schedules shutdown in 60s. "
            "restart  — schedules restart in 60s. "
            "abort    — cancels a pending shutdown or restart."
        ),
    )


class SystemPowerTool(BaseTool):
    """
    Controls OS power state.

    ADMIN level — this is the highest-risk tool in the registry.
    A 60-second delay is enforced on shutdown and restart to give the
    user a recovery window. The abort action cancels a pending operation.

    Windows-only implementation (uses shutdown.exe).
    Raises a clear error on non-Windows platforms instead of silently
    doing nothing or crashing with an obscure subprocess error.
    """

    name           = "SYSTEM_POWER"
    description    = "Controls OS power state: shutdown, restart, or abort a pending shutdown."
    security_level = SecurityLevel.ADMIN
    args_schema    = SystemPowerSchema

    # Seconds between the command and actual power-off. Gives the user
    # time to abort if the command was accidental.
    _DELAY_SECONDS: int = 60

    def execute(self, action: PowerAction) -> str:  # type: ignore[override]
        if sys.platform != "win32":
            return (
                "Error: System power control is currently only supported on Windows. "
                f"Platform detected: {sys.platform}"
            )

        handler = {
            PowerAction.SHUTDOWN: self._shutdown,
            PowerAction.RESTART:  self._restart,
            PowerAction.ABORT:    self._abort,
        }.get(action)

        if handler is None:
            return f"Error: Unknown power action '{action}'."

        return handler()

    def _shutdown(self) -> str:
        try:
            subprocess.run(
                ["shutdown", "/s", "/t", str(self._DELAY_SECONDS)],
                check=True,
                capture_output=True,
                shell=False,
                timeout=10,
            )
            logger.warning(
                f"[POWER] Shutdown scheduled in {self._DELAY_SECONDS}s."
            )
            return (
                f"Success: PC shutting down in {self._DELAY_SECONDS} seconds. "
                f"Say 'abort' to cancel."
            )
        except subprocess.CalledProcessError as exc:
            err_msg = exc.stderr.decode(errors="replace").strip() if exc.stderr else "Unknown OS error"
            logger.error(f"[POWER] Command failed: {exc} | {err_msg}")
            return f"Error: Command failed — {err_msg}"
        except subprocess.TimeoutExpired:
            return "Error: Shutdown command timed out."
        except FileNotFoundError:
            return "Error: shutdown.exe not found. Is this a Windows system?"
        except Exception as exc:
            return f"Error: Unexpected error during shutdown — {exc}"

    def _restart(self) -> str:
        try:
            subprocess.run(
                ["shutdown", "/r", "/t", str(self._DELAY_SECONDS)],
                check=True,
                capture_output=True,
                shell=False,
                timeout=10,
            )
            logger.warning(
                f"[POWER] Restart scheduled in {self._DELAY_SECONDS}s."
            )
            return (
                f"Success: PC restarting in {self._DELAY_SECONDS} seconds. "
                f"Say 'abort' to cancel."
            )
        except subprocess.CalledProcessError as exc:
            logger.error(f"[POWER] Restart command failed: {exc}")
            return f"Error: Restart command failed — {exc.stderr.decode(errors='replace').strip()}"
        except subprocess.TimeoutExpired:
            return "Error: Restart command timed out."
        except FileNotFoundError:
            return "Error: shutdown.exe not found. Is this a Windows system?"
        except Exception as exc:
            return f"Error: Unexpected error during restart — {exc}"

    def _abort(self) -> str:
        try:
            subprocess.run(
                ["shutdown", "/a"],
                check=True,
                capture_output=True,
                shell=False,
                timeout=10,
            )
            logger.info("[POWER] Pending shutdown/restart aborted.")
            return "Success: Pending shutdown or restart has been cancelled."
        except subprocess.CalledProcessError:
            # Return code is non-zero when there is no pending shutdown.
            return "Error: No pending shutdown or restart to abort."
        except subprocess.TimeoutExpired:
            return "Error: Abort command timed out."
        except FileNotFoundError:
            return "Error: shutdown.exe not found. Is this a Windows system?"
        except Exception as exc:
            return f"Error: Unexpected error during abort — {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _no_pyautogui() -> str:
    return (
        "Error: pyautogui is not installed. "
        "Media key simulation is unavailable. "
        "Run: pip install pyautogui"
    )