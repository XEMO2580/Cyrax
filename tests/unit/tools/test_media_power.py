"""
tests/tools/test_media_power.py — Certification suite for tools/media_power.py

TECHNIQUE NOTE (see Phase 4.5A audit): Unlike desktop_ops.py and
pc_actions.py, this file's third-party availability flags
(_PYAUTOGUI_AVAILABLE, _PYCAW_AVAILABLE) are computed ONCE at module
import time and are NOT re-evaluated per call. block_import (sys.modules
patching) has no effect on already-imported module state, so it is NOT
used here. Instead:
  - _PYAUTOGUI_AVAILABLE / pyautogui are monkeypatched directly on the
    tools.media_power module object.
  - pycaw's COM interop is mocked at the module's own helper-function
    boundary (_set_volume_pycaw, _get_volume_pycaw, _set_mute_pycaw)
    rather than attempting to mock ctypes/comtypes COM pointers directly.
  - sys.platform is monkeypatched for SystemPowerTool, since execute()
    hard-blocks on non-Windows before reaching any other mockable code.

CAVEAT: This suite is written against the version of media_power.py
authored earlier in this session. If the file on disk has since diverged,
re-run this audit against the actual current source before treating this
certification as final.

NO REAL SUBPROCESSES, NO REAL KEYSTROKES, NO REAL AUDIO CHANGES.
"""

from __future__ import annotations

import subprocess
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from tools import media_power
from tools.media_power import (
    MediaControlTool, MediaControlSchema, MediaAction,
    SystemPowerTool, SystemPowerSchema, PowerAction,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def with_pyautogui(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """
    Forces _PYAUTOGUI_AVAILABLE = True and injects a Mock pyautogui object
    into the module namespace, regardless of whether the real library is
    actually installed in the test environment.
    """
    mock_pyautogui = Mock()
    monkeypatch.setattr(media_power, "_PYAUTOGUI_AVAILABLE", True)
    monkeypatch.setattr(media_power, "pyautogui", mock_pyautogui, raising=False)
    return mock_pyautogui


@pytest.fixture
def without_pyautogui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forces _PYAUTOGUI_AVAILABLE = False regardless of real install state."""
    monkeypatch.setattr(media_power, "_PYAUTOGUI_AVAILABLE", False)


@pytest.fixture
def with_pycaw(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    """
    Forces _PYCAW_AVAILABLE = True and mocks the module's own pycaw helper
    functions directly — the pragmatic testing boundary for COM interop,
    per the Phase 4.5A audit rationale.
    """
    monkeypatch.setattr(media_power, "_PYCAW_AVAILABLE", True)
    mocks = {
        "set_volume": Mock(return_value=True),
        "get_volume": Mock(return_value=0.5),
        "set_mute":   Mock(return_value=True),
    }
    monkeypatch.setattr(media_power, "_set_volume_pycaw", mocks["set_volume"])
    monkeypatch.setattr(media_power, "_get_volume_pycaw", mocks["get_volume"])
    monkeypatch.setattr(media_power, "_set_mute_pycaw",   mocks["set_mute"])
    return mocks


@pytest.fixture
def without_pycaw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_power, "_PYCAW_AVAILABLE", False)


@pytest.fixture
def windows_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """SystemPowerTool hard-blocks on non-Windows before anything else runs."""
    monkeypatch.setattr(media_power.sys, "platform", "win32")


# ══════════════════════════════════════════════════════════════════════════════
# MediaControlTool
# ══════════════════════════════════════════════════════════════════════════════

class TestMediaControlTool:

    def test_happy_path_play_pause(self, with_pyautogui: Mock) -> None:
        result = MediaControlTool().execute(action=MediaAction.PLAY_PAUSE)
        assert result.startswith("Success:")
        with_pyautogui.press.assert_called_once_with("playpause")

    def test_happy_path_next(self, with_pyautogui: Mock) -> None:
        result = MediaControlTool().execute(action=MediaAction.NEXT)
        assert result.startswith("Success:")
        with_pyautogui.press.assert_called_once_with("nexttrack")

    def test_happy_path_volume_up_with_pycaw(
        self, with_pycaw: dict[str, Mock]
    ) -> None:
        result = MediaControlTool().execute(action=MediaAction.VOLUME_UP)

        assert result.startswith("Success:")
        with_pycaw["get_volume"].assert_called_once()
        with_pycaw["set_volume"].assert_called_once()
        # 0.5 (mocked current) + 0.10 step = 0.6
        new_level_arg = with_pycaw["set_volume"].call_args[0][0]
        assert abs(new_level_arg - 0.6) < 1e-6

    def test_volume_up_falls_back_to_pyautogui_without_pycaw(
        self, without_pycaw: None, with_pyautogui: Mock
    ) -> None:
        result = MediaControlTool().execute(action=MediaAction.VOLUME_UP)

        assert result.startswith("Success:")
        with_pyautogui.press.assert_called_once_with("volumeup", presses=10)

    def test_mute_via_pycaw(self, with_pycaw: dict[str, Mock]) -> None:
        result = MediaControlTool().execute(action=MediaAction.MUTE)
        assert result.startswith("Success:")
        with_pycaw["set_mute"].assert_called_once_with(True)

    def test_volume_max_via_pycaw(self, with_pycaw: dict[str, Mock]) -> None:
        result = MediaControlTool().execute(action=MediaAction.VOLUME_MAX)
        assert result.startswith("Success:")
        with_pycaw["set_volume"].assert_called_once_with(1.0)

    def test_invalid_args_bad_action_string(self) -> None:
        with pytest.raises(ValidationError):
            MediaControlSchema(action="not_a_real_action")

    def test_invalid_args_missing_action(self) -> None:
        with pytest.raises(ValidationError):
            MediaControlSchema()

    def test_missing_dependency_both_unavailable(
        self, without_pyautogui: None, without_pycaw: None
    ) -> None:
        """
        Neither pyautogui nor pycaw available — every action must return
        the graceful missing-dependency error, never raise NameError/AttributeError.
        """
        result = MediaControlTool().execute(action=MediaAction.VOLUME_UP)
        assert result.startswith("Error:")
        assert "pyautogui" in result.lower()

    def test_pycaw_available_but_returns_none_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, with_pyautogui: Mock
    ) -> None:
        """
        _PYCAW_AVAILABLE True but _get_volume_pycaw() returns None
        (COM interface construction failed at call time) — must fall
        through to pyautogui, not crash.
        """
        monkeypatch.setattr(media_power, "_PYCAW_AVAILABLE", True)
        monkeypatch.setattr(media_power, "_get_volume_pycaw", Mock(return_value=None))

        result = MediaControlTool().execute(action=MediaAction.VOLUME_UP)

        assert result.startswith("Success:")
        with_pyautogui.press.assert_called_once_with("volumeup", presses=10)

    def test_pyautogui_press_exception_handled(
        self, with_pyautogui: Mock
    ) -> None:
        with_pyautogui.press.side_effect = RuntimeError("input backend crashed")

        result = MediaControlTool().execute(action=MediaAction.PLAY_PAUSE)

        assert result.startswith("Error:")

    def test_unknown_action_enum_gap(self, with_pyautogui: Mock) -> None:
        """
        Defensive: if a MediaAction value somehow bypasses the handler
        dict mapping, the tool must return a structured error, not raise.
        This exercises the `handler is None` branch.
        """
        tool = MediaControlTool()
        # Bypass Pydantic to directly test the internal dispatch guard.
        result = tool.execute(action="totally_unmapped_action")  # type: ignore[arg-type]
        assert result.startswith("Error:")


# ══════════════════════════════════════════════════════════════════════════════
# SystemPowerTool
# ══════════════════════════════════════════════════════════════════════════════

class TestSystemPowerTool:

    def test_happy_path_shutdown(self, windows_platform: None) -> None:
        mock_run = Mock(return_value=Mock(returncode=0))
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "tools.media_power.subprocess.run", mock_run
        ):
            result = SystemPowerTool().execute(action=PowerAction.SHUTDOWN)

        assert result.startswith("Success:")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[:2] == ["shutdown", "/s"]

    def test_happy_path_restart(self, windows_platform: None, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_run = Mock(return_value=Mock(returncode=0))
        monkeypatch.setattr(media_power.subprocess, "run", mock_run)

        result = SystemPowerTool().execute(action=PowerAction.RESTART)

        assert result.startswith("Success:")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[:2] == ["shutdown", "/r"]

    def test_happy_path_abort(self, windows_platform: None, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_run = Mock(return_value=Mock(returncode=0))
        monkeypatch.setattr(media_power.subprocess, "run", mock_run)

        result = SystemPowerTool().execute(action=PowerAction.ABORT)

        assert result.startswith("Success:")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == ["shutdown", "/a"]

    def test_invalid_args_bad_action(self) -> None:
        with pytest.raises(ValidationError):
            SystemPowerSchema(action="destroy_everything")

    def test_non_windows_blocked_before_any_subprocess_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Malicious/erroneous input isn't the concern here — the real risk
        is SystemPowerTool executing OS commands on an unsupported platform.
        Confirm the guard fires before subprocess.run is ever reached.
        """
        monkeypatch.setattr(media_power.sys, "platform", "linux")
        mock_run = Mock()
        monkeypatch.setattr(media_power.subprocess, "run", mock_run)

        result = SystemPowerTool().execute(action=PowerAction.SHUTDOWN)

        assert result.startswith("Error:")
        assert "windows" in result.lower()
        mock_run.assert_not_called()

    def test_abort_with_no_pending_shutdown(
        self, windows_platform: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CalledProcessError on abort means there was nothing to abort."""
        monkeypatch.setattr(
            media_power.subprocess,
            "run",
            Mock(side_effect=subprocess.CalledProcessError(returncode=1, cmd=["shutdown", "/a"])),
        )

        result = SystemPowerTool().execute(action=PowerAction.ABORT)

        assert result.startswith("Error:")
        assert "no pending" in result.lower()

    def test_timeout(self, windows_platform: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            media_power.subprocess,
            "run",
            Mock(side_effect=subprocess.TimeoutExpired(cmd="shutdown", timeout=10)),
        )

        result = SystemPowerTool().execute(action=PowerAction.SHUTDOWN)

        assert result.startswith("Error:")
        assert "timed out" in result.lower()

    def test_permission_failure(self, windows_platform: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            media_power.subprocess,
            "run",
            Mock(side_effect=PermissionError("Access is denied")),
        )

        result = SystemPowerTool().execute(action=PowerAction.RESTART)

        assert result.startswith("Error:")

    def test_shutdown_binary_missing(self, windows_platform: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            media_power.subprocess,
            "run",
            Mock(side_effect=FileNotFoundError()),
        )

        result = SystemPowerTool().execute(action=PowerAction.SHUTDOWN)

        assert result.startswith("Error:")
        assert "windows" in result.lower() or "not found" in result.lower()

    def test_called_process_error_generic(
        self, windows_platform: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            media_power.subprocess,
            "run",
            Mock(side_effect=subprocess.CalledProcessError(returncode=5, cmd=["shutdown", "/s"])),
        )

        result = SystemPowerTool().execute(action=PowerAction.SHUTDOWN)

        assert result.startswith("Error:")