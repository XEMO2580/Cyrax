"""
tests/tools/test_pc_actions.py — Certification suite for tools/pc_actions.py

NO REAL SUBPROCESSES: subprocess.Popen and subprocess.run are patched in
every test. NO REAL BROWSER LAUNCHES: webbrowser.open is patched. NO REAL
KEYSTROKES: pyautogui is blocked or mocked.

NOTE ON TEST TIER: tool.execute() is called directly, bypassing
ToolRegistry.execute_tool(). Registry-level integration is a separate suite.
"""

from __future__ import annotations

import subprocess
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from tools.pc_actions import (
    OpenAppTool, OpenAppSchema,
    CloseAppTool, CloseAppSchema,
    TypeTextTool, TypeTextSchema,
    ALLOWED_APPS,
    WEB_FALLBACKS,
)


# ══════════════════════════════════════════════════════════════════════════════
# OpenAppTool
# ══════════════════════════════════════════════════════════════════════════════

class TestOpenAppTool:

    def test_happy_path_web_fallback(self) -> None:
        mock_open = Mock()
        with patch("tools.pc_actions.webbrowser.open", mock_open):
            result = OpenAppTool().execute(app="youtube")

        assert result.startswith("Success:")
        mock_open.assert_called_once_with(WEB_FALLBACKS["youtube"])

    def test_happy_path_native_app(self) -> None:
        mock_popen = Mock()
        with patch("tools.pc_actions.subprocess.Popen", mock_popen):
            result = OpenAppTool().execute(app="chrome")

        assert result.startswith("Success:")
        mock_popen.assert_called_once()
        assert mock_popen.call_args[0][0] == ALLOWED_APPS["chrome"]

    def test_case_insensitive_and_whitespace_stripped(self) -> None:
        mock_open = Mock()
        with patch("tools.pc_actions.webbrowser.open", mock_open):
            result = OpenAppTool().execute(app="  YouTube  ")

        assert result.startswith("Success:")

    def test_invalid_args_missing_app(self) -> None:
        with pytest.raises(ValidationError):
            OpenAppSchema()

    def test_malicious_input_unwhitelisted_app_blocked(self) -> None:
        """Attempt to open something outside both whitelists must hard-block."""
        mock_popen = Mock()
        mock_open  = Mock()

        with patch("tools.pc_actions.subprocess.Popen", mock_popen), \
             patch("tools.pc_actions.webbrowser.open", mock_open):
            result = OpenAppTool().execute(app="malware.exe")

        assert result.startswith("Security Block:")
        mock_popen.assert_not_called()
        mock_open.assert_not_called()

    def test_empty_input(self) -> None:
        result = OpenAppTool().execute(app="")
        assert result.startswith("Security Block:")

    def test_large_input_does_not_crash(self) -> None:
        """Oversized app name — no whitelist entry can match, no crash."""
        mock_popen = Mock()
        huge_name  = "a" * 100_000

        with patch("tools.pc_actions.subprocess.Popen", mock_popen):
            result = OpenAppTool().execute(app=huge_name)

        assert result.startswith("Security Block:")
        mock_popen.assert_not_called()

    def test_native_app_executable_not_found(self) -> None:
        with patch(
            "tools.pc_actions.subprocess.Popen",
            Mock(side_effect=FileNotFoundError()),
        ):
            result = OpenAppTool().execute(app="chrome")

        assert result.startswith("Error:")
        assert "could not find" in result.lower()

    def test_permission_failure(self) -> None:
        """OS denies process launch — caught, not propagated."""
        with patch(
            "tools.pc_actions.subprocess.Popen",
            Mock(side_effect=PermissionError("Access is denied")),
        ):
            result = OpenAppTool().execute(app="chrome")

        assert result.startswith("Error:")

    def test_web_open_failure_handled(self) -> None:
        with patch(
            "tools.pc_actions.webbrowser.open",
            Mock(side_effect=RuntimeError("no browser configured")),
        ):
            result = OpenAppTool().execute(app="youtube")

        assert result.startswith("Error:")


# ══════════════════════════════════════════════════════════════════════════════
# CloseAppTool
# ══════════════════════════════════════════════════════════════════════════════

class TestCloseAppTool:

    def test_happy_path(self) -> None:
        mock_run = Mock(return_value=Mock(returncode=0, stderr=""))
        with patch("tools.pc_actions.subprocess.run", mock_run):
            result = CloseAppTool().execute(app="chrome")

        assert result.startswith("Success:")
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[0] == "taskkill"
        assert "chrome.exe" in called_cmd

    def test_invalid_args_missing_app(self) -> None:
        with pytest.raises(ValidationError):
            CloseAppSchema()

    def test_malicious_input_unwhitelisted_app_blocked(self) -> None:
        mock_run = Mock()
        with patch("tools.pc_actions.subprocess.run", mock_run):
            result = CloseAppTool().execute(app="critical_system_process.exe")

        assert result.startswith("Security Block:")
        mock_run.assert_not_called()

    def test_empty_input(self) -> None:
        mock_run = Mock()
        with patch("tools.pc_actions.subprocess.run", mock_run):
            result = CloseAppTool().execute(app="")

        assert result.startswith("Security Block:")
        mock_run.assert_not_called()

    def test_large_input_does_not_crash(self) -> None:
        mock_run = Mock()
        with patch("tools.pc_actions.subprocess.run", mock_run):
            result = CloseAppTool().execute(app="x" * 50_000)

        assert result.startswith("Security Block:")
        mock_run.assert_not_called()

    def test_process_not_running(self) -> None:
        mock_run = Mock(return_value=Mock(returncode=128, stderr="not found"))
        with patch("tools.pc_actions.subprocess.run", mock_run):
            result = CloseAppTool().execute(app="notepad")

        assert result.startswith("Error:")
        assert "not currently running" in result.lower()

    def test_timeout(self) -> None:
        with patch(
            "tools.pc_actions.subprocess.run",
            Mock(side_effect=subprocess.TimeoutExpired(cmd="taskkill", timeout=10)),
        ):
            result = CloseAppTool().execute(app="chrome")

        assert result.startswith("Error:")
        assert "timed out" in result.lower()

    def test_permission_failure(self) -> None:
        with patch(
            "tools.pc_actions.subprocess.run",
            Mock(side_effect=PermissionError("Access is denied")),
        ):
            result = CloseAppTool().execute(app="chrome")

        assert result.startswith("Error:")

    def test_taskkill_binary_missing_non_windows(self) -> None:
        """taskkill not found (e.g. non-Windows host) — caught cleanly."""
        with patch(
            "tools.pc_actions.subprocess.run",
            Mock(side_effect=FileNotFoundError()),
        ):
            result = CloseAppTool().execute(app="chrome")

        assert result.startswith("Error:")
        assert "not available" in result.lower()

    def test_unexpected_nonzero_returncode(self) -> None:
        mock_run = Mock(return_value=Mock(returncode=1, stderr="access denied"))
        with patch("tools.pc_actions.subprocess.run", mock_run):
            result = CloseAppTool().execute(app="notepad")

        assert result.startswith("Error:")


# ══════════════════════════════════════════════════════════════════════════════
# TypeTextTool
# ══════════════════════════════════════════════════════════════════════════════

class TestTypeTextTool:

    def test_happy_path_no_enter(self) -> None:
        mock_pyautogui = Mock()
        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}), \
             patch("tools.pc_actions.time.sleep", Mock()):
            result = TypeTextTool().execute(text="hello world")

        assert result.startswith("Success:")
        mock_pyautogui.write.assert_called_once_with("hello world", interval=0.01)
        mock_pyautogui.press.assert_not_called()

    def test_happy_path_with_enter(self) -> None:
        mock_pyautogui = Mock()
        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}), \
             patch("tools.pc_actions.time.sleep", Mock()):
            result = TypeTextTool().execute(text="search query", press_enter=True)

        assert result.startswith("Success:")
        mock_pyautogui.press.assert_called_once_with("enter")

    def test_invalid_args_missing_text(self) -> None:
        with pytest.raises(ValidationError):
            TypeTextSchema()

    def test_missing_dependency(self, block_import) -> None:
        with block_import("pyautogui"):
            result = TypeTextTool().execute(text="hello")

        assert result.startswith("Error:")
        assert "pyautogui" in result.lower()

    def test_malicious_input_newlines_and_nulls_stripped(self) -> None:
        """
        Payload attempting to smuggle an implicit Enter keystroke via
        embedded newline must be sanitised before reaching pyautogui.write.
        """
        mock_pyautogui = Mock()
        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}), \
             patch("tools.pc_actions.time.sleep", Mock()):
            result = TypeTextTool().execute(
                text="rm -rf /\n\x00malicious", press_enter=False
            )

        assert result.startswith("Success:")
        written_text = mock_pyautogui.write.call_args[0][0]
        assert "\n" not in written_text
        assert "\x00" not in written_text

    def test_empty_input_raw(self) -> None:
        mock_pyautogui = Mock()
        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = TypeTextTool().execute(text="")

        assert result.startswith("Error:")
        assert "empty" in result.lower()
        mock_pyautogui.write.assert_not_called()

    def test_empty_after_sanitisation(self) -> None:
        """Text consisting only of stripped characters becomes empty."""
        mock_pyautogui = Mock()
        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}):
            result = TypeTextTool().execute(text="\n\r\x00")

        assert result.startswith("Error:")
        assert "empty after sanitisation" in result.lower()
        mock_pyautogui.write.assert_not_called()

    def test_large_input(self) -> None:
        """
        AUDIT FINDING: TypeTextSchema has no max_length constraint (unlike
        ClipboardWriteSchema in desktop_ops.py). This test documents current
        unguarded behaviour rather than asserting a limit that doesn't exist.
        Flagged as a recommended (non-blocking) hardening item.
        """
        mock_pyautogui = Mock()
        huge_text = "a" * 200_000

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}), \
             patch("tools.pc_actions.time.sleep", Mock()):
            result = TypeTextTool().execute(text=huge_text)

        assert result.startswith("Success:")
        mock_pyautogui.write.assert_called_once()

    def test_pyautogui_write_failure_handled(self) -> None:
        mock_pyautogui = Mock()
        mock_pyautogui.write.side_effect = RuntimeError("display not accessible")

        with patch.dict("sys.modules", {"pyautogui": mock_pyautogui}), \
             patch("tools.pc_actions.time.sleep", Mock()):
            result = TypeTextTool().execute(text="hello")

        assert result.startswith("Error:")
        assert "keyboard simulation failed" in result.lower()