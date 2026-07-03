"""
tests/tools/test_desktop_ops.py — Certification suite for tools/desktop_ops.py

Scope: ClipboardReadTool, ClipboardWriteTool, SendNotificationTool.

NOTE ON TEST TIER: These tests call tool.execute() directly, bypassing
ToolRegistry.execute_tool(). Registry-level integration is covered
separately in tests/integration/test_registry_gate.py.

Mocking boundaries:
  - pyperclip.copy / pyperclip.paste are mocked. No real clipboard I/O.
  - plyer.notification.notify is mocked. No real desktop notifications fire.

Requires: tests/conftest.py providing the `block_import` fixture.
Requires (dev dependency, not mocked): real `pyperclip` package installed,
since test_pyperclip_exception_handled imports pyperclip.PyperclipException
directly to assert against the real exception type.
"""

from __future__ import annotations

import threading
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from tools.desktop_ops import (
    ClipboardReadTool,
    ClipboardWriteTool,
    SendNotificationTool,
    ClipboardReadSchema,
    ClipboardWriteSchema,
    SendNotificationSchema,
    _MAX_CLIPBOARD_READ_CHARS,
    _MAX_CLIPBOARD_WRITE_CHARS,
    _MAX_NOTIFICATION_TITLE,
)


# ══════════════════════════════════════════════════════════════════════════════
# ClipboardReadTool
# ══════════════════════════════════════════════════════════════════════════════

class TestClipboardReadTool:

    def test_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_paste = Mock(return_value="hello from the clipboard")
        with patch("pyperclip.paste", mock_paste):
            result = ClipboardReadTool().execute()

        assert result == "hello from the clipboard"
        mock_paste.assert_called_once()

    def test_missing_dependency(self, block_import) -> None:
        with block_import("pyperclip"):
            result = ClipboardReadTool().execute()

        assert result.startswith("Error:")
        assert "pyperclip" in result.lower()

    def test_empty_clipboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with patch("pyperclip.paste", Mock(return_value="   ")):
            result = ClipboardReadTool().execute()

        assert result.startswith("Error:")
        assert "empty" in result.lower()

    def test_none_clipboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with patch("pyperclip.paste", Mock(return_value=None)):
            result = ClipboardReadTool().execute()

        assert result.startswith("Error:")

    def test_non_text_clipboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with patch("pyperclip.paste", Mock(return_value=b"\x89PNG\r\n\x1a\n")):
            result = ClipboardReadTool().execute()

        assert result.startswith("Error:")
        assert "non-text" in result.lower()

    def test_pyperclip_exception_handled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pyperclip

        with patch(
            "pyperclip.paste",
            Mock(side_effect=pyperclip.PyperclipException("no clipboard mechanism")),
        ):
            result = ClipboardReadTool().execute()

        assert result.startswith("Error:")

    def test_large_clipboard_content_is_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        oversized = "A" * (_MAX_CLIPBOARD_READ_CHARS + 10_000)

        with patch("pyperclip.paste", Mock(return_value=oversized)):
            result = ClipboardReadTool().execute()

        assert len(result) < len(oversized)
        assert "truncated" in result.lower()
        assert result.startswith("A" * 100)

    def test_schema_accepts_no_args(self) -> None:
        schema = ClipboardReadSchema()
        assert schema.model_dump() == {}


# ══════════════════════════════════════════════════════════════════════════════
# ClipboardWriteTool
# ══════════════════════════════════════════════════════════════════════════════

class TestClipboardWriteTool:

    def test_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_copy = Mock()
        with patch("pyperclip.copy", mock_copy):
            result = ClipboardWriteTool().execute(text="hello world")

        assert result.startswith("Success:")
        mock_copy.assert_called_once_with("hello world")

    def test_invalid_args_missing_text(self) -> None:
        with pytest.raises(ValidationError):
            ClipboardWriteSchema()

    def test_missing_dependency(self, block_import) -> None:
        with block_import("pyperclip"):
            result = ClipboardWriteTool().execute(text="hello")

        assert result.startswith("Error:")
        assert "pyperclip" in result.lower()

    def test_null_byte_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_copy = Mock()
        with patch("pyperclip.copy", mock_copy):
            result = ClipboardWriteTool().execute(text="hello\x00world\x00")

        assert result.startswith("Success:")
        written_value = mock_copy.call_args[0][0]
        assert "\x00" not in written_value
        assert written_value == "helloworld"

    def test_empty_string_via_schema(self) -> None:
        mock_copy = Mock()
        with patch("pyperclip.copy", mock_copy):
            result = ClipboardWriteTool().execute(text="")

        assert result.startswith("Success:")
        mock_copy.assert_called_once_with("")

    def test_large_input_rejected_by_schema(self) -> None:
        oversized = "A" * (_MAX_CLIPBOARD_WRITE_CHARS + 1)

        with pytest.raises(ValidationError):
            ClipboardWriteSchema(text=oversized)

    def test_large_input_rejected_by_execute_directly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        oversized = "A" * (_MAX_CLIPBOARD_WRITE_CHARS + 1)
        mock_copy = Mock()

        with patch("pyperclip.copy", mock_copy):
            result = ClipboardWriteTool().execute(text=oversized)

        assert result.startswith("Error:")
        assert "too long" in result.lower()
        mock_copy.assert_not_called()

    def test_pyperclip_exception_handled(self) -> None:
        import pyperclip

        with patch(
            "pyperclip.copy",
            Mock(side_effect=pyperclip.PyperclipException("clipboard locked")),
        ):
            result = ClipboardWriteTool().execute(text="hello")

        assert result.startswith("Error:")


# ══════════════════════════════════════════════════════════════════════════════
# SendNotificationTool
# ══════════════════════════════════════════════════════════════════════════════

class TestSendNotificationTool:

    def test_happy_path(self) -> None:
        called = threading.Event()
        mock_notify = Mock(side_effect=lambda **kw: called.set())

        with patch("plyer.notification.notify", mock_notify):
            result = SendNotificationTool().execute(
                title="Test Title", message="Test message body."
            )

        assert result.startswith("Success:")
        assert called.wait(timeout=2.0), "Notification thread never fired."
        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        assert kwargs["title"] == "Test Title"
        assert kwargs["message"] == "Test message body."
        assert kwargs["app_name"] == "CYRAX"

    def test_invalid_args_missing_title(self) -> None:
        with pytest.raises(ValidationError):
            SendNotificationSchema(message="body only")

    def test_invalid_args_missing_message(self) -> None:
        with pytest.raises(ValidationError):
            SendNotificationSchema(title="title only")

    def test_missing_dependency(self, block_import) -> None:
        with block_import("plyer"):
            result = SendNotificationTool().execute(title="T", message="M")

        assert result.startswith("Error:")
        assert "plyer" in result.lower()

    def test_empty_title_after_strip(self) -> None:
        mock_notify = Mock()
        with patch("plyer.notification.notify", mock_notify):
            result = SendNotificationTool().execute(title="   ", message="valid message")

        assert result.startswith("Error:")
        assert "title" in result.lower()
        mock_notify.assert_not_called()

    def test_empty_message_after_strip(self) -> None:
        mock_notify = Mock()
        with patch("plyer.notification.notify", mock_notify):
            result = SendNotificationTool().execute(title="valid title", message="   ")

        assert result.startswith("Error:")
        assert "message" in result.lower()
        mock_notify.assert_not_called()

    def test_large_title_truncated_by_schema(self) -> None:
        with pytest.raises(ValidationError):
            SendNotificationSchema(
                title="X" * (_MAX_NOTIFICATION_TITLE + 1),
                message="valid",
            )

    def test_notify_exception_is_caught_in_thread_not_raised(self) -> None:
        error_logged = threading.Event()

        def _raise(**kwargs):
            error_logged.set()
            raise RuntimeError("OS notification backend unavailable")

        with patch("plyer.notification.notify", Mock(side_effect=_raise)):
            result = SendNotificationTool().execute(title="T", message="M")

        assert result.startswith("Success:")
        assert error_logged.wait(timeout=2.0), "Thread target never executed."