"""
tests/tools/test_email_ops.py — Certification suite for tools/email_ops.py

NO REAL NETWORK CALLS: smtplib.SMTP and imaplib.IMAP4_SSL are fully mocked
at the module-path level (tools.email_ops.smtplib.SMTP, etc.) — no socket
is ever opened.

NOTE: smtplib/imaplib are Python stdlib, always present. No
"missing dependency" test category applies to this file (see Phase 4.5A
audit — this is a deliberate omission, not an oversight).
"""

from __future__ import annotations

import imaplib
import smtplib
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from tools.email_ops import (
    SendEmailTool, SendEmailSchema,
    ReadEmailTool, ReadEmailSchema,
    _MAX_EMAILS_TO_READ,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures — credentials
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def configured_credentials(monkeypatch: pytest.MonkeyPatch):
    """
    Ensures settings.CYRAX_EMAIL / CYRAX_EMAIL_PASSWORD are present for
    every test in this file by default. Tests targeting the missing-
    credential path override this explicitly per-test.
    """
    from config import settings as settings_module

    monkeypatch.setattr(settings_module.settings, "CYRAX_EMAIL", "cyrax@example.com")
    monkeypatch.setattr(settings_module.settings, "CYRAX_EMAIL_PASSWORD", "app_password_123")
    monkeypatch.setattr(settings_module.settings, "SMTP_SERVER", "smtp.gmail.com", raising=False)
    monkeypatch.setattr(settings_module.settings, "SMTP_PORT", 587, raising=False)
    monkeypatch.setattr(settings_module.settings, "IMAP_SERVER", "imap.gmail.com", raising=False)


# ══════════════════════════════════════════════════════════════════════════════
# SendEmailTool
# ══════════════════════════════════════════════════════════════════════════════

class TestSendEmailTool:

    def test_happy_path(self) -> None:
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.__enter__ = Mock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__  = Mock(return_value=False)

        with patch("tools.email_ops.smtplib.SMTP", return_value=mock_smtp_instance):
            result = SendEmailTool().execute(
                to_address="recipient@example.com",
                subject="Test Subject",
                body="Test body content.",
            )

        assert result.startswith("Success:")
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with(
            "cyrax@example.com", "app_password_123"
        )
        mock_smtp_instance.sendmail.assert_called_once()

    def test_invalid_args_missing_subject(self) -> None:
        with pytest.raises(ValidationError):
            SendEmailSchema(to_address="a@b.com", body="hi")

    def test_missing_credentials_email(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from config import settings as settings_module
        monkeypatch.setattr(settings_module.settings, "CYRAX_EMAIL", "")

        result = SendEmailTool().execute(
            to_address="a@b.com", subject="s", body="b"
        )

        assert result.startswith("Error:")
        assert "CYRAX_EMAIL" in result

    def test_missing_credentials_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from config import settings as settings_module
        monkeypatch.setattr(settings_module.settings, "CYRAX_EMAIL_PASSWORD", "")

        result = SendEmailTool().execute(
            to_address="a@b.com", subject="s", body="b"
        )

        assert result.startswith("Error:")
        assert "CYRAX_EMAIL_PASSWORD" in result

    def test_smtp_authentication_failure(self) -> None:
        """CRITICAL per task directive: SMTP auth failure must be caught cleanly."""
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.__enter__ = Mock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__  = Mock(return_value=False)
        mock_smtp_instance.login.side_effect = smtplib.SMTPAuthenticationError(
            535, b"5.7.8 Username and Password not accepted"
        )

        with patch("tools.email_ops.smtplib.SMTP", return_value=mock_smtp_instance):
            result = SendEmailTool().execute(
                to_address="a@b.com", subject="s", body="b"
            )

        assert result.startswith("Error:")
        assert "authentication failed" in result.lower()
        assert "App Password" in result

    def test_malicious_input_empty_recipient(self) -> None:
        mock_smtp_instance = MagicMock()
        with patch("tools.email_ops.smtplib.SMTP", return_value=mock_smtp_instance):
            result = SendEmailTool().execute(to_address="   ", subject="s", body="b")

        assert result.startswith("Error:")
        assert "recipient" in result.lower()
        mock_smtp_instance.assert_not_called()

    def test_empty_body(self) -> None:
        result = SendEmailTool().execute(to_address="a@b.com", subject="s", body="")
        assert result.startswith("Error:")
        assert "body" in result.lower()

    def test_large_body_rejected_by_schema(self) -> None:
        from tools.email_ops import _MAX_BODY_CHARS
        with pytest.raises(ValidationError):
            SendEmailSchema(
                to_address="a@b.com", subject="s", body="x" * (_MAX_BODY_CHARS + 1)
            )

    def test_recipients_refused(self) -> None:
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.__enter__ = Mock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__  = Mock(return_value=False)
        mock_smtp_instance.sendmail.side_effect = smtplib.SMTPRecipientsRefused(
            {"bad@address.invalid": (550, b"No such user")}
        )

        with patch("tools.email_ops.smtplib.SMTP", return_value=mock_smtp_instance):
            result = SendEmailTool().execute(
                to_address="bad@address.invalid", subject="s", body="b"
            )

        assert result.startswith("Error:")
        assert "refused" in result.lower()

    def test_connection_timeout(self) -> None:
        with patch(
            "tools.email_ops.smtplib.SMTP",
            side_effect=__import__("socket").timeout(),
        ):
            result = SendEmailTool().execute(to_address="a@b.com", subject="s", body="b")

        assert result.startswith("Error:")
        assert "timed out" in result.lower()

    def test_smtp_connect_error(self) -> None:
        with patch(
            "tools.email_ops.smtplib.SMTP",
            side_effect=smtplib.SMTPConnectError(421, b"Cannot connect"),
        ):
            result = SendEmailTool().execute(to_address="a@b.com", subject="s", body="b")

        assert result.startswith("Error:")
        assert "connect" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# ReadEmailTool
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_imap(
    search_status: str = "OK",
    search_ids: bytes = b"1 2 3",
    fetch_status: str = "OK",
    fetch_raw_message: bytes | None = None,
    login_side_effect: Exception | None = None,
    select_status: str = "OK",
) -> MagicMock:
    """
    Builds a MagicMock replicating imaplib.IMAP4_SSL's real (status, data)
    tuple-return shape, including the nested tuple structure FETCH responses
    actually use — not a simplified stand-in.
    """
    mock_mail = MagicMock()

    if login_side_effect:
        mock_mail.login.side_effect = login_side_effect

    mock_mail.select.return_value = (select_status, [b"3"])
    mock_mail.search.return_value = (search_status, [search_ids])

    if fetch_raw_message is None:
        fetch_raw_message = (
            b"From: sender@example.com\r\n"
            b"Subject: Test Subject\r\n"
            b"Date: Mon, 1 Jan 2026 12:00:00 +0000\r\n"
            b"Content-Type: text/plain\r\n\r\n"
            b"This is the plain text body."
        )

    # Real imaplib FETCH shape: [ (b'1 (BODY[] {123}', raw_bytes), b')' ]
    mock_mail.fetch.return_value = (
        fetch_status,
        [(b"1 (BODY[] {%d}" % len(fetch_raw_message), fetch_raw_message)],
    )

    return mock_mail


class TestReadEmailTool:

    def test_happy_path(self) -> None:
        mock_mail = _make_mock_imap()
        with patch("tools.email_ops.imaplib.IMAP4_SSL", return_value=mock_mail):
            result = ReadEmailTool().execute()

        assert "Unread Emails" in result
        assert "sender@example.com" in result
        assert "Test Subject" in result
        assert "This is the plain text body." in result
        mock_mail.close.assert_called_once()
        mock_mail.logout.assert_called_once()

    def test_invalid_args_max_results_out_of_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ReadEmailSchema(max_results=_MAX_EMAILS_TO_READ + 1)

    def test_invalid_args_max_results_zero(self) -> None:
        with pytest.raises(ValidationError):
            ReadEmailSchema(max_results=0)

    def test_imap_authentication_failure(self) -> None:
        """CRITICAL per task directive: IMAP auth failure must be caught cleanly."""
        mock_mail = _make_mock_imap(
            login_side_effect=imaplib.IMAP4.error("[AUTHENTICATIONFAILED] Invalid credentials")
        )
        with patch("tools.email_ops.imaplib.IMAP4_SSL", return_value=mock_mail):
            result = ReadEmailTool().execute()

        assert result.startswith("Error:")
        assert "authentication failed" in result.lower()
        # Connection must still be closed even on auth failure, via finally block.
        mock_mail.logout.assert_called_once()

    def test_imap_search_failure(self) -> None:
        """CRITICAL per task directive: IMAP SEARCH command failure handled."""
        mock_mail = _make_mock_imap(search_status="NO")
        with patch("tools.email_ops.imaplib.IMAP4_SSL", return_value=mock_mail):
            result = ReadEmailTool().execute()

        assert result.startswith("Error:")
        assert "search" in result.lower()

    def test_mailbox_select_failure(self) -> None:
        mock_mail = _make_mock_imap(select_status="NO")
        with patch("tools.email_ops.imaplib.IMAP4_SSL", return_value=mock_mail):
            result = ReadEmailTool().execute(mailbox="NONEXISTENT")

        assert result.startswith("Error:")
        assert "could not open mailbox" in result.lower()

    def test_no_unread_messages(self) -> None:
        mock_mail = _make_mock_imap(search_ids=b"")
        with patch("tools.email_ops.imaplib.IMAP4_SSL", return_value=mock_mail):
            result = ReadEmailTool().execute()

        assert "No unread messages" in result

    def test_missing_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from config import settings as settings_module
        monkeypatch.setattr(settings_module.settings, "CYRAX_EMAIL", "")

        result = ReadEmailTool().execute()

        assert result.startswith("Error:")
        assert "CYRAX_EMAIL" in result

    def test_connection_timeout(self) -> None:
        with patch(
            "tools.email_ops.imaplib.IMAP4_SSL",
            side_effect=__import__("socket").timeout(),
        ):
            result = ReadEmailTool().execute()

        assert result.startswith("Error:")
        assert "connect" in result.lower()

    def test_html_only_message_stripped_to_text(self) -> None:
        """
        Message with only an HTML part — must fall back to stripped
        plain text, not return raw markup to the LLM context.
        """
        html_msg = (
            b"From: sender@example.com\r\n"
            b"Subject: HTML Email\r\n"
            b"Content-Type: text/html\r\n\r\n"
            b"<html><body><p>Hello <b>World</b></p><script>evil()</script></body></html>"
        )
        mock_mail = _make_mock_imap(fetch_raw_message=html_msg)

        with patch("tools.email_ops.imaplib.IMAP4_SSL", return_value=mock_mail):
            result = ReadEmailTool().execute()

        assert "Hello" in result
        assert "World" in result
        assert "<script>" not in result
        assert "evil()" not in result

    def test_fetch_failure_for_individual_message_does_not_crash_whole_read(self) -> None:
        """Malformed/corrupt single message must not abort the entire read."""
        mock_mail = _make_mock_imap(search_ids=b"1")
        mock_mail.fetch.return_value = ("NO", [None])

        with patch("tools.email_ops.imaplib.IMAP4_SSL", return_value=mock_mail):
            result = ReadEmailTool().execute()

        assert "Error" in result  # per-message error, embedded in output
        # Connection is still cleanly closed.
        mock_mail.logout.assert_called_once()

    def test_connection_always_closed_on_exception(self) -> None:
        """
        Even when an unexpected exception occurs mid-processing, the
        finally block must still close/logout the IMAP connection.
        """
        mock_mail = _make_mock_imap()
        mock_mail.select.side_effect = RuntimeError("unexpected failure")

        with patch("tools.email_ops.imaplib.IMAP4_SSL", return_value=mock_mail):
            with pytest.raises(RuntimeError):
                ReadEmailTool().execute()

        mock_mail.close.assert_called_once()
        mock_mail.logout.assert_called_once()