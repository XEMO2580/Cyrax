"""
tools/email_ops.py — CYRAX 3.0 Email Tools

Tools:
    SendEmailTool — Sends an email via SMTP with TLS
    ReadEmailTool — Reads recent unseen emails via IMAP

Security:
    Both tools — ADMIN (credentials are sensitive; reading/sending
    email is a high-trust action with real-world consequences)

Credentials sourced exclusively from config.settings:
    settings.CYRAX_EMAIL          — sender address
    settings.CYRAX_EMAIL_PASSWORD — app password or SMTP password
    settings.SMTP_SERVER          — SMTP hostname  (default: smtp.gmail.com)
    settings.SMTP_PORT            — SMTP port      (default: 587)
    settings.IMAP_SERVER          — IMAP hostname  (default: imap.gmail.com)

Required additions to config/settings.py:
    SMTP_SERVER: str = Field(default="smtp.gmail.com", ...)
    SMTP_PORT:   int = Field(default=587, ...)
    IMAP_SERVER: str = Field(default="imap.gmail.com", ...)

Context window protection (ReadEmailTool):
    - Only UNSEEN messages are fetched.
    - Maximum 5 most recent messages.
    - HTML parts are stripped; only plain-text parts are kept.
    - Each message body truncated to 2,000 characters.
    - Full output truncated to 8,000 characters.

All execute() methods are synchronous, never raise, and return plain
strings. Failures return strings starting with 'Error: '.
"""

from __future__ import annotations

import email
import email.header
import imaplib
import logging
import re
import smtplib
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from pydantic import BaseModel, EmailStr, Field

from security.auth import SecurityLevel
from tools.registry import BaseTool

logger = logging.getLogger(__name__)

# Hard limits.
_MAX_SUBJECT_CHARS:      int = 256
_MAX_BODY_CHARS:         int = 50_000
_MAX_EMAILS_TO_READ:     int = 5
_MAX_BODY_PER_EMAIL:     int = 2_000
_MAX_TOTAL_OUTPUT_CHARS: int = 8_000

# Network timeouts in seconds.
_SMTP_TIMEOUT: int = 15
_IMAP_TIMEOUT: int = 15


# ══════════════════════════════════════════════════════════════════════════════
# CREDENTIAL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _get_credentials() -> tuple[str, str] | str:
    """
    Returns (email_address, password) from settings, or an error string.
    Centralised so both tools use the same validation path.
    """
    from config.settings import settings

    address  = (settings.CYRAX_EMAIL or "").strip()
    password = (settings.CYRAX_EMAIL_PASSWORD or "").strip()

    if not address:
        return (
            "Error: CYRAX_EMAIL is not configured. "
            "Add CYRAX_EMAIL=your@email.com to your .env file."
        )
    if not password:
        return (
            "Error: CYRAX_EMAIL_PASSWORD is not configured. "
            "Add CYRAX_EMAIL_PASSWORD=yourpassword to your .env file."
        )
    return address, password


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: SEND EMAIL
# ══════════════════════════════════════════════════════════════════════════════

class SendEmailSchema(BaseModel):
    to_address: str = Field(
        ...,
        description="Recipient email address.",
    )
    subject: str = Field(
        ...,
        max_length=_MAX_SUBJECT_CHARS,
        description="Email subject line.",
    )
    body: str = Field(
        ...,
        max_length=_MAX_BODY_CHARS,
        description="Plain text body of the email.",
    )


class SendEmailTool(BaseTool):
    """
    Sends a plain-text email via SMTP with STARTTLS.

    Uses the credentials and server settings from config.settings.
    Establishes a fresh SMTP connection per call — no persistent
    connection pool is maintained to avoid stale connection issues.
    """

    name           = "SEND_EMAIL"
    description    = (
        "Sends a plain-text email to the specified address. "
        "Uses the CYRAX email credentials configured in settings."
    )
    security_level = SecurityLevel.ADMIN
    args_schema    = SendEmailSchema

    def execute(  # type: ignore[override]
        self,
        to_address: str,
        subject:    str,
        body:       str,
    ) -> str:
        from config.settings import settings

        # ── Credential validation ─────────────────────────────────────────────
        creds = _get_credentials()
        if isinstance(creds, str):
            return creds
        from_address, password = creds

        # ── Sanitise inputs ───────────────────────────────────────────────────
        to_address = to_address.strip()
        subject    = subject.strip()[:_MAX_SUBJECT_CHARS]
        body       = body.strip()

        if not to_address:
            return "Error: Recipient email address cannot be empty."
        if not subject:
            return "Error: Email subject cannot be empty."
        if not body:
            return "Error: Email body cannot be empty."

        # ── Build MIME message ────────────────────────────────────────────────
        msg                    = MIMEMultipart("alternative")
        msg["From"]            = from_address
        msg["To"]              = to_address
        msg["Subject"]         = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # ── SMTP send ─────────────────────────────────────────────────────────
        smtp_server = getattr(settings, "SMTP_SERVER", "smtp.gmail.com")
        smtp_port   = int(getattr(settings, "SMTP_PORT", 587))

        logger.info(
            f"[SEND_EMAIL] Sending to '{to_address}' via "
            f"{smtp_server}:{smtp_port}"
        )

        try:
            with smtplib.SMTP(
                smtp_server,
                smtp_port,
                timeout=_SMTP_TIMEOUT,
            ) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(from_address, password)
                server.sendmail(
                    from_address,
                    [to_address],
                    msg.as_string(),
                )

            logger.info(f"[SEND_EMAIL] Email sent to '{to_address}'.")
            return f"Success: Email sent to '{to_address}' with subject '{subject}'."

        except smtplib.SMTPAuthenticationError:
            logger.error("[SEND_EMAIL] SMTP authentication failed.")
            return (
                "Error: SMTP authentication failed. "
                "Check CYRAX_EMAIL and CYRAX_EMAIL_PASSWORD in your .env file. "
                "If using Gmail, ensure you are using an App Password."
            )
        except smtplib.SMTPRecipientsRefused:
            logger.warning(f"[SEND_EMAIL] Recipient refused: '{to_address}'.")
            return (
                f"Error: Recipient '{to_address}' was refused by the mail server. "
                f"Check the address and try again."
            )
        except smtplib.SMTPConnectError as exc:
            logger.error(f"[SEND_EMAIL] SMTP connection error: {exc}")
            return (
                f"Error: Could not connect to SMTP server '{smtp_server}:{smtp_port}'. "
                f"Check your network and SMTP settings."
            )
        except socket.timeout:
            return (
                f"Error: SMTP connection timed out after {_SMTP_TIMEOUT}s. "
                f"Check your network connection."
            )
        except smtplib.SMTPException as exc:
            logger.error(f"[SEND_EMAIL] SMTP error: {exc}")
            return f"Error: Email send failed — {exc}"
        except Exception as exc:
            logger.exception(f"[SEND_EMAIL] Unexpected error: {exc}")
            return f"Error: Unexpected error sending email — {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: READ EMAIL
# ══════════════════════════════════════════════════════════════════════════════

class ReadEmailSchema(BaseModel):
    max_results: int = Field(
        default=5,
        ge=1,
        le=_MAX_EMAILS_TO_READ,
        description=(
            f"Number of recent unseen emails to retrieve. "
            f"Maximum: {_MAX_EMAILS_TO_READ}."
        ),
    )
    mailbox: str = Field(
        default="INBOX",
        description="Mailbox folder to read from. Defaults to INBOX.",
    )


class ReadEmailTool(BaseTool):
    """
    Reads recent unseen (unread) emails via IMAP.

    Fetches only UNSEEN messages to avoid overwhelming the context
    window with already-processed mail. Maximum 5 messages per call.

    Content extraction:
        MIME multipart messages are walked — only text/plain parts are
        kept. HTML parts are stripped entirely. If a message contains
        only an HTML part, the HTML is reduced to plain text via tag
        removal before inclusion.

    Each message is capped at _MAX_BODY_PER_EMAIL characters.
    Total output is capped at _MAX_TOTAL_OUTPUT_CHARS characters.

    Does NOT mark messages as read — PEEK is used so the seen/unseen
    state of the mailbox is not altered by this tool.
    """

    name           = "READ_EMAIL"
    description    = (
        "Reads recent unread emails from the configured inbox. "
        f"Returns up to {_MAX_EMAILS_TO_READ} messages as plain text. "
        "Does not mark emails as read."
    )
    security_level = SecurityLevel.ADMIN
    args_schema    = ReadEmailSchema

    def execute(  # type: ignore[override]
        self,
        max_results: int = 5,
        mailbox:     str = "INBOX",
    ) -> str:
        from config.settings import settings

        # ── Credential validation ─────────────────────────────────────────────
        creds = _get_credentials()
        if isinstance(creds, str):
            return creds
        email_address, password = creds

        imap_server = getattr(settings, "IMAP_SERVER", "imap.gmail.com")
        max_results = min(max(1, max_results), _MAX_EMAILS_TO_READ)

        logger.info(
            f"[READ_EMAIL] Connecting to {imap_server} for '{email_address}'"
        )

        # ── IMAP connection ───────────────────────────────────────────────────
        try:
            mail = imaplib.IMAP4_SSL(imap_server, timeout=_IMAP_TIMEOUT)
        except (socket.timeout, OSError) as exc:
            return (
                f"Error: Could not connect to IMAP server '{imap_server}'. "
                f"Check your network and IMAP settings. Details: {exc}"
            )
        except Exception as exc:
            return f"Error: IMAP connection failed — {exc}"

        try:
            # ── Login ─────────────────────────────────────────────────────────
            try:
                mail.login(email_address, password)
            except imaplib.IMAP4.error as exc:
                return (
                    f"Error: IMAP authentication failed — {exc}. "
                    f"Check CYRAX_EMAIL and CYRAX_EMAIL_PASSWORD. "
                    f"If using Gmail, enable IMAP and use an App Password."
                )

            # ── Select mailbox ────────────────────────────────────────────────
            status, data = mail.select(mailbox, readonly=True)
            if status != "OK":
                return (
                    f"Error: Could not open mailbox '{mailbox}'. "
                    f"Check the mailbox name and try again."
                )

            # ── Search for UNSEEN messages ────────────────────────────────────
            status, message_numbers = mail.search(None, "UNSEEN")
            if status != "OK":
                return "Error: IMAP SEARCH command failed."

            ids_raw = message_numbers[0]
            if not ids_raw:
                return (
                    f"No unread messages found in '{mailbox}'. "
                    f"Your inbox is up to date."
                )

            # Most recent messages are at the end of the id list.
            all_ids   = ids_raw.split()
            recent_ids = all_ids[-max_results:][::-1]   # Newest first.

            logger.info(
                f"[READ_EMAIL] Found {len(all_ids)} unseen messages. "
                f"Fetching {len(recent_ids)}."
            )

            # ── Fetch and parse messages ──────────────────────────────────────
            output_parts: list[str] = [
                f"=== Unread Emails from '{mailbox}' "
                f"({len(recent_ids)} of {len(all_ids)} unread) ===\n"
            ]

            for idx, msg_id in enumerate(recent_ids, start=1):
                part_str = self._fetch_message(mail, msg_id, idx)
                output_parts.append(part_str)

            full_output = "\n".join(output_parts)

            # ── Context window protection ─────────────────────────────────────
            if len(full_output) > _MAX_TOTAL_OUTPUT_CHARS:
                full_output = (
                    full_output[:_MAX_TOTAL_OUTPUT_CHARS]
                    + f"\n\n... [Output truncated at {_MAX_TOTAL_OUTPUT_CHARS:,} chars]"
                )

            return full_output

        finally:
            # Always close the IMAP connection, even on error.
            try:
                mail.close()
                mail.logout()
            except Exception:
                pass

    def _fetch_message(
        self,
        mail:   imaplib.IMAP4_SSL,
        msg_id: bytes,
        index:  int,
    ) -> str:
        """
        Fetches a single message by ID and returns a formatted plain-text
        summary. Uses BODY.PEEK[] to avoid marking the message as read.
        """
        try:
            status, data = mail.fetch(msg_id, "(BODY.PEEK[])")
            if status != "OK" or not data or data[0] is None:
                return f"[{index}] Error: Could not fetch message {msg_id.decode()}.\n"

            # data[0][1] is the raw RFC 822 message bytes.
            raw: bytes = data[0][1]  # type: ignore[index]
            parsed = email.message_from_bytes(raw)

            from_addr = _decode_header_value(parsed.get("From", "Unknown"))
            subject   = _decode_header_value(parsed.get("Subject", "(No subject)"))
            date_str  = parsed.get("Date", "Unknown date")

            body = _extract_plain_body(parsed)
            if not body:
                body = "(No readable text content)"
            elif len(body) > _MAX_BODY_PER_EMAIL:
                body = (
                    body[:_MAX_BODY_PER_EMAIL]
                    + f"\n... [Body truncated at {_MAX_BODY_PER_EMAIL:,} chars]"
                )

            return (
                f"[{index}] From:    {from_addr}\n"
                f"     Date:    {date_str}\n"
                f"     Subject: {subject}\n"
                f"     ---\n"
                f"     {body.strip()}\n"
            )

        except Exception as exc:
            logger.warning(f"[READ_EMAIL] Error parsing message {msg_id}: {exc}")
            return f"[{index}] Error: Could not parse message — {exc}\n"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _decode_header_value(raw_value: str) -> str:
    """
    Decodes RFC 2047 encoded email headers (e.g. =?UTF-8?b?...?=)
    into plain Unicode strings.
    """
    try:
        parts   = email.header.decode_header(raw_value)
        decoded = []
        for part, charset in parts:
            if isinstance(part, bytes):
                decoded.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                decoded.append(str(part))
        return " ".join(decoded).strip()
    except Exception:
        return str(raw_value)


def _extract_plain_body(parsed_msg: email.message.Message) -> str:
    """
    Walks a MIME message and extracts only text/plain parts.
    If no plain-text parts exist, falls back to stripping HTML from
    text/html parts.
    Returns a single concatenated plain-text string.
    """
    plain_parts: list[str] = []
    html_parts:  list[str] = []

    if parsed_msg.is_multipart():
        for part in parsed_msg.walk():
            content_type = part.get_content_type()
            disposition  = str(part.get("Content-Disposition", ""))

            # Skip attachments.
            if "attachment" in disposition:
                continue

            if content_type == "text/plain":
                charset = part.get_content_charset() or "utf-8"
                try:
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        plain_parts.append(
                            payload.decode(charset, errors="replace")
                        )
                except Exception:
                    pass

            elif content_type == "text/html":
                charset = part.get_content_charset() or "utf-8"
                try:
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        html_parts.append(
                            payload.decode(charset, errors="replace")
                        )
                except Exception:
                    pass
    else:
        # Single-part message.
        content_type = parsed_msg.get_content_type()
        charset      = parsed_msg.get_content_charset() or "utf-8"
        try:
            payload = parsed_msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                text = payload.decode(charset, errors="replace")
                if content_type == "text/plain":
                    plain_parts.append(text)
                elif content_type == "text/html":
                    html_parts.append(text)
        except Exception:
            pass

    if plain_parts:
        return "\n".join(plain_parts)

    if html_parts:
        # Strip HTML tags from fallback HTML content.
        combined = "\n".join(html_parts)
        return _strip_html(combined)

    return ""


def _strip_html(html: str) -> str:
    """
    Removes HTML tags and decodes common entities.
    Lightweight regex approach — does not require BeautifulSoup.
    """
    # Remove <script> and <style> blocks entirely.
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags.
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities.
    text = (
        text.replace("&amp;",  "&")
            .replace("&lt;",   "<")
            .replace("&gt;",   ">")
            .replace("&quot;", '"')
            .replace("&#39;",  "'")
            .replace("&nbsp;", " ")
    )
    # Normalise whitespace.
    text = re.sub(r"[ \t]+",  " ",  text)
    text = re.sub(r"\n{3,}",  "\n\n", text)
    return text.strip()