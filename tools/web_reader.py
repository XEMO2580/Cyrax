"""
tools/web_reader.py — CYRAX 3.0 Web Page Reader Tool

Tools:
    ReadWebpageTool — Fetches a URL and extracts clean readable text

Security:
    ReadWebpageTool — USER (makes outbound HTTP requests on behalf of user)

Dependencies (all optional — OS boots normally if missing):
    requests        — HTTP client
    beautifulsoup4  — HTML parsing and text extraction
    lxml            — Faster BS4 parser (optional — falls back to html.parser)

Context window protection:
    All HTML, JavaScript, CSS, and invisible elements are stripped.
    Final extracted text is truncated to 25,000 characters maximum.
    This prevents a large webpage from exhausting the LLM context window.

Network safety:
    Hard 10-second total request timeout (connect + read).
    Redirects are followed up to a limit of 5.
    Private/loopback IP ranges are blocked to prevent SSRF attacks.
    User-Agent is set to a standard browser string to avoid bot blocks.

execute() is synchronous, never raises, and returns a plain string.
Failures return strings starting with 'Error: '.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from security.auth import SecurityLevel
from tools.registry import BaseTool

logger = logging.getLogger(__name__)

# Hard limits.
_MAX_OUTPUT_CHARS:    int = 25_000
_REQUEST_TIMEOUT:     int = 10        # seconds (connect + read combined)
_MAX_REDIRECTS:       int = 5
_MAX_CONTENT_BYTES:   int = 10 * 1024 * 1024   # 10 MB raw HTML ceiling

# Permitted URL schemes — no file://, ftp://, etc.
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# Private/loopback ranges blocked to prevent SSRF.
_BLOCKED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# HTML elements whose entire subtree (tag + content) should be removed.
_STRIP_TAGS: frozenset[str] = frozenset({
    "script", "style", "noscript", "head", "meta",
    "link", "iframe", "svg", "img", "video", "audio",
    "form", "input", "button", "select", "textarea",
    "nav", "footer", "header", "aside", "advertisement",
})


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class ReadWebpageSchema(BaseModel):
    url: str = Field(
        ...,
        description=(
            "Full URL of the webpage to read, including scheme "
            "(e.g. https://example.com)."
        ),
    )
    include_links: bool = Field(
        default=False,
        description=(
            "If True, appends a list of hyperlinks found on the page "
            "after the main content."
        ),
    )

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        parsed = urlparse(v.strip())
        if parsed.scheme not in _ALLOWED_SCHEMES:
            raise ValueError(
                f"URL scheme '{parsed.scheme}' is not permitted. "
                f"Only http and https are supported."
            )
        if not parsed.netloc:
            raise ValueError("URL must include a valid hostname.")
        return v.strip()


# ══════════════════════════════════════════════════════════════════════════════
# TOOL
# ══════════════════════════════════════════════════════════════════════════════

class ReadWebpageTool(BaseTool):
    """
    Fetches a URL and returns clean, readable plain text extracted from the page.

    Processing pipeline:
        1. URL scheme and format validation (Pydantic schema).
        2. SSRF guard — hostname resolved to IP, checked against blocked ranges.
        3. HTTP GET with timeout, redirect limit, and raw size ceiling.
        4. HTML parsed with BeautifulSoup (lxml if available, html.parser fallback).
        5. Noise tags (script, style, nav, footer, etc.) removed entirely.
        6. Remaining text extracted, whitespace normalised.
        7. Optional link list appended.
        8. Output truncated to _MAX_OUTPUT_CHARS.

    Returns the extracted text, or an 'Error: ...' string on any failure.
    """

    name           = "READ_WEBPAGE"
    description    = (
        "Fetches a URL and returns the readable plain-text content of the webpage. "
        "Strips all HTML, JavaScript, CSS, and navigation elements. "
        f"Returns up to {_MAX_OUTPUT_CHARS:,} characters."
    )
    security_level = SecurityLevel.USER
    args_schema    = ReadWebpageSchema

    def execute(  # type: ignore[override]
        self,
        url:           str,
        include_links: bool = False,
    ) -> str:
        # ── Defensive imports ─────────────────────────────────────────────────
        try:
            import requests  # type: ignore[import]
        except ImportError:
            return (
                "Error: requests is not installed. "
                "Run: pip install requests"
            )

        try:
            from bs4 import BeautifulSoup  # type: ignore[import]
        except ImportError:
            return (
                "Error: beautifulsoup4 is not installed. "
                "Run: pip install beautifulsoup4"
            )

        # ── SSRF guard ────────────────────────────────────────────────────────
        ssrf_error = _check_ssrf(url)
        if ssrf_error:
            return ssrf_error

        # ── HTTP request ──────────────────────────────────────────────────────
        logger.info(f"[WEB_READER] Fetching: {url}")

        try:
            response = requests.get(
                url,
                timeout=_REQUEST_TIMEOUT,
                headers={"User-Agent": _USER_AGENT},
                allow_redirects=True,
                stream=True,      # Stream so we can enforce the size ceiling.
            )
            response.raise_for_status()

        except requests.exceptions.Timeout:
            return (
                f"Error: Request timed out after {_REQUEST_TIMEOUT}s. "
                f"The server may be slow or unreachable."
            )
        except requests.exceptions.TooManyRedirects:
            return (
                f"Error: Too many redirects (>{_MAX_REDIRECTS}). "
                f"The URL may be in a redirect loop."
            )
        except requests.exceptions.ConnectionError as exc:
            return f"Error: Could not connect to '{url}' — {exc}"
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            return f"Error: HTTP {status} returned by '{url}'."
        except requests.exceptions.RequestException as exc:
            return f"Error: Request failed — {exc}"

        # ── Content type guard ────────────────────────────────────────────────
        content_type = response.headers.get("Content-Type", "")
        if content_type and not any(
            ct in content_type.lower()
            for ct in ("text/html", "text/plain", "application/xhtml")
        ):
            return (
                f"Error: URL returned non-HTML content "
                f"(Content-Type: {content_type}). "
                f"This tool only reads HTML and plain-text pages."
            )

        # ── Raw content size ceiling ──────────────────────────────────────────
        try:
            raw_bytes = b""
            for chunk in response.iter_content(chunk_size=65536):
                raw_bytes += chunk
                if len(raw_bytes) > _MAX_CONTENT_BYTES:
                    logger.warning(
                        f"[WEB_READER] Response exceeded {_MAX_CONTENT_BYTES // 1024 // 1024} MB. "
                        f"Truncating raw HTML before parse."
                    )
                    break
        except Exception as exc:
            return f"Error: Failed reading response body — {exc}"

        # ── Detect encoding ───────────────────────────────────────────────────
        encoding = response.encoding or "utf-8"
        try:
            html = raw_bytes.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            html = raw_bytes.decode("utf-8", errors="replace")

        # ── HTML parsing and text extraction ──────────────────────────────────
        try:
            text, links = _extract_text(html, BeautifulSoup)
        except Exception as exc:
            logger.error(f"[WEB_READER] Parsing error: {exc}")
            return f"Error: Failed to parse HTML from '{url}' — {exc}"

        if not text.strip():
            return (
                f"Error: No readable text found at '{url}'. "
                f"The page may be JavaScript-rendered or require a login."
            )

        # ── Assemble output ───────────────────────────────────────────────────
        output_parts: list[str] = [
            f"=== Content from: {url} ===\n",
            text,
        ]

        if include_links and links:
            link_block = "\n".join(f"  - {href}" for href in links[:50])
            output_parts.append(f"\n\n=== Links found on page ===\n{link_block}")

        output = "\n".join(output_parts)

        # ── Context window protection ──────────────────────────────────────────
        if len(output) > _MAX_OUTPUT_CHARS:
            output = (
                output[:_MAX_OUTPUT_CHARS]
                + f"\n\n... [Content truncated at {_MAX_OUTPUT_CHARS:,} characters]"
            )
            logger.info(
                f"[WEB_READER] Output truncated to {_MAX_OUTPUT_CHARS:,} chars."
            )

        logger.info(
            f"[WEB_READER] Extracted {len(output):,} chars from '{url}'."
        )
        return output


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _check_ssrf(url: str) -> str | None:
    """
    Resolves the URL hostname to an IP address and checks it against
    the blocked private/loopback network list.

    Returns an error string if the host resolves to a private range.
    Returns None if the host is safe to contact.

    Performed BEFORE the HTTP request — we do not make any connection
    to the target until this check passes.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        return "Error: URL has no resolvable hostname."

    try:
        # getaddrinfo returns all addresses — check every one.
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        return f"Error: Could not resolve hostname '{hostname}' — {exc}"
    except Exception as exc:
        return f"Error: DNS resolution failed for '{hostname}' — {exc}"

    for addr_info in addr_infos:
        raw_ip = addr_info[4][0]
        try:
            ip_obj = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue

        for blocked_net in _BLOCKED_NETWORKS:
            if ip_obj in blocked_net:
                logger.warning(
                    f"[WEB_READER] SSRF blocked: '{hostname}' → {raw_ip} "
                    f"is in {blocked_net}"
                )
                return (
                    f"Error: Access to '{hostname}' is blocked. "
                    f"Private and loopback addresses are not permitted."
                )

    return None


def _extract_text(
    html:         str,
    BeautifulSoup: type,
) -> tuple[str, list[str]]:
    """
    Parses raw HTML and extracts clean readable text.

    Parser preference: lxml (fast C parser) → html.parser (stdlib fallback).
    Noise elements (script, style, nav, etc.) are decomposed entirely
    before text extraction so their content does not appear in the output.

    Returns:
        text:  Normalised plain-text content.
        links: List of absolute or relative href values found in <a> tags.
    """
    # Choose the fastest available parser.
    try:
        import lxml  # type: ignore[import] # noqa: F401
        parser = "lxml"
    except ImportError:
        parser = "html.parser"

    soup = BeautifulSoup(html, parser)

    # Remove all noise subtrees in-place.
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove HTML comments.
    from bs4 import Comment  # type: ignore[import]
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Extract links before stripping remaining tags.
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href and not href.startswith(("#", "javascript:")):
            links.append(href)

    # Get text with a single space as separator between elements.
    raw_text: str = soup.get_text(separator=" ", strip=True)

    # Normalise whitespace: collapse runs of spaces/tabs, preserve paragraphs.
    import re
    # Collapse multiple spaces and tabs to a single space.
    text = re.sub(r"[ \t]+", " ", raw_text)
    # Collapse more than two consecutive newlines to two.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text, links