# ═══════════════════════════════════════════════════════════════════════════
# tests/unit/tools/test_web_reader.py
# ═══════════════════════════════════════════════════════════════════════════

"""
tests/unit/tools/test_web_reader.py — Certification suite for tools/web_reader.py

SOURCE NOTE: No tools/web_reader.py content was included in this batch's
uploaded_files. This suite is written against the version authored earlier
in this session (ReadWebpageTool, SSRF guard via socket.getaddrinfo,
BeautifulSoup-based extraction). If the file on disk has since diverged,
re-run this audit against the actual current source before treating this
certification as final — same caveat pattern established for media_power.py
under TD-027.

NO REAL NETWORK TRAFFIC: requests.get is mocked in every test.
NO REAL DNS RESOLUTION: socket.getaddrinfo is mocked for SSRF tests.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from tools.web_reader import (
    ReadWebpageTool,
    ReadWebpageSchema,
    _MAX_OUTPUT_CHARS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mock_response(
    text: str = "",
    status_code: int = 200,
    content_type: str = "text/html",
    raise_for_status_side_effect=None,
) -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.headers     = {"Content-Type": content_type}
    resp.encoding    = "utf-8"

    body_bytes = text.encode("utf-8")
    resp.iter_content = Mock(return_value=[body_bytes])

    if raise_for_status_side_effect:
        resp.raise_for_status = Mock(side_effect=raise_for_status_side_effect)
    else:
        resp.raise_for_status = Mock(return_value=None)

    return resp


def _mock_public_dns():
    """Returns a getaddrinfo result resolving to a public, non-blocked IP."""
    return [
        (2, 1, 6, "", ("93.184.216.34", 0)),  # example.com-style public IP
    ]


def _mock_private_dns(ip: str = "127.0.0.1"):
    return [
        (2, 1, 6, "", (ip, 0)),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# ReadWebpageTool — happy path / core extraction
# ══════════════════════════════════════════════════════════════════════════════

class TestReadWebpageToolHappyPath:

    def test_happy_path_extracts_plain_text(self) -> None:
        html = "<html><body><p>Hello World</p></body></html>"
        resp = _mock_response(text=html)

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert "Hello World" in result
        assert "=== Content from: https://example.com ===" in result

    def test_include_links_appends_link_block(self) -> None:
        html = '<html><body><p>Text</p><a href="https://other.com/page">Link</a></body></html>'
        resp = _mock_response(text=html)

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(
                url="https://example.com", include_links=True
            )

        assert "Links found on page" in result
        assert "https://other.com/page" in result

    def test_links_excluded_by_default(self) -> None:
        html = '<html><body><a href="https://other.com/page">Link</a></body></html>'
        resp = _mock_response(text=html)

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert "Links found on page" not in result


# ══════════════════════════════════════════════════════════════════════════════
# Invalid arguments
# ══════════════════════════════════════════════════════════════════════════════

class TestInvalidArgs:

    def test_ftp_scheme_rejected_at_schema_level(self) -> None:
        with pytest.raises(ValidationError):
            ReadWebpageSchema(url="ftp://example.com/file.txt")

    def test_file_scheme_rejected_at_schema_level(self) -> None:
        with pytest.raises(ValidationError):
            ReadWebpageSchema(url="file:///etc/passwd")

    def test_missing_hostname_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReadWebpageSchema(url="https://")

    def test_missing_url_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReadWebpageSchema()

    def test_valid_https_url_accepted_by_schema(self) -> None:
        schema = ReadWebpageSchema(url="https://example.com/page")
        assert schema.url == "https://example.com/page"


# ══════════════════════════════════════════════════════════════════════════════
# Missing dependency
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingDependency:

    def test_missing_requests(self, block_import) -> None:
        with block_import("requests"):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert result.startswith("Error:")
        assert "requests" in result.lower()

    def test_missing_beautifulsoup(self, block_import) -> None:
        with block_import("bs4"):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert result.startswith("Error:")
        assert "beautifulsoup4" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# SSRF protection
# ══════════════════════════════════════════════════════════════════════════════

class TestSSRFProtection:

    @pytest.mark.parametrize(
        "blocked_ip",
        [
            "127.0.0.1",      # loopback
            "10.0.0.5",       # private class A
            "172.16.0.1",     # private class B
            "192.168.1.1",    # private class C
            "169.254.1.1",    # link-local
        ],
    )
    def test_private_ip_resolution_blocked(self, blocked_ip: str) -> None:
        mock_get = Mock()

        with patch("socket.getaddrinfo", return_value=_mock_private_dns(blocked_ip)), \
             patch("requests.get", mock_get):
            result = ReadWebpageTool().execute(url="https://internal-service.local")

        assert result.startswith("Error:")
        assert "blocked" in result.lower()
        mock_get.assert_not_called()

    def test_public_ip_resolution_allowed(self) -> None:
        resp = _mock_response(text="<html><body>public content</body></html>")

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp) as mock_get:
            result = ReadWebpageTool().execute(url="https://example.com")

        assert not result.startswith("Error:")
        mock_get.assert_called_once()

    def test_dns_resolution_failure_handled(self) -> None:
        import socket as socket_module

        mock_get = Mock()

        with patch(
            "socket.getaddrinfo",
            side_effect=socket_module.gaierror("Name or service not known"),
        ), patch("requests.get", mock_get):
            result = ReadWebpageTool().execute(url="https://nonexistent-domain-xyz123.invalid")

        assert result.startswith("Error:")
        assert "resolve" in result.lower()
        mock_get.assert_not_called()

    def test_ssrf_check_happens_before_http_request(self) -> None:
        """Confirms DNS/SSRF check is a hard gate BEFORE any network I/O."""
        mock_get = Mock()

        with patch("socket.getaddrinfo", return_value=_mock_private_dns("127.0.0.1")), \
             patch("requests.get", mock_get):
            ReadWebpageTool().execute(url="https://localhost")

        mock_get.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# HTML stripping
# ══════════════════════════════════════════════════════════════════════════════

class TestHTMLStripping:

    def test_script_tags_stripped(self) -> None:
        html = (
            "<html><body>"
            "<p>Visible text</p>"
            "<script>alert('malicious payload');</script>"
            "</body></html>"
        )
        resp = _mock_response(text=html)

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert "Visible text" in result
        assert "<script>" not in result
        assert "alert(" not in result
        assert "malicious payload" not in result

    def test_style_tags_stripped(self) -> None:
        html = (
            "<html><head><style>body { color: red; font-size: 999px; }</style></head>"
            "<body><p>Content here</p></body></html>"
        )
        resp = _mock_response(text=html)

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert "Content here" in result
        assert "font-size" not in result
        assert "<style>" not in result

    def test_nav_and_footer_stripped(self) -> None:
        html = (
            "<html><body>"
            "<nav>Home | About | Contact</nav>"
            "<p>Main article content.</p>"
            "<footer>Copyright 2026 Example Corp</footer>"
            "</body></html>"
        )
        resp = _mock_response(text=html)

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert "Main article content." in result
        assert "Home | About | Contact" not in result
        assert "Copyright 2026" not in result

    def test_html_comments_stripped(self) -> None:
        html = (
            "<html><body>"
            "<!-- This is an internal comment with secrets -->"
            "<p>Public content</p>"
            "</body></html>"
        )
        resp = _mock_response(text=html)

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert "Public content" in result
        assert "internal comment with secrets" not in result

    def test_form_and_input_elements_stripped(self) -> None:
        html = (
            "<html><body>"
            "<p>Article text</p>"
            '<form><input type="password" value="leaked"><button>Submit</button></form>'
            "</body></html>"
        )
        resp = _mock_response(text=html)

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert "Article text" in result
        assert "leaked" not in result


# ══════════════════════════════════════════════════════════════════════════════
# Empty input / empty results
# ══════════════════════════════════════════════════════════════════════════════

class TestEmptyInput:

    def test_page_with_no_extractable_text_returns_error(self) -> None:
        html = "<html><head><script>only_js_here();</script></head><body></body></html>"
        resp = _mock_response(text=html)

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert result.startswith("Error:")
        assert "no readable text" in result.lower()

    def test_completely_empty_html_body(self) -> None:
        resp = _mock_response(text="")

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert result.startswith("Error:")


# ══════════════════════════════════════════════════════════════════════════════
# Large input / context-window truncation
# ══════════════════════════════════════════════════════════════════════════════

class TestLargeInputTruncation:

    def test_output_truncated_at_max_output_chars(self) -> None:
        huge_paragraph = "word " * 20_000  # comfortably exceeds _MAX_OUTPUT_CHARS
        html = f"<html><body><p>{huge_paragraph}</p></body></html>"
        resp = _mock_response(text=html)

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com")

        assert len(result) <= _MAX_OUTPUT_CHARS + 200  # allow for truncation notice
        assert "truncated" in result.lower()

    def test_non_html_content_type_rejected(self) -> None:
        resp = _mock_response(
            text="binary garbage", content_type="application/pdf"
        )

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com/file.pdf")

        assert result.startswith("Error:")
        assert "non-html" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Timeout
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeout:

    def test_timeout_caught_gracefully(self) -> None:
        import requests as requests_module

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch(
                 "requests.get",
                 side_effect=requests_module.exceptions.Timeout("Connection timed out"),
             ):
            result = ReadWebpageTool().execute(url="https://slow-server.example.com")

        assert result.startswith("Error:")
        assert "timed out" in result.lower()

    def test_too_many_redirects_handled(self) -> None:
        import requests as requests_module

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch(
                 "requests.get",
                 side_effect=requests_module.exceptions.TooManyRedirects("Redirect loop"),
             ):
            result = ReadWebpageTool().execute(url="https://redirect-loop.example.com")

        assert result.startswith("Error:")
        assert "redirect" in result.lower()

    def test_connection_error_handled(self) -> None:
        import requests as requests_module

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch(
                 "requests.get",
                 side_effect=requests_module.exceptions.ConnectionError("DNS failure"),
             ):
            result = ReadWebpageTool().execute(url="https://unreachable.example.com")

        assert result.startswith("Error:")

    def test_http_error_status_handled(self) -> None:
        import requests as requests_module

        resp = _mock_response(
            text="",
            status_code=404,
            raise_for_status_side_effect=requests_module.exceptions.HTTPError(
                response=Mock(status_code=404)
            ),
        )

        with patch("socket.getaddrinfo", return_value=_mock_public_dns()), \
             patch("requests.get", return_value=resp):
            result = ReadWebpageTool().execute(url="https://example.com/missing-page")

        assert result.startswith("Error:")
        assert "404" in result