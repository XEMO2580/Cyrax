"""
tests/tools/test_web_search.py — Certification suite for tools/web_search.py

NO REAL WEB SEARCHES: ddgs.DDGS is fully mocked as a context manager.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from tools.web_search import (
    WebSearchTool,
    WebSearchSchema,
    _MAX_RESULTS,
    _MAX_SNIPPET_CHARS,
    _REQUEST_TIMEOUT,
)


def _make_mock_ddgs(text_return: list[dict] | None = None, text_side_effect=None) -> MagicMock:
    """
    Builds a MagicMock standing in for the ddgs.DDGS class such that:
        with DDGS(timeout=...) as ddgs:
            ddgs.text(query, max_results=...)
    behaves as configured.
    """
    instance = MagicMock()
    if text_side_effect is not None:
        instance.text.side_effect = text_side_effect
    else:
        instance.text.return_value = text_return or []

    instance.__enter__ = MagicMock(return_value=instance)
    instance.__exit__  = MagicMock(return_value=False)

    ddgs_class = MagicMock(return_value=instance)
    return ddgs_class


class TestWebSearchTool:

    def test_happy_path(self) -> None:
        results = [
            {"title": "Result One", "href": "https://example.com/1", "body": "First snippet."},
            {"title": "Result Two", "href": "https://example.com/2", "body": "Second snippet."},
        ]
        mock_ddgs_class = _make_mock_ddgs(text_return=results)

        with patch("ddgs.DDGS", mock_ddgs_class):
            result = WebSearchTool().execute(query="test query")

        assert "Result One" in result
        assert "https://example.com/1" in result
        assert "First snippet." in result
        assert "[1]" in result and "[2]" in result

    def test_invalid_args_empty_query(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchSchema(query="")

    def test_invalid_args_too_long_query(self) -> None:
        with pytest.raises(ValidationError):
            WebSearchSchema(query="a" * 501)

    def test_missing_dependency(self, block_import) -> None:
        with block_import("ddgs"):
            result = WebSearchTool().execute(query="anything")

        assert result.startswith("Error:")
        assert "ddgs" in result.lower()

    def test_malformed_ddgs_results_handled_gracefully(self) -> None:
        """
        Defensive parsing test: DDGS returns malformed/hostile result dicts
        (None values, missing keys, non-string types). Formatter must not crash.
        """
        hostile_results = [
            {"title": None, "href": 12345, "body": None},
            {},  # completely empty dict
            {"title": "Valid Title", "href": "https://ok.com", "body": "fine"},
        ]
        mock_ddgs_class = _make_mock_ddgs(text_return=hostile_results)

        with patch("ddgs.DDGS", mock_ddgs_class):
            result = WebSearchTool().execute(query="test")

        assert "No title" in result
        assert "No URL" in result
        assert "No description" in result
        assert "Valid Title" in result

    def test_empty_input_whitespace_only(self) -> None:
        """Passes Pydantic min_length=1 (whitespace counts) but strips to empty."""
        result = WebSearchTool().execute(query="   ")
        assert result.startswith("Error:")
        assert "cannot be empty" in result.lower()

    def test_empty_results_from_ddgs(self) -> None:
        mock_ddgs_class = _make_mock_ddgs(text_return=[])

        with patch("ddgs.DDGS", mock_ddgs_class):
            result = WebSearchTool().execute(query="extremely obscure query xyz123")

        assert result.startswith("Error:")
        assert "no results" in result.lower()

    def test_large_snippet_truncated(self) -> None:
        oversized_snippet = "A" * (_MAX_SNIPPET_CHARS + 500)
        results = [{"title": "T", "href": "https://x.com", "body": oversized_snippet}]
        mock_ddgs_class = _make_mock_ddgs(text_return=results)

        with patch("ddgs.DDGS", mock_ddgs_class):
            result = WebSearchTool().execute(query="test")

        assert "..." in result
        # Confirm the raw oversized snippet does not appear in full.
        assert oversized_snippet not in result

    def test_query_at_max_length_boundary_succeeds(self) -> None:
        boundary_query = "a" * 500  # exactly at max_length
        mock_ddgs_class = _make_mock_ddgs(
            text_return=[{"title": "T", "href": "https://x.com", "body": "b"}]
        )

        with patch("ddgs.DDGS", mock_ddgs_class):
            result = WebSearchTool().execute(query=boundary_query)

        assert result.startswith("Web search results")

    def test_timeout(self) -> None:
        mock_ddgs_class = _make_mock_ddgs(
            text_side_effect=TimeoutError(f"Request timed out after {_REQUEST_TIMEOUT}s")
        )

        with patch("ddgs.DDGS", mock_ddgs_class):
            result = WebSearchTool().execute(query="slow query")

        assert result.startswith("Error:")
        assert "timed out" in result.lower()

    def test_generic_network_failure(self) -> None:
        mock_ddgs_class = _make_mock_ddgs(
            text_side_effect=ConnectionError("DNS resolution failed")
        )

        with patch("ddgs.DDGS", mock_ddgs_class):
            result = WebSearchTool().execute(query="anything")

        assert result.startswith("Error:")
        assert "ConnectionError" in result

    def test_max_results_param_passed_correctly(self) -> None:
        instance = MagicMock()
        instance.text.return_value = []
        instance.__enter__ = MagicMock(return_value=instance)
        instance.__exit__  = MagicMock(return_value=False)
        mock_ddgs_class = MagicMock(return_value=instance)

        with patch("ddgs.DDGS", mock_ddgs_class):
            WebSearchTool().execute(query="test")

        instance.text.assert_called_once_with("test", max_results=_MAX_RESULTS)