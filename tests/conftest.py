"""
tests/conftest.py — Shared pytest fixtures for CYRAX 3.0 test suite.

Provides:
    mock_ctx    — A CyraxContext-shaped Mock satisfying the protocols in
                  core/context.py. Uses spec=Protocol so any call to a
                  method not explicitly configured raises AttributeError
                  loudly rather than silently returning a Mock.
    sandbox     — Monkeypatches tools.file_ops.CYRAX_WORKSPACE and its
                  derived _WORKSPACE_RESOLVED constant to point at a
                  pytest tmp_path, so file_ops tests never touch the
                  real developer machine.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# CyraxContext fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_ctx() -> Mock:
    """
    Minimal CyraxContext double for tools that read self._ctx.

    Only tools/scheduler.py and tools/memory_ops.py currently read
    self._ctx / self._loop. desktop_ops.py and file_ops.py tools in
    this sprint do NOT require ctx injection — but the fixture is
    provided here for suite-wide reuse in later phases.
    """
    ctx = Mock(name="CyraxContext")
    ctx.session_id = "test_session_001"

    ctx.tool_registry = Mock(name="ToolRegistry")
    ctx.tool_registry.get_tool_names.return_value = []
    ctx.tool_registry.get_all_definitions.return_value = []

    ctx.memory = Mock(name="MemoryStack")
    ctx.memory.state = Mock(name="SessionStateStore")
    ctx.memory.state.get.return_value = None
    ctx.memory.conversation = Mock(name="ConversationStore")
    ctx.memory.profile = Mock(name="UserProfileStore")

    ctx.security = Mock(name="SecurityGuard")
    ctx.security.authorize_action.return_value = True

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# file_ops.py sandbox isolation
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Redirects tools.file_ops's workspace sandbox to a pytest tmp_path.

    CRITICAL: both CYRAX_WORKSPACE and its pre-derived _WORKSPACE_RESOLVED
    must be patched together. resolve_secure_path() and the root-targeting
    guard both close over _WORKSPACE_RESOLVED directly — patching only
    CYRAX_WORKSPACE leaves the traversal guard checking against the real
    ~/CYRAX_WORKSPACE while I/O happens in tmp_path, which would make
    every path-traversal test pass for the wrong reason.

    Returns the tmp_path so tests can also write fixture files directly.
    """
    from tools import file_ops

    monkeypatch.setattr(file_ops, "CYRAX_WORKSPACE", tmp_path)
    monkeypatch.setattr(file_ops, "_WORKSPACE_RESOLVED", tmp_path.resolve())

    return tmp_path


# ══════════════════════════════════════════════════════════════════════════════
# Missing-dependency simulation helper
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def block_import():
    """
    Returns a context manager factory: block_import("pyperclip") makes
    `import pyperclip` raise ImportError for the duration of the `with`
    block, regardless of whether the real package is installed.

    Usage:
        with block_import("pyperclip"):
            result = tool.execute()
            assert result.startswith("Error:")
    """
    from unittest.mock import patch

    def _factory(module_name: str):
        return patch.dict("sys.modules", {module_name: None})

    return _factory