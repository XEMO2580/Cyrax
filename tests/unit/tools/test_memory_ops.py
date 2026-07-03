"""
tests/tools/test_memory_ops.py — Certification suite for tools/memory_ops.py

TECHNIQUE NOTE (see Phase 4.5A audit): execute() bridges sync → async via
asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=...).
This REQUIRES self._loop to be a real, running event loop — a bare Mock()
cannot satisfy run_coroutine_threadsafe's contract.

To avoid deadlock, every test calls tool.execute() via asyncio.to_thread()
from within an async test, with self._loop set to the CURRENT running loop
(obtained via asyncio.get_running_loop() inside the test). This lets
execute() (on the worker thread) schedule work back onto the test's own
loop (on the main thread) without the two blocking each other.

MEMORY STORE: A real UserProfileStore backed by tmp_path is used (per task
directive) — NOT a mock — so profile persistence logic is genuinely exercised.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from memory.profile.profile_store import UserProfileStore
from tools.memory_ops import (
    CoreMemoryWriteTool, CoreMemoryWriteSchema,
    CoreMemoryReadTool, CoreMemoryReadSchema,
    _MAX_PROFILE_ENTRIES,
)

pytestmark = pytest.mark.asyncio


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def profile_store(tmp_path: Path) -> UserProfileStore:
    """Real UserProfileStore on a tmp_path sandbox — per task directive."""
    return UserProfileStore(profile_path=tmp_path / "user_profile.json")


@pytest.fixture
def mock_ctx_with_real_profile(profile_store: UserProfileStore) -> Mock:
    ctx = Mock(name="CyraxContext")
    ctx.memory = Mock(name="MemoryStack")
    ctx.memory.profile = profile_store
    return ctx


async def _run_tool_execute(tool, ctx, loop, **kwargs) -> str:
    """
    Runs tool.execute() on a worker thread (asyncio.to_thread) with
    self._ctx / self._loop injected, mirroring exactly what
    ToolRegistry.execute_tool() does in production. self._loop is the
    CURRENT test's running loop, obtained by the caller before this
    helper is invoked — never fetched inside the worker thread itself.
    """
    tool._ctx  = ctx
    tool._loop = loop
    return await asyncio.wait_for(
        asyncio.to_thread(tool.execute, **kwargs),
        timeout=5.0,  # Hard guard: fails fast on any deadlock regression.
    )


# ══════════════════════════════════════════════════════════════════════════════
# CoreMemoryWriteTool
# ══════════════════════════════════════════════════════════════════════════════

class TestCoreMemoryWriteTool:

    async def test_happy_path_new_key(self, mock_ctx_with_real_profile: Mock) -> None:
        loop = asyncio.get_running_loop()
        result = await _run_tool_execute(
            CoreMemoryWriteTool(), mock_ctx_with_real_profile, loop,
            key="user_name", value="Xemo",
        )

        assert result.startswith("Success:")
        assert "Saved" in result

        stored = await mock_ctx_with_real_profile.memory.profile.get("user_name")
        assert stored == "Xemo"

    async def test_happy_path_overwrite_existing_key(
        self, mock_ctx_with_real_profile: Mock
    ) -> None:
        loop = asyncio.get_running_loop()
        await _run_tool_execute(
            CoreMemoryWriteTool(), mock_ctx_with_real_profile, loop,
            key="favorite_browser", value="brave",
        )
        result = await _run_tool_execute(
            CoreMemoryWriteTool(), mock_ctx_with_real_profile, loop,
            key="favorite_browser", value="chrome",
        )

        assert result.startswith("Success:")
        assert "Updated" in result
        stored = await mock_ctx_with_real_profile.memory.profile.get("favorite_browser")
        assert stored == "chrome"

    def test_invalid_args_missing_value(self) -> None:
        with pytest.raises(ValidationError):
            CoreMemoryWriteSchema(key="x")

    async def test_missing_context_dependency(self) -> None:
        """Tool constructed without _ctx injected must fail gracefully, not raise."""
        tool = CoreMemoryWriteTool()
        result = await asyncio.to_thread(tool.execute, key="k", value="v")

        assert result.startswith("Error:")
        assert "CyraxContext" in result

    async def test_missing_loop_dependency(self, mock_ctx_with_real_profile: Mock) -> None:
        tool = CoreMemoryWriteTool()
        tool._ctx = mock_ctx_with_real_profile
        # _loop deliberately NOT set.
        result = await asyncio.to_thread(tool.execute, key="k", value="v")

        assert result.startswith("Error:")
        assert "event loop" in result.lower()

    async def test_malicious_input_key_sanitised(
        self, mock_ctx_with_real_profile: Mock
    ) -> None:
        """Key with spaces/mixed case is normalised, not passed through raw."""
        loop = asyncio.get_running_loop()
        result = await _run_tool_execute(
            CoreMemoryWriteTool(), mock_ctx_with_real_profile, loop,
            key="  Favorite Color  ", value="blue",
        )

        assert result.startswith("Success:")
        stored = await mock_ctx_with_real_profile.memory.profile.get("favorite_color")
        assert stored == "blue"

    async def test_empty_key_rejected(self, mock_ctx_with_real_profile: Mock) -> None:
        loop = asyncio.get_running_loop()
        result = await _run_tool_execute(
            CoreMemoryWriteTool(), mock_ctx_with_real_profile, loop,
            key="   ", value="something",
        )
        assert result.startswith("Error:")
        assert "key cannot be empty" in result.lower()

    async def test_empty_value_rejected(self, mock_ctx_with_real_profile: Mock) -> None:
        loop = asyncio.get_running_loop()
        result = await _run_tool_execute(
            CoreMemoryWriteTool(), mock_ctx_with_real_profile, loop,
            key="some_key", value="   ",
        )
        assert result.startswith("Error:")
        assert "value cannot be empty" in result.lower()

    async def test_profile_entry_limit_enforced(
        self, mock_ctx_with_real_profile: Mock
    ) -> None:
        """New key rejected once profile has _MAX_PROFILE_ENTRIES entries."""
        loop = asyncio.get_running_loop()
        for i in range(_MAX_PROFILE_ENTRIES):
            await mock_ctx_with_real_profile.memory.profile.set(f"key_{i}", "v")

        result = await _run_tool_execute(
            CoreMemoryWriteTool(), mock_ctx_with_real_profile, loop,
            key="one_too_many", value="v",
        )

        assert result.startswith("Error:")
        assert "full" in result.lower()

    async def test_profile_entry_limit_does_not_block_updates(
        self, mock_ctx_with_real_profile: Mock
    ) -> None:
        """Updating an EXISTING key when at capacity must still succeed."""
        loop = asyncio.get_running_loop()
        for i in range(_MAX_PROFILE_ENTRIES):
            await mock_ctx_with_real_profile.memory.profile.set(f"key_{i}", "v")

        result = await _run_tool_execute(
            CoreMemoryWriteTool(), mock_ctx_with_real_profile, loop,
            key="key_0", value="updated_value",
        )

        assert result.startswith("Success:")

    async def test_large_value_rejected_by_schema(self) -> None:
        from tools.memory_ops import _MAX_VALUE_CHARS
        with pytest.raises(ValidationError):
            CoreMemoryWriteSchema(key="k", value="x" * (_MAX_VALUE_CHARS + 1))


# ══════════════════════════════════════════════════════════════════════════════
# CoreMemoryReadTool
# ══════════════════════════════════════════════════════════════════════════════

class TestCoreMemoryReadTool:

    async def test_happy_path_single_key(self, mock_ctx_with_real_profile: Mock) -> None:
        loop = asyncio.get_running_loop()
        await mock_ctx_with_real_profile.memory.profile.set("user_name", "Xemo")

        result = await _run_tool_execute(
            CoreMemoryReadTool(), mock_ctx_with_real_profile, loop, key="user_name"
        )

        assert result == "Memory 'user_name': Xemo"

    async def test_happy_path_read_all(self, mock_ctx_with_real_profile: Mock) -> None:
        loop = asyncio.get_running_loop()
        await mock_ctx_with_real_profile.memory.profile.set("a", "1")
        await mock_ctx_with_real_profile.memory.profile.set("b", "2")

        result = await _run_tool_execute(
            CoreMemoryReadTool(), mock_ctx_with_real_profile, loop, key=""
        )

        assert "2 entries" in result
        assert "a: 1" in result
        assert "b: 2" in result

    def test_invalid_args_key_too_long(self) -> None:
        from tools.memory_ops import _MAX_KEY_CHARS
        with pytest.raises(ValidationError):
            CoreMemoryReadSchema(key="x" * (_MAX_KEY_CHARS + 1))

    async def test_missing_context_dependency(self) -> None:
        tool = CoreMemoryReadTool()
        result = await asyncio.to_thread(tool.execute, key="anything")
        assert result.startswith("Error:")

    async def test_nonexistent_key(self, mock_ctx_with_real_profile: Mock) -> None:
        loop = asyncio.get_running_loop()
        result = await _run_tool_execute(
            CoreMemoryReadTool(), mock_ctx_with_real_profile, loop, key="ghost_key"
        )

        assert "does not exist" in result.lower()

    async def test_empty_profile_read_all(self, mock_ctx_with_real_profile: Mock) -> None:
        loop = asyncio.get_running_loop()
        result = await _run_tool_execute(
            CoreMemoryReadTool(), mock_ctx_with_real_profile, loop, key=""
        )

        assert "empty" in result.lower()

    async def test_read_all_truncates_long_values_in_listing(
        self, mock_ctx_with_real_profile: Mock
    ) -> None:
        loop = asyncio.get_running_loop()
        long_value = "x" * 500
        await mock_ctx_with_real_profile.memory.profile.set("big", long_value)

        result = await _run_tool_execute(
            CoreMemoryReadTool(), mock_ctx_with_real_profile, loop, key=""
        )

        assert "..." in result
        assert long_value not in result  # full 500-char value not dumped raw