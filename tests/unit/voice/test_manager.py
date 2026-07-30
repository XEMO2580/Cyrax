"""
tests/unit/voice/test_manager.py — Phase 5.0 VoiceManager certification.

All hardware and network I/O is mocked. No real microphone, no real
OpenAI API calls. AudioCapture and BaseSTT are replaced with Mock/fake
doubles at the point of injection into VoiceManager.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from voice.manager import VoiceManager
from voice.state import VoiceState, VoiceStateError
from voice.stt.base import STTResult

pytestmark = pytest.mark.asyncio


def _make_ctx(dispatcher_return: dict | None = None) -> Mock:
    ctx = Mock(name="CyraxContext")
    ctx.session_id = "test_session"
    ctx.dispatcher = Mock(name="Dispatcher")
    ctx.dispatcher.handle = AsyncMock(
        return_value=dispatcher_return
        or {"status": "success", "response": "Done."}
    )
    ctx.events = Mock(name="EventBus")
    ctx.events.publish = AsyncMock()
    return ctx


def _make_audio_capture(return_bytes: bytes = b"\x01\x02\x03") -> Mock:
    capture = Mock()
    capture.record_audio = Mock(return_value=return_bytes)
    return capture


def _make_stt(result: STTResult) -> Mock:
    stt = Mock()
    stt.transcribe = Mock(return_value=result)
    stt.is_available = Mock(return_value=True)
    return stt


class TestVoiceManagerPhase50:

    async def test_voice_success_path_end_to_end(self) -> None:
        ctx = _make_ctx(dispatcher_return={"status": "success", "response": "Chrome opened."})
        audio = _make_audio_capture(b"\x01\x02\x03\x04")
        stt = _make_stt(STTResult(status="success", text="open chrome"))

        manager = VoiceManager(ctx=ctx, session_id="s1", audio_capture=audio, stt=stt)

        result = await manager.handle_voice_command()

        audio.record_audio.assert_called_once()
        stt.transcribe.assert_called_once()
        called_args = stt.transcribe.call_args[0]
        assert called_args[0] == b"\x01\x02\x03\x04"

        ctx.dispatcher.handle.assert_awaited_once()
        _, kwargs = ctx.dispatcher.handle.call_args
        assert kwargs["user_input"] == "open chrome"
        assert kwargs["session_id"] == "s1"
        assert kwargs["ctx"] is ctx

        assert result["status"] == "success"
        assert result["response"] == "Chrome opened."
        assert await manager.get_state() == VoiceState.IDLE

        ctx.events.publish.assert_awaited_once()

    async def test_voice_aborts_on_empty_audio_capture(self) -> None:
        ctx = _make_ctx()
        audio = _make_audio_capture(return_bytes=b"")  # Empty — capture failed/stalled.
        stt = _make_stt(STTResult(status="success", text="should never be reached"))

        manager = VoiceManager(ctx=ctx, session_id="s1", audio_capture=audio, stt=stt)

        result = await manager.handle_voice_command()

        stt.transcribe.assert_not_called()
        ctx.dispatcher.handle.assert_not_awaited()

        assert result["status"] == "error"
        assert "no audio" in result["response"].lower()
        assert await manager.get_state() == VoiceState.IDLE

    async def test_voice_aborts_on_empty_stt_transcript(self) -> None:
        ctx = _make_ctx()
        audio = _make_audio_capture(b"\x01\x02")
        stt = _make_stt(STTResult(status="empty", text="", message="No speech detected."))

        manager = VoiceManager(ctx=ctx, session_id="s1", audio_capture=audio, stt=stt)

        result = await manager.handle_voice_command()

        ctx.dispatcher.handle.assert_not_awaited()
        assert result["status"] == "error"
        assert "no speech" in result["response"].lower()
        assert await manager.get_state() == VoiceState.IDLE

    async def test_voice_handles_stt_timeout_and_resets(self) -> None:
        ctx = _make_ctx()
        audio = _make_audio_capture(b"\x01\x02")
        stt = _make_stt(
            STTResult(status="timeout", text="", message="STT request timed out after 15s.")
        )

        manager = VoiceManager(ctx=ctx, session_id="s1", audio_capture=audio, stt=stt)

        result = await manager.handle_voice_command()

        ctx.dispatcher.handle.assert_not_awaited()
        assert result["status"] == "error"
        assert "timed out" in result["response"].lower()
        assert await manager.get_state() == VoiceState.IDLE

    async def test_voice_handles_dispatcher_failure(self) -> None:
        """
        Primary interpretation: Dispatcher's own contract never raises —
        "failure" means a returned status="error" dict. Also verifies the
        defense-in-depth case (an actual exception from the mock) lands
        the manager in ERROR, not a crash.
        """
        ctx = _make_ctx(
            dispatcher_return={"status": "error", "response": "Tool execution failed."}
        )
        audio = _make_audio_capture(b"\x01\x02")
        stt = _make_stt(STTResult(status="success", text="open chrome"))

        manager = VoiceManager(ctx=ctx, session_id="s1", audio_capture=audio, stt=stt)
        result = await manager.handle_voice_command()

        assert result["status"] == "error"
        assert result["response"] == "Tool execution failed."
        assert await manager.get_state() == VoiceState.IDLE

        # Defense-in-depth: an actual raised exception from the dispatcher
        # mock must land in ERROR, not propagate or crash the manager.
        ctx2 = _make_ctx()
        ctx2.dispatcher.handle = AsyncMock(side_effect=RuntimeError("dispatcher crashed"))
        audio2 = _make_audio_capture(b"\x01\x02")
        stt2 = _make_stt(STTResult(status="success", text="open chrome"))

        manager2 = VoiceManager(ctx=ctx2, session_id="s1", audio_capture=audio2, stt=stt2)
        result2 = await manager2.handle_voice_command()

        assert result2["status"] == "error"
        assert await manager2.get_state() == VoiceState.ERROR

    async def test_voice_rejects_reentry_when_not_idle(self) -> None:
        ctx = _make_ctx()
        audio = _make_audio_capture(b"\x01\x02")
        stt = _make_stt(STTResult(status="success", text="open chrome"))

        manager = VoiceManager(ctx=ctx, session_id="s1", audio_capture=audio, stt=stt)

        # Force a non-IDLE state directly, simulating an in-flight command.
        await manager._set_state(VoiceState.RECORDING)

        with pytest.raises(VoiceStateError):
            await manager.handle_voice_command()

        # State must remain unchanged by the rejected attempt.
        assert await manager.get_state() == VoiceState.RECORDING