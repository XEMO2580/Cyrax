"""
voice/manager.py — CYRAX 3.0 Phase 5.0 Push-to-Talk VoiceManager.

Scope (strict — do not extend):
  - Text-triggered only (:voice CLI command). No hotkeys, no wake word.
  - No TTS. Dispatcher's text response is returned as-is; nothing speaks it.
  - No barge-in / cancellation — a voice command runs to completion or
    fails; there is nothing to interrupt.

Architectural contracts:
  - AudioCapture.record_audio() and BaseSTT.transcribe() run via
    asyncio.to_thread() — the event loop is never blocked.
  - Only this class calls ctx.events.publish(). Worker-thread functions
    (AudioCapture, STT adapters) return plain data and hold no reference
    to ctx or ctx.events — this is structural, not just documented.
  - Every exit path from handle_voice_command() ends in IDLE (known,
    anticipated failures) or ERROR (unexpected/unhandled failures) —
    never left in RECORDING/TRANSCRIBING/DISPATCHING.
  - Re-entrant :voice commands while not IDLE raise VoiceStateError
    immediately — no queuing, no silent ignoring.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from config.settings import settings
from core.context import CyraxContext
from core.trace import new_trace_id
from voice.audio_in import AudioCapture
from voice.state import VoiceState, VoiceStateError
from voice.stt.base import BaseSTT

logger = logging.getLogger(__name__)


class VoiceManager:
    """
    Orchestrates one :voice command: Audio -> STT -> Dispatcher.

    Args:
        ctx:          CyraxContext — same object the text interface uses.
        session_id:   Passed through unmodified to ctx.dispatcher.handle().
        audio_capture: Injected AudioCapture instance (or a test double).
        stt:           Injected BaseSTT implementation (or a test double).
    """

    def __init__(
        self,
        ctx:            CyraxContext,
        session_id:     str,
        audio_capture:  AudioCapture,
        stt:            BaseSTT,
    ) -> None:
        self._ctx           = ctx
        self._session_id    = session_id
        self._audio_capture = audio_capture
        self._stt           = stt

        self._state:      VoiceState   = VoiceState.IDLE
        self._state_lock: asyncio.Lock = asyncio.Lock()

    async def get_state(self) -> VoiceState:
        async with self._state_lock:
            return self._state

    async def _set_state(self, new_state: VoiceState) -> None:
        async with self._state_lock:
            self._state = new_state

    async def handle_voice_command(self) -> dict[str, Any]:
        """
        Full :voice command lifecycle. Returns a dict compatible with
        ctx.dispatcher.handle()'s own return shape: {"status": ..., "response": ...}.

        Raises:
            VoiceStateError: if called while not currently IDLE. This is
                              a hard rejection, not a queued retry.
        """
        async with self._state_lock:
            if self._state != VoiceState.IDLE:
                raise VoiceStateError(self._state)
            self._state = VoiceState.RECORDING

        trace_id = new_trace_id()
        logger.info(f"[VOICE:{trace_id}] Command started.")

        try:
            # ── RECORDING ─────────────────────────────────────────────────────
            audio_bytes: bytes = await asyncio.to_thread(
                self._audio_capture.record_audio,
                settings.VOICE_MAX_RECORD_SECONDS,
            )

            if not audio_bytes:
                logger.warning(f"[VOICE:{trace_id}] Empty audio capture. Resetting to IDLE.")
                await self._set_state(VoiceState.IDLE)
                return {"status": "error", "response": "No audio was captured."}

            # ── TRANSCRIBING ──────────────────────────────────────────────────
            await self._set_state(VoiceState.TRANSCRIBING)

            stt_result = await asyncio.to_thread(
                self._stt.transcribe,
                audio_bytes,
                settings.VOICE_STT_TIMEOUT_SECONDS,
            )

            if stt_result.status == "timeout":
                logger.error(f"[VOICE:{trace_id}] STT timed out. Resetting to IDLE.")
                await self._set_state(VoiceState.IDLE)
                return {"status": "error", "response": "Transcription timed out."}

            if stt_result.status != "success" or not stt_result.text.strip():
                logger.warning(
                    f"[VOICE:{trace_id}] Empty/failed transcript "
                    f"(status={stt_result.status}). Resetting to IDLE."
                )
                await self._set_state(VoiceState.IDLE)
                return {
                    "status": "error",
                    "response": stt_result.message or "No speech was transcribed.",
                }

            # ── DISPATCHING ───────────────────────────────────────────────────
            await self._set_state(VoiceState.DISPATCHING)

            dispatcher_result = await self._ctx.dispatcher.handle(
                user_input=stt_result.text,
                session_id=self._session_id,
                trace_id=trace_id,
                ctx=self._ctx,
            )

            await self._set_state(VoiceState.IDLE)

            if dispatcher_result.get("status") == "error":
                logger.warning(f"[VOICE:{trace_id}] Dispatcher returned error status.")
            else:
                logger.info(f"[VOICE:{trace_id}] Command completed successfully.")

            # Only VoiceManager (main-loop code) touches ctx.events.
            if hasattr(self._ctx, "events") and self._ctx.events is not None:
                await self._ctx.events.publish(
                    {"type": "voice_command_completed", "trace_id": trace_id}
                )

            dispatcher_result["transcript"] = stt_result.text

            return dispatcher_result

        except Exception as exc:
            logger.exception(f"[VOICE:{trace_id}] Unhandled failure: {exc}")
            await self._set_state(VoiceState.ERROR)
            return {"status": "error", "response": "Voice command failed unexpectedly."}