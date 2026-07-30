# """
# voice/audio_in.py — Synchronous, bounded hardware audio capture interface.

# CRITICAL (Constraint 5): record_audio() MUST enforce a hard upper time
# limit and must NEVER block indefinitely on a stalled microphone. If the
# backend stalls or fails, it returns empty bytes or raises — VoiceManager
# is responsible for catching either and resetting to IDLE.

# This file is fully synchronous by contract. VoiceManager wraps every call
# via asyncio.to_thread() — do not add async here.

# No concrete hardware SDK (pyaudio, sounddevice, etc.) is wired in this
# phase. AudioCapture is a real class with a real bounded-loop contract,
# but the actual microphone read is a placeholder point clearly marked
# for Phase 5.2 hardware integration — wiring a specific library was not
# authorized by this directive and is not guessed at here.
# """

# from __future__ import annotations

# import logging
# import time

# from config.settings import settings

# logger = logging.getLogger(__name__)


# class AudioCapture:
#     """
#     Bounded microphone capture.

#     Args:
#         sample_rate: Hz, sourced from settings.VOICE_SAMPLE_RATE.
#                      Stored for use by the real backend in Phase 5.2;
#                      not consumed elsewhere in this phase's mocked flow.
#     """

#     def __init__(self, sample_rate: int | None = None) -> None:
#         self._sample_rate = sample_rate or settings.VOICE_SAMPLE_RATE

#     def record_audio(self, max_duration_sec: int | None = None) -> bytes:
#         """
#         Captures audio from the microphone for at most max_duration_sec.

#         Contract (Constraint 5):
#           - MUST return within max_duration_sec regardless of backend state.
#           - Returns b"" (empty bytes) on failure, stall, or no input device —
#             never blocks indefinitely waiting for a fix.
#           - May also raise on a hard backend fault (e.g., device not found);
#             VoiceManager catches this and resets to IDLE, per Constraint 3.

#         Args:
#             max_duration_sec: Overrides settings.VOICE_MAX_RECORD_SECONDS
#                                for this call. Defaults to the config value
#                                so callers never hardcode a duration.

#         Returns:
#             Raw audio bytes, or b"" if no audio was captured.

#         PHASE 5.2 NOTE: The actual hardware read (e.g., a real backend's
#         blocking stream read) belongs inside the loop below, replacing the
#         placeholder. The bounded-loop structure and the deadline check are
#         the real Phase 5.0 deliverable — the hardware call itself is not
#         specified by this directive and is not invented here.
#         """
#         limit = max_duration_sec or settings.VOICE_MAX_RECORD_SECONDS
#         deadline = time.monotonic() + limit

#         logger.info(f"[AUDIO_IN] Recording started. Max duration: {limit}s.")

#         try:
#             audio_chunks: list[bytes] = []

#             while time.monotonic() < deadline:
#                 # PHASE 5.2 INTEGRATION POINT:
#                 # Replace this placeholder with a real bounded read from the
#                 # chosen audio backend, e.g.:
#                 #     chunk = stream.read(frame_count, exception_on_overflow=False)
#                 # The loop's own deadline check above already guarantees
#                 # termination — a real backend read should itself also use a
#                 # short per-chunk timeout so a single stalled read cannot
#                 # consume the entire remaining budget silently.
#                 break  # No hardware backend wired in this phase.

#             recorded = b"".join(audio_chunks)

#             if not recorded:
#                 logger.warning("[AUDIO_IN] No audio captured (empty result).")

#             return recorded

#         except Exception as exc:
#             logger.error(f"[AUDIO_IN] Capture failed: {exc}")
#             return b""




"""
voice/audio_in.py — Synchronous, bounded hardware audio capture.
"""
from __future__ import annotations
import io
import logging
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from config.settings import settings

logger = logging.getLogger(__name__)

_CHANNELS: int = 1
_DTYPE:    str = "int16"

class AudioCapture:
    def __init__(self, sample_rate: int | None = None) -> None:
        self._sample_rate = sample_rate or settings.VOICE_SAMPLE_RATE

    def record_audio(self, max_duration_sec: int | None = None) -> bytes:
        duration = max_duration_sec or settings.VOICE_MAX_RECORD_SECONDS
        frame_count = int(duration * self._sample_rate)

        logger.info(
            f"[AUDIO_IN] Recording {duration}s at {self._sample_rate}Hz "
            f"({frame_count} frames)."
        )

        try:
            recording: np.ndarray = sd.rec(
                frame_count,
                samplerate=self._sample_rate,
                channels=_CHANNELS,
                dtype=_DTYPE,
            )
            sd.wait()
        except sd.PortAudioError as exc:
            logger.error(f"[AUDIO_IN] PortAudio error: {exc}")
            return b""
        except Exception as exc:
            logger.error(f"[AUDIO_IN] Unexpected capture failure: {exc}")
            return b""

        if recording is None or recording.size == 0:
            logger.warning("[AUDIO_IN] Recording produced no samples.")
            return b""

        try:
            buffer = io.BytesIO()
            wavfile.write(buffer, self._sample_rate, recording)
            wav_bytes = buffer.getvalue()
            buffer.close()
        except Exception as exc:
            logger.error(f"[AUDIO_IN] WAV encoding failed: {exc}")
            return b""

        if not wav_bytes:
            logger.warning("[AUDIO_IN] WAV encoding produced empty bytes.")
            return b""

        logger.info(f"[AUDIO_IN] Captured {len(wav_bytes):,} bytes of WAV audio.")
        return wav_bytes