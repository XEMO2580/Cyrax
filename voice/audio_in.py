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