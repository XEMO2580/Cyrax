"""
voice/stt/base.py — Abstract, synchronous STT adapter contract.

Fully synchronous by contract — VoiceManager wraps every call via
asyncio.to_thread(). Do not add async here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class STTResult:
    """
    status:  "success" | "empty" | "timeout" | "error"
    text:    Transcribed text. Empty unless status == "success".
    message: Human-readable detail for non-success statuses.
    """
    status:  str
    text:    str
    message: str = ""


class BaseSTT(ABC):
    """
    Implementations MUST:
      - Be fully synchronous.
      - Never raise from transcribe() — all failures are represented as
        STTResult.status in {"empty", "timeout", "error"}.
      - Respect the timeout_seconds argument for the network/API call itself
        (independent of AudioCapture's recording bound).
    """

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, timeout_seconds: float) -> STTResult:
        """
        Transcribes audio_bytes to text.

        Args:
            audio_bytes:     Raw audio from AudioCapture.record_audio().
                              Empty bytes MUST be handled by returning
                              STTResult(status="empty", ...) without
                              attempting an API call.
            timeout_seconds:  Maximum time to wait for the STT API response.

        Returns:
            STTResult. NEVER raises.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """True if this adapter is configured and ready (e.g., API key present)."""
        raise NotImplementedError