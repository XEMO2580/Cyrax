# """
# voice/stt/openai_adapter.py — OpenAI Whisper API STT adapter.

# Synchronous by contract (see base.py). The OpenAI SDK call itself may be
# sync or async depending on the client used; this adapter guarantees a
# synchronous public interface regardless, since VoiceManager wraps it in
# asyncio.to_thread() and must not need to know which.

# No API call is exercised in this phase's test suite — tests mock this
# class entirely. The concrete request/response handling here is real
# structure, not a stub, but is the first place actual OpenAI SDK
# integration work happens in Phase 5.2+.
# """

# from __future__ import annotations

# import logging

# from config.settings import settings
# from voice.stt.base import BaseSTT, STTResult

# logger = logging.getLogger(__name__)


# class OpenAISTTAdapter(BaseSTT):
#     """
#     Wraps the OpenAI Whisper transcription endpoint.

#     Availability requires settings.OPENAI_API_KEY (or equivalent) to be
#     configured. If absent, is_available() returns False and VoiceManager
#     is responsible for surfacing that as a degraded/unavailable voice
#     subsystem rather than attempting a doomed API call.
#     """

#     def __init__(self) -> None:
#         self._api_key = getattr(settings, "OPENAI_API_KEY", "") or ""

#     def transcribe(self, audio_bytes: bytes, timeout_seconds: float) -> STTResult:
#         if not audio_bytes:
#             return STTResult(status="empty", text="", message="No audio to transcribe.")

#         if not self.is_available():
#             return STTResult(
#                 status="error", text="",
#                 message="OpenAI STT adapter is not configured (missing API key).",
#             )

#         try:
#             # PHASE 5.2 INTEGRATION POINT:
#             # Real call shape, illustrative of the contract this method
#             # must satisfy — replace the body with an actual OpenAI SDK
#             # invocation (e.g. client.audio.transcriptions.create(...))
#             # inside a timeout-bounded call. No SDK is imported or invoked
#             # here; wiring a concrete library was not authorized by this
#             # directive.
#             raise NotImplementedError(
#                 "OpenAI SDK integration is a Phase 5.2 deliverable, not "
#                 "authorized in this Phase 5.0 directive."
#             )

#         except TimeoutError:
#             logger.error(f"[STT:OpenAI] Request timed out after {timeout_seconds}s.")
#             return STTResult(
#                 status="timeout", text="",
#                 message=f"STT request timed out after {timeout_seconds}s.",
#             )
#         except NotImplementedError:
#             raise
#         except Exception as exc:
#             logger.error(f"[STT:OpenAI] Transcription failed: {exc}")
#             return STTResult(status="error", text="", message=str(exc))

#     def is_available(self) -> bool:
#         return bool(self._api_key)


"""
voice/stt/openai_adapter.py — OpenAI-compatible STT adapter.

Uses the official openai Python SDK. settings.OPENAI_BASE_URL may
reroute requests to Groq's free Whisper endpoint.
"""
from __future__ import annotations

import io
import logging

from openai import OpenAI, APITimeoutError, APIError, APIConnectionError

from config.settings import settings
from voice.stt.base import BaseSTT, STTResult

logger = logging.getLogger(__name__)

_AUDIO_FILENAME: str = "voice_command.wav"


class OpenAISTTAdapter(BaseSTT):
    def __init__(self) -> None:
        self._api_key = getattr(settings, "OPENAI_API_KEY", "") or ""
        self._base_url = getattr(settings, "OPENAI_BASE_URL", None)
        self._model = getattr(settings, "OPENAI_STT_MODEL", "whisper-large-v3")
        
        self._client: OpenAI | None = None

        if self._api_key:
            client_kwargs: dict = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = OpenAI(**client_kwargs)

    def transcribe(self, audio_bytes: bytes, timeout_seconds: float) -> STTResult:
        if not audio_bytes:
            return STTResult(status="empty", text="", message="No audio to transcribe.")

        if not self.is_available():
            return STTResult(
                status="error", text="",
                message="STT adapter is not configured (missing OPENAI_API_KEY).",
            )

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = _AUDIO_FILENAME  # SDK requires a filename attribute.

        try:
            response = self._client.audio.transcriptions.create(
                model=self._model,
                file=audio_file,
                timeout=timeout_seconds,
            )

            text = (response.text or "").strip()

            if not text:
                return STTResult(status="empty", text="", message="Transcription was empty.")

            return STTResult(status="success", text=text)

        except APITimeoutError:
            logger.error(f"[STT:OpenAI] Request timed out after {timeout_seconds}s.")
            return STTResult(status="timeout", text="", message=f"STT request timed out after {timeout_seconds}s.")
        except APIConnectionError as exc:
            logger.error(f"[STT:OpenAI] Connection error: {exc}")
            return STTResult(status="error", text="", message=f"Connection error: {exc}")
        except APIError as exc:
            logger.error(f"[STT:OpenAI] API error: {exc}")
            return STTResult(status="error", text="", message=f"API error: {exc}")
        except Exception as exc:
            logger.error(f"[STT:OpenAI] Unexpected error: {exc}")
            return STTResult(status="error", text="", message=str(exc))
        finally:
            audio_file.close()

    def is_available(self) -> bool:
        return self._client is not None