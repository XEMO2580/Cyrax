"""
brain/providers/gemini_provider.py — CYRAX 3.0 Gemini Provider Adapter

Translates canonical CYRAX message format into Google GenAI SDK format.
Uses the modern google-genai SDK (google.genai), not the legacy
google-generativeai SDK.

Key translation differences from Groq:
  - Gemini uses "model" role instead of "assistant"
  - System instructions are passed via GenerateContentConfig, not as a message
  - Multi-turn history uses types.Content and types.Part objects
  - Error types come from google.api_core.exceptions, not an OpenAI-compatible lib

All Gemini SDK exceptions are caught and re-raised as ProviderError.
Nothing from the Google SDK leaks above this file.
"""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from brain.providers.base import BaseProvider, ProviderCapabilities, ProviderError
from config.settings import settings

logger = logging.getLogger(__name__)

# HTTP status codes Google maps to specific exception types.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class GeminiProvider(BaseProvider):
    """
    Google Gemini LLM adapter using the modern google-genai SDK.

    Capabilities:
        - System instructions via GenerateContentConfig.system_instruction
        - Native multi-turn history via types.Content objects
        - Very large context window (1M tokens on Gemini 1.5 / 2.x)
        - JSON mode not natively supported — planner uses Groq instead

    Message format translation:
        Canonical role "assistant" → Gemini role "model"
        Canonical role "system"   → extracted, passed as system_instruction
        Canonical role "tool"     → mapped to "user" (Gemini tool-call format
                                    differs; full tool-call support is Priority 3)
        All messages converted to types.Content(role, parts=[types.Part.from_text()])
    """

    provider_name = "gemini"
    capabilities  = ProviderCapabilities(
        supports_json_mode     = False,
        supports_system_prompt = True,
        supports_tool_schemas  = False,
        max_output_tokens      = 8192,
        context_window_tokens  = 1048576,
    )

    def __init__(self) -> None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GeminiProvider cannot be initialised: "
                "GEMINI_API_KEY is empty. "
                "Check config/settings.py cross-field validation."
            )
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model  = settings.GEMINI_MODEL
        logger.info(
            f"[GEMINI] Provider initialised. Model: {self._model}"
        )

    async def generate(
        self,
        messages:      list[dict],
        system_prompt: str,
        *,
        max_tokens:    int   = 800,
        temperature:   float = 0.7,
        json_mode:     bool  = False,
    ) -> str:
        """
        Translates canonical messages into Gemini's types.Content format.

        system_prompt is injected via GenerateContentConfig.system_instruction.
        The last message in the list is treated as the current user turn.
        All preceding messages form the multi-turn history.
        """
        if json_mode:
            logger.warning(
                "[GEMINI] json_mode=True requested but Gemini does not support "
                "native JSON mode. Proceeding without — caller should use Groq "
                "for planning tasks."
            )

        contents, system_instruction = self._build_contents(
            messages, system_prompt
        )

        config = types.GenerateContentConfig(
            system_instruction = system_instruction or None,
            temperature        = temperature,
            max_output_tokens  = max_tokens,
        )

        logger.debug(
            f"[GEMINI] generate() | "
            f"contents={len(contents)} | "
            f"max_tokens={max_tokens} | "
            f"temperature={temperature}"
        )

        try:
            response = await self._client.aio.models.generate_content(
                model    = self._model,
                contents = contents,
                config   = config,
            )

            text = response.text
            if not text or not text.strip():
                raise ProviderError(
                    "Gemini returned an empty response.",
                    provider=self.provider_name,
                    status_code=0,
                    retryable=True,
                )

            logger.debug(
                f"[GEMINI] Response received. "
                f"Length: {len(text)} chars."
            )
            return text.strip()

        except google_exceptions.ResourceExhausted as exc:
            raise ProviderError(
                f"Gemini quota exhausted (429): {exc}",
                provider=self.provider_name,
                status_code=429,
                retryable=True,
            ) from exc

        except google_exceptions.Unauthenticated as exc:
            raise ProviderError(
                f"Gemini authentication failed — check GEMINI_API_KEY: {exc}",
                provider=self.provider_name,
                status_code=401,
                retryable=False,
            ) from exc

        except google_exceptions.PermissionDenied as exc:
            raise ProviderError(
                f"Gemini permission denied — API key may lack required scopes: {exc}",
                provider=self.provider_name,
                status_code=403,
                retryable=False,
            ) from exc

        except google_exceptions.InvalidArgument as exc:
            raise ProviderError(
                f"Gemini invalid argument — check message format or model name: {exc}",
                provider=self.provider_name,
                status_code=400,
                retryable=False,
            ) from exc

        except google_exceptions.DeadlineExceeded as exc:
            raise ProviderError(
                f"Gemini request deadline exceeded: {exc}",
                provider=self.provider_name,
                status_code=408,
                retryable=True,
            ) from exc

        except google_exceptions.ServiceUnavailable as exc:
            raise ProviderError(
                f"Gemini service unavailable (503): {exc}",
                provider=self.provider_name,
                status_code=503,
                retryable=True,
            ) from exc

        except google_exceptions.GoogleAPICallError as exc:
            status_code = getattr(exc, "code", 0) or 0
            retryable   = int(status_code) in _RETRYABLE_STATUS_CODES
            raise ProviderError(
                f"Gemini API error {status_code}: {exc}",
                provider=self.provider_name,
                status_code=int(status_code),
                retryable=retryable,
            ) from exc

        except ProviderError:
            raise

        except Exception as exc:
            raise ProviderError(
                f"Gemini unexpected error: {exc}",
                provider=self.provider_name,
                status_code=0,
                retryable=False,
            ) from exc

    async def health_check(self) -> bool:
        """
        Sends a minimal single-token request to verify Gemini is reachable.
        Returns False on any exception — never raises.
        """
        try:
            response = await self._client.aio.models.generate_content(
                model    = self._model,
                contents = [
                    types.Content(
                        role  = "user",
                        parts = [types.Part.from_text(text="ping")],
                    )
                ],
                config = types.GenerateContentConfig(max_output_tokens=1),
            )
            return bool(response.text)
        except Exception as exc:
            logger.warning(f"[GEMINI] health_check failed: {exc}")
            return False

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_contents(
        messages:      list[dict],
        system_prompt: str,
    ) -> tuple[list[types.Content], str]:
        """
        Converts canonical message dicts into Gemini types.Content objects.

        Returns:
            contents:           List of types.Content for the API call.
            system_instruction: Extracted system prompt string (may be empty).

        Role mapping:
            "user"      → "user"
            "assistant" → "model"   (Gemini's term for the AI turn)
            "system"    → extracted into system_instruction, not a content turn
            "tool"      → "user"    (simplified; full tool-call support is Priority 3)

        Empty content strings are filtered out to avoid Gemini API validation errors.
        """
        contents: list[types.Content] = []

        # Collect system messages into the instruction string.
        # Gemini handles system context best via system_instruction, not turns.
        system_parts: list[str] = []
        if system_prompt and system_prompt.strip():
            system_parts.append(system_prompt.strip())

        for msg in messages:
            role    = msg.get("role", "")
            content = str(msg.get("content", "")).strip()

            if not role or not content:
                continue

            if role == "system":
                system_parts.append(content)
                continue

            # Translate canonical roles to Gemini roles.
            if role == "assistant":
                gemini_role = "model"
            elif role in {"user", "tool"}:
                gemini_role = "user"
            else:
                logger.warning(
                    f"[GEMINI] Unknown role '{role}' — skipping message."
                )
                continue

            contents.append(
                types.Content(
                    role  = gemini_role,
                    parts = [types.Part.from_text(text=content)],
                )
            )

        system_instruction = "\n\n".join(system_parts)

        # Gemini requires the last content turn to be from "user".
        # If the history ends on a "model" turn (e.g. injected summary),
        # the API returns a 400. Validate and warn — the caller is responsible
        # for ensuring the user turn is last.
        if contents and contents[-1].role != "user":
            logger.warning(
                "[GEMINI] Last content turn is not 'user'. "
                "Gemini requires the final turn to be from 'user'. "
                "Check the message list passed to generate()."
            )

        return contents, system_instruction