"""
brain/providers/groq_provider.py — CYRAX 3.0 Groq Provider Adapter

Translates canonical CYRAX message format into Groq's OpenAI-compatible SDK.
All Groq SDK exceptions are caught and re-raised as ProviderError.
Nothing from the Groq SDK leaks above this file.
"""

from __future__ import annotations

import logging
from typing import Any

from groq import AsyncGroq
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)

from brain.providers.base import BaseProvider, ProviderCapabilities, ProviderError
from config.settings import settings

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):
    """
    Groq LLM adapter using the official AsyncGroq SDK.

    Capabilities:
        - Supports JSON mode via response_format={"type": "json_object"}
        - OpenAI-compatible role format (system / user / assistant)
        - Low latency — preferred for planning and classification calls

    Message format translation:
        Canonical "system" role messages are extracted and passed as the
        system parameter in the API call, not as a message in the array,
        because Groq (like OpenAI) expects system instructions separately
        when using the chat completions endpoint.
    """

    provider_name = "groq"
    capabilities  = ProviderCapabilities(
        supports_json_mode     = True,
        supports_system_prompt = True,
        supports_tool_schemas  = True,
        max_output_tokens      = 8192,
        context_window_tokens  = 131072,
    )

    def __init__(self) -> None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GroqProvider cannot be initialised: "
                "GROQ_API_KEY is empty. "
                "Check config/settings.py cross-field validation."
            )
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self._model  = settings.GROQ_MODEL
        logger.info(
            f"[GROQ] Provider initialised. Model: {self._model}"
        )

    async def generate(
        self,
        messages:      list[dict],
        system_prompt: str,
        *,
        max_tokens:    int   = 800,
        temperature:   float = 0.7,
        json_mode:     bool  = False,
        tools:         list[dict] | None = None,
    ) -> str:
        api_messages = self._build_messages(messages, system_prompt)

        request_kwargs: dict[str, Any] = {
            "model":       self._model,
            "messages":    api_messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

        if json_mode and self.capabilities.supports_json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        if tools:
            request_kwargs["tools"] = self.get_tool_schemas(tools)
            request_kwargs["tool_choice"] = "auto"

        logger.debug(
            f"[GROQ] generate() | messages={len(api_messages)} | "
            f"max_tokens={max_tokens} | temperature={temperature} | "
            f"json_mode={json_mode} | tools={len(tools) if tools else 0}"
        )

        try:
            response = await self._client.chat.completions.create(**request_kwargs)

            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ProviderError(
                    "Groq returned an empty response.",
                    provider=self.provider_name,
                    status_code=0,
                    retryable=True,
                )

            logger.debug(
                f"[GROQ] Response received. "
                f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}"
            )
            return content.strip()

        except RateLimitError as exc:
            raise ProviderError(
                f"Groq rate limit exceeded: {exc}",
                provider=self.provider_name,
                status_code=429,
                retryable=True,
            ) from exc

        except AuthenticationError as exc:
            raise ProviderError(
                f"Groq authentication failed — check GROQ_API_KEY: {exc}",
                provider=self.provider_name,
                status_code=401,
                retryable=False,
            ) from exc

        except BadRequestError as exc:
            raise ProviderError(
                f"Groq bad request — check message format or model name: {exc}",
                provider=self.provider_name,
                status_code=400,
                retryable=False,
            ) from exc

        except APITimeoutError as exc:
            raise ProviderError(
                f"Groq request timed out: {exc}",
                provider=self.provider_name,
                status_code=408,
                retryable=True,
            ) from exc

        except APIConnectionError as exc:
            raise ProviderError(
                f"Groq connection error — check network: {exc}",
                provider=self.provider_name,
                status_code=0,
                retryable=True,
            ) from exc

        except APIStatusError as exc:
            retryable = exc.status_code in {429, 500, 502, 503, 504}
            raise ProviderError(
                f"Groq API error {exc.status_code}: {exc.message}",
                provider=self.provider_name,
                status_code=exc.status_code,
                retryable=retryable,
            ) from exc

        except ProviderError:
            raise

        except Exception as exc:
            raise ProviderError(
                f"Groq unexpected error: {exc}",
                provider=self.provider_name,
                status_code=0,
                retryable=False,
            ) from exc

    async def health_check(self) -> bool:
        """
        Sends a minimal single-token request to verify Groq is reachable.
        Returns False on any exception — never raises.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
            return bool(response.choices)
        except Exception as exc:
            logger.warning(f"[GROQ] health_check failed: {exc}")
            return False

    # ── Schema translation (Phase 6, Constraint 4) ───────────────────────────

    @staticmethod
    def get_tool_schemas(tool_definitions: list[dict]) -> list[dict[str, Any]]:
        """
        Translates canonical ToolRegistry definitions (name, description,
        Pydantic-derived JSON Schema "parameters") into Groq's
        OpenAI-compatible function-calling array:

            [{"type": "function", "function": {
                "name": ..., "description": ..., "parameters": {...}
            }}, ...]

        Groq's format is a near-direct passthrough of standard JSON Schema —
        no restructuring of the parameters object is needed, only wrapping.
        """
        schemas: list[dict[str, Any]] = []
        for tool_def in tool_definitions:
            schemas.append({
                "type": "function",
                "function": {
                    "name":        tool_def.get("name", ""),
                    "description": tool_def.get("description", ""),
                    "parameters":  tool_def.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return schemas

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        messages:      list[dict],
        system_prompt: str,
    ) -> list[dict[str, str]]:
        """
        Constructs the Groq-compatible message array.

        Groq (OpenAI-compatible) format:
            [
                {"role": "system",    "content": "<system_prompt>"},
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]

        The injected system_prompt is always first.
        Timestamps and any non-standard keys are stripped.
        Empty content strings are filtered out.
        """
        api_messages: list[dict[str, str]] = []

        if system_prompt and system_prompt.strip():
            api_messages.append({
                "role":    "system",
                "content": system_prompt.strip(),
            })

        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")

            if not role or not content or not str(content).strip():
                continue

            # Normalise role: "tool" → "user" for Groq compatibility.
            # Groq does not support role="tool" in standard chat completions.
            if role == "tool":
                role = "user"

            api_messages.append({
                "role":    role,
                "content": str(content).strip(),
            })

        return api_messages