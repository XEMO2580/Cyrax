"""
brain/providers/groq_provider.py — CYRAX 3.0 Groq Provider Adapter

Translates canonical CYRAX message format into Groq's OpenAI-compatible SDK.
All Groq SDK exceptions are caught and re-raised as ProviderError.
Nothing from the Groq SDK leaks above this file.
"""

from __future__ import annotations

import asyncio
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

from brain.providers.base import (
    BaseProvider,
    ProviderCapabilities,
    ProviderError,
    GenerationCancelledError,
    GenerationMode,
    StreamLifecycleState,
)
from core.interrupt_controller import CancellationToken
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
        text                   = True,
        streaming              = True,
        structured_json        = True,
        cancellation           = True,
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
        generation_mode: "GenerationMode | str" = GenerationMode.TEXT,
        json_mode:     bool  = False,
        tools:         list[dict] | None = None,
        cancel_token: Any = None,
    ) -> str:
        api_messages = self._build_messages(messages, system_prompt)

        # ── Streaming path (if cancel_token supplied) ───────────────────────
        if cancel_token is not None:
            try:
                mode = generation_mode if isinstance(generation_mode, GenerationMode) else GenerationMode(generation_mode)
            except Exception:
                mode = GenerationMode.TEXT
            stream_json = mode == GenerationMode.STRUCTURED_JSON
            return await self._generate_streaming(
                api_messages, max_tokens, temperature, stream_json, cancel_token
            )

        # ── Unchanged non-streaming path ────────────────────────────────────
        request_kwargs: dict[str, Any] = {
            "model":       self._model,
            "messages":    api_messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

        # Backwards-compatible handling: if caller supplied generation_mode use that,
        # otherwise fall back to the legacy json_mode boolean.
        try:
            mode = generation_mode if isinstance(generation_mode, GenerationMode) else GenerationMode(generation_mode)
        except Exception:
            mode = GenerationMode.TEXT

        json_request = (mode == GenerationMode.STRUCTURED_JSON) or json_mode
        if json_request and self.capabilities.supports_json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        if tools:
            request_kwargs["tools"] = self.get_tool_schemas(tools)
            request_kwargs["tool_choice"] = "auto"

        logger.debug(
            f"[GROQ] generate() | messages={len(api_messages)} | "
            f"max_tokens={max_tokens} | temperature={temperature} | "
            f"generation_mode={mode.value if 'mode' in locals() else str(generation_mode)} | "
            f"response_format_present={'response_format' in request_kwargs} | tools={len(tools) if tools else 0}"
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

            cleaned_content = content.strip()

            if json_request:
                import json
                try:
                    json.loads(cleaned_content)
                except Exception as json_err:
                    raise ProviderError(
                        f"Groq output is not valid JSON in structured mode: {json_err}",
                        provider=self.provider_name,
                        status_code=0,
                        retryable=True,
                    ) from json_err

            logger.debug(
                f"[GROQ] Response received. "
                f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}"
            )
            return cleaned_content

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
            # Groq returns a 400 when server-side JSON validation fails
            # (json_validate_failed). Treat that as a provider-formatting
            # failure that should allow failover to another model/provider
            # rather than a hard stop. Log developer-safe diagnostics
            # (no API keys or full prompts).
            text = str(exc)
            is_json_validate = "json_validate_failed" in text or "Failed to validate JSON" in text or "json_validate" in text

            logger.warning(
                f"[GROQ] BadRequestError (400) from Groq. json_validate_failed={is_json_validate}."
                f" model={self._model} response_format_set={('response_format' in request_kwargs)}"
            )

            # When it's a JSON validation problem, allow failover by marking
            # the ProviderError as retryable=True so MoERouter can try others.
            retryable_flag = True if is_json_validate else False

            raise ProviderError(
                f"Groq bad request — check message format or model name: {exc}",
                provider=self.provider_name,
                status_code=400,
                retryable=retryable_flag,
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

    async def _generate_streaming(
        self,
        api_messages:  list[dict],
        max_tokens:    int,
        temperature:   float,
        json_mode:     bool,
        cancel_token:  CancellationToken,
    ) -> str:
        """
        Requirement 17/18: streams chunks via AsyncGroq's stream=True
        interface, checking cancel_token.is_cancelled after EVERY chunk.
        On cancellation, immediately stops iterating (closes the async
        generator, which per the SDK's httpx-backed transport terminates
        the underlying HTTP connection rather than reading it to completion),
        discards all accumulated chunks, and raises GenerationCancelledError.
        Never returns partial content as a "success" — either the full
        stream completes and is returned, or it's cancelled and nothing
        is returned.
        """
        request_kwargs: dict[str, Any] = {
            "model":       self._model,
            "messages":    api_messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      True,
        }
        
        if json_mode and self.capabilities.supports_json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        logger.debug(f"[GROQ:STREAM] {StreamLifecycleState.STARTED.value} | model={self._model}")
        accumulated: list[str] = []

        try:
            stream = await self._client.chat.completions.create(**request_kwargs)

            async for chunk in stream:
                # Requirement 16: worker observes the cancellation primitive
                # DURING the active chunk-generation loop, not just before
                # or after the call.
                if cancel_token.is_cancelled:
                    logger.info(
                        f"[GROQ:STREAM] {StreamLifecycleState.CANCELLED.value} observed mid-stream "
                        f"after {len(accumulated)} chunk(s). Closing stream, "
                        f"discarding partial output."
                    )
                    if hasattr(stream, "close"):
                        close_fn = getattr(stream, "close")
                        if asyncio.iscoroutinefunction(close_fn) or asyncio.iscoroutine(close_fn):
                            await stream.close()
                        else:
                            try:
                                res = stream.close()
                                if asyncio.iscoroutine(res):
                                    await res
                            except Exception:
                                pass
                    elif hasattr(stream, "aclose"):
                        try:
                            await stream.aclose()
                        except Exception:
                            pass
                    raise GenerationCancelledError(partial_chunks_discarded=len(accumulated))

                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    accumulated.append(delta)
                    logger.debug(
                        f"[GROQ:STREAM] {StreamLifecycleState.CHUNK_RECEIVED.value} | "
                        f"chunk_len={len(delta)} | total_chunks={len(accumulated)}"
                    )

            full_content = "".join(accumulated).strip()
            if not full_content:
                logger.warning(f"[GROQ:STREAM] {StreamLifecycleState.EMPTY.value} response received.")
                raise ProviderError(
                    "Groq streaming call returned no content.",
                    provider=self.provider_name, 
                    status_code=0, 
                    retryable=True,
                )

            if json_mode:
                import json
                try:
                    json.loads(full_content)
                except Exception as json_err:
                    logger.warning(f"[GROQ:STREAM] {StreamLifecycleState.FAILED.value} | invalid JSON in structured mode: {json_err}")
                    raise ProviderError(
                        f"Groq output is not valid JSON in structured streaming mode: {json_err}",
                        provider=self.provider_name,
                        status_code=0,
                        retryable=True,
                    ) from json_err

            logger.debug(f"[GROQ:STREAM] {StreamLifecycleState.COMPLETED.value} | Length: {len(full_content)} chars.")
            return full_content

        except GenerationCancelledError:
            raise

        except RateLimitError as exc:
            logger.warning(f"[GROQ:STREAM] {StreamLifecycleState.FAILED.value} | Rate limit: {exc}")
            raise ProviderError(f"Groq rate limit exceeded: {exc}", provider=self.provider_name, status_code=429, retryable=True) from exc
        except AuthenticationError as exc:
            logger.error(f"[GROQ:STREAM] {StreamLifecycleState.FAILED.value} | Auth failed: {exc}")
            raise ProviderError(f"Groq authentication failed: {exc}", provider=self.provider_name, status_code=401, retryable=False) from exc
        except (APITimeoutError, TimeoutError, asyncio.TimeoutError) as exc:
            logger.warning(f"[GROQ:STREAM] {StreamLifecycleState.TIMEOUT.value} | {exc}")
            raise ProviderError(f"Groq streaming request timed out: {exc}", provider=self.provider_name, status_code=408, retryable=True) from exc
        except APIConnectionError as exc:
            logger.warning(f"[GROQ:STREAM] {StreamLifecycleState.FAILED.value} | Connection error: {exc}")
            raise ProviderError(f"Groq connection error: {exc}", provider=self.provider_name, status_code=0, retryable=True) from exc
        except ProviderError:
            raise
        except BadRequestError as exc:
            text = str(exc)
            is_json_validate = "json_validate_failed" in text or "Failed to validate JSON" in text or "json_validate" in text
            logger.warning(
                f"[GROQ] Streaming BadRequestError (400). json_validate_failed={is_json_validate}. model={self._model}"
            )
            retryable_flag = True if is_json_validate else False
            raise ProviderError(f"Groq streaming bad request: {exc}", provider=self.provider_name, status_code=400, retryable=retryable_flag) from exc
        except Exception as exc:
            err_msg = str(exc).lower()
            is_timeout = "timeout" in err_msg or "timed out" in err_msg
            status_code = 408 if is_timeout else 0
            retryable = is_timeout or "connection" in err_msg or "stream" in err_msg
            logger.warning(f"[GROQ:STREAM] {StreamLifecycleState.FAILED.value} | {exc}")
            raise ProviderError(f"Groq streaming unexpected error: {exc}", provider=self.provider_name, status_code=status_code, retryable=retryable) from exc

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