"""
brain/providers/gemini_provider.py — CYRAX 3.0 Gemini Provider Adapter (Phase 6)

FIX: Corrected undefined-name bug — all exception handlers referenced
"google_exceptions" but the actual import was aliased "genai_errors".
Every handler below is fixed to use genai_errors.

ADDED: get_tool_schemas() — translates ToolRegistry's canonical definitions
into Gemini's types.FunctionDeclaration / types.Tool format, per Phase 6
Constraint 4. Gemini's schema dialect differs from OpenAI's: only a subset
of JSON Schema types is supported natively (STRING, NUMBER, INTEGER,
BOOLEAN, ARRAY, OBJECT), and $defs/$ref are not supported — nested
Pydantic models must be inlined, which this translation performs.

FIX (Phase 7 SDK Patch): Replaced granular genai_errors.ResourceExhausted,
Unauthenticated, PermissionDenied etc. with a single catch of
genai_errors.APIError, which is the only exception type the new
google.genai SDK raises. Status-code-specific messages are preserved
via conditional checks on exc.code.
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

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Gemini's Schema.type only accepts this subset — JSON Schema's broader
# type vocabulary must be mapped down onto these.
_JSON_SCHEMA_TO_GEMINI_TYPE: dict[str, str] = {
    "string":  "STRING",
    "number":  "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array":   "ARRAY",
    "object":  "OBJECT",
}


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
        supports_tool_schemas  = True,
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
        tools:         list[dict] | None = None,
    ) -> str:
        if json_mode:
            logger.warning(
                "[GEMINI] json_mode=True requested but Gemini does not support "
                "native JSON mode. Proceeding without — caller should use Groq "
                "for planning tasks."
            )

        contents, system_instruction = self._build_contents(messages, system_prompt)

        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction or None,
            "temperature":        temperature,
            "max_output_tokens":  max_tokens,
        }

        if tools:
            config_kwargs["tools"] = [self.get_tool_schemas(tools)]

        config = types.GenerateContentConfig(**config_kwargs)

        logger.debug(
            f"[GEMINI] generate() | contents={len(contents)} | "
            f"max_tokens={max_tokens} | temperature={temperature} | "
            f"tools={len(tools) if tools else 0}"
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
                    provider=self.provider_name, status_code=0, retryable=True,
                )

            logger.debug(f"[GEMINI] Response received. Length: {len(text)} chars.")
            return text.strip()

        except genai_errors.APIError as exc:
            # The new google.genai SDK wraps most HTTP/API errors here
            status_code = getattr(exc, "code", 0) or 0
            retryable = int(status_code) in _RETRYABLE_STATUS_CODES
            
            if status_code == 429:
                msg = f"Gemini quota exhausted (429): {exc}"
            elif status_code == 401:
                msg = f"Gemini authentication failed — check GEMINI_API_KEY: {exc}"
            elif status_code == 403:
                msg = f"Gemini permission denied: {exc}"
            elif status_code == 400:
                msg = f"Gemini invalid argument: {exc}"
            else:
                msg = f"Gemini API error {status_code}: {exc}"
                
            raise ProviderError(
                msg, provider=self.provider_name, status_code=int(status_code), retryable=retryable,
            ) from exc

        except ProviderError:
            raise

        except Exception as exc:
            raise ProviderError(
                f"Gemini unexpected error: {exc}",
                provider=self.provider_name, status_code=0, retryable=False,
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

    # ── Schema translation (Phase 6, Constraint 4) ───────────────────────────

    @classmethod
    def get_tool_schemas(cls, tool_definitions: list[dict]) -> types.Tool:
        """
        Translates canonical ToolRegistry definitions into a single
        types.Tool wrapping one types.FunctionDeclaration per tool.

        Nested JSON Schema "parameters" objects are converted via
        _convert_schema_properties() into Gemini's types.Schema tree.
        """
        declarations: list[types.FunctionDeclaration] = []

        for tool_def in tool_definitions:
            params_schema = tool_def.get("parameters", {"type": "object", "properties": {}})
            gemini_schema = cls._convert_json_schema(params_schema)

            declarations.append(
                types.FunctionDeclaration(
                    name=tool_def.get("name", ""),
                    description=tool_def.get("description", ""),
                    parameters=gemini_schema,
                )
            )

        return types.Tool(function_declarations=declarations)

    @classmethod
    def _convert_json_schema(cls, json_schema: dict[str, Any]) -> types.Schema:
        """
        Recursively converts a standard JSON Schema dict (as produced by
        Pydantic's model_json_schema()) into Gemini's types.Schema.

        Handles: type mapping, nested object properties, array items,
        required fields, and enum values. Does not resolve $ref/$defs —
        Pydantic schemas passed through the Registry are expected to be
        flat (single-level args_schema classes, no nested BaseModel
        fields) per existing tool design; if that assumption changes,
        this method needs $ref resolution added.
        """
        schema_type = json_schema.get("type", "object")
        gemini_type = _JSON_SCHEMA_TO_GEMINI_TYPE.get(schema_type, "STRING")

        kwargs: dict[str, Any] = {"type": gemini_type}

        if "description" in json_schema:
            kwargs["description"] = json_schema["description"]

        if "enum" in json_schema:
            kwargs["enum"] = [str(v) for v in json_schema["enum"]]

        if gemini_type == "OBJECT":
            properties = json_schema.get("properties", {})
            kwargs["properties"] = {
                key: cls._convert_json_schema(sub_schema)
                for key, sub_schema in properties.items()
            }
            required = json_schema.get("required", [])
            if required:
                kwargs["required"] = required

        if gemini_type == "ARRAY":
            items_schema = json_schema.get("items", {"type": "string"})
            kwargs["items"] = cls._convert_json_schema(items_schema)

        return types.Schema(**kwargs)

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
