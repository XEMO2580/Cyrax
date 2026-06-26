"""
brain/providers/base.py — CYRAX 3.0 Abstract Provider Contract

Every LLM provider adapter (Groq, Gemini, Ollama) must subclass BaseProvider
and implement generate(). The MoERouter interacts exclusively with this
interface — it never imports a concrete provider class directly.

Provider responsibilities:
  - Translate canonical message dicts into the vendor's native API format.
  - Translate the vendor's response back into a plain string.
  - Handle vendor-specific errors internally and re-raise as ProviderError.
  - Never leak vendor SDK objects, raw HTTP responses, or vendor exceptions
    above this boundary.

Canonical message format (what the router passes in):
    [
        {"role": "user",      "content": "open chrome"},
        {"role": "assistant", "content": "I opened chrome."},
        ...
    ]
    Valid roles: "user", "assistant", "system", "tool"

The system_prompt is passed separately so each provider can inject it
according to its native convention (system message, system_instruction
field, prepended user turn, etc.) without the router knowing the difference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER EXCEPTION
# ══════════════════════════════════════════════════════════════════════════════

class ProviderError(Exception):
    """
    Raised by provider adapters when a vendor API call fails unrecoverably.

    Attributes:
        provider:   Name of the provider that raised (e.g. "groq", "gemini").
        status_code: HTTP status code from the vendor, if available.
        retryable:  True if the caller may retry (e.g. transient 503).
                    False for hard failures (e.g. 401 invalid key, 400 bad request).
    """

    def __init__(
        self,
        message:     str,
        provider:    str  = "unknown",
        status_code: int  = 0,
        retryable:   bool = False,
    ) -> None:
        super().__init__(message)
        self.provider    = provider
        self.status_code = status_code
        self.retryable   = retryable

    def __repr__(self) -> str:
        return (
            f"ProviderError(provider={self.provider!r}, "
            f"status_code={self.status_code}, "
            f"retryable={self.retryable}, "
            f"message={str(self)!r})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER CAPABILITY FLAGS
# ══════════════════════════════════════════════════════════════════════════════

class ProviderCapabilities:
    """
    Declares what a provider natively supports.
    MoERouter reads these flags when selecting a provider for a given task.

    Attributes:
        supports_json_mode:     Provider can constrain output to valid JSON.
        supports_system_prompt: Provider accepts a dedicated system prompt field.
        supports_tool_schemas:  Provider can consume OpenAI-style function schemas.
        max_output_tokens:      Hard ceiling on tokens the provider will generate.
        context_window_tokens:  Maximum tokens the provider accepts as input.
    """

    def __init__(
        self,
        supports_json_mode:     bool = False,
        supports_system_prompt: bool = True,
        supports_tool_schemas:  bool = False,
        max_output_tokens:      int  = 1024,
        context_window_tokens:  int  = 8192,
    ) -> None:
        self.supports_json_mode     = supports_json_mode
        self.supports_system_prompt = supports_system_prompt
        self.supports_tool_schemas  = supports_tool_schemas
        self.max_output_tokens      = max_output_tokens
        self.context_window_tokens  = context_window_tokens


# ══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE PROVIDER
# ══════════════════════════════════════════════════════════════════════════════

class BaseProvider(ABC):
    """
    Abstract base class all CYRAX LLM provider adapters must subclass.

    Subclass responsibilities:
        1. Implement generate() — translate canonical messages to vendor format,
           call the vendor API, return a plain string.
        2. Set provider_name — used in logs and ProviderError.
        3. Set capabilities — MoERouter uses these for task-aware routing.
        4. Catch ALL vendor SDK exceptions inside generate() and re-raise
           as ProviderError. Nothing from the vendor SDK leaks above this class.

    The router always calls generate(). It never calls vendor SDK methods.
    """

    #: Short lowercase identifier. Set on every concrete subclass.
    provider_name: str = "base"

    #: Capability declaration. Override on every concrete subclass.
    capabilities: ProviderCapabilities = ProviderCapabilities()

    @abstractmethod
    async def generate(
        self,
        messages:      list[dict],
        system_prompt: str,
        *,
        max_tokens:    int  = 800,
        temperature:   float = 0.7,
        json_mode:     bool = False,
    ) -> str:
        """
        Generate a text response from the provider.

        Args:
            messages:      Canonical conversation history.
                           Format: [{"role": str, "content": str}, ...]
                           Roles: "user" | "assistant" | "system" | "tool"

            system_prompt: Instruction context for this call. Injected
                           according to the provider's native convention —
                           the caller never specifies how.

            max_tokens:    Maximum tokens to generate. Provider adapters
                           may cap this at their own ceiling.

            temperature:   Sampling temperature. 0.0 for deterministic
                           (planning), 0.7 for conversational.

            json_mode:     If True, the provider must return only valid JSON.
                           Only valid when capabilities.supports_json_mode is True.
                           Raises ProviderError if requested but unsupported.

        Returns:
            Plain string response from the provider. Never empty — raises
            ProviderError if the provider returns no content.

        Raises:
            ProviderError: On any vendor API failure, quota exhaustion, timeout,
                           or empty response. Always includes provider_name and
                           retryable flag so the failover layer can decide.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Performs a minimal liveness check against the provider.

        Used by MoERouter to verify a provider is reachable before routing
        a real request to it. Must be fast — a single minimal API call or
        a TCP connection check to the provider's endpoint.

        Returns:
            True if the provider is reachable and responding.
            False if unreachable or returning consistent errors.

        Must NOT raise — catch all exceptions and return False.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name!r})"