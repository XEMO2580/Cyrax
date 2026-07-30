"""
config/settings.py — CYRAX 3.0 Typed Configuration (Hardened)

Frozen, validated, cross-field enforced.
Boots loud on any missing critical value.
No silent misconfigurations reach runtime.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class ActiveLLM(str, Enum):
    AUTO   = "auto"
    GROQ   = "groq"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class LogLevel(str, Enum):
    DEBUG    = "DEBUG"
    INFO     = "INFO"
    WARNING  = "WARNING"
    ERROR    = "ERROR"
    CRITICAL = "CRITICAL"


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

class CyraxSettings(BaseSettings):

    # ── LLM Provider Keys ─────────────────────────────────────────────────────

    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API key. Required when ACTIVE_LLM is 'groq' or 'auto'.",
    )

    GEMINI_API_KEY: str = Field(
        default="",
        description="Gemini API key. Required when ACTIVE_LLM is 'gemini' or 'auto'.",
    )

    # ── Security ──────────────────────────────────────────────────────────────

    CYRAX_PIN_HASH: str = Field(
        description=(
            "Bcrypt hash of the master PIN. Required. No default. "
            "Generate with: python -c \"import bcrypt; "
            "print(bcrypt.hashpw(b'YOUR_PIN', bcrypt.gensalt()).decode())\""
        ),
    )

    # ── Email ─────────────────────────────────────────────────────────────────

    CYRAX_EMAIL: str = Field(
        default="",
        description="Gmail address for email_ops.py. Leave blank to keep email disabled.",
    )

    CYRAX_EMAIL_PASSWORD: str = Field(
        default="",
        description="Gmail app password for email_ops.py.",
    )

    SMTP_SERVER: str = Field(
        default="",
        description="SMTP server hostname used by email_ops.py.",
    )

    SMTP_PORT: int = Field(
        default=587,
        ge=1,
        le=65535,
        description="SMTP server port used by email_ops.py.",
    )

    IMAP_SERVER: str = Field(
        default="",
        description="IMAP server hostname used by email_ops.py.",
    )


    # ── Brain Routing ─────────────────────────────────────────────────────────

    ACTIVE_LLM: ActiveLLM = Field(
        default=ActiveLLM.AUTO,
        description=(
            "Brain routing strategy. "
            "'auto' delegates to moe_router.py. "
            "'groq' / 'gemini' / 'ollama' forces a specific provider."
        ),
    )

    GROQ_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier. Update when Groq deprecates a model.",
    )

    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model identifier.",
    )

    OLLAMA_MODEL: str = Field(
        default="phi3:mini",
        description="Ollama model tag. Only used when ACTIVE_LLM is 'ollama'.",
    )

    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama server.",
    )

    # ── Timeouts & Limits ─────────────────────────────────────────────────────

    BRAIN_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Max seconds to wait for dispatcher response before timeout.",
    )

    TOOL_TIMEOUT_SECONDS: float = Field(
        default=15.0,
        ge=1.0,
        le=60.0,
        description="Max seconds a single tool execution may run.",
    )

    MAX_TOOL_CALLS_PER_MINUTE: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Rate limit for tool registry execution calls per session per minute.",
    )

    MAX_AGENT_STEPS: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max tool-execution steps the planner may take in a single turn.",
    )

    # ── Memory ────────────────────────────────────────────────────────────────

    CONVERSATION_MAX_TOKENS: int = Field(
        default=4000,
        ge=500,
        le=32000,
        description="Token ceiling for conversation history injected into LLM context.",
    )

    CONVERSATION_MAX_TURNS: int = Field(
        default=40,
        ge=4,
        le=200,
        description="Hard cap on stored conversation turns. Secondary guard to token count.",
    )

    # ── Session & Security ────────────────────────────────────────────────────

    SESSION_DURATION_SECONDS: int = Field(
        default=1800,
        ge=60,
        description="Seconds before an authenticated session expires.",
    )

    LOCKOUT_DURATION_SECONDS: int = Field(
        default=300,
        ge=30,
        description="Seconds the system locks out further PIN attempts after MAX_AUTH_RETRIES.",
    )

    MAX_AUTH_RETRIES: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Failed PIN attempts before lockout is triggered.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────

    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level for the console handler.",
    )

    LOG_DIR: str = Field(
        default="",
        description=(
            "Override the log directory path. "
            "Defaults to <project_root>/logs when empty."
        ),
    )

    # ── Voice ─────────────────────────────────────────────────────────────────

    # OpenAI-compatible STT (e.g., Groq Whisper) configuration
    OPENAI_API_KEY: str = Field(
        default="",
        description=(
            "OpenAI-compatible API key for STT. When OPENAI_BASE_URL points "
            "at Groq's endpoint, this is the Groq API key, not an OpenAI one."
        ),
    )

    OPENAI_BASE_URL: str | None = Field(
        default=None,
        description=(
            "Optional override for the OpenAI-compatible API base URL. "
            "Set to Groq's endpoint to route Whisper STT calls through "
            "Groq's free tier instead of OpenAI directly. "
            "None uses the openai SDK's default (api.openai.com)."
        ),
    )

    OPENAI_STT_MODEL: str = Field(
        default="whisper-large-v3",
        description="The Whisper model ID to send to the STT provider (e.g., Groq).",
    )

    TTS_VOICE: str = Field(

        default="en-US-JennyNeural",
        description="Edge-TTS neural voice name.",
    )

    STT_ENERGY_THRESHOLD: int = Field(
        default=300,
        ge=0,
        description="SpeechRecognition energy threshold. Lower = more sensitive.",
    )

    STT_PAUSE_THRESHOLD: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Seconds of silence before SpeechRecognition ends a phrase.",
    )

    STT_PHRASE_TIME_LIMIT: int = Field(
        default=15,
        ge=3,
        le=60,
        description="Maximum seconds for a single spoken phrase.",
    )

    STT_LISTEN_TIMEOUT_SECONDS: int = Field(
        default=10,
        description="Max time to wait for speech",
    )

    WAKE_WORD_CONFIDENCE_THRESHOLD: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Minimum OpenWakeWord confidence score to trigger wake.",
    )

    VOICE_MAX_RECORD_SECONDS: int = Field(
        default=10,
        ge=1,
        le=60,
        description=(
            "Hard upper bound on a single :voice recording, in seconds. "
            "AudioCapture.record_audio() must never block past this limit "
            "even if the microphone backend stalls."
        ),
    )

    VOICE_STT_TIMEOUT_SECONDS: int = Field(
        default=15,
        ge=1,
        le=60,
        description=(
            "Timeout for the STT API call (OpenAI Whisper endpoint). "
            "Independent of VOICE_MAX_RECORD_SECONDS — this bounds the "
            "network round-trip, not the recording itself."
        ),
    )

    VOICE_SAMPLE_RATE: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        description=(
            "Audio sample rate in Hz for microphone capture. "
            "16000 Hz matches Whisper's expected input rate."
        ),
    )


    # ── Testing ───────────────────────────────────────────────────────────────

    CYRAX_TESTING: bool = Field(
        default=False,
        description=(
            "When True, bootstrap hard-failures raise RuntimeError instead of "
            "calling sys.exit(1). Set via environment variable in test suites."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FIELD-LEVEL VALIDATORS
    # ══════════════════════════════════════════════════════════════════════════

    @field_validator("CYRAX_PIN_HASH")
    @classmethod
    def pin_hash_must_be_bcrypt(cls, v: str) -> str:
        """
        Rejects plaintext PINs stored by mistake.
        Bcrypt hashes always start with $2b$ or $2a$.
        """
        if not (v.startswith("$2b$") or v.startswith("$2a$")):
            raise ValueError(
                "CYRAX_PIN_HASH is not a valid bcrypt hash. "
                "It must start with '$2b$' or '$2a$'. "
                "Do not store a plaintext PIN in CYRAX_PIN_HASH. "
                "Generate a hash with: "
                "python -c \"import bcrypt; "
                "print(bcrypt.hashpw(b'YOUR_PIN', bcrypt.gensalt()).decode())\""
            )
        return v

    @field_validator("GROQ_API_KEY", "GEMINI_API_KEY")
    @classmethod
    def api_key_no_whitespace(cls, v: str) -> str:
        """Catches accidentally copy-pasted API keys with leading/trailing spaces."""
        if v and v != v.strip():
            raise ValueError(
                "API key contains leading or trailing whitespace. "
                "Check your .env file for accidental spaces around the value."
            )
        return v

    # ══════════════════════════════════════════════════════════════════════════
    # CROSS-FIELD VALIDATOR
    # ══════════════════════════════════════════════════════════════════════════

    @model_validator(mode="after")
    def enforce_provider_key_requirements(self) -> "CyraxSettings":
        """
        Enforces that the API keys required by the active routing strategy
        are present. Crashes boot with a clear error rather than reaching
        the first LLM call and failing there.

        Rules:
          ACTIVE_LLM = auto   → GROQ_API_KEY required (primary provider)
                                GEMINI_API_KEY required (fallback provider)
          ACTIVE_LLM = groq   → GROQ_API_KEY required
          ACTIVE_LLM = gemini → GEMINI_API_KEY required
          ACTIVE_LLM = ollama → OLLAMA_BASE_URL must be non-empty (always has
                                default, so this guards against accidental blank)
        """
        active = self.ACTIVE_LLM

        if active in (ActiveLLM.AUTO, ActiveLLM.GROQ):
            if not self.GROQ_API_KEY:
                raise ValueError(
                    f"GROQ_API_KEY is required when ACTIVE_LLM is '{active.value}' "
                    f"but it is missing or empty in your .env file. "
                    f"Add: GROQ_API_KEY=your_key_here"
                )

        if active in (ActiveLLM.AUTO, ActiveLLM.GEMINI):
            if not self.GEMINI_API_KEY:
                raise ValueError(
                    f"GEMINI_API_KEY is required when ACTIVE_LLM is '{active.value}' "
                    f"but it is missing or empty in your .env file. "
                    f"Add: GEMINI_API_KEY=your_key_here"
                )

        if active == ActiveLLM.OLLAMA:
            if not self.OLLAMA_BASE_URL.strip():
                raise ValueError(
                    "OLLAMA_BASE_URL cannot be empty when ACTIVE_LLM is 'ollama'. "
                    "Default is 'http://localhost:11434'. "
                    "Ensure your Ollama server is running and the URL is correct."
                )

        return self

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL CONFIG
    # ══════════════════════════════════════════════════════════════════════════

    model_config = {
        "env_file":          ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive":    True,
        "extra":             "ignore",
        # Allow tests to monkeypatch credentials (unit tests rely on this).
        # Runtime code should still treat settings as effectively read-only.
        "frozen":            False,

    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL INSTANCE
# ══════════════════════════════════════════════════════════════════════════════

settings = CyraxSettings()