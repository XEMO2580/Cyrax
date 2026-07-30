"""
voice/state.py — Phase 5.0 Push-to-Talk voice state machine.

Five states only. No SPEAKING, no INTERRUPTED — this phase has no TTS
and no barge-in, per explicit scope exclusion. Do not add states not
listed here without a new directive.
"""

from __future__ import annotations

from enum import Enum, auto


class VoiceState(Enum):
    """
    IDLE          — Ready to accept a new :voice command. Default state.
    RECORDING     — Actively capturing microphone audio (bounded duration).
    TRANSCRIBING  — Audio captured; STT API call in progress.
    DISPATCHING   — Transcription complete; ctx.dispatcher.handle() in progress.
    ERROR         — An unhandled/unexpected failure occurred. Does not
                    auto-recover. Distinct from known failure modes (empty
                    audio, empty transcript, STT timeout, dispatcher error
                    status) — those return the system to IDLE directly,
                    per Constraint 3. ERROR is reserved for the case none
                    of those specific handlers caught.
    """
    IDLE         = auto()
    RECORDING    = auto()
    TRANSCRIBING = auto()
    DISPATCHING  = auto()
    ERROR        = auto()


class VoiceStateError(RuntimeError):
    """
    Raised when a :voice command is issued while VoiceManager is not IDLE.
    Prevents re-entrant triggering (Constraint / mandated test:
    test_voice_rejects_reentry_when_not_idle).
    """

    def __init__(self, current_state: VoiceState) -> None:
        self.current_state = current_state
        super().__init__(
            f"Cannot start a new voice command while in state "
            f"'{current_state.name}'. Wait for the current command to finish."
        )