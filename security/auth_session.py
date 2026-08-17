"""
security/auth_session.py — CYRAX 3.0 Distributed API JWT Authentication (Phase 9.5C)

Implements standard JWT generation and decoding for the distributed mobile
API. This is the server-side token layer that backs the frozen
`docs/api/authentication.md` contract.

Scope & design:
  - stdlib-only HS256 implementation (hmac + sha256 + base64url). No PyJWT
    dependency — the codebase already favours dependency-minimal components
    for transport/security primitives, and the algorithm is fixed at HS256
    for this phase.
  - Payload contract (see `docs/api/authentication.md` §2):
        sub         = "{device_id}:{session_id}"
        device_id   = client-generated UUID, stable across reinstalls
        session_id  = server-generated UUID, new per login
        scope       = "mobile_client"
        iat         = issued-at (unix seconds)
        exp         = expiry (unix seconds)
        iss         = "cyrax-os"
        aud         = "cyrax-client"
  - Errors map 1:1 to `docs/api/error_codes.md` §2:
        TokenExpiredError  -> AUTH_TOKEN_EXPIRED (401)
        TokenInvalidError  -> AUTH_TOKEN_INVALID (401)
  - A missing JWT_SECRET is a deliberate hard-stop: the API must not issue
    unsigned/guessable tokens, and must not silently accept tokens it cannot
    verify. Both create_token() and decode_token() raise TokenInvalidError
    with a clear message when settings.JWT_SECRET is empty.

The JWT is strictly tied to a device_id + session_id pair (see
`docs/api/versioning.md` §4) — the sub claim is the combined subject and the
individual claims are carried explicitly for per-device policy enforcement.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ERROR TYPES (map to error_codes.md)
# ══════════════════════════════════════════════════════════════════════════════

class TokenExpiredError(Exception):
    """
    Raised when a JWT's `exp` claim has passed.
    Maps to error_code `AUTH_TOKEN_EXPIRED` / HTTP 401.
    """

    error_code = "AUTH_TOKEN_EXPIRED"


class TokenInvalidError(Exception):
    """
    Raised when a JWT is malformed, has an invalid signature, an unknown
    algorithm, missing mandatory claims, or the secret is not configured.
    Maps to error_code `AUTH_TOKEN_INVALID` / HTTP 401.
    """

    error_code = "AUTH_TOKEN_INVALID"


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_SCOPE_MOBILE_CLIENT = "mobile_client"
_ISSUER              = "cyrax-os"
_AUDIENCE            = "cyrax-client"

# Claims that MUST be present and non-empty for a token to be considered valid.
_REQUIRED_CLAIMS: frozenset[str] = frozenset({
    "sub", "device_id", "session_id", "iat", "exp", "scope", "iss", "aud",
})


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _b64url_encode(data: bytes) -> str:
    """Base64url-encode without padding (RFC 7515 §2)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    """Base64url-decode, re-adding padding if a caller stripped it."""
    if not data:
        raise TokenInvalidError("Empty base64url segment.")
    padding = "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(data + padding)
    except (ValueError, base64.binascii.Error) as exc:
        raise TokenInvalidError(f"Malformed base64url segment: {exc}") from exc


def _sign(header_b64: str, payload_b64: str, secret: str) -> str:
    """Computes the HMAC-SHA256 signature over the JWT signing input."""
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    digest = hmac.new(
        secret.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    return _b64url_encode(digest)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def create_token(
    device_id:     str,
    session_id:    str,
    *,
    expires_in_seconds: int | None = None,
    scope:               str       = _SCOPE_MOBILE_CLIENT,
) -> str:
    """
    Generates a signed JWT for a device_id + session_id pair.

    Args:
        device_id:  Client-generated stable device UUID.
        session_id: Server-generated session UUID (new per login).
        expires_in_seconds: Optional override for JWT lifetime. Defaults to
            settings.JWT_EXPIRY_MINUTES * 60.
        scope:      Claim scope. Frozen at "mobile_client" in this phase.

    Returns:
        The signed JWT string.

    Raises:
        TokenInvalidError: If settings.JWT_SECRET is empty (server refuses to
            issue unsigned tokens), or if device_id/session_id are blank.
    """
    secret = settings.JWT_SECRET
    if not secret:
        raise TokenInvalidError(
            "JWT_SECRET is not configured. Refusing to issue a token."
        )

    if not device_id or not session_id:
        raise TokenInvalidError("device_id and session_id are required to mint a token.")

    if expires_in_seconds is None:
        expires_in_seconds = settings.JWT_EXPIRY_MINUTES * 60

    now = int(time.time())
    payload: dict[str, Any] = {
        "sub":        f"{device_id}:{session_id}",
        "device_id":  device_id,
        "session_id": session_id,
        "scope":      scope,
        "iat":        now,
        "exp":        now + expires_in_seconds,
        "iss":        _ISSUER,
        "aud":        _AUDIENCE,
    }

    header = {"alg": settings.JWT_ALGORITHM, "typ": "JWT"}

    header_b64  = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    signature = _sign(header_b64, payload_b64, secret)
    token = f"{header_b64}.{payload_b64}.{signature}"

    logger.debug(
        f"[AUTH_SESSION] Token issued | device_id={device_id} | "
        f"session_id={session_id} | expires_in={expires_in_seconds}s"
    )
    return token


def decode_token(token: str) -> dict[str, Any]:
    """
    Verifies and decodes a JWT.

    Performs full validation in order:
      1. Structure (3 dot-separated segments).
      2. Secret configured.
      3. Algorithm header == HS256.
      4. Signature verification.
      5. Required claims present.
      6. `exp` not passed (raises TokenExpiredError).

    Args:
        token: The JWT string.

    Returns:
        The decoded payload dict (including `sub`, `device_id`, `session_id`,
        `scope`, `iat`, `exp`, `iss`, `aud`).

    Raises:
        TokenExpiredError: The token's `exp` has passed (AUTH_TOKEN_EXPIRED).
        TokenInvalidError: Any other validation failure (AUTH_TOKEN_INVALID).
    """
    secret = settings.JWT_SECRET
    if not secret:
        raise TokenInvalidError(
            "JWT_SECRET is not configured. Refusing to verify a token."
        )

    parts = token.split(".")
    if len(parts) != 3:
        raise TokenInvalidError("Token must have exactly three segments.")

    header_b64, payload_b64, signature_b64 = parts

    # 1. Header + algorithm check.
    try:
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TokenInvalidError(f"Malformed header segment: {exc}") from exc

    alg = header.get("alg")
    if alg != settings.JWT_ALGORITHM:
        raise TokenInvalidError(
            f"Unexpected algorithm '{alg}'. Expected '{settings.JWT_ALGORITHM}'."
        )

    # 2. Signature verification (constant-time HMAC compare).
    expected_sig = _sign(header_b64, payload_b64, secret)
    if not hmac.compare_digest(expected_sig, signature_b64):
        raise TokenInvalidError("Signature verification failed.")

    # 3. Payload decode.
    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TokenInvalidError(f"Malformed payload segment: {exc}") from exc

    if not isinstance(payload, dict):
        raise TokenInvalidError("Payload must be a JSON object.")

    # 4. Required claims present and non-empty.
    missing = [c for c in _REQUIRED_CLAIMS if not payload.get(c)]
    if missing:
        raise TokenInvalidError(f"Missing required claim(s): {', '.join(missing)}")

    # 5. Expiry check.
    exp = payload.get("exp")
    try:
        exp_int = int(exp)
    except (TypeError, ValueError) as exc:
        raise TokenInvalidError(f"Invalid 'exp' claim: {exp!r}") from exc

    if time.time() > exp_int:
        raise TokenExpiredError(
            f"Token expired at {exp_int} (server time {int(time.time())})."
        )

    logger.debug(
        f"[AUTH_SESSION] Token verified | sub={payload.get('sub')}"
    )
    return payload


def get_device_and_session(token: str) -> tuple[str, str]:
    """
    Convenience helper returning (device_id, session_id) from a verified token.

    Raises the same errors as decode_token() on any validation failure.
    """
    payload = decode_token(token)
    return payload["device_id"], payload["session_id"]
