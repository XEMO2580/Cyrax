"""
app/api/routes/auth.py — CYRAX 3.0 POST /api/v1/auth/login (Phase 9.5C)

Implements the login endpoint frozen in `docs/api/rest.md` §3 and
`docs/api/schemas.md` §1:

  POST /api/v1/auth/login
    Auth: None
    Body: LoginRequest {device_id, pin}
    Response: 200 OK -> LoginResponse {token, session_id, expires_at, expires_in_seconds}
    Failure:
      401 AUTH_INVALID_CREDENTIALS — bad PIN.
      423 AUTH_DEVICE_LOCKED — device-global SecurityGuard lockout active
          (includes retry_after_seconds beyond the base ErrorResponse shape).

Flow:
  1. Check device-global lockout first (SecurityGuard.is_locked_out()).
  2. Verify the PIN via SecurityGuard.authenticate() (charges the persistent
     lockout store on failure).
  3. On success, create a fresh mobile session via SessionManager.
  4. Mint a JWT strictly bound to (device_id, session_id).
  5. Return LoginResponse.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.api.schemas import LoginRequest, LoginResponse
from app.sessions.session_manager import get_session_manager
from config.settings import settings
from security import auth_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


def _error_response(
    status_code: int,
    error_code: str,
    message: str,
    headers: dict[str, str] | None = None,
    extra: dict | None = None,
) -> JSONResponse:
    """
    Builds a JSONResponse matching the frozen ErrorResponse shape
    (`docs/api/schemas.md` §9) at the TOP level of the body — not wrapped in
    FastAPI's `{"detail": ...}` layer. Optional extra fields (e.g.
    retry_after_seconds for AUTH_DEVICE_LOCKED) are merged in.
    """
    body: dict = {
        "error_code": error_code,
        "message": message,
        "trace_id": None,
    }
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status_code, content=body, headers=headers)


@router.post(
    "/auth/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate a mobile device and receive a JWT.",
    description=(
        "Verify device_id + PIN, create a fresh mobile session, and return a "
        "JWT strictly bound to the device_id:session_id pair."
    ),
)
async def login(payload: LoginRequest) -> LoginResponse:
    """
    POST /api/v1/auth/login.

    Accepts a device_id and a PIN. Verifies the PIN against the shared
    SecurityGuard (device-global lockout applies), then creates a new
    mobile session and mints a JWT.
    """
    # Resolve the shared security guard from the base boot context.
    session_manager = get_session_manager()
    security_guard = session_manager.get_security_guard()

    # ── 1. Device-global lockout check (423 LOCKED) ──────────────────────────
    locked, remaining = security_guard.is_locked_out()
    if locked:
        retry_after = int(remaining)
        logger.warning(
            f"[API_LOGIN] Rejected — device locked. "
            f"retry_after_seconds={retry_after}"
        )
        return _error_response(
            status_code=status.HTTP_423_LOCKED,
            error_code="AUTH_DEVICE_LOCKED",
            message=(
                "Device is locked out due to too many failed PIN attempts. "
                f"Retry in {retry_after} seconds."
            ),
            extra={"retry_after_seconds": retry_after},
        )

    # ── 2. Verify PIN via SecurityGuard (401 AUTH_INVALID_CREDENTIALS) ───────
    authenticated = security_guard.authenticate(payload.pin)
    if not authenticated:
        # Re-check lockout: a failed attempt may have just triggered one.
        locked_now, remaining_now = security_guard.is_locked_out()
        if locked_now:
            retry_after = int(remaining_now)
            return _error_response(
                status_code=status.HTTP_423_LOCKED,
                error_code="AUTH_DEVICE_LOCKED",
                message=(
                    "Too many failed PIN attempts. Device locked. "
                    f"Retry in {retry_after} seconds."
                ),
                extra={"retry_after_seconds": retry_after},
            )

        return _error_response(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTH_INVALID_CREDENTIALS",
            message="Incorrect PIN.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── 3. Create a fresh mobile session ─────────────────────────────────────
    session = await session_manager.create_session(payload.device_id)

    # ── 4. Mint a JWT bound to (device_id, session_id) ───────────────────────
    expires_in_seconds = settings.JWT_EXPIRY_MINUTES * 60
    token = auth_session.create_token(
        device_id=payload.device_id,
        session_id=session.session_id,
        expires_in_seconds=expires_in_seconds,
    )

    expires_at = datetime.now(timezone.utc).timestamp() + expires_in_seconds

    logger.info(
        f"[API_LOGIN] Success | device_id={payload.device_id} | "
        f"session_id={session.session_id}"
    )

    return LoginResponse(
        token=token,
        session_id=session.session_id,
        expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        expires_in_seconds=expires_in_seconds,
    )
