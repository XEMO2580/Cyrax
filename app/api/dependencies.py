"""FastAPI dependencies for mobile API authentication and session resolution.

This module exposes two FastAPI dependency callables used by the mobile
API routes:

- get_current_device_id: extracts the authenticated device_id from the
  Bearer token in the Authorization header.
- get_session_context: resolves the session-scoped CyraxContext for the
  device/session pair encoded in the token (resumes evicted sessions
  transparently via the process-wide SessionManager).

Both functions delegate JWT verification to the project's security.auth_session
module so behavior is consistent with other places that create/verify
mobile tokens (websocket, auth endpoints, etc.).
"""

from fastapi import Header, HTTPException
from app.sessions.session_manager import get_session_manager
from security import auth_session
from core.context import CyraxContext


async def get_current_device_id(authorization: str = Header(...)) -> str:
    token = authorization.removeprefix("Bearer ").strip()
    try:
        device_id, _ = auth_session.get_device_and_session(token)
    except auth_session.TokenExpiredError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    except auth_session.TokenInvalidError:
        raise HTTPException(status_code=401, detail="Invalid token.")
    return device_id


async def get_session_context(authorization: str = Header(...)) -> CyraxContext:
    """FastAPI dependency that returns the session-scoped CyraxContext.

    Verifies the Bearer token, extracts (device_id, session_id), and asks
    the process-wide SessionManager to resume or create the corresponding
    MobileSession. On failure this raises an HTTPException with an
    appropriate status code.
    """
    token = authorization.removeprefix("Bearer ").strip()
    try:
        device_id, session_id = auth_session.get_device_and_session(token)
    except auth_session.TokenExpiredError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    except auth_session.TokenInvalidError:
        raise HTTPException(status_code=401, detail="Invalid token.")

    try:
        manager = get_session_manager()
        session = await manager.resume_session(device_id, session_id)
        return session.ctx
    except Exception:
        # Hide internal errors from clients; they get a generic 500 instead.
        raise HTTPException(status_code=500, detail="Session resolution failed.")


async def get_current_context(authorization: str = Header(...)) -> CyraxContext:
    """Backward-compatible alias for get_session_context used by older modules."""
    return await get_session_context(authorization)
