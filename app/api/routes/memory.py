"""
app/api/routes/memory.py — CYRAX 3.0 Memory Profile Endpoints (Phase 9.5D)

Implements the memory profile endpoints frozen in `docs/api/rest.md` §3
and `docs/api/schemas.md` §8:

  GET /api/v1/memory/profile     -> 200 ProfileGetResponse
  POST /api/v1/memory/profile    -> 200 {"status": "success"}

Behavior:
  GET  -> ctx.memory.profile.get_all()
  POST -> ctx.memory.profile.set(key, value)

Failure:
  400 PROFILE_LIMIT_REACHED — mirrors tools/memory_ops.py's existing
  _MAX_PROFILE_ENTRIES behavior.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_context
from app.api.schemas import ProfileGetResponse, ProfileSetRequest
from core.context import CyraxContext

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memory"])


@router.get(
    "/memory/profile",
    response_model=ProfileGetResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve the caller's stored long-term profile memory.",
)
async def get_profile(
    ctx: CyraxContext = Depends(get_current_context),
) -> ProfileGetResponse:
    """
    GET /api/v1/memory/profile.

    Returns the caller's stored long-term profile memory as a dict.
    """
    entries = await ctx.memory.profile.get_all()
    return ProfileGetResponse(entries=entries)


@router.post(
    "/memory/profile",
    response_model=None,
    status_code=status.HTTP_200_OK,
    summary="Write a key/value into long-term profile memory.",
    responses={400: {"description": "Profile entry limit reached."}},
)
async def set_profile(
    payload: ProfileSetRequest,
    ctx: CyraxContext = Depends(get_current_context),
) -> dict[str, str]:
    """
    POST /api/v1/memory/profile.

    Writes a key into the user's profile memory. Returns 200 {"status":
    "success"} on success, or 400 PROFILE_LIMIT_REACHED if the store's
    entry limit is hit.
    """
    try:
        await ctx.memory.profile.set(payload.key, payload.value)
    except Exception as exc:
        # Profile stores raise a limit-exceeded error when _MAX_PROFILE_ENTRIES
        # is reached. We surface that as 400 PROFILE_LIMIT_REACHED.
        if "limit" in str(exc).lower() or "max" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "PROFILE_LIMIT_REACHED",
                    "message": str(exc),
                    "trace_id": None,
                },
            ) from exc
        raise

    return {"status": "success"}
