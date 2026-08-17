"""
app/api/websocket.py — CYRAX 3.0 WebSocket Event Endpoint (Phase 9.5E)

Implements the single multiplexed WebSocket endpoint frozen in
`docs/api/websocket.md`:

  ws://<host>/api/v1/events?token=<jwt>

Connection lifecycle:
  1. Client opens WebSocket with ?token=<jwt> query param.
  2. Server validates the JWT during the handshake (before upgrade).
     - Invalid/expired token -> close code 4401 immediately.
     - Valid token -> resolve to the session-scoped CyraxContext.
  3. Server subscribes an internal listener to ctx.notification_center.
  4. Server sends a `connected` event upon successful upgrade.
  5. On disconnect, the listener is unsubscribed (finally block).

Envelope format (websocket.md §2):
  {"type": str, "payload": {...}, "timestamp": ISO 8601 UTC}

Non-blocking delivery (websocket.md §4):
  Each listener's send_json() is wrapped in asyncio.wait_for(..., timeout=2.0).
  If a client's socket is stalled, the event is dropped for that client only.

Event types (websocket.md §3): connected, task_progress, task_completed,
task_failed, task_cancelled, notification, heartbeat, error.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.sessions.session_manager import get_session_manager
from security import auth_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Close codes (error_codes.md §3).
_WS_CLOSE_AUTH_INVALID = 4401
_WS_CLOSE_INTERNAL     = 4500

# Non-blocking delivery timeout (websocket.md §4).
_SEND_TIMEOUT_SECONDS = 2.0


def _iso_now() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _envelope(event_type: str, payload: dict) -> dict:
    """Builds a websocket.md §2 envelope."""
    return {
        "type": event_type,
        "payload": payload,
        "timestamp": _iso_now(),
    }


@router.websocket("/api/v1/events")
async def events_endpoint(websocket: WebSocket) -> None:
    """
    The single multiplexed event WebSocket.

    Authenticates from the ?token=<jwt> query param before accepting the
    upgrade, then streams task events to the client until disconnect.
    """
    # ── 1. Validate the JWT from the query param (pre-upgrade) ───────────────
    query = websocket.query_params
    token = query.get("token")

    if not token:
        await websocket.close(code=_WS_CLOSE_AUTH_INVALID)
        logger.warning("[WS] Rejected: missing token query param.")
        return

    try:
        payload = auth_session.decode_token(token)
    except auth_session.TokenExpiredError as exc:
        logger.warning(f"[WS] Rejected: token expired — {exc}")
        await websocket.close(code=_WS_CLOSE_AUTH_INVALID)
        return
    except auth_session.TokenInvalidError as exc:
        logger.warning(f"[WS] Rejected: token invalid — {exc}")
        await websocket.close(code=_WS_CLOSE_AUTH_INVALID)
        return

    device_id  = payload["device_id"]
    session_id = payload["session_id"]

    # ── 2. Resolve the session-scoped context ────────────────────────────────
    try:
        manager = get_session_manager()
        session = await manager.resume_session(device_id, session_id)
        session.touch()
        ctx = session.ctx
    except Exception as exc:
        logger.error(f"[WS] Session resolution failed: {exc}", exc_info=True)
        await websocket.close(code=_WS_CLOSE_INTERNAL)
        return

    # ── 3. Accept the connection ─────────────────────────────────────────────
    await websocket.accept()

    # ── 4. Define the event listener (bounded send_json) ─────────────────────
    async def listener(event) -> None:
        """NotificationCenter listener → maps TaskNotificationEvent to envelope."""
        try:
            event_type = _map_event_type(event.status)
            if event_type is None:
                logger.debug(f"[WS] Ignoring unmapped event status: {event.status}")
                return

            payload = {
                "task_id": event.task_id,
                "status": event.status,
            }
            if event_type == "task_completed":
                payload["title"] = event.title
                payload["summary"] = event.summary
                payload["result"] = event.payload or ""
            elif event_type == "task_failed":
                payload["title"] = event.title
                payload["summary"] = event.summary
                payload["error"] = event.payload or ""
            elif event_type == "task_cancelled":
                payload["reason"] = event.payload or ""

            await _send_bounded(websocket, _envelope(event_type, payload))
        except Exception as exc:
            logger.error(f"[WS] Listener error: {exc}", exc_info=True)

    # ── 5. Subscribe to the notification center ──────────────────────────────
    try:
        await ctx.notification_center.subscribe(listener)
    except Exception as exc:
        logger.error(f"[WS] subscribe failed: {exc}", exc_info=True)
        await websocket.close(code=_WS_CLOSE_INTERNAL)
        return

    # ── 6. Send the initial `connected` event ────────────────────────────────
    try:
        await _send_bounded(websocket, _envelope("connected", {"session_id": session_id}))
    except Exception:
        pass

    logger.info(f"[WS] Connected | device_id={device_id} | session_id={session_id}")

    try:
        # ── 7. Keep the connection open + send periodic heartbeats ───────────
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Idle for 30s — send a heartbeat (websocket.md §3).
                try:
                    await _send_bounded(websocket, _envelope("heartbeat", {}))
                except Exception:
                    break
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        logger.info(f"[WS] Disconnected | session_id={session_id}")
    finally:
        # ── 8. Unsubscribe the listener (websocket.md §1 step 5) ─────────────
        try:
            await ctx.notification_center.unsubscribe(listener)
        except Exception as exc:
            logger.error(f"[WS] unsubscribe failed: {exc}", exc_info=True)
        logger.info(f"[WS] Listener unsubscribed | session_id={session_id}")


def _map_event_type(status: str) -> str | None:
    """
    Maps a TaskNotificationEvent.status string to a websocket.md §3 event type.

    status values come from orchestrator/executor.py via NotificationCenter.
    """
    normalized = status.lower()
    if "complete" in normalized:
        return "task_completed"
    if "fail" in normalized:
        return "task_failed"
    if "cancel" in normalized:
        return "task_cancelled"
    if "running" in normalized:
        return "task_progress"
    return None


async def _send_bounded(websocket: WebSocket, message: dict) -> None:
    """
    Sends a JSON message with a bounded timeout (websocket.md §4).

    If the client's socket is stalled and does not accept a write within
    _SEND_TIMEOUT_SECONDS, the event is dropped for that client only.
    """
    await asyncio.wait_for(websocket.send_json(message), timeout=_SEND_TIMEOUT_SECONDS)
