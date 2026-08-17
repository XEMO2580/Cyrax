# docs/api/websocket.md — CYRAX 3.0 WebSocket Event Contract

**Status:** FROZEN (Phase 9.5A)
**Endpoint:** `wss://<host>/api/v1/events?token=<jwt>`
**Model:** Single multiplexed connection per session — all event types flow over one socket, distinguished by the `type` field. Client does not open separate sockets per event category.

---

## 1. Connection Lifecycle

```
1. Client opens WebSocket with ?token=<jwt> query param.
2. Server validates JWT during handshake (before upgrade).
   - Invalid/expired token -> connection closed immediately, close code 4401.
   - Valid token -> connection upgraded, resolved to the same
     session-scoped CyraxContext used by REST requests for this session_id.
3. Server subscribes an internal listener to ctx.notification_center for
   the duration of the connection.
4. Server sends a `connected` event immediately upon successful upgrade
   (see §3) — the client can use this as the "socket is ready" signal
   rather than relying on the WebSocket open event alone.
5. On disconnect (client or server initiated), the server unsubscribes
   the listener from ctx.notification_center. No events are missed
   between disconnect and a subsequent reconnect being buffered — this
   phase does NOT implement event replay/buffering. A client reconnecting
   after a gap should call GET /jobs to reconcile state, not assume the
   WebSocket delivered everything that happened while disconnected.
```

---

## 2. Envelope Format

Every message sent from server to client uses this envelope:

```json
{
  "type": "string, one of the Event Types in §3",
  "payload": { "...": "type-specific fields, see §3" },
  "timestamp": "ISO 8601 UTC string"
}
```

There is no client-to-server message format in this phase — the
WebSocket is **server-to-client only**. Client actions (cancel a task,
send a chat message) go through REST endpoints, never over the socket.
This keeps the contract asymmetric and simple; bidirectional socket
messaging is explicitly out of scope for 9.5.

---

## 3. Event Types

| `type` | Fires when | `payload` shape |
|---|---|---|
| `connected` | Immediately after successful WebSocket upgrade. | `{"session_id": "string"}` |
| `task_progress` | A background task transitions `PENDING` → `RUNNING`, or (future phases) emits an intermediate progress update. In this phase, only the `RUNNING` transition is emitted — fine-grained progress percentages are not yet implemented anywhere in the task pipeline, so the client should not expect a progress bar's worth of updates, just a single "it started" signal. | `{"task_id": "string", "status": "running"}` |
| `task_completed` | A background task reaches `COMPLETED`. Mirrors `TaskNotificationEvent` from `core/notification_center.py`. | `{"task_id": "string", "status": "completed", "title": "string", "summary": "string", "result": "string"}` |
| `task_failed` | A background task reaches `FAILED`. | `{"task_id": "string", "status": "failed", "title": "string", "summary": "string", "error": "string"}` |
| `task_cancelled` | A background task reaches `CANCELLED` (via user cancel or `cancel_all`). | `{"task_id": "string", "status": "cancelled", "reason": "string"}` |
| `notification` | A general system notification not tied to a specific task (reserved for future use — e.g. security alerts, lockout warnings). | `{"title": "string", "body": "string", "severity": "info" \| "warning" \| "critical"}` |
| `heartbeat` | Sent by the server every 30 seconds on an otherwise-idle connection, to let mobile clients detect a silently-dropped connection (common on cellular networks) faster than TCP-level timeout. | `{}` (empty payload) |
| `error` | A server-side error occurred that the client should surface (e.g. the notification bridge itself failed for this connection). Does NOT close the connection. | `{"code": "string, see error_codes.md", "message": "string"}` |

**Direct mapping note:** `task_completed`, `task_failed`, `task_cancelled`
map 1:1 onto `TaskNotificationEvent.status` values already produced by
`orchestrator/executor.py` — the WebSocket layer is a thin
`NotificationCenter` subscriber, not a second source of truth for task
outcomes. `title`/`summary` fields are passed through unmodified from
the existing `TaskNotificationEvent` dataclass.

---

## 4. Non-Blocking Delivery Requirement

(Implementation note, included here because it affects the contract's
reliability guarantees, not just the backend's internal structure.)

Each connected client's `NotificationCenter` listener MUST wrap its
`send_json()` call in a bounded timeout. If a client's socket is stalled
(e.g. poor cellular connectivity) and does not accept a write within
**2 seconds**, the event is dropped for that client only — other
connected clients/sessions still receive it. A client that misses events
this way should reconcile via `GET /jobs` after reconnecting, per §1 step 5.
