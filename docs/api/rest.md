# docs/api/rest.md — CYRAX 3.0 Distributed REST API Contract

**Status:** FROZEN (Phase 9.5A)
**Base path:** `/api/v1`
**Content-Type:** `application/json` for all request/response bodies.

---

## 1. Endpoint Summary

| Method | Path | Auth Required | Description |
|---|---|---|---|
| `POST` | `/auth/login` | No | Authenticate with device_id + PIN, receive JWT. |
| `POST` | `/chat` | Yes | Send a message through the same Decision Engine → Planner/chat pipeline the CLI uses. Synchronous — blocks until `immediate`/`interactive` execution completes, or returns a queued acknowledgement for `background` mode. |
| `POST` | `/tasks` | Yes | Explicitly submit a background task (bypasses conversational classification — direct queue submission). |
| `GET` | `/tasks/{task_id}` | Yes | Poll the current status/result of a specific task. |
| `GET` | `/jobs` | Yes | List the caller's tasks, most recent first. Supports pagination. |
| `POST` | `/tasks/{task_id}/cancel` | Yes | Cancel a specific running/pending task. Routes to `ctx.interrupt_controller.cancel()`. |
| `POST` | `/tasks/cancel_all` | Yes | Global stop — routes to `ctx.interrupt_controller.cancel_all()`. Equivalent to the CLI's "stop"/"cancel"/"abort" interceptor. |
| `GET` | `/memory/profile` | Yes | Retrieve the caller's stored long-term profile memory (`ctx.memory.profile.get_all()`). |
| `POST` | `/memory/profile` | Yes | Write a key/value into long-term profile memory. |
| `GET` | `/health` | No | Liveness/readiness probe. No auth — used by infra, not the mobile client's normal flow. |

---

## 2. Session Resolution (applies to every authenticated route)

Every authenticated request resolves to a `CyraxContext` via this order:

```
1. Extract session_id from the validated JWT's `sub` claim.
2. Look up an existing CyraxContext in the session pool by session_id.
3. If found and not evicted -> use it directly.
4. If not found (evicted, or first request after login) ->
   construct a fresh CyraxContext for this session_id, add to pool.
5. Proceed with the route handler using the resolved CyraxContext.
```

This is a Phase 9.5B (Session Architecture) implementation concern —
documented here only so route handlers' behavior is unambiguous: **no
route handler ever receives a shared, cross-session `CyraxContext`.**

---

## 3. Endpoint Details

### `POST /auth/login`
- **Auth:** None
- **Body:** `LoginRequest` (see `schemas.md §1`)
- **Response:** `200 OK` → `LoginResponse` (JWT + expiry)
- **Failure:** `401 UNAUTHORIZED` on bad PIN, `423 LOCKED` if `SecurityGuard`'s lockout is currently active (mirrors `security/lockout_store.py`'s existing device-global lockout — **not** per-mobile-session, since the lockout store is device-scoped by design; a locked-out device is locked out for CLI, voice, AND mobile simultaneously).

### `POST /chat`
- **Auth:** Required
- **Body:** `ChatRequest` (see `schemas.md §2`)
- **Response:** `200 OK` → `ChatResponse`
- **Behavior:** Calls `ctx.dispatcher.handle()` exactly as `cli_main.py`'s text worker does. If `DecisionEngine` classifies the request as `execution_mode=background`, the response is the same queued-acknowledgement string the CLI shows (`"✓ Task Accepted..."`) — the client should treat that as a signal to watch `/jobs` or the WebSocket for the eventual outcome, not as a final answer.

### `POST /tasks`
- **Auth:** Required
- **Body:** `TaskSubmitRequest` (see `schemas.md §3`)
- **Response:** `202 ACCEPTED` → `TaskSubmitResponse` (`task_id`, initial `status`)
- **Behavior:** Direct `ctx.task_queue.submit()` call — bypasses `DecisionEngine.classify()` entirely. Intended for client-side "explicit background action" UI (e.g. a dedicated "Run in background" button), not the default chat flow.

### `GET /tasks/{task_id}`
- **Auth:** Required
- **Response:** `200 OK` → `TaskStatusResponse`, or `404 NOT_FOUND` if `task_id` doesn't belong to this session or doesn't exist.

### `GET /jobs`
- **Auth:** Required
- **Query params:** `limit` (int, default 20, max 100), `offset` (int, default 0), `status` (optional filter, one of `TaskStatus` values)
- **Response:** `200 OK` → `JobListResponse` (array of `TaskStatusResponse` + `total_count`)

### `POST /tasks/{task_id}/cancel`
- **Auth:** Required
- **Response:** `200 OK` → `{"cancelled": true}` or `{"cancelled": false, "reason": "..."}` (e.g. task is `interruptible=False`, or already terminal)
- **Failure:** `404 NOT_FOUND` if unknown task_id.

### `POST /tasks/cancel_all`
- **Auth:** Required
- **Response:** `200 OK` → `{"cancelled_task_ids": ["...", "..."], "count": N}`

### `GET /memory/profile`
- **Auth:** Required
- **Response:** `200 OK` → `{"entries": {"key": "value", ...}}`

### `POST /memory/profile`
- **Auth:** Required
- **Body:** `{"key": "string", "value": "string"}`
- **Response:** `200 OK` → `{"status": "success"}`
- **Failure:** `400 BAD_REQUEST` if the profile store's entry limit is reached (mirrors `tools/memory_ops.py`'s existing `_MAX_PROFILE_ENTRIES` behavior).

### `GET /health`
- **Auth:** None
- **Response:** `200 OK` → `{"status": "ok", "version": "3.0"}`
