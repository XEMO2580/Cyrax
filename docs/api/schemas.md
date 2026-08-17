# docs/api/schemas.md — CYRAX 3.0 Distributed API Schema Definitions

**Status:** FROZEN (Phase 9.5A)
**Format:** Pydantic-equivalent JSON Schema. Field types map directly to
Pydantic `BaseModel` field types for the eventual FastAPI implementation
(Phase 9.5D).

---

## 1. `LoginRequest` / `LoginResponse`

```json
// LoginRequest
{
  "device_id": "string, required, client-generated UUID",
  "pin": "string, required, 4-8 digits"
}
```

```json
// LoginResponse
{
  "token": "string, JWT",
  "session_id": "string",
  "expires_at": "ISO 8601 UTC string",
  "expires_in_seconds": 3600
}
```

---

## 2. `ChatRequest` / `ChatSubmitResponse` (Phase 10.3R — was ChatResponse)

**BREAKING CHANGE from the original Phase 9.5A contract:** `/chat` now
returns `202 ACCEPTED` immediately instead of `200 OK` with the final
answer. Clients must poll `GET /api/v1/tasks/{task_id}` (see rest.md §3)
to retrieve the actual response. The synchronous `ChatResponse` shape
from the original contract is retired.

```json
// ChatRequest — unchanged
{
  "message": "string, required, 1-4000 chars"
}
```

```json
// ChatSubmitResponse
{
  "task_id": "string, UUID",
  "trace_id": "string",
  "status": "queued"
}
```

Poll `GET /api/v1/tasks/{task_id}` — returns `TaskStatusResponse` (§4,
unchanged). The assistant's reply is in `TaskStatusResponse.result` once
`status == "completed"`.

**Note:** `status` values are the same three-value set already returned
by `Dispatcher.handle()` internally (`"shutdown"` is excluded — the
mobile client does not have a shutdown-phrase concept; if a user types
"shutdown" via mobile chat, it is treated as ordinary conversational
input, not a system command, since there is no local process for a
mobile session to shut down).

---

## 3. `TaskSubmitRequest` / `TaskSubmitResponse`

```json
// TaskSubmitRequest
{
  "user_input": "string, required, 1-4000 chars",
  "tools_required": "boolean, default false",
  "provider_name": "string, optional, one of: groq | gemini, defaults to backend routing decision"
}
```

```json
// TaskSubmitResponse
{
  "task_id": "string, UUID",
  "status": "pending"
}
```

---

## 4. `TaskStatusResponse`

Directly mirrors `core/task.py`'s `Task` model, minus internal-only
fields (`interruptible`, `priority` are included since they're useful
client-side signal; nothing security-sensitive is exposed).

```json
{
  "task_id": "string, UUID",
  "user_input": "string",
  "task_type": "immediate | background | cron",
  "status": "pending | scheduled | running | cancelling | completed | failed | cancelled",
  "priority": "integer, 0-4 (see Priority enum)",
  "result": "string | null",
  "error": "string | null",
  "created_at": "ISO 8601 UTC string",
  "updated_at": "ISO 8601 UTC string",
  "scheduled_at": "ISO 8601 UTC string | null"
}
```

---

## 5. `JobListResponse`

```json
{
  "jobs": [ /* array of TaskStatusResponse, see §4 */ ],
  "total_count": "integer",
  "limit": "integer",
  "offset": "integer"
}
```

---

## 6. `CancelResponse`

```json
{
  "cancelled": "boolean",
  "reason": "string | null, populated only when cancelled=false"
}
```

---

## 7. `CancelAllResponse`

```json
{
  "cancelled_task_ids": ["string", "..."],
  "count": "integer"
}
```

---

## 8. `ProfileGetResponse` / `ProfileSetRequest`

```json
// ProfileGetResponse
{
  "entries": { "key": "value", "...": "..." }
}
```

```json
// ProfileSetRequest
{
  "key": "string, required, 1-128 chars",
  "value": "string, required, 1-4000 chars"
}
```

---

## 9. `ErrorResponse` (all endpoints, on non-2xx)

```json
{
  "error_code": "string, see error_codes.md",
  "message": "string, human-readable",
  "trace_id": "string | null"
}
