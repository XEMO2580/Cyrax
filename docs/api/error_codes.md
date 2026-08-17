# docs/api/error_codes.md — CYRAX 3.0 API Error Code Contract

**Status:** FROZEN (Phase 9.5A)

All non-2xx REST responses use the `ErrorResponse` shape from
`schemas.md §9`. WebSocket `error` events (`websocket.md §3`) use the
same `error_code` values in their `code` field.

---

## 1. HTTP Status Code Usage

| Status | Meaning in CYRAX context |
|---|---|
| `200 OK` | Successful GET/POST with a synchronous result. |
| `202 ACCEPTED` | Task successfully queued (`POST /tasks`, or `/chat` when routed to background mode). |
| `400 BAD_REQUEST` | Malformed request body, or a request rejected by an internal validation rule (e.g. profile entry limit reached). |
| `401 UNAUTHORIZED` | Missing/invalid/expired JWT, or failed login PIN. |
| `403 FORBIDDEN` | Authenticated, but the action is blocked by `SecurityLevel` gating (e.g. attempting an ADMIN-tier action without... — see note below, mobile has no PIN-reprompt flow in this phase) or a sandbox violation (`tools/file_ops.py`-style path traversal rejection surfaced through the API). |
| `404 NOT_FOUND` | Unknown `task_id`, or a `task_id` that exists but belongs to a different session. |
| `423 LOCKED` | Device-global `SecurityGuard` lockout is currently active. |
| `429 TOO_MANY_REQUESTS` | Rate limit exceeded — either `ToolRegistry`'s own per-session rate limit surfacing through `/chat`, or a future API-gateway-level rate limit. |
| `500 INTERNAL_SERVER_ERROR` | Unhandled backend exception. Should be rare given `Dispatcher`/`Planner`/`ToolRegistry`'s existing never-raise contracts — a 500 here indicates those contracts were violated somewhere, which is itself a bug worth flagging, not just a user-facing error. |
| `503 SERVICE_UNAVAILABLE` | All LLM providers exhausted (mirrors `MoERouter`'s "all providers exhausted" internal string response) — surfaced as an HTTP-level failure for `/chat` rather than a `200` with an apologetic message body, since the mobile client's error-handling UI should treat this differently from a normal assistant reply. |

---

## 2. Custom `error_code` Values

| `error_code` | HTTP Status | Meaning |
|---|---|---|
| `AUTH_INVALID_CREDENTIALS` | 401 | Login PIN incorrect. |
| `AUTH_TOKEN_EXPIRED` | 401 | JWT `exp` has passed. |
| `AUTH_TOKEN_INVALID` | 401 | JWT signature/structure invalid. |
| `AUTH_DEVICE_LOCKED` | 423 | `SecurityGuard` device-global lockout active. Response body includes `retry_after_seconds` as an additional field beyond the base `ErrorResponse` shape. |
| `PERMISSION_DENIED` | 403 | `SecurityLevel` gate blocked the action. **Known gap, documented not resolved:** the CLI's PIN-reprompt flow (`pending_auth` state) has no mobile equivalent in this phase — an ADMIN-tier action requested via mobile fails with this code rather than prompting for a PIN inline. Mobile PIN-reprompt is deferred to a later phase. |
| `SANDBOX_VIOLATION` | 403 | A file-operation path traversal or similar sandbox boundary was rejected — mirrors `tools/file_ops.py`'s `PermissionError` path. |
| `TASK_NOT_FOUND` | 404 | Unknown or foreign `task_id`. |
| `RATE_LIMIT_EXCEEDED` | 429 | `ToolRegistry.MAX_TOOL_CALLS_PER_MINUTE` or equivalent exceeded. |
| `PROFILE_LIMIT_REACHED` | 400 | `_MAX_PROFILE_ENTRIES` reached on `POST /memory/profile`. |
| `PROVIDERS_EXHAUSTED` | 503 | `MoERouter` failover chain fully exhausted for this request. |
| `INTERNAL_ERROR` | 500 | Unhandled exception; see HTTP table note above. |

---

## 3. WebSocket Close Codes

Standard WebSocket close codes (1000-1015) are used for normal
protocol-level closure. CYRAX-specific closes use the private-use range:

| Close Code | Meaning |
|---|---|
| `4401` | JWT invalid/expired at handshake time — mirrors `AUTH_TOKEN_INVALID`/`AUTH_TOKEN_EXPIRED`. |
| `4423` | Device-global lockout active at handshake time — mirrors `AUTH_DEVICE_LOCKED`. |
| `4500` | Internal server error during connection setup (e.g. `CyraxContext` construction failed for this session). |

Non-fatal errors during an established connection use the `error` event
type (`websocket.md §3`), not a close code — the connection stays open.
