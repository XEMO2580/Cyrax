# docs/api/authentication.md — CYRAX 3.0 Distributed API Authentication Contract

**Status:** FROZEN (Phase 9.5A)
**Applies to:** REST endpoints and WebSocket connection at `/api/v1/*`

---

## 1. Overview

Distributed CYRAX introduces `MobileSession` as a new concept — it does **not**
reuse the local CLI's implicit single-session model (`device_master_001`).
Each authenticated mobile client is issued a JWT scoped to a specific
`device_id` + `session_id` pair, which the API layer resolves into a
per-request `CyraxContext` (see `docs/api/rest.md` §2 for the session
resolution contract).

Authentication is JWT-based. There is no separate refresh-token endpoint
in this phase — token lifetime and reissuance are handled via the login
endpoint only (see §4). Refresh tokens are explicitly deferred to a later
phase; noted here so the client does not assume their existence.

---

## 2. JWT Payload Structure

```json
{
  "sub": "device_id:session_id",
  "device_id": "string, client-generated UUID, stable across app reinstalls if possible",
  "session_id": "string, server-generated UUID, new per login",
  "iat": 1735689600,
  "exp": 1735693200,
  "scope": "mobile_client"
}
```

| Field        | Type   | Required | Notes |
|---|---|---|---|
| `sub`        | string | yes | `"{device_id}:{session_id}"` — combined subject claim, used as the CyraxContext pool lookup key. |
| `device_id`  | string | yes | Identifies the physical device. Stable identity across sessions from the same device. |
| `session_id` | string | yes | Identifies this specific login session. A new `session_id` is issued on every successful login — logging in again from the same device does not reuse a prior `session_id`. |
| `iat`        | int (unix ts) | yes | Issued-at time. |
| `exp`        | int (unix ts) | yes | Expiry. Default lifetime: **1 hour** (`3600` seconds). |
| `scope`      | string | yes | Fixed value `"mobile_client"` in this phase. Reserved for future scopes (e.g. `"admin_client"`). |

**Design note:** `device_id` and `session_id` are both carried in the JWT
so the server-side session pool (Phase 9.5B) can key `CyraxContext`
instances by `session_id` while still being able to enforce
per-`device_id` policies (e.g. max concurrent sessions per device) without
a second lookup.

---

## 3. Token Transport

| Transport | Mechanism |
|---|---|
| REST requests | `Authorization: Bearer <jwt>` header. Required on every `/api/v1/*` route except `POST /api/v1/auth/login`. |
| WebSocket connection | Query parameter: `wss://.../api/v1/events?token=<jwt>`. **Not** a header — WebSocket upgrade requests from mobile HTTP clients cannot reliably attach custom headers pre-handshake on all platforms, so the query-param form is the frozen contract. The server validates the token during the WebSocket handshake, before upgrading the connection, and closes with code `4401` (see `error_codes.md` §3) if invalid. |

**Security note (documented, not resolved here):** query-param tokens can
appear in server access logs. This is an accepted tradeoff for this phase
given mobile WebSocket header limitations; log redaction of the `token`
query parameter is a backend implementation requirement for Phase 9.5E,
not an API contract concern.

---

## 4. Lifecycle

```
1. Client -> POST /api/v1/auth/login {device_id, credentials}
2. Server validates credentials, generates session_id, issues JWT (exp = now + 1h)
3. Client stores JWT, attaches as Bearer token to all REST calls
4. Client opens WebSocket with ?token=<jwt>
5. On JWT expiry (401 response OR WebSocket close 4401):
     Client MUST re-authenticate via POST /api/v1/auth/login again.
     There is no silent refresh in this phase.
6. Server-side session (CyraxContext instance) is evicted from the pool
   after IDLE_SESSION_TIMEOUT (backend config, default 30 min of no
   REST/WebSocket activity) — independent of JWT expiry. A valid,
   unexpired JWT against an evicted session results in a fresh
   CyraxContext being constructed transparently (same session_id,
   new in-memory state) — this is a session **resume**, not a re-login.
```

---

## 5. Credentials (Login Payload)

`POST /api/v1/auth/login` request body — see `docs/api/schemas.md §1` for
the full schema. Distributed CYRAX reuses the existing `SecurityGuard`
PIN-based authentication (`security/auth.py`) as the credential check —
there is no separate mobile-specific password system in this phase.

```json
{
  "device_id": "a1b2c3d4-...",
  "pin": "147147"
}
```

Failure returns `401 UNAUTHORIZED` per `error_codes.md §2`.
