# docs/api/versioning.md — CYRAX 3.0 API Versioning Contract

**Status:** FROZEN (Phase 9.5A)
**Applies to:** All REST endpoints and the WebSocket connection at `/api/v1/*`.

---

## 1. Base Path

Every CYRAX distributed API endpoint is versioned through a URL path prefix:

| Surface | Base Path |
|---|---|
| REST endpoints | `/api/v1` |
| WebSocket connection | `/api/v1/events` |

There is no unversioned surface. The current and only version is **v1**.

---

## 2. Non-Breaking Changes (allowed within v1)

The following are considered **non-breaking** and may be added at any time
within the v1 lifecycle without a version bump. A conforming client MUST
tolerate them:

- **Adding new fields** to an existing request/response schema (e.g. a new
  optional member of `ChatResponse`). Existing fields are never removed or
  retyped.
- **Adding new optional query parameters** to an existing endpoint.
- **Adding new endpoints** to a new path (a new resource under `/api/v1`).
- **Adding new `error_code` values** or new HTTP status dictionary entries.
- **Adding new WebSocket event `type` values** — clients must ignore unknown
  event types rather than failing the connection.
- **Adding new enum members** to an existing enum (e.g. a new `TaskStatus`
  value) as long as existing members are unchanged.

**Client rule:** a conforming client must treat any unknown field, unknown
enum member, unknown event type, or unknown `error_code` as forward-compatible
and ignore it gracefully.

---

## 3. Breaking Changes (require a `/v2` migration)

The following are considered **breaking** and MUST NOT be introduced within
v1. They require a new major version (`/api/v2`) and a coordinated migration:

- **Removing, renaming, or retyping** an existing field.
- **Changing the meaning** of an existing field.
- **Removing or renaming** an existing endpoint.
- **Changing the HTTP status code** or response shape of an existing endpoint.
- **Changing the JWT payload's required claims** (adding a required claim is
  not breaking; making an existing optional claim required, or renaming a
  claim, is).
- **Changing the WebSocket envelope format** or renaming an existing event
  `type`.
- **Changing the token transport mechanism** (e.g. moving the WebSocket token
  from the `?token=` query parameter to a header).

When a breaking change is required, `/api/v2` is introduced while `/api/v1`
remains available for a deprecation window sufficient for existing clients to
migrate.

---

## 4. JWT Binding

JWT tokens are **strictly tied** to a `device_id` + `session_id` pair:

- The `sub` claim is the combined subject `"{device_id}:{session_id}"`.
- The `device_id` and `session_id` claims are carried explicitly in the
  payload (see `docs/api/authentication.md` §2).
- A token is only valid for the exact `device_id`/`session_id` pair it was
  issued to. A token minted for one session can never be used to access
  another session's `CyraxContext`.

This binding is fixed at the v1 contract level and cannot change without a
major version migration.

---

## 5. Backward & Forward Compatibility Matrix

| Change Type | Introduced in v1? | Requires v2? |
|---|---|---|
| Add a field to an existing schema | ✅ | ❌ |
| Add an endpoint | ✅ | ❌ |
| Add an `error_code` | ✅ | ❌ |
| Add a WebSocket event `type` | ✅ | ❌ |
| Add an enum member | ✅ | ❌ |
| Remove/rename/retype a field | ❌ | ✅ |
| Remove/rename an endpoint | ❌ | ✅ |
| Change a status code / response shape | ❌ | ✅ |
| Change a required JWT claim | ❌ | ✅ |
| Change the WebSocket envelope | ❌ | ✅ |
| Change the token transport | ❌ | ✅ |
