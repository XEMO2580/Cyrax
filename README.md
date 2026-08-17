# CYRAX 3.0 — Distributed OS Assistant + API

CYRAX 3.0 is a local-first OS assistant (CLI + Voice) wrapped in a
**JWT-authenticated, multi-tenant distributed API** (Phases 9.5B → 9.5E).
Mobile clients authenticate per-device with a PIN, receive a session-scoped
`CyraxContext`, and drive the frozen V1.0 Core OS over REST + WebSocket.

This README walks through **every step and command** needed to run the
project: environment setup, CLI use, booting the API server, exercising the
API, and running the test suites.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Project Layout](#3-project-layout)
4. [Step 1 — Set Up the Virtual Environment](#step-1--set-up-the-virtual-environment)
5. [Step 2 — Configure `.env`](#step-2--configure-env)
6. [Step 3 — Install Dependencies](#step-3--install-dependencies)
7. [Step 4 — Run the Test Suites](#step-4--run-the-test-suites)
8. [Step 5 — Run the CLI (Core OS)](#step-5--run-the-cli-core-os)
9. [Step 6 — Boot the Distributed API](#step-6--boot-the-distributed-api)
10. [Step 7 — Exercise the API End-to-End](#step-7--exercise-the-api-end-to-end)
11. [WebSocket Events](#websocket-events)
12. [API Contract Documents](#api-contract-documents)
13. [Quick Troubleshooting](#quick-troubleshooting)

---

## 1. Architecture Overview

```
                    ┌─────────────────────────────────────────────┐
                    │          CYRAX 3.0 Distributed API          │
                    │               app/api/main.py               │
                    ├─────────────────────────────────────────────┤
                    │  REST /api/v1/*      WebSocket /events      │
                    │  (Bearer JWT)        (?token= JWT)          │
                    ├─────────────────────────────────────────────┤
                    │   SessionManager (multi-tenant)             │
                    │   ┌─────────────────────────────────────┐   │
                    │   │ MobileSession -> CyraxContext       │   │
                    │   │  REUSE: tool_registry, brain_router │   │
                    │   │         decision_engine, job_store  │   │
                    │   │  FRESH: MemoryStack + Interrupt     │   │
                    │   └─────────────────────────────────────┘   │
                    ├─────────────────────────────────────────────┤
                    │  security/auth_session.py (JWT HS256)       │
                    │  security/auth.py (SecurityGuard / PIN)     │
                    └─────────────────────────────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────────┐
                    │        Frozen V1.0 Core OS (bootstrap)      │
                    │  Dispatcher → DecisionEngine → Planner →    │
                    │  ToolRegistry → MoERouter (Groq/Gemini)     │
                    └─────────────────────────────────────────────┘
```

**Key layers (Phases):**

- **9.5A** — Frozen API contracts under `docs/api/` (auth, REST, WebSocket,
  schemas, error codes, versioning).
- **9.5B** — `app/sessions/session_manager.py`: multi-tenant `SessionManager`
  + `MobileSession`, idle/absolute session eviction.
- **9.5C** — `security/auth_session.py`: stdlib HS256 JWT create/decode,
  strictly bound to `device_id` + `session_id`.
- **9.5D** — REST routes: `/auth/login`, `/chat`, `/tasks`, `/tasks/{id}`,
  `/jobs`, `/tasks/{id}/cancel`, `/tasks/cancel_all`, `/memory/profile`,
  `/health`.
- **9.5E** — WebSocket `/api/v1/events?token=<jwt>`: real-time task/notification
  events with a 2-second non-blocking send timeout.

---

## 2. Prerequisites

- **Windows 11** (this project is developed on Windows; commands below are
  PowerShell/`cmd` compatible).
- **Python 3.11+** (tested on 3.11.7).
- Internet access for package installs and LLM provider calls
  (Groq / Gemini API keys).
- Optional: an Android device / DevTools client to test the distributed API
  from a phone.

---

## 3. Project Layout

```
CYRAX 3.0/
├── app/                 # Application layer
│   ├── bootstrap.py     # Phase 1 synchronous boot sequence
│   ├── cli_main.py      # CLI entry point
│   ├── api/             # Phase 9.5D/9.5E FastAPI layer
│   │   ├── main.py      # FastAPI app + lifespan + exception handler
│   │   ├── schemas.py   # Pydantic request/response models
│   │   ├── dependencies.py  # get_current_session / get_current_context
│   │   ├── websocket.py # WS /api/v1/events handler
│   │   └── routes/      # auth.py, chat.py, jobs.py, memory.py
│   └── sessions/        # session_manager.py (Phase 9.5B)
├── brain/               # LLM routing (MoERouter, providers)
├── config/              # settings.py, loader.py, YAML configs
├── core/                # context, dispatcher, task_queue, job_store, ...
├── docs/                # API contracts + this quick-start guide
├── memory/              # conversation, profile, state, semantic stores
├── orchestrator/        # decision_engine, planner, executor, dispatcher
├── security/            # auth.py (PIN), auth_session.py (JWT), sandboxing
├── tests/               # unit/integration/stress + smoke_test_9_5*.py
├── tools/               # file_ops, web_search, scheduler, email, ...
├── voice/               # STT/TTS subsystem
├── requirements.txt     # runtime deps (fastapi, uvicorn)
├── requirements-dev.txt # test deps (pytest, pytest-asyncio)
└── README.md            # this file
```

---

## Step 1 — Set Up the Virtual Environment

Create and activate a project-local virtual environment so dependencies don't
pollute your global Python:


Your prompt should now show `(.venv)`. To deactivate later:

```bash
deactivate
```

---

## Step 2 — Configure `.env`

CYRAX reads configuration from a `.env` file at the project root. This file
is **git-ignored** (contains secrets). Create it if it does not exist:

```bash
notepad .env
```

Minimal required variables:

```dotenv
# ── LLM Providers (required when ACTIVE_LLM is auto) ────────────
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key

# ── Security: bcrypt hash of your master PIN (REQUIRED) ─────────
CYRAX_PIN_HASH=$2b$12$...

# ── JWT (Phase 9.5C) ────────────────────────────────────────────
JWT_SECRET=generate_a_long_random_secret_here
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=60
```

Generate the bcrypt PIN hash and a JWT secret:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'147147', bcrypt.gensalt()).decode())"
python -c "import secrets; print(secrets.token_hex(32))"
```

Optional variables (see `config/settings.py` for the full list):

```dotenv
ACTIVE_LLM=auto            # auto | groq | gemini | ollama
EMAIL_* / SMTP_*           # email tools
LOG_LEVEL=INFO
SESSION_DURATION_SECONDS=1800
```

> **Note:** `CYRAX_PIN_HASH` must start with `$2b$` or `$2a$`. The PIN
> `147147` is used in all examples below — the mobile login endpoint sends
> `{device_id, pin}` and the shared `SecurityGuard` verifies it.

---

## Step 3 — Install Dependencies

With the venv activated, install runtime and test dependencies:

```bash
# Runtime (from requirements.txt)
pip install -r requirements.txt

# Test (from requirements-dev.txt)
pip install -r requirements-dev.txt
```

`requirements.txt` contains the full runtime set — the API layer
(`fastapi`, `uvicorn[standard]`), the Core OS LLM provider SDKs (`groq`,
`google-genai`) required by the boot sequence's Step 6, and the core config/
persistence libraries (`pydantic-settings`, `python-dotenv`, `bcrypt`,
`aiosqlite`). JWT signing is stdlib-only (no PyJWT).

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
groq>=1.0.0
google-genai>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
bcrypt>=4.0.0
aiosqlite>=0.19.0
```

> **Note:** If you ever see `[CYRAX BOOT FAILURE] LLM provider initialisation
> failed: No module named 'groq'` (or `google.genai`), you are missing the
> provider SDKs — re-run `pip install groq google-genai` (or `pip install -r
> requirements.txt`). These are required by `app.bootstrap()` Step 6.

---

## Step 4 — Run the Test Suites

### All tests (recommended)

```bash
python -m pytest -v
```

### Phase 9.5B/9.5C smoke tests (JWT + session manager + login)

```bash
python -m pytest tests/smoke_test_9_5.py -v
```

Expected: **8 passed** — JWT round-trip, expiry, invalid signature, missing
secret, session lifecycle, session isolation contract, FastAPI login
round-trip, and wrong-PIN rejection.

### Phase 9.5D/9.5E smoke tests (REST + WebSocket)

```bash
python -m pytest tests/smoke_test_9_5d.py -v
```

Expected: **14 passed** — chat (success + auth enforcement), task submit/
status/404, jobs listing, cancel 404, cancel_all, memory get/set, WebSocket
connected event + invalid/missing token → close code 4401.

### Run just the JWT + session tests

```bash
python -m pytest tests/smoke_test_9_5.py -k "jwt or session" -v
```

---

## Step 5 — Run the CLI (Core OS)

The local CLI drives the Core OS directly (single-session,
`device_master_001`). This is independent of the distributed API.

```bash
.venv\Scripts\activate
python -m app.cli_main --mode text
```

Or equivalently:

```bash
python app/cli_main.py --mode text
```

CLI options:

| Flag | Description |
|---|---|
| `--mode text` | Text interface (default). Type `:voice` to speak. |
| `--mode voice` | Voice-first interface (legacy). |
| `--session <id>` | Override the session id (default `device_master_001`). |

Commands while in the CLI:

| Command | Effect |
|---|---|
| `shutdown` / `exit` / `quit` | Stop CYRAX cleanly. |
| `:voice` | Trigger voice capture (text mode). |
| Any other text | Routes through the Decision Engine → Planner/chat pipeline. |

---

## Step 6 — Boot the Distributed API

### Development mode (auto-reload)

```bash
.venv\Scripts\activate
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production mode (no reload)

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

On startup the lifespan:

1. Calls `app.bootstrap.bootstrap()` → boots the Core OS.
2. Builds the process-wide `SessionManager`.
3. Starts the background idle-eviction loop.

Expected log output:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     [CYRAX] CYRAX 3.0 — Phase 1 Boot Sequence
INFO:     [API] Core OS booted.
INFO:     [API] Session eviction loop started.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Open the interactive docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Step 7 — Exercise the API End-to-End

All examples use `curl` (PowerShell-compatible quoting). Adjust the PIN
(`147147`) and device id as desired.

### 7.1 Health (no auth)

```bash
curl http://localhost:8000/api/v1/health
```

```json
{"status":"ok","version":"3.0"}
```

### 7.2 Login → get a JWT

```bash
curl -X POST http://localhost:8000/api/v1/auth/login ^
  -H "Content-Type: application/json" ^
  -d "{\"device_id\":\"device-mobile-1\",\"pin\":\"147147\"}"
```

Response:

```json
{
  "token": "<jwt>",
  "session_id": "<uuid>",
  "expires_at": "2025-01-01T00:00:00+00:00",
  "expires_in_seconds": 3600
}
```

Save the `token` value — every other route needs it.

### 7.3 Chat (requires JWT)

```bash
curl -X POST http://localhost:8000/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer <jwt>" ^
  -d "{\"message\":\"open notepad\"}"
```

### 7.4 Submit a background task

```bash
curl -X POST http://localhost:8000/api/v1/tasks ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer <jwt>" ^
  -d "{\"user_input\":\"open notepad\",\"tools_required\":true}"
```

Returns `202` with `{"task_id":"...","status":"pending"}`.

### 7.5 Poll a task status

```bash
curl http://localhost:8000/api/v1/tasks/<task_id> ^
  -H "Authorization: Bearer <jwt>"
```

### 7.6 List your jobs

```bash
curl "http://localhost:8000/api/v1/jobs?limit=20&offset=0" ^
  -H "Authorization: Bearer <jwt>"
```

You can add `&status=running` (or any `TaskStatus` value) to filter.

### 7.7 Cancel a task / cancel all

```bash
curl -X POST http://localhost:8000/api/v1/tasks/<task_id>/cancel ^
  -H "Authorization: Bearer <jwt>"

curl -X POST http://localhost:8000/api/v1/tasks/cancel_all ^
  -H "Authorization: Bearer <jwt>"
```

### 7.8 Memory profile read/write

```bash
# Read
curl http://localhost:8000/api/v1/memory/profile ^
  -H "Authorization: Bearer <jwt>"

# Write
curl -X POST http://localhost:8000/api/v1/memory/profile ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer <jwt>" ^
  -d "{\"key\":\"nickname\",\"value\":\"cyrax\"}"
```

### 7.9 Error shapes

Non-2xx responses follow the frozen `ErrorResponse` shape:

```json
{
  "error_code": "AUTH_TOKEN_EXPIRED",
  "message": "Token expired at ... (server time ...).",
  "trace_id": null
}
```

See `docs/api/error_codes.md` for the full catalogue.

---

## WebSocket Events

Connect with the JWT as a **query parameter** (not a header) per the frozen
contract:

```
ws://localhost:8000/api/v1/events?token=<jwt>
```

Or from JavaScript:

```js
const ws = new WebSocket(`ws://localhost:8000/api/v1/events?token=${jwt}`);
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

You will immediately receive a `connected` envelope:

```json
{
  "type": "connected",
  "payload": { "session_id": "<uuid>" },
  "timestamp": "2025-01-01T00:00:00.000Z"
}
```

Followed by `task_completed`, `task_failed`, `task_cancelled`, `task_progress`,
`notification`, and `heartbeat` (every 30s on idle connections) events.

**Invalid/missing token** → the server closes the connection with code **4401**
before upgrade.

---

## API Contract Documents

All contracts are FROZEN — the backend and the Android client build against
these:

| Document | Contents |
|---|---|
| `docs/api/authentication.md` | JWT payload, token transport, lifecycle |
| `docs/api/rest.md` | Endpoint summary, session resolution, route details |
| `docs/api/websocket.md` | WS lifecycle, envelope, event types, delivery guarantees |
| `docs/api/schemas.md` | Pydantic-equivalent JSON schemas |
| `docs/api/error_codes.md` | HTTP statuses, custom `error_code` values, WS close codes |
| `docs/api/versioning.md` | `/api/v1` versioning rules |
| `docs/QUICKSTART_UVICORN.md` | Fast boot of the uvicorn server |

---

## Quick Troubleshooting

| Symptom | Fix |
|---|---|
| `GROQ_API_KEY is required` | Add `GROQ_API_KEY` to `.env` (and `GEMINI_API_KEY` for `auto`). |
| `CYRAX_PIN_HASH is not a valid bcrypt hash` | Generate with `bcrypt.hashpw(...)` and set it in `.env`. |
| `ModuleNotFoundError: app` | Run from the project root. |
| `uvicorn is not recognised` | Activate venv and `pip install -r requirements.txt`. |
| Port 8000 in use | `--port 9000` or kill the occupying process. |
| `401 AUTH_TOKEN_INVALID` | You sent a missing/bad/expired JWT — re-login. |
| `423 LOCKED` | Too many failed PIN attempts — wait for lockout to expire. |
| WebSocket closes 4401 | Token missing/expired at handshake — re-login and reconnect. |

---

## License / Status

Phase 9.5D/9.5E complete. All 22 smoke tests pass. Contracts frozen for the
Android client build phase.

