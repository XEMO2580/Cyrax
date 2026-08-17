# 🚀 CYRAX 3.0 Distributed API — Uvicorn Quick-Start Guide

This guide gets the **CYRAX 3.0 Distributed API** (FastAPI + Uvicorn) up and
running quickly. It assumes you already have a virtual environment with
dependencies installed. For a full step-by-step setup, see `../README.md`.

---

## 1. One-Line Boot (from project root)

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

If you use the project's virtual environment (Windows):

```bash
.venv\Scripts\uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

or activate the venv first:

```bash
.venv\Scripts\activate
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** The `--reload` flag is for development. **Remove it in production.**

---

## 2. What You Should See

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     [API] Core OS booted.
INFO:     [API] Session eviction loop started.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

If you see `[CYRAX BOOT FAILURE]`, the Core OS failed to boot — see
Section 6 (Troubleshooting) below.

---

## 3. Verify It's Alive

Open another terminal and hit the no-auth health probe:

```bash
curl http://localhost:8000/api/v1/health
```

Expected:

```json
{"status":"ok","version":"3.0"}
```

---

## 4. Useful URLs Once Running

| URL | What it is |
|---|---|
| `http://localhost:8000/api/v1/health` | Liveness/readiness probe (no auth) |
| `http://localhost:8000/docs` | Interactive Swagger UI (beautiful `/docs`) |
| `http://localhost:8000/redoc` | ReDoc alternative API docs |
| `http://localhost:8000/openapi.json` | Raw OpenAPI schema |

---

## 5. Common Uvicorn Command Variations

| What you want | Command |
|---|---|
| Development server with auto-reload | `uvicorn app.api.main:app --reload` |
| Bind on all interfaces (for a phone on your LAN) | `uvicorn app.api.main:app --host 0.0.0.0` |
| Different port | `uvicorn app.api.main:app --port 9000` |
| Production (no reload, 4 workers) | `uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 4` |
| Access logs off (quiet) | `uvicorn app.api.main:app --no-access-log` |
| HTTPS via TLS cert | `uvicorn app.api.main:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem` |

> **Production note:** `--workers > 1` requires the app to be import-safe
> (module-level state is created per-process). The `SessionManager` is a
> per-process singleton, so with multiple workers each worker keeps its own
> in-memory session pool and JWT verification still works across workers
> (it is stateless HMAC). For a single-device deployment, `--workers 1` is
> the recommended default.

---

## 6. Troubleshooting

### `[CYRAX BOOT FAILURE] GROQ_API_KEY is required...`
Add the missing key(s) to your `.env` file (project root). The settings
validator requires:
- `GROQ_API_KEY` when `ACTIVE_LLM` is `auto` or `groq`
- `GEMINI_API_KEY` when `ACTIVE_LLM` is `auto` or `gemini`

### `[CYRAX BOOT FAILURE] CYRAX_PIN_HASH`
Generate a bcrypt hash and put it in `.env`:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'YOUR_PIN', bcrypt.gensalt()).decode())"
```

### `uvicorn` is not recognised
Either you haven't installed dependencies, or you're not using the venv:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Module not found: `app.api.main`
Make sure you are running from the **project root**
(`c:/Users/subha/OneDrive/Documents/CYRAX 3.0`), where `app/` lives.

### Port already in use
Find and stop the process, or pick a different port with `--port`.

---

## 7. Next Steps for Mobile Development

1. **Login first** to get a JWT:
   ```bash
   curl -X POST http://localhost:8000/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"device_id":"<your-device-uuid>","pin":"<your-pin>"}'
   ```
2. Use the returned `token` as a Bearer header on every authenticated route:
   ```bash
   curl http://localhost:8000/api/v1/chat \
     -H "Authorization: Bearer <jwt>" \
     -H "Content-Type: application/json" \
     -d '{"message":"hello"}'
   ```
3. Open the WebSocket event stream:
   ```
   ws://localhost:8000/api/v1/events?token=<jwt>
   ```
4. Browse `http://localhost:8000/docs` to explore every endpoint interactively.

Full endpoint contract: `docs/api/rest.md`, `docs/api/websocket.md`,
`docs/api/authentication.md`.

