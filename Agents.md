# CYRAX ENGINEERING AGENT CONTRACT

## Project

CYRAX is being developed as a distributed personal AI operating system.

The current architecture contains:

- FastAPI backend
- Python V12 asynchronous execution engine
- MoERouter
- Decision Engine
- Planner
- Task Queue
- SQLite Job Store
- Notification Center
- Interrupt Controller
- Learning Router
- Tool Registry
- Android Jetpack Compose client
- Room persistence
- authentication/security layer
- provider integrations

## Human Authority

XEMO is the Product Owner and Final Authority.

## Architecture Authority

GPT is the Chief Architect.

Agents must not make architectural changes without an explicit task allowing such changes.

## Backend Owner

Google Antigravity is responsible for backend/V12 engineering.

## Android Owner

Gemini Agent in Android Studio is responsible for Android engineering.

## Product Design Owner

Google Stitch is responsible for product/UI design exploration and design-system artifacts.

## QA Owner

Gemini QA validates implementation.

Firebase Test Lab is used for device-matrix validation.

## Security Principle

Authentication and authorization are different concerns.

Authentication answers:
"Who is the user?"

Authorization answers:
"What is this user/device allowed to do?"

Do not remove or bypass V12 authorization controls when introducing a user-authentication provider.

## Device Principle

CYRAX must eventually support explicit target-device execution.

A task must not implicitly execute on the backend host simply because that host happens to run the TaskExecutor.

## Backend Principle

The V12 engine is the system of record for:

- task execution
- routing
- provider selection
- authorization
- tool selection
- device selection
- retries
- failover
- task lifecycle
- observability

## Modification Policy

Never silently change architecture.

Never silently change provider models.

Never silently remove security gates.

Never silently change API contracts.

Never silently change database schemas.

Every modification requires:

- reason
- affected files
- expected behavior
- tests
- known risks
- follow-up

## Audit Policy

Audit tasks are READ-ONLY unless explicitly stated otherwise.

## Code Quality

Prefer:

- explicit contracts
- typed models
- deterministic state machines
- bounded retries
- observable failures
- idempotent operations
- testable components
- clear ownership boundaries

Avoid:

- hidden global state
- duplicated business rules
- UI-driven authorization
- provider-specific logic leaking into orchestration
- hardcoded model assumptions
- device execution without an explicit target
- silent fallback behavior