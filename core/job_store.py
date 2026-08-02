# core/job_store.py

"""
core/job_store.py — CYRAX 3.0 Phase 8.5 SQLite Persistence Layer

Async via aiosqlite — no blocking I/O on the main event loop.
This is the single source of truth for durability across reboots.
TaskQueue remains the RAM-side execution order; this file is disk.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from core.task import Task, TaskStatus, TaskType

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.home() / "CYRAX_WORKSPACE" / ".cyrax_memory" / "jobs.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    user_input      TEXT NOT NULL,
    task_type       TEXT NOT NULL,
    status          TEXT NOT NULL,
    tools_required  INTEGER NOT NULL DEFAULT 0,
    provider_name   TEXT NOT NULL DEFAULT 'groq',
    interruptible   INTEGER NOT NULL DEFAULT 1,
    priority        INTEGER NOT NULL DEFAULT 3,
    result          TEXT,
    error           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    scheduled_at    TEXT
);
"""

# Phase 8.6 migration: add the priority column to databases created by
# earlier phases (the CREATE TABLE IF NOT EXISTS above only helps fresh
# DBs, not an existing jobs table lacking the column).
_ADD_PRIORITY_COLUMN_SQL = "ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 3"

# ── Phase 7.3: LearningRouter routing_history table ──────────────────────────

_CREATE_ROUTING_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS routing_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    intent_family   TEXT NOT NULL,
    provider        TEXT NOT NULL,
    success         INTEGER NOT NULL,
    latency_ms      REAL NOT NULL,
    fallback_used   INTEGER NOT NULL DEFAULT 0,
    timestamp       TEXT NOT NULL,
    reason          TEXT
);
"""

_CREATE_ROUTING_HISTORY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_routing_history_intent_provider
ON routing_history (intent_family, provider);
"""


def _row_to_task(row: aiosqlite.Row) -> Task:
    task = Task(
        task_id=row["job_id"],
        user_input=row["user_input"],
        task_type=TaskType(row["task_type"]),
        status=TaskStatus(row["status"]),
        tools_required=bool(row["tools_required"]),
        provider_name=row["provider_name"],
        interruptible=bool(row["interruptible"]),
        result=row["result"],
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        scheduled_at=datetime.fromisoformat(row["scheduled_at"]) if row["scheduled_at"] else None,
    )
    # Phase 8.6 — restore persisted priority. Defaults to BACKGROUND (3)
    # if the column is absent on a not-yet-migrated old database.
    if "priority" in row.keys():
        task.priority = row["priority"]
    return task


class SQLiteJobStore:
    """
    Async SQLite persistence for Task records. One connection is opened
    lazily and reused — aiosqlite connections are safe for sequential
    use from a single event loop, which matches CYRAX's single-loop model.

    init() must be awaited once before any other method — creates the
    table if missing. Not done in __init__ since __init__ can't be async.
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute(_CREATE_TABLE_SQL)
        await self._conn.execute(_CREATE_ROUTING_HISTORY_TABLE_SQL)
        await self._conn.execute(_CREATE_ROUTING_HISTORY_INDEX_SQL)

        # Phase 8.6 migration — add priority to pre-existing databases.
        # Best-effort: if the column already exists, SQLite raises
        # OperationalError("duplicate column name") which we swallow.
        try:
            await self._conn.execute(_ADD_PRIORITY_COLUMN_SQL)
            await self._conn.commit()
            logger.info("[JOB_STORE] Applied Phase 8.6 migration: added 'priority' column.")
        except Exception as exc:
            logger.debug(
                f"[JOB_STORE] priority column already present (or migration "
                f"not needed): {exc}"
            )

        await self._conn.commit()
        logger.info(f"[JOB_STORE] Initialised. DB: {self._db_path}")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError(
                "SQLiteJobStore.init() must be awaited before any other method."
            )
        return self._conn

    async def insert_job(self, task: Task) -> None:
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO jobs (
                job_id, user_input, task_type, status, tools_required,
                provider_name, interruptible, priority, result, error,
                created_at, updated_at, scheduled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.task_id, task.user_input, task.task_type.value,
                task.status.value, int(task.tools_required),
                task.provider_name, int(task.interruptible),
                int(task.priority),
                task.result, task.error,
                task.created_at.isoformat(), task.updated_at.isoformat(),
                task.scheduled_at.isoformat() if task.scheduled_at else None,
            ),
        )
        await conn.commit()
        logger.debug(f"[JOB_STORE] Inserted: {task.task_id}")

    async def update_status(
        self,
        job_id:  str,
        status:  TaskStatus,
        result:  str | None = None,
        error:   str | None = None,
    ) -> None:
        conn = self._require_conn()
        now = datetime.now(timezone.utc).isoformat()

        if result is not None and error is not None:
            await conn.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, updated_at = ? WHERE job_id = ?",
                (status.value, result, error, now, job_id),
            )
        elif result is not None:
            await conn.execute(
                "UPDATE jobs SET status = ?, result = ?, updated_at = ? WHERE job_id = ?",
                (status.value, result, now, job_id),
            )
        elif error is not None:
            await conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?",
                (status.value, error, now, job_id),
            )
        else:
            await conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (status.value, now, job_id),
            )
        await conn.commit()
        logger.debug(f"[JOB_STORE] Updated: {job_id} | status={status.value}")

    async def get_due_jobs(self) -> list[Task]:
        """Returns SCHEDULED jobs whose scheduled_at has already passed."""
        conn = self._require_conn()
        now = datetime.now(timezone.utc).isoformat()

        cursor = await conn.execute(
            "SELECT * FROM jobs WHERE status = ? AND scheduled_at IS NOT NULL AND scheduled_at <= ?",
            (TaskStatus.SCHEDULED.value, now),
        )
        rows = await cursor.fetchall()
        return [_row_to_task(row) for row in rows]

    async def recover_pending_jobs(self) -> list[Task]:
        """
        Boot-time crash recovery. Any job left RUNNING when the process
        died is reverted to PENDING (it never got to finish, so it must
        be re-executed). Returns the combined list of already-PENDING
        and just-reverted jobs, ready for the executor to pick up.
        """
        conn = self._require_conn()
        now = datetime.now(timezone.utc).isoformat()

        cursor = await conn.execute(
            "SELECT job_id FROM jobs WHERE status = ?", (TaskStatus.RUNNING.value,)
        )
        stuck_rows = await cursor.fetchall()
        stuck_ids = [row["job_id"] for row in stuck_rows]

        if stuck_ids:
            placeholders = ",".join("?" for _ in stuck_ids)
            await conn.execute(
                f"UPDATE jobs SET status = ?, updated_at = ? WHERE job_id IN ({placeholders})",
                (TaskStatus.PENDING.value, now, *stuck_ids),
            )
            await conn.commit()
            logger.warning(
                f"[JOB_STORE] Recovered {len(stuck_ids)} crashed RUNNING "
                f"job(s) back to PENDING: {stuck_ids}"
            )

        cursor = await conn.execute(
            "SELECT * FROM jobs WHERE status = ?", (TaskStatus.PENDING.value,)
        )
        rows = await cursor.fetchall()
        return [_row_to_task(row) for row in rows]

    # ── Phase 7.3: LearningRouter routing_history ─────────────────────────────

    async def record_routing_outcome(
        self,
        intent_family:  str,
        provider:       str,
        success:        bool,
        latency_ms:     float,
        fallback_used:  bool,
        reason:         str = "",
    ) -> None:
        conn = self._require_conn()
        now = datetime.now(timezone.utc).isoformat()

        await conn.execute(
            """
            INSERT INTO routing_history (
                intent_family, provider, success, latency_ms,
                fallback_used, timestamp, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent_family, provider, int(success), latency_ms,
                int(fallback_used), now, reason,
            ),
        )
        await conn.commit()
        logger.debug(
            f"[JOB_STORE] routing_history recorded: "
            f"{intent_family}/{provider} | success={success} | latency={latency_ms:.0f}ms"
        )

    async def get_routing_stats(self, intent_family: str, limit: int = 200) -> list[dict]:
        """
        Returns the most recent `limit` routing_history rows for this
        intent_family, newest first, as plain dicts (not Task objects —
        this table has nothing to do with Task's shape).

        limit exists so LearningRouter never has to load an unbounded
        history from disk; the decay function itself makes very old
        rows contribute negligibly, so a bounded window is sufficient.
        """
        conn = self._require_conn()

        cursor = await conn.execute(
            """
            SELECT provider, success, latency_ms, fallback_used, timestamp, reason
            FROM routing_history
            WHERE intent_family = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (intent_family, limit),
        )
        rows = await cursor.fetchall()

        return [
            {
                "provider":       row["provider"],
                "success":        bool(row["success"]),
                "latency_ms":     row["latency_ms"],
                "fallback_used":  bool(row["fallback_used"]),
                "timestamp":      row["timestamp"],
                "reason":         row["reason"],
            }
            for row in rows
        ]
