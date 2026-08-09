"""Immutable run/audit persistence for WealthWise advisory runs.

Every advisory run surfaced through the API is appended as an audit record
(profile summary, decision, portfolio risk level, status, spend).
Default is in-memory (diskless — tests and eval stay clean); set
RUN_STORE=sqlite for durable, restart-surviving audit with zero extra
dependencies (stdlib sqlite3). Postgres is reserved.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = Field(default_factory=_now)
    # Profile summary (serialisable subset — no raw PII)
    profile_summary: str = ""              # e.g. "C3 | 500000 CNY | 5y"
    status: str = "pending"
    decision: str = ""                     # compliance decision: PASS/DOWNGRADE/REJECT/""
    portfolio_r_level: str = ""            # e.g. "R3"
    tokens_used: int = 0
    cost_usd: float = 0.0
    # Full dashboard JSON stored as a string for audit retrieval
    dashboard_json: str = "{}"


@runtime_checkable
class RunStore(Protocol):
    def save(self, record: RunRecord) -> str: ...
    def list(self, limit: int = 50) -> list[RunRecord]: ...
    def get(self, run_id: str) -> RunRecord | None: ...


class InMemoryRunStore:
    def __init__(self) -> None:
        self._rows: list[RunRecord] = []

    def save(self, record: RunRecord) -> str:
        self._rows.append(record)
        return record.run_id

    def list(self, limit: int = 50) -> list[RunRecord]:
        return list(reversed(self._rows))[:limit]

    def get(self, run_id: str) -> RunRecord | None:
        return next((r for r in self._rows if r.run_id == run_id), None)


_COLUMNS = (
    "run_id", "created_at", "profile_summary", "status",
    "decision", "portfolio_r_level", "tokens_used", "cost_usd", "dashboard_json",
)


class SqliteRunStore:
    """Durable append-only audit log backed by stdlib sqlite3.

    For `:memory:` paths a single persistent connection is kept open so that
    the schema created at __init__ time is visible to all subsequent calls
    (each sqlite3.connect(":memory:") opens an independent empty database).
    For file paths a new connection is opened per operation as usual.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        # For in-memory databases, keep a single shared connection alive.
        self._shared_conn: sqlite3.Connection | None = None
        if path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS runs(
                run_id TEXT PRIMARY KEY,
                created_at TEXT,
                profile_summary TEXT,
                status TEXT,
                decision TEXT,
                portfolio_r_level TEXT,
                tokens_used INTEGER,
                cost_usd REAL,
                dashboard_json TEXT
            )""")
            c.commit()

    def _conn(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        return sqlite3.connect(self._path)

    def save(self, record: RunRecord) -> str:
        with self._conn() as c:
            c.execute(
                f"INSERT INTO runs({','.join(_COLUMNS)}) "
                f"VALUES({','.join('?' * len(_COLUMNS))})",
                (
                    record.run_id,
                    record.created_at,
                    record.profile_summary,
                    record.status,
                    record.decision,
                    record.portfolio_r_level,
                    record.tokens_used,
                    record.cost_usd,
                    record.dashboard_json,
                ),
            )
        return record.run_id

    def _list_all(self, limit: int) -> list[RunRecord]:
        """Fetch the most recent *limit* records ordered newest-first."""
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {','.join(_COLUMNS)} FROM runs "
                f"ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def _get_by_id(self, run_id: str) -> list[RunRecord]:
        """Fetch zero or one record by primary key."""
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {','.join(_COLUMNS)} FROM runs "
                f"WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def list(self, limit: int = 50) -> list[RunRecord]:
        return self._list_all(limit)

    def get(self, run_id: str) -> RunRecord | None:
        rows = self._get_by_id(run_id)
        return rows[0] if rows else None

    @staticmethod
    def _row(r: tuple) -> RunRecord:
        return RunRecord(
            run_id=r[0],
            created_at=r[1],
            profile_summary=r[2],
            status=r[3],
            decision=r[4],
            portfolio_r_level=r[5],
            tokens_used=r[6],
            cost_usd=r[7],
            dashboard_json=r[8],
        )


def build_run_store(settings) -> RunStore:
    """Build the RunStore from settings.run_store.

    "sqlite"   → SqliteRunStore at settings.run_store_path.
    "memory"   → InMemoryRunStore (default).
    "postgres" → NotImplementedError (reserved).
    """
    backend = getattr(settings, "run_store", "memory")
    if backend == "sqlite":
        path = settings.run_store_path
        return SqliteRunStore(path)
    if backend == "postgres":
        raise NotImplementedError("Postgres RunStore not yet implemented")
    return InMemoryRunStore()
