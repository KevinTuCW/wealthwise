"""Immutable run/audit persistence for WealthWise advisory runs.

Every advisory run surfaced through the API is appended as an audit record
(profile summary, decision, portfolio risk level, status, spend).

Three backends, chosen by `RUN_STORE`:

``memory``
    Default. Diskless, so tests and eval leave nothing behind.
``sqlite``
    Durable and restart-surviving with zero extra dependencies (stdlib
    sqlite3). One writer, one file — right up to the point the app runs as
    more than one process.
``postgres``
    The multi-process answer. Two uvicorn workers against one SQLite file
    serialise on a write lock and, on a network filesystem, corrupt it; an
    audit log that cannot be written concurrently is the wrong shape for the
    thing it is auditing. Needs `psycopg` (`pip install -e '.[pg]'`) and
    `RUN_STORE_DSN`.

All three implement the same `RunStore` protocol and the same append-only
semantics, so the backend is a deployment decision rather than a code one.
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


class PostgresRunStore:
    """Durable append-only audit log backed by Postgres via psycopg 3.

    A connection is opened per operation rather than pooled. The store is
    written once per advisory run — a run that costs seconds of model latency —
    so the few milliseconds of connect are invisible next to it, and not holding
    a long-lived socket means a database restart heals by itself instead of
    leaving every worker holding a dead handle.

    `dashboard_json` is stored as TEXT, not JSONB, which looks like leaving the
    native type on the table. It is the point: JSONB normalises whitespace,
    reorders keys and drops duplicates, so what came back out would no longer be
    the bytes that were served to the investor. An audit record that has been
    helpfully tidied is not evidence. Indexing on the JSON is not a use case
    here — `/runs` reads by primary key and by recency, both of which are
    columns.
    """

    #: Same column list and order as the SQLite store, so a record written by
    #: one backend reads identically from the other.
    _DDL = """
        CREATE TABLE IF NOT EXISTS runs(
            run_id            TEXT PRIMARY KEY,
            created_at        TIMESTAMPTZ NOT NULL,
            profile_summary   TEXT,
            status            TEXT,
            decision          TEXT,
            portfolio_r_level TEXT,
            tokens_used       INTEGER,
            cost_usd          DOUBLE PRECISION,
            dashboard_json    TEXT
        )
    """

    # `list()` is always "most recent N", and created_at is not the primary key,
    # so without this the common read degrades to a full scan plus a sort as the
    # log grows — which is exactly what an append-only table does.
    _INDEX = "CREATE INDEX IF NOT EXISTS runs_created_at_idx ON runs(created_at DESC)"

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("RUN_STORE=postgres requires RUN_STORE_DSN")
        self._dsn = dsn
        with self._conn() as c, c.cursor() as cur:
            cur.execute(self._DDL)
            cur.execute(self._INDEX)

    def _conn(self):
        """Open a connection. psycopg is imported lazily — it is an extra."""
        import psycopg  # lazy: `pip install -e '.[pg]'`, not a base dependency

        return psycopg.connect(self._dsn)

    def save(self, record: RunRecord) -> str:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"INSERT INTO runs({','.join(_COLUMNS)}) "
                f"VALUES({','.join(['%s'] * len(_COLUMNS))})",
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

    def list(self, limit: int = 50) -> list[RunRecord]:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"SELECT {','.join(_COLUMNS)} FROM runs "
                f"ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [self._row(r) for r in rows]

    def get(self, run_id: str) -> RunRecord | None:
        with self._conn() as c, c.cursor() as cur:
            cur.execute(
                f"SELECT {','.join(_COLUMNS)} FROM runs WHERE run_id = %s",
                (run_id,),
            )
            rows = cur.fetchall()
        return self._row(rows[0]) if rows else None

    @staticmethod
    def _row(r: tuple) -> RunRecord:
        created_at = r[1]
        # TIMESTAMPTZ comes back as an aware datetime rendered in the *session's*
        # timezone, so the same instant reads as +08:00 here and +00:00 in CI.
        # Records are written in UTC, and an audit log whose timestamps depend on
        # where they were read from is one nobody can line up with a trace.
        if isinstance(created_at, datetime):
            created_at = created_at.astimezone(timezone.utc).isoformat()

        return RunRecord(
            run_id=r[0],
            created_at=created_at if isinstance(created_at, str) else str(created_at),
            profile_summary=r[2] or "",
            status=r[3] or "",
            decision=r[4] or "",
            portfolio_r_level=r[5] or "",
            tokens_used=r[6] or 0,
            cost_usd=r[7] or 0.0,
            dashboard_json=r[8] or "{}",
        )


def build_run_store(settings) -> RunStore:
    """Build the RunStore from settings.run_store.

    "sqlite"   → SqliteRunStore at settings.run_store_path.
    "postgres" → PostgresRunStore at settings.run_store_dsn.
    "memory"   → InMemoryRunStore (default).
    """
    backend = getattr(settings, "run_store", "memory")
    if backend == "sqlite":
        path = settings.run_store_path
        return SqliteRunStore(path)
    if backend == "postgres":
        return PostgresRunStore(getattr(settings, "run_store_dsn", ""))
    return InMemoryRunStore()
