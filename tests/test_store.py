"""TDD tests for wealthwise.store — RunStore Protocol + InMemoryRunStore + SqliteRunStore."""
from __future__ import annotations

import tempfile
import os

import pytest

from wealthwise.store import (
    InMemoryRunStore,
    PostgresRunStore,
    RunRecord,
    RunStore,
    SqliteRunStore,
    build_run_store,
)

#: Postgres integration tests run only against a database the operator supplies.
PG_DSN = os.environ.get("WEALTHWISE_TEST_PG_DSN", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**kwargs) -> RunRecord:
    defaults = {
        "profile_summary": "C3 | 500,000 CNY | 5y",
        "status": "done",
        "decision": "PASS",
        "portfolio_r_level": "R3",
        "tokens_used": 1200,
        "cost_usd": 0.00024,
        "dashboard_json": '{"status": "done"}',
    }
    defaults.update(kwargs)
    return RunRecord(**defaults)


# ---------------------------------------------------------------------------
# InMemoryRunStore
# ---------------------------------------------------------------------------

class TestInMemoryRunStore:
    def test_protocol(self):
        store = InMemoryRunStore()
        assert isinstance(store, RunStore)

    def test_save_returns_run_id(self):
        store = InMemoryRunStore()
        rec = _make_record()
        run_id = store.save(rec)
        assert run_id == rec.run_id

    def test_list_empty(self):
        store = InMemoryRunStore()
        assert store.list() == []

    def test_save_and_list_round_trip(self):
        store = InMemoryRunStore()
        rec1 = _make_record(status="done", decision="PASS")
        rec2 = _make_record(status="NEEDS_HUMAN_REVIEW", decision="DOWNGRADE")
        store.save(rec1)
        store.save(rec2)
        result = store.list()
        # Most recent first
        assert result[0].run_id == rec2.run_id
        assert result[1].run_id == rec1.run_id

    def test_list_limit(self):
        store = InMemoryRunStore()
        for i in range(10):
            store.save(_make_record())
        assert len(store.list(limit=3)) == 3

    def test_get_existing(self):
        store = InMemoryRunStore()
        rec = _make_record(status="done")
        store.save(rec)
        fetched = store.get(rec.run_id)
        assert fetched is not None
        assert fetched.run_id == rec.run_id
        assert fetched.decision == "PASS"
        assert fetched.portfolio_r_level == "R3"

    def test_get_missing(self):
        store = InMemoryRunStore()
        assert store.get("nonexistent_id") is None

    def test_record_fields_preserved(self):
        store = InMemoryRunStore()
        rec = _make_record(tokens_used=5000, cost_usd=0.001, profile_summary="C5 | 1,000,000 CNY | 10y")
        store.save(rec)
        fetched = store.get(rec.run_id)
        assert fetched.tokens_used == 5000
        assert fetched.cost_usd == pytest.approx(0.001)
        assert fetched.profile_summary == "C5 | 1,000,000 CNY | 10y"


# ---------------------------------------------------------------------------
# SqliteRunStore
# ---------------------------------------------------------------------------

class TestSqliteRunStore:
    def _temp_store(self) -> SqliteRunStore:
        """Create a SqliteRunStore backed by a temp file (auto-deleted by OS)."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        # We delete the file so SqliteRunStore creates it fresh
        os.unlink(path)
        return SqliteRunStore(path)

    def test_memory_backend(self):
        """sqlite3 :memory: backend — no file created."""
        store = SqliteRunStore(":memory:")
        assert isinstance(store, RunStore)

    def test_save_and_list_round_trip(self):
        store = SqliteRunStore(":memory:")
        rec1 = _make_record(status="done")
        rec2 = _make_record(status="NEEDS_HUMAN_REVIEW", decision="DOWNGRADE")
        store.save(rec1)
        store.save(rec2)
        result = store.list()
        assert len(result) == 2
        # Most recent first (ordered by created_at DESC)
        ids = [r.run_id for r in result]
        assert rec2.run_id in ids
        assert rec1.run_id in ids

    def test_get_existing(self):
        store = SqliteRunStore(":memory:")
        rec = _make_record(tokens_used=800, cost_usd=0.00016)
        store.save(rec)
        fetched = store.get(rec.run_id)
        assert fetched is not None
        assert fetched.run_id == rec.run_id
        assert fetched.tokens_used == 800
        assert fetched.cost_usd == pytest.approx(0.00016)

    def test_get_missing(self):
        store = SqliteRunStore(":memory:")
        assert store.get("no_such_id") is None

    def test_list_limit(self):
        store = SqliteRunStore(":memory:")
        for _ in range(8):
            store.save(_make_record())
        assert len(store.list(limit=3)) == 3

    def test_dashboard_json_persisted(self):
        store = SqliteRunStore(":memory:")
        rec = _make_record(dashboard_json='{"status":"done","allocation":{}}')
        store.save(rec)
        fetched = store.get(rec.run_id)
        assert fetched.dashboard_json == '{"status":"done","allocation":{}}'

    def test_file_backend(self):
        """Verify SQLite file is created and survives re-open."""
        store = self._temp_store()
        rec = _make_record(status="done")
        store.save(rec)
        # Re-open with the same path
        store2 = SqliteRunStore(store._path)
        fetched = store2.get(rec.run_id)
        assert fetched is not None
        assert fetched.run_id == rec.run_id
        # Cleanup
        try:
            os.unlink(store._path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# build_run_store gating
# ---------------------------------------------------------------------------

class TestBuildRunStore:
    class _Settings:
        def __init__(self, backend: str, path: str = ":memory:", dsn: str = ""):
            self.run_store = backend
            self.run_store_path = path
            self.run_store_dsn = dsn

    def test_memory_backend(self):
        settings = self._Settings("memory")
        store = build_run_store(settings)
        assert isinstance(store, InMemoryRunStore)

    def test_sqlite_backend(self):
        settings = self._Settings("sqlite", ":memory:")
        store = build_run_store(settings)
        assert isinstance(store, SqliteRunStore)

    def test_postgres_without_dsn_fails_loudly(self):
        """Refuse to start rather than write the audit log somewhere unread."""
        settings = self._Settings("postgres")
        with pytest.raises(ValueError, match="RUN_STORE_DSN"):
            build_run_store(settings)

    @pytest.mark.skipif(not PG_DSN, reason="set WEALTHWISE_TEST_PG_DSN to run")
    def test_postgres_backend(self):
        settings = self._Settings("postgres", dsn=PG_DSN)
        assert isinstance(build_run_store(settings), PostgresRunStore)

    def test_default_is_memory(self):
        class MinimalSettings:
            pass
        store = build_run_store(MinimalSettings())
        assert isinstance(store, InMemoryRunStore)


# ---------------------------------------------------------------------------
# PostgresRunStore — integration, skipped unless a test database is supplied
#
# Gated on an env var rather than faked in-process: a fake would test the fake,
# and the reason this backend exists at all is how a real Postgres behaves with
# more than one writer.
#
#   createdb wealthwise
#   WEALTHWISE_TEST_PG_DSN=postgresql:///wealthwise pytest tests/test_store.py
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PG_DSN, reason="set WEALTHWISE_TEST_PG_DSN to run")
class TestPostgresRunStore:
    @pytest.fixture
    def store(self):
        import psycopg

        s = PostgresRunStore(PG_DSN)
        with psycopg.connect(PG_DSN) as c, c.cursor() as cur:
            cur.execute("TRUNCATE runs")      # append-only log, per-test clean slate
        return s

    def test_save_then_get_round_trips_every_field(self, store):
        record = _make_record()
        run_id = store.save(record)
        got = store.get(run_id)
        assert got is not None
        assert got.model_dump() == record.model_dump()

    def test_get_unknown_run_id_returns_none(self, store):
        assert store.get("no-such-run") is None

    def test_list_is_newest_first_and_honours_limit(self, store):
        import datetime

        base = datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc)
        for i in range(5):
            store.save(_make_record(
                created_at=(base + datetime.timedelta(minutes=i)).isoformat(),
                profile_summary=f"run-{i}",
            ))
        got = store.list(limit=3)
        assert [r.profile_summary for r in got] == ["run-4", "run-3", "run-2"]

    def test_dashboard_json_is_stored_byte_for_byte(self, store):
        """An audit record tidied on the way in is no longer evidence."""
        raw = '{"b": 1, "a": [2,  3], "spaced": "  keep  "}'
        run_id = store.save(_make_record(dashboard_json=raw))
        assert store.get(run_id).dashboard_json == raw

    def test_survives_a_new_connection(self, store):
        """Durability is the whole point: a second store must see the record."""
        run_id = store.save(_make_record(profile_summary="C4 | 1,000,000 CNY | 8y"))
        assert PostgresRunStore(PG_DSN).get(run_id).profile_summary == \
            "C4 | 1,000,000 CNY | 8y"

    def test_satisfies_the_run_store_protocol(self, store):
        assert isinstance(store, RunStore)
