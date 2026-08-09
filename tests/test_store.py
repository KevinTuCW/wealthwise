"""TDD tests for wealthwise.store — RunStore Protocol + InMemoryRunStore + SqliteRunStore."""
from __future__ import annotations

import tempfile
import os

import pytest

from wealthwise.store import (
    InMemoryRunStore,
    RunRecord,
    RunStore,
    SqliteRunStore,
    build_run_store,
)


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
        def __init__(self, backend: str, path: str = ":memory:"):
            self.run_store = backend
            self.run_store_path = path

    def test_memory_backend(self):
        settings = self._Settings("memory")
        store = build_run_store(settings)
        assert isinstance(store, InMemoryRunStore)

    def test_sqlite_backend(self):
        settings = self._Settings("sqlite", ":memory:")
        store = build_run_store(settings)
        assert isinstance(store, SqliteRunStore)

    def test_postgres_raises(self):
        settings = self._Settings("postgres")
        with pytest.raises(NotImplementedError):
            build_run_store(settings)

    def test_default_is_memory(self):
        class MinimalSettings:
            pass
        store = build_run_store(MinimalSettings())
        assert isinstance(store, InMemoryRunStore)
