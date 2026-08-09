"""Tests for wealthwise FastAPI app — /health + workbench routes."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from wealthwise.app import app, _STORE

client = TestClient(app)

# ---------------------------------------------------------------------------
# Sample profile payload
# ---------------------------------------------------------------------------

_SAMPLE_PROFILE = {
    "risk_level": "C3",
    "investable": 500_000.0,
    "horizon_years": 5,
    "goals": ["balanced_growth"],
    "liquidity_min": 0.2,
    "accept_cross_border": True,
    "holdings": [],
}


# ---------------------------------------------------------------------------
# /health (existing)
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /workbench
# ---------------------------------------------------------------------------

def test_workbench_page_returns_200():
    response = client.get("/workbench")
    assert response.status_code == 200


def test_workbench_page_is_html():
    response = client.get("/workbench")
    ct = response.headers.get("content-type", "")
    assert "html" in ct or response.text.strip().startswith("<!doctype") or "<html" in response.text


# ---------------------------------------------------------------------------
# POST /workbench/run
# ---------------------------------------------------------------------------

def test_workbench_run_returns_200():
    response = client.post("/workbench/run", json=_SAMPLE_PROFILE)
    assert response.status_code == 200


def test_workbench_run_returns_five_panels():
    response = client.post("/workbench/run", json=_SAMPLE_PROFILE)
    body = response.json()
    for panel in ("allocation", "experts", "crosscheck", "compliance", "cost"):
        assert panel in body, f"response missing panel: {panel}"


def test_workbench_run_allocation_has_r_level():
    response = client.post("/workbench/run", json=_SAMPLE_PROFILE)
    alloc = response.json()["allocation"]
    assert "portfolio_r_level" in alloc
    assert alloc["portfolio_r_level"]  # non-empty


def test_workbench_run_compliance_has_cr_matrix():
    response = client.post("/workbench/run", json=_SAMPLE_PROFILE)
    comp = response.json()["compliance"]
    assert "cr_matrix" in comp
    assert len(comp["cr_matrix"]) == 5  # one row per C-level


def test_workbench_run_compliance_has_disclosure_checklist():
    response = client.post("/workbench/run", json=_SAMPLE_PROFILE)
    comp = response.json()["compliance"]
    assert "disclosure_checklist" in comp
    assert len(comp["disclosure_checklist"]) >= 4


def test_workbench_run_cost_has_tokens():
    response = client.post("/workbench/run", json=_SAMPLE_PROFILE)
    cost = response.json()["cost"]
    assert "tokens_used" in cost
    assert "cost_usd" in cost


def test_workbench_run_includes_run_id():
    response = client.post("/workbench/run", json=_SAMPLE_PROFILE)
    assert "run_id" in response.json()


def test_workbench_run_persists_to_store():
    """After POST /workbench/run, GET /runs should include the new run."""
    # Clear in-memory store by running and comparing before/after counts
    runs_before = client.get("/runs").json()["runs"]
    count_before = len(runs_before)

    client.post("/workbench/run", json=_SAMPLE_PROFILE)

    runs_after = client.get("/runs").json()["runs"]
    count_after = len(runs_after)
    assert count_after == count_before + 1


def test_workbench_run_c1_profile():
    """A conservative C1 investor should still get a dashboard (possibly with violations)."""
    profile = {
        "risk_level": "C1",
        "investable": 200_000.0,
        "horizon_years": 2,
        "goals": ["capital_preservation"],
        "liquidity_min": 0.5,
        "accept_cross_border": False,
        "holdings": [],
    }
    response = client.post("/workbench/run", json=profile)
    assert response.status_code == 200
    body = response.json()
    for panel in ("allocation", "experts", "crosscheck", "compliance", "cost"):
        assert panel in body


# ---------------------------------------------------------------------------
# GET /workbench/dashboard
# ---------------------------------------------------------------------------

def test_workbench_dashboard_after_run():
    """After a run, GET /workbench/dashboard (no run_id) returns the stored dashboard."""
    # Ensure there is at least one run
    run_resp = client.post("/workbench/run", json=_SAMPLE_PROFILE)
    run_id = run_resp.json().get("run_id", "")

    # Fetch by run_id
    dash_resp = client.get(f"/workbench/dashboard?run_id={run_id}")
    assert dash_resp.status_code == 200
    dash = dash_resp.json()
    # Should have the dashboard panels (not an error dict)
    assert "error" not in dash
    for panel in ("allocation", "experts", "crosscheck", "compliance", "cost"):
        assert panel in dash, f"dashboard missing panel: {panel}"


def test_workbench_dashboard_latest_run():
    """GET /workbench/dashboard with no run_id returns the most recent run."""
    client.post("/workbench/run", json=_SAMPLE_PROFILE)
    resp = client.get("/workbench/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert "error" not in body


def test_workbench_dashboard_unknown_run_id():
    resp = client.get("/workbench/dashboard?run_id=nonexistent_id_xyz")
    assert resp.status_code == 200
    body = resp.json()
    assert "error" in body


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------

def test_runs_returns_list():
    response = client.get("/runs")
    assert response.status_code == 200
    body = response.json()
    assert "runs" in body
    assert isinstance(body["runs"], list)


def test_runs_contains_recent_run():
    # Add a run and check it appears
    run_resp = client.post("/workbench/run", json=_SAMPLE_PROFILE)
    run_id = run_resp.json().get("run_id", "")

    runs_resp = client.get("/runs")
    run_ids = [r["run_id"] for r in runs_resp.json()["runs"]]
    assert run_id in run_ids


def test_runs_record_has_expected_fields():
    client.post("/workbench/run", json=_SAMPLE_PROFILE)
    runs_resp = client.get("/runs")
    runs = runs_resp.json()["runs"]
    assert len(runs) >= 1
    rec = runs[0]
    for field in ("run_id", "created_at", "profile_summary", "status",
                  "decision", "portfolio_r_level", "tokens_used", "cost_usd"):
        assert field in rec, f"run record missing field: {field}"


def test_runs_limit_param():
    # Add multiple runs
    for _ in range(3):
        client.post("/workbench/run", json=_SAMPLE_PROFILE)
    runs_resp = client.get("/runs?limit=2")
    assert len(runs_resp.json()["runs"]) <= 2
