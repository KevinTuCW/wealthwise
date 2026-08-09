"""Wealthwise FastAPI application.

Routes:
  GET  /health              — liveness probe
  GET  /workbench           — serves static/workbench.html
  POST /workbench/run       — run advisory pipeline, persist audit record, return dashboard
  GET  /workbench/dashboard — return stored dashboard (run_id query param, or latest run)
  GET  /workbench/stream    — SSE streaming of advisory pipeline nodes
  GET  /runs                — audit log of recent runs
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from wealthwise.agents.state import InvestorProfile
from wealthwise.bootstrap import build_runtime_deps
from wealthwise.config import get_settings
from wealthwise.runner import run_advisory
from wealthwise.store import RunRecord, build_run_store
from wealthwise.workbench import build_dashboard, sse_events

app = FastAPI(
    title="WealthWise",
    description="金融投顾多 Agent 系统 (阵03)",
)

_STATIC = Path(__file__).parent.parent.parent / "web" / "static"
_STORE = build_run_store(get_settings())


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    risk_level: str = "C3"
    investable: float = 500_000.0
    horizon_years: int = 5
    goals: list[str] = ["balanced_growth"]
    liquidity_min: float = 0.2
    accept_cross_border: bool = True
    holdings: list[str] = []


def _profile_from_request(req: RunRequest) -> InvestorProfile:
    return InvestorProfile(
        risk_level=req.risk_level,  # type: ignore[arg-type]
        investable=req.investable,
        horizon_years=req.horizon_years,
        goals=req.goals,
        liquidity_min=req.liquidity_min,
        accept_cross_border=req.accept_cross_border,
        holdings=req.holdings,
    )


def _profile_summary(profile: InvestorProfile) -> str:
    return f"{profile.risk_level} | {profile.investable:,.0f} CNY | {profile.horizon_years}y"


def _run_and_record(profile: InvestorProfile) -> dict:
    """Run the advisory pipeline, persist an audit record, and return dashboard."""
    settings = get_settings()
    deps = build_runtime_deps(settings)
    state = run_advisory(profile, deps)
    dashboard = build_dashboard(state, settings)

    record = RunRecord(
        profile_summary=_profile_summary(profile),
        status=state.status,
        decision=state.compliance.decision if state.compliance else "",
        portfolio_r_level=state.portfolio.portfolio_r_level if state.portfolio else "",
        tokens_used=state.tokens_used,
        cost_usd=dashboard["cost"]["cost_usd"],
        dashboard_json=json.dumps(dashboard, ensure_ascii=False),
    )
    run_id = _STORE.save(record)
    return {"run_id": run_id, **dashboard}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/workbench")
def workbench_page() -> FileResponse:
    return FileResponse(_STATIC / "workbench.html")


@app.post("/workbench/run")
def workbench_run(req: RunRequest) -> dict:
    """Run the advisory pipeline for the given profile and persist the result."""
    profile = _profile_from_request(req)
    return _run_and_record(profile)


@app.get("/workbench/dashboard")
def workbench_dashboard(run_id: str = Query("")) -> dict:
    """Return a stored run's dashboard (by run_id), or the most recent run."""
    if run_id:
        record = _STORE.get(run_id)
        if record is None:
            return {"error": f"run_id={run_id!r} not found"}
        return json.loads(record.dashboard_json)
    # Fall back to most recent run
    runs = _STORE.list(limit=1)
    if not runs:
        return {"error": "no runs found — POST /workbench/run first"}
    return json.loads(runs[0].dashboard_json)


@app.get("/workbench/stream")
def workbench_stream(
    risk_level: str = Query("C3"),
    investable: float = Query(500_000.0),
    horizon_years: int = Query(5),
    goals: str = Query("balanced_growth"),
    liquidity_min: float = Query(0.2),
    accept_cross_border: bool = Query(True),
) -> StreamingResponse:
    """SSE stream of pipeline nodes; accepts profile via query params."""
    profile = InvestorProfile(
        risk_level=risk_level,  # type: ignore[arg-type]
        investable=investable,
        horizon_years=horizon_years,
        goals=goals.split(","),
        liquidity_min=liquidity_min,
        accept_cross_border=accept_cross_border,
    )
    settings = get_settings()
    deps = build_runtime_deps(settings)
    return StreamingResponse(
        sse_events(profile, deps, settings),
        media_type="text/event-stream",
    )


@app.get("/runs")
def runs(limit: int = Query(50, ge=1, le=500)) -> dict:
    """Audit log of recent advisory runs (most recent first)."""
    return {"runs": [r.model_dump() for r in _STORE.list(limit)]}
