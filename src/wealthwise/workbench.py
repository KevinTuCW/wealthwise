"""Advisory workbench — dashboard builder + SSE streaming.

Five-panel dashboard surfaced from a completed AdvisoryState:
  1. allocation  — class_weights, per-symbol weights, portfolio metrics, r_level
  2. experts     — each expert agent's contribution (macro/equity/portfolio/compliance)
  3. crosscheck  — multi-source consensus + multi-model jury agreement signals
  4. compliance  — C×R suitability matrix, misleading-term hits, disclosure checklist
  5. cost        — tokens, cost_usd, node latencies from trace_events

SSE streaming via sse_events(): yields start/node/complete events while the
LangGraph pipeline runs, so the frontend can animate the pipeline in real time.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi.encoders import jsonable_encoder

from wealthwise.agents.deps import AdvisoryDeps
from wealthwise.agents.state import AdvisoryState, InvestorProfile
from wealthwise.agents.supervisor.graph import build_graph
from wealthwise.compliance.suitability import C_ORDER
from wealthwise.guardrails.output import has_complete_disclosures
from wealthwise.portfolio.metrics import R_ORDER


# ---------------------------------------------------------------------------
# Cost helper
# ---------------------------------------------------------------------------

def _cost_usd(tokens: int, settings) -> float:
    price = getattr(settings, "token_price_per_1k", 0.0002)
    return round(tokens / 1000 * price, 6)


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _build_allocation_panel(state: AdvisoryState) -> dict:
    """Panel 1: allocation — class_weights, per-symbol weights, portfolio metrics."""
    portfolio = state.portfolio
    if portfolio is None:
        return {
            "portfolio_r_level": "",
            "class_weights": {},
            "weights": {},
            "metrics": {},
            "fx_exposure": 0.0,
        }
    return {
        "portfolio_r_level": portfolio.portfolio_r_level,
        "class_weights": portfolio.class_weights,
        "weights": portfolio.weights,
        "fx_exposure": round(portfolio.fx_exposure, 4),
        "metrics": {
            # Standard portfolio risk metrics; may be empty if optimizer didn't fill them
            "volatility": portfolio.metrics.get("volatility", portfolio.metrics.get("vol")),
            "max_drawdown": portfolio.metrics.get("max_drawdown", portfolio.metrics.get("mdd")),
            "sharpe": portfolio.metrics.get("sharpe"),
            "diversification": portfolio.metrics.get("diversification"),
        },
    }


def _build_experts_panel(state: AdvisoryState) -> dict:
    """Panel 2: each expert's contribution pulled from state fields."""
    # Macro
    macro = state.macro_view
    macro_summary = {
        "tilt": macro.get("tilt", macro.get("equity_tilt", "")),
        "confidence": macro.get("confidence"),
        "regime": macro.get("regime", ""),
        "signals": macro.get("signals", []),
    }

    # Equity candidates
    equity_count = len(state.equity_candidates)
    fi_count = len(state.fixedincome_candidates)
    # Summarise top candidates for display
    equity_top = [
        {"symbol": c.symbol, "name": c.name, "r_level": c.r_level, "market": c.market}
        for c in state.equity_candidates[:5]
    ]
    fi_top = [
        {"symbol": c.symbol, "name": c.name, "r_level": c.r_level, "market": c.market}
        for c in state.fixedincome_candidates[:5]
    ]

    # Portfolio construction summary
    portfolio = state.portfolio
    port_summary: dict = {}
    if portfolio:
        port_summary = {
            "portfolio_r_level": portfolio.portfolio_r_level,
            "fx_exposure": round(portfolio.fx_exposure, 4),
            "n_positions": len([w for w in portfolio.weights.values() if w > 0]),
        }

    # Compliance summary
    compliance = state.compliance
    comp_summary: dict = {}
    if compliance:
        comp_summary = {
            "decision": compliance.decision,
            "matched": compliance.matched,
            "confidence": compliance.confidence,
            "violations": compliance.violations,
        }

    # Extract per-node timings from trace_events
    node_latencies: dict[str, float] = {}
    prev_ts: float | None = None
    for ev in state.trace_events:
        ts = ev.get("ts", 0.0)
        node = ev.get("node", "")
        if prev_ts is not None and node:
            node_latencies[node] = round(ts - prev_ts, 3)
        prev_ts = ts

    return {
        "macro": macro_summary,
        "equity": {
            "candidate_count": equity_count,
            "top_candidates": equity_top,
        },
        "fixed_income": {
            "candidate_count": fi_count,
            "top_candidates": fi_top,
        },
        "portfolio_construction": port_summary,
        "compliance": comp_summary,
        "node_latencies": node_latencies,
    }


def _build_crosscheck_panel(state: AdvisoryState) -> dict:
    """Panel 3: multi-source consensus + multi-model jury agreement signals."""
    # Extract crosscheck / jury signals from trace_events
    jury_events = [e for e in state.trace_events if "jury" in e.get("node", "").lower()
                   or "crosscheck" in e.get("node", "").lower()]
    compliance_events = [e for e in state.trace_events if "compliance" in e.get("node", "")]

    # Macro tilt consensus (from macro_view)
    macro = state.macro_view
    macro_tilt = macro.get("tilt", macro.get("equity_tilt", "neutral"))
    macro_confidence = macro.get("confidence")

    # Jury/model agreement: look for jury sub-events
    jury_votes: list[dict] = []
    for ev in state.trace_events:
        if "votes" in ev:
            jury_votes = ev["votes"]
            break
        if "jury" in ev.get("node", "").lower() and "label" in ev:
            jury_votes.append({"node": ev["node"], "label": ev.get("label", "")})

    # Agreement signal
    labels = [v.get("label", "") for v in jury_votes if v.get("label")]
    unique_labels = set(labels)
    agreement = "unanimous" if len(unique_labels) <= 1 and labels else "split" if len(unique_labels) > 1 else "n/a"

    # Escalation signals
    escalation_signals: list[str] = []
    if state.compliance and state.compliance.decision in {"REJECT", "DOWNGRADE"}:
        escalation_signals.append(f"compliance:{state.compliance.decision}")
    for ev in state.trace_events:
        if ev.get("status") in {"BUDGET_EXCEEDED", "GUARDRAIL_BLOCKED"}:
            escalation_signals.append(ev["status"])

    return {
        "macro_tilt": macro_tilt,
        "macro_confidence": macro_confidence,
        "jury_votes": jury_votes,
        "agreement": agreement,
        "escalation_signals": escalation_signals,
        "jury_event_count": len(jury_events),
        "compliance_event_count": len(compliance_events),
    }


def _build_compliance_panel(state: AdvisoryState) -> dict:
    """Panel 4: C×R suitability matrix, disclosure checklist, misleading-term hits."""
    profile = state.profile
    portfolio = state.portfolio
    compliance = state.compliance

    # --- C×R suitability matrix ---
    # Rows: C-levels (C1..C5); columns: portfolio R-level + per-asset R-levels
    c_levels = ["C1", "C2", "C3", "C4", "C5"]
    r_levels_used: list[str] = []
    if portfolio:
        r_levels_used.append(portfolio.portfolio_r_level)
    all_candidates = list(state.equity_candidates) + list(state.fixedincome_candidates)
    for c in all_candidates:
        if c.r_level not in r_levels_used:
            r_levels_used.append(c.r_level)
    # Sort R-levels by order
    r_levels_used = sorted(set(r_levels_used), key=lambda r: R_ORDER.get(r, 0))

    cr_matrix: list[dict] = []
    for c in c_levels:
        row: dict = {"c_level": c}
        for r in r_levels_used:
            c_num = C_ORDER.get(c, 0)
            r_num = R_ORDER.get(r, 0)
            cell = "suitable" if r_num <= c_num else "over-level"
            row[r] = cell
        cr_matrix.append(row)

    # Highlight the investor's own C-level row
    investor_c = profile.risk_level if profile else None
    portfolio_r = portfolio.portfolio_r_level if portfolio else None

    # --- Disclosure checklist ---
    ok, missing = has_complete_disclosures(state)
    # Build a structured checklist from the four required disclosures
    disclosures_raw = compliance.disclosures if compliance else []
    combined = " ".join(disclosures_raw).lower()

    def _check(keywords: list[str], field: str) -> dict:
        present = any(kw in combined for kw in keywords)
        return {"field": field, "present": present}

    checklist = [
        _check(["适当性", "suitability", "匹配", "match"], "suitability_match"),
        _check(["投资有风险", "入市须谨慎", "过往业绩", "risk disclosure"], "risk_disclosure"),
        _check(["不构成投资建议"], "not_advice_disclaimer"),
        _check(["汇率"], "fx_cross_border_disclosure"),
    ]

    # --- Violations ---
    violations = compliance.violations if compliance else []

    return {
        "investor_c_level": investor_c,
        "portfolio_r_level": portfolio_r,
        "cr_matrix": cr_matrix,
        "r_levels_used": r_levels_used,
        "disclosure_checklist": checklist,
        "disclosures_complete": ok,
        "disclosures_missing": missing,
        "violations": violations,
        "compliance_decision": compliance.decision if compliance else "",
        "compliance_matched": compliance.matched if compliance else None,
    }


def _build_cost_panel(state: AdvisoryState, settings) -> dict:
    """Panel 5: tokens_used, cost_usd, node latency/count from trace_events."""
    tokens = state.tokens_used
    cost = _cost_usd(tokens, settings)

    # Node latencies: compute per-node elapsed time from sequential trace events
    node_timings: list[dict] = []
    prev_ts: float | None = None
    prev_node: str = ""
    for ev in state.trace_events:
        ts = ev.get("ts", 0.0)
        node = ev.get("node", "")
        if prev_ts is not None and node:
            elapsed = round(ts - prev_ts, 3)
            node_timings.append({"node": prev_node or node, "latency_s": elapsed})
        prev_ts = ts
        prev_node = node

    # Summary counts
    node_names = [e.get("node", "") for e in state.trace_events if e.get("node")]
    unique_nodes = list(dict.fromkeys(node_names))  # ordered dedup

    return {
        "tokens_used": tokens,
        "cost_usd": cost,
        "budget_spent": state.budget_spent,
        "node_count": len(unique_nodes),
        "unique_nodes": unique_nodes,
        "node_timings": node_timings,
        "trace_event_count": len(state.trace_events),
    }


# ---------------------------------------------------------------------------
# Main dashboard builder
# ---------------------------------------------------------------------------

def build_dashboard(state: AdvisoryState, settings) -> dict:
    """Build the five-panel dashboard from a completed AdvisoryState.

    Parameters
    ----------
    state:
        Completed AdvisoryState (from run_advisory or from a streamed run).
    settings:
        Settings instance (for token_price_per_1k).

    Returns
    -------
    dict with keys: allocation, experts, crosscheck, compliance, cost.
    """
    return {
        "status": state.status,
        "allocation": _build_allocation_panel(state),
        "experts": _build_experts_panel(state),
        "crosscheck": _build_crosscheck_panel(state),
        "compliance": _build_compliance_panel(state),
        "cost": _build_cost_panel(state, settings),
    }


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

def sse_events(profile: InvestorProfile | None, deps: AdvisoryDeps,
               settings=None) -> Iterator[str]:
    """Yield SSE-formatted events while the advisory pipeline runs.

    Events emitted:
      - "start"    — immediately, carrying the profile summary.
      - "node"     — once per LangGraph node as it completes; carries node
                     name + state patch.
      - "complete" — final event, carrying the full five-panel dashboard.

    Each event is formatted as:  event: <name>\\ndata: <json>\\n\\n
    (matching shopscout's SSE framing exactly).
    """
    if settings is None:
        from wealthwise.config import get_settings
        settings = get_settings()

    graph = build_graph(deps)
    initial = AdvisoryState(profile=profile)

    profile_desc = ""
    if profile:
        profile_desc = (
            f"{profile.risk_level} | "
            f"{profile.investable:,.0f} CNY | "
            f"{profile.horizon_years}y"
        )

    yield _sse("start", {"profile": profile_desc})

    # Accumulate state patches so we can reconstruct the final state
    state_data = initial.model_dump()

    for update in graph.stream(initial, stream_mode="updates"):
        for node, patch in update.items():
            payload = {"node": node, **_safe_patch(patch)}
            yield _sse("node", payload)
            # Merge patch into state_data (handle list fields by replacement)
            if isinstance(patch, dict):
                state_data.update(patch)

    # Reconstruct final state from accumulated patches
    final = AdvisoryState.model_validate(state_data)
    yield _sse("complete", build_dashboard(final, settings))


def _safe_patch(patch: dict) -> dict:
    """Return a JSON-serialisable subset of a state patch (drop large blobs)."""
    safe: dict = {}
    for k, v in patch.items():
        if k in ("trace_events", "equity_candidates", "fixedincome_candidates"):
            # Summarise rather than dump full lists
            if isinstance(v, list):
                safe[k + "_count"] = len(v)
            continue
        safe[k] = v
    return safe


def _sse(event: str, data: dict) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(jsonable_encoder(data), ensure_ascii=False)}\n\n"
    )
