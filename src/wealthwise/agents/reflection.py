"""Reflection node — bounded de-risk retry after compliance verdict.

Decision logic:
- REJECT  → terminal; set status to CANNOT_ISSUE (no retry).
- DOWNGRADE + budget remains + not already retried → single retry:
    lower the risk ceiling and raise the liquidity floor in goal_constraints,
    append a note, record trace_event, increment budget_spent.
    Signal: route back to portfolio node for one re-optimisation pass.
- DOWNGRADE + budget exhausted or already retried → terminal; set status to
    NEEDS_HUMAN_REVIEW (compliance will output-guard it anyway).
- PASS → signal proceed to explanation.

The routing hint is stored in state.notes[-1] as a structured prefix so the
conditional edge can read it cheaply:
    "__route__: portfolio"   → re-run portfolio
    "__route__: explanation" → proceed
    "__route__: finalize"    → terminal (cannot issue)

The retry cap is enforced via a dedicated budget_spent counter increment plus
a check against `deps.max_llm_judgments`.  We use `budget_spent` both for
LLM calls (incremented by macro_node/compliance_node) and reflection retries
(+1 per retry attempt).  The graph's budget guard nodes check this before
heavy jury stages; reflection here checks it as a secondary guard.

IMPORTANT: `budget_spent` is also incremented by each jury call inside
`macro_node` / `compliance_node` (they don't change it; they return a delta
which LangGraph merges).  Reflection does NOT re-count those; it only adds 1
per retry iteration to signal its own cost.
"""
from __future__ import annotations

import time

from wealthwise.agents.state import AdvisoryState

# Status constants
STATUS_CANNOT_ISSUE = "CANNOT_ISSUE"
STATUS_NEEDS_REVIEW = "NEEDS_HUMAN_REVIEW"

# Routing hint prefix embedded in notes
_ROUTE_PREFIX = "__route__: "

# How much to tighten the risk ceiling on a downgrade retry:
# drop one R level (e.g. R4 → R3) and raise liquidity floor by 10 pp.
_R_DOWNGRADE_MAP = {
    "R5": "R4",
    "R4": "R3",
    "R3": "R2",
    "R2": "R1",
    "R1": "R1",  # already floor
}
_LIQUIDITY_BUMP = 0.10


def _tighten_constraints(goal_constraints: dict) -> dict:
    """Return a copy of goal_constraints with a tighter risk ceiling and higher
    liquidity floor for the de-risk retry pass."""
    gc = dict(goal_constraints)
    current_ceiling = gc.get("risk_ceiling", "R5")
    new_ceiling = _R_DOWNGRADE_MAP.get(current_ceiling, "R1")
    current_liq = float(gc.get("liquidity_min", 0.0))
    new_liq = min(1.0, current_liq + _LIQUIDITY_BUMP)
    # Also cap equity accordingly
    current_max_equity = float(gc.get("max_equity", 1.0))
    new_max_equity = max(0.0, current_max_equity - 0.10)

    gc["risk_ceiling"] = new_ceiling
    gc["liquidity_min"] = new_liq
    gc["max_equity"] = new_max_equity
    gc["_reflection_retry"] = True  # sentinel so we don't retry twice
    return gc


def reflection_node(state: AdvisoryState, deps) -> dict:
    """Post-compliance decision: route to retry, explanation, or terminal.

    Returns a state increment dict with keys:
        goal_constraints (updated on retry), notes, trace_events,
        budget_spent, status (on terminal paths).
    """
    if state.compliance is None:
        # Should not happen in normal flow, but guard gracefully
        note = f"{_ROUTE_PREFIX}explanation"
        event = {"node": "reflection", "ts": time.time(),
                 "decision": "no_compliance", "route": "explanation"}
        return {
            "notes": state.notes + [note],
            "trace_events": state.trace_events + [event],
        }

    decision = state.compliance.decision
    already_retried = state.goal_constraints.get("_reflection_retry", False)
    budget_ok = state.budget_spent < deps.max_llm_judgments

    if decision == "REJECT":
        # Hard terminal — never loop
        note = f"{_ROUTE_PREFIX}finalize"
        event = {"node": "reflection", "ts": time.time(),
                 "decision": "REJECT", "route": "finalize"}
        return {
            "status": STATUS_CANNOT_ISSUE,
            "notes": state.notes + [
                note,
                "reflection_node: REJECT decision — advisory cannot be issued",
            ],
            "trace_events": state.trace_events + [event],
        }

    if decision == "DOWNGRADE" and not already_retried and budget_ok:
        # Single de-risk retry
        new_gc = _tighten_constraints(state.goal_constraints)
        budget_delta = state.budget_spent + 1
        note = f"{_ROUTE_PREFIX}portfolio"
        event = {
            "node": "reflection",
            "ts": time.time(),
            "decision": "DOWNGRADE",
            "route": "portfolio",
            "retry": True,
            "new_risk_ceiling": new_gc["risk_ceiling"],
            "new_liquidity_min": new_gc["liquidity_min"],
        }
        return {
            "goal_constraints": new_gc,
            "budget_spent": budget_delta,
            "notes": state.notes + [
                note,
                (
                    f"reflection_node: DOWNGRADE — de-risk retry; "
                    f"new ceiling={new_gc['risk_ceiling']} "
                    f"liquidity_min={new_gc['liquidity_min']:.0%}"
                ),
            ],
            "trace_events": state.trace_events + [event],
        }

    if decision == "DOWNGRADE" and (already_retried or not budget_ok):
        # Budget exhausted or already retried — send to human review
        note = f"{_ROUTE_PREFIX}explanation"
        reason = "already_retried" if already_retried else "budget_exhausted"
        event = {"node": "reflection", "ts": time.time(),
                 "decision": "DOWNGRADE", "route": "explanation", "reason": reason}
        return {
            "status": STATUS_NEEDS_REVIEW,
            "notes": state.notes + [
                note,
                f"reflection_node: DOWNGRADE ({reason}) — flagging for human review",
            ],
            "trace_events": state.trace_events + [event],
        }

    # PASS — proceed to explanation
    note = f"{_ROUTE_PREFIX}explanation"
    event = {"node": "reflection", "ts": time.time(),
             "decision": "PASS", "route": "explanation"}
    return {
        "notes": state.notes + [note],
        "trace_events": state.trace_events + [event],
    }


def route_after_reflection(state: AdvisoryState) -> str:
    """Read the routing hint from the most recent __route__ note."""
    for note in reversed(state.notes):
        if note.startswith(_ROUTE_PREFIX):
            return note[len(_ROUTE_PREFIX):]
    return "explanation"
