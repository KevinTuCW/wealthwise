"""Advisory LangGraph state-machine — supervisor + expert pipeline.

Node topology (in execution order):
  START
  → intake          (normalize / set initial status)
  → input_guard     (screen_profile; GUARDRAIL_BLOCKED → END)
  → planner         (compute goal_constraints + planner hints)
  → budget_macro    (estimate jury calls; BUDGET_EXCEEDED → END)
  → macro           (macro_view via RAG + jury)
  → equity          (screen equity candidates)
  → cap             (process guardrail: dedupe + truncate)
  → portfolio       (optimize allocation) ←──────────────────┐
  → budget_compliance (estimate jury calls; BUDGET_EXCEEDED → END)  │
  → compliance      (suitability + jury)                        │
  → reflection      (PASS → explanation                         │
                     DOWNGRADE + budget → portfolio ────────────┘  (single retry)
                     DOWNGRADE exhausted / REJECT → finalize)
  → explanation     (deterministic template: compose advisory text)
  → output_guard    (enforce_output disclosure checks)
  → END

Budget guard nodes check projected jury calls against deps.max_llm_judgments
BEFORE the heavy jury-invoking nodes (macro, compliance) run.  If the projected
total would exceed the cap, the pipeline terminates with BUDGET_EXCEEDED.

Every node wraps its body with `traced(...)` for Langfuse instrumentation
(no-op when tracing is disabled offline).

Deps are bound at graph-build time via closures (same pattern as shopscout).
"""
from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from wealthwise.agents.deps import AdvisoryDeps
from wealthwise.agents.experts.compliance import compliance_node
from wealthwise.agents.experts.equity import equity_node
from wealthwise.agents.experts.goal import goal_node
from wealthwise.agents.experts.macro import macro_node
from wealthwise.agents.experts.portfolio import portfolio_node
from wealthwise.agents.reflection import reflection_node, route_after_reflection
from wealthwise.agents.state import AdvisoryState
from wealthwise.agents.supervisor.planner import build_planner_hints, plan
from wealthwise.guardrails.input import screen_profile
from wealthwise.guardrails.output import STATUS_NEEDS_REVIEW, enforce_output
from wealthwise.guardrails.process import cap_candidates
from wealthwise.obs import traced

# Status constants
_STATUS_GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"
_STATUS_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
_STATUS_DONE = "done"

# Each heavy node costs one deliberation across the whole jury.
#
# Derived from the injected jury rather than hard-coded: the constant used to be
# a literal 2, with the comment "2 clients × 1 call". Adding a third juror would
# have left every budget check under-estimating the real spend by a third — a
# cost guard that silently stops matching the cost is worse than no guard.
_JURY_DELIBERATIONS_MACRO = 1
_JURY_DELIBERATIONS_COMPLIANCE = 1


def _jury_calls(deps, deliberations: int = 1) -> int:
    """Model calls one deliberation costs: one per juror."""
    return max(1, len(deps.jury_clients)) * deliberations


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

def _append_event(state: AdvisoryState, node: str, status: str = "OK",
                  extra: dict | None = None) -> list[dict]:
    """Return a new trace_events list with a new event appended."""
    event: dict = {
        "node": node,
        "ts": time.time(),
        "status": status,
        "budget_spent": state.budget_spent,
    }
    if extra:
        event.update(extra)
    return [*state.trace_events, event]


# ---------------------------------------------------------------------------
# Graph node factories (closure-bind deps)
# ---------------------------------------------------------------------------

@traced("wealthwise.node.intake")
def _intake_node(state: AdvisoryState) -> dict:
    """Normalize the incoming state and mark pipeline as running."""
    event = _append_event(state, "intake")
    return {
        "status": "running",
        "trace_events": event,
    }


@traced("wealthwise.node.input_guard")
def _input_guard_node(state: AdvisoryState) -> dict:
    """Screen the investor profile; route to END on failure."""
    if state.profile is None:
        notes = [*state.notes, "input_guard: profile is None — blocking"]
        ev_state = state.model_copy(update={"status": _STATUS_GUARDRAIL_BLOCKED, "notes": notes})
        return {
            "status": _STATUS_GUARDRAIL_BLOCKED,
            "notes": notes,
            "trace_events": _append_event(ev_state, "input_guard", _STATUS_GUARDRAIL_BLOCKED),
        }

    ok, reasons = screen_profile(state.profile)
    if not ok:
        msg = "; ".join(reasons)
        notes = [*state.notes, f"input_guard: {msg}"]
        ev_state = state.model_copy(update={"status": _STATUS_GUARDRAIL_BLOCKED, "notes": notes})
        return {
            "status": _STATUS_GUARDRAIL_BLOCKED,
            "notes": notes,
            "trace_events": _append_event(ev_state, "input_guard", _STATUS_GUARDRAIL_BLOCKED,
                                          {"reasons": reasons}),
        }

    ev_state = state.model_copy(update={"status": "running"})
    return {
        "status": "running",
        "trace_events": _append_event(ev_state, "input_guard"),
    }


def _route_after_input_guard(state: AdvisoryState) -> str:
    return "end" if state.status == _STATUS_GUARDRAIL_BLOCKED else "planner"


def _make_planner_node():
    @traced("wealthwise.node.planner")
    def planner_node(state: AdvisoryState) -> dict:
        """Derive planner hints and merge into goal_constraints."""
        node_plan = plan(state.profile)
        hints = build_planner_hints(state.profile)

        # Merge hints into existing goal_constraints (goal_node hasn't run yet;
        # we store hints for goal_node to pick up and for audit trail).
        updated_gc = {**state.goal_constraints, "planner_hints": hints}

        note = (
            f"planner_node: plan={node_plan}; "
            f"conservative_mode={hints['conservative_mode']}; "
            f"skip_cross_border={hints['skip_cross_border']}"
        )
        event = {
            "node": "planner",
            "ts": time.time(),
            "plan": node_plan,
            "conservative_mode": hints["conservative_mode"],
        }
        return {
            "goal_constraints": updated_gc,
            "notes": state.notes + [note],
            "trace_events": state.trace_events + [event],
        }
    return planner_node


def _make_budget_node(name: str, estimated_calls: int, deps: AdvisoryDeps):
    """Factory: budget guard node that checks projected jury calls before a heavy node."""
    @traced(f"wealthwise.node.budget_guard.{name}")
    def budget_node(state: AdvisoryState) -> dict:
        projected = state.budget_spent + estimated_calls
        if projected > deps.max_llm_judgments:
            notes = [
                *state.notes,
                f"budget_guard: blocked {name}: "
                f"{projected} > {deps.max_llm_judgments} estimated judgment calls",
            ]
            ev_state = state.model_copy(update={"status": _STATUS_BUDGET_EXCEEDED,
                                                "notes": notes})
            return {
                "status": _STATUS_BUDGET_EXCEEDED,
                "notes": notes,
                "trace_events": _append_event(ev_state, "budget_guard",
                                              _STATUS_BUDGET_EXCEEDED,
                                              {"stage": name,
                                               "estimated_calls": estimated_calls}),
            }
        ev_state = state.model_copy(update={"budget_spent": projected})
        return {
            "budget_spent": projected,
            "trace_events": _append_event(ev_state, "budget_guard", "OK",
                                          {"stage": name,
                                           "estimated_calls": estimated_calls}),
        }
    budget_node.__name__ = f"budget_{name}"
    return budget_node


def _route_after_budget(state: AdvisoryState) -> str:
    return "end" if state.status == _STATUS_BUDGET_EXCEEDED else "continue"


def _make_expert_node(expert_fn, deps: AdvisoryDeps, node_name: str):
    """Wrap an (state, deps)->dict expert function as a traced LangGraph node."""
    @traced(f"wealthwise.node.{node_name}")
    def node(state: AdvisoryState) -> dict:
        result = expert_fn(state, deps)
        # Append a node-level trace event on top of whatever the expert emitted
        merged_trace = result.get("trace_events", state.trace_events)
        ev_state = state.model_copy(update={**result, "trace_events": merged_trace})
        trace = _append_event(ev_state, f"{node_name}_done", "OK")
        return {**result, "trace_events": trace}
    node.__name__ = node_name
    return node


def _make_goal_node(deps: AdvisoryDeps):
    @traced("wealthwise.node.goal")
    def node(state: AdvisoryState) -> dict:
        result = goal_node(state, deps)
        # After goal_node runs, merge the planner hints into goal_constraints
        # (goal_node replaces goal_constraints wholesale; re-attach hints)
        hints = state.goal_constraints.get("planner_hints", {})
        if hints:
            gc = dict(result.get("goal_constraints", {}))
            gc["planner_hints"] = hints
            result = {**result, "goal_constraints": gc}
        return result
    return node


def _make_macro_node(deps: AdvisoryDeps):
    return _make_expert_node(macro_node, deps, "macro")


def _make_equity_node(deps: AdvisoryDeps):
    @traced("wealthwise.node.equity")
    def node(state: AdvisoryState) -> dict:
        result = equity_node(state, deps)
        merged_trace = result.get("trace_events", state.trace_events)
        ev_state = state.model_copy(update={**result, "trace_events": merged_trace})
        trace = _append_event(ev_state, "equity_done", "OK")
        return {**result, "trace_events": trace}
    return node


def _make_cap_node(deps: AdvisoryDeps):
    @traced("wealthwise.node.cap")
    def node(state: AdvisoryState) -> dict:
        all_candidates = list(state.equity_candidates) + list(state.fixedincome_candidates)
        cleaned = cap_candidates(all_candidates, max_candidates=50)

        # Split back into equity vs fixed-income
        equity_cleaned = [c for c in cleaned if c.asset_class == "equity"]
        fi_cleaned = [c for c in cleaned if c.asset_class != "equity"]

        note = (
            f"cap_node: {len(all_candidates)} → {len(cleaned)} candidates "
            f"({len(equity_cleaned)} equity, {len(fi_cleaned)} fi/cash)"
        )
        ev_state = state.model_copy(update={
            "equity_candidates": equity_cleaned,
            "fixedincome_candidates": fi_cleaned,
        })
        return {
            "equity_candidates": equity_cleaned,
            "fixedincome_candidates": fi_cleaned,
            "notes": state.notes + [note],
            "trace_events": _append_event(ev_state, "cap"),
        }
    return node


def _make_portfolio_node(deps: AdvisoryDeps):
    @traced("wealthwise.node.portfolio")
    def node(state: AdvisoryState) -> dict:
        result = portfolio_node(state, deps)
        merged_trace = result.get("trace_events", state.trace_events)
        ev_state = state.model_copy(update={**result, "trace_events": merged_trace})
        trace = _append_event(ev_state, "portfolio_done", "OK")
        return {**result, "trace_events": trace}
    return node


def _make_compliance_node(deps: AdvisoryDeps):
    @traced("wealthwise.node.compliance")
    def node(state: AdvisoryState) -> dict:
        result = compliance_node(state, deps)
        merged_trace = result.get("trace_events", state.trace_events)
        ev_state = state.model_copy(update={**result, "trace_events": merged_trace})
        trace = _append_event(ev_state, "compliance_done", "OK")
        return {**result, "trace_events": trace}
    return node


def _make_reflection_node(deps: AdvisoryDeps):
    @traced("wealthwise.node.reflection")
    def node(state: AdvisoryState) -> dict:
        return reflection_node(state, deps)
    return node


def _make_explanation_node():
    """Deterministic template: compose the human-readable advisory explanation."""

    # Terminal statuses that must not be overwritten with 'done'.
    _TERMINAL_REVIEW_STATUSES = frozenset({
        "NEEDS_HUMAN_REVIEW", "CANNOT_ISSUE", "GUARDRAIL_BLOCKED", "BUDGET_EXCEEDED",
    })

    @traced("wealthwise.node.explanation")
    def node(state: AdvisoryState) -> dict:
        profile = state.profile
        portfolio = state.portfolio
        compliance = state.compliance
        gc = state.goal_constraints

        if profile is None or portfolio is None:
            explanation = "本次咨询因输入不完整，无法生成投资建议。不构成投资建议。"
            confidence = 0.0
        else:
            # Build readable allocation summary (top 5 positions)
            sorted_weights = sorted(portfolio.weights.items(),
                                    key=lambda x: x[1], reverse=True)
            top5 = sorted_weights[:5]
            alloc_lines = ", ".join(f"{sym}: {w:.1%}" for sym, w in top5)
            if len(sorted_weights) > 5:
                alloc_lines += f" 及其他 {len(sorted_weights) - 5} 个标的"

            risk_ceiling = gc.get("risk_ceiling", portfolio.portfolio_r_level)
            decision = compliance.decision if compliance else "N/A"

            # Compose disclosures from the real compliance output (not static strings).
            # This ensures the output guard validates substantive compliance disclosures.
            disclosures_text = (
                "\n".join(compliance.disclosures) if compliance and compliance.disclosures
                else "不构成投资建议。"
            )

            explanation = (
                f"根据您的风险承受等级（{profile.risk_level}）和投资目标，"
                f"本系统建议以下资产配置方案（风险等级 {portfolio.portfolio_r_level}，"
                f"上限 {risk_ceiling}）：\n"
                f"主要持仓：{alloc_lines}。\n"
                f"投资组合境外敞口：{portfolio.fx_exposure:.1%}。\n"
                f"合规审核结论：{decision}。\n"
                f"{disclosures_text}"
            )
            # Confidence: compliance confidence × portfolio sharpe proxy
            base_conf = compliance.confidence if compliance else 0.5
            confidence = round(base_conf * 0.9, 3)  # slight discount for model uncertainty

        note = f"explanation_node: composed {len(explanation)} chars; confidence={confidence}"
        event = {"node": "explanation", "ts": time.time(),
                 "len_explanation": len(explanation), "confidence": confidence}

        # Only advance status to 'done' if the current status is not already a
        # terminal review/blocked status (e.g. NEEDS_HUMAN_REVIEW set by reflection
        # when DOWNGRADE was exhausted).
        new_status = (
            state.status if state.status in _TERMINAL_REVIEW_STATUSES
            else _STATUS_DONE
        )

        return {
            "explanation": explanation,
            "confidence": confidence,
            "status": new_status,
            "notes": state.notes + [note],
            "trace_events": state.trace_events + [event],
        }
    return node


def _make_output_guard_node():
    @traced("wealthwise.node.output_guard")
    def node(state: AdvisoryState) -> dict:
        enforced = enforce_output(state)
        note = f"output_guard: status={enforced.status}"
        event = {"node": "output_guard", "ts": time.time(),
                 "status": enforced.status}
        return {
            "status": enforced.status,
            "explanation": enforced.explanation,
            "notes": enforced.notes + [note],
            "trace_events": enforced.trace_events + [event],
        }
    return node


def _make_finalize_node():
    """Terminal node for REJECT and CANNOT_ISSUE paths."""
    @traced("wealthwise.node.finalize")
    def node(state: AdvisoryState) -> dict:
        note = f"finalize_node: pipeline terminated; status={state.status}"
        event = {"node": "finalize", "ts": time.time(), "status": state.status}
        return {
            "notes": state.notes + [note],
            "trace_events": state.trace_events + [event],
        }
    return node


# ---------------------------------------------------------------------------
# Routing helpers for reflection
# ---------------------------------------------------------------------------

def _route_after_reflection_state(state: AdvisoryState) -> str:
    """Read routing hint from the last __route__ note in state."""
    route = route_after_reflection(state)
    if route == "portfolio":
        return "portfolio"
    if route == "finalize":
        return "finalize"
    return "explanation"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(deps: AdvisoryDeps):
    """Compile the advisory pipeline as a LangGraph state machine.

    Parameters
    ----------
    deps:
        AdvisoryDeps — providers and jury clients bound into all node closures.

    Returns
    -------
    A compiled LangGraph StateGraph ready to .invoke().
    """
    g = StateGraph(AdvisoryState)

    # Register nodes
    g.add_node("intake", _intake_node)
    g.add_node("input_guard", _input_guard_node)
    g.add_node("planner", _make_planner_node())
    g.add_node("budget_macro", _make_budget_node(
        "macro", _jury_calls(deps, _JURY_DELIBERATIONS_MACRO), deps))
    g.add_node("goal", _make_goal_node(deps))
    g.add_node("macro", _make_macro_node(deps))
    g.add_node("equity", _make_equity_node(deps))
    g.add_node("cap", _make_cap_node(deps))
    g.add_node("portfolio", _make_portfolio_node(deps))
    g.add_node("budget_compliance", _make_budget_node(
        "compliance", _jury_calls(deps, _JURY_DELIBERATIONS_COMPLIANCE), deps))
    g.add_node("compliance", _make_compliance_node(deps))
    g.add_node("reflection", _make_reflection_node(deps))
    g.add_node("explanation", _make_explanation_node())
    g.add_node("output_guard", _make_output_guard_node())
    g.add_node("finalize", _make_finalize_node())

    # Wire edges
    g.add_edge(START, "intake")
    g.add_edge("intake", "input_guard")
    g.add_conditional_edges(
        "input_guard",
        _route_after_input_guard,
        {"planner": "planner", "end": END},
    )
    g.add_edge("planner", "budget_macro")
    g.add_conditional_edges(
        "budget_macro",
        _route_after_budget,
        {"continue": "goal", "end": END},
    )
    g.add_edge("goal", "macro")
    g.add_edge("macro", "equity")
    g.add_edge("equity", "cap")
    g.add_edge("cap", "portfolio")
    g.add_edge("portfolio", "budget_compliance")
    g.add_conditional_edges(
        "budget_compliance",
        _route_after_budget,
        {"continue": "compliance", "end": END},
    )
    g.add_edge("compliance", "reflection")
    g.add_conditional_edges(
        "reflection",
        _route_after_reflection_state,
        {
            "portfolio": "portfolio",
            "explanation": "explanation",
            "finalize": "finalize",
        },
    )
    g.add_edge("explanation", "output_guard")
    g.add_edge("output_guard", END)
    g.add_edge("finalize", END)

    return g.compile()
