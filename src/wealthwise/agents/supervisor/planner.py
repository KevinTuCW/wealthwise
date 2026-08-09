"""Supervisor planner — decide expert ordering / skipping based on investor profile.

Deterministic, rule-based: no LLM call.  Returns an ordered list of expert
node names for the graph to execute.  The graph itself hard-wires the full
sequence; this planner's output is recorded on AdvisoryState.notes for
auditability and used only to annotate goal_constraints with planner hints.

Skipping rules
--------------
C1 (capital-preservation):
    - Skip deep equity screening (equity node still runs but the risk ceiling
      means virtually no equity candidates will survive; we mark the profile
      so downstream nodes short-circuit HK/US emphasis).
accept_cross_border=False:
    - Mark no_cross_border=True so equity_node, portfolio_node, and compliance
      omit HK/US emphasis from their logic.

The returned list is informational and used for trace/notes; the actual graph
topology is static (nodes always run in fixed order); behavior is shaped by the
goal_constraints dict that the planner augments.
"""
from __future__ import annotations

from wealthwise.agents.state import InvestorProfile

# Canonical expert execution order
_FULL_ORDER = ["goal", "macro", "equity", "cap", "portfolio", "compliance",
               "reflection", "explanation"]

# Profiles that warrant a note about reduced equity emphasis
_CONSERVATIVE_LEVELS = frozenset({"C1", "C2"})


def plan(profile: InvestorProfile | None) -> list[str]:
    """Return the ordered list of expert node names for *profile*.

    The graph topology is static (nodes always execute in this order); the
    planner's value is in the ``build_planner_hints`` dict that it writes into
    ``goal_constraints``, which downstream nodes (equity_node, goal_node) then
    consume to tighten screening for conservative profiles.

    An empty/None profile signals that the input guard will block the pipeline;
    the returned plan is still the full order (the graph routes to END via the
    input guard node, so the plan is never executed).

    Parameters
    ----------
    profile:
        InvestorProfile to plan for, or None.

    Returns
    -------
    list[str]
        Ordered expert node names (always the full static order).
    """
    return list(_FULL_ORDER)


def build_planner_hints(profile: InvestorProfile | None) -> dict:
    """Derive a planner_hints sub-dict to merge into goal_constraints.

    Keys
    ----
    skip_cross_border : bool
        True when accept_cross_border is False.
    conservative_mode : bool
        True for C1/C2 profiles — equity cap floor is already handled by
        goal_node, but we flag it here for transparency.
    plan : list[str]
        The ordered node list for audit trail.
    """
    if profile is None:
        return {
            "skip_cross_border": False,
            "conservative_mode": False,
            "plan": list(_FULL_ORDER),
        }

    skip_cross_border = not profile.accept_cross_border
    conservative_mode = profile.risk_level in _CONSERVATIVE_LEVELS

    return {
        "skip_cross_border": skip_cross_border,
        "conservative_mode": conservative_mode,
        "plan": list(_FULL_ORDER),
    }
