"""portfolio_node — combine equity + fixed-income candidates and optimize.

Screens fixed-income/cash candidates, combines with state.equity_candidates,
and calls build_portfolio() to produce a PortfolioAllocation within the risk
ceiling from goal_constraints.  Deterministic (no LLM).
"""
from __future__ import annotations

import time
from dataclasses import asdict

from wealthwise.agents.state import AdvisoryState
from wealthwise.portfolio.execution import build_execution_plan, realised_allocation
from wealthwise.portfolio.guidance import build_guidance
from wealthwise.portfolio.optimize import build_portfolio

# Fixed-income / cash markets and asset classes to screen
_FIXED_INCOME_CLASSES = ["bond", "cash"]


def portfolio_node(state: AdvisoryState, deps) -> dict:
    """Build PortfolioAllocation from equity + fixed-income candidates.

    Parameters
    ----------
    state:
        AdvisoryState — must have equity_candidates and goal_constraints set.
    deps:
        AdvisoryDeps — uses .market for fixed-income screening.
        If None (test helper path), uses only equity_candidates already in state.

    Returns
    -------
    dict
        State increment with keys: portfolio, fixedincome_candidates,
        tokens_used, trace_events, notes.
    """
    gc = state.goal_constraints
    risk_ceiling = gc.get("risk_ceiling", "R5")

    # Screen fixed-income / cash candidates from the A market
    fi_candidates = []
    if deps is not None:
        for asset_class in _FIXED_INCOME_CLASSES:
            batch = deps.market.screen("A", {"asset_class": asset_class})
            fi_candidates.extend(batch)
    else:
        fi_candidates = list(state.fixedincome_candidates)

    # Combine all candidates (equity + fixed-income)
    all_candidates = list(state.equity_candidates) + fi_candidates

    # Build portfolio
    portfolio = build_portfolio(
        candidates=all_candidates,
        goal_constraints=gc,
        risk_ceiling=risk_ceiling,
        method=gc.get("risk_budget_method", "risk_parity"),
    )

    # Continuous weights are a target, not an order. Convert to whole lots and
    # report what the investor will actually hold: compliance runs downstream of
    # this, and approving weights nobody can execute is an audit gap.
    plan_dict: dict = {}
    investable = state.profile.investable if state.profile else 0.0
    if investable > 0 and portfolio.weights:
        plan = build_execution_plan(portfolio, all_candidates, investable)
        if plan.positions:
            portfolio = realised_allocation(portfolio, plan)
            plan_dict = {
                "positions": [asdict(pos) for pos in plan.positions],
                "cash_residual": round(plan.cash_residual, 2),
                "invested": round(plan.invested, 2),
                "investable": investable,
                "dropped": plan.dropped,
                "guidance": build_guidance(state.profile, portfolio, plan),
            }

    event = {
        "node": "portfolio",
        "ts": time.time(),
        "n_equity": len(state.equity_candidates),
        "n_fi": len(fi_candidates),
        "n_total": len(all_candidates),
        "portfolio_r_level": portfolio.portfolio_r_level,
        "fx_exposure": portfolio.fx_exposure,
        "n_weights": len(portfolio.weights),
    }
    note = (
        f"portfolio_node: {len(state.equity_candidates)} equity + {len(fi_candidates)} "
        f"fi/cash → {len(portfolio.weights)} positions; "
        f"portfolio_r_level={portfolio.portfolio_r_level}; "
        f"fx_exposure={portfolio.fx_exposure:.1%}"
    )

    return {
        "portfolio": portfolio,
        "fixedincome_candidates": fi_candidates,
        "execution_plan": plan_dict,
        "trace_events": state.trace_events + [event],
        "notes": state.notes + [note],
    }
