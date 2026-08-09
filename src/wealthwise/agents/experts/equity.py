"""equity_node — screen equity candidates using goal_constraints + macro_view.

Screens across A / HK / US markets using goal_constraints (risk_ceiling,
accept_cross_border) and macro_view tilt to filter and annotate candidates.
No LLM call needed here — purely deterministic rule-based screening.
"""
from __future__ import annotations

import time

from wealthwise.agents.state import AdvisoryState, AssetCandidate
from wealthwise.portfolio.metrics import R_ORDER

# ---------------------------------------------------------------------------
# Markets to consider and their PE-filter heuristics
# ---------------------------------------------------------------------------

_ALL_MARKETS = ["A", "HK", "US"]

# When macro tilt is 'underweight' equity, apply a tighter PE cap to screen
# out expensive names; neutral/overweight use a generous cap (no hard filter).
_PE_CAP_UNDERWEIGHT = 35.0
_PE_CAP_DEFAULT = 9999.0   # effectively no PE filter


def equity_node(state: AdvisoryState, deps) -> dict:
    """Screen equity candidates from market providers.

    Parameters
    ----------
    state:
        AdvisoryState — must have goal_constraints set (from goal_node).
    deps:
        AdvisoryDeps — uses .market.  (deps may be None in test helper calls
        when the caller manages providers directly; if so we skip screening and
        return whatever equity_candidates is already in state.)

    Returns
    -------
    dict
        State increment with keys: equity_candidates, trace_events, notes.
    """
    gc = state.goal_constraints
    risk_ceiling = gc.get("risk_ceiling", "R5")
    accept_cross_border = gc.get("accept_cross_border", True)
    ceiling_order = R_ORDER.get(risk_ceiling, 5)

    # Determine markets to screen
    if accept_cross_border:
        markets = _ALL_MARKETS
    else:
        markets = ["A"]

    # Determine PE filter from macro tilt
    tilt = state.macro_view.get("tilt", "neutral") if state.macro_view else "neutral"
    pe_cap = _PE_CAP_UNDERWEIGHT if tilt == "underweight" else _PE_CAP_DEFAULT

    # Screen from providers
    candidates: list[AssetCandidate] = []
    if deps is not None:
        filters: dict = {"asset_class": "equity"}
        if pe_cap < _PE_CAP_DEFAULT:
            filters["max_pe"] = pe_cap

        for market in markets:
            batch = deps.market.screen(market, filters)
            candidates.extend(batch)
    else:
        # No deps: return existing equity_candidates unchanged (test helper path)
        candidates = list(state.equity_candidates)

    # Filter by risk ceiling
    eligible = [c for c in candidates if R_ORDER.get(c.r_level, 5) <= ceiling_order]

    event = {
        "node": "equity",
        "ts": time.time(),
        "markets": markets,
        "risk_ceiling": risk_ceiling,
        "tilt": tilt,
        "total_screened": len(candidates),
        "eligible": len(eligible),
    }
    note = (
        f"equity_node: screened {len(candidates)} candidates across {markets}; "
        f"{len(eligible)} within {risk_ceiling} ceiling; macro_tilt={tilt}"
    )

    return {
        "equity_candidates": eligible,
        "trace_events": state.trace_events + [event],
        "notes": state.notes + [note],
    }
