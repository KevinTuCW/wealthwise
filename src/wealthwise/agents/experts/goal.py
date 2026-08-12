"""goal_node — derive goal_constraints from investor profile (rule-based, no LLM).

Maps C-level → risk_ceiling, goals + horizon → equity cap + target return band,
and carries liquidity_min and accept_cross_border through as constraint floors.

All logic is deterministic: no LLM call, no external data needed.
"""
from __future__ import annotations

import time

from wealthwise.agents.state import AdvisoryState

# ---------------------------------------------------------------------------
# C-level → R-level ceiling (1-to-1 numeric mapping)
# ---------------------------------------------------------------------------

_C_TO_R: dict[str, str] = {
    "C1": "R1",
    "C2": "R2",
    "C3": "R3",
    "C4": "R4",
    "C5": "R5",
}

# ---------------------------------------------------------------------------
# Equity cap table  — (goal_bucket, horizon_bucket) → max_equity
# ---------------------------------------------------------------------------
# "aggressive" goals (retirement / growth / education):
#   long  (≥ 8y)  → 0.80
#   mid   (4–7y)  → 0.65
#   short (< 4y)  → 0.45
# "conservative" goals (capital_preservation / income / liquidity):
#   long           → 0.50
#   mid            → 0.35
#   short          → 0.20

_AGGRESSIVE_GOALS = frozenset({"retirement", "growth", "education", "wealth_appreciation"})

_EQUITY_CAPS: dict[tuple[str, str], float] = {
    ("aggressive", "long"):  0.80,
    ("aggressive", "mid"):   0.65,
    ("aggressive", "short"): 0.45,
    ("conservative", "long"):  0.50,
    ("conservative", "mid"):   0.35,
    ("conservative", "short"): 0.20,
}

# ---------------------------------------------------------------------------
# Equity FLOOR table — the other half of suitability
# ---------------------------------------------------------------------------
# A cap alone only prevents "too aggressive". Answering a ten-year retirement
# mandate with a pile of money-market funds is unsuitable in the other
# direction, and nothing downstream was checking for it. The floor sits at
# roughly two-thirds of the cap: enough to keep the mandate honest, loose enough
# that the liquidity floor and the de-risk retry still have room to move.
_EQUITY_FLOORS: dict[tuple[str, str], float] = {
    ("aggressive", "long"):  0.55,
    ("aggressive", "mid"):   0.40,
    ("aggressive", "short"): 0.25,
    ("conservative", "long"):  0.30,
    ("conservative", "mid"):   0.20,
    ("conservative", "short"): 0.05,
}

# Target return bands  (low, high) as approximate annualized targets
_RETURN_BANDS: dict[tuple[str, str], tuple[float, float]] = {
    ("aggressive", "long"):  (0.07, 0.12),
    ("aggressive", "mid"):   (0.05, 0.09),
    ("aggressive", "short"): (0.03, 0.06),
    ("conservative", "long"):  (0.04, 0.07),
    ("conservative", "mid"):   (0.03, 0.05),
    ("conservative", "short"): (0.02, 0.04),
}


def _goal_bucket(goals: list[str]) -> str:
    """Return 'aggressive' if any goal is in the growth set, else 'conservative'."""
    for g in goals:
        if g.lower() in _AGGRESSIVE_GOALS:
            return "aggressive"
    return "conservative"


def _horizon_bucket(horizon_years: int) -> str:
    if horizon_years >= 8:
        return "long"
    if horizon_years >= 4:
        return "mid"
    return "short"


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def goal_node(state: AdvisoryState, deps) -> dict:
    """Derive goal_constraints from state.profile deterministically.

    Parameters
    ----------
    state:
        AdvisoryState — must have a valid .profile.
    deps:
        AdvisoryDeps (unused by this node; present for uniform node signature).

    Returns
    -------
    dict
        State increment with keys: goal_constraints, trace_events, notes.
    """
    profile = state.profile
    if profile is None:
        raise ValueError("goal_node requires state.profile to be set")

    risk_ceiling = _C_TO_R[profile.risk_level]
    g_bucket = _goal_bucket(profile.goals)
    h_bucket = _horizon_bucket(profile.horizon_years)

    key = (g_bucket, h_bucket)
    max_equity = _EQUITY_CAPS[key]
    # The floor must never fight the investor's own liquidity requirement.
    min_equity = min(_EQUITY_FLOORS[key], max(0.0, 1.0 - profile.liquidity_min))
    return_band = _RETURN_BANDS[key]

    goal_constraints = {
        "risk_ceiling": risk_ceiling,
        "max_equity": max_equity,
        "min_equity": min_equity,
        "target_return_low": return_band[0],
        "target_return_high": return_band[1],
        "liquidity_min": profile.liquidity_min,
        "accept_cross_border": profile.accept_cross_border,
        "goal_bucket": g_bucket,
        "horizon_bucket": h_bucket,
    }

    event = {
        "node": "goal",
        "ts": time.time(),
        "risk_ceiling": risk_ceiling,
        "max_equity": max_equity,
        "min_equity": min_equity,
        "goal_bucket": g_bucket,
        "horizon_bucket": h_bucket,
    }
    note = (
        f"goal_node: {profile.risk_level}→{risk_ceiling} ceiling; "
        f"goals={profile.goals} ({g_bucket}); "
        f"horizon={profile.horizon_years}y ({h_bucket}); "
        f"equity={min_equity:.0%}–{max_equity:.0%}; "
        f"liquidity_min={profile.liquidity_min:.0%}"
    )

    return {
        "goal_constraints": goal_constraints,
        "trace_events": state.trace_events + [event],
        "notes": state.notes + [note],
    }
