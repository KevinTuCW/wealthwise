"""goal_node — derive goal_constraints from investor profile (rule-based, no LLM).

Two independent limits on how much equity the book may hold, and the tighter
one wins:

* **risk tolerance** (C1–C5) — how much the investor may take on
* **goal + horizon** — how much the mandate actually calls for

Plus `risk_ceiling` (C→R), which is a different instrument: it filters *which
securities* are eligible, not *how much* equity there is.

That distinction is the bug this file used to have. The equity cap was keyed on
goal and horizon alone, and risk tolerance reached the pipeline only as the R
ceiling — so a C5 and a C3 with the same goal received the same 35% equity in
different tickers, and raising the risk rating changed nothing an investor would
recognise as risk. Suitability was enforcing "may not exceed" while never asking
"does this match".

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

# `balanced_growth` belongs here despite its name, and its absence was a live
# bug: it is the workbench's own default goal, so every default demo run was
# silently planned as capital preservation. "Balanced" describes the resulting
# mix — which is exactly what the cap below is for — not the mandate.
_AGGRESSIVE_GOALS = frozenset({
    "retirement", "growth", "education", "wealth_appreciation", "balanced_growth",
})

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

# ---------------------------------------------------------------------------
# Equity cap by risk tolerance — the limit the C rating is actually for
# ---------------------------------------------------------------------------
# A house view, not a calibrated number, and deliberately shaped like the C→R
# ladder it sits beside: each step up buys roughly another fifth of the book in
# equity. C5 stops at 0.85 because the liquidity floor has to live somewhere.
#
# This is a *cap*, never a floor. Tolerance is permission, not instruction: a C5
# saving for a two-year goal does not get a growth book because he could stomach
# one. That is what keeps the goal table below meaningful.
#
# C1 is 0.10 rather than 0.00, and the difference is not cosmetic. Zero looked
# like the honest reading of 保守型 — and it silently disarmed a security gate.
# A C1 book is already all cash without any help from this table, because the R1
# ceiling admits no equity; setting the cap to zero adds nothing to that, and it
# means an *unauthorised* instrument that does clear the ceiling (the
# cross-border leak the status_routing suite injects) gets zero weight, drops
# out of the portfolio, and is never seen by the compliance node. The violation
# stops being rejected and starts being invisible.
#
# This repo has made that exact mistake once before, from the other direction:
# dropping wrong-market names during screening produced a cleaner portfolio and
# no detection. Same lesson, second sighting — **a filter that removes evidence
# is not a control.** Refusal has to happen where it can be recorded.
_RISK_EQUITY_CAPS: dict[str, float] = {
    "C1": 0.10,
    "C2": 0.20,
    "C3": 0.40,
    "C4": 0.65,
    "C5": 0.85,
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
    goal_cap = _EQUITY_CAPS[key]
    risk_cap = _RISK_EQUITY_CAPS[profile.risk_level]
    max_equity = min(goal_cap, risk_cap)

    # Which limit bound, recorded rather than inferred. A cap nobody can
    # attribute is a cap nobody can argue with — and "why is my C5 book only 35%
    # equity" has two completely different answers depending on this field.
    equity_cap_source = "goal" if goal_cap <= risk_cap else "risk"

    # The floor must clear neither the cap above it nor the investor's own
    # liquidity requirement; an inverted band would leave the optimiser to
    # violate one of the two without saying which.
    min_equity = min(_EQUITY_FLOORS[key], max_equity,
                     max(0.0, 1.0 - profile.liquidity_min))
    return_band = _RETURN_BANDS[key]

    goal_constraints = {
        "risk_ceiling": risk_ceiling,
        "max_equity": max_equity,
        "min_equity": min_equity,
        "goal_equity_cap": goal_cap,
        "risk_equity_cap": risk_cap,
        "equity_cap_source": equity_cap_source,
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
        "equity_cap_source": equity_cap_source,
        "goal_bucket": g_bucket,
        "horizon_bucket": h_bucket,
    }
    note = (
        f"goal_node: {profile.risk_level}→{risk_ceiling} ceiling; "
        f"goals={profile.goals} ({g_bucket}); "
        f"horizon={profile.horizon_years}y ({h_bucket}); "
        f"equity={min_equity:.0%}–{max_equity:.0%} (capped by {equity_cap_source}); "
        f"liquidity_min={profile.liquidity_min:.0%}"
    )

    return {
        "goal_constraints": goal_constraints,
        "trace_events": state.trace_events + [event],
        "notes": state.notes + [note],
    }
