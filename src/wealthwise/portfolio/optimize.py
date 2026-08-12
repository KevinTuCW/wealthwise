"""Portfolio optimizer — deterministic two-level risk-budget heuristic.

No solver, no numpy.  All math delegated to metrics.py (pure functions).

Why two levels
--------------
Flat inverse-volatility weighting across every candidate degenerates the moment
a money-market fund is in the pool: at vol ≈ 0.01 its raw weight is 1/0.01 = 100
against an equity's 1/0.22 ≈ 4.5, so cash swallows the portfolio.  In practice
that produced an **83% cash** allocation for a C4 investor with a ten-year
retirement goal — perfectly "safe", entirely unsuitable, and waved through by
every downstream guardrail, because they all check for *too much* risk and none
of them checks for too little.

So allocation runs top-down, the way it is actually done:

  1. **Class targets** — equity / bond / cash shares derived from goal
     constraints (`min_equity`, `max_equity`, `liquidity_min`).  Suitability is
     two-sided: the equity floor is as real a constraint as the equity cap.
  2. **Within-class inverse volatility** — spread each class's target over its
     members, with a volatility floor and a per-asset cap so one ultra-low-vol
     instrument cannot dominate its own class either.

Assumptions, all deliberately conservative:

    ASSUMED_CROSS_CORR = 0.3   moderate positive correlation across mixed assets
    WEIGHT_VOL_FLOOR   = 0.03  floor on the vol used for *weighting* (not risk)
    MAX_ASSET_IN_CLASS = 0.40  no single name takes more than 40% of its class
"""
from __future__ import annotations

from wealthwise.agents.state import AssetCandidate, PortfolioAllocation
from wealthwise.portfolio.metrics import (
    R_ORDER,
    diversification_ratio,
    fx_exposure,
    max_drawdown_estimate,
    normalize,
    portfolio_r_level,
    portfolio_volatility,
    sharpe,
)

# Default volatility assigned to an asset that carries no volatility metric.
DEFAULT_VOL: float = 0.15

# Fixed cross-asset correlation assumed when building portfolio vol estimate.
ASSUMED_CROSS_CORR: float = 0.3

# Default assumed expected return used for Sharpe estimate (in absence of
# forward-return signals — a 5% placeholder for a mixed growth portfolio).
DEFAULT_EXP_RETURN: float = 0.05

# Default investment horizon used for max-drawdown estimate (years).
DEFAULT_HORIZON_YEARS: float = 3.0

# Risk-free rate used for Sharpe (approximate China short-term rate, 2025).
RISK_FREE_RATE: float = 0.02

# Volatility floor applied when *weighting* only. Risk metrics still use the real
# volatility; this exists purely so a 1%-vol money-market fund does not receive a
# 20x weight over an equity and turn every portfolio into a cash pile.
WEIGHT_VOL_FLOOR: float = 0.03

# No single instrument may take more than this share of its own asset class.
MAX_ASSET_IN_CLASS: float = 0.40

# Asset classes treated as the liquid/"safe" bucket for the liquidity floor.
SAFE_CLASSES: tuple[str, ...] = ("cash", "bond")

# Share of the safe bucket held as cash when both cash and bond are available.
# Cash is for liquidity, not for return; bonds carry the rest.
CASH_SHARE_OF_SAFE: float = 0.30


def build_portfolio(
    candidates: list[AssetCandidate],
    goal_constraints: dict,
    risk_ceiling: str,
    method: str = "risk_parity",
) -> PortfolioAllocation:
    """Build a PortfolioAllocation via two-level risk-budget weighting.

    Steps
    -----
    1. Filter: remove any candidate whose r_level > risk_ceiling.
    2. Class targets: equity / bond / cash shares from goal_constraints
       (``min_equity`` floor, ``max_equity`` cap, ``liquidity_min`` floor),
       redistributed when a class has no eligible members.
    3. Within each class: inverse-volatility weights over a floored volatility,
       capped so no single name exceeds ``MAX_ASSET_IN_CLASS`` of its class.
    4. Normalize to sum 1.
    5. Compute class_weights, portfolio_r_level, fx_exposure, metrics dict.

    Parameters
    ----------
    candidates:
        All candidate assets to consider (may include ineligible r_levels).
    goal_constraints:
        Constraint bag from AdvisoryState.  Recognized keys:
        - ``liquidity_min`` / ``min_cash`` (float): minimum cash+bond allocation.
        - ``max_equity`` (float): maximum equity allocation.
        - ``min_equity`` (float): **minimum** equity allocation — the other half
          of suitability. Without it a growth mandate can be answered with a pile
          of cash, which is its own kind of unsuitable.
    risk_ceiling:
        Maximum permitted r_level (inclusive).  Assets with higher r_level
        are excluded from the portfolio.
    method:
        Weighting scheme identifier (only "risk_parity" is implemented;
        kept as a parameter for future extension without API change).

    Returns
    -------
    PortfolioAllocation
        A fully populated allocation object.
    """
    if method != "risk_parity":
        raise NotImplementedError(f"unknown method: {method}")

    ceiling_order = R_ORDER.get(risk_ceiling, 1)

    # Step 1: filter by risk ceiling
    eligible = [c for c in candidates if R_ORDER.get(c.r_level, 1) <= ceiling_order]

    if not eligible:
        # Edge case: nothing eligible — return empty allocation
        return PortfolioAllocation(
            weights={},
            class_weights={},
            portfolio_r_level="R1",
            fx_exposure=0.0,
            metrics={},
        )

    # Step 2: class targets (top-down)
    liquidity_floor = float(
        goal_constraints.get("liquidity_min", goal_constraints.get("min_cash", 0.0))
    )
    equity_cap = float(goal_constraints.get("max_equity", 1.0))
    equity_floor = float(goal_constraints.get("min_equity", 0.0))
    targets = class_targets(eligible, equity_floor, equity_cap, liquidity_floor)

    # Step 3: within-class inverse volatility, then scale to the class target
    weights: dict[str, float] = {}
    for cls, target in targets.items():
        members = [c for c in eligible if c.asset_class == cls]
        for sym, share in _within_class_weights(members).items():
            weights[sym] = weights.get(sym, 0.0) + share * target

    # Step 4: normalize
    weights = normalize(weights)

    if not weights:
        # Degenerate: all raw weights zeroed out
        weights = normalize({c.symbol: 1.0 for c in eligible})

    # Step 5: derived quantities

    # class_weights
    symbol_to_class: dict[str, str] = {c.symbol: c.asset_class for c in eligible}
    class_weights: dict[str, float] = {}
    for sym, w in weights.items():
        ac = symbol_to_class.get(sym, "equity")
        class_weights[ac] = class_weights.get(ac, 0.0) + w

    # portfolio_r_level
    p_r_level = portfolio_r_level(weights, eligible)

    # fx_exposure
    fx_exp = fx_exposure(weights, eligible)

    # portfolio vol estimate (use symmetric correlation matrix with ASSUMED_CROSS_CORR)
    syms = list(weights.keys())
    n = len(syms)
    sym_to_vol: dict[str, float] = {
        c.symbol: float(c.metrics.get("volatility", DEFAULT_VOL)) for c in eligible
    }
    w_list = [weights[s] for s in syms]
    vol_list = [sym_to_vol.get(s, DEFAULT_VOL) for s in syms]
    corr_matrix = [
        [1.0 if i == j else ASSUMED_CROSS_CORR for j in range(n)] for i in range(n)
    ]
    p_vol = portfolio_volatility(w_list, vol_list, corr_matrix)
    dr = diversification_ratio(w_list, vol_list, corr_matrix)
    mdd = max_drawdown_estimate(p_vol, DEFAULT_HORIZON_YEARS)
    sr = sharpe(DEFAULT_EXP_RETURN, p_vol, RISK_FREE_RATE)

    # Compute actual cash+bond weight achieved to record liquidity compliance honestly
    liquidity_achieved = class_weights.get("cash", 0.0) + class_weights.get("bond", 0.0)
    metrics = {
        "volatility": p_vol,
        "sharpe": sr,
        "max_drawdown_estimate": mdd,
        "diversification_ratio": dr,
        "assumed_cross_corr": ASSUMED_CROSS_CORR,
        "n_assets": n,
        "liquidity_target": liquidity_floor,
        "liquidity_achieved": liquidity_achieved,
        "constraints_met": liquidity_achieved >= liquidity_floor - 1e-9,
    }

    return PortfolioAllocation(
        weights=weights,
        class_weights=class_weights,
        portfolio_r_level=p_r_level,
        fx_exposure=fx_exp,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def class_targets(
    eligible: list[AssetCandidate],
    equity_floor: float,
    equity_cap: float,
    liquidity_floor: float,
) -> dict[str, float]:
    """Resolve asset-class target weights from the goal constraints.

    Everything happens in normalized weight space (shares of 1.0). The previous
    implementation mixed spaces — it added a normalized shortfall (0..1) onto a
    raw 1/vol figure (magnitude 5..100) — so the liquidity floor was a nudge
    rather than a constraint and could silently miss its target.

    Rules, in order:
      * the equity target starts at its cap, and is never allowed to squeeze the
        liquidity floor out;
      * it is then pulled back up to the equity floor if the floor still leaves
        room for the liquidity requirement;
      * whatever a missing class cannot absorb is redistributed to the classes
        that do have eligible members, so the targets always sum to 1.
    """
    classes = {c.asset_class for c in eligible}
    has_equity = "equity" in classes
    safe_present = [c for c in SAFE_CLASSES if c in classes]
    other = sorted(classes - {"equity"} - set(SAFE_CLASSES))

    liquidity_floor = min(max(liquidity_floor, 0.0), 1.0)
    equity_cap = min(max(equity_cap, 0.0), 1.0)
    equity_floor = min(max(equity_floor, 0.0), 1.0)

    if not has_equity:
        equity_target = 0.0
    elif not safe_present and not other:
        equity_target = 1.0
    else:
        headroom = 1.0 - liquidity_floor
        equity_target = min(equity_cap, headroom)
        equity_target = max(equity_target, min(equity_floor, headroom))
        equity_target = max(0.0, min(1.0, equity_target))

    remainder = 1.0 - equity_target
    targets: dict[str, float] = {}
    if has_equity:
        targets["equity"] = equity_target

    if not safe_present and not other:
        return targets or {c.asset_class: 1.0 for c in eligible[:1]}

    # Non-equity classes: 'other' (e.g. commodity) shares the remainder evenly
    # with the safe bucket; the safe bucket splits cash/bond by CASH_SHARE_OF_SAFE.
    if other and safe_present:
        other_total = remainder * 0.5 / len(other)
        for cls in other:
            targets[cls] = other_total
        safe_total = remainder - other_total * len(other)
    elif other:
        for cls in other:
            targets[cls] = remainder / len(other)
        safe_total = 0.0
    else:
        safe_total = remainder

    if safe_present and safe_total > 0:
        if len(safe_present) == 1:
            targets[safe_present[0]] = safe_total
        else:
            targets["cash"] = safe_total * CASH_SHARE_OF_SAFE
            targets["bond"] = safe_total * (1.0 - CASH_SHARE_OF_SAFE)

    total = sum(targets.values())
    return {k: v / total for k, v in targets.items()} if total > 0 else targets


def _within_class_weights(members: list[AssetCandidate]) -> dict[str, float]:
    """Inverse-volatility weights inside one asset class, summing to 1.

    Two guards keep this from degenerating the way the flat version did:
      * volatility is floored at WEIGHT_VOL_FLOOR for weighting purposes (risk
        metrics still use the real number), so a 1%-vol money-market fund does
        not out-weigh an equity twentyfold;
      * no member takes more than MAX_ASSET_IN_CLASS of its class, with the
        excess redistributed to the rest.
    """
    if not members:
        return {}
    if len(members) == 1:
        return {members[0].symbol: 1.0}

    raw: dict[str, float] = {}
    for c in members:
        vol = float(c.metrics.get("volatility", DEFAULT_VOL))
        if vol <= 0.0:
            vol = DEFAULT_VOL
        raw[c.symbol] = 1.0 / max(vol, WEIGHT_VOL_FLOOR)

    weights = normalize(raw)
    cap = max(MAX_ASSET_IN_CLASS, 1.0 / len(members))   # cap must stay feasible
    for _ in range(len(members)):                        # bounded redistribution
        over = {s: w for s, w in weights.items() if w > cap + 1e-12}
        if not over:
            break
        excess = sum(w - cap for w in over.values())
        under = {s: w for s, w in weights.items() if s not in over}
        under_total = sum(under.values())
        for s in over:
            weights[s] = cap
        if under_total <= 0:
            break
        for s, w in under.items():
            weights[s] = w + excess * (w / under_total)
    return normalize(weights)
