"""Portfolio optimizer — deterministic risk-budget / inverse-volatility heuristic.

No solver, no numpy.  All math delegated to metrics.py (pure functions).

Cross-asset correlation assumption
-----------------------------------
When building the portfolio volatility estimate we have no per-pair correlation
data at this stage.  We use a conservative fixed assumption:

    ASSUMED_CROSS_CORR = 0.3   (moderate positive correlation across mixed assets)

This is documented here so it is easy to replace with real correlation data later.
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


def _r_order(r_level: str) -> int:
    return R_ORDER.get(r_level, 1)


def build_portfolio(
    candidates: list[AssetCandidate],
    goal_constraints: dict,
    risk_ceiling: str,
    method: str = "risk_parity",
) -> PortfolioAllocation:
    """Build a PortfolioAllocation via inverse-volatility risk-budget weighting.

    Steps
    -----
    1. Filter: remove any candidate whose r_level > risk_ceiling.
    2. Inverse-vol weights: weight_i ∝ 1 / vol_i.
    3. Apply goal_constraints floors/caps if present:
       - ``min_cash`` or ``liquidity_min`` (float, 0..1): ensure cash/bond
         assets collectively receive at least that share.
       - ``max_equity`` (float, 0..1): cap the total equity weight.
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
    ceiling_order = _r_order(risk_ceiling)

    # Step 1: filter by risk ceiling
    eligible = [c for c in candidates if _r_order(c.r_level) <= ceiling_order]

    if not eligible:
        # Edge case: nothing eligible — return empty allocation
        return PortfolioAllocation(
            weights={},
            class_weights={},
            portfolio_r_level="R1",
            fx_exposure=0.0,
            metrics={},
        )

    # Step 2: inverse-vol raw weights
    raw: dict[str, float] = {}
    for c in eligible:
        vol_i = float(c.metrics.get("volatility", DEFAULT_VOL))
        if vol_i <= 0.0:
            vol_i = DEFAULT_VOL
        raw[c.symbol] = 1.0 / vol_i

    # Step 3: apply goal_constraints
    liquidity_floor = float(
        goal_constraints.get("liquidity_min", goal_constraints.get("min_cash", 0.0))
    )
    equity_cap = float(goal_constraints.get("max_equity", 1.0))

    if liquidity_floor > 0.0 or equity_cap < 1.0:
        raw = _apply_constraints(eligible, raw, liquidity_floor, equity_cap)

    # Step 4: normalize
    weights = normalize(raw)

    if not weights:
        # Degenerate: all raw weights zeroed out
        weights = {c.symbol: 1.0 / len(eligible) for c in eligible}
        weights = normalize(weights)

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

    metrics = {
        "volatility": p_vol,
        "sharpe": sr,
        "max_drawdown_estimate": mdd,
        "diversification_ratio": dr,
        "assumed_cross_corr": ASSUMED_CROSS_CORR,
        "n_assets": n,
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

def _apply_constraints(
    eligible: list[AssetCandidate],
    raw: dict[str, float],
    liquidity_floor: float,
    equity_cap: float,
) -> dict[str, float]:
    """Adjust raw inverse-vol weights to respect liquidity floor and equity cap.

    Algorithm (simple iterative two-pass):
    1. Compute tentative normalized weights from raw.
    2. If equity weight exceeds equity_cap, scale equity weights down uniformly
       so they collectively equal equity_cap; redistribute surplus to non-equity.
    3. If cash+bond weight is below liquidity_floor, scale cash/bond up to meet
       the floor; scale equity down proportionally to free up the share.
    4. Re-normalize.

    This is a heuristic, not an optimizer.  It converges in O(n) with no loops.
    """
    # Categorize symbols
    equity_syms = [c.symbol for c in eligible if c.asset_class == "equity"]
    safe_syms = [
        c.symbol for c in eligible if c.asset_class in ("bond", "cash")
    ]

    tentative = normalize(raw)
    if not tentative:
        return raw

    # --- equity cap ---
    equity_total = sum(tentative.get(s, 0.0) for s in equity_syms)
    if equity_total > equity_cap and equity_total > 0:
        scale = equity_cap / equity_total
        for s in equity_syms:
            raw[s] *= scale
        # Non-equity gets a proportional boost to absorb the freed weight
        surplus = equity_total - equity_cap
        non_equity_total = 1.0 - equity_total
        if non_equity_total > 0:
            boost = 1.0 + surplus / non_equity_total
            for c in eligible:
                if c.symbol not in equity_syms:
                    raw[c.symbol] *= boost
        tentative = normalize(raw)

    # --- liquidity floor ---
    safe_total = sum(tentative.get(s, 0.0) for s in safe_syms)
    if safe_total < liquidity_floor and len(safe_syms) > 0:
        needed = liquidity_floor - safe_total
        # Boost safe assets uniformly
        for s in safe_syms:
            raw[s] = raw.get(s, 0.0) + needed / len(safe_syms)
        # Scale equity down to keep sum manageable (re-normalize handles the rest)
        if len(equity_syms) > 0:
            equity_total_now = sum(tentative.get(s, 0.0) for s in equity_syms)
            available_equity = max(0.0, equity_total_now - needed)
            if equity_total_now > 0:
                eq_scale = available_equity / equity_total_now
                for s in equity_syms:
                    raw[s] *= eq_scale

    return raw
