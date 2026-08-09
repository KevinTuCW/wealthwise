"""Pure portfolio math functions — stdlib only (math module), no numpy.

All functions are deterministic, stateless, and side-effect-free so they can
be tested exhaustively.

Correlation assumption used by optimize.py for cross-asset pairs where the
caller does not supply an explicit matrix: ASSUMED_CROSS_CORR = 0.3.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wealthwise.agents.state import AssetCandidate

# ---------------------------------------------------------------------------
# R-level ordering — single source of truth, imported by optimize.py too.
# ---------------------------------------------------------------------------

R_ORDER: dict[str, int] = {
    "R1": 1,
    "R2": 2,
    "R3": 3,
    "R4": 4,
    "R5": 5,
}


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def normalize(raw: dict[str, float]) -> dict[str, float]:
    """Scale weights so they sum to 1.0.

    Zero-sum behavior:
        - Empty dict or all-zero values → return empty dict.

    Parameters
    ----------
    raw:
        Mapping of symbol → non-negative weight.

    Returns
    -------
    dict[str, float]
        Normalized weights summing to 1.0, or {} if total == 0.
    """
    total = sum(raw.values())
    if total == 0.0:
        return {}
    return {k: v / total for k, v in raw.items()}


# ---------------------------------------------------------------------------
# portfolio_volatility
# ---------------------------------------------------------------------------

def portfolio_volatility(
    weights: list[float],
    vols: list[float],
    corr: list[list[float]],
) -> float:
    """Compute annualized portfolio volatility via √(wᵀ Σ w).

    Σ_ij = corr[i][j] * vol[i] * vol[j]

    Parameters
    ----------
    weights:
        Asset weights (need not sum to 1 — caller's responsibility).
    vols:
        Per-asset annualized volatilities (same length as weights).
    corr:
        n×n correlation matrix (1 on diagonal, values in [-1, 1]).

    Returns
    -------
    float
        Portfolio volatility ≥ 0.
    """
    n = len(weights)
    variance = 0.0
    for i in range(n):
        for j in range(n):
            variance += weights[i] * weights[j] * corr[i][j] * vols[i] * vols[j]
    return math.sqrt(max(variance, 0.0))  # guard tiny negative float rounding


# ---------------------------------------------------------------------------
# max_drawdown_estimate
# ---------------------------------------------------------------------------

def max_drawdown_estimate(vol: float, horizon_years: float) -> float:
    """Conservative estimate of expected maximum drawdown.

    Formula (documented):
        MDD_est = 2 × σ_annual × √horizon_years

    Rationale: a 2-sigma adverse move over the horizon is a widely-cited
    rule-of-thumb for a rough worst-case envelope; it is NOT a rigorous
    statistical estimate and is intentionally conservative.

    Parameters
    ----------
    vol:
        Annualized portfolio volatility (e.g. 0.15 for 15%).
    horizon_years:
        Investment horizon in years.

    Returns
    -------
    float
        Estimated maximum drawdown fraction (0..∞, typically < 1).
    """
    return 2.0 * vol * math.sqrt(horizon_years)


# ---------------------------------------------------------------------------
# sharpe
# ---------------------------------------------------------------------------

def sharpe(exp_return: float, vol: float, rf: float = 0.0) -> float:
    """Sharpe ratio (exp_return - rf) / vol.

    Returns 0.0 when vol == 0 to avoid ZeroDivisionError.
    """
    if vol == 0.0:
        return 0.0
    return (exp_return - rf) / vol


# ---------------------------------------------------------------------------
# diversification_ratio
# ---------------------------------------------------------------------------

def diversification_ratio(
    weights: list[float],
    vols: list[float],
    corr: list[list[float]],
) -> float:
    """Diversification ratio: (Σ wᵢ σᵢ) / σ_portfolio.

    A value of 1.0 means no diversification benefit (single asset or perfectly
    correlated).  Higher values indicate more diversification.

    Returns 1.0 when portfolio volatility is 0 to avoid ZeroDivisionError.
    """
    weighted_vol_sum = sum(w * v for w, v in zip(weights, vols))
    port_vol = portfolio_volatility(weights, vols, corr)
    if port_vol == 0.0:
        return 1.0
    return weighted_vol_sum / port_vol


# ---------------------------------------------------------------------------
# fx_exposure
# ---------------------------------------------------------------------------

def fx_exposure(
    weights: dict[str, float],
    candidates: list[AssetCandidate],
    base_currency: str = "CNY",
) -> float:
    """Fraction of portfolio held in non-base-currency assets.

    Parameters
    ----------
    weights:
        symbol → weight mapping (values should sum to ~1).
    candidates:
        Full candidate list; symbols not present in weights are ignored.
    base_currency:
        Home currency; any candidate.currency != base_currency is foreign.

    Returns
    -------
    float
        Sum of weights for non-base-currency assets (0 ≤ result ≤ 1).
    """
    symbol_to_currency: dict[str, str] = {c.symbol: c.currency for c in candidates}
    return sum(
        w
        for sym, w in weights.items()
        if symbol_to_currency.get(sym, base_currency) != base_currency
    )


# ---------------------------------------------------------------------------
# portfolio_r_level
# ---------------------------------------------------------------------------

def portfolio_r_level(
    weights: dict[str, float],
    candidates: list[AssetCandidate],
) -> str:
    """Aggregate risk level = max single-asset R among assets with weight > 0.

    This is intentionally simple, monotone, and explainable: the portfolio
    cannot be safer than its riskiest held position.  Explainability is
    important for regulatory/compliance contexts.

    Returns "R1" if no asset has weight > 0 (degenerate case).
    """
    symbol_to_r: dict[str, str] = {c.symbol: c.r_level for c in candidates}
    max_order = 0
    max_r = "R1"
    for sym, w in weights.items():
        if w > 0:
            r = symbol_to_r.get(sym, "R1")
            order = R_ORDER.get(r, 1)
            if order > max_order:
                max_order = order
                max_r = r
    return max_r
