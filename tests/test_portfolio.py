"""Tests for portfolio metrics and optimizer — TDD, written BEFORE implementation.

Expected failure mode before implementation:
    ModuleNotFoundError: No module named 'wealthwise.portfolio.metrics'
"""
from __future__ import annotations

import math

import pytest

from wealthwise.agents.state import AssetCandidate, PortfolioAllocation
from wealthwise.portfolio.metrics import (
    diversification_ratio,
    fx_exposure,
    normalize,
    portfolio_r_level,
    portfolio_volatility,
    sharpe,
)
from wealthwise.portfolio.optimize import build_portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    symbol: str,
    r_level: str,
    asset_class: str = "equity",
    currency: str = "CNY",
    vol: float = 0.2,
) -> AssetCandidate:
    return AssetCandidate(
        symbol=symbol,
        market="A",
        asset_class=asset_class,
        name=symbol,
        currency=currency,
        r_level=r_level,
        metrics={"volatility": vol},
    )


# ---------------------------------------------------------------------------
# test_weights_normalize
# ---------------------------------------------------------------------------

def test_weights_normalize():
    result = normalize({"a": 2.0, "b": 2.0})
    assert abs(result["a"] - 0.5) < 1e-9
    assert abs(result["b"] - 0.5) < 1e-9
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_weights_normalize_zero_sum_empty():
    """All-zero input → empty dict (defined behavior)."""
    result = normalize({"a": 0.0, "b": 0.0})
    assert result == {}


def test_weights_normalize_empty():
    """Empty input → empty dict."""
    result = normalize({})
    assert result == {}


def test_weights_normalize_single():
    result = normalize({"x": 5.0})
    assert abs(result["x"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# test_portfolio_vol
# ---------------------------------------------------------------------------

def test_portfolio_vol():
    """Two uncorrelated assets: vol = sqrt(w1²σ1² + w2²σ2²)."""
    weights = [0.5, 0.5]
    vols = [0.2, 0.1]
    corr = [[1.0, 0.0], [0.0, 1.0]]
    expected = math.sqrt(0.5**2 * 0.2**2 + 0.5**2 * 0.1**2)  # ≈ 0.11180
    result = portfolio_volatility(weights, vols, corr)
    assert abs(result - expected) < 1e-4


def test_portfolio_vol_correlated():
    """Perfectly correlated (ρ=1): vol == weighted sum of individual vols."""
    weights = [0.5, 0.5]
    vols = [0.2, 0.1]
    corr = [[1.0, 1.0], [1.0, 1.0]]
    expected = 0.5 * 0.2 + 0.5 * 0.1  # 0.15
    result = portfolio_volatility(weights, vols, corr)
    assert abs(result - expected) < 1e-4


# ---------------------------------------------------------------------------
# test_diversification
# ---------------------------------------------------------------------------

def test_diversification():
    """Equal-weight 4 uncorrelated assets has higher DR than a single-asset portfolio."""
    n = 4
    weights_equal = [1 / n] * n
    vols_equal = [0.2] * n
    corr_identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    dr_diversified = diversification_ratio(weights_equal, vols_equal, corr_identity)

    # Single asset: weight=1 on first, 0 on rest.  DR = 1.0 (no diversification).
    weights_single = [1.0, 0.0, 0.0, 0.0]
    dr_single = diversification_ratio(weights_single, vols_equal, corr_identity)

    assert dr_diversified > dr_single


# ---------------------------------------------------------------------------
# test_fx_exposure
# ---------------------------------------------------------------------------

def test_fx_exposure():
    """FX exposure == sum of weights for non-CNY assets."""
    candidates = [
        _make_candidate("A", "R2", currency="CNY"),
        _make_candidate("B", "R2", currency="USD"),
        _make_candidate("C", "R2", currency="HKD"),
    ]
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    result = fx_exposure(weights, candidates)
    assert abs(result - 0.5) < 1e-9  # B(0.3) + C(0.2) = 0.5


# ---------------------------------------------------------------------------
# test_risk_budget_allocates
# ---------------------------------------------------------------------------

def test_risk_budget_allocates():
    """build_portfolio returns PortfolioAllocation with weights summing to 1."""
    candidates = [
        _make_candidate("E1", "R2", asset_class="equity", vol=0.25),
        _make_candidate("B1", "R1", asset_class="bond", vol=0.05),
        _make_candidate("C1", "R1", asset_class="cash", vol=0.01),
    ]
    goal_constraints: dict = {}
    alloc = build_portfolio(candidates, goal_constraints, risk_ceiling="R3")
    assert isinstance(alloc, PortfolioAllocation)
    assert abs(sum(alloc.weights.values()) - 1.0) < 1e-6
    # No asset exceeds risk ceiling R3
    r_order = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
    for sym, w in alloc.weights.items():
        if w > 0:
            cand = next(c for c in candidates if c.symbol == sym)
            assert r_order[cand.r_level] <= r_order["R3"]


# ---------------------------------------------------------------------------
# test_portfolio_r_level_monotone
# ---------------------------------------------------------------------------

def test_portfolio_r_level_monotone():
    """All-R2 portfolio → R2; adding an R4 asset raises it to R4."""
    c_r2 = [
        _make_candidate("A", "R2"),
        _make_candidate("B", "R2"),
    ]
    weights_r2 = {"A": 0.5, "B": 0.5}
    assert portfolio_r_level(weights_r2, c_r2) == "R2"

    c_mixed = [
        _make_candidate("A", "R2"),
        _make_candidate("B", "R2"),
        _make_candidate("C", "R4"),
    ]
    weights_mixed = {"A": 0.4, "B": 0.4, "C": 0.2}
    assert portfolio_r_level(weights_mixed, c_mixed) == "R4"


# ---------------------------------------------------------------------------
# test_build_excludes_over_ceiling
# ---------------------------------------------------------------------------

def test_build_excludes_over_ceiling():
    """risk_ceiling=R3 must exclude any R5 asset from returned weights."""
    candidates = [
        _make_candidate("Safe", "R2", vol=0.1),
        _make_candidate("Risky", "R5", vol=0.4),
    ]
    alloc = build_portfolio(candidates, {}, risk_ceiling="R3")
    # R5 asset must not appear with positive weight
    assert alloc.weights.get("Risky", 0.0) == 0.0
    # Safe asset should carry the full weight
    assert abs(alloc.weights.get("Safe", 0.0) - 1.0) < 1e-6
