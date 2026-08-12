"""Suitability runs in two directions — this file guards the neglected one.

Every existing guardrail asks "is this too risky for the investor?". None asked
"does this actually answer the mandate?", so a C4 ten-year retirement profile
could be handed 82.8% money-market funds, 6.8% equity, and a clean PASS.
"""
from wealthwise.agents.state import AssetCandidate, InvestorProfile
from wealthwise.bootstrap import build_sample_deps
from wealthwise.portfolio.optimize import (MAX_ASSET_IN_CLASS, build_portfolio,
                                           class_targets)
from wealthwise.runner import run_advisory


def _cand(sym, r_level, asset_class, vol, market="A", currency="CNY"):
    return AssetCandidate(symbol=sym, name=sym, market=market, asset_class=asset_class,
                          r_level=r_level, currency=currency,
                          metrics={"volatility": vol})


def _pool():
    return [
        _cand("EQ1", "R3", "equity", 0.22),
        _cand("EQ2", "R3", "equity", 0.28),
        _cand("EQ3", "R2", "equity", 0.18),
        _cand("BD1", "R2", "bond", 0.05),
        _cand("CA1", "R1", "cash", 0.01),
    ]


def test_cash_no_longer_swallows_the_portfolio():
    # The regression: 1/0.01 = 100 vs 1/0.22 ≈ 4.5 made cash dominate everything.
    alloc = build_portfolio(_pool(), {"min_equity": 0.55, "max_equity": 0.80,
                                      "liquidity_min": 0.20}, risk_ceiling="R4")
    assert alloc.class_weights["equity"] >= 0.55
    assert alloc.class_weights.get("cash", 0.0) < 0.25


def test_equity_floor_is_honoured_for_a_growth_mandate():
    alloc = build_portfolio(_pool(), {"min_equity": 0.55, "max_equity": 0.80,
                                      "liquidity_min": 0.10}, risk_ceiling="R4")
    assert alloc.class_weights["equity"] >= 0.55


def test_equity_cap_still_binds():
    alloc = build_portfolio(_pool(), {"min_equity": 0.05, "max_equity": 0.20,
                                      "liquidity_min": 0.10}, risk_ceiling="R4")
    assert alloc.class_weights["equity"] <= 0.20 + 1e-9


def test_liquidity_floor_is_met_exactly_not_approximately():
    # The old weight-space mixing could undershoot the floor and only record it.
    for floor in (0.0, 0.3, 0.5, 0.7, 0.9):
        alloc = build_portfolio(_pool(), {"liquidity_min": floor, "max_equity": 0.8},
                                risk_ceiling="R4")
        liquid = (alloc.class_weights.get("cash", 0.0)
                  + alloc.class_weights.get("bond", 0.0))
        assert liquid >= floor - 1e-9, f"floor {floor} -> {liquid}"
        assert alloc.metrics["constraints_met"] is True


def test_liquidity_floor_beats_the_equity_floor_when_they_conflict():
    alloc = build_portfolio(_pool(), {"min_equity": 0.80, "liquidity_min": 0.90},
                            risk_ceiling="R4")
    liquid = alloc.class_weights.get("cash", 0.0) + alloc.class_weights.get("bond", 0.0)
    assert liquid >= 0.90 - 1e-9


def test_no_single_name_dominates_its_class():
    alloc = build_portfolio(_pool(), {"min_equity": 0.6, "max_equity": 0.8},
                            risk_ceiling="R4")
    equity_total = alloc.class_weights["equity"]
    for sym in ("EQ1", "EQ2", "EQ3"):
        share_of_class = alloc.weights.get(sym, 0.0) / equity_total
        assert share_of_class <= MAX_ASSET_IN_CLASS + 1e-9


def test_class_targets_redistribute_when_a_class_is_missing():
    equity_only = [_cand("EQ1", "R3", "equity", 0.2)]
    targets = class_targets(equity_only, equity_floor=0.5, equity_cap=0.8,
                            liquidity_floor=0.4)
    assert targets == {"equity": 1.0}      # nothing else to hold it in


def test_weights_still_sum_to_one_and_respect_the_ceiling():
    alloc = build_portfolio(_pool() + [_cand("RISK", "R5", "equity", 0.5)],
                            {"min_equity": 0.5}, risk_ceiling="R3")
    assert abs(sum(alloc.weights.values()) - 1.0) < 1e-6
    assert alloc.weights.get("RISK", 0.0) == 0.0


def test_end_to_end_growth_mandate_is_not_answered_with_cash():
    profile = InvestorProfile(risk_level="C4", investable=1_000_000.0,
                              horizon_years=10, goals=["retirement"],
                              liquidity_min=0.2, accept_cross_border=True)
    state = run_advisory(profile, build_sample_deps())
    equity = state.portfolio.class_weights.get("equity", 0.0)
    cash = state.portfolio.class_weights.get("cash", 0.0)
    assert equity >= 0.45, f"growth mandate answered with {equity:.1%} equity"
    assert cash <= 0.30, f"{cash:.1%} in cash for a ten-year growth goal"
    assert state.status == "done"
