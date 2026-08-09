"""Tests for investor suitability C-R hard gate.

TDD: these tests are written before implementation. Expected failure mode
before implementation: ImportError / ModuleNotFoundError.

Zero-miss semantics (test_zero_miss): if *any* over-level portfolio escapes
check_suitability without being flagged, the test fails — this is the
compliance hard gate.
"""
from __future__ import annotations

import pytest

from wealthwise.agents.state import (
    AssetCandidate,
    InvestorProfile,
    PortfolioAllocation,
)
from wealthwise.compliance.suitability import check_suitability, is_over_level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(
    risk_level: str = "C3",
    liquidity_min: float = 0.1,
    accept_cross_border: bool = True,
    investable: float = 500_000.0,
    horizon_years: int = 5,
) -> InvestorProfile:
    return InvestorProfile(
        risk_level=risk_level,
        investable=investable,
        horizon_years=horizon_years,
        goals=["growth"],
        liquidity_min=liquidity_min,
        accept_cross_border=accept_cross_border,
    )


def _candidate(
    symbol: str,
    r_level: str,
    asset_class: str = "equity",
    market: str = "A",
) -> AssetCandidate:
    return AssetCandidate(
        symbol=symbol,
        market=market,
        asset_class=asset_class,
        name=symbol,
        currency="CNY",
        r_level=r_level,
    )


def _portfolio(
    weights: dict[str, float],
    class_weights: dict[str, float] | None = None,
    portfolio_r_level: str = "R3",
) -> PortfolioAllocation:
    if class_weights is None:
        class_weights = {"equity": 1.0}
    return PortfolioAllocation(
        weights=weights,
        class_weights=class_weights,
        portfolio_r_level=portfolio_r_level,
        fx_exposure=0.0,
    )


# ---------------------------------------------------------------------------
# test_match_ok
# ---------------------------------------------------------------------------

def test_match_ok():
    """C4 investor + portfolio of R3-max assets → PASS, matched=True, no violations."""
    profile = _profile(risk_level="C4")
    candidates = [
        _candidate("AA", "R2"),
        _candidate("BB", "R3"),
        _candidate("CC", "R1", asset_class="bond"),
    ]
    portfolio = _portfolio(
        weights={"AA": 0.4, "BB": 0.4, "CC": 0.2},
        class_weights={"equity": 0.8, "bond": 0.2},
        portfolio_r_level="R3",
    )

    verdict = check_suitability(profile, portfolio, candidates)

    assert verdict.matched is True
    assert verdict.decision == "PASS"
    assert verdict.violations == []


# ---------------------------------------------------------------------------
# test_over_level_rejected
# ---------------------------------------------------------------------------

def test_over_level_rejected():
    """C2 investor + portfolio containing an R5 asset → not matched, violations name offending symbol."""
    profile = _profile(risk_level="C2")
    candidates = [
        _candidate("SAFE", "R1", asset_class="bond"),
        _candidate("RISKY", "R5"),
    ]
    portfolio = _portfolio(
        weights={"SAFE": 0.6, "RISKY": 0.4},
        class_weights={"equity": 0.4, "bond": 0.6},
        portfolio_r_level="R5",
    )

    verdict = check_suitability(profile, portfolio, candidates)

    assert verdict.matched is False
    assert verdict.decision in {"DOWNGRADE", "REJECT"}
    # Violations must name the offending R5 symbol
    assert any("RISKY" in v for v in verdict.violations), (
        f"Expected 'RISKY' in violations, got: {verdict.violations}"
    )


# ---------------------------------------------------------------------------
# test_liquidity_floor
# ---------------------------------------------------------------------------

def test_liquidity_floor():
    """profile.liquidity_min=0.3 but cash+bond class weight < 0.3 → liquidity violation."""
    profile = _profile(risk_level="C5", liquidity_min=0.30)
    candidates = [
        _candidate("EQ1", "R3"),
        _candidate("EQ2", "R2"),
    ]
    portfolio = _portfolio(
        weights={"EQ1": 0.8, "EQ2": 0.2},
        class_weights={"equity": 1.0},  # 0 cash/bond → below 0.3 floor
        portfolio_r_level="R3",
    )

    verdict = check_suitability(profile, portfolio, candidates)

    assert any("liquidity" in v.lower() for v in verdict.violations), (
        f"Expected a liquidity violation, got: {verdict.violations}"
    )


# ---------------------------------------------------------------------------
# test_cross_border_gate
# ---------------------------------------------------------------------------

def test_cross_border_gate():
    """accept_cross_border=False but portfolio holds HK asset → cross-border violation."""
    profile = _profile(risk_level="C4", accept_cross_border=False)
    candidates = [
        _candidate("DOMESTIC", "R2", market="A"),
        _candidate("HK_STOCK", "R2", market="HK"),
    ]
    portfolio = _portfolio(
        weights={"DOMESTIC": 0.7, "HK_STOCK": 0.3},
        class_weights={"equity": 1.0},
        portfolio_r_level="R2",
    )

    verdict = check_suitability(profile, portfolio, candidates)

    assert any("cross-border" in v.lower() or "跨境" in v for v in verdict.violations), (
        f"Expected a cross-border violation, got: {verdict.violations}"
    )


# ---------------------------------------------------------------------------
# test_zero_miss  — the compliance hard gate
# ---------------------------------------------------------------------------

def test_zero_miss():
    """Every portfolio containing at least one over-level asset must be flagged.

    Loop over a batch of portfolios — if check_suitability misses any one,
    the test fails. This is the 0-漏判 hard-gate requirement.
    """
    # Each entry: (profile_c, asset_r, should_be_flagged)
    cases = [
        # C1 investor: R2..R5 all over-level
        ("C1", "R2", True),
        ("C1", "R3", True),
        ("C1", "R4", True),
        ("C1", "R5", True),
        # C2: R3..R5 over-level
        ("C2", "R3", True),
        ("C2", "R4", True),
        ("C2", "R5", True),
        # C3: R4..R5 over-level
        ("C3", "R4", True),
        ("C3", "R5", True),
        # C4: R5 over-level
        ("C4", "R5", True),
        # C5: nothing over-level (boundary check)
        ("C5", "R5", False),
        # exact-match: not over-level
        ("C3", "R3", False),
    ]

    for c_level, r_level, should_flag in cases:
        profile = _profile(risk_level=c_level)
        candidates = [
            _candidate("TARGET", r_level),
            _candidate("SAFE", "R1", asset_class="cash"),
        ]
        # Give enough cash/bond so liquidity doesn't contaminate the R check
        portfolio = _portfolio(
            weights={"TARGET": 0.5, "SAFE": 0.5},
            class_weights={"equity": 0.5, "cash": 0.5},
            portfolio_r_level=r_level,
        )
        verdict = check_suitability(profile, portfolio, candidates)

        if should_flag:
            assert verdict.matched is False, (
                f"MISS: C={c_level}, R={r_level} — expected matched=False, "
                f"got matched={verdict.matched}, decision={verdict.decision}"
            )
            assert verdict.decision in {"DOWNGRADE", "REJECT"}, (
                f"MISS: C={c_level}, R={r_level} — expected DOWNGRADE or REJECT, "
                f"got {verdict.decision}"
            )
        else:
            assert verdict.matched is True, (
                f"FALSE POSITIVE: C={c_level}, R={r_level} — "
                f"expected matched=True, got matched={verdict.matched}"
            )


# ---------------------------------------------------------------------------
# test_clean_cross_border_ok
# ---------------------------------------------------------------------------

def test_clean_cross_border_ok():
    """accept_cross_border=True + HK/US assets within C-level → PASS."""
    profile = _profile(risk_level="C4", accept_cross_border=True)
    candidates = [
        _candidate("US_ETF", "R3", market="US"),
        _candidate("HK_FUND", "R2", market="HK"),
        _candidate("CASH_A", "R1", asset_class="cash", market="A"),
    ]
    portfolio = _portfolio(
        weights={"US_ETF": 0.4, "HK_FUND": 0.3, "CASH_A": 0.3},
        class_weights={"equity": 0.7, "cash": 0.3},
        portfolio_r_level="R3",
    )

    verdict = check_suitability(profile, portfolio, candidates)

    assert verdict.matched is True
    assert verdict.decision == "PASS"
    assert verdict.violations == []


# ---------------------------------------------------------------------------
# test_is_over_level helper
# ---------------------------------------------------------------------------

def test_is_over_level():
    """Direct unit tests for the is_over_level pure helper."""
    assert is_over_level("R3", "C2") is True    # 3 > 2
    assert is_over_level("R2", "C2") is False   # equal: not over-level
    assert is_over_level("R1", "C2") is False   # below: not over-level
    assert is_over_level("R5", "C4") is True    # 5 > 4
    assert is_over_level("R5", "C5") is False   # max C allows max R


def test_absent_symbol_treated_as_high_risk():
    """A symbol held in the portfolio but absent from candidates must be treated
    as R5 (fail-closed), so a C2 investor is flagged for a violation."""
    profile = _profile(risk_level="C2")
    # UNKNOWN_SYM is NOT in the candidates list
    candidates = [
        _candidate("KNOWN", "R1", asset_class="bond"),
    ]
    portfolio = _portfolio(
        weights={"KNOWN": 0.5, "UNKNOWN_SYM": 0.5},
        class_weights={"equity": 0.5, "bond": 0.5},
        portfolio_r_level="R5",
    )
    verdict = check_suitability(profile, portfolio, candidates)
    # With fail-closed default of R5, UNKNOWN_SYM exceeds C2 → must be flagged
    assert verdict.matched is False, (
        "UNKNOWN_SYM absent from candidates should default to R5 and trigger a violation"
    )
    assert any("UNKNOWN_SYM" in v for v in verdict.violations), (
        f"Expected UNKNOWN_SYM in violations, got: {verdict.violations}"
    )
