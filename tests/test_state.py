"""Tests for AdvisoryState data contract — Task 1.1."""
import pytest
from pydantic import ValidationError


def test_investor_profile_valid():
    from wealthwise.agents.state import InvestorProfile

    p = InvestorProfile(
        risk_level="C3",
        investable=500000,
        horizon_years=5,
        goals=["retirement"],
        liquidity_min=0.1,
        accept_cross_border=True,
    )
    assert p.risk_level == "C3"
    assert p.investable == 500000.0
    assert p.horizon_years == 5
    assert p.goals == ["retirement"]
    assert p.liquidity_min == 0.1
    assert p.accept_cross_border is True
    assert p.holdings == []  # default


def test_investor_profile_invalid_risk_level():
    from wealthwise.agents.state import InvestorProfile

    with pytest.raises(ValidationError):
        InvestorProfile(
            risk_level="C6",  # invalid — only C1..C5
            investable=100000,
            horizon_years=3,
            goals=["growth"],
            liquidity_min=0.05,
            accept_cross_border=False,
        )


def test_advisory_state_defaults():
    from wealthwise.agents.state import AdvisoryState

    state = AdvisoryState()
    assert state.status == "pending"
    assert state.trace_events == []
    assert state.budget_spent == 0
    assert state.tokens_used == 0
    assert state.equity_candidates == []
    assert state.fixedincome_candidates == []
    assert state.profile is None
    assert state.portfolio is None
    assert state.compliance is None
    assert state.explanation == ""
    assert state.confidence == 0.0
    assert state.notes == []


def test_asset_candidate_valid():
    from wealthwise.agents.state import AssetCandidate

    c = AssetCandidate(
        symbol="600519",
        market="A",
        asset_class="equity",
        name="贵州茅台",
        currency="CNY",
        r_level="R3",
    )
    assert c.market == "A"
    assert c.asset_class == "equity"
    assert c.r_level == "R3"
    assert c.metrics == {}
    assert c.tags == []


def test_asset_candidate_invalid_market():
    from wealthwise.agents.state import AssetCandidate

    with pytest.raises(ValidationError):
        AssetCandidate(
            symbol="AAPL",
            market="JP",  # invalid — only A/HK/US
            asset_class="equity",
            name="Apple",
            currency="USD",
            r_level="R2",
        )


def test_portfolio_allocation_valid():
    from wealthwise.agents.state import PortfolioAllocation

    pa = PortfolioAllocation(
        weights={"600519": 0.6, "510300": 0.4},
        class_weights={"equity": 1.0},
        portfolio_r_level="R3",
        fx_exposure=0.0,
    )
    assert pa.weights["600519"] == 0.6
    assert pa.fx_exposure == 0.0


def test_compliance_verdict_valid():
    from wealthwise.agents.state import ComplianceVerdict

    cv = ComplianceVerdict(decision="PASS", matched=True, confidence=0.95)
    assert cv.decision == "PASS"
    assert cv.violations == []
    assert cv.disclosures == []


def test_compliance_verdict_invalid_decision():
    from wealthwise.agents.state import ComplianceVerdict

    with pytest.raises(ValidationError):
        ComplianceVerdict(decision="MAYBE", matched=True, confidence=0.5)


def test_advisory_state_mutable_defaults_are_independent():
    """Each AdvisoryState instance must get its own list/dict, not a shared one."""
    from wealthwise.agents.state import AdvisoryState

    s1 = AdvisoryState()
    s2 = AdvisoryState()
    s1.notes.append("x")
    assert s2.notes == [], "mutable default leaked between instances"
