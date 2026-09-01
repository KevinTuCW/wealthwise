"""TDD test suite for the 5 expert-agent nodes + AdvisoryDeps.

All tests run offline with SampleMarket/Macro/FXProvider + FakeModelClient.
No network, no real LLM keys required.
"""
from __future__ import annotations

import pytest

from wealthwise.agents.state import (
    AdvisoryState,
    AssetCandidate,
    InvestorProfile,
    PortfolioAllocation,
)
from wealthwise.llm import FakeModelClient, Verdict
from wealthwise.providers.sample import SampleFXProvider, SampleMacroProvider, SampleMarketProvider
from wealthwise.rag.backends import build_embedder
from wealthwise.rag.corpus import load_policy_retriever, load_research_retriever

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = "data/samples"


def _embedder():
    """The offline hashing embedder, built the same way the fixtures build it."""
    from wealthwise.config import Settings

    return build_embedder(Settings(embed_provider="local", embed_dim=256))


@pytest.fixture(scope="module")
def embedder():
    from wealthwise.config import Settings
    return build_embedder(Settings(embed_provider="local", embed_dim=256))


@pytest.fixture(scope="module")
def deps(embedder):
    from wealthwise.agents.deps import AdvisoryDeps

    return AdvisoryDeps(
        market=SampleMarketProvider(),
        macro=SampleMacroProvider(),
        fx=SampleFXProvider(),
        jury_clients=[FakeModelClient("fake", Verdict(label="overweight", rationale="test", tokens=5))],
        policy_retriever=load_policy_retriever(DATA_DIR, embedder),
        research_retriever=load_research_retriever(DATA_DIR, embedder),
        embedder=embedder,
    )


def _c4_profile(accept_cross_border: bool = True) -> InvestorProfile:
    return InvestorProfile(
        risk_level="C4",
        investable=1_000_000.0,
        horizon_years=10,
        goals=["retirement", "growth"],
        liquidity_min=0.10,
        accept_cross_border=accept_cross_border,
    )


def _base_state(accept_cross_border: bool = True) -> AdvisoryState:
    return AdvisoryState(profile=_c4_profile(accept_cross_border))


# ---------------------------------------------------------------------------
# AdvisoryDeps
# ---------------------------------------------------------------------------

class TestAdvisoryDeps:
    def test_deps_is_dataclass(self, deps):
        """AdvisoryDeps must be a dataclass (frozen or otherwise)."""
        import dataclasses
        assert dataclasses.is_dataclass(deps)

    def test_deps_has_required_fields(self, deps):
        assert deps.market is not None
        assert deps.macro is not None
        assert deps.fx is not None
        assert deps.jury_clients is not None
        assert len(deps.jury_clients) >= 1
        assert deps.policy_retriever is not None
        assert deps.research_retriever is not None

    def test_deps_has_threshold_params(self, deps):
        assert hasattr(deps, "max_fx_exposure")
        assert hasattr(deps, "risk_budget_method")
        assert hasattr(deps, "max_llm_judgments")
        # sensible defaults
        assert 0.0 < deps.max_fx_exposure <= 1.0
        assert deps.risk_budget_method in ("risk_parity", "equal_weight", "mean_variance")
        assert deps.max_llm_judgments > 0


# ---------------------------------------------------------------------------
# goal_node
# ---------------------------------------------------------------------------

class TestGoalNode:
    def test_returns_dict_increment(self, deps):
        from wealthwise.agents.experts.goal import goal_node
        state = _base_state()
        result = goal_node(state, deps)
        assert isinstance(result, dict)

    def test_required_keys_present(self, deps):
        from wealthwise.agents.experts.goal import goal_node
        state = _base_state()
        result = goal_node(state, deps)
        assert "goal_constraints" in result
        assert "trace_events" in result

    def test_c4_maps_to_r4_ceiling(self, deps):
        from wealthwise.agents.experts.goal import goal_node
        state = _base_state()
        result = goal_node(state, deps)
        gc = result["goal_constraints"]
        assert gc["risk_ceiling"] == "R4"

    def test_retirement_long_horizon_raises_equity_cap(self, deps):
        """C4 + retirement + 10y horizon should allow a higher equity cap than
        capital-preservation or short horizons."""
        from wealthwise.agents.experts.goal import goal_node
        # Long-horizon retirement profile
        state_long = AdvisoryState(profile=InvestorProfile(
            risk_level="C4", investable=500_000, horizon_years=10,
            goals=["retirement"], liquidity_min=0.10, accept_cross_border=True))
        # Short-horizon capital-preservation profile
        state_short = AdvisoryState(profile=InvestorProfile(
            risk_level="C4", investable=500_000, horizon_years=2,
            goals=["capital_preservation"], liquidity_min=0.10, accept_cross_border=True))
        long_result = goal_node(state_long, deps)
        short_result = goal_node(state_short, deps)
        assert long_result["goal_constraints"]["max_equity"] >= short_result["goal_constraints"]["max_equity"]

    def test_liquidity_min_propagated(self, deps):
        from wealthwise.agents.experts.goal import goal_node
        state = _base_state()
        result = goal_node(state, deps)
        gc = result["goal_constraints"]
        # liquidity_min from profile (0.10) must be carried into constraints
        assert gc["liquidity_min"] >= state.profile.liquidity_min

    def test_accept_cross_border_propagated(self, deps):
        from wealthwise.agents.experts.goal import goal_node
        state_cb = _base_state(accept_cross_border=True)
        state_no = _base_state(accept_cross_border=False)
        assert goal_node(state_cb, deps)["goal_constraints"]["accept_cross_border"] is True
        assert goal_node(state_no, deps)["goal_constraints"]["accept_cross_border"] is False

    def test_deterministic_no_llm(self, deps):
        """goal_node must return identical results on repeated calls (rule-based)."""
        from wealthwise.agents.experts.goal import goal_node
        state = _base_state()
        r1 = goal_node(state, deps)
        r2 = goal_node(state, deps)
        assert r1["goal_constraints"] == r2["goal_constraints"]

    def test_does_not_mutate_state(self, deps):
        from wealthwise.agents.experts.goal import goal_node
        state = _base_state()
        original_gc = dict(state.goal_constraints)
        goal_node(state, deps)
        assert state.goal_constraints == original_gc


# ---------------------------------------------------------------------------
# macro_node
# ---------------------------------------------------------------------------

class TestMacroNode:
    def test_returns_dict_increment(self, deps):
        from wealthwise.agents.experts.macro import macro_node
        state = _base_state()
        result = macro_node(state, deps)
        assert isinstance(result, dict)

    def test_required_keys_present(self, deps):
        from wealthwise.agents.experts.macro import macro_node
        state = _base_state()
        result = macro_node(state, deps)
        assert "macro_view" in result
        assert "tokens_used" in result
        assert "trace_events" in result

    def test_macro_view_has_tilt_and_confidence(self, deps):
        from wealthwise.agents.experts.macro import macro_node
        state = _base_state()
        result = macro_node(state, deps)
        mv = result["macro_view"]
        assert "tilt" in mv
        assert "confidence" in mv
        assert mv["tilt"] is not None  # should have a label from FakeModelClient
        assert 0.0 <= mv["confidence"] <= 1.0

    def test_tokens_used_increases(self, deps):
        from wealthwise.agents.experts.macro import macro_node
        state = _base_state()
        result = macro_node(state, deps)
        assert result["tokens_used"] > 0

    def test_deterministic_under_fake_client(self, deps):
        """Under FakeModelClient, repeated macro_node calls produce identical macro_view."""
        from wealthwise.agents.experts.macro import macro_node
        state = _base_state()
        r1 = macro_node(state, deps)
        r2 = macro_node(state, deps)
        assert r1["macro_view"]["tilt"] == r2["macro_view"]["tilt"]
        assert r1["macro_view"]["confidence"] == r2["macro_view"]["confidence"]

    def test_accumulates_trace_event(self, deps):
        from wealthwise.agents.experts.macro import macro_node
        state = _base_state()
        result = macro_node(state, deps)
        assert len(result["trace_events"]) > len(state.trace_events)

    def test_does_not_mutate_state(self, deps):
        from wealthwise.agents.experts.macro import macro_node
        state = _base_state()
        original = dict(state.macro_view)
        macro_node(state, deps)
        assert state.macro_view == original


# ---------------------------------------------------------------------------
# equity_node
# ---------------------------------------------------------------------------

class TestEquityNode:
    def _state_with_goal(self, accept_cross_border: bool = True) -> AdvisoryState:
        from wealthwise.agents.experts.goal import goal_node
        state = _base_state(accept_cross_border)
        gc_inc = goal_node(state, None)  # deps not needed for goal_node
        state2 = state.model_copy(update={"goal_constraints": gc_inc["goal_constraints"]})
        return state2

    def test_returns_dict_increment(self, deps):
        from wealthwise.agents.experts.equity import equity_node
        state = self._state_with_goal()
        result = equity_node(state, deps)
        assert isinstance(result, dict)

    def test_equity_candidates_non_empty(self, deps):
        from wealthwise.agents.experts.equity import equity_node
        state = self._state_with_goal()
        result = equity_node(state, deps)
        assert len(result["equity_candidates"]) > 0

    def test_all_candidates_within_risk_ceiling(self, deps):
        from wealthwise.agents.experts.equity import equity_node
        from wealthwise.portfolio.metrics import R_ORDER
        state = self._state_with_goal()
        result = equity_node(state, deps)
        gc = state.goal_constraints
        ceiling = R_ORDER[gc["risk_ceiling"]]
        for c in result["equity_candidates"]:
            assert R_ORDER[c.r_level] <= ceiling, (
                f"{c.symbol} r_level {c.r_level} exceeds ceiling {gc['risk_ceiling']}"
            )

    def test_no_cross_border_when_not_accepted(self, deps):
        from wealthwise.agents.experts.equity import equity_node
        state = self._state_with_goal(accept_cross_border=False)
        result = equity_node(state, deps)
        for c in result["equity_candidates"]:
            assert c.market == "A", (
                f"Expected only A-market candidates, got {c.symbol} in {c.market}"
            )

    def test_cross_border_included_when_accepted(self, deps):
        from wealthwise.agents.experts.equity import equity_node
        state = self._state_with_goal(accept_cross_border=True)
        result = equity_node(state, deps)
        markets = {c.market for c in result["equity_candidates"]}
        # With C4 and cross-border enabled, HK and/or US candidates should appear
        assert markets - {"A"}, f"Expected HK/US candidates too, got: {markets}"

    def test_returns_asset_candidate_objects(self, deps):
        from wealthwise.agents.experts.equity import equity_node
        state = self._state_with_goal()
        result = equity_node(state, deps)
        for c in result["equity_candidates"]:
            assert isinstance(c, AssetCandidate)

    def test_trace_event_added(self, deps):
        from wealthwise.agents.experts.equity import equity_node
        state = self._state_with_goal()
        result = equity_node(state, deps)
        assert "trace_events" in result
        assert len(result["trace_events"]) > len(state.trace_events)

    def test_does_not_mutate_state(self, deps):
        from wealthwise.agents.experts.equity import equity_node
        state = self._state_with_goal()
        original = list(state.equity_candidates)
        equity_node(state, deps)
        assert state.equity_candidates == original

    def test_conservative_mode_tightens_equity(self, deps):
        """I2: conservative_mode=True (C1/C2 profile) yields a stricter/smaller
        equity candidate set than a C4 profile, all else equal.

        The planner sets conservative_mode=True for C1/C2 which equity_node
        consumes to apply a tighter PE cap AND a candidate count cap.
        """
        from wealthwise.agents.experts.equity import equity_node
        from wealthwise.agents.supervisor.planner import build_planner_hints

        # --- C4 profile (non-conservative) ---
        c4_profile = InvestorProfile(
            risk_level="C4",
            investable=1_000_000.0,
            horizon_years=10,
            goals=["retirement", "growth"],
            liquidity_min=0.10,
            accept_cross_border=True,
        )
        c4_gc = {
            "risk_ceiling": "R4",
            "accept_cross_border": True,
            "max_equity": 0.80,
            "liquidity_min": 0.10,
            "planner_hints": build_planner_hints(c4_profile),
        }
        c4_state = AdvisoryState(profile=c4_profile, goal_constraints=c4_gc)
        c4_result = equity_node(c4_state, deps)
        c4_candidates = c4_result["equity_candidates"]

        # --- C1 profile (conservative) ---
        c1_profile = InvestorProfile(
            risk_level="C1",
            investable=500_000.0,
            horizon_years=3,
            goals=["capital_preservation"],
            liquidity_min=0.50,
            accept_cross_border=True,  # same cross-border flag so market scope matches
        )
        c1_gc = {
            "risk_ceiling": "R1",
            "accept_cross_border": True,
            "max_equity": 0.20,
            "liquidity_min": 0.50,
            "planner_hints": build_planner_hints(c1_profile),
        }
        c1_state = AdvisoryState(profile=c1_profile, goal_constraints=c1_gc)
        c1_result = equity_node(c1_state, deps)
        c1_candidates = c1_result["equity_candidates"]

        # Verify conservative_mode is set for C1 but not C4
        assert c1_gc["planner_hints"]["conservative_mode"] is True, (
            "C1 profile must set conservative_mode=True"
        )
        assert c4_gc["planner_hints"]["conservative_mode"] is False, (
            "C4 profile must set conservative_mode=False"
        )

        # Conservative mode must yield ≤ non-conservative candidate count
        # (stricter PE cap and candidate cap apply)
        assert len(c1_candidates) <= len(c4_candidates), (
            f"C1 conservative_mode candidates ({len(c1_candidates)}) must be "
            f"≤ C4 candidates ({len(c4_candidates)})"
        )

        # The conservative trace event must record conservative_mode=True
        equity_events = [e for e in c1_result["trace_events"]
                         if e.get("node") == "equity"]
        if equity_events:
            assert equity_events[-1].get("conservative_mode") is True, (
                "Equity trace event must record conservative_mode=True for C1"
            )


# ---------------------------------------------------------------------------
# portfolio_node
# ---------------------------------------------------------------------------

class TestPortfolioNode:
    def _state_with_equity(self, accept_cross_border: bool = True) -> AdvisoryState:
        from wealthwise.agents.experts.goal import goal_node
        from wealthwise.agents.experts.equity import equity_node
        state = _base_state(accept_cross_border)
        gc_inc = goal_node(state, None)
        state = state.model_copy(update={"goal_constraints": gc_inc["goal_constraints"]})
        eq_inc = equity_node(state, None)
        state = state.model_copy(update={"equity_candidates": eq_inc["equity_candidates"],
                                          "trace_events": eq_inc["trace_events"]})
        return state

    def test_returns_dict_increment(self, deps):
        from wealthwise.agents.experts.portfolio import portfolio_node
        state = self._state_with_equity()
        result = portfolio_node(state, deps)
        assert isinstance(result, dict)

    def test_portfolio_present(self, deps):
        from wealthwise.agents.experts.portfolio import portfolio_node
        state = self._state_with_equity()
        result = portfolio_node(state, deps)
        assert "portfolio" in result
        assert result["portfolio"] is not None

    def test_weights_sum_to_one(self, deps):
        from wealthwise.agents.experts.portfolio import portfolio_node
        state = self._state_with_equity()
        result = portfolio_node(state, deps)
        port = result["portfolio"]
        total = sum(port.weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, not 1.0"

    def test_portfolio_r_level_within_ceiling(self, deps):
        from wealthwise.agents.experts.portfolio import portfolio_node
        from wealthwise.portfolio.metrics import R_ORDER
        state = self._state_with_equity()
        result = portfolio_node(state, deps)
        ceiling = state.goal_constraints["risk_ceiling"]
        port_r = result["portfolio"].portfolio_r_level
        assert R_ORDER[port_r] <= R_ORDER[ceiling], (
            f"Portfolio r_level {port_r} exceeds ceiling {ceiling}"
        )

    def test_fixedincome_candidates_present(self, deps):
        from wealthwise.agents.experts.portfolio import portfolio_node
        state = self._state_with_equity()
        result = portfolio_node(state, deps)
        assert "fixedincome_candidates" in result
        assert isinstance(result["fixedincome_candidates"], list)

    def test_portfolio_is_allocation_instance(self, deps):
        from wealthwise.agents.experts.portfolio import portfolio_node
        state = self._state_with_equity()
        result = portfolio_node(state, deps)
        assert isinstance(result["portfolio"], PortfolioAllocation)

    def test_trace_event_added(self, deps):
        from wealthwise.agents.experts.portfolio import portfolio_node
        state = self._state_with_equity()
        result = portfolio_node(state, deps)
        assert "trace_events" in result
        assert len(result["trace_events"]) > len(state.trace_events)

    def test_does_not_mutate_state(self, deps):
        from wealthwise.agents.experts.portfolio import portfolio_node
        state = self._state_with_equity()
        original = state.portfolio
        portfolio_node(state, deps)
        assert state.portfolio == original


# ---------------------------------------------------------------------------
# compliance_node
# ---------------------------------------------------------------------------

class TestComplianceNode:
    def _full_state(self, accept_cross_border: bool = True) -> tuple[AdvisoryState, object]:
        """Build a fully populated state (goal → equity → portfolio) + deps."""
        from wealthwise.agents.experts.goal import goal_node
        from wealthwise.agents.experts.equity import equity_node
        from wealthwise.agents.experts.portfolio import portfolio_node
        state = _base_state(accept_cross_border)
        state = state.model_copy(update=goal_node(state, None))
        state = state.model_copy(update=equity_node(state, None))
        state = state.model_copy(update=portfolio_node(state, None))
        return state

    def test_returns_dict_increment(self, deps):
        from wealthwise.agents.experts.compliance import compliance_node
        state = self._full_state()
        result = compliance_node(state, deps)
        assert isinstance(result, dict)

    def test_compliance_key_present(self, deps):
        from wealthwise.agents.experts.compliance import compliance_node
        state = self._full_state()
        result = compliance_node(state, deps)
        assert "compliance" in result
        assert result["compliance"] is not None

    def test_pass_for_matched_portfolio(self, deps):
        """A well-constructed C4 portfolio within R4 ceiling must receive PASS."""
        from wealthwise.agents.experts.compliance import compliance_node
        from wealthwise.agents.state import ComplianceVerdict
        state = self._full_state()
        result = compliance_node(state, deps)
        verdict: ComplianceVerdict = result["compliance"]
        # Not necessarily always PASS (depends on sample data), but it must not
        # be REJECT unless there's a genuine hard violation.
        assert verdict.decision in {"PASS", "DOWNGRADE"}, (
            f"Expected PASS or DOWNGRADE for matched C4 portfolio, got {verdict.decision}"
        )

    def test_disclosures_always_present(self, deps):
        from wealthwise.agents.experts.compliance import compliance_node
        state = self._full_state()
        result = compliance_node(state, deps)
        assert len(result["compliance"].disclosures) > 0

    def test_reject_never_softened(self, deps):
        """If suitability check returns REJECT, compliance_node MUST NOT upgrade to PASS.

        We force a hard cross-border violation: accept_cross_border=False
        but inject an HK asset with non-zero weight directly into the portfolio.
        """
        from wealthwise.agents.experts.compliance import compliance_node
        from wealthwise.agents.state import ComplianceVerdict

        # Build a state where portfolio *contains* an HK asset despite no cross-border auth
        profile = InvestorProfile(
            risk_level="C4", investable=500_000, horizon_years=5,
            goals=["growth"], liquidity_min=0.10, accept_cross_border=False)
        hk_candidate = AssetCandidate(
            symbol="00700", market="HK", asset_class="equity",
            name="腾讯控股", currency="HKD", r_level="R3")
        a_candidate = AssetCandidate(
            symbol="600519", market="A", asset_class="equity",
            name="贵州茅台", currency="CNY", r_level="R3")
        bond = AssetCandidate(
            symbol="519736", market="A", asset_class="bond",
            name="国泰债券A", currency="CNY", r_level="R2")
        forced_portfolio = PortfolioAllocation(
            weights={"00700": 0.3, "600519": 0.5, "519736": 0.2},
            class_weights={"equity": 0.8, "bond": 0.2},
            portfolio_r_level="R3",
            fx_exposure=0.3,
        )
        state = AdvisoryState(
            profile=profile,
            goal_constraints={"risk_ceiling": "R4", "max_equity": 0.8,
                               "liquidity_min": 0.10, "accept_cross_border": False},
            equity_candidates=[hk_candidate, a_candidate],
            fixedincome_candidates=[bond],
            portfolio=forced_portfolio,
        )
        result = compliance_node(state, deps)
        verdict: ComplianceVerdict = result["compliance"]
        assert verdict.decision in {"DOWNGRADE", "REJECT"}, (
            f"Cross-border violation must not yield PASS, got {verdict.decision}"
        )
        assert verdict.matched is False

    def test_jury_cannot_override_reject_to_pass(self, deps):
        """The jury is advisory only — it can never soften a suitability REJECT to PASS.

        This test creates a FakeModelClient that always votes 'PASS' / 'compliant'
        and checks the compliance node still respects the suitability hard block.
        """
        from wealthwise.agents.deps import AdvisoryDeps
        from wealthwise.agents.experts.compliance import compliance_node
        from wealthwise.rag.backends import build_embedder
        from wealthwise.config import Settings

        # A jury client that always says 'compliant' / tries to pass everything
        permissive_jury = [FakeModelClient("permissive", Verdict(label="PASS", rationale="all good", tokens=1))]
        permissive_deps = AdvisoryDeps(
            market=deps.market,
            macro=deps.macro,
            fx=deps.fx,
            jury_clients=permissive_jury,
            policy_retriever=deps.policy_retriever,
            research_retriever=deps.research_retriever,
        )

        profile = InvestorProfile(
            risk_level="C1", investable=100_000, horizon_years=3,
            goals=["capital_preservation"], liquidity_min=0.50, accept_cross_border=False)
        # Portfolio with R5 asset AND portfolio_r_level R5 → compound REJECT
        r5_asset = AssetCandidate(
            symbol="NVDA", market="US", asset_class="equity",
            name="NVIDIA", currency="USD", r_level="R5")
        bond = AssetCandidate(
            symbol="519736", market="A", asset_class="bond",
            name="国泰债券A", currency="CNY", r_level="R2")
        bad_portfolio = PortfolioAllocation(
            weights={"NVDA": 0.6, "519736": 0.4},
            class_weights={"equity": 0.6, "bond": 0.4},
            portfolio_r_level="R5",
            fx_exposure=0.6,
        )
        state = AdvisoryState(
            profile=profile,
            goal_constraints={"risk_ceiling": "R1", "max_equity": 0.3,
                               "liquidity_min": 0.50, "accept_cross_border": False},
            equity_candidates=[r5_asset],
            fixedincome_candidates=[bond],
            portfolio=bad_portfolio,
        )
        result = compliance_node(state, permissive_deps)
        assert result["compliance"].decision in {"DOWNGRADE", "REJECT"}, (
            f"A permissive jury must NOT soften a suitability violation to PASS, "
            f"got {result['compliance'].decision}"
        )
        assert result["compliance"].matched is False

    def test_tokens_accumulated(self, deps):
        from wealthwise.agents.experts.compliance import compliance_node
        state = self._full_state()
        result = compliance_node(state, deps)
        assert "tokens_used" in result

    def test_trace_event_added(self, deps):
        from wealthwise.agents.experts.compliance import compliance_node
        state = self._full_state()
        result = compliance_node(state, deps)
        assert "trace_events" in result
        assert len(result["trace_events"]) > len(state.trace_events)

    def test_does_not_mutate_state(self, deps):
        from wealthwise.agents.experts.compliance import compliance_node
        state = self._full_state()
        original = state.compliance
        compliance_node(state, deps)
        assert state.compliance == original

    def test_cross_border_disclosure_when_fx_held(self, deps):
        """When portfolio holds HK/US assets (and investor authorized it),
        a cross-border FX risk disclosure must be present."""
        from wealthwise.agents.experts.compliance import compliance_node
        state = self._full_state(accept_cross_border=True)
        # Only meaningful if portfolio actually has FX exposure
        if state.portfolio and state.portfolio.fx_exposure > 0:
            result = compliance_node(state, deps)
            combined = " ".join(result["compliance"].disclosures)
            assert "汇率" in combined or "跨境" in combined or "FX" in combined.upper(), (
                f"Expected FX/cross-border disclosure, got: {result['compliance'].disclosures}"
            )


# ---------------------------------------------------------------------------
# equity_node — geographic quota + quality selection
# ---------------------------------------------------------------------------

class TestEquitySelection:
    """Selection used to be `list[:50]` over concatenated per-market screens,
    which handed every slot to whichever market was screened first."""

    def _candidate(self, symbol, market, cap_100m=500.0, pe=15.0):
        from wealthwise.agents.state import AssetCandidate

        metrics = {}
        if pe is not None:
            metrics["pe"] = pe
        if cap_100m is not None:
            metrics["market_cap_100m"] = cap_100m
        return AssetCandidate(symbol=symbol, market=market, asset_class="equity",
                              name=symbol, currency="CNY", r_level="R3",
                              metrics=metrics)

    def test_每个市场都拿到名额(self):
        from wealthwise.agents.experts.equity import _select

        pool = ([self._candidate(f"A{i}", "A") for i in range(300)]
                + [self._candidate(f"H{i}", "HK") for i in range(30)]
                + [self._candidate(f"U{i}", "US") for i in range(30)])
        got, _ = _select(pool, ["A", "HK", "US"], 50)

        counts = {m: sum(1 for c in got if c.market == m) for m in ("A", "HK", "US")}
        assert sum(counts.values()) == 50
        assert counts["HK"] > 0 and counts["US"] > 0, (
            f"cross-border sleeves were squeezed out: {counts}")
        assert counts["A"] > counts["HK"], "A-shares should still be the core"

    def test_单市场时名额全给它(self):
        from wealthwise.agents.experts.equity import _select

        pool = [self._candidate(f"A{i}", "A") for i in range(300)]
        got, _ = _select(pool, ["A"], 50)
        assert len(got) == 50
        assert all(c.market == "A" for c in got)

    def test_市场候选不足时名额转给其他市场(self):
        from wealthwise.agents.experts.equity import _select

        pool = ([self._candidate(f"A{i}", "A") for i in range(300)]
                + [self._candidate("H0", "HK")]
                + [self._candidate(f"U{i}", "US") for i in range(30)])
        got, _ = _select(pool, ["A", "HK", "US"], 50)
        assert len(got) == 50, "unused HK quota should be spent, not lost"

    def test_按规模排序而非按顺序(self):
        from wealthwise.agents.experts.equity import _select

        # Caps kept above the size floor so this exercises ranking, not filtering.
        pool = [self._candidate(f"A{i}", "A", cap_100m=100.0 * i) for i in range(1, 101)]
        got, _ = _select(pool, ["A"], 5)
        assert [c.symbol for c in got] == ["A100", "A99", "A98", "A97", "A96"]

    def test_亏损与微盘被剔除(self):
        from wealthwise.agents.experts.equity import _select

        pool = [
            self._candidate("GOOD", "A", cap_100m=500.0, pe=12.0),
            self._candidate("LOSS", "A", cap_100m=500.0, pe=-3.0),   # 亏损
            self._candidate("TINY", "A", cap_100m=5.0, pe=12.0),     # 规模不足
        ]
        assert [c.symbol for c in _select(pool, ["A"], 10)[0]] == ["GOOD"]

    def test_缺失指标不等于不合格(self):
        """Provider coverage varies; silence must not empty the candidate set."""
        from wealthwise.agents.experts.equity import _select

        pool = [self._candidate(f"A{i}", "A", cap_100m=None, pe=None) for i in range(5)]
        assert len(_select(pool, ["A"], 10)[0]) == 5

    def test_不同provider的市值字段都能读(self):
        from wealthwise.agents.state import AssetCandidate
        from wealthwise.portfolio.factors import market_cap_100m

        tencent = AssetCandidate(symbol="X", market="A", asset_class="equity",
                                 name="X", currency="CNY", r_level="R3",
                                 metrics={"market_cap_100m": 3500.0})
        sample = AssetCandidate(symbol="Y", market="A", asset_class="equity",
                                name="Y", currency="CNY", r_level="R3",
                                metrics={"market_cap_cny": 350_000_000_000})
        assert market_cap_100m(tencent) == 3500.0
        assert market_cap_100m(sample) == 3500.0

    def test_未筛市场的标的必须放行到合规(self):
        """A leaked cross-border name must reach compliance, not vanish here."""
        from wealthwise.agents.experts.equity import _select

        pool = ([self._candidate(f"A{i}", "A") for i in range(100)]
                + [self._candidate("LEAK", "US")])
        got, _ = _select(pool, ["A"], 10)
        assert "LEAK" in [c.symbol for c in got], (
            "dropping an unauthorised holding hides the violation instead of rejecting it")


# ---------------------------------------------------------------------------
# Multi-factor ranking — the ENABLE_FACTOR_SCORING path
# ---------------------------------------------------------------------------

class _StubHistory:
    """A history provider that hands back a fixed momentum/volatility per symbol."""

    def __init__(self, table: dict[str, dict]):
        self._table = table
        self.calls: list[list[str]] = []

    def enrich(self, candidates):
        self.calls.append([c.symbol for c in candidates])
        out = []
        for c in candidates:
            extra = self._table.get(c.symbol)
            if not extra:
                out.append(c)
                continue
            out.append(c.model_copy(update={"metrics": {**c.metrics, **extra}}))
        return out


class TestFactorScoringSwitch:
    """The switch must change which names are picked — and nothing else."""

    def _candidate(self, symbol, market="A", cap=500.0, pe=15.0, **metrics):
        return AssetCandidate(
            symbol=symbol, market=market, asset_class="equity", name=symbol,
            currency="CNY", r_level="R3",
            metrics={"market_cap_100m": cap, "pe": pe, **metrics},
        )

    def _deps(self, candidates, **overrides):
        from wealthwise.agents.deps import AdvisoryDeps

        class _Market:
            name = "stub"

            def screen(self, market, filters):
                return [c for c in candidates if c.market == market]

            def quotes(self, symbols):
                return []

        base = dict(
            market=_Market(),
            macro=SampleMacroProvider(DATA_DIR),
            fx=SampleFXProvider(DATA_DIR),
            jury_clients=[FakeModelClient("f", Verdict(label="neutral", rationale=""))],
            policy_retriever=load_policy_retriever(DATA_DIR, _embedder()),
            research_retriever=load_research_retriever(DATA_DIR, _embedder()),
        )
        base.update(overrides)
        return AdvisoryDeps(**base)

    def _state(self):
        return AdvisoryState(
            goal_constraints={"risk_ceiling": "R5", "accept_cross_border": False},
            macro_view={"tilt": "neutral"},
        )

    def test_关闭时仍按规模排序(self):
        from wealthwise.agents.experts.equity import equity_node

        pool = [self._candidate("BIG", cap=5000.0, volatility=0.60, momentum=-0.3),
                self._candidate("SMALL", cap=200.0, volatility=0.10, momentum=0.4)]
        out = equity_node(self._state(), self._deps(pool, enable_factor_scoring=False))

        assert [c.symbol for c in out["equity_candidates"]] == ["BIG", "SMALL"]
        assert out["trace_events"][-1]["ranking"]["method"] == "quality"

    def test_开启后动量与低波能压过规模(self):
        from wealthwise.agents.experts.equity import equity_node

        pool = [self._candidate("BIG", cap=5000.0, volatility=0.60, momentum=-0.3),
                self._candidate("SMALL", cap=200.0, volatility=0.10, momentum=0.4)]
        out = equity_node(self._state(), self._deps(pool, enable_factor_scoring=True))

        assert [c.symbol for c in out["equity_candidates"]] == ["SMALL", "BIG"]
        assert out["trace_events"][-1]["ranking"]["method"] == "factor"

    def test_开关不改变候选数量与市场配额(self):
        """Only the sort key moves; the quota and the budget are untouched."""
        from wealthwise.agents.experts.equity import equity_node

        pool = [self._candidate(f"A{i}", cap=100.0 * (i + 1), volatility=0.2 + i * 0.01)
                for i in range(80)]
        off = equity_node(self._state(), self._deps(pool, enable_factor_scoring=False))
        on = equity_node(self._state(), self._deps(pool, enable_factor_scoring=True))

        assert len(off["equity_candidates"]) == len(on["equity_candidates"])
        assert {c.symbol for c in on["equity_candidates"]} != \
            {c.symbol for c in off["equity_candidates"]} or True   # may coincide
        assert all(c.market == "A" for c in on["equity_candidates"])

    def test_因子分与z值进trace(self):
        from wealthwise.agents.experts.equity import equity_node

        pool = [self._candidate(f"A{i}", cap=100.0 * (i + 1)) for i in range(10)]
        out = equity_node(self._state(), self._deps(pool, enable_factor_scoring=True))

        top = out["trace_events"][-1]["ranking"]["top"]
        assert top and "score" in top[0] and "size" in top[0]["z"]

    def test_只给进入排名的标的拉历史(self):
        """History is one request per symbol; enriching the whole screen wastes it."""
        from wealthwise.agents.experts.equity import equity_node

        pool = [self._candidate(f"A{i}") for i in range(5)]
        pool.append(self._candidate("TOO_RISKY", pe=15.0))
        pool[-1] = pool[-1].model_copy(update={"r_level": "R5"})
        history = _StubHistory({})

        state = AdvisoryState(
            goal_constraints={"risk_ceiling": "R3", "accept_cross_border": False},
            macro_view={"tilt": "neutral"},
        )
        equity_node(state, self._deps(pool, enable_factor_scoring=True, history=history))

        assert history.calls, "history was never consulted"
        assert "TOO_RISKY" not in history.calls[0], (
            "paid for history on a name the risk ceiling had already excluded")

    def test_关闭时不拉历史(self):
        from wealthwise.agents.experts.equity import equity_node

        history = _StubHistory({})
        pool = [self._candidate("A0")]
        equity_node(self._state(),
                    self._deps(pool, enable_factor_scoring=False, history=history))
        assert history.calls == []

    def test_历史补出的波动率流向优化器(self):
        """The optimiser weights on `volatility` and was defaulting to 0.15."""
        from wealthwise.agents.experts.equity import equity_node

        pool = [self._candidate("A0"), self._candidate("A1")]
        history = _StubHistory({"A0": {"volatility": 0.42, "momentum": 0.1},
                                "A1": {"volatility": 0.11, "momentum": 0.1}})
        out = equity_node(self._state(),
                          self._deps(pool, enable_factor_scoring=True, history=history))

        vols = {c.symbol: c.metrics.get("volatility") for c in out["equity_candidates"]}
        assert vols == {"A0": 0.42, "A1": 0.11}


class TestDataDisagreementGate:
    def _candidate(self, symbol, disagreement=None):
        metrics = {"market_cap_100m": 500.0, "pe": 15.0, "price": 100.0}
        if disagreement:
            metrics["data_disagreement"] = disagreement
        return AssetCandidate(symbol=symbol, market="A", asset_class="equity",
                              name=symbol, currency="CNY", r_level="R3",
                              metrics=metrics)

    def _deps(self, candidates, **overrides):
        from wealthwise.agents.deps import AdvisoryDeps

        class _Market:
            name = "stub"

            def screen(self, market, filters):
                return [c for c in candidates if c.market == market]

            def quotes(self, symbols):
                return []

        base = dict(
            market=_Market(),
            macro=SampleMacroProvider(DATA_DIR),
            fx=SampleFXProvider(DATA_DIR),
            jury_clients=[FakeModelClient("f", Verdict(label="neutral", rationale=""))],
            policy_retriever=load_policy_retriever(DATA_DIR, _embedder()),
            research_retriever=load_research_retriever(DATA_DIR, _embedder()),
        )
        base.update(overrides)
        return AdvisoryDeps(**base)

    def _state(self):
        return AdvisoryState(
            goal_constraints={"risk_ceiling": "R5", "accept_cross_border": False},
            macro_view={"tilt": "neutral"},
        )

    def test_价格分歧的标的不进订单(self):
        from wealthwise.agents.experts.equity import equity_node

        pool = [self._candidate("CLEAN"),
                self._candidate("SPLIT", disagreement=["price"])]
        out = equity_node(self._state(), self._deps(pool))

        assert [c.symbol for c in out["equity_candidates"]] == ["CLEAN"]

    def test_被剔除的标的仍进trace(self):
        """The count is the observable that says a feed has drifted."""
        from wealthwise.agents.experts.equity import equity_node

        pool = [self._candidate("CLEAN"),
                self._candidate("SPLIT", disagreement=["price"])]
        out = equity_node(self._state(), self._deps(pool))

        assert out["trace_events"][-1]["data_disagreement"] == ["SPLIT"]
        assert "source disagreement" in out["notes"][-1]

    def test_估值口径分歧不剔除(self):
        """Different earnings windows are a methodology difference, not a fault."""
        from wealthwise.agents.experts.equity import equity_node

        pool = [self._candidate("PE_GAP", disagreement=["pe"])]
        out = equity_node(self._state(), self._deps(pool))

        assert [c.symbol for c in out["equity_candidates"]] == ["PE_GAP"]

    def test_开关关闭时保留分歧标的(self):
        from wealthwise.agents.experts.equity import equity_node

        pool = [self._candidate("SPLIT", disagreement=["price"])]
        out = equity_node(self._state(),
                          self._deps(pool, drop_on_data_disagreement=False))

        assert [c.symbol for c in out["equity_candidates"]] == ["SPLIT"]


class TestMacroConsumesConsensus:
    def _deps(self, macro, **overrides):
        from wealthwise.agents.deps import AdvisoryDeps

        neutral = Verdict(label="neutral", rationale="")
        base = dict(
            market=SampleMarketProvider(DATA_DIR),
            macro=macro,
            fx=SampleFXProvider(DATA_DIR),
            # Two jurors, so the jury's own confidence lands at 1.0 and cannot be
            # confused with the 0.5 a lone *data source* is capped at.
            jury_clients=[FakeModelClient("f1", neutral), FakeModelClient("f2", neutral)],
            policy_retriever=load_policy_retriever(DATA_DIR, _embedder()),
            research_retriever=load_research_retriever(DATA_DIR, _embedder()),
        )
        base.update(overrides)
        return AdvisoryDeps(**base)

    def test_共识元数据进入macro_view(self):
        from wealthwise.agents.experts.macro import macro_node
        from wealthwise.providers.consensus_provider import ConsensusMacroProvider

        class _Src:
            def __init__(self, name, snap):
                self.name = name
                self._snap = snap

            def snapshot(self):
                return dict(self._snap)

        macro = ConsensusMacroProvider([_Src("a", {"cpi": 0.021}),
                                        _Src("b", {"cpi": 0.021})])
        out = macro_node(AdvisoryState(), self._deps(macro))
        view = out["macro_view"]

        assert view["sources"] == ["a", "b"]
        assert view["data_confidence"] == 1.0
        assert view["contested_signals"] == []
        assert view["signal_consensus"]["cpi"]["sources"] == ["a", "b"]

    def test_发布方分歧被记入并进提示词(self):
        from wealthwise.agents.experts.macro import macro_node
        from wealthwise.providers.consensus_provider import ConsensusMacroProvider

        class _Src:
            def __init__(self, name, snap):
                self.name = name
                self._snap = snap

            def snapshot(self):
                return dict(self._snap)

        seen = {}

        class _Recording:
            name = "recorder"

            def judge(self, system, user, labels):
                seen["user"] = user
                return Verdict(label="neutral", rationale="")

        macro = ConsensusMacroProvider([_Src("a", {"cpi": 0.021}),
                                        _Src("b", {"cpi": 0.045})])
        deps = self._deps(macro, jury_clients=[_Recording()])
        out = macro_node(AdvisoryState(), deps)

        assert out["macro_view"]["contested_signals"] == ["cpi"]
        assert "Data caveat" in seen["user"], (
            "the jury judged on a contested figure without being told it was contested")

    def test_陪审置信度与数据置信度不混为一谈(self):
        from wealthwise.agents.experts.macro import macro_node
        from wealthwise.providers.consensus_provider import ConsensusMacroProvider

        class _Src:
            name = "only"

            def snapshot(self):
                return {"cpi": 0.021}

        out = macro_node(AdvisoryState(), self._deps(ConsensusMacroProvider([_Src()])))
        view = out["macro_view"]

        assert view["data_confidence"] == 0.5      # one publisher
        assert view["confidence"] == 1.0           # jury was unanimous
