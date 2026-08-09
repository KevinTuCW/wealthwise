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
