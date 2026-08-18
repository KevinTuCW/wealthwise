"""TDD tests for wealthwise.workbench — build_dashboard + sse_events."""
from __future__ import annotations

import json

import pytest

from wealthwise.agents.state import (
    AdvisoryState,
    AssetCandidate,
    ComplianceVerdict,
    InvestorProfile,
    PortfolioAllocation,
)
from wealthwise.bootstrap import build_sample_deps
from wealthwise.config import get_settings
from wealthwise.workbench import build_dashboard, sse_events


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _c3_profile(accept_cross_border: bool = True) -> InvestorProfile:
    return InvestorProfile(
        risk_level="C3",
        investable=500_000.0,
        horizon_years=5,
        goals=["balanced_growth"],
        liquidity_min=0.2,
        accept_cross_border=accept_cross_border,
    )


def _c3_cross_border_state() -> AdvisoryState:
    """Build a realistic completed AdvisoryState with cross-border assets."""
    profile = _c3_profile(accept_cross_border=True)

    eq_candidates = [
        AssetCandidate(symbol="AAPL", market="US", asset_class="equity",
                       name="Apple Inc.", currency="USD", r_level="R3"),
        AssetCandidate(symbol="600519", market="A", asset_class="equity",
                       name="贵州茅台", currency="CNY", r_level="R3"),
    ]
    fi_candidates = [
        AssetCandidate(symbol="519736", market="A", asset_class="bond",
                       name="债券基金A", currency="CNY", r_level="R2"),
        AssetCandidate(symbol="000198", market="A", asset_class="cash",
                       name="货币基金", currency="CNY", r_level="R1"),
    ]

    portfolio = PortfolioAllocation(
        weights={"AAPL": 0.1, "600519": 0.2, "519736": 0.3, "000198": 0.4},
        class_weights={"equity": 0.3, "bond": 0.3, "cash": 0.4},
        portfolio_r_level="R3",
        fx_exposure=0.1,
        metrics={
            "volatility": 0.08,
            "sharpe": 1.5,
            "max_drawdown": -0.12,
            "diversification": 1.8,
        },
    )

    compliance = ComplianceVerdict(
        decision="PASS",
        matched=True,
        violations=[],
        disclosures=[
            "投资者风险等级 C3，组合风险等级 R3，符合适当性匹配要求。",
            "投资有风险，入市须谨慎，过往业绩不代表未来表现。",
            "本内容不构成投资建议，仅供参考，请结合自身情况审慎决策。",
            "跨境标的涉及汇率波动、通道（港股通/QDII）与税收风险。",
        ],
        confidence=1.0,
    )

    import time
    trace_events = [
        {"node": "intake", "ts": time.time(), "status": "OK", "budget_spent": 0},
        {"node": "input_guard", "ts": time.time() + 0.01, "status": "OK", "budget_spent": 0},
        {"node": "macro", "ts": time.time() + 0.05, "status": "OK", "budget_spent": 2},
        {"node": "compliance", "ts": time.time() + 0.10, "status": "OK", "budget_spent": 4},
    ]

    return AdvisoryState(
        profile=profile,
        equity_candidates=eq_candidates,
        fixedincome_candidates=fi_candidates,
        portfolio=portfolio,
        compliance=compliance,
        macro_view={"tilt": "neutral", "confidence": 0.7, "regime": "stable"},
        status="done",
        tokens_used=1200,
        trace_events=trace_events,
        budget_spent=4,
        explanation="本内容不构成投资建议。",
        confidence=0.85,
    )


# ---------------------------------------------------------------------------
# build_dashboard — five panels
# ---------------------------------------------------------------------------

class TestBuildDashboard:
    def setup_method(self):
        self.state = _c3_cross_border_state()
        self.settings = get_settings()
        self.dash = build_dashboard(self.state, self.settings)

    def test_has_all_five_panels(self):
        assert set(self.dash) >= {"allocation", "experts", "crosscheck", "compliance", "cost"}

    # ---- 1. allocation panel ----
    def test_allocation_has_required_keys(self):
        alloc = self.dash["allocation"]
        for key in ("portfolio_r_level", "class_weights", "weights", "fx_exposure", "metrics"):
            assert key in alloc, f"allocation missing key: {key}"

    def test_allocation_portfolio_r_level(self):
        assert self.dash["allocation"]["portfolio_r_level"] == "R3"

    def test_allocation_class_weights_present(self):
        cw = self.dash["allocation"]["class_weights"]
        assert isinstance(cw, dict)
        assert "equity" in cw and "bond" in cw and "cash" in cw

    def test_allocation_metrics_present(self):
        m = self.dash["allocation"]["metrics"]
        # volatility, sharpe, max_drawdown, diversification all provided by fixture
        assert m.get("volatility") is not None
        assert m.get("sharpe") is not None

    # ---- 2. experts panel ----
    def test_experts_has_required_keys(self):
        exp = self.dash["experts"]
        for key in ("macro", "equity", "fixed_income", "portfolio_construction", "compliance"):
            assert key in exp, f"experts missing key: {key}"

    def test_experts_macro_tilt(self):
        assert self.dash["experts"]["macro"]["tilt"] == "neutral"

    def test_experts_equity_candidate_count(self):
        assert self.dash["experts"]["equity"]["candidate_count"] == 2

    def test_experts_compliance_decision(self):
        assert self.dash["experts"]["compliance"]["decision"] == "PASS"

    # ---- 3. crosscheck panel ----
    def test_crosscheck_has_required_keys(self):
        cc = self.dash["crosscheck"]
        for key in ("macro_tilt", "agreement", "escalation_signals", "jury_event_count"):
            assert key in cc, f"crosscheck missing key: {key}"

    # ---- 4. compliance panel ----
    def test_compliance_has_required_keys(self):
        comp = self.dash["compliance"]
        for key in ("cr_matrix", "r_levels_used", "disclosure_checklist",
                    "investor_c_level", "portfolio_r_level", "compliance_decision"):
            assert key in comp, f"compliance missing key: {key}"

    def test_compliance_cr_matrix_shape(self):
        comp = self.dash["compliance"]
        matrix = comp["cr_matrix"]
        # Five C-levels → five rows
        assert len(matrix) == 5
        assert all("c_level" in row for row in matrix)

    def test_compliance_investor_c_level(self):
        assert self.dash["compliance"]["investor_c_level"] == "C3"

    def test_compliance_portfolio_r_level(self):
        assert self.dash["compliance"]["portfolio_r_level"] == "R3"

    def test_compliance_disclosure_checklist_present(self):
        checklist = self.dash["compliance"]["disclosure_checklist"]
        assert isinstance(checklist, list)
        assert len(checklist) >= 4

    def test_compliance_fx_disclosure_present_for_cross_border(self):
        """For a cross-border portfolio with 汇率 in disclosures, FX row must be present."""
        checklist = self.dash["compliance"]["disclosure_checklist"]
        fx_items = [item for item in checklist if item["field"] == "fx_cross_border_disclosure"]
        assert len(fx_items) == 1
        assert fx_items[0]["present"] is True

    def test_compliance_suitability_match_present(self):
        checklist = self.dash["compliance"]["disclosure_checklist"]
        suit_items = [item for item in checklist if item["field"] == "suitability_match"]
        assert len(suit_items) == 1
        assert suit_items[0]["present"] is True

    def test_compliance_risk_disclosure_present(self):
        checklist = self.dash["compliance"]["disclosure_checklist"]
        risk_items = [item for item in checklist if item["field"] == "risk_disclosure"]
        assert len(risk_items) == 1
        assert risk_items[0]["present"] is True

    def test_cr_matrix_suitable_cell_for_matching_level(self):
        """C3 investor, R3 portfolio → the C3/R3 cell should be 'suitable'."""
        matrix = self.dash["compliance"]["cr_matrix"]
        r_levels = self.dash["compliance"]["r_levels_used"]
        c3_row = next((r for r in matrix if r["c_level"] == "C3"), None)
        assert c3_row is not None
        if "R3" in r_levels:
            assert c3_row["R3"] == "suitable"

    def test_cr_matrix_over_level_cell(self):
        """C1 investor against R3 product → 'over-level'."""
        matrix = self.dash["compliance"]["cr_matrix"]
        r_levels = self.dash["compliance"]["r_levels_used"]
        c1_row = next((r for r in matrix if r["c_level"] == "C1"), None)
        assert c1_row is not None
        if "R3" in r_levels:
            assert c1_row["R3"] == "over-level"

    # ---- 5. cost panel ----
    def test_cost_has_required_keys(self):
        cost = self.dash["cost"]
        for key in ("tokens_used", "cost_usd", "node_count", "trace_event_count"):
            assert key in cost, f"cost missing key: {key}"

    def test_cost_computes_cost_usd_from_tokens(self):
        """cost_usd = tokens_used / 1000 * token_price_per_1k."""
        tokens = self.state.tokens_used
        expected = round(tokens / 1000 * self.settings.token_price_per_1k, 6)
        assert self.dash["cost"]["cost_usd"] == pytest.approx(expected)

    def test_cost_tokens_used(self):
        assert self.dash["cost"]["tokens_used"] == self.state.tokens_used

    def test_cost_trace_event_count(self):
        assert self.dash["cost"]["trace_event_count"] == len(self.state.trace_events)

    def test_cost_no_offline_note_when_tokens_nonzero(self):
        """When tokens_used > 0, no offline note should be present."""
        cost = self.dash["cost"]
        # Fixture has tokens_used=1200 — should NOT carry the offline note
        assert cost["tokens_used"] == 1200
        assert "note" not in cost, (
            f"Offline note should not appear when tokens_used > 0, got: {cost.get('note')}"
        )


# ---------------------------------------------------------------------------
# sse_events — offline streaming
# ---------------------------------------------------------------------------

class TestSseEvents:
    def _collect(self, profile: InvestorProfile | None = None):
        p = profile or _c3_profile()
        deps = build_sample_deps()
        settings = get_settings()
        events = list(sse_events(p, deps, settings))
        return events

    def test_yields_start_event(self):
        events = self._collect()
        starts = [e for e in events if e.startswith("event: start")]
        assert len(starts) == 1

    def test_yields_node_events(self):
        events = self._collect()
        nodes = [e for e in events if e.startswith("event: node")]
        assert len(nodes) >= 1

    def test_yields_complete_event(self):
        events = self._collect()
        completes = [e for e in events if e.startswith("event: complete")]
        assert len(completes) == 1

    def test_event_order_start_then_nodes_then_complete(self):
        events = self._collect()
        kinds = []
        for e in events:
            if e.startswith("event: start"):
                kinds.append("start")
            elif e.startswith("event: node"):
                kinds.append("node")
            elif e.startswith("event: complete"):
                kinds.append("complete")
        assert kinds[0] == "start"
        assert kinds[-1] == "complete"
        assert "node" in kinds

    def test_complete_event_has_five_panels(self):
        events = self._collect()
        complete_event = next(e for e in events if e.startswith("event: complete"))
        # Extract JSON from data line
        data_line = next(l for l in complete_event.splitlines() if l.startswith("data:"))
        payload = json.loads(data_line[len("data: "):])
        for panel in ("allocation", "experts", "crosscheck", "compliance", "cost"):
            assert panel in payload, f"complete event missing panel: {panel}"

    def test_sse_framing(self):
        """Each event must follow the SSE wire format: event: X\\ndata: JSON\\n\\n"""
        events = self._collect()
        for e in events:
            lines = e.rstrip("\n").split("\n")
            assert lines[0].startswith("event: ")
            assert lines[1].startswith("data: ")
            assert e.endswith("\n\n")

    def test_node_event_has_node_field(self):
        events = self._collect()
        node_events = [e for e in events if e.startswith("event: node")]
        for e in node_events:
            data_line = next(l for l in e.splitlines() if l.startswith("data:"))
            payload = json.loads(data_line[len("data: "):])
            assert "node" in payload

    def test_start_event_has_profile_field(self):
        events = self._collect()
        start_event = next(e for e in events if e.startswith("event: start"))
        data_line = next(l for l in start_event.splitlines() if l.startswith("data:"))
        payload = json.loads(data_line[len("data: "):])
        assert "profile" in payload

    def test_none_profile_still_yields_events(self):
        """A None profile triggers guardrail block — pipeline still yields start + complete."""
        deps = build_sample_deps()
        settings = get_settings()
        events = list(sse_events(None, deps, settings))
        starts = [e for e in events if e.startswith("event: start")]
        completes = [e for e in events if e.startswith("event: complete")]
        assert len(starts) == 1
        assert len(completes) == 1


# ---------------------------------------------------------------------------
# build_dashboard with real pipeline run
# ---------------------------------------------------------------------------

class TestBuildDashboardLive:
    """End-to-end: run the pipeline offline and verify dashboard structure."""

    def setup_method(self):
        from wealthwise.runner import run_advisory
        profile = _c3_profile()
        deps = build_sample_deps()
        self.state = run_advisory(profile, deps)
        self.settings = get_settings()
        self.dash = build_dashboard(self.state, self.settings)

    def test_five_panels(self):
        assert {"allocation", "experts", "crosscheck", "compliance", "cost"} <= set(self.dash)

    def test_status_present(self):
        assert "status" in self.dash

    def test_allocation_r_level_nonempty(self):
        assert self.dash["allocation"]["portfolio_r_level"]

    def test_compliance_cr_matrix_five_rows(self):
        assert len(self.dash["compliance"]["cr_matrix"]) == 5

    def test_compliance_checklist_four_items(self):
        assert len(self.dash["compliance"]["disclosure_checklist"]) == 4

    def test_cost_tokens_consistent(self):
        assert self.dash["cost"]["tokens_used"] == self.state.tokens_used

    def test_cost_offline_note_present_when_tokens_zero(self):
        """I3: when tokens_used == 0 (offline mode), cost panel must carry offline note."""
        # Offline pipeline produces tokens_used=0 (no real LLM calls)
        assert self.state.tokens_used == 0, (
            f"Offline pipeline should have tokens_used=0, got {self.state.tokens_used}"
        )
        cost = self.dash["cost"]
        assert "note" in cost, "Cost panel must include 'note' key when tokens_used == 0"
        assert "离线" in cost["note"] or "offline" in cost["note"].lower(), (
            f"Offline note should mention offline mode: {cost['note']!r}"
        )


class TestSseEventsFinalStateAccuracy:
    """M2: verify SSE complete event dashboard matches direct run_advisory."""

    def test_complete_event_trace_count_matches_direct_run(self):
        """The SSE complete-event dashboard must reflect the fully accumulated state.

        Specifically: trace_event_count in the SSE complete dashboard should match
        what a direct run_advisory call produces (both invoke the full pipeline).
        """
        from wealthwise.runner import run_advisory

        profile = _c3_profile()
        deps = build_sample_deps()
        settings = get_settings()

        # Direct run for reference
        direct_state = run_advisory(profile, deps)
        direct_dash = build_dashboard(direct_state, settings)

        # SSE streaming run
        events = list(sse_events(profile, deps, settings))
        complete_event = next(e for e in events if e.startswith("event: complete"))
        data_line = next(l for l in complete_event.splitlines() if l.startswith("data:"))
        sse_dash = json.loads(data_line[len("data: "):])

        # The SSE complete dashboard must have the same trace_event_count
        # as the direct run (both are full pipeline invocations, same deps).
        sse_count = sse_dash["cost"]["trace_event_count"]
        direct_count = direct_dash["cost"]["trace_event_count"]
        assert sse_count == direct_count, (
            f"SSE complete trace_event_count={sse_count} != "
            f"direct run trace_event_count={direct_count}; "
            "SSE final state may be using accumulated patches instead of full invoke"
        )

    def test_complete_event_status_matches_direct_run(self):
        """SSE complete status must match the direct pipeline status."""
        from wealthwise.runner import run_advisory

        profile = _c3_profile()
        deps = build_sample_deps()
        settings = get_settings()

        direct_state = run_advisory(profile, deps)
        events = list(sse_events(profile, deps, settings))
        complete_event = next(e for e in events if e.startswith("event: complete"))
        data_line = next(l for l in complete_event.splitlines() if l.startswith("data:"))
        sse_dash = json.loads(data_line[len("data: "):])

        assert sse_dash["status"] == direct_state.status


# ---------------------------------------------------------------------------
# sse_events runs the pipeline exactly once
# ---------------------------------------------------------------------------

class TestStreamRunsPipelineOnce:
    """The streamed dashboard must describe the run the client just watched.

    The generator used to stream node updates and then re-invoke the whole graph
    to build the final dashboard. With real providers that second invoke is an
    independent advisory: double the latency and jury spend, and a dashboard that
    does not correspond to the streamed run or to its trace.
    """

    def _profile(self):
        from wealthwise.agents.state import InvestorProfile

        return InvestorProfile(
            risk_level="C3", investable=500_000.0, horizon_years=5,
            goals=["balanced_growth"], liquidity_min=0.2, accept_cross_border=True,
        )

    def test_graph_is_invoked_once(self, monkeypatch):
        from wealthwise.bootstrap import build_sample_deps
        from wealthwise.config import get_settings
        import wealthwise.workbench as wb

        deps = build_sample_deps()
        real_build = wb.build_graph
        runs = {"stream": 0, "invoke": 0}

        def counting_build_graph(d):
            graph = real_build(d)
            real_stream, real_invoke = graph.stream, graph.invoke

            def stream(*a, **kw):
                runs["stream"] += 1
                return real_stream(*a, **kw)

            def invoke(*a, **kw):
                runs["invoke"] += 1
                return real_invoke(*a, **kw)

            graph.stream, graph.invoke = stream, invoke
            return graph

        monkeypatch.setattr(wb, "build_graph", counting_build_graph)
        list(sse_events(self._profile(), deps, get_settings()))

        assert runs["stream"] == 1
        assert runs["invoke"] == 0, "the pipeline must not be re-run to build the dashboard"

    def test_streamed_dashboard_matches_a_direct_run(self):
        from wealthwise.bootstrap import build_sample_deps
        from wealthwise.config import get_settings
        from wealthwise.runner import run_advisory

        deps, settings = build_sample_deps(), get_settings()
        profile = self._profile()

        events = list(sse_events(profile, deps, settings))
        complete = [e for e in events if e.startswith("event: complete")]
        assert len(complete) == 1
        streamed = json.loads(complete[0].split("data: ", 1)[1])

        reference = build_dashboard(run_advisory(profile, deps), settings)
        # trace_event_count and the token totals are exactly the accumulated-list
        # fields the discarded second invoke existed to get right.
        assert streamed["cost"]["trace_event_count"] == reference["cost"]["trace_event_count"]
        assert streamed["cost"]["tokens_used"] == reference["cost"]["tokens_used"]
        assert streamed["cost"]["node_count"] == reference["cost"]["node_count"]
        assert streamed["status"] == reference["status"]
        assert streamed["allocation"]["weights"] == reference["allocation"]["weights"]


class TestNodeStartEvents:
    """Nodes emit on completion, so a slow node makes the stream silent.

    node_start is the only signal the UI has that a long jury call is in flight
    rather than the connection being dead.
    """

    def _profile(self):
        from wealthwise.agents.state import InvestorProfile

        return InvestorProfile(
            risk_level="C3", investable=500_000.0, horizon_years=5,
            goals=["balanced_growth"], liquidity_min=0.2, accept_cross_border=True,
        )

    def _events(self):
        from wealthwise.bootstrap import build_sample_deps
        from wealthwise.config import get_settings

        return list(sse_events(self._profile(), build_sample_deps(), get_settings()))

    def test_every_node_announces_itself_before_it_completes(self):
        events = self._events()
        order = []
        for raw in events:
            kind = raw.split("\n", 1)[0].removeprefix("event: ")
            if kind in ("node_start", "node"):
                order.append((kind, json.loads(raw.split("data: ", 1)[1])["node"]))

        started = [n for k, n in order if k == "node_start"]
        completed = [n for k, n in order if k == "node"]
        assert started, "no node_start events were emitted"
        assert started == completed, "each node should start and complete exactly once, in order"

        # And the start must precede the completion for the *same* occurrence.
        seen_starts: list[str] = []
        for kind, node in order:
            if kind == "node_start":
                seen_starts.append(node)
            else:
                assert node in seen_starts, f"{node} completed without announcing a start"
                seen_starts.remove(node)

    def test_node_start_carries_only_the_name(self):
        """It fires before execution, so there is no state to report yet."""
        for raw in self._events():
            if raw.startswith("event: node_start"):
                assert json.loads(raw.split("data: ", 1)[1]).keys() == {"node"}


class TestExecutionPanel:
    """The five original panels describe the decision; none says what to buy."""

    def _dash(self):
        from wealthwise.agents.state import InvestorProfile
        from wealthwise.bootstrap import build_sample_deps
        from wealthwise.runner import run_advisory

        profile = InvestorProfile(
            risk_level="C4", investable=800_000.0, horizon_years=7,
            goals=["balanced_growth"], liquidity_min=0.15, accept_cross_border=True,
        )
        state = run_advisory(profile, build_sample_deps())
        return build_dashboard(state, get_settings())

    def test_dashboard_carries_the_order_list(self):
        ex = self._dash()["execution"]
        assert ex["position_count"] > 0, "offline demo produced no executable plan"
        for key in ("positions", "invested", "cash_residual", "guidance"):
            assert key in ex

    def test_every_position_is_placeable(self):
        for p in self._dash()["execution"]["positions"]:
            assert p["shares"] > 0 and p["shares"] % p["lot_size"] == 0
            assert p["amount"] > 0
            assert p["name"] and p["symbol"] and p["market"]

    def test_guidance_reaches_the_ui(self):
        g = self._dash()["execution"]["guidance"]
        assert g["entry"]["detail"] and g["rebalance"]["detail"]
        assert g["channels"], "no channel guidance surfaced"

    def test_offline_sample_data_can_price_every_instrument(self):
        """The demo path is the repo's headline; it must not degrade to weights."""
        from wealthwise.bootstrap import build_sample_deps

        market = build_sample_deps().market
        for asset_class in ("equity", "bond", "cash"):
            for c in market.screen("A", {"asset_class": asset_class}):
                assert c.metrics.get("price"), f"{c.symbol} has no price"
                assert c.metrics.get("lot_size"), f"{c.symbol} has no lot size"
