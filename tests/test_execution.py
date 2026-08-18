"""Tests for the execution layer — weights to a placeable order list."""
from __future__ import annotations

from wealthwise.agents.state import AssetCandidate, InvestorProfile, PortfolioAllocation
from wealthwise.portfolio.execution import build_execution_plan, realised_allocation
from wealthwise.portfolio.guidance import build_guidance


def _candidate(symbol, price, lot=100, market="A", asset_class="equity", name=None):
    return AssetCandidate(
        symbol=symbol, market=market, asset_class=asset_class,
        name=name or symbol, currency={"A": "CNY", "HK": "HKD", "US": "USD"}[market],
        r_level="R3", metrics={"price": price, "lot_size": lot},
    )


def _portfolio(weights, class_weights=None):
    return PortfolioAllocation(
        weights=weights,
        class_weights=class_weights or {"equity": sum(weights.values())},
        portfolio_r_level="R3", fx_exposure=0.0, metrics={},
    )


class TestLotRounding:
    def test_每个持仓都买得起至少一手(self):
        """The defect this module exists for: 5,600 元 allocated to a 129,799 元 lot."""
        cands = [_candidate("EXPENSIVE", 1298.0), _candidate("CHEAP", 11.0)]
        plan = build_execution_plan(_portfolio({"EXPENSIVE": 0.5, "CHEAP": 0.5}),
                                    cands, investable=100_000)

        assert plan.positions, "plan should not be empty"
        for p in plan.positions:
            assert p.amount >= p.price * p.lot_size, f"{p.symbol} cannot buy one lot"
            assert p.shares % p.lot_size == 0, f"{p.symbol} is not a whole number of lots"

    def test_买不起的标的被剔除并记录原因(self):
        cands = [_candidate("EXPENSIVE", 1298.0), _candidate("CHEAP", 11.0)]
        plan = build_execution_plan(_portfolio({"EXPENSIVE": 0.5, "CHEAP": 0.5}),
                                    cands, investable=50_000)
        assert "EXPENSIVE" not in [p.symbol for p in plan.positions]
        assert any("EXPENSIVE" in d for d in plan.dropped)

    def test_美股按股计不按手计(self):
        plan = build_execution_plan(
            _portfolio({"AAPL": 1.0}),
            [_candidate("AAPL", 306.0, lot=1, market="US")], investable=100_000)
        assert plan.positions[0].lot_size == 1
        assert plan.positions[0].shares == plan.positions[0].lots

    def test_港股按实际每手股数计算(self):
        """HK board lots vary; assuming 100 everywhere misprices the book."""
        plan = build_execution_plan(
            _portfolio({"00005": 1.0}),
            [_candidate("00005", 100.0, lot=400, market="HK")], investable=100_000)
        assert plan.positions[0].shares % 400 == 0


class TestSleevesSurvive:
    def test_低权重同类不会被整片丢掉(self):
        """Dropping every sub-minimum name at once left 35% of the book in idle cash."""
        cands = [_candidate(f"E{i}", 10.0) for i in range(50)]
        weights = {f"E{i}": 0.35 / 50 for i in range(50)}   # 每只 0.7%，全在 2% 门槛下
        plan = build_execution_plan(_portfolio(weights, {"equity": 0.35}),
                                    cands, investable=800_000)

        equity = [p for p in plan.positions if p.asset_class == "equity"]
        assert equity, "the whole equity sleeve was dropped"
        assert 0.30 <= sum(p.weight for p in equity) <= 0.40, (
            "the equity sleeve should survive at roughly its target weight")

    def test_丢弃顺序按难塞程度而非权重(self):
        """Near-identical risk-parity weights made 'smallest first' arbitrary."""
        cands = [_candidate("HUGE_LOT", 1298.0), _candidate("A1", 6.0),
                 _candidate("A2", 6.5), _candidate("A3", 7.0)]
        weights = {"HUGE_LOT": 0.09, "A1": 0.09, "A2": 0.09, "A3": 0.08}
        plan = build_execution_plan(_portfolio(weights, {"equity": 0.35}),
                                    cands, investable=300_000)

        kept = {p.symbol for p in plan.positions}
        assert "HUGE_LOT" not in kept, "the hardest name to fit should go first"
        assert {"A1", "A2", "A3"} <= kept, "cheap-lot names should survive"

    def test_同类内部再分配不改变大类比例(self):
        cands = ([_candidate("EXPENSIVE", 2000.0)]
                 + [_candidate(f"E{i}", 10.0) for i in range(5)]
                 + [_candidate(f"B{i}", 100.0, asset_class="bond") for i in range(3)])
        weights = {"EXPENSIVE": 0.10}
        weights.update({f"E{i}": 0.05 for i in range(5)})
        weights.update({f"B{i}": 0.65 / 3 for i in range(3)})
        plan = build_execution_plan(
            _portfolio(weights, {"equity": 0.35, "bond": 0.65}),
            cands, investable=1_000_000)

        cw = plan.class_weights()
        assert abs(cw.get("bond", 0) - 0.65) < 0.08, (
            f"bond sleeve drifted: {cw}")


class TestResidualSweep:
    def test_取整余款进货币基金而非闲置(self):
        cands = ([_candidate("E1", 10.0)]
                 + [_candidate("MMF", 100.0, asset_class="cash", name="货币ETF")])
        plan = build_execution_plan(
            _portfolio({"E1": 0.5, "MMF": 0.5}, {"equity": 0.5, "cash": 0.5}),
            cands, investable=100_000)
        assert plan.cash_residual < 100_000 * 0.05, (
            f"too much left idle: {plan.cash_residual}")

    def test_没有货币标的时余款如实报告(self):
        plan = build_execution_plan(_portfolio({"E1": 1.0}),
                                    [_candidate("E1", 999.0)], investable=100_000)
        assert plan.cash_residual > 0
        assert abs(plan.invested + plan.cash_residual - 100_000) < 1e-6


class TestRealisedAllocation:
    def test_合规看到的是取整后的真实权重(self):
        """Approving pre-rounding weights the investor never holds is an audit gap."""
        cands = [_candidate("EXPENSIVE", 1298.0), _candidate("CHEAP", 11.0)]
        target = _portfolio({"EXPENSIVE": 0.5, "CHEAP": 0.5})
        plan = build_execution_plan(target, cands, investable=50_000)
        realised = realised_allocation(target, plan)

        assert set(realised.weights) == {p.symbol for p in plan.positions}
        assert "EXPENSIVE" not in realised.weights

    def test_外币敞口按真实持仓重算(self):
        cands = [_candidate("A1", 10.0), _candidate("AAPL", 300.0, lot=1, market="US")]
        target = _portfolio({"A1": 0.5, "AAPL": 0.5})
        plan = build_execution_plan(target, cands, investable=200_000)
        realised = realised_allocation(target, plan)
        assert 0.4 < realised.fx_exposure < 0.6


class TestGuidance:
    def _profile(self, horizon=7, investable=800_000.0):
        return InvestorProfile(risk_level="C4", investable=investable,
                               horizon_years=horizon, goals=["balanced_growth"],
                               liquidity_min=0.15, accept_cross_border=True)

    def test_权益重则分批建仓(self):
        g = build_guidance(self._profile(),
                           _portfolio({}, {"equity": 0.6}),
                           build_execution_plan(_portfolio({}), [], 800_000))
        assert g["entry"]["stages"] > 1

    def test_权益轻则一次性建仓(self):
        g = build_guidance(self._profile(),
                           _portfolio({}, {"equity": 0.05}),
                           build_execution_plan(_portfolio({}), [], 800_000))
        assert g["entry"]["stages"] == 1

    def test_再平衡周期随期限放宽(self):
        plan = build_execution_plan(_portfolio({}), [], 800_000)
        short = build_guidance(self._profile(horizon=2), _portfolio({}), plan)
        long = build_guidance(self._profile(horizon=20), _portfolio({}), plan)
        assert short["rebalance"]["cadence"] != long["rebalance"]["cadence"]

    def test_港股通门槛按本金判定(self):
        cands = [_candidate("00700", 400.0, lot=100, market="HK")]
        target = _portfolio({"00700": 1.0})
        rich = build_guidance(self._profile(investable=800_000), target,
                              build_execution_plan(target, cands, 800_000))
        poor = build_guidance(self._profile(investable=200_000), target,
                              build_execution_plan(target, cands, 200_000))
        assert any("已满足港股通" in c for c in rich["channels"])
        assert any("未达港股通" in c for c in poor["channels"])
