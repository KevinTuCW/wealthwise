"""Tests for the consensus provider layer — pillar one, wired to actual sources.

`test_consensus.py` covers the resolver's arithmetic. These cover the wrappers
that put it in the pipeline: what a second feed changes, what it must not
change, and what happens when it disagrees or falls over.
"""
from __future__ import annotations

import pytest

from wealthwise.agents.state import AssetCandidate


def _candidate(symbol="600519", market="A", **metrics) -> AssetCandidate:
    return AssetCandidate(
        symbol=symbol, market=market, asset_class="equity", name=symbol,
        currency="CNY", r_level="R3", metrics=metrics,
    )


class _StubMarket:
    """A market provider that returns whatever it was handed."""

    def __init__(self, name, candidates, fail=False):
        self.name = name
        self._candidates = candidates
        self._fail = fail
        self.quote_calls = 0

    def screen(self, market, filters):
        return [c for c in self._candidates if c.market == market]

    def quotes(self, symbols):
        self.quote_calls += 1
        if self._fail:
            raise RuntimeError("feed down")
        wanted = {s.casefold() for s in symbols}
        return [c for c in self._candidates if c.symbol.casefold() in wanted]


class _StubMacro:
    def __init__(self, name, snapshot, fail=False):
        self.name = name
        self._snapshot = snapshot
        self._fail = fail

    def snapshot(self):
        if self._fail:
            raise RuntimeError("publisher down")
        return dict(self._snapshot)


# ---------------------------------------------------------------------------
# Market consensus
# ---------------------------------------------------------------------------

class TestConsensusMarketProvider:
    def test_单源时数值不变且置信封顶(self):
        """The offline stack runs through this layer; it must not move a number."""
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        source = _StubMarket("sample", [_candidate(price=100.0, pe=20.0)])
        provider = ConsensusMarketProvider([source])

        got = provider.screen("A", {})
        assert got[0].metrics["price"] == 100.0
        assert got[0].metrics["pe"] == 20.0
        assert got[0].metrics["consensus"]["price"]["confidence"] == 0.5
        assert got[0].metrics["consensus"]["price"]["disagreement"] is False
        assert got[0].metrics["data_confidence"] == 0.5

    def test_双源一致时取中位数且高置信(self):
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        primary = _StubMarket("tencent", [_candidate(price=100.0, pe=20.0)])
        second = _StubMarket("sina", [_candidate(price=100.0)])
        provider = ConsensusMarketProvider([primary, second])

        got = provider.screen("A", {})[0]
        assert got.metrics["price"] == 100.0
        assert got.metrics["consensus"]["price"]["confidence"] == 1.0
        assert got.metrics["consensus"]["price"]["sources"] == ["tencent", "sina"]
        assert "data-disagreement" not in got.tags

    def test_价格分歧被标记而不是被平均掉(self):
        """A 10% price gap is not an average worth taking — it is a data fault."""
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        primary = _StubMarket("tencent", [_candidate(price=100.0)])
        second = _StubMarket("sina", [_candidate(price=110.0)])
        provider = ConsensusMarketProvider([primary, second])

        got = provider.screen("A", {})[0]
        assert got.metrics["data_disagreement"] == ["price"]
        assert "data-disagreement" in got.tags
        assert got.metrics["consensus"]["price"]["readings"] == {
            "tencent": 100.0, "sina": 110.0,
        }

    def test_估值口径差异容忍度更宽(self):
        """P/E windows legitimately differ; a 5% gap must not cry wolf."""
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        primary = _StubMarket("tencent", [_candidate(price=100.0, pe=20.0)])
        second = _StubMarket("sina", [_candidate(price=100.0, pe=21.0)])

        got = ConsensusMarketProvider([primary, second]).screen("A", {})[0]
        assert got.metrics["consensus"]["pe"]["disagreement"] is False
        assert got.metrics["pe"] == 20.5

    def test_停牌的零报价不参与中位数(self):
        """A halted feed's 0.00 alongside a real price must not halve it."""
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        primary = _StubMarket("tencent", [_candidate(price=1299.0)])
        second = _StubMarket("sina", [_candidate(price=0.0)])

        got = ConsensusMarketProvider([primary, second]).screen("A", {})[0]
        assert got.metrics["price"] == 1299.0
        assert got.metrics["consensus"]["price"]["sources"] == ["tencent"]

    def test_负市盈率是信息不是脏数据(self):
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        primary = _StubMarket("tencent", [_candidate(price=10.0, pe=-3.0)])
        second = _StubMarket("sina", [_candidate(price=10.0, pe=-3.0)])

        got = ConsensusMarketProvider([primary, second]).screen("A", {})[0]
        assert got.metrics["pe"] == -3.0
        assert got.metrics["consensus"]["pe"]["confidence"] == 1.0

    def test_佐证源故障不拖垮筛选(self):
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        primary = _StubMarket("tencent", [_candidate(price=100.0)])
        broken = _StubMarket("sina", [], fail=True)

        got = ConsensusMarketProvider([primary, broken]).screen("A", {})
        assert len(got) == 1
        assert got[0].metrics["price"] == 100.0
        assert got[0].metrics["data_confidence"] == 0.5   # degraded, not failed

    def test_佐证源缺失标的时不错位署名(self):
        """A feed with nothing to say must not have another's reading filed under it."""
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        primary = _StubMarket("tencent", [_candidate("600519", price=100.0)])
        silent = _StubMarket("sina", [])
        third = _StubMarket("third", [_candidate("600519", price=100.0)])

        got = ConsensusMarketProvider([primary, silent, third]).screen("A", {})[0]
        assert got.metrics["consensus"]["price"]["sources"] == ["tencent", "third"]

    def test_按共识后的市盈率复筛只会更严(self):
        """A name that only clears max_pe on one feed's number does not clear it."""
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        primary = _StubMarket("tencent", [_candidate(price=100.0, pe=24.0)])
        second = _StubMarket("sina", [_candidate(price=100.0, pe=28.0)])

        provider = ConsensusMarketProvider([primary, second])
        # Reconciled P/E is the median, 26.0 — above the cap the primary passed.
        assert provider.screen("A", {"max_pe": 25.0}) == []

    def test_非数值字段来自主源(self):
        """Lot size has no median; an averaged one is an unplaceable order."""
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        primary = _StubMarket("tencent", [_candidate(price=100.0, lot_size=500)])
        second = _StubMarket("sina", [_candidate(price=100.0, lot_size=100)])

        got = ConsensusMarketProvider([primary, second]).screen("A", {})[0]
        assert got.metrics["lot_size"] == 500

    def test_空源列表被拒绝(self):
        from wealthwise.providers.consensus_provider import ConsensusMarketProvider

        with pytest.raises(ValueError):
            ConsensusMarketProvider([])


# ---------------------------------------------------------------------------
# Macro consensus
# ---------------------------------------------------------------------------

class TestConsensusMacroProvider:
    def test_两个发布方的cpi被真正调和(self):
        from wealthwise.providers.consensus_provider import ConsensusMacroProvider

        provider = ConsensusMacroProvider([
            _StubMacro("akshare-cpi-yearly", {"cpi": 0.021}),
            _StubMacro("akshare-cpi-nbs", {"cpi": 0.022}),
        ])
        snap = provider.snapshot()
        assert snap["cpi"] == pytest.approx(0.0215)
        assert snap["consensus"]["cpi"]["sources"] == [
            "akshare-cpi-yearly", "akshare-cpi-nbs",
        ]
        assert snap["consensus"]["cpi"]["confidence"] > 0.9

    def test_单发布方信号封顶在0点5(self):
        """One publisher is one publisher, however live the endpoint is."""
        from wealthwise.providers.consensus_provider import ConsensusMacroProvider

        snap = ConsensusMacroProvider([
            _StubMacro("akshare-lpr", {"interest_rate": 0.031}),
            _StubMacro("akshare-cpi-yearly", {"cpi": 0.021}),
        ]).snapshot()

        assert snap["interest_rate"] == 0.031
        assert snap["consensus"]["interest_rate"]["confidence"] == 0.5

    def test_发布方分歧被记名(self):
        from wealthwise.providers.consensus_provider import ConsensusMacroProvider

        snap = ConsensusMacroProvider([
            _StubMacro("a", {"cpi": 0.021}),
            _StubMacro("b", {"cpi": 0.034}),
        ]).snapshot()

        assert snap["disagreement"] == ["cpi"]
        assert snap["consensus"]["cpi"]["disagreement"] is True

    def test_定性字段取第一个发布它的源(self):
        from wealthwise.providers.consensus_provider import ConsensusMacroProvider

        snap = ConsensusMacroProvider([
            _StubMacro("a", {"cpi": 0.021}),
            _StubMacro("b", {"cpi": 0.021, "risk_sentiment": "cautious",
                             "as_of": "2026-08-01"}),
        ]).snapshot()

        assert snap["risk_sentiment"] == "cautious"
        assert snap["as_of"] == "2026-08-01"

    def test_定性源不投数值票(self):
        """A static JSON file must not corroborate a live reading."""
        from wealthwise.providers.consensus_provider import (
            ConsensusMacroProvider, QualitativeMacroSource,
        )

        live = _StubMacro("akshare-lpr", {"interest_rate": 0.031})
        sample = _StubMacro("sample", {"interest_rate": 0.034,
                                       "risk_sentiment": "cautious"})

        snap = ConsensusMacroProvider([
            live, QualitativeMacroSource(sample, name="sample-views"),
        ]).snapshot()

        assert snap["interest_rate"] == 0.031                     # not medianed
        assert snap["consensus"]["interest_rate"]["sources"] == ["akshare-lpr"]
        assert snap["consensus"]["interest_rate"]["confidence"] == 0.5
        assert snap["risk_sentiment"] == "cautious"               # views still land

    def test_布尔值不参与中位数(self):
        from wealthwise.providers.consensus_provider import ConsensusMacroProvider

        snap = ConsensusMacroProvider([_StubMacro("a", {"recession": True})]).snapshot()
        assert snap["recession"] is True
        assert "recession" not in snap["consensus"]

    def test_发布方故障不拖垮快照(self):
        from wealthwise.providers.consensus_provider import ConsensusMacroProvider

        snap = ConsensusMacroProvider([
            _StubMacro("a", {"cpi": 0.021}),
            _StubMacro("b", {}, fail=True),
        ]).snapshot()

        assert snap["cpi"] == 0.021
        assert snap["consensus"]["cpi"]["sources"] == ["a"]

    def test_每个源每次快照只取一次(self):
        """Twelve signals must cost one fetch per publisher, not twelve."""
        from wealthwise.providers.consensus_provider import ConsensusMacroProvider

        calls = {"n": 0}

        class Counting(_StubMacro):
            def snapshot(self):
                calls["n"] += 1
                return dict(self._snapshot)

        provider = ConsensusMacroProvider([
            Counting("a", {"cpi": 0.021, "interest_rate": 0.031, "ppi": -0.008}),
        ])
        provider.snapshot()
        assert calls["n"] == 1

        provider.snapshot()          # a second run refetches, it does not cache
        assert calls["n"] == 2
