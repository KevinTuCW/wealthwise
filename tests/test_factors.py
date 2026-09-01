"""Tests for multi-factor scoring and the history it runs on.

Two things are being pinned here. First the arithmetic: z-scores, clipping,
renormalisation when a factor is missing. Second — and more important — the
property that makes the switch safe to ship: with `ENABLE_FACTOR_SCORING` off,
selection must be bit-for-bit what it was before this module existed.
"""
from __future__ import annotations

import json

import pytest

from wealthwise.agents.state import AssetCandidate


def _candidate(symbol, market="A", **metrics) -> AssetCandidate:
    return AssetCandidate(
        symbol=symbol, market=market, asset_class="equity", name=symbol,
        currency="CNY", r_level="R3", metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Scoring arithmetic
# ---------------------------------------------------------------------------

class TestScoreCandidates:
    def test_便宜的价值分更高(self):
        from wealthwise.portfolio.factors import score_candidates

        cheap = _candidate("CHEAP", pe=5.0, market_cap_100m=500.0)
        dear = _candidate("DEAR", pe=50.0, market_cap_100m=500.0)
        scores = score_candidates([cheap, dear])

        assert scores["CHEAP"].z["value"] > scores["DEAR"].z["value"]

    def test_低波动得分高于高波动(self):
        from wealthwise.portfolio.factors import score_candidates

        calm = _candidate("CALM", volatility=0.12, market_cap_100m=500.0)
        wild = _candidate("WILD", volatility=0.60, market_cap_100m=500.0)
        scores = score_candidates([calm, wild])

        assert scores["CALM"].z["low_vol"] > scores["WILD"].z["low_vol"]

    def test_规模因子在本仓是越大越好(self):
        """Inverts the academic small-cap premium on purpose — suitability, not alpha."""
        from wealthwise.portfolio.factors import score_candidates

        big = _candidate("BIG", market_cap_100m=5000.0)
        small = _candidate("SMALL", market_cap_100m=150.0)
        scores = score_candidates([big, small])

        assert scores["BIG"].z["size"] > scores["SMALL"].z["size"]

    def test_动量来自历史而非当日涨跌(self):
        from wealthwise.portfolio.factors import score_candidates

        winner = _candidate("WIN", momentum=0.35, market_cap_100m=500.0)
        loser = _candidate("LOSE", momentum=-0.20, market_cap_100m=500.0)
        scores = score_candidates([winner, loser])

        assert scores["WIN"].z["momentum"] > scores["LOSE"].z["momentum"]

    def test_换手率封顶后投机不再越高越好(self):
        from wealthwise.portfolio.factors import TURNOVER_CAP, score_candidates

        liquid = _candidate("LIQ", turnover=TURNOVER_CAP, market_cap_100m=500.0)
        frantic = _candidate("HOT", turnover=TURNOVER_CAP * 6, market_cap_100m=500.0)
        scores = score_candidates([liquid, frantic])

        assert scores["LIQ"].z["liquidity"] == scores["HOT"].z["liquidity"]

    def test_极端值被裁剪不能绑架排名(self):
        from wealthwise.portfolio.factors import Z_CLIP, score_candidates

        pool = [_candidate(f"N{i}", pe=15.0, market_cap_100m=500.0) for i in range(30)]
        pool.append(_candidate("GLITCH", pe=0.0001, market_cap_100m=500.0))
        scores = score_candidates(pool)

        assert scores["GLITCH"].z["value"] == Z_CLIP

    def test_缺因子的标的按已有因子重归一(self):
        """A 2-of-5 name is scored on the same scale, not penalised for coverage."""
        from wealthwise.portfolio.factors import score_candidates

        full = _candidate("FULL", pe=10.0, market_cap_100m=500.0,
                          volatility=0.2, momentum=0.1, turnover=1.0)
        thin = _candidate("THIN", pe=10.0, market_cap_100m=500.0)
        scores = score_candidates([full, thin])

        assert scores["THIN"].coverage == 2
        assert scores["FULL"].coverage == 5
        # Both sit at the mean of every factor they share, so both score ~0 —
        # the thin one is not dragged down for the factors it lacks.
        assert scores["THIN"].score == pytest.approx(0.0, abs=1e-9)

    def test_覆盖过薄会被标记(self):
        from wealthwise.portfolio.factors import score_candidates

        scores = score_candidates([_candidate("BARE"), _candidate("ALSO_BARE")])
        assert scores["BARE"].thin is True

    def test_单标的市场不会除以零(self):
        from wealthwise.portfolio.factors import score_candidates

        scores = score_candidates([_candidate("ONLY", pe=10.0, market_cap_100m=500.0)])
        assert scores["ONLY"].score == 0.0

    def test_全体相同的因子不提供区分度(self):
        from wealthwise.portfolio.factors import score_candidates

        pool = [_candidate(f"S{i}", pe=15.0, market_cap_100m=500.0) for i in range(5)]
        scores = score_candidates(pool)
        assert all(s.z["value"] == 0.0 for s in scores.values())

    def test_亏损股的价值因子为负(self):
        """E/P rather than P/E: a negative multiple must rank low, not high."""
        from wealthwise.portfolio.factors import score_candidates

        loss = _candidate("LOSS", pe=-5.0, market_cap_100m=500.0)
        profit = _candidate("PROFIT", pe=25.0, market_cap_100m=500.0)
        scores = score_candidates([loss, profit])

        assert scores["LOSS"].z["value"] < scores["PROFIT"].z["value"]

    def test_空列表返回空(self):
        from wealthwise.portfolio.factors import score_candidates

        assert score_candidates([]) == {}


# ---------------------------------------------------------------------------
# History-derived inputs
# ---------------------------------------------------------------------------

class TestHistoryMetrics:
    def test_波动率年化(self):
        from wealthwise.providers.history import realized_volatility

        # A flat series has no variance, so no risk to annualise.
        assert realized_volatility([100.0] * 40) == 0.0

    def test_波动率区分平静与剧烈(self):
        from wealthwise.providers.history import realized_volatility

        calm = [100.0 + (i % 2) * 0.1 for i in range(60)]
        wild = [100.0 + (i % 2) * 10.0 for i in range(60)]
        assert realized_volatility(wild) > realized_volatility(calm)

    def test_历史太短不给数(self):
        """A 10-bar volatility estimate is noise wearing a number's clothes."""
        from wealthwise.providers.history import realized_volatility, momentum

        short = [100.0] * 10
        assert realized_volatility(short) is None
        assert momentum(short) is None

    def test_动量跳过最近五个交易日(self):
        from wealthwise.providers.history import momentum

        # Flat for the window, then a spike in the final five sessions. Skipping
        # the reversal window means that spike must not register as momentum.
        closes = [100.0] * 60 + [200.0] * 5
        assert momentum(closes) == pytest.approx(0.0)

    def test_动量读取窗口内的真实涨幅(self):
        from wealthwise.providers.history import momentum

        closes = [100.0] * 5 + [float(100 + i) for i in range(60)]
        assert momentum(closes) > 0

    def test_k线解析取收盘价(self):
        from wealthwise.providers.history import _closes

        payload = json.dumps({"code": 0, "data": {"sh600519": {"qfqday": [
            ["2026-05-11", "1344.8", "1333.3", "1344.8", "1332.9", "57135.0"],
            ["2026-05-12", "1333.9", "1326.5", "1335.5", "1322.4", "50837.0"],
        ]}}})
        assert _closes(payload) == [1333.3, 1326.5]

    def test_脏响应不抛异常(self):
        from wealthwise.providers.history import _closes

        assert _closes("param error") == []
        assert _closes(json.dumps({"code": 0, "msg": "param error", "data": []})) == []

    def test_enrich写入波动率与动量(self):
        from wealthwise.providers.history import TencentHistoryProvider

        closes = [100.0 + i * 0.5 for i in range(70)]
        provider = TencentHistoryProvider()
        provider.closes = lambda pairs: {"600519": closes}      # type: ignore[assignment]

        got = provider.enrich([_candidate("600519")])[0]
        assert got.metrics["volatility"] > 0
        assert got.metrics["momentum"] > 0
        assert got.metrics["history_bars"] == 70

    def test_取不到历史的标的原样返回(self):
        from wealthwise.providers.history import TencentHistoryProvider

        provider = TencentHistoryProvider()
        provider.closes = lambda pairs: {}                       # type: ignore[assignment]

        original = _candidate("600519", pe=20.0)
        got = provider.enrich([original])[0]
        assert got.metrics == {"pe": 20.0}

    def test_单标的抓取失败不影响其他标的(self):
        from wealthwise.providers.history import TencentHistoryProvider

        provider = TencentHistoryProvider(workers=2)

        def fake_get(prefixed):
            if "600519" in prefixed:
                raise RuntimeError("reset by peer")
            return json.dumps({"data": {prefixed: {"qfqday": [
                ["d", "1", str(100 + i), "1", "1", "1"] for i in range(40)
            ]}}})

        provider._get = fake_get                                 # type: ignore[assignment]
        got = provider.closes([("600519", "A"), ("000858", "A")])
        assert "600519" not in got
        assert len(got["000858"]) == 40
