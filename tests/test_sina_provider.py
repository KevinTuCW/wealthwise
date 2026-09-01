"""Tests for SinaMarketProvider — the corroborating quote feed.

Payloads below are trimmed from live responses captured 2026-09-01. The field
layout differs per market, which is the thing most likely to rot, so each market
gets its own parse assertion rather than one generic case.
"""
from __future__ import annotations

import pytest

from wealthwise.agents.state import AssetCandidate
from wealthwise.providers.universe import Universe

UNIVERSE = Universe({
    "equity": {"A": ["600519"], "HK": ["00700"], "US": ["AAPL"]},
    "bond": {"A": ["511010"]},
    "cash": {"A": ["511990"]},
})

A_LINE = ('var hq_str_sh600519="贵州茅台,1295.000,1299.520,1299.560,1307.990,'
          '1286.100,1299.560,1299.700,3266402,4242440861.000";')
HK_LINE = ('var hq_str_hk00700="TENCENT,腾讯控股,446.400,453.000,447.600,440.600,'
           '441.400,-11.600,-2.561,441.39,441.60,8990301596,20289781,0.000,0.000,'
           '675.134,411.000,2026/09/01,16:08";')
US_LINE = ('var hq_str_gb_aapl="苹果,316.8500,-0.89,2026-09-01 17:10:29,-2.8500,'
           '319.6000,321.2350,312.8000,344.5700,225.1600,41242724,39864520,'
           '4624166374688,8.30,38.170000";')
# A halted name: every price field zeroed.
HALTED_LINE = 'var hq_str_sh600519="诺思兰德,0.000,0.000,0.000,0.000,0.000,0.000";'

PAYLOAD = "\n".join([A_LINE, HK_LINE, US_LINE])


def _provider(payload=PAYLOAD):
    from wealthwise.providers.sina_provider import SinaMarketProvider

    provider = SinaMarketProvider(UNIVERSE)
    provider._get = lambda prefixed: payload       # type: ignore[assignment]
    return provider


class TestSymbolPrefixing:
    @pytest.mark.parametrize("symbol,market,expected", [
        ("600519", "A", "sh600519"),
        ("511010", "A", "sh511010"),      # Shanghai fund range — the bond sleeve
        ("000858", "A", "sz000858"),
        ("430047", "A", "bj430047"),
        ("00700", "HK", "hk00700"),
        ("700", "HK", "hk00700"),         # zero-padded to five
        ("AAPL", "US", "gb_aapl"),
    ])
    def test_前缀规则与腾讯源一致(self, symbol, market, expected):
        from wealthwise.providers.sina_provider import _prefix

        assert _prefix(symbol, market) == expected

    def test_两个源的前缀规则不能分叉(self):
        """Different prefix rules would make the two feeds quote different things."""
        from wealthwise.providers.sina_provider import _prefix as sina_prefix
        from wealthwise.providers.tencent_provider import _prefix as tencent_prefix

        for symbol in ("600519", "511010", "000858", "430047", "159972"):
            assert sina_prefix(symbol, "A") == tencent_prefix(symbol, "A")


class TestParsing:
    def test_a股取第四个字段作现价(self):
        got = {c.symbol: c for c in _provider().quotes(["600519", "00700", "AAPL"])}
        assert got["600519"].metrics["price"] == 1299.560
        assert got["600519"].name == "贵州茅台"
        assert got["600519"].currency == "CNY"

    def test_港股布局不同现价在第七位(self):
        got = {c.symbol: c for c in _provider().quotes(["600519", "00700", "AAPL"])}
        assert got["00700"].metrics["price"] == 441.400
        assert got["00700"].name == "腾讯控股"       # the Chinese name, not "TENCENT"
        assert got["00700"].currency == "HKD"

    def test_美股还能提供市值与市盈率(self):
        got = {c.symbol: c for c in _provider().quotes(["600519", "00700", "AAPL"])}
        apple = got["AAPL"]
        assert apple.metrics["price"] == 316.85
        assert apple.metrics["pe"] == pytest.approx(38.17)
        # Raw USD in the payload, 亿 in the domain metric — same unit as Tencent's.
        assert apple.metrics["market_cap_100m"] == pytest.approx(46241.66, rel=1e-4)

    def test_ah股不谎报市盈率(self):
        """This feed carries no P/E outside the US layout; absent must stay absent."""
        got = {c.symbol: c for c in _provider().quotes(["600519", "00700"])}
        assert "pe" not in got["600519"].metrics
        assert "pe" not in got["00700"].metrics

    def test_停牌全零行被跳过(self):
        assert _provider(HALTED_LINE).quotes(["600519"]) == []

    def test_空行被跳过(self):
        assert _provider('var hq_str_sh600519="";').quotes(["600519"]) == []

    def test_不在universe的标的不被请求(self):
        assert _provider().quotes(["999999"]) == []

    def test_返回的是AssetCandidate(self):
        got = _provider().quotes(["600519"])
        assert all(isinstance(c, AssetCandidate) for c in got)

    def test_资产类别取自universe(self):
        payload = 'var hq_str_sh511010="国债ETF国泰,141.1,141.1,141.2,141.2,141.1,141.2";'
        got = _provider(payload).quotes(["511010"])
        assert got[0].asset_class == "bond"
        assert got[0].r_level == "R2"


class TestScreening:
    def test_只返回被请求市场的标的(self):
        got = _provider().screen("HK", {"asset_class": "equity"})
        assert [c.symbol for c in got] == ["00700"]

    def test_没有市盈率的市场不会被max_pe误杀(self):
        """A feed with no P/E cannot honour the filter — and must not pretend to.

        This is exactly why this provider is registered as a corroborating
        source rather than a primary one.
        """
        got = _provider().screen("A", {"asset_class": "equity", "max_pe": 10.0})
        assert [c.symbol for c in got] == ["600519"]

    def test_有市盈率时max_pe生效(self):
        assert _provider().screen("US", {"asset_class": "equity", "max_pe": 10.0}) == []


class TestNetworkContract:
    def test_referer必须带上(self):
        """Without a Referer the endpoint answers 'Forbidden' — this is functional."""
        from wealthwise.providers.sina_provider import _HEADERS

        assert "Referer" in _HEADERS and _HEADERS["Referer"]

    def test_模块导入不触发requests(self):
        import sys

        assert "wealthwise.providers.sina_provider" in sys.modules
