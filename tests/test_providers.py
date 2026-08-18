"""Tests for the data-provider layer — Task 1.2.

All tests are offline: no network, no akshare import.
"""
from __future__ import annotations

import json
import pathlib
import pytest

SAMPLES = pathlib.Path(__file__).parent.parent / "data" / "samples"


# ---------------------------------------------------------------------------
# SampleMarketProvider
# ---------------------------------------------------------------------------

class TestSampleMarketProvider:
    def test_quotes_returns_asset_candidates(self):
        from wealthwise.providers.sample import SampleMarketProvider
        from wealthwise.agents.state import AssetCandidate

        p = SampleMarketProvider(str(SAMPLES))
        results = p.quotes(["600519", "AAPL"])
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, AssetCandidate)

    def test_quotes_case_insensitive(self):
        from wealthwise.providers.sample import SampleMarketProvider

        p = SampleMarketProvider(str(SAMPLES))
        upper = p.quotes(["AAPL"])
        lower = p.quotes(["aapl"])
        assert len(upper) == len(lower) == 1
        assert upper[0].symbol == lower[0].symbol

    def test_quotes_unknown_symbol_returns_empty(self):
        from wealthwise.providers.sample import SampleMarketProvider

        p = SampleMarketProvider(str(SAMPLES))
        results = p.quotes(["DOES_NOT_EXIST_XYZ"])
        assert results == []

    def test_screen_by_market_a(self):
        from wealthwise.providers.sample import SampleMarketProvider

        p = SampleMarketProvider(str(SAMPLES))
        results = p.screen("A", {})
        assert len(results) >= 4
        for r in results:
            assert r.market == "A"

    def test_screen_by_market_hk(self):
        from wealthwise.providers.sample import SampleMarketProvider

        p = SampleMarketProvider(str(SAMPLES))
        results = p.screen("HK", {})
        assert len(results) >= 4
        for r in results:
            assert r.market == "HK"

    def test_screen_by_market_us(self):
        from wealthwise.providers.sample import SampleMarketProvider

        p = SampleMarketProvider(str(SAMPLES))
        results = p.screen("US", {})
        assert len(results) >= 4
        for r in results:
            assert r.market == "US"

    def test_screen_filter_asset_class(self):
        from wealthwise.providers.sample import SampleMarketProvider

        p = SampleMarketProvider(str(SAMPLES))
        results = p.screen("A", {"asset_class": "equity"})
        for r in results:
            assert r.asset_class == "equity"
            assert r.market == "A"

    def test_screen_filter_max_pe(self):
        from wealthwise.providers.sample import SampleMarketProvider

        p = SampleMarketProvider(str(SAMPLES))
        results = p.screen("A", {"max_pe": 30.0})
        for r in results:
            pe = r.metrics.get("pe")
            if pe is not None:
                assert pe <= 30.0

    def test_screen_unknown_market_returns_empty(self):
        from wealthwise.providers.sample import SampleMarketProvider

        p = SampleMarketProvider(str(SAMPLES))
        results = p.screen("JP", {})
        assert results == []


# ---------------------------------------------------------------------------
# SampleMacroProvider
# ---------------------------------------------------------------------------

class TestSampleMacroProvider:
    def test_snapshot_returns_dict_with_required_keys(self):
        from wealthwise.providers.sample import SampleMacroProvider

        p = SampleMacroProvider(str(SAMPLES))
        snap = p.snapshot()
        assert isinstance(snap, dict)
        assert "interest_rate" in snap
        assert "cpi" in snap

    def test_snapshot_values_are_numeric(self):
        from wealthwise.providers.sample import SampleMacroProvider

        p = SampleMacroProvider(str(SAMPLES))
        snap = p.snapshot()
        assert isinstance(snap["interest_rate"], (int, float))
        assert isinstance(snap["cpi"], (int, float))


# ---------------------------------------------------------------------------
# SampleFXProvider
# ---------------------------------------------------------------------------

class TestSampleFXProvider:
    def test_rate_usdcnh(self):
        from wealthwise.providers.sample import SampleFXProvider

        p = SampleFXProvider(str(SAMPLES))
        rate = p.rate("USDCNH")
        assert isinstance(rate, float)
        assert rate > 0

    def test_rate_hkdcnh(self):
        from wealthwise.providers.sample import SampleFXProvider

        p = SampleFXProvider(str(SAMPLES))
        rate = p.rate("HKDCNH")
        assert isinstance(rate, float)
        assert rate > 0

    def test_rate_case_insensitive(self):
        from wealthwise.providers.sample import SampleFXProvider

        p = SampleFXProvider(str(SAMPLES))
        upper = p.rate("USDCNH")
        lower = p.rate("usdcnh")
        assert upper == lower

    def test_rate_unknown_pair_raises(self):
        from wealthwise.providers.sample import SampleFXProvider

        p = SampleFXProvider(str(SAMPLES))
        with pytest.raises(KeyError):
            p.rate("EURJPY")


# ---------------------------------------------------------------------------
# build_provider factory
# ---------------------------------------------------------------------------

class TestBuildProvider:
    def test_returns_sample_providers_when_use_real_providers_false(self):
        from wealthwise.providers.akshare_provider import build_provider
        from wealthwise.providers.sample import SampleMarketProvider, SampleMacroProvider, SampleFXProvider
        from wealthwise.config import Settings

        s = Settings(use_real_providers=False, sample_data_dir=str(SAMPLES))
        market, macro, fx = build_provider(s)
        assert isinstance(market, SampleMarketProvider)
        assert isinstance(macro, SampleMacroProvider)
        assert isinstance(fx, SampleFXProvider)

    def test_returns_akshare_provider_when_use_real_providers_true(self):
        from wealthwise.providers.akshare_provider import build_provider, AkShareMarketProvider
        from wealthwise.config import Settings

        s = Settings(use_real_providers=True, sample_data_dir=str(SAMPLES))
        market, macro, fx = build_provider(s)
        assert isinstance(market, AkShareMarketProvider)


# ---------------------------------------------------------------------------
# AkShareMarketProvider — _get seam stub (no real akshare import)
# ---------------------------------------------------------------------------

class TestAkShareMarketProviderStubbed:
    def test_quotes_maps_stubbed_payload_to_asset_candidates(self, monkeypatch):
        """Provider maps _get stub payload into AssetCandidate without importing akshare."""
        from wealthwise.providers.akshare_provider import AkShareMarketProvider
        from wealthwise.agents.state import AssetCandidate

        provider = AkShareMarketProvider()

        # Stub _get so no network / no akshare needed
        stub_payload = [
            {
                "symbol": "600519",
                "market": "A",
                "name": "贵州茅台",
                "currency": "CNY",
                "asset_class": "equity",
                "r_level": "R3",
                "metrics": {"pe": 28.5, "volatility": 0.22},
                "tags": ["白酒", "消费"],
            }
        ]
        monkeypatch.setattr(provider, "_get", lambda *args, **kwargs: stub_payload)

        results = provider.quotes(["600519"])
        assert len(results) == 1
        c = results[0]
        assert isinstance(c, AssetCandidate)
        assert c.symbol == "600519"
        assert c.market == "A"
        assert c.asset_class == "equity"
        assert c.r_level == "R3"
        assert c.metrics["pe"] == 28.5

    def test_screen_maps_stubbed_payload(self, monkeypatch):
        from wealthwise.providers.akshare_provider import AkShareMarketProvider
        from wealthwise.agents.state import AssetCandidate

        provider = AkShareMarketProvider()

        stub_payload = [
            {
                "symbol": "AAPL",
                "market": "US",
                "name": "Apple Inc.",
                "currency": "USD",
                "asset_class": "equity",
                "r_level": "R4",
                "metrics": {"pe": 30.0, "volatility": 0.25},
                "tags": ["科技"],
            }
        ]
        monkeypatch.setattr(provider, "_get", lambda *args, **kwargs: stub_payload)

        results = provider.screen("US", {})
        assert len(results) == 1
        assert isinstance(results[0], AssetCandidate)
        assert results[0].market == "US"

    def test_akshare_not_imported_at_module_level(self):
        """Importing akshare_provider must NOT trigger an akshare import."""
        import sys
        # If akshare were imported at module top, it'd be in sys.modules after import
        # We just assert the provider module loaded fine without it
        assert "wealthwise.providers.akshare_provider" in sys.modules
        assert "akshare" not in sys.modules


# ---------------------------------------------------------------------------
# Protocol conformance check
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_sample_providers_satisfy_protocols(self):
        from wealthwise.providers.base import MarketProvider, MacroProvider, FXProvider
        from wealthwise.providers.sample import SampleMarketProvider, SampleMacroProvider, SampleFXProvider

        assert isinstance(SampleMarketProvider(str(SAMPLES)), MarketProvider)
        assert isinstance(SampleMacroProvider(str(SAMPLES)), MacroProvider)
        assert isinstance(SampleFXProvider(str(SAMPLES)), FXProvider)


# ---------------------------------------------------------------------------
# TencentMarketProvider — quote-based A/HK/US provider
# ---------------------------------------------------------------------------

# Real response shapes, trimmed to the fields the mapper reads. Positions match
# the live payloads: name=1, code=2, price=3, pe=39, mcap=45, pb=46.
def _row(values: dict, width: int) -> str:
    fields = [""] * width
    for idx, val in values.items():
        fields[idx] = val
    return "~".join(fields)


A_ROW = _row({1: "贵州茅台", 2: "600519", 3: "1297.99", 39: "19.93", 45: "16225.93",
              46: "6.46"}, 88)
HK_ROW = _row({1: "腾讯控股", 2: "00700", 3: "443.200", 39: "16.21", 45: "40344.99",
               46: "TENCENT"}, 78)
US_ROW = _row({1: "苹果", 2: "AAPL.OQ", 3: "305.59", 39: "35.04", 45: "44598.35",
               46: "Apple Inc."}, 71)

PAYLOAD = (
    f'v_sh600519="{A_ROW}";\n'
    f'v_hk00700="{HK_ROW}";\n'
    f'v_usAAPL="{US_ROW}";\n'
)

UNIVERSE = {
    "equity": {"A": ["600519"], "HK": ["00700"], "US": ["AAPL"]},
    "bond": {"A": ["511010"]},
    "cash": {"A": ["511990"]},
}


def _provider(monkeypatch, payload=PAYLOAD):
    from wealthwise.providers.tencent_provider import TencentMarketProvider
    from wealthwise.providers.universe import Universe

    p = TencentMarketProvider(Universe(UNIVERSE))
    monkeypatch.setattr(p, "_get", lambda prefixed: payload)
    return p


class TestTencentMarketProvider:
    def test_parses_all_three_market_layouts(self, monkeypatch):
        from wealthwise.agents.state import AssetCandidate

        results = _provider(monkeypatch).quotes(["600519", "00700", "AAPL"])
        assert len(results) == 3
        by_symbol = {r.symbol: r for r in results}
        assert all(isinstance(r, AssetCandidate) for r in results)

        a = by_symbol["600519"]
        assert (a.market, a.currency, a.name) == ("A", "CNY", "贵州茅台")
        assert a.metrics["pe"] == 19.93
        assert a.metrics["pb"] == 6.46          # A-share layout only

        hk = by_symbol["00700"]
        assert (hk.market, hk.currency) == ("HK", "HKD")
        assert hk.metrics["pe"] == 16.21
        assert "pb" not in hk.metrics           # slot 46 is the English name here

        us = by_symbol["AAPL"]                  # venue suffix ".OQ" stripped
        assert (us.market, us.currency) == ("US", "USD")
        assert us.metrics["pe"] == 35.04
        assert "pb" not in us.metrics

    def test_screen_filters_by_max_pe(self, monkeypatch):
        p = _provider(monkeypatch)
        assert [c.symbol for c in p.screen("A", {"max_pe": 25.0})] == ["600519"]
        assert p.screen("A", {"max_pe": 10.0}) == []

    def test_screen_drops_candidates_with_no_pe(self, monkeypatch):
        """An unpriceable P/E must not pass a max_pe screen as if it were 0."""
        blank = _row({1: "无PE", 2: "600519", 3: "10.0", 39: "-"}, 88)
        p = _provider(monkeypatch, f'v_sh600519="{blank}";\n')
        assert p.screen("A", {}) != []                 # unfiltered: still returned
        assert p.screen("A", {"max_pe": 25.0}) == []   # filtered: dropped, not kept

    def test_serves_the_fixed_income_sleeve(self, monkeypatch):
        """An all-equity candidate set can never satisfy a liquidity floor."""
        bond = _row({1: "国债ETF", 2: "511010", 3: "141.28"}, 88)
        cash = _row({1: "华宝添益", 2: "511990", 3: "100.00"}, 88)
        p = _provider(monkeypatch, f'v_sh511010="{bond}";\nv_sh511990="{cash}";\n')

        bonds = p.screen("A", {"asset_class": "bond"})
        assert [c.symbol for c in bonds] == ["511010"]
        assert bonds[0].asset_class == "bond"
        assert bonds[0].r_level == "R2"       # bond ETFs are genuinely low-risk

        cashes = p.screen("A", {"asset_class": "cash"})
        assert [c.symbol for c in cashes] == ["511990"]
        assert cashes[0].asset_class == "cash"
        assert cashes[0].r_level == "R1"

    def test_quotes_ignores_symbols_outside_universe(self, monkeypatch):
        p = _provider(monkeypatch, "")
        assert p.quotes(["NOT_IN_UNIVERSE"]) == []

    def test_skips_truncated_rows(self, monkeypatch):
        p = _provider(monkeypatch, 'v_sh600519="1~短~600519~10.0";\n')
        assert p.screen("A", {}) == []

    def test_symbol_prefixing_per_exchange(self):
        from wealthwise.providers.tencent_provider import _prefix

        assert _prefix("600519", "A") == "sh600519"    # Shanghai equity
        assert _prefix("511990", "A") == "sh511990"    # Shanghai fund/ETF
        assert _prefix("159001", "A") == "sz159001"    # Shenzhen fund/ETF
        assert _prefix("000001", "A") == "sz000001"    # Shenzhen equity
        assert _prefix("830799", "A") == "bj830799"    # Beijing
        assert _prefix("700", "HK") == "hk00700"       # zero-padded to 5
        assert _prefix("aapl", "US") == "usAAPL"

    def test_satisfies_market_provider_protocol(self):
        from wealthwise.providers.base import MarketProvider
        from wealthwise.providers.tencent_provider import TencentMarketProvider
        from wealthwise.providers.universe import Universe

        assert isinstance(TencentMarketProvider(Universe(UNIVERSE)), MarketProvider)

    def test_requests_not_imported_at_module_level(self):
        import sys
        import wealthwise.providers.tencent_provider as mod  # noqa: F401

        src = (
            pathlib.Path(mod.__file__).read_text(encoding="utf-8").split("def _get")[0]
        )
        assert "import requests" not in src


class TestUniverse:
    def test_shipped_universe_covers_all_markets(self):
        from wealthwise.providers.universe import Universe

        u = Universe.load()
        for market in ("A", "HK", "US"):
            assert len(u.symbols(market)) > 0, f"{market} equity universe is empty"

    def test_shipped_universe_has_a_fixed_income_sleeve(self):
        """Without bond+cash the portfolio is 100% equity and always downgrades."""
        from wealthwise.providers.universe import Universe

        u = Universe.load()
        for asset_class in ("bond", "cash"):
            assert len(u.symbols("A", asset_class)) > 0, f"no {asset_class} symbols"

    def test_market_of_reverse_lookup(self):
        from wealthwise.providers.universe import Universe

        u = Universe(UNIVERSE)
        assert u.market_of("600519") == "A"
        assert u.market_of("aapl") == "US"      # case-insensitive
        assert u.market_of("UNKNOWN") is None
        assert u.asset_class_of("600519") == "equity"
        assert u.asset_class_of("511990") == "cash"
        assert u.asset_class_of("UNKNOWN") is None

    def test_screen_never_returns_other_markets(self, monkeypatch):
        """Cross-border exclusion must not depend on the endpoint's good behaviour."""
        p = _provider(monkeypatch)   # stub echoes A + HK + US rows for every call
        assert [c.symbol for c in p.screen("A", {})] == ["600519"]
        assert [c.symbol for c in p.screen("HK", {})] == ["00700"]

    def test_screen_never_returns_unrequested_symbols(self, monkeypatch):
        """The asset-class tag feeds the liquidity floor; it must not be guessed."""
        equity = _row({1: "贵州茅台", 2: "600519", 3: "1297.99", 39: "19.93"}, 88)
        cash = _row({1: "华宝添益", 2: "511990", 3: "100.00"}, 88)
        # Stub echoes an equity row on a cash screen — it must not be tagged cash.
        p = _provider(monkeypatch, f'v_sh511990="{cash}";\nv_sh600519="{equity}";\n')

        got = p.screen("A", {"asset_class": "cash"})
        assert [c.symbol for c in got] == ["511990"]
        assert all(c.asset_class == "cash" for c in got)
