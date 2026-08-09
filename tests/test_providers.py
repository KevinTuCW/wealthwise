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
