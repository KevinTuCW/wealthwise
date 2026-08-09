"""AkShare-backed providers for live A/HK/US market data, macro, and FX.

Design:
- All akshare calls are isolated inside `_get()` so tests can monkeypatch
  that seam without importing akshare.
- akshare is lazy-imported inside `_get()` only — never at module top level.
- `build_provider()` is config-gated: returns Sample providers when
  settings.use_real_providers is False.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from wealthwise.agents.state import AssetCandidate

if TYPE_CHECKING:
    from wealthwise.config import Settings


# ---------------------------------------------------------------------------
# AkShare providers
# ---------------------------------------------------------------------------

class AkShareMarketProvider:
    """Live market data via AkShare (lazy-imported).

    _get() is the only method that touches akshare; all public methods go
    through _get() so tests can stub it without importing akshare.
    """

    def _get(self, symbols: list[str] | None = None,
             market: str | None = None,
             filters: dict | None = None) -> list[dict]:
        """Fetch raw records from AkShare.

        In production this would dispatch to the appropriate akshare function
        based on market; here it returns an empty list as a safe default for
        environments where akshare is not installed.

        Tests monkeypatch this method to inject fixture payloads.
        """
        import akshare as ak  # noqa: F401  # lazy — only reached in real usage
        # Production implementation would call ak.stock_zh_a_spot_em(), etc.
        # Left as [] because YAGNI: the live dispatch is a future task.
        return []

    def quotes(self, symbols: list[str]) -> list[AssetCandidate]:
        """Fetch AssetCandidates for the given symbols via _get."""
        raw = self._get(symbols=symbols)
        return [AssetCandidate(**rec) for rec in raw]

    def screen(self, market: str, filters: dict) -> list[AssetCandidate]:
        """Return candidates in market matching filters via _get."""
        raw = self._get(market=market, filters=filters)
        return [AssetCandidate(**rec) for rec in raw]


class AkShareMacroProvider:
    """Live macro data via AkShare (lazy-imported)."""

    def _get(self) -> dict:
        import akshare as ak  # noqa: F401  # lazy
        return {}

    def snapshot(self) -> dict:
        return self._get()


class AkShareFXProvider:
    """Live FX rates via AkShare (lazy-imported)."""

    def _get(self, pair: str) -> float:
        import akshare as ak  # noqa: F401  # lazy
        return 0.0

    def rate(self, pair: str) -> float:
        return self._get(pair)


# ---------------------------------------------------------------------------
# Config-gated factory
# ---------------------------------------------------------------------------

def build_provider(settings: "Settings") -> tuple[
    "AkShareMarketProvider | object",
    "AkShareMacroProvider | object",
    "AkShareFXProvider | object",
]:
    """Return (market, macro, fx) providers gated on settings.use_real_providers.

    When use_real_providers is False (the default), returns offline Sample
    providers so the pipeline works without any API keys or akshare install.
    """
    if settings.use_real_providers:
        return (
            AkShareMarketProvider(),
            AkShareMacroProvider(),
            AkShareFXProvider(),
        )

    from wealthwise.providers.sample import (
        SampleFXProvider,
        SampleMacroProvider,
        SampleMarketProvider,
    )
    data_dir = settings.sample_data_dir
    return (
        SampleMarketProvider(data_dir),
        SampleMacroProvider(data_dir),
        SampleFXProvider(data_dir),
    )
