"""Provider Protocols — the stable interfaces the pipeline depends on.

Swap Sample providers for live AkShare providers without touching any
downstream agent — they only depend on these interfaces.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from wealthwise.agents.state import AssetCandidate


@runtime_checkable
class MarketProvider(Protocol):
    """Source of asset candidates and screening results."""

    def quotes(self, symbols: list[str]) -> list[AssetCandidate]:
        """Fetch AssetCandidates for the given symbol list (best-effort)."""
        ...

    def screen(self, market: str, filters: dict) -> list[AssetCandidate]:
        """Return candidates in the given market matching the filter dict.

        Supported filter keys (all optional):
            asset_class (str)  — exact match
            max_pe (float)     — metrics["pe"] <= max_pe
        """
        ...


@runtime_checkable
class MacroProvider(Protocol):
    """Source of macro-economic snapshots."""

    def snapshot(self) -> dict:
        """Return the latest macro data as a plain dict."""
        ...


@runtime_checkable
class FXProvider(Protocol):
    """Source of FX spot rates."""

    def rate(self, pair: str) -> float:
        """Return the spot rate for the given pair (e.g. 'USDCNH').

        Raises KeyError if the pair is not available.
        """
        ...
