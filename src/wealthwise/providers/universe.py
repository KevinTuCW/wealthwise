"""Symbol universe backing the quote-based market provider.

`TencentMarketProvider` reads quotes, not screens, so it needs to be told which
symbols exist. That list is a shipped data file rather than a live call: the
whole point of moving off the eastmoney screener was to stop a flaky network
fetch from being able to fail an advisory run, and re-introducing one here to
fetch the constituent list would put the same failure back a layer down.

Shape: ``{asset_class: {market: [symbol, ...]}}``. The asset-class dimension is
not decoration — `portfolio_node` screens for bond and cash separately from
equity, and a universe that only knows about equities leaves the portfolio 100%
stock, which no liquidity floor can ever satisfy.

Refresh the file with `scripts/refresh_universe.py` (index constituents change
each quarter); the pipeline itself never fetches it.
"""
from __future__ import annotations

import json
from pathlib import Path

MARKETS = ("A", "HK", "US")
ASSET_CLASSES = ("equity", "bond", "cash")

DEFAULT_UNIVERSE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "universe.json"
)


class Universe:
    """Per-asset-class, per-market symbol lists, with reverse lookup by symbol."""

    def __init__(self, by_class: dict[str, dict[str, list[str]]]) -> None:
        self._by_class: dict[str, dict[str, list[str]]] = {
            cls: {m: list(by_class.get(cls, {}).get(m, [])) for m in MARKETS}
            for cls in ASSET_CLASSES
        }
        # Reverse index for quotes(): callers pass bare symbols with no market.
        self._lookup: dict[str, tuple[str, str]] = {}
        for cls in ASSET_CLASSES:
            for market in MARKETS:
                for symbol in self._by_class[cls][market]:
                    self._lookup.setdefault(symbol.casefold(), (market, cls))

    @classmethod
    def load(cls, path: str | Path | None = None) -> Universe:
        """Load the universe from JSON: {asset_class: {market: [symbols]}}."""
        target = Path(path) if path is not None else DEFAULT_UNIVERSE_PATH
        with open(target, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def symbols(self, market: str, asset_class: str = "equity") -> list[str]:
        return list(self._by_class.get(asset_class, {}).get(market, []))

    def market_of(self, symbol: str) -> str | None:
        """Return the market a bare symbol belongs to, or None if unknown."""
        found = self._lookup.get(symbol.strip().casefold())
        return found[0] if found else None

    def asset_class_of(self, symbol: str) -> str | None:
        """Return the asset class a bare symbol belongs to, or None if unknown."""
        found = self._lookup.get(symbol.strip().casefold())
        return found[1] if found else None

    def __len__(self) -> int:
        return sum(len(v) for cls in self._by_class.values() for v in cls.values())
