"""Symbol universe backing the quote-based market provider.

`TencentMarketProvider` reads quotes, not screens, so it needs to be told which
symbols exist. That list is a shipped data file rather than a live call: the
whole point of moving off the eastmoney screener was to stop a flaky network
fetch from being able to fail an advisory run, and re-introducing one here to
fetch the constituent list would put the same failure back a layer down.

Refresh the file with `scripts/refresh_universe.py` (index constituents change
each quarter); the pipeline itself never fetches it.
"""
from __future__ import annotations

import json
from pathlib import Path

_MARKETS = ("A", "HK", "US")

DEFAULT_UNIVERSE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "universe.json"
)


class Universe:
    """Per-market symbol lists, with reverse lookup from symbol to market."""

    def __init__(self, by_market: dict[str, list[str]]) -> None:
        self._by_market = {m: list(by_market.get(m, [])) for m in _MARKETS}
        # Reverse index for quotes(): callers pass bare symbols with no market.
        self._market_of: dict[str, str] = {}
        for market in _MARKETS:
            for symbol in self._by_market[market]:
                self._market_of.setdefault(symbol.casefold(), market)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Universe:
        """Load the universe from JSON: {"A": [...], "HK": [...], "US": [...]}."""
        target = Path(path) if path is not None else DEFAULT_UNIVERSE_PATH
        with open(target, encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls({m: raw.get(m, []) for m in _MARKETS})

    def symbols(self, market: str) -> list[str]:
        return list(self._by_market.get(market, []))

    def market_of(self, symbol: str) -> str | None:
        """Return the market a bare symbol belongs to, or None if unknown."""
        return self._market_of.get(symbol.strip().casefold())

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_market.values())
