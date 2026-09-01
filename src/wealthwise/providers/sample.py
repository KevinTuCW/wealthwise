"""Offline sample providers — backed by data/samples/*.json.

Zero network, zero external dependencies. Used by default when
settings.use_real_providers is False, and as the canonical
offline stack for all tests.
"""
from __future__ import annotations

import json
from pathlib import Path

from wealthwise.agents.state import AssetCandidate


def _load_json(base: Path, filename: str) -> object:
    return json.loads((base / filename).read_text())


class SampleMarketProvider:
    """Offline provider backed by equities.json and funds.json.

    Symbol lookups use casefold so tests and live usage are case-insensitive.
    """

    name = "sample"

    def __init__(self, data_dir: str | None = None) -> None:
        from wealthwise.config import get_settings
        base = Path(data_dir) if data_dir else Path(get_settings().sample_data_dir)

        equities: list[dict] = _load_json(base, "equities.json")  # type: ignore[assignment]
        funds: list[dict] = _load_json(base, "funds.json")        # type: ignore[assignment]
        all_records = equities + funds

        # symbol → record (casefold key)
        self._by_symbol: dict[str, dict] = {
            r["symbol"].casefold(): r for r in all_records
        }
        # all records as list (for screening)
        self._all: list[dict] = all_records

    def quotes(self, symbols: list[str]) -> list[AssetCandidate]:
        out: list[AssetCandidate] = []
        for sym in symbols:
            rec = self._by_symbol.get(sym.casefold())
            if rec:
                out.append(AssetCandidate(**rec))
        return out

    def screen(self, market: str, filters: dict) -> list[AssetCandidate]:
        results: list[AssetCandidate] = []
        for rec in self._all:
            if rec["market"] != market:
                continue
            if "asset_class" in filters and rec["asset_class"] != filters["asset_class"]:
                continue
            if "max_pe" in filters:
                pe = rec.get("metrics", {}).get("pe")
                if pe is not None and pe > filters["max_pe"]:
                    continue
            results.append(AssetCandidate(**rec))
        return results


class SampleMacroProvider:
    """Offline macro provider backed by macro.json."""

    name = "sample"

    def __init__(self, data_dir: str | None = None) -> None:
        from wealthwise.config import get_settings
        base = Path(data_dir) if data_dir else Path(get_settings().sample_data_dir)
        self._data: dict = _load_json(base, "macro.json")  # type: ignore[assignment]

    def snapshot(self) -> dict:
        return dict(self._data)


class SampleFXProvider:
    """Offline FX provider backed by fx.json.

    Pair lookups use casefold (case-insensitive). Raises KeyError for
    unknown pairs so callers can handle missing data explicitly.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        from wealthwise.config import get_settings
        base = Path(data_dir) if data_dir else Path(get_settings().sample_data_dir)
        raw: dict = _load_json(base, "fx.json")  # type: ignore[assignment]
        self._rates: dict[str, dict] = {k.casefold(): v for k, v in raw.items()}

    def rate(self, pair: str) -> float:
        entry = self._rates.get(pair.casefold())
        if entry is None:
            raise KeyError(f"FX pair {pair!r} not available in sample data")
        return float(entry["rate"])
