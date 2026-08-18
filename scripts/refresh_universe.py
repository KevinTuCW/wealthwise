"""Regenerate data/universe.json — the symbol universe the screener quotes.

Run manually when index constituents change (roughly quarterly):

    PYTHONPATH=src .venv/bin/python scripts/refresh_universe.py

A-shares come from the CSI 300 + CSI 500 constituent lists published by China
Securities Index (host `csindex.com.cn`, reached via akshare) — a different host from the
eastmoney screener this provider replaced, and reachable where that one is not.

HK and US are curated large-cap lists. There is no equally clean free
constituent feed for those, and inventing one is worse than a reviewed list: an
advisory universe is a decision about what the product will ever recommend, so
it should be legible in the diff rather than whatever an endpoint returned.

Every symbol is verified against the live quote endpoint before the file is
written — a symbol that does not price is dropped and reported, so a delisting
shows up here rather than as a silently thinner screen at runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wealthwise.providers.tencent_provider import (  # noqa: E402
    TencentMarketProvider,
    _prefix,
)
from wealthwise.providers.universe import DEFAULT_UNIVERSE_PATH, Universe  # noqa: E402

# CSI 300 + CSI 500 = the CSI 800, the standard large- and mid-cap A-share
# universe. CSI 300 alone was too narrow to select from once the screener
# started ranking rather than taking whatever came first.
A_INDEXES = ("000300", "000905")

# Hang Seng large caps across the main sectors (tech, financials, energy,
# property, consumer) — the HK slice an advisory portfolio would draw from.
HK_SYMBOLS = [
    "00700", "09988", "03690", "00941", "00939", "01299", "00388", "00005",
    "01398", "03988", "00883", "00857", "00386", "01810", "02318", "02020",
    "00001", "00016", "00011", "00027", "00175", "02331", "01113", "00688",
    "09618", "09999", "01024", "02269", "06862", "01876", "00291", "00322",
]

# Exchange-traded money-market funds — the cash sleeve. These quote on the same
# endpoint as equities, so the whole universe stays on one reliable request.
CASH_SYMBOLS = ["511990", "511880", "511660", "511690", "511900", "159001"]

# Government / local-government / corporate bond ETFs — the fixed-income sleeve,
# spread across the curve rather than all sitting in ten-year treasuries.
BOND_SYMBOLS = [
    "511010", "511260", "511020", "511090", "511030", "511220",
    "159972", "159926",
]

# US large caps: the mega-cap tech complex plus enough non-tech (healthcare,
# financials, staples, energy) that the screener is not a one-sector bet.
US_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "BRK.B", "JPM", "V", "MA", "UNH", "JNJ", "LLY", "ABBV",
    "XOM", "CVX", "WMT", "PG", "KO", "PEP", "COST", "MCD",
    "HD", "CRM", "ORCL", "AMD", "NFLX", "ADBE", "INTC", "QCOM",
    "CSCO", "TXN", "IBM", "GE", "CAT", "BA", "DIS", "NKE",
]


def fetch_a_universe() -> list[str]:
    """Return CSI 800 constituent codes from the index publisher."""
    import akshare as ak

    codes: list[str] = []
    seen: set[str] = set()
    for index in A_INDEXES:
        df = ak.index_stock_cons_csindex(symbol=index)
        column = next((c for c in ("成分券代码", "品种代码", "成分股代码")
                       if c in df.columns), None)
        if column is None:
            raise SystemExit(f"unexpected constituent columns: {list(df.columns)}")
        for value in df[column].tolist():
            code = str(value).zfill(6)
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def verify(symbols: list[str], market: str) -> tuple[list[str], list[str]]:
    """Split `symbols` into those that price on the live endpoint and those that don't."""
    provider = TencentMarketProvider(Universe({}))
    priced = {
        c.symbol.casefold()
        for c in provider._fetch([_prefix(s, market) for s in symbols])
    }
    # US tickers come back without the venue suffix; A/HK codes round-trip as-is.
    ok = [s for s in symbols if s.split(".")[0].casefold() in priced]
    missing = [s for s in symbols if s.split(".")[0].casefold() not in priced]
    return ok, missing


def main() -> int:
    universe: dict[str, dict[str, list[str]]] = {}
    total = 0
    for asset_class, market, symbols in (
        ("equity", "A", fetch_a_universe()),
        ("equity", "HK", HK_SYMBOLS),
        ("equity", "US", US_SYMBOLS),
        ("bond", "A", BOND_SYMBOLS),
        ("cash", "A", CASH_SYMBOLS),
    ):
        ok, missing = verify(symbols, market)
        universe.setdefault(asset_class, {})[market] = ok
        total += len(ok)
        print(f"{asset_class:6} {market:3} kept {len(ok):4} / {len(symbols):4}")
        if missing:
            print(f"    dropped (no quote): {', '.join(missing)}")

    DEFAULT_UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_UNIVERSE_PATH, "w", encoding="utf-8") as fh:
        json.dump(universe, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {DEFAULT_UNIVERSE_PATH} ({total} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
