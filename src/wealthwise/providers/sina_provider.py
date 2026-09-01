"""Sina-backed market provider — the second opinion on every quote.

Why a second quote source exists
--------------------------------
`ConsensusResolver` has been in the tree since Phase 1 with nothing to
reconcile. One provider yields one reading, and a median over one number is that
number, so pillar one of the cross-check was a class with tests and no caller.
A second *independent* feed is the whole mechanism: different operator,
different upstream, so agreement between them is evidence rather than a feed
agreeing with itself.

`hq.sinajs.cn` batches an arbitrary symbol list into one request, the same shape
as `qt.gtimg.cn`. Cross-checking therefore costs one extra HTTP round trip per
screen, not one per symbol.

What it can corroborate
-----------------------
Price, on all three markets. Market cap and P/E on US names only — the A-share
and Hong Kong payloads carry neither. The consensus layer corroborates what this
feed actually reports and leaves everything else at single-source confidence,
rather than manufacturing agreement out of one source's numbers.

Response format
---------------
GBK-encoded lines of ``var hq_str_<symbol>="f0,f1,f2,…";`` — comma-separated,
with a *different field layout per market* (verified live 2026-09-01):

    A     [0] name   [1] open  [2] prev close  [3] last  [4] high  [5] low
                     [8] volume  [9] turnover
    HK    [0] name(en)  [1] name(zh)  [2] open  [3] prev close  [4] high
                     [5] low  [6] last
    US    [0] name   [1] last  [2] change %  [3] timestamp  [4] change
                     [5] open  [6] high  [7] low  [12] market cap (raw ccy)
                     [13] EPS  [14] P/E

A `Referer` header is mandatory; without one the endpoint answers `Forbidden`.
`_get()` is the only method that touches the network; tests monkeypatch it.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from wealthwise.agents.state import AssetCandidate

if TYPE_CHECKING:
    from wealthwise.providers.universe import Universe

_QUOTE_URL = "https://hq.sinajs.cn/list="

# Matched to the Tencent provider's chunking so both sources page the universe
# the same way and a disagreement can never be an artefact of one feed having
# been asked for a different slice than the other.
_CHUNK = 200

_TIMEOUT = 15

# Sina rejects a request with no Referer outright ("Forbidden"), so this is a
# functional requirement rather than politeness.
_HEADERS = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
}

_LINE = re.compile(r'var\s+hq_str_([A-Za-z0-9_]+)\s*=\s*"(.*)"\s*;')

_CURRENCY = {"A": "CNY", "HK": "HKD", "US": "USD"}

# Same rating table as the Tencent provider: real ratings for the fixed-income
# sleeve, a placeholder for equity, which a spot quote cannot rate.
_R_LEVEL = {"cash": "R1", "bond": "R2", "equity": "R3"}

# Per-market field positions (see module docstring). Only fields this feed
# genuinely carries are listed — an absent entry means "this market's payload
# does not report it", which is different from "it reported nothing today".
_LAYOUT = {
    "A": {"name": 0, "price": 3},
    "HK": {"name": 1, "price": 6},
    "US": {"name": 0, "price": 1, "market_cap_raw": 12, "pe": 14},
}

# Shortest usable payload per market. A suspended or delisted symbol comes back
# as a bare `""`, which must be skipped rather than parsed into zeros.
_MIN_FIELDS = {"A": 6, "HK": 7, "US": 15}


def _prefix(symbol: str, market: str) -> str:
    """Map a bare symbol to the exchange-prefixed form hq.sinajs.cn expects.

    The A-share rules match the Tencent provider's exactly — including the
    5xxxxx Shanghai fund range that carries the entire fixed-income sleeve.
    """
    s = symbol.strip()
    if market == "HK":
        return f"hk{s.zfill(5)}"
    if market == "US":
        return f"gb_{s.lower()}"
    head = s[:1]
    if head in ("5", "6", "9"):
        return f"sh{s}"
    if head in ("4", "8"):
        return f"bj{s}"
    return f"sz{s}"


def _unprefix(key: str) -> tuple[str, str]:
    """Recover (symbol, market) from a prefixed response key."""
    if key.startswith("gb_"):
        return key[3:].upper(), "US"
    if key.startswith("hk"):
        # Hong Kong codes are zero-padded to five on the way out; the universe
        # stores them the same way, so no stripping is wanted here.
        return key[2:], "HK"
    return key[2:], "A"


def _to_float(raw: str) -> float | None:
    """Parse a numeric field, treating blanks and sentinels as absent."""
    text = (raw or "").strip()
    if not text or text in ("-", "--"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _map_quote(key: str, raw: str, asset_class: str) -> dict | None:
    """Map one response line to AssetCandidate kwargs, or None if unusable.

    A halted symbol answers with every price field at 0.000 (verified live on a
    Beijing-listed name). Those zeros must not become readings: a 0 alongside a
    real 1299 makes the two-source median 649, which is not a disagreement the
    resolver can catch — it is a fabricated price presented with confidence.
    """
    symbol, market = _unprefix(key)
    fields = raw.split(",")
    if len(fields) < _MIN_FIELDS[market]:
        return None

    layout = _LAYOUT[market]
    price = _to_float(fields[layout["price"]])
    if price is None or price <= 0:
        return None

    metrics: dict = {"price": price}

    idx = layout.get("market_cap_raw")
    if idx is not None and len(fields) > idx:
        cap = _to_float(fields[idx])
        if cap is not None and cap > 0:
            # Sina reports US market cap in raw currency units; the domain metric
            # is 亿 (100M) of the local currency, matching the Tencent feed.
            metrics["market_cap_100m"] = cap / 1e8

    idx = layout.get("pe")
    if idx is not None and len(fields) > idx:
        pe = _to_float(fields[idx])
        # A P/E of exactly 0 is Sina's "not reported" for this field, not a
        # company trading at zero earnings multiple.
        if pe is not None and pe != 0:
            metrics["pe"] = pe

    return {
        "symbol": symbol,
        "market": market,
        "asset_class": asset_class,
        "name": (fields[layout["name"]] or "").strip(),
        "currency": _CURRENCY[market],
        "r_level": _R_LEVEL.get(asset_class, "R3"),
        "metrics": metrics,
        "tags": [],
    }


def _parse(payload: str, class_of: dict[str, str] | None = None) -> list[dict]:
    """Parse a full quote response into AssetCandidate kwarg dicts."""
    out: list[dict] = []
    for match in _LINE.finditer(payload):
        key, raw = match.group(1), match.group(2)
        mapped = _map_quote(key, raw, (class_of or {}).get(key, "equity"))
        if mapped is not None:
            out.append(mapped)
    return out


class SinaMarketProvider:
    """A/HK/US spot quotes via hq.sinajs.cn — the corroborating source.

    Parameters
    ----------
    universe:
        Symbol universe backing `screen()`, injected the same way the Tencent
        provider takes it so both sources screen the identical symbol list.
    """

    name = "sina"

    def __init__(self, universe: Universe) -> None:
        self._universe = universe

    def _get(self, prefixed: list[str]) -> str:
        """Fetch raw GBK quote text for already-prefixed symbols.

        The only method that touches the network; tests monkeypatch it.
        """
        import requests  # lazy — keeps the import off the offline test path

        session = requests.Session()
        # Same reason as the Tencent provider: the macOS system proxy mangles
        # these endpoints, and nothing here sits behind a corporate egress.
        session.trust_env = False

        chunks: list[str] = []
        for i in range(0, len(prefixed), _CHUNK):
            batch = prefixed[i:i + _CHUNK]
            response = session.get(_QUOTE_URL + ",".join(batch),
                                   headers=_HEADERS, timeout=_TIMEOUT)
            response.raise_for_status()
            response.encoding = "gbk"
            chunks.append(response.text)
        return "\n".join(chunks)

    def _fetch(self, prefixed: list[str],
               class_of: dict[str, str] | None = None) -> list[AssetCandidate]:
        if not prefixed:
            return []
        return [AssetCandidate(**row) for row in _parse(self._get(prefixed), class_of)]

    def quotes(self, symbols: list[str]) -> list[AssetCandidate]:
        """Fetch AssetCandidates for the given bare symbols, resolved via the universe."""
        prefixed: list[str] = []
        class_of: dict[str, str] = {}
        for symbol in symbols:
            market = self._universe.market_of(symbol)
            if market is None:
                continue
            key = _prefix(symbol, market)
            prefixed.append(key)
            class_of[key] = self._universe.asset_class_of(symbol) or "equity"
        return self._fetch(prefixed, class_of)

    def screen(self, market: str, filters: dict) -> list[AssetCandidate]:
        """Return universe candidates in `market` matching the filter dict.

        Supported filters: asset_class, max_pe. This feed reports no P/E outside
        the US layout, so a `max_pe` screen run against it alone would pass every
        A-share and Hong Kong name through unfiltered. That is precisely why this
        provider is registered as a corroborating source rather than as a primary
        one — see providers/consensus_provider.py, which screens on the primary
        and uses this feed only to cross-check the numbers that come back.
        """
        asset_class = filters.get("asset_class", "equity")
        wanted = self._universe.symbols(market, asset_class)
        expected = {s.casefold() for s in wanted}
        prefixed = [_prefix(s, market) for s in wanted]
        class_of = dict.fromkeys(prefixed, asset_class)

        out: list[AssetCandidate] = []
        for candidate in self._fetch(prefixed, class_of):
            if candidate.market != market:
                continue
            if candidate.symbol.casefold() not in expected:
                continue
            if "max_pe" in filters:
                pe = candidate.metrics.get("pe")
                if pe is not None and pe > filters["max_pe"]:
                    continue
            out.append(candidate)
        return out
