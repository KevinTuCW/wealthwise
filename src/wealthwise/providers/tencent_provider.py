"""Tencent-backed market provider — batched A/HK/US spot quotes.

Why this exists
---------------
The AkShare eastmoney screener (`stock_zh_a_spot_em`, host
`82.push2.eastmoney.com`) is not dependably reachable: TCP and TLS succeed, then
the connection is reset mid-stream on roughly two calls out of three. A provider
that fails a third of the time is worse than a slower one that does not, because
the failure lands inside `equity_node` and takes the whole advisory run with it.

`qt.gtimg.cn` answers A, HK and US in a *single* batched request — 200 symbols in
~2s, one round trip instead of the screener's paginated fetch. Fewer requests is
the reliability win here; every extra page is another chance to be reset.

Universe, not full-market scan
------------------------------
This is a quote API, so it needs a symbol list rather than returning the whole
market. `screen()` therefore quotes a curated index universe (see
`data/universe.json`) instead of all ~5,400 A-shares. That is a deliberate
narrowing, not a limitation worked around: `cap_node` truncates to 50 candidates
anyway, so the full-market scan was discarded downstream, and screening from
index constituents is the more defensible universe for an advisory system.

Response format
---------------
GBK-encoded lines of `v_<symbol>="f0~f1~f2~...";`. Field positions verified live
against all three markets:

    [1]  name          [2]  code          [3]  last price
    [39] P/E (TTM)     [45] market cap (100M, local currency)
    [46] P/B  —  A-shares only; HK/US carry the English name at this position

`_get()` is the only method that touches the network; tests monkeypatch it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from wealthwise.agents.state import AssetCandidate

if TYPE_CHECKING:
    from wealthwise.providers.universe import Universe

_QUOTE_URL = "https://qt.gtimg.cn/q="

# 200 verified to answer in one request; kept as the chunk size so a large
# universe degrades into a few requests rather than one oversized URL.
_CHUNK = 200

_TIMEOUT = 15

# Field positions in the `~`-split payload (verified live, see module docstring).
_F_NAME = 1
_F_CODE = 2
_F_PRICE = 3
_F_PE = 39
_F_MCAP = 45
_F_PB = 46

# Shortest payload observed is the US layout at 71 fields; anything below the
# P/E index is a truncated or error row and is skipped rather than half-parsed.
_MIN_FIELDS = _F_PE + 1

_CURRENCY = {"A": "CNY", "HK": "HKD", "US": "USD"}


def _prefix(symbol: str, market: str) -> str:
    """Map a bare symbol to the exchange-prefixed form qt.gtimg.cn expects."""
    s = symbol.strip()
    if market == "HK":
        return f"hk{s.zfill(5)}"
    if market == "US":
        return f"us{s.upper()}"
    # A-shares: 6/9 → Shanghai, 0/2/3 → Shenzhen, 4/8 → Beijing.
    head = s[:1]
    if head in ("6", "9"):
        return f"sh{s}"
    if head in ("4", "8"):
        return f"bj{s}"
    return f"sz{s}"


def _market_of(prefixed: str) -> str:
    if prefixed.startswith("hk"):
        return "HK"
    if prefixed.startswith("us"):
        return "US"
    return "A"


def _to_float(raw: str) -> float | None:
    """Parse a numeric field, treating blanks and sentinels as absent.

    Tencent writes an empty string or "-" for a metric it has no value for.
    Those must stay None: a P/E coerced to 0.0 reads as "extremely cheap" and
    would sail through every max_pe filter in the screener.
    """
    text = (raw or "").strip()
    if not text or text in ("-", "--"):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value


def _map_quote_row(fields: list[str], market: str) -> dict | None:
    """Map one `~`-split quote row to AssetCandidate kwargs, or None if unusable."""
    if len(fields) < _MIN_FIELDS:
        return None

    # US codes come back suffixed with the venue ("AAPL.OQ"); keep the ticker.
    code = (fields[_F_CODE] or "").strip()
    symbol = code.split(".")[0] if market == "US" else code
    if not symbol:
        return None

    metrics: dict = {}
    pe = _to_float(fields[_F_PE])
    if pe is not None:
        metrics["pe"] = pe
    price = _to_float(fields[_F_PRICE])
    if price is not None:
        metrics["price"] = price
    if len(fields) > _F_MCAP:
        mcap = _to_float(fields[_F_MCAP])
        if mcap is not None:
            metrics["market_cap_100m"] = mcap
    # P/B only occupies this slot on the A-share layout; on HK/US it is the
    # English company name, which _to_float rejects — but guard on market too
    # so the intent is readable rather than relying on the parse failing.
    if market == "A" and len(fields) > _F_PB:
        pb = _to_float(fields[_F_PB])
        if pb is not None:
            metrics["pb"] = pb

    return {
        "symbol": symbol,
        "market": market,
        "asset_class": "equity",
        "name": (fields[_F_NAME] or "").strip(),
        "currency": _CURRENCY[market],
        # Spot quotes carry no suitability rating. R3 is the same neutral
        # placeholder the AkShare provider used; deriving a real r_level needs a
        # volatility history fetch and is left to the risk-scoring task rather
        # than guessed from a single day's price.
        "r_level": "R3",
        "metrics": metrics,
        "tags": [],
    }


def _parse(payload: str) -> list[dict]:
    """Parse a full quote response into AssetCandidate kwarg dicts."""
    out: list[dict] = []
    for line in payload.splitlines():
        line = line.strip()
        if not line.startswith("v_") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        fields = value.strip().rstrip(";").strip('"').split("~")
        mapped = _map_quote_row(fields, _market_of(key[2:]))
        if mapped is not None:
            out.append(mapped)
    return out


class TencentMarketProvider:
    """Live A/HK/US equity quotes via qt.gtimg.cn.

    Parameters
    ----------
    universe:
        Symbol universe backing `screen()`. Injected so tests and the refresh
        script can supply their own list without touching the network.
    """

    def __init__(self, universe: Universe) -> None:
        self._universe = universe

    def _get(self, prefixed: list[str]) -> str:
        """Fetch raw GBK quote text for already-prefixed symbols.

        The only method that touches the network; tests monkeypatch it.
        """
        import requests  # lazy — keeps the import off the offline test path

        session = requests.Session()
        # The macOS system proxy (read via urllib's getproxies(), not just the
        # env vars) mangles this endpoint; go direct. Nothing here is behind a
        # corporate egress, so bypassing costs nothing.
        session.trust_env = False

        chunks: list[str] = []
        for i in range(0, len(prefixed), _CHUNK):
            batch = prefixed[i:i + _CHUNK]
            response = session.get(_QUOTE_URL + ",".join(batch), timeout=_TIMEOUT)
            response.raise_for_status()
            response.encoding = "gbk"
            chunks.append(response.text)
        return "\n".join(chunks)

    def _fetch(self, prefixed: list[str]) -> list[AssetCandidate]:
        if not prefixed:
            return []
        return [AssetCandidate(**row) for row in _parse(self._get(prefixed))]

    def quotes(self, symbols: list[str]) -> list[AssetCandidate]:
        """Fetch AssetCandidates for the given symbols (best-effort).

        Symbols are resolved against the universe to recover their market, so a
        bare "600519" or "AAPL" works the same way it did on the sample provider.
        """
        prefixed: list[str] = []
        for symbol in symbols:
            market = self._universe.market_of(symbol)
            if market is not None:
                prefixed.append(_prefix(symbol, market))
        return self._fetch(prefixed)

    def screen(self, market: str, filters: dict) -> list[AssetCandidate]:
        """Return universe candidates in `market` matching the filter dict.

        Supported filters: asset_class (exact match), max_pe.
        """
        if filters.get("asset_class") and filters["asset_class"] != "equity":
            return []   # this provider only serves equities

        prefixed = [_prefix(s, market) for s in self._universe.symbols(market)]
        out: list[AssetCandidate] = []
        for candidate in self._fetch(prefixed):
            # Only requested symbols come back in practice, so this looks
            # redundant — but "A-shares only" is a compliance constraint when
            # the investor declined cross-border exposure, and that invariant
            # should not rest on the endpoint echoing exactly what it was asked.
            if candidate.market != market:
                continue
            if "max_pe" in filters:
                pe = candidate.metrics.get("pe")
                # A name with no P/E (loss-making, or simply not reported) is
                # unscreenable on this filter, so it is dropped rather than
                # admitted — a max_pe screen that silently keeps unknowns is
                # not the screen the caller asked for.
                if pe is None or pe > filters["max_pe"]:
                    continue
            out.append(candidate)
        return out
