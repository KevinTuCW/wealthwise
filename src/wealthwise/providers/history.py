"""Daily price history — the input the factor model cannot fake.

Why this exists
---------------
Two of the five factors, momentum and realized volatility, are not in any spot
quote. Without them the "multi-factor" score is three cross-sectional ranks over
valuation and size, which is a screen, not a factor model.

It also closes a hole that predates factor scoring. `optimize.py` weights inside
each asset class by inverse volatility and falls back to `DEFAULT_VOL = 0.15`
for any candidate carrying no `volatility` metric — and no live provider carried
one. Every real-provider run was therefore weighting a money-market ETF and a
small-cap equity as though they had identical risk, which makes inverse-vol
weighting an elaborate way to write equal weight.

Cost
----
`web.ifzq.gtimg.cn` serves one symbol per request; the `;`-joined batch form the
quote endpoint accepts is rejected here (`param error`, verified live). Fetching
is therefore concurrent rather than batched: measured 20 symbols in ~0.4s at 8
workers, so a 300-name book costs a few seconds, once, before ranking.

`_get()` is the only method that touches the network; tests monkeypatch it.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wealthwise.agents.state import AssetCandidate
    from wealthwise.providers.universe import Universe

_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

_TIMEOUT = 15

# Concurrency for the per-symbol fetch. Eight is where the measured wall time
# stopped improving on this endpoint; higher just makes a public feed angrier.
_WORKERS = 8

# Trading days pulled per symbol. Enough for a 60-day momentum window with room
# for the reversal skip below, and short enough that one request stays small.
_BARS = 70

# Momentum is measured over this many trading days...
_MOMENTUM_WINDOW = 60

# ...ending five sessions before today. The skip is standard practice and it is
# not cosmetic: raw "last 60 days including this week" momentum is dominated by
# short-term reversal, so the factor ends up buying whatever just spiked and
# selling whatever just dipped — the opposite of the effect it is named for.
_MOMENTUM_SKIP = 5

# Below this many bars a volatility estimate is noise dressed as a number, and a
# momentum window does not exist at all. Such names get no history metrics and
# are scored on the factors they do have.
_MIN_BARS = 30

# Trading days per year, for annualising a daily standard deviation.
_TRADING_DAYS = 252


def _prefix(symbol: str, market: str) -> str:
    """Exchange-prefixed symbol form, matching the Tencent quote provider's rules."""
    s = symbol.strip()
    if market == "HK":
        return f"hk{s.zfill(5)}"
    if market == "US":
        return f"us{s.upper()}"
    head = s[:1]
    if head in ("5", "6", "9"):
        return f"sh{s}"
    if head in ("4", "8"):
        return f"bj{s}"
    return f"sz{s}"


def _closes(payload: str) -> list[float]:
    """Extract the close series from one k-line response, oldest first.

    Shape: ``{"data": {"<prefixed>": {"qfqday"|"day": [[date, open, close,
    high, low, volume], …]}}}``. The key varies with whether the series is
    adjusted, so both are accepted.
    """
    try:
        data = json.loads(payload).get("data")
    except (ValueError, AttributeError):
        return []
    if not isinstance(data, dict):
        return []

    for series in data.values():
        if not isinstance(series, dict):
            continue
        bars = series.get("qfqday") or series.get("day") or []
        closes: list[float] = []
        for bar in bars:
            try:
                closes.append(float(bar[2]))
            except (IndexError, TypeError, ValueError):
                continue
        if closes:
            return closes
    return []


def realized_volatility(closes: list[float]) -> float | None:
    """Annualised standard deviation of daily log returns.

    Log returns rather than simple ones: they are additive across days, which is
    what makes the √252 scaling to an annual figure valid in the first place.
    """
    if len(closes) < _MIN_BARS:
        return None
    rets = [
        math.log(b / a)
        for a, b in zip(closes, closes[1:])
        if a > 0 and b > 0
    ]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    variance = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(variance) * math.sqrt(_TRADING_DAYS)


def momentum(closes: list[float]) -> float | None:
    """Total return over `_MOMENTUM_WINDOW` sessions, skipping the last `_MOMENTUM_SKIP`."""
    if len(closes) < _MIN_BARS:
        return None
    end = len(closes) - _MOMENTUM_SKIP
    if end <= 0:
        return None
    start = max(0, end - _MOMENTUM_WINDOW)
    first, last = closes[start], closes[end - 1]
    if first <= 0:
        return None
    return last / first - 1.0


class TencentHistoryProvider:
    """Daily bars via web.ifzq.gtimg.cn, fetched concurrently.

    Parameters
    ----------
    universe:
        Used only to resolve a bare symbol's market when a candidate does not
        carry one; candidates from the pipeline always do, so this is optional.
    """

    name = "tencent-kline"

    def __init__(self, universe: "Universe | None" = None,
                 workers: int = _WORKERS) -> None:
        self._universe = universe
        self._workers = workers

    def _get(self, prefixed: str) -> str:
        """Fetch the raw k-line JSON for one prefixed symbol."""
        import requests  # lazy — keeps the import off the offline test path

        session = requests.Session()
        session.trust_env = False       # same proxy reason as the quote providers
        response = session.get(
            _KLINE_URL,
            params={"param": f"{prefixed},day,,,{_BARS},qfq"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return response.text

    def closes(self, symbols: list[tuple[str, str]]) -> dict[str, list[float]]:
        """Fetch close series for (symbol, market) pairs, keyed by bare symbol.

        A symbol whose fetch fails returns no series rather than raising. History
        is an enrichment: losing it costs two factors on one name, and taking the
        whole advisory down over a k-line request would trade a better ranking
        for no ranking at all.
        """
        if not symbols:
            return {}

        def fetch(pair: tuple[str, str]) -> tuple[str, list[float]]:
            symbol, market = pair
            try:
                return symbol, _closes(self._get(_prefix(symbol, market)))
            except Exception:
                return symbol, []

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            results = list(pool.map(fetch, symbols))
        return {sym: series for sym, series in results if series}

    def enrich(self, candidates: list["AssetCandidate"]) -> list["AssetCandidate"]:
        """Return candidates with `momentum` and `volatility` filled in from history.

        `volatility` is written even though nothing in the factor path strictly
        needs it there: the portfolio optimiser reads that exact key and has been
        falling back to a 0.15 constant for every live candidate.
        """
        pairs = [(c.symbol, c.market) for c in candidates]
        series = self.closes(pairs)

        out: list["AssetCandidate"] = []
        for candidate in candidates:
            closes = series.get(candidate.symbol)
            if not closes:
                out.append(candidate)
                continue
            metrics = dict(candidate.metrics)
            vol = realized_volatility(closes)
            if vol is not None:
                metrics["volatility"] = round(vol, 6)
            mom = momentum(closes)
            if mom is not None:
                metrics["momentum"] = round(mom, 6)
            metrics["history_bars"] = len(closes)
            out.append(candidate.model_copy(update={"metrics": metrics}))
        return out
