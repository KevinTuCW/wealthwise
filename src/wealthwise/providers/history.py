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

#: One daily bar: (date "YYYY-MM-DD", close, volume).
Bar = tuple[str, float, float]

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


# Venue suffixes tried for a US name whose quote did not carry one, Nasdaq
# first. Guessing costs at most one extra request and only for the fallback
# path; the quote provider supplies the real suffix in `venue_code`.
_US_VENUES = (".OQ", ".N")


def _prefix(symbol: str, market: str, venue_code: str | None = None) -> str:
    """Exchange-prefixed symbol form, matching the Tencent quote provider's rules.

    US names need the venue suffix the quote feed reports (`AAPL.OQ`, `JPM.N`).
    The bare ticker is accepted by the k-line endpoint and answers with two bars
    regardless of the requested count — enough to look like a working request and
    never enough to clear `_MIN_BARS`.
    """
    s = symbol.strip()
    if market == "HK":
        return f"hk{s.zfill(5)}"
    if market == "US":
        if venue_code and "." in venue_code:
            return f"us{venue_code.strip().upper()}"
        return f"us{s.upper()}"
    head = s[:1]
    if head in ("5", "6", "9"):
        return f"sh{s}"
    if head in ("4", "8"):
        return f"bj{s}"
    return f"sz{s}"


def _series(payload: str) -> list[Bar]:
    """Extract (date, close, volume) bars from one k-line response, oldest first.

    Shape: ``{"data": {"<prefixed>": {"qfqday"|"day": [[date, open, close,
    high, low, volume], …]}}}``. The key varies with whether the series is
    adjusted, so both are accepted.

    Date and volume are carried alongside the close because the factor
    backtest needs them — it has to know *when* a bar was and how much traded,
    to line the cross-section up across symbols and to rebuild turnover as of a
    past date. The pipeline itself only ever reads closes.
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
        out: list[Bar] = []
        for bar in bars:
            try:
                close = float(bar[2])
            except (IndexError, TypeError, ValueError):
                continue
            try:
                volume = float(bar[5])
            except (IndexError, TypeError, ValueError):
                volume = 0.0        # optional: only the backtest reads it
            out.append((str(bar[0]), close, volume))
        if out:
            return out
    return []


def _closes(payload: str) -> list[float]:
    """The close series from one k-line response, oldest first."""
    return [close for _, close, _ in _series(payload)]


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
    bars:
        Daily bars requested per symbol. The pipeline wants the minimum that
        supports the momentum window; `scripts/backtest_factors.py` asks for
        years of it. Same endpoint, same parser, one parameter apart — the
        alternative was a second fetcher, which is how a backtest ends up
        measuring data the live path never sees.
    """

    name = "tencent-kline"

    def __init__(self, universe: "Universe | None" = None,
                 workers: int = _WORKERS, bars: int = _BARS) -> None:
        self._universe = universe
        self._workers = workers
        self._bars = bars

    def _get(self, prefixed: str) -> str:
        """Fetch the raw k-line JSON for one prefixed symbol."""
        import requests  # lazy — keeps the import off the offline test path

        session = requests.Session()
        session.trust_env = False       # same proxy reason as the quote providers
        response = session.get(
            _KLINE_URL,
            params={"param": f"{prefixed},day,,,{self._bars},qfq"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return response.text

    def series(self, symbols: list[tuple[str, str]],
               venues: dict[str, str] | None = None) -> dict[str, list[Bar]]:
        """Fetch (date, close, volume) bars for (symbol, market) pairs.

        `venues` maps a US symbol to the venue-suffixed code its quote carried
        (`AAPL` → `AAPL.OQ`). Without it a US name is retried across the known
        suffixes, because the bare ticker returns a two-bar stub rather than an
        error.

        A symbol whose fetch fails returns no series rather than raising. History
        is an enrichment: losing it costs two factors on one name, and taking the
        whole advisory down over a k-line request would trade a better ranking
        for no ranking at all.
        """
        if not symbols:
            return {}

        venue_of = venues or {}

        def candidates(symbol: str, market: str) -> list[str]:
            known = venue_of.get(symbol)
            if market != "US" or (known and "." in known):
                return [_prefix(symbol, market, known)]
            return [f"us{symbol.strip().upper()}{v}" for v in _US_VENUES]

        def fetch(pair: tuple[str, str]) -> tuple[str, list[Bar]]:
            symbol, market = pair
            for prefixed in candidates(symbol, market):
                try:
                    bars = _series(self._get(prefixed))
                except Exception:
                    continue
                if len(bars) >= _MIN_BARS:
                    return symbol, bars
            return symbol, []

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            results = list(pool.map(fetch, symbols))
        return {sym: bars for sym, bars in results if bars}

    def closes(self, symbols: list[tuple[str, str]],
               venues: dict[str, str] | None = None) -> dict[str, list[float]]:
        """Close series for (symbol, market) pairs, keyed by bare symbol."""
        return {
            symbol: [close for _, close, _ in bars]
            for symbol, bars in self.series(symbols, venues).items()
        }

    def enrich(self, candidates: list["AssetCandidate"]) -> list["AssetCandidate"]:
        """Return candidates with `momentum` and `volatility` filled in from history.

        `volatility` is written even though nothing in the factor path strictly
        needs it there: the portfolio optimiser reads that exact key and has been
        falling back to a 0.15 constant for every live candidate.
        """
        pairs = [(c.symbol, c.market) for c in candidates]
        venues = {
            c.symbol: str(c.metrics["venue_code"])
            for c in candidates if c.metrics.get("venue_code")
        }
        series = self.closes(pairs, venues)

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
