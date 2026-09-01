"""Backtest the five-factor weights on this repo's own candidate pool.

Why
---
`portfolio/factors.py` ships weights of 25/25/20/15/15 that were a house view,
and `ENABLE_FACTOR_SCORING` defaults to off for exactly that reason: a scoring
formula nobody has measured is worse than a legible rule, because it looks
quantitative. This script is what turns that default into a decision instead of
a hedge. It measures the factors on the 872 equity names in
`data/universe.json`, over the daily history the pipeline already fetches, and
prints what each one is worth.

Method
------
* **Universe** — the shipped equity universe (A / HK / US), scored per market,
  the same way `score_candidates` does. Cross-market z-scores would rank
  countries rather than companies.
* **Rebalance** — every 21 trading days on each market's own calendar, using a
  65-session lookback and a 21-session forward return. With ~800 bars of
  history that is roughly 33 observation dates.
* **Rank IC** — Spearman correlation between the factor value at the rebalance
  date and the forward 21-day return, computed per date, then summarised as
  mean IC, its t-statistic across dates, and the share of dates with IC > 0.
  The t-statistic is the number to read: a mean IC of 0.02 over 33 dates with a
  t of 0.4 is noise wearing a decimal point.
* **Selection test** — replays the real thing: the `_is_investable` floors, the
  60/20/20 market quota, 50 names, equally weighted. Two rankings, same
  pipeline, one number each.

What this backtest cannot see
-----------------------------
1. **Point-in-time fundamentals.** Only today's P/E, P/B and market cap are
   available; past values are reconstructed by scaling today's back along the
   price series. That is sound for size and turnover (shares outstanding move
   slowly) and *look-ahead* for value, because it assumes today's earnings and
   book value were known then. Value's IC below is therefore an upper bound and
   should be read as one.
2. **Survivorship.** The universe is today's index membership. Names that fell
   out of the index — the ones that did badly — are not in it. Every ranking
   measured here shares that tailwind, which is why the honest comparison is
   factor-vs-quality rather than either against zero.
3. **Costs.** No commission, no spread, no FX. A monthly rebalance of a 50-name
   book is not free; a strategy that wins by less than its turnover costs has
   not won.

Neither (1) nor (2) can be fixed with a free data source, so they are stated
rather than papered over. They bound how much this result is allowed to prove.

Run
---
    PYTHONPATH=src .venv/bin/python scripts/backtest_factors.py
    …--cache /tmp/ww_backtest.json   reuse the fetched bars (default)
    …--refresh                       re-fetch even if the cache is warm
    …--markdown docs/factor-backtest.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wealthwise.portfolio.factors import (      # noqa: E402
    FACTOR_WEIGHTS,
    TURNOVER_CAP,
    Z_CLIP,
)
from wealthwise.providers.history import (      # noqa: E402
    TencentHistoryProvider,
    realized_volatility,
)
from wealthwise.providers.tencent_provider import TencentMarketProvider  # noqa: E402
from wealthwise.providers.universe import Universe                       # noqa: E402

# --- windows, matched to the live path -------------------------------------
LOOKBACK = 65          # 60-day momentum window + the 5-session reversal skip
VOL_WINDOW = 60
MOMENTUM_WINDOW = 60
MOMENTUM_SKIP = 5
FORWARD = 21           # one month of trading days
STEP = 21              # rebalance cadence

# --- selection, matched to agents/experts/equity.py ------------------------
EQUITY_BUDGET = 50
MARKET_QUOTA = {"A": 0.6, "HK": 0.2, "US": 0.2}
MIN_MARKET_CAP_100M = 100.0

# Shares per unit of reported volume. Mainland bars quote 手 (100 shares); HK
# and US quote shares. Verified in `_report_turnover_calibration`, which prints
# the reconstructed turnover against the live quote field so the assumption is
# checked rather than asserted.
VOLUME_UNIT = {"A": 100.0, "HK": 1.0, "US": 1.0}

BARS = 800             # ~3.2 years, the most this endpoint returns in one call

CACHE_DEFAULT = "/tmp/ww_backtest.json"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch(cache_path: str, refresh: bool) -> dict:
    """Quotes + daily bars for the shipped equity universe, memoised on disk."""
    cache = Path(cache_path)
    if cache.exists() and not refresh:
        print(f"cache: {cache}")
        return json.loads(cache.read_text())

    universe = Universe.load()
    market_provider = TencentMarketProvider(universe)

    quotes: dict[str, dict] = {}
    t0 = time.time()
    for market in ("A", "HK", "US"):
        for candidate in market_provider.screen(market, {"asset_class": "equity"}):
            quotes[candidate.symbol] = {
                "market": candidate.market,
                "name": candidate.name,
                **{k: candidate.metrics.get(k) for k in
                   ("price", "pe", "pb", "market_cap_100m", "turnover", "venue_code")},
            }
    print(f"quotes: {len(quotes)} in {time.time() - t0:.1f}s")

    history = TencentHistoryProvider(universe, bars=BARS)
    pairs = [(s, q["market"]) for s, q in quotes.items()]
    venues = {s: q["venue_code"] for s, q in quotes.items() if q.get("venue_code")}
    t0 = time.time()
    bars = history.series(pairs, venues)
    print(f"bars:   {len(bars)} symbols in {time.time() - t0:.1f}s")

    payload = {"quotes": quotes, "bars": bars}
    cache.write_text(json.dumps(payload))
    print(f"cached: {cache}")
    return payload


# ---------------------------------------------------------------------------
# Statistics — kept in the stdlib on purpose; the project has no numpy
# ---------------------------------------------------------------------------

def _ranks(values: list[float]) -> list[float]:
    """Ascending ranks with ties averaged."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation — the standard way to score a cross-sectional signal.

    Ranks rather than levels because a factor's job is to order names, and one
    stock that trebled would otherwise carry a whole month's correlation.
    """
    return _pearson(_ranks(xs), _ranks(ys))


def t_stat(series: list[float]) -> float | None:
    """t-statistic of a mean against zero."""
    if len(series) < 3:
        return None
    sd = statistics.stdev(series)
    if sd <= 0:
        return None
    return statistics.fmean(series) / (sd / math.sqrt(len(series)))


def zscores(values: dict[str, float]) -> dict[str, float]:
    """Clipped cross-sectional z-scores — same rule as `factors._zscores`."""
    if len(values) < 2:
        return dict.fromkeys(values, 0.0)
    series = list(values.values())
    mean = statistics.fmean(series)
    sd = statistics.stdev(series)
    if sd <= 0:
        return dict.fromkeys(values, 0.0)
    return {k: max(-Z_CLIP, min(Z_CLIP, (v - mean) / sd)) for k, v in values.items()}


# ---------------------------------------------------------------------------
# Point-in-time factor reconstruction
# ---------------------------------------------------------------------------

def _annualised_vol(closes: list[float]) -> float | None:
    return realized_volatility(closes)


def factor_panel(symbol: str, quote: dict, bars: list[list],
                 idx: int) -> dict[str, float] | None:
    """Every factor for one name as of bar `idx`, or None if it cannot be scored.

    `idx` is a position in that symbol's own series, so a market holiday shifts
    nobody: each name is measured on its own most recent 65 sessions.
    """
    if idx < LOOKBACK or idx + FORWARD >= len(bars):
        return None

    closes = [b[1] for b in bars]
    price_now = quote.get("price")
    price_then = closes[idx]
    if not price_now or price_then <= 0:
        return None
    # Everything reconstructed below is today's fundamental carried back along
    # the price. Sound for size, look-ahead for value — see the module docstring.
    drift = price_then / price_now

    out: dict[str, float] = {}

    cap_now = quote.get("market_cap_100m")
    cap_then = cap_now * drift if cap_now else None
    if cap_then and cap_then > 0:
        out["size"] = math.log10(cap_then)
        out["_cap"] = cap_then

    yields: list[float] = []
    pe_now = quote.get("pe")
    if pe_now:
        pe_then = pe_now * drift
        out["_pe"] = pe_then
        if pe_then != 0:
            yields.append(1.0 / pe_then)
    pb_now = quote.get("pb")
    if pb_now and pb_now > 0:
        pb_then = pb_now * drift
        if pb_then > 0:
            yields.append(1.0 / pb_then)
    if yields:
        out["value"] = statistics.fmean(yields)

    window = closes[idx - VOL_WINDOW:idx + 1]
    vol = _annualised_vol(window)
    if vol is not None and vol > 0:
        out["low_vol"] = -vol

    end = idx - MOMENTUM_SKIP
    start = end - MOMENTUM_WINDOW
    if start >= 0 and closes[start] > 0:
        out["momentum"] = closes[end] / closes[start] - 1.0

    # Liquidity is reconstructed only where the live quote reports a turnover
    # figure at all. The bars carry volume for every market, so a turnover
    # number *could* be built for Hong Kong too — and the pipeline would never
    # have one, because `qt.gtimg.cn` returns 0 in that field for HK. Scoring a
    # factor here that the live path cannot score is how a backtest ends up
    # measuring a strategy nobody can run.
    if cap_then and cap_then > 0 and quote.get("turnover"):
        shares = cap_now * 1e8 / price_now
        unit = VOLUME_UNIT.get(quote["market"], 1.0)
        turnover = bars[idx][2] * unit / shares * 100.0
        if turnover > 0:
            out["liquidity"] = min(turnover, TURNOVER_CAP)

    out["_fwd"] = closes[idx + FORWARD] / price_then - 1.0
    return out


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

def rebalance_dates(data: dict) -> list[str]:
    """Calendar dates to rebalance on, shared by all three markets.

    One global spine rather than a calendar per market. Per-market calendars
    look more precise and are wrong for the part that matters: the three sets of
    dates barely intersect, so the "50 names under a 60/20/20 quota" test ends up
    holding 30 A-shares on an A trading day and 10 US names on a US one — three
    single-market books wearing one name, never the diversified book the
    pipeline actually builds.
    """
    bars = data["bars"]
    longest = max(bars, key=lambda s: len(bars[s]))
    calendar = [b[0] for b in bars[longest]]
    dates = [calendar[pos]
             for pos in range(len(calendar) - FORWARD - 1, LOOKBACK, -STEP)]
    return sorted(dates)


def _position(dates: list[str], target: str) -> int | None:
    """Index of the last bar dated on or before `target`, or None.

    "On or before" rather than exact: a name that did not trade on the spine
    date — suspended, or on a market with a different holiday — is scored on its
    most recent close instead of dropping out of the cross-section, which is
    what a portfolio holding it would have to do.
    """
    lo, hi = 0, len(dates)
    while lo < hi:
        mid = (lo + hi) // 2
        if dates[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1 if lo else None


def build_panels(data: dict) -> dict[str, list[dict]]:
    """One panel per market: a list of {date, rows: {symbol: factors}} snapshots."""
    quotes, bars = data["quotes"], data["bars"]
    spine = rebalance_dates(data)
    panels: dict[str, list[dict]] = {}

    for market in ("A", "HK", "US"):
        symbols = [s for s in bars if quotes.get(s, {}).get("market") == market]
        if not symbols:
            continue
        dates_of = {s: [b[0] for b in bars[s]] for s in symbols}

        snapshots: list[dict] = []
        for date in spine:
            rows: dict[str, dict] = {}
            for symbol in symbols:
                idx = _position(dates_of[symbol], date)
                if idx is None:
                    continue
                panel = factor_panel(symbol, quotes[symbol], bars[symbol], idx)
                if panel:
                    rows[symbol] = panel
            if len(rows) >= 10:
                snapshots.append({"date": date, "rows": rows})
        panels[market] = snapshots
    return panels


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def factor_ics(panels: dict[str, list[dict]]) -> dict[str, dict[str, list[float]]]:
    """Per-market, per-factor rank IC series across rebalance dates."""
    out: dict[str, dict[str, list[float]]] = {}
    for market, snapshots in panels.items():
        per_factor: dict[str, list[float]] = {f: [] for f in FACTOR_WEIGHTS}
        for snap in snapshots:
            for factor in FACTOR_WEIGHTS:
                xs, ys = [], []
                for panel in snap["rows"].values():
                    if factor in panel:
                        xs.append(panel[factor])
                        ys.append(panel["_fwd"])
                ic = spearman(xs, ys)
                if ic is not None:
                    per_factor[factor].append(ic)
        out[market] = per_factor
    return out


def composite(snap_rows: dict[str, dict], weights: dict[str, float]) -> dict[str, float]:
    """Weighted composite of clipped z-scores, renormalised over coverage.

    Deliberately the same arithmetic as `factors.score_candidates`, including
    the renormalisation: a backtest of a slightly different formula measures a
    strategy the pipeline does not run.
    """
    z_by_factor: dict[str, dict[str, float]] = {}
    for factor in weights:
        raw = {s: p[factor] for s, p in snap_rows.items() if factor in p}
        if raw:
            z_by_factor[factor] = zscores(raw)

    scores: dict[str, float] = {}
    for symbol in snap_rows:
        weighted = total = 0.0
        for factor, weight in weights.items():
            z = z_by_factor.get(factor, {}).get(symbol)
            if z is None:
                continue
            weighted += weight * z
            total += weight
        scores[symbol] = weighted / total if total > 0 else 0.0
    return scores


def composite_ic(panels: dict[str, list[dict]],
                 weights: dict[str, float]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for market, snapshots in panels.items():
        ics = []
        for snap in snapshots:
            scores = composite(snap["rows"], weights)
            xs = list(scores.values())
            ys = [snap["rows"][s]["_fwd"] for s in scores]
            ic = spearman(xs, ys)
            if ic is not None:
                ics.append(ic)
        out[market] = ics
    return out


def _investable(panel: dict) -> bool:
    """`equity._is_investable`, on reconstructed point-in-time fields."""
    pe = panel.get("_pe")
    if pe is not None and pe <= 0:
        return False
    cap = panel.get("_cap")
    return cap is None or cap >= MIN_MARKET_CAP_100M


def selection_returns(panels: dict[str, list[dict]],
                      weights: dict[str, float] | None) -> list[float]:
    """Forward return of the book the pipeline would have held, per date.

    `weights=None` runs the default quality rule (largest first, cheaper P/E
    breaks ties); a weight dict runs the factor composite. Everything else —
    floors, quota, 50 names, equal weighting — is held identical, so the
    difference between the two series is the ranking and nothing else.
    """
    dates = sorted({snap["date"] for snaps in panels.values() for snap in snaps})
    by_market_date = {
        market: {snap["date"]: snap["rows"] for snap in snaps}
        for market, snaps in panels.items()
    }

    out: list[float] = []
    for date in dates:
        picked: list[float] = []
        for market, quota in MARKET_QUOTA.items():
            rows = by_market_date.get(market, {}).get(date)
            if not rows:
                continue
            eligible = {s: p for s, p in rows.items() if _investable(p)}
            if not eligible:
                continue
            budget = max(1, round(EQUITY_BUDGET * quota))
            if weights is None:
                ranked = sorted(
                    eligible,
                    key=lambda s: (-(eligible[s].get("_cap") or 0.0),
                                   eligible[s].get("_pe") or float("inf"), s),
                )
            else:
                scores = composite(eligible, weights)
                ranked = sorted(
                    eligible,
                    key=lambda s: (-scores.get(s, 0.0),
                                   -(eligible[s].get("_cap") or 0.0),
                                   eligible[s].get("_pe") or float("inf"), s),
                )
            picked += [eligible[s]["_fwd"] for s in ranked[:budget]]
        if picked:
            out.append(statistics.fmean(picked))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value, digits=3, dash="—"):
    return dash if value is None else f"{value:+.{digits}f}"


def _summary_row(name: str, ics: list[float]) -> list[str]:
    if not ics:
        return [name, "—", "—", "—", "0"]
    return [
        name,
        _fmt(statistics.fmean(ics)),
        _fmt(t_stat(ics), 2),
        f"{sum(1 for i in ics if i > 0) / len(ics):.0%}",
        str(len(ics)),
    ]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join(
        "  ".join(r[i].ljust(widths[i]) for i in range(len(headers))) for r in rows
    )
    return f"{line}\n{sep}\n{body}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def report_turnover_calibration(data: dict) -> list[list[str]]:
    """Check the 手-vs-shares volume assumption against the live turnover field."""
    rows = []
    for market in ("A", "HK", "US"):
        ratios = []
        for symbol, bars in data["bars"].items():
            quote = data["quotes"].get(symbol, {})
            if quote.get("market") != market:
                continue
            live = quote.get("turnover")
            cap, price = quote.get("market_cap_100m"), quote.get("price")
            if not (live and cap and price) or not bars:
                continue
            shares = cap * 1e8 / price
            rebuilt = bars[-1][2] * VOLUME_UNIT[market] / shares * 100.0
            if rebuilt > 0:
                ratios.append(rebuilt / live)
        if ratios:
            rows.append([market, f"{VOLUME_UNIT[market]:.0f}",
                         f"{statistics.median(ratios):.2f}", str(len(ratios))])
        else:
            # HK lands here: the quote feed returns 0 in the turnover field, so
            # there is nothing to calibrate against and no liquidity factor.
            rows.append([market, f"{VOLUME_UNIT[market]:.0f}",
                         "no turnover reported", "0"])
    return rows


#: Comparison set. Every alternative is a *mechanical* transform of the shipped
#: weights — equal weighting, or dropping one factor and rescaling the rest in
#: their existing proportions. None of them is fitted to the numbers below.
#: Searching the weight space on 34 dates of one universe would return whichever
#: mix best explains this sample, and nothing about the next one.
ALTERNATIVE_WEIGHTS: dict[str, dict[str, float]] = {
    "house 25/25/20/15/15": FACTOR_WEIGHTS,
    "equal 20×5": dict.fromkeys(FACTOR_WEIGHTS, 0.2),
}


def _without(factor: str) -> dict[str, float]:
    """House weights minus one factor, the rest kept in proportion."""
    kept = {k: v for k, v in FACTOR_WEIGHTS.items() if k != factor}
    total = sum(kept.values())
    return {k: v / total for k, v in kept.items()}


for _dropped in FACTOR_WEIGHTS:
    ALTERNATIVE_WEIGHTS[f"house minus {_dropped}"] = _without(_dropped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=CACHE_DEFAULT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()

    data = fetch(args.cache, args.refresh)
    panels = build_panels(data)

    sections: list[tuple[str, str, str]] = []   # (title, text table, md table)

    def emit(title: str, headers: list[str], rows: list[list[str]]) -> None:
        sections.append((title, _table(headers, rows), _md_table(headers, rows)))

    dates = {m: len(s) for m, s in panels.items()}
    span = {m: (s[0]["date"], s[-1]["date"]) for m, s in panels.items() if s}
    coverage = [[m, str(dates[m]), span[m][0], span[m][1],
                 str(len(panels[m][-1]["rows"]))] for m in panels if panels[m]]
    emit("Coverage", ["market", "dates", "from", "to", "names (last date)"], coverage)

    emit("Volume-unit calibration (rebuilt ÷ live turnover, median)",
         ["market", "shares per unit", "ratio", "names"],
         report_turnover_calibration(data))

    ics = factor_ics(panels)
    for market in ("A", "HK", "US"):
        if market not in ics:
            continue
        rows = [_summary_row(f, ics[market][f]) for f in FACTOR_WEIGHTS]
        emit(f"Factor rank IC — {market}",
             ["factor", "mean IC", "t", "IC>0", "dates"], rows)

    rows = []
    for label, weights in ALTERNATIVE_WEIGHTS.items():
        per_market = composite_ic(panels, weights)
        pooled = [ic for market in per_market for ic in per_market[market]]
        rows.append(_summary_row(label, pooled))
    emit("Composite rank IC (all markets pooled)",
         ["weights", "mean IC", "t", "IC>0", "dates"], rows)

    quality = selection_returns(panels, None)
    rows = [[
        "quality rule (default)",
        _fmt(statistics.fmean(quality), 4) if quality else "—",
        "—", "—", "—", str(len(quality)),
    ]]
    for label, weights in ALTERNATIVE_WEIGHTS.items():
        picked = selection_returns(panels, weights)
        if not picked or len(picked) != len(quality):
            continue
        diff = [f - q for f, q in zip(picked, quality)]
        rows.append([
            label,
            _fmt(statistics.fmean(picked), 4),
            _fmt(statistics.fmean(diff), 4),
            _fmt(t_stat(diff), 2),
            f"{sum(1 for d in diff if d > 0) / len(diff):.0%}",
            str(len(picked)),
        ])
    emit("Selected book — mean 21-day forward return, equal-weighted 50 names",
         ["ranking", "mean fwd", "vs quality", "t", "win rate", "dates"], rows)

    for title, text, _ in sections:
        print(f"\n## {title}\n")
        print(text)

    if args.markdown:
        out = Path(args.markdown)
        body = [f"# Factor backtest — {data['quotes'].__len__()} names, "
                f"{BARS} daily bars\n"]
        body.append("Generated by `scripts/backtest_factors.py`. Method, and what "
                    "it cannot see, are documented in that file's docstring.\n")
        for title, _, md in sections:
            body.append(f"\n## {title}\n\n{md}\n")
        out.write_text("\n".join(body), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
