"""AkShare-backed macro and FX sources — column mappings verified live.

Scope
-----
Macro (benchmark rate, CPI) and FX only. Equity quotes moved to
`tencent_provider.py`: the eastmoney spot endpoints AkShare wraps
(`stock_zh_a_spot_em` and its HK/US siblings) resolve and complete a TLS
handshake, then reset mid-stream, and a screener that fails on two calls in
three inside `equity_node` takes the whole advisory down with it. The AkShare
equity and fund providers that used to live here were deleted rather than kept
as a fallback — an uncallable path cannot be calibrated, and an uncalibrated
fallback is a liability, not a safety net.

Calibration
-----------
Every mapping below was checked against akshare 1.18.83 live output on
2026-09-01 (recorded in `docs/real-data-verification.md`), which is why no
`TODO(live-calibration)` markers remain. Three of them were wrong:

* `macro_china_cpi` is published **newest-first**, so reading the last row
  returned the CPI print for January 2008.
* `macro_china_cpi_yearly` ends with the *scheduled* release, whose 今值 is
  NaN until the number is out — the snapshot published a NaN CPI on any day
  between two prints.
* `currency_boc_sina` has a frozen default date range baked into the signature
  (`start_date='20230304', end_date='20231110'`), so calling it without dates
  quoted a 2023 rate as today's.

All three fail silently and produce a plausible-looking number, which is the
argument for calibrating against live output rather than reasoning about the
docs. `_newest()` and the freshness guard exist to make them fail loudly if any
of the three publishers changes its ordering again.

Design:
- All akshare calls are isolated inside `_get()` / `_table()` so tests can
  monkeypatch that seam without importing akshare.
- akshare is lazy-imported inside those seams only — never at module top level.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Shared parsing helpers
# ---------------------------------------------------------------------------


def _to_records(df) -> list[dict]:
    """Turn an AkShare DataFrame into a list of plain dict rows.

    AkShare returns pandas DataFrames; `.to_dict(orient="records")` is the
    stable way to get row dicts without importing pandas here.
    """
    if df is None:
        return []
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records")
    return list(df)  # already record-like (e.g. a test stub)


def _number(value: object) -> float | None:
    """Parse a published cell as a float, or None if it carries no reading.

    NaN counts as no reading. Pandas hands one back for any cell the publisher
    has not filled in yet, and NaN survives arithmetic silently, so a missing
    print would otherwise reach the consensus layer as a number and poison the
    median it lands in.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _as_date(value: object) -> date | None:
    """Parse the several date shapes AkShare macro tables use, or None.

    Seen live: `datetime.date` (LPR, CPI-yearly), `pandas.Timestamp`, and the
    NBS table's `"2026年07月份"` string.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "")
    match = re.search(r"(\d{4})\D+(\d{1,2})(?:\D+(\d{1,2}))?", text)
    if not match:
        return None
    year, month, day = match.group(1), match.group(2), match.group(3)
    try:
        return date(int(year), int(month), int(day or 1))
    except ValueError:
        return None


def _newest(rows: list[dict], date_key: str, value_keys: tuple[str, ...]) -> dict | None:
    """The most recent row that actually carries one of `value_keys`.

    Two failure modes, one helper. Publishers disagree on ordering — LPR ships
    oldest-first, the NBS CPI table newest-first — so `rows[-1]` is a coin flip
    on which end of the history it lands. And a table that lists its next
    scheduled release ends with a row whose value is NaN, so "most recent row"
    and "most recent reading" are not the same row. Picking by date, among rows
    that have a value, is right under both.

    Rows with no parseable date sort last but stay eligible: losing the reading
    entirely because a publisher renamed its date column would be a worse
    failure than reading it in file order.
    """
    usable = [
        row for row in rows
        if any(_number(row.get(k)) is not None for k in value_keys)
    ]
    if not usable:
        return None
    dated = [(d, row) for row in usable if (d := _as_date(row.get(date_key)))]
    if not dated:
        return usable[-1]
    return max(dated, key=lambda pair: pair[0])[1]


# ---------------------------------------------------------------------------
# Macro sources
# ---------------------------------------------------------------------------


# How old a macro print may be before the source declines to publish it.
#
# Both series here are monthly, and the NBS table is labelled by the month it
# describes rather than the day it was released, so the label already lags
# publication by around six weeks. A hundred days leaves roughly two cycles of
# headroom for a late release, and still catches a feed that has stopped: the
# jin10-backed `macro_china_cpi_yearly` last printed in September 2025 and
# hands that reading back as though it were current.
#
# This is the failure mode a consensus layer is least able to catch by itself.
# Two publishers a year apart do not look like a disagreement — a 0.0% reading
# from August 2025 and a 0.5% reading from July 2026 are both plausible CPI
# prints, so the resolver reconciles them into a narrow spread and reports high
# confidence in a number that describes no month at all.
_MACRO_MAX_STALENESS_DAYS = 100


class _AkShareMacroSource:
    """One AkShare macro endpoint, published as a partial snapshot.

    Macro consensus needs *sources*, not one provider that happens to call
    several endpoints internally: a snapshot assembled behind a single interface
    cannot be cross-checked, because by the time it is returned the individual
    publishers have already been collapsed into one dict. Splitting endpoint by
    endpoint is what lets `ConsensusMacroProvider` see two independent CPI
    readings and report the spread between them.

    Subclasses declare which table they read and which column carries the
    number; `_table()` is the only method that touches akshare, so tests can
    stub it without importing akshare.
    """

    name = "akshare"
    #: akshare callables tried in order; the first one present is used.
    functions: tuple[str, ...] = ()
    #: Snapshot key this source fills in.
    field: str = ""
    #: Column carrying the print date, used for both ordering and freshness.
    date_key: str = ""
    #: Columns carrying the reading, most preferred first.
    value_keys: tuple[str, ...] = ()

    def _table(self) -> list[dict]:
        """Return the endpoint's rows, or [] when the function is unavailable."""
        import akshare as ak  # lazy — never imported at module top

        for fn_name in self.functions:
            fn = getattr(ak, fn_name, None)
            if fn is not None:
                return _to_records(fn())
        return []

    def _convert(self, raw: float) -> float | None:
        """Turn the published cell into a decimal fraction."""
        return raw / 100.0

    def _get(self) -> dict:
        row = _newest(self._table(), self.date_key, self.value_keys)
        if row is None:
            return {}

        as_of = _as_date(row.get(self.date_key))
        if as_of and (date.today() - as_of).days > _MACRO_MAX_STALENESS_DAYS:
            return {}          # a stopped feed reports nothing, not its last word

        raw = next(
            (v for k in self.value_keys if (v := _number(row.get(k))) is not None),
            None,
        )
        if raw is None:
            return {}
        value = self._convert(raw)
        return {self.field: value} if value is not None else {}

    def snapshot(self) -> dict:
        return self._get()


class AkShareLprSource(_AkShareMacroSource):
    """Benchmark lending rate — the 1-year LPR.

    Live 2026-09-01: columns `TRADE_DATE / LPR1Y / LPR5Y / RATE_1 / RATE_2`,
    1575 rows oldest-first, latest print 2026-08-20 at LPR1Y = 3.00.
    """

    name = "akshare-lpr"
    functions = ("macro_china_lpr",)
    field = "interest_rate"
    date_key = "TRADE_DATE"
    value_keys = ("LPR1Y", "1年", "lpr_1y")


class AkShareCpiYearlySource(_AkShareMacroSource):
    """CPI year-on-year as republished by the market-data aggregator table.

    Live 2026-09-01: columns `商品 / 日期 / 今值 / 预测值 / 前值`, oldest-first,
    and the final row is the *next scheduled* release with 今值 = NaN — so the
    last row and the last reading are not the same row.

    That table is also **not currently publishing**: its last actual print is
    2025-08-09, a year behind the statistics bureau, and every other jin10-backed
    series in this akshare version stops within days of the same date. The
    mapping below is correct and this source still returns nothing today — that
    is the freshness guard working, not a bug. CPI therefore reconciles from one
    publisher and is reported at confidence 0.5: the honest reading, and visible
    in the consensus record rather than hidden inside a false agreement.
    """

    name = "akshare-cpi-yearly"
    functions = ("macro_china_cpi_yearly",)
    field = "cpi"
    date_key = "日期"
    value_keys = ("今值", "value", "cpi")


class AkShareCpiNbsSource(_AkShareMacroSource):
    """CPI year-on-year straight from the statistics-bureau index table.

    Deliberately a different publisher from `AkShareCpiYearlySource` rather than
    a different column of the same one. Two readings drawn from a single table
    would agree by construction, and a consensus that cannot disagree is
    decoration.

    Live 2026-09-01: 223 rows **newest-first** (`月份` = "2026年07月份" at row 0,
    "2008年01月份" at the end), `全国-同比增长` = 0.5 for July 2026. Reading the
    last row — which is what this source used to do — published the January 2008
    print, 0.5% and 7.1% being equally plausible-looking numbers.
    """

    name = "akshare-cpi-nbs"
    functions = ("macro_china_cpi",)
    field = "cpi"
    date_key = "月份"
    value_keys = ("全国-同比增长", "同比增长", "全国-当月")

    def _convert(self, raw: float) -> float | None:
        """Convert, allowing for the two forms this table publishes.

        `全国-同比增长` quotes the change itself (0.5 → +0.5%); the fallback
        column `全国-当月` quotes the index on "last year = 100" (100.5 → +0.5%).
        As fractions the two forms are two orders of magnitude apart, so which
        one was read is recoverable from the value.
        """
        cpi = raw / 100.0
        return cpi - 1.0 if cpi > 0.5 else cpi


class AkShareMacroProvider:
    """Live macro data via AkShare — every endpoint merged into one snapshot.

    Kept for callers that want a single macro provider rather than a consensus
    of sources. `build_macro_sources()` is the entry point the consensus layer
    uses.
    """

    name = "akshare"

    def __init__(self, sources: list[_AkShareMacroSource] | None = None) -> None:
        self._sources = sources if sources is not None else build_macro_sources()

    def _get(self) -> dict:
        snapshot: dict = {}
        for source in self._sources:
            snapshot.update(source.snapshot())
        return snapshot

    def snapshot(self) -> dict:
        return self._get()


def build_macro_sources() -> list[_AkShareMacroSource]:
    """The macro publishers, in the order their readings should be labelled.

    CPI has two independent publishers and is genuinely corroborated. The
    benchmark rate has one, and the resolver caps it at confidence 0.5 — the
    right answer, and the reason no second rate endpoint was invented for it:
    Shibor and LPR measure different things, and medianing them would produce a
    number that no publisher reports and no borrower pays.
    """
    return [AkShareLprSource(), AkShareCpiYearlySource(), AkShareCpiNbsSource()]


# ---------------------------------------------------------------------------
# FX provider
# ---------------------------------------------------------------------------

# AkShare currency_boc_sina uses Chinese currency names; map our pair codes.
# Verified live 2026-09-01: 美元 → 中行折算价 678.09, 港币 → 91.93 (per 100 units).
_FX_PAIR_TO_BOC = {
    "USDCNH": "美元",
    "HKDCNH": "港币",
}

# How far back to ask for. The table only carries business days, and a long
# weekend plus a holiday clears a week, so a fortnight is the smallest window
# that reliably contains a print.
_FX_LOOKBACK_DAYS = 21

# A rate older than this is refused rather than returned. The endpoint's default
# date range is hard-coded in its signature and ends in November 2023, so the
# original no-arguments call did not fail — it quoted a two-and-a-half-year-old
# rate with a straight face. A stale FX rate is worse than a missing one:
# missing is visible, stale is not.
_FX_MAX_STALENESS_DAYS = 10


class AkShareFXProvider:
    """Live BOC middle rates via AkShare (lazy-imported).

    Columns verified live 2026-09-01: `日期 / 中行汇买价 / 中行钞买价 /
    中行钞卖价/汇卖价 / 央行中间价 / 中行折算价`, quoted per **100** units of
    foreign currency, oldest-first.
    """

    name = "akshare-boc-fx"

    def _get(self, pair: str) -> list[dict]:
        """Fetch a recent BOC spot window for the currency behind `pair`."""
        import akshare as ak  # lazy

        symbol = _FX_PAIR_TO_BOC.get(pair.upper())
        if symbol is None:
            return []
        fn = getattr(ak, "currency_boc_sina", None)
        if fn is None:
            return []
        today = date.today()
        start = today - timedelta(days=_FX_LOOKBACK_DAYS)
        # The date arguments are not optional in practice: their defaults are
        # frozen constants in the signature, not "latest".
        return _to_records(fn(
            symbol=symbol,
            start_date=start.strftime("%Y%m%d"),
            end_date=today.strftime("%Y%m%d"),
        ))

    def rate(self, pair: str) -> float:
        rows = self._get(pair)
        last = _newest(rows, "日期", ("中行折算价", "中行汇买价", "value"))
        if last is None:
            raise KeyError(f"FX pair {pair!r} not available from akshare")

        as_of = _as_date(last.get("日期"))
        if as_of and (date.today() - as_of).days > _FX_MAX_STALENESS_DAYS:
            raise KeyError(f"stale FX quote for {pair!r}: last print {as_of}")

        mid = next(
            (v for k in ("中行折算价", "中行汇买价", "value")
             if (v := _number(last.get(k))) is not None),
            None,
        )
        if mid is None:
            raise KeyError(f"no rate column in akshare payload for {pair!r}")
        return mid / 100.0            # quoted per 100 units of foreign currency
