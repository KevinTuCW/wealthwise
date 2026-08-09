"""AkShare-backed providers for live A/HK/US market data, macro, and FX.

Design:
- All akshare calls are isolated inside `_get()` so tests can monkeypatch
  that seam without importing akshare.
- akshare is lazy-imported inside `_get()` only — never at module top level.
- `_get()` returns raw list-of-dict / dict payloads (AkShare DataFrames turned
  into records); the public methods map those into AssetCandidate / dict / float.
- `build_provider()` is config-gated: returns Sample providers when
  settings.use_real_providers is False.

AkShare is not installed in the offline dev/test environment, so the exact
DataFrame column names below cannot be verified live. Every column→field
mapping is marked `# TODO(live-calibration): verify akshare column names
against live output` — same pattern shopscout used for its junglescout/spapi
parse layers. Tests bypass the network entirely by monkeypatching `_get`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from wealthwise.agents.state import AssetCandidate

if TYPE_CHECKING:
    from wealthwise.config import Settings


# ---------------------------------------------------------------------------
# Market provider
# ---------------------------------------------------------------------------

# Which AkShare spot function backs each market's full snapshot.
# TODO(live-calibration): verify these function names exist in the installed
# akshare version (they are renamed occasionally across releases).
_MARKET_SPOT_FN = {
    "A": "stock_zh_a_spot_em",
    "HK": "stock_hk_spot_em",
    "US": "stock_us_spot_em",
}


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


def _map_equity_row(row: dict, market: str) -> dict:
    """Map one AkShare spot-quote row to AssetCandidate kwargs.

    AkShare A-share columns are Chinese ("代码"/"名称"/"市盈率-动态"…); HK/US
    columns differ again. Each get() falls back through the Chinese label, an
    English alias, and finally the already-domain-shaped key, so the same
    mapper survives minor akshare schema drift and pre-shaped fixture rows.
    """
    # TODO(live-calibration): verify akshare column names against live output
    symbol = str(row.get("代码") or row.get("symbol") or "")
    name = str(row.get("名称") or row.get("name") or "")
    # TODO(live-calibration): verify akshare column names against live output
    pe = row.get("市盈率-动态") or row.get("市盈率") or row.get("pe")
    # TODO(live-calibration): verify akshare column names against live output
    #   AkShare spot tables carry a daily change pct ("涨跌幅"), not annualized
    #   vol; real annualized volatility needs a separate history fetch, so it is
    #   left absent here rather than fabricated.
    metrics = dict(row.get("metrics") or {})   # honour a pre-shaped metrics bag
    if pe is not None and "pe" not in metrics:
        metrics["pe"] = float(pe)

    currency = row.get("currency") or {"A": "CNY", "HK": "HKD", "US": "USD"}[market]
    return {
        "symbol": symbol,
        "market": row.get("market") or market,
        "asset_class": row.get("asset_class") or "equity",
        "name": name,
        "currency": currency,
        # TODO(live-calibration): AkShare spot tables have no suitability rating;
        # r_level must be derived downstream (e.g. from volatility). Default R3
        # is a neutral placeholder until the risk-scoring task fills it.
        "r_level": row.get("r_level") or "R3",
        "metrics": metrics,
        "tags": list(row.get("tags") or []),
    }


class AkShareMarketProvider:
    """Live market data via AkShare (lazy-imported).

    _get() is the only method that touches akshare; all public methods go
    through _get() so tests can stub it without importing akshare.
    """

    def _get(self, symbols: list[str] | None = None,
             market: str | None = None) -> list[dict]:
        """Fetch spot-snapshot rows from AkShare, as record dicts.

        When `market` is one of A/HK/US, dispatches to that single spot
        function. When `market` is None (the quotes path), fetches all three
        snapshots and tags each row with its market so the mapper can tell them
        apart. Tests monkeypatch this method, so no akshare import happens then.
        """
        import akshare as ak  # lazy — never imported at module top

        targets = [market] if market in _MARKET_SPOT_FN else list(_MARKET_SPOT_FN)
        out: list[dict] = []
        for mkt in targets:
            fn = getattr(ak, _MARKET_SPOT_FN[mkt], None)
            if fn is None:
                continue
            for row in _to_records(fn()):
                row.setdefault("market", mkt)   # tag origin market for the mapper
                out.append(row)
        return out

    def quotes(self, symbols: list[str]) -> list[AssetCandidate]:
        """Fetch AssetCandidates for the given symbols across A/HK/US snapshots."""
        wanted = {s.casefold() for s in symbols}
        out: list[AssetCandidate] = []
        for row in self._get(symbols=symbols):
            mapped = _map_equity_row(row, row.get("market", "A"))
            if mapped["symbol"].casefold() in wanted:
                out.append(AssetCandidate(**mapped))
        return out

    def screen(self, market: str, filters: dict) -> list[AssetCandidate]:
        """Return candidates in `market` matching simple filters.

        Supported filters: asset_class (equity only for spot tables), max_pe.
        """
        out: list[AssetCandidate] = []
        for row in self._get(market=market):
            mapped = _map_equity_row(row, market)
            if filters.get("asset_class") and mapped["asset_class"] != filters["asset_class"]:
                continue
            pe = mapped["metrics"].get("pe")
            if "max_pe" in filters and pe is not None and pe > filters["max_pe"]:
                continue
            out.append(AssetCandidate(**mapped))
        return out


# ---------------------------------------------------------------------------
# Fund provider (bond / money-market / QDII)
# ---------------------------------------------------------------------------

def _map_fund_row(row: dict) -> dict:
    """Map one AkShare open-fund row to AssetCandidate kwargs."""
    # TODO(live-calibration): verify akshare column names against live output
    symbol = str(row.get("基金代码") or row.get("代码") or row.get("symbol") or "")
    name = str(row.get("基金简称") or row.get("名称") or row.get("name") or "")
    return {
        "symbol": symbol,
        "market": "A",
        # TODO(live-calibration): AkShare fund lists don't expose a clean
        # asset_class; classify from name/type downstream. Default "bond".
        "asset_class": "bond",
        "name": name,
        "currency": "CNY",
        "r_level": "R2",
        "metrics": {},
        "tags": [],
    }


class AkShareFundProvider:
    """Live open-end fund data via AkShare (lazy-imported)."""

    def _get(self) -> list[dict]:
        """Fetch the open-fund catalogue from AkShare, as record dicts."""
        import akshare as ak  # lazy
        # TODO(live-calibration): verify akshare column names against live output
        #   fund_open_fund_daily_em() is the daily NAV table for open funds;
        #   fund_name_em() is the fuller name/type catalogue. Pick per need.
        fn = getattr(ak, "fund_open_fund_daily_em", None) or getattr(ak, "fund_name_em", None)
        if fn is None:
            return []
        return _to_records(fn())

    def quotes(self, symbols: list[str]) -> list[AssetCandidate]:
        wanted = {s.casefold() for s in symbols}
        out: list[AssetCandidate] = []
        for row in self._get():
            mapped = _map_fund_row(row)
            if mapped["symbol"].casefold() in wanted:
                out.append(AssetCandidate(**mapped))
        return out

    def screen(self, market: str, filters: dict) -> list[AssetCandidate]:
        out: list[AssetCandidate] = []
        for row in self._get():
            mapped = _map_fund_row(row)
            if market and mapped["market"] != market:
                continue
            if filters.get("asset_class") and mapped["asset_class"] != filters["asset_class"]:
                continue
            out.append(AssetCandidate(**mapped))
        return out


# ---------------------------------------------------------------------------
# Macro provider
# ---------------------------------------------------------------------------

class AkShareMacroProvider:
    """Live macro data via AkShare (lazy-imported)."""

    def _get(self) -> dict:
        """Fetch a macro snapshot (rate + CPI) from AkShare, as a flat dict."""
        import akshare as ak  # lazy

        snapshot: dict = {}

        # --- benchmark interest rate (LPR) ---
        # TODO(live-calibration): verify akshare column names against live output
        rate_fn = getattr(ak, "macro_china_lpr", None)
        if rate_fn is not None:
            rate_rows = _to_records(rate_fn())
            if rate_rows:
                last = rate_rows[-1]
                # LPR published in percent (e.g. 3.45) → store as decimal fraction
                lpr = last.get("LPR1Y") or last.get("1年") or last.get("lpr_1y")
                if lpr is not None:
                    snapshot["interest_rate"] = float(lpr) / 100.0

        # --- CPI (YoY) ---
        # TODO(live-calibration): verify akshare column names against live output
        cpi_fn = getattr(ak, "macro_china_cpi_yearly", None) or getattr(ak, "macro_china_cpi", None)
        if cpi_fn is not None:
            cpi_rows = _to_records(cpi_fn())
            if cpi_rows:
                last = cpi_rows[-1]
                cpi = last.get("今值") or last.get("value") or last.get("cpi")
                if cpi is not None:
                    # published in percent (e.g. 2.1) → decimal fraction
                    snapshot["cpi"] = float(cpi) / 100.0

        return snapshot

    def snapshot(self) -> dict:
        return self._get()


# ---------------------------------------------------------------------------
# FX provider
# ---------------------------------------------------------------------------

# AkShare currency_boc_sina uses Chinese currency names; map our pair codes.
# TODO(live-calibration): verify akshare column names against live output
_FX_PAIR_TO_BOC = {
    "USDCNH": "美元",
    "HKDCNH": "港币",
}


class AkShareFXProvider:
    """Live FX rates via AkShare (lazy-imported)."""

    def _get(self, pair: str) -> list[dict]:
        """Fetch the BOC spot history for the currency behind `pair`."""
        import akshare as ak  # lazy

        symbol = _FX_PAIR_TO_BOC.get(pair.upper())
        if symbol is None:
            return []
        # TODO(live-calibration): verify akshare column names against live output
        #   currency_boc_sina(symbol=…) returns a dated middle-rate history;
        #   fx_spot_quote() is the alternative interbank quote table.
        fn = getattr(ak, "currency_boc_sina", None)
        if fn is None:
            return []
        return _to_records(fn(symbol=symbol))

    def rate(self, pair: str) -> float:
        rows = self._get(pair)
        if not rows:
            raise KeyError(f"FX pair {pair!r} not available from akshare")
        last = rows[-1]
        # TODO(live-calibration): verify akshare column names against live output
        #   BOC quotes middle rate per 100 units of foreign currency ("中行折算价"),
        #   so divide by 100 to get CNY-per-unit.
        mid = last.get("中行折算价") or last.get("中行汇买价") or last.get("value")
        if mid is None:
            raise KeyError(f"no rate column in akshare payload for {pair!r}")
        return float(mid) / 100.0


# ---------------------------------------------------------------------------
# Config-gated factory
# ---------------------------------------------------------------------------

def build_provider(settings: "Settings") -> tuple[
    "AkShareMarketProvider | object",
    "AkShareMacroProvider | object",
    "AkShareFXProvider | object",
]:
    """Return (market, macro, fx) providers gated on settings.use_real_providers.

    When use_real_providers is False (the default), returns offline Sample
    providers so the pipeline works without any API keys or akshare install.
    """
    if settings.use_real_providers:
        return (
            AkShareMarketProvider(),
            AkShareMacroProvider(),
            AkShareFXProvider(),
        )

    from wealthwise.providers.sample import (
        SampleFXProvider,
        SampleMacroProvider,
        SampleMarketProvider,
    )
    data_dir = settings.sample_data_dir
    return (
        SampleMarketProvider(data_dir),
        SampleMacroProvider(data_dir),
        SampleFXProvider(data_dir),
    )
