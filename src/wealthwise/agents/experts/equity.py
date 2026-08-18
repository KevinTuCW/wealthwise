"""equity_node — screen equity candidates using goal_constraints + macro_view.

Screens across A / HK / US markets using goal_constraints (risk_ceiling,
accept_cross_border) and macro_view tilt to filter and annotate candidates.
No LLM call needed here — purely deterministic rule-based screening.
"""
from __future__ import annotations

import time

from wealthwise.agents.state import AdvisoryState, AssetCandidate
from wealthwise.portfolio.metrics import R_ORDER

# ---------------------------------------------------------------------------
# Markets to consider and their PE-filter heuristics
# ---------------------------------------------------------------------------

_ALL_MARKETS = ["A", "HK", "US"]

# When macro tilt is 'underweight' equity, apply a tighter PE cap to screen
# out expensive names; neutral/overweight use a generous cap (no hard filter).
_PE_CAP_UNDERWEIGHT = 35.0
_PE_CAP_DEFAULT = 9999.0   # effectively no PE filter

# Conservative mode (C1/C2 profiles): tighter PE cap regardless of macro tilt.
# This ensures the planner hint genuinely affects the candidate set.
_PE_CAP_CONSERVATIVE = 25.0

# Conservative mode: cap the number of equity candidates to keep the set small
# and weighted toward lower-risk names.
_MAX_CANDIDATES_CONSERVATIVE = 10

# How many equity names reach the optimiser. Matches the process guardrail's own
# ceiling: the guardrail is a safety net, and a net that routinely fires is doing
# the selecting. Choosing here, on the merits, is this expert's actual job.
_EQUITY_BUDGET = 50

# Geographic split of that budget. Screening ran market by market and the results
# were concatenated, so the process guardrail's `list[:50]` handed every slot to
# whichever market came first — A-shares, all 300 of them. An investor who had
# explicitly accepted cross-border exposure ended up with 0% of it, and nothing
# in the pipeline was even deciding that; it fell out of list order.
#
# These weights are a house view on geographic diversification, not an
# optimisation: an A-share core with meaningful HK and US sleeves. Unfilled quota
# is redistributed, so declining cross-border simply gives A-shares the lot.
_MARKET_QUOTA = {"A": 0.6, "HK": 0.2, "US": 0.2}

# Ranking needs a size floor to be meaningful; below it, market cap is mostly
# noise and liquidity gets thin. 100e8 = 10 billion in the local currency.
_MIN_MARKET_CAP_100M = 100.0


def _market_cap_100m(candidate: AssetCandidate) -> float | None:
    """Market cap in units of 100M local currency, or None if unreported.

    Providers disagree on how to say this: the quote-backed one reports
    `market_cap_100m` already in 亿, the sample one reports `market_cap_<ccy>` in
    raw units. Reading only one spelling silently disqualified every candidate
    from the other provider, which is a screening rule failing on a naming
    difference rather than on anything about the companies.
    """
    direct = candidate.metrics.get("market_cap_100m")
    if direct is not None:
        return float(direct)
    for key in ("market_cap_cny", "market_cap_hkd", "market_cap_usd", "market_cap"):
        raw = candidate.metrics.get(key)
        if raw is not None:
            return float(raw) / 1e8
    return None


def _is_investable(candidate: AssetCandidate) -> bool:
    """Quality floor: reject on evidence of unsuitability, not on missing data.

    A reported P/E at or below zero means the company is loss-making, and a
    reported market cap under the floor means it is too small to carry the
    liquidity an advisory portfolio assumes. Both are disqualifying.

    An *absent* metric is not. Provider coverage varies, and treating silence as
    a negative verdict would let a thinner data source empty the candidate set
    entirely — the pipeline would then look like it had screened carefully when
    it had merely failed to read anything.
    """
    pe = candidate.metrics.get("pe")
    if pe is not None and pe <= 0:
        return False
    cap = _market_cap_100m(candidate)
    return cap is None or cap >= _MIN_MARKET_CAP_100M


def _quality_key(candidate: AssetCandidate) -> tuple:
    """Sort key: larger companies first, ties broken by cheaper valuation.

    Size stands in for liquidity and stability, which is what suitability
    actually cares about — not for expected return. There is deliberately no
    attempt at a multi-factor score: the fields available here (P/E, P/B, market
    cap) cannot support one, and a scoring formula that looks quantitative
    without being validated is worse than an honest, legible rule.

    Names with no reported size sort last rather than being dropped, so a
    thin-data provider degrades to "quota respected, order arbitrary" instead of
    to an empty book.
    """
    cap = _market_cap_100m(candidate)
    pe = candidate.metrics.get("pe")
    return (-(cap if cap is not None else 0.0),
            pe if pe is not None else float("inf"),
            candidate.symbol)


def _allocate_quota(markets: list[str], available: dict[str, int],
                    budget: int) -> dict[str, int]:
    """Split `budget` across `markets` by house weights, redistributing shortfalls.

    A market that cannot fill its share (few candidates, or excluded outright)
    releases the remainder to the others rather than shrinking the portfolio.
    """
    weights = {m: _MARKET_QUOTA.get(m, 0.0) for m in markets}
    total = sum(weights.values())
    if total <= 0:
        return {m: 0 for m in markets}

    quota = {m: min(available.get(m, 0), int(budget * w / total)) for m, w in weights.items()}
    # Hand any unused slots to markets that still have candidates, in house-weight
    # order, so the budget is actually spent.
    leftover = budget - sum(quota.values())
    for market in sorted(markets, key=lambda m: -weights[m]):
        if leftover <= 0:
            break
        room = available.get(market, 0) - quota[market]
        take = min(room, leftover)
        quota[market] += take
        leftover -= take
    return quota


def _select(candidates: list[AssetCandidate], markets: list[str],
            budget: int) -> list[AssetCandidate]:
    """Pick the best `budget` names, respecting the geographic split.

    Candidates from a market that was never screened are passed through untouched
    rather than quota-selected or dropped. They should not exist — a provider
    returning a US name from an A-share screen is malfunctioning — and that is
    exactly why they must reach the compliance node. Filtering them out here
    would keep the portfolio clean while destroying the pipeline's ability to
    notice, turning an unauthorised cross-border holding into a silent omission
    instead of the REJECT it is supposed to trigger.
    """
    by_market: dict[str, list[AssetCandidate]] = {m: [] for m in markets}
    unscreened: list[AssetCandidate] = []
    for c in candidates:
        if c.market not in by_market:
            unscreened.append(c)
        elif _is_investable(c):
            by_market[c.market].append(c)
    for market in by_market:
        by_market[market].sort(key=_quality_key)

    quota = _allocate_quota(markets, {m: len(v) for m, v in by_market.items()}, budget)
    selected: list[AssetCandidate] = []
    for market in markets:
        selected.extend(by_market[market][:quota[market]])
    return selected + unscreened


def equity_node(state: AdvisoryState, deps) -> dict:
    """Screen equity candidates from market providers.

    Parameters
    ----------
    state:
        AdvisoryState — must have goal_constraints set (from goal_node).
    deps:
        AdvisoryDeps — uses .market.  (deps may be None in test helper calls
        when the caller manages providers directly; if so we skip screening and
        return whatever equity_candidates is already in state.)

    Returns
    -------
    dict
        State increment with keys: equity_candidates, trace_events, notes.
    """
    gc = state.goal_constraints
    risk_ceiling = gc.get("risk_ceiling", "R5")
    accept_cross_border = gc.get("accept_cross_border", True)
    ceiling_order = R_ORDER.get(risk_ceiling, 5)

    # Consume conservative_mode planner hint (C1/C2 profiles).
    hints = gc.get("planner_hints", {})
    conservative_mode = hints.get("conservative_mode", False)

    # Determine markets to screen
    if accept_cross_border:
        markets = _ALL_MARKETS
    else:
        markets = ["A"]

    # Determine PE filter:
    # - conservative_mode (C1/C2) → tightest cap regardless of tilt
    # - underweight macro tilt    → moderate tighter cap
    # - otherwise                 → no hard filter
    tilt = state.macro_view.get("tilt", "neutral") if state.macro_view else "neutral"
    if conservative_mode:
        pe_cap = _PE_CAP_CONSERVATIVE
    elif tilt == "underweight":
        pe_cap = _PE_CAP_UNDERWEIGHT
    else:
        pe_cap = _PE_CAP_DEFAULT

    # Screen from providers
    candidates: list[AssetCandidate] = []
    if deps is not None:
        filters: dict = {"asset_class": "equity"}
        if pe_cap < _PE_CAP_DEFAULT:
            filters["max_pe"] = pe_cap

        for market in markets:
            batch = deps.market.screen(market, filters)
            candidates.extend(batch)
    else:
        # No deps: return existing equity_candidates unchanged (test helper path)
        candidates = list(state.equity_candidates)

    # Filter by risk ceiling
    within_ceiling = [c for c in candidates if R_ORDER.get(c.r_level, 5) <= ceiling_order]

    # Conservative mode keeps a smaller book, but still one built by choosing
    # rather than by truncating a list.
    budget = _MAX_CANDIDATES_CONSERVATIVE if conservative_mode else _EQUITY_BUDGET
    eligible = _select(within_ceiling, markets, budget)

    event = {
        "node": "equity",
        "ts": time.time(),
        "markets": markets,
        "risk_ceiling": risk_ceiling,
        "tilt": tilt,
        "conservative_mode": conservative_mode,
        "pe_cap": pe_cap,
        "total_screened": len(candidates),
        "within_ceiling": len(within_ceiling),
        "eligible": len(eligible),
        # The geographic mix is the point of the selection step, so it belongs in
        # the trace rather than only in the resulting weights.
        "selected_by_market": {
            m: sum(1 for c in eligible if c.market == m) for m in markets
        },
    }
    by_market = ", ".join(
        f"{m}:{sum(1 for c in eligible if c.market == m)}" for m in markets
    )
    note = (
        f"equity_node: screened {len(candidates)} candidates across {markets}; "
        f"{len(within_ceiling)} within {risk_ceiling} ceiling; "
        f"selected {len(eligible)} by quota ({by_market}); macro_tilt={tilt}; "
        f"conservative_mode={conservative_mode}; pe_cap={pe_cap}"
    )

    return {
        "equity_candidates": eligible,
        "trace_events": state.trace_events + [event],
        "notes": state.notes + [note],
    }
