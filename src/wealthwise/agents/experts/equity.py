"""equity_node — screen equity candidates using goal_constraints + macro_view.

Screens across A / HK / US markets using goal_constraints (risk_ceiling,
accept_cross_border) and macro_view tilt to filter and annotate candidates.
No LLM call needed here — purely deterministic rule-based screening and ranking.

Ranking runs one of two ways, chosen by `deps.enable_factor_scoring`:

* **quality** (default) — largest first, cheaper valuation breaks ties. Legible,
  and it commits to nothing it cannot defend.
* **factor** — the five-factor cross-sectional composite in
  `portfolio/factors.py`, fed by daily history when a `HistoryProvider` is
  wired. Both paths run through the same quota and the same guardrails; only the
  sort key differs, which keeps the switch honest — it changes which names are
  picked, never how many or from where.
"""
from __future__ import annotations

import time

from wealthwise.agents.state import AdvisoryState, AssetCandidate
from wealthwise.portfolio.factors import FactorScore, market_cap_100m, score_candidates
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
    cap = market_cap_100m(candidate)
    return cap is None or cap >= _MIN_MARKET_CAP_100M


def _quality_key(candidate: AssetCandidate) -> tuple:
    """Sort key: larger companies first, ties broken by cheaper valuation.

    Size stands in for liquidity and stability, which is what suitability
    actually cares about — not for expected return. This rule commits to nothing
    it cannot defend from two fields, which is why it remains the default and the
    fallback: `portfolio/factors.py` ranks on more, but on weights that are a
    house view rather than a validated one.

    Names with no reported size sort last rather than being dropped, so a
    thin-data provider degrades to "quota respected, order arbitrary" instead of
    to an empty book.
    """
    cap = market_cap_100m(candidate)
    pe = candidate.metrics.get("pe")
    return (-(cap if cap is not None else 0.0),
            pe if pe is not None else float("inf"),
            candidate.symbol)


def _factor_key(scores: dict[str, FactorScore]):
    """Sort key factory: highest composite first, quality rule breaks ties.

    The quality tuple is kept as the tiebreak rather than the symbol, because
    every score collapses to 0.0 when a market has one candidate or when no
    factor had data — and in that case the ranking should degrade to the rule it
    replaced, not to alphabetical order.
    """
    def key(candidate: AssetCandidate) -> tuple:
        score = scores.get(candidate.symbol)
        return (-(score.score if score else 0.0), *_quality_key(candidate))
    return key


def _disagreed(candidate: AssetCandidate) -> bool:
    """True when the consensus layer found the feeds materially at odds on price.

    Only price counts. Two sources quoting different P/E are using different
    earnings windows, which is a methodology difference; two sources quoting
    different *prices* for the same instrument means at least one of them is
    wrong about what it is looking at, and that name should not be sized into an
    order list on the strength of it.
    """
    return "price" in (candidate.metrics.get("data_disagreement") or [])


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


def _select(candidates: list[AssetCandidate], markets: list[str], budget: int,
            factor_scoring: bool = False) -> tuple[list[AssetCandidate], dict]:
    """Pick the best `budget` names, respecting the geographic split.

    Candidates from a market that was never screened are passed through untouched
    rather than quota-selected or dropped. They should not exist — a provider
    returning a US name from an A-share screen is malfunctioning — and that is
    exactly why they must reach the compliance node. Filtering them out here
    would keep the portfolio clean while destroying the pipeline's ability to
    notice, turning an unauthorised cross-border holding into a silent omission
    instead of the REJECT it is supposed to trigger.

    Returns the selection plus a ranking record for the trace: which rule ran,
    and the factor scores behind the names that made it.
    """
    by_market: dict[str, list[AssetCandidate]] = {m: [] for m in markets}
    unscreened: list[AssetCandidate] = []
    for c in candidates:
        if c.market not in by_market:
            unscreened.append(c)
        elif _is_investable(c):
            by_market[c.market].append(c)

    # Scored per market, never pooled: the z-scores are relative to the list they
    # are computed over, so pooling would rank markets against each other and
    # quietly duplicate the job _MARKET_QUOTA already does explicitly.
    scores: dict[str, FactorScore] = {}
    for market, members in by_market.items():
        if factor_scoring:
            scores.update(score_candidates(members))
        members.sort(key=_factor_key(scores) if factor_scoring else _quality_key)

    quota = _allocate_quota(markets, {m: len(v) for m, v in by_market.items()}, budget)
    selected: list[AssetCandidate] = []
    for market in markets:
        selected.extend(by_market[market][:quota[market]])

    ranking = {"method": "factor" if factor_scoring else "quality"}
    if factor_scoring:
        # Selection order, not a global leaderboard. Scores are z-scores within
        # one market, so ranking an A-share against a US name by score would
        # compare two numbers that were never on the same scale — the exact
        # mistake per-market scoring exists to avoid.
        ranking["top"] = [
            {"symbol": c.symbol, "market": c.market,
             "score": scores[c.symbol].score, "z": scores[c.symbol].z}
            for c in selected[:5] if c.symbol in scores
        ]
        ranking["thin_evidence"] = sorted(
            s.symbol for s in scores.values() if s.thin and s.symbol in
            {c.symbol for c in selected}
        )
    return selected + unscreened, ranking


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

    # Data-quality gate. A name whose two feeds disagree on price is excluded
    # from selection but counted in the trace: the count is the observable that
    # tells you a feed has drifted, and silently dropping the names would hide
    # exactly the signal the consensus layer exists to produce.
    drop_disagreed = getattr(deps, "drop_on_data_disagreement", True) if deps else False
    disagreed = [c for c in within_ceiling if _disagreed(c)]
    rankable = [c for c in within_ceiling if not _disagreed(c)] if drop_disagreed \
        else within_ceiling

    # Momentum and realized volatility are not in any spot quote, so the factor
    # path fetches history first. Only for the names still in contention — this
    # is one request per symbol, and enriching everything screened would pay for
    # hundreds of names the quota was never going to reach.
    factor_scoring = bool(getattr(deps, "enable_factor_scoring", False)) if deps else False
    history = getattr(deps, "history", None) if deps else None
    if factor_scoring and history is not None and rankable:
        rankable = history.enrich(rankable)

    # Conservative mode keeps a smaller book, but still one built by choosing
    # rather than by truncating a list.
    budget = _MAX_CANDIDATES_CONSERVATIVE if conservative_mode else _EQUITY_BUDGET
    eligible, ranking = _select(rankable, markets, budget, factor_scoring)

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
        "ranking": ranking,
        "data_disagreement": [c.symbol for c in disagreed],
        # The geographic mix is the point of the selection step, so it belongs in
        # the trace rather than only in the resulting weights.
        "selected_by_market": {
            m: sum(1 for c in eligible if c.market == m) for m in markets
        },
    }
    by_market = ", ".join(
        f"{m}:{sum(1 for c in eligible if c.market == m)}" for m in markets
    )
    dropped = (f"; dropped {len(disagreed)} on source disagreement"
               if drop_disagreed and disagreed else "")
    note = (
        f"equity_node: screened {len(candidates)} candidates across {markets}; "
        f"{len(within_ceiling)} within {risk_ceiling} ceiling; "
        f"selected {len(eligible)} by quota ({by_market}) ranked by "
        f"{ranking['method']}{dropped}; macro_tilt={tilt}; "
        f"conservative_mode={conservative_mode}; pe_cap={pe_cap}"
    )

    return {
        "equity_candidates": eligible,
        "trace_events": state.trace_events + [event],
        "notes": state.notes + [note],
    }
