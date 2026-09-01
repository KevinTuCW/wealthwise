"""Multi-factor cross-sectional scoring for equity candidates.

What this replaces
------------------
Selection used to be "largest market cap first, cheaper P/E breaks ties". That
rule was defended in `equity.py` on the grounds that the available fields could
not support a factor model and that an unvalidated scoring formula is worse than
a legible one. The first half of that has stopped being true — daily history now
supplies momentum and realized volatility (see `providers/history.py`).

The second half has now been measured rather than asserted. `scripts/
backtest_factors.py` runs these exact weights over 34 monthly rebalances of this
repo's own 872-name universe (`docs/factor-backtest.md`): the composite picks a
book returning **−4bp a month against the rule it would replace, t = −0.09**.
Positive rank IC across the full cross-section, and nothing left of it once the
quota and the size floor cut the list to the top fifty. So `ENABLE_FACTOR_SCORING`
stays **off by default** — no longer because the formula is unmeasured, but
because it was measured and did not win.

The weights below are unchanged from the house view, deliberately. Re-fitting
them on the same 34 dates would produce whichever mix best explains this sample
and would tell us nothing about the next one; the backtest is allowed to reject
the default, not to author a replacement.

The five factors
----------------
======== =========================== =========================================
factor   input                        direction and why
======== =========================== =========================================
value    E/P from P/E, B/P from P/B   cheap is better; the oldest documented
                                      cross-sectional effect there is
momentum 60-day return, skipping the  winners keep winning over months; the
         last 5 sessions              skip avoids short-term reversal
low_vol  annualised realized vol      lower is better — the low-volatility
                                      anomaly, and it is the factor most
                                      aligned with a suitability mandate
size     log market cap               **larger** is better here, inverting the
                                      academic small-cap premium on purpose:
                                      this book is built for suitability and
                                      liquidity, not for maximum expected
                                      return, and a C2 investor is not being
                                      served a micro-cap for its risk premium
liquidity capped daily turnover       tradability; capped rather than
                                      monotonic, because past a few percent a
                                      day extra churn is speculation and
                                      rewarding it would rank the hottest
                                      name highest every time
======== =========================== =========================================

How scoring works
-----------------
Each factor is turned into a cross-sectional z-score **within one market**, then
combined by weight. Per-market is not a detail: A-shares, Hong Kong and US names
sit in different valuation regimes, and z-scoring the pooled set would rank
markets against each other rather than companies — which is a job the geographic
quota in `equity.py` already does, deliberately and visibly.

Missing inputs do not count against a name. A factor a candidate has no data for
is dropped from *its* composite and the remaining weights are renormalised, so a
thin-data name is scored on what is known about it. The alternative — treating
silence as a bad score — would systematically sink whichever names a weaker feed
happens to cover less well, and dress a data-coverage artefact up as a judgment
about the company.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from wealthwise.agents.state import AssetCandidate

# House-view weights: close to equal, tilted toward the two factors with the
# longest-documented evidence in the literature and away from the two that are
# here for tradability rather than return.
#
# Measured on this universe (`docs/factor-backtest.md`), the literature and the
# candidate pool disagree in three places, and the numbers are recorded here so
# the next person to touch this dict starts from them:
#
#   value      the only factor positive in all three markets (+0.06 / +0.18 /
#              +0.07 mean rank IC) — and the one whose backtest carries
#              look-ahead bias, since past P/E is reconstructed from today's
#              earnings. Read it as an upper bound.
#   momentum   negative in all three markets on this pool. Its 25% is the least
#              defensible number in this dict; dropping it is the one change the
#              evidence points at, and it needs dates this backtest has not seen
#              before it earns the edit.
#   size       significantly negative in A-shares (t = −2.11) and positive in
#              HK/US. Nothing to fix: the tilt to large caps is there for
#              suitability and liquidity, and the A-share reading is what that
#              choice costs, now quantified rather than assumed.
#   low_vol    carries HK (+0.13), flat in A, negative in US.
#   liquidity  no signal anywhere, and no data at all in HK — the quote feed
#              reports 0 in that field, so a fifth of the book is scored on four
#              factors whatever this weight says.
FACTOR_WEIGHTS: dict[str, float] = {
    "value": 0.25,
    "momentum": 0.25,
    "low_vol": 0.20,
    "size": 0.15,
    "liquidity": 0.15,
}

# Z-scores are clipped here before weighting. One name with a 40σ P/E artefact —
# a stale denominator, a mis-parsed column — would otherwise decide the whole
# ranking by itself.
Z_CLIP: float = 3.0

# Daily turnover, in percent, above which more is not better. Blue chips run
# well under 1%; a name turning over 20% of its float in a session is being
# traded, not held.
TURNOVER_CAP: float = 5.0

# A name scored on fewer than this many factors is ranked, but flagged: the
# composite is real, and it is also thinner evidence than its neighbours'.
MIN_COVERAGE: int = 2


@dataclass
class FactorScore:
    """One candidate's composite score and the z-scores behind it."""

    symbol: str
    score: float
    z: dict[str, float] = field(default_factory=dict)
    coverage: int = 0

    @property
    def thin(self) -> bool:
        """True when too few factors had data for this name to be well evidenced."""
        return self.coverage < MIN_COVERAGE


def market_cap_100m(candidate: AssetCandidate) -> float | None:
    """Market cap in units of 100M local currency, or None if unreported.

    Providers disagree on how to say this: the quote-backed ones report
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


# ---------------------------------------------------------------------------
# Raw factor inputs — each returns None when the candidate cannot support it
# ---------------------------------------------------------------------------

def _value(candidate: AssetCandidate) -> float | None:
    """Mean of the earnings yield and book yield that are actually reported.

    Yields rather than the ratios themselves: P/E is unbounded above and
    discontinuous through zero, so averaging or z-scoring it lets one 900×
    multiple dominate a whole market. E/P is bounded, and a loss-making company
    lands negative, which is the ranking it deserves.
    """
    parts: list[float] = []
    pe = candidate.metrics.get("pe")
    if pe is not None and float(pe) != 0:
        parts.append(1.0 / float(pe))
    pb = candidate.metrics.get("pb")
    if pb is not None and float(pb) > 0:
        parts.append(1.0 / float(pb))
    return sum(parts) / len(parts) if parts else None


def _momentum(candidate: AssetCandidate) -> float | None:
    raw = candidate.metrics.get("momentum")
    return float(raw) if raw is not None else None


def _low_vol(candidate: AssetCandidate) -> float | None:
    """Negated volatility, so that — like every other factor — higher is better."""
    raw = candidate.metrics.get("volatility")
    if raw is None:
        return None
    vol = float(raw)
    return -vol if vol > 0 else None


def _size(candidate: AssetCandidate) -> float | None:
    """Log market cap. Raw cap spans four orders of magnitude across a market,
    so the untransformed number makes the largest name an outlier by definition."""
    cap = market_cap_100m(candidate)
    if cap is None or cap <= 0:
        return None
    return math.log10(cap)


def _liquidity(candidate: AssetCandidate) -> float | None:
    raw = candidate.metrics.get("turnover")
    if raw is None:
        return None
    turnover = float(raw)
    if turnover <= 0:
        return None
    return min(turnover, TURNOVER_CAP)


_EXTRACTORS = {
    "value": _value,
    "momentum": _momentum,
    "low_vol": _low_vol,
    "size": _size,
    "liquidity": _liquidity,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _zscores(values: dict[str, float]) -> dict[str, float]:
    """Clipped cross-sectional z-scores for one factor over one market.

    A factor every name agrees on carries no information about which name to
    pick, so a zero standard deviation yields zeros rather than a division by
    one — the difference matters when a market has a single candidate in it.
    """
    if len(values) < 2:
        return dict.fromkeys(values, 0.0)
    series = list(values.values())
    mean = sum(series) / len(series)
    variance = sum((v - mean) ** 2 for v in series) / (len(series) - 1)
    sd = math.sqrt(variance)
    if sd <= 0:
        return dict.fromkeys(values, 0.0)
    return {
        sym: max(-Z_CLIP, min(Z_CLIP, (v - mean) / sd)) for sym, v in values.items()
    }


def score_candidates(candidates: list[AssetCandidate]) -> dict[str, FactorScore]:
    """Score one market's candidates against each other. Keyed by symbol.

    Callers must pass a single market's names: the z-scores are relative to
    whatever is in this list, so mixing markets silently changes what every
    score means.
    """
    if not candidates:
        return {}

    z_by_factor: dict[str, dict[str, float]] = {}
    for factor, extract in _EXTRACTORS.items():
        raw = {}
        for candidate in candidates:
            value = extract(candidate)
            if value is not None:
                raw[candidate.symbol] = value
        if raw:
            z_by_factor[factor] = _zscores(raw)

    out: dict[str, FactorScore] = {}
    for candidate in candidates:
        symbol = candidate.symbol
        z: dict[str, float] = {}
        weighted = 0.0
        total_weight = 0.0
        for factor, weight in FACTOR_WEIGHTS.items():
            value = z_by_factor.get(factor, {}).get(symbol)
            if value is None:
                continue
            z[factor] = round(value, 4)
            weighted += weight * value
            total_weight += weight
        # Renormalise over the factors this name actually had, so a 3-of-5 name
        # is compared on the same scale as a 5-of-5 one instead of being
        # penalised for its provider's coverage.
        score = weighted / total_weight if total_weight > 0 else 0.0
        out[symbol] = FactorScore(
            symbol=symbol, score=round(score, 6), z=z, coverage=len(z)
        )
    return out
