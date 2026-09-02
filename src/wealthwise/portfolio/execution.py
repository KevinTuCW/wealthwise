"""Turn target weights into an order list a person can actually place.

The optimiser produces continuous weights over every surviving candidate. On a
800k CNY book that came out as 64 positions averaging 0.7%, and 18 of them were
allocated less than the cost of a single board lot — a 5,600 CNY slice of a stock
whose minimum trade is 129,799 CNY. Weights like that are not a recommendation;
they are a recommendation-shaped object.

This module closes that gap deterministically, with no model in the loop:

1. Drop positions that cannot afford one board lot, or that fall under a minimum
   meaningful weight, and redistribute their weight within their own asset class
   so the class mix the optimiser chose survives.
2. Repeat until stable — redistribution can lift a previously-unaffordable name
   over the line, and can push a marginal one under it.
3. Round each survivor down to whole lots. The rounding residual goes to cash,
   never to an unplanned overweight.

This runs *after* compliance has ruled, not before. Handing the gate the rounded
result reads as the more auditable order and is the opposite: an unauthorised
holding small enough to be rounded away would leave a clean book behind, and the
verdict would be PASS on a portfolio that should have been rejected. Rounding
must not be able to launder a violation. Compliance judges the recommendation;
this implements the recommendation it approved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wealthwise.agents.state import AssetCandidate, PortfolioAllocation

# A position below this is not worth its transaction and monitoring cost: it
# cannot move the portfolio, but it still has to be tracked and rebalanced.
MIN_POSITION_WEIGHT = 0.02

# Guard against pathological inputs rather than a real product limit.
_MAX_ITERATIONS = 20


@dataclass
class Position:
    """One executable line: what to buy, how much of it, and for how much."""

    symbol: str
    name: str
    market: str
    asset_class: str
    currency: str
    price: float
    lot_size: int
    lots: int
    shares: int
    amount: float          # in the instrument's own currency
    weight: float          # realised share of the book, post-rounding


@dataclass
class ExecutionPlan:
    positions: list[Position] = field(default_factory=list)
    cash_residual: float = 0.0      # unallocated after lot rounding
    investable: float = 0.0
    dropped: list[str] = field(default_factory=list)   # symbol: reason

    @property
    def invested(self) -> float:
        return sum(p.amount for p in self.positions)

    def class_weights(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self.positions:
            out[p.asset_class] = out.get(p.asset_class, 0.0) + p.weight
        if self.cash_residual > 0 and self.investable > 0:
            out["cash"] = out.get("cash", 0.0) + self.cash_residual / self.investable
        return out


def _lot_cost(candidate: AssetCandidate) -> float | None:
    price = candidate.metrics.get("price")
    if not price or price <= 0:
        return None
    lot = int(candidate.metrics.get("lot_size") or 1)
    return float(price) * max(lot, 1)


def _redistribute(weights: dict[str, float], dropped: set[str],
                  class_of: dict[str, str]) -> dict[str, float]:
    """Move dropped weight onto the survivors of the same asset class.

    Spreading it across the whole book instead would quietly rewrite the
    equity/bond/cash split that the optimiser picked to meet the investor's
    risk ceiling and liquidity floor.
    """
    freed: dict[str, float] = {}
    for symbol in dropped:
        freed[class_of[symbol]] = freed.get(class_of[symbol], 0.0) + weights[symbol]

    survivors = {s: w for s, w in weights.items() if s not in dropped}
    for asset_class, amount in freed.items():
        peers = {s: w for s, w in survivors.items() if class_of[s] == asset_class}
        total = sum(peers.values())
        if total <= 0:
            # No peer left to carry this sleeve; the weight falls through to the
            # cash residual rather than being forced onto a different asset class.
            continue
        for symbol, weight in peers.items():
            survivors[symbol] += amount * weight / total
    return survivors


def build_execution_plan(portfolio: PortfolioAllocation,
                         candidates: list[AssetCandidate],
                         investable: float,
                         min_position_weight: float = MIN_POSITION_WEIGHT) -> ExecutionPlan:
    """Convert target weights into whole-lot positions for `investable`."""
    plan = ExecutionPlan(investable=investable)
    if investable <= 0 or not portfolio.weights:
        return plan

    by_symbol = {c.symbol: c for c in candidates}
    weights = {s: w for s, w in portfolio.weights.items() if w > 0}
    class_of = {s: (by_symbol[s].asset_class if s in by_symbol else "equity")
                for s in weights}

    reasons: dict[str, str] = {}

    def _violation(symbol: str, weight: float) -> str | None:
        candidate = by_symbol.get(symbol)
        if candidate is None:
            return "no quote"
        cost = _lot_cost(candidate)
        if cost is None:
            return "no price"
        if weight < min_position_weight:
            return f"below {min_position_weight:.0%} minimum"
        if investable * weight < cost:
            return f"one lot costs {cost:,.0f}"
        return None

    def _difficulty(symbol: str, weight: float) -> float:
        """How many times its target allocation one lot of this name costs.

        This is the drop order. Ranking by weight instead looked reasonable and
        was not: risk parity hands a sleeve near-identical weights, so "smallest
        first" resolved to floating-point noise and evicted names arbitrarily.
        The A-share sleeve disappeared that way while HK and US survived, which
        is not a decision anyone made. Ranking by difficulty evicts the names
        that are genuinely hard to fit — a stock whose board lot costs more than
        its entire target allocation — and keeps the cheap-lot ones that fit.
        """
        candidate = by_symbol.get(symbol)
        cost = _lot_cost(candidate) if candidate else None
        if cost is None:
            return float("inf")
        target = investable * weight
        return cost / target if target > 0 else float("inf")

    # Drop one position at a time, hardest to fit first, redistributing after
    # each. Dropping every violator in a single pass looked equivalent and was
    # not: an equity sleeve spread thinly enough that *all* of its names sat
    # under the minimum lost every one of them at once, leaving no peer to
    # inherit the weight. 35% of the book fell through to idle cash and the
    # investor was handed a portfolio with no equities in it. Removing the worst
    # offender first lifts its peers, which is the point of the floor.
    for _ in range(_MAX_ITERATIONS * max(len(weights), 1)):
        violations = {s: _violation(s, w) for s, w in weights.items()}
        offenders = [s for s, why in violations.items() if why]
        if not offenders or len(offenders) == len(weights):
            reasons.update({s: why for s, why in violations.items() if why})
            break
        victim = max(offenders, key=lambda s: _difficulty(s, weights[s]))
        reasons[victim] = violations[victim] or "not executable"
        weights = _redistribute(weights, {victim}, class_of)

    # Anything still failing the floor after the loop settles is not executable.
    for symbol in list(weights):
        candidate = by_symbol.get(symbol)
        cost = _lot_cost(candidate) if candidate else None
        if cost is None or investable * weights[symbol] < cost:
            reasons.setdefault(symbol, "not executable at this size")
            del weights[symbol]

    for symbol, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        candidate = by_symbol[symbol]
        price = float(candidate.metrics["price"])
        lot = max(int(candidate.metrics.get("lot_size") or 1), 1)
        lots = int((investable * weight) // (price * lot))
        if lots <= 0:
            continue
        shares = lots * lot
        amount = shares * price
        plan.positions.append(Position(
            symbol=symbol, name=candidate.name, market=candidate.market,
            asset_class=candidate.asset_class, currency=candidate.currency,
            price=price, lot_size=lot, lots=lots, shares=shares,
            amount=amount, weight=amount / investable,
        ))

    # Sweep the rounding remainder into money-market funds. Rounding 30 positions
    # down leaves real money behind — 22% of a 100k book at the small end — and
    # "we could not fit it into a whole lot" is not a reason to advise leaving it
    # idle. A money-market ETF is where that cash belongs, and it is the one
    # instrument in the book whose whole purpose is absorbing exactly this.
    residual = investable - plan.invested
    sweeps = sorted((p for p in plan.positions if p.asset_class == "cash"),
                    key=lambda p: p.price * p.lot_size)
    # One lot at a time, round-robin. Filling the cheapest fund to exhaustion
    # first is equivalent in risk — money-market funds are interchangeable here —
    # but it lands as one 13.7% holding beside five 2.5% ones, which reads like a
    # defect to anyone looking at the sheet. Spreading it keeps the sleeve even.
    progressed = True
    while residual > 0 and sweeps and progressed:
        progressed = False
        for position in sweeps:
            lot_cost = position.price * position.lot_size
            if lot_cost <= 0 or residual < lot_cost:
                continue
            position.lots += 1
            position.shares += position.lot_size
            position.amount += lot_cost
            position.weight = position.amount / investable
            residual -= lot_cost
            progressed = True

    plan.cash_residual = residual
    plan.dropped = [f"{s}: {r}" for s, r in reasons.items()]
    return plan


def realised_allocation(portfolio: PortfolioAllocation,
                        plan: ExecutionPlan) -> PortfolioAllocation:
    """Rebuild the allocation from what the plan actually buys.

    This is reporting, not a second gate. Compliance rules on the *pre-rounding*
    target and the plan is built afterwards, because a holding small enough to be
    rounded away would otherwise leave a clean book behind — rounding must not be
    able to launder a violation. So this describes what the investor ends up
    holding; the approved recommendation is `portfolio`.
    """
    if not plan.positions:
        return portfolio
    fx = sum(p.weight for p in plan.positions if p.currency != "CNY")
    return portfolio.model_copy(update={
        "weights": {p.symbol: round(p.weight, 6) for p in plan.positions},
        "class_weights": {k: round(v, 6) for k, v in plan.class_weights().items()},
        "fx_exposure": round(fx, 6),
    })
