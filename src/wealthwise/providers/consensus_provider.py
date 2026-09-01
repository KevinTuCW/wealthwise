"""Consensus providers — where pillar one stops being a class and starts running.

`ConsensusResolver` and `SourceRegistry` have existed since Phase 1 with no
caller. Everything downstream read a single provider, so "多源共识" described a
median over one number. These two wrappers are the callers: they sit in the same
`MarketProvider` / `MacroProvider` slots the expert nodes already depend on, so
nothing downstream has to know that a reading is now corroborated.

What each one reconciles
------------------------
`ConsensusMarketProvider` screens on its **primary** source and cross-checks the
result against the others. The primary defines the universe and owns the filter
semantics; corroborating feeds are asked only "what do you have for these
symbols?". The alternative — screening on every source and unioning — sounds
more symmetric and is worse: a feed that reports no P/E cannot honour a `max_pe`
screen, so it would contribute a set that was never filtered at all and the
union would quietly re-admit everything the screen just removed.

`ConsensusMacroProvider` has no such asymmetry: each source publishes a flat
snapshot, so every numeric key is resolved through `SourceRegistry` across all
of them. Non-numeric fields (`as_of`, the qualitative view blocks) cannot be
median-ed and are taken from the first source that reports them.

Honest single-source behaviour
------------------------------
Registered with one source, both wrappers return that source's numbers unchanged
and record `confidence = 0.5` — the resolver's cap for a reading nothing
corroborates. That is the correct reading of the offline stack, and it is why
the offline pipeline can run through this layer without its results moving.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from wealthwise.agents.state import AssetCandidate
from wealthwise.providers.consensus import ConsensusResolver, ConsensusResult, Reading
from wealthwise.providers.registry import SourceRegistry

if TYPE_CHECKING:
    from wealthwise.providers.base import MacroProvider, MarketProvider

# Metrics worth reconciling, and how far two feeds may drift before the
# disagreement is reported rather than averaged away.
#
# The tolerances differ by an order of magnitude because the quantities do. Two
# feeds quoting the same exchange should agree on price to within a tick — 2% is
# already generous and catches the failure that matters, a stale or mis-mapped
# symbol. Valuation ratios legitimately differ on methodology (trailing vs
# rolling earnings windows, share counts as of different dates), so a 15% gap
# there is a methodology difference, not a data error, and flagging it would
# train everyone to ignore the flag.
_METRIC_TOLERANCE = {
    "price": 0.02,
    "market_cap_100m": 0.10,
    "pe": 0.15,
    "pb": 0.15,
}

# Metrics that cannot legitimately be zero or negative. A halted symbol comes
# back as 0.00 on some feeds, and a 0 alongside a real 1299 makes the two-source
# median 649 — not a disagreement the resolver can flag, but a fabricated number
# delivered with full confidence. P/E and P/B are absent from this set because a
# negative P/E is real information: the company is losing money.
_POSITIVE_ONLY = ("price", "market_cap_100m")

# Default spread threshold for macro signals. Macro series are revised,
# published on different calendars, and quoted by different aggregators, so they
# are held to a looser standard than an intraday quote.
_MACRO_THRESHOLD = 0.10


def _source_name(source: object, index: int) -> str:
    """Label a provider for the readings it contributes."""
    return str(getattr(source, "name", None) or f"{type(source).__name__}#{index}")


def _numeric(value: object) -> float | None:
    """Return value as a float when it is a real number, else None.

    `bool` is excluded deliberately: it is an `int` subclass in Python, and a
    flag reconciled into a median is a category error.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _summarise(result: ConsensusResult) -> dict:
    """Flatten a ConsensusResult into the JSON-safe bag carried through state."""
    return {
        "value": result.value,
        "confidence": round(result.confidence, 4),
        "disagreement": result.disagreement,
        "sources": [r.source for r in result.readings],
        "readings": {r.source: r.value for r in result.readings},
    }


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

class ConsensusMarketProvider:
    """Reconcile per-symbol metrics across several market feeds.

    Parameters
    ----------
    sources:
        Market providers, **primary first**. The primary owns screening; the
        rest are asked for quotes on whatever the primary returned.
    """

    name = "consensus"

    def __init__(self, sources: list["MarketProvider"]) -> None:
        if not sources:
            raise ValueError("ConsensusMarketProvider needs at least one source")
        self._sources = list(sources)
        self._names = [_source_name(s, i) for i, s in enumerate(sources)]
        self._resolvers = {
            metric: ConsensusResolver(threshold=tol)
            for metric, tol in _METRIC_TOLERANCE.items()
        }

    @property
    def source_names(self) -> list[str]:
        return list(self._names)

    @property
    def sources(self) -> list["MarketProvider"]:
        """The wrapped feeds, primary first — so callers can still see through."""
        return list(self._sources)

    def quotes(self, symbols: list[str]) -> list[AssetCandidate]:
        primary = self._sources[0].quotes(symbols)
        return self._reconcile(primary, [c.symbol for c in primary])

    def screen(self, market: str, filters: dict) -> list[AssetCandidate]:
        """Screen on the primary source, then cross-check every candidate.

        The `max_pe` filter is re-applied against the *reconciled* P/E, so a name
        that only clears the cap on the primary feed's number does not clear it
        here. Re-checking can only tighten the screen the caller asked for; it
        never re-admits a name the primary rejected.
        """
        primary = self._sources[0].screen(market, filters)
        reconciled = self._reconcile(primary, [c.symbol for c in primary])

        max_pe = filters.get("max_pe")
        if max_pe is None:
            return reconciled
        kept: list[AssetCandidate] = []
        for candidate in reconciled:
            pe = candidate.metrics.get("pe")
            if pe is not None and pe > max_pe:
                continue
            kept.append(candidate)
        return kept

    # -- internals ----------------------------------------------------------

    def _corroborating(
        self, symbols: list[str]
    ) -> list[tuple[str, dict[str, AssetCandidate]]]:
        """Quote `symbols` on every non-primary source, keyed by symbol.

        A corroborating feed that fails is not allowed to fail the advisory. The
        primary already returned a usable answer, and degrading from "two sources
        agree" to "one source, confidence 0.5" is the honest outcome — the
        alternative is an outage in a cross-check taking down the thing it was
        added to make safer.
        """
        out: list[tuple[str, dict[str, AssetCandidate]]] = []
        for name, source in zip(self._names[1:], self._sources[1:]):
            try:
                quoted = source.quotes(symbols)
            except Exception:
                quoted = []
            out.append((name, {c.symbol.casefold(): c for c in quoted}))
        return out

    def _reconcile(self, primary: list[AssetCandidate],
                   symbols: list[str]) -> list[AssetCandidate]:
        if not primary:
            return []
        others = self._corroborating(symbols) if len(self._sources) > 1 else []

        out: list[AssetCandidate] = []
        for candidate in primary:
            key = candidate.symbol.casefold()
            # Pairs, not a bare list: a feed missing this symbol must drop out of
            # the labelling too, or every later source's reading gets filed under
            # the name of the one that had nothing to say.
            peers = [(name, table[key]) for name, table in others if key in table]
            out.append(self._merge(candidate, peers))
        return out

    def _merge(self, candidate: AssetCandidate,
               peers: list[tuple[str, AssetCandidate]]) -> AssetCandidate:
        """Return `candidate` with reconciled metrics and a consensus record.

        Non-numeric fields (name, market, asset class, lot size) stay the
        primary's. Those are not measurements and have no median; a lot size
        averaged across two feeds would be an unplaceable order.
        """
        metrics = dict(candidate.metrics)
        consensus: dict[str, dict] = {}
        disagreements: list[str] = []
        confidences: list[float] = []

        for metric, resolver in self._resolvers.items():
            readings: list[Reading] = []
            for name, source in [(self._names[0], candidate), *peers]:
                value = _numeric(source.metrics.get(metric))
                if value is None:
                    continue
                if metric in _POSITIVE_ONLY and value <= 0:
                    continue
                readings.append(Reading(source=name, value=value))
            if not readings:
                continue

            result = resolver.resolve(readings)
            metrics[metric] = result.value
            consensus[metric] = _summarise(result)
            confidences.append(result.confidence)
            if result.disagreement:
                disagreements.append(metric)

        metrics["consensus"] = consensus
        # Deliberately the floor across every metric, not an average: a name whose
        # price two feeds dispute is not rescued by their agreeing on its market
        # cap. In practice coverage binds before disagreement does — the second
        # feed publishes no P/E for A-shares or Hong Kong, so mainland names sit
        # at 0.5 on a normal day and drop toward 0 when a price is contested. Read
        # it as "the least-corroborated number on this name", and read the
        # per-metric records above for which number that is.
        metrics["data_confidence"] = round(min(confidences), 4) if confidences else 0.0

        tags = list(candidate.tags)
        if disagreements:
            metrics["data_disagreement"] = disagreements
            tag = "data-disagreement"
            if tag not in tags:
                tags.append(tag)

        return candidate.model_copy(update={"metrics": metrics, "tags": tags})


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------

class SnapshotSignalSource:
    """Adapt a `MacroProvider` to the `SignalProvider` interface the registry wants.

    `SourceRegistry` pulls one signal at a time, but a macro provider publishes a
    whole snapshot per call — and for the live sources that call is a network
    round trip. The snapshot is therefore fetched once and cached until
    `refresh()`, so resolving twelve signals costs one fetch per source rather
    than twelve.
    """

    def __init__(self, provider: "MacroProvider", name: str) -> None:
        self.name = name
        self.provider = provider
        self._cache: dict | None = None

    def refresh(self) -> None:
        self._cache = None

    def snapshot(self) -> dict:
        if self._cache is None:
            try:
                self._cache = dict(self.provider.snapshot() or {})
            except Exception:
                # Same rule as the market side: a source that is down contributes
                # nothing rather than failing the run.
                self._cache = {}
        return self._cache

    def signal_readings(self, signal: str) -> list[Reading]:
        value = _numeric(self.snapshot().get(signal))
        return [Reading(source=self.name, value=value)] if value is not None else []


class QualitativeMacroSource:
    """A macro provider stripped down to its non-numeric fields.

    The offline sample snapshot is the only thing in the stack that carries the
    qualitative view blocks (`asset_class_view`, `market_view`,
    `risk_sentiment`) that no free feed publishes, and the pipeline would be
    poorer without them. But it also carries numbers — a static CPI, a static
    LPR — and registering it as a full source would let a JSON file vote in the
    median and hand a live reading corroboration it never received. Numbers out,
    labels in.
    """

    def __init__(self, provider: "MacroProvider", name: str | None = None) -> None:
        self._provider = provider
        self.name = name or f"{_source_name(provider, 0)}-qualitative"

    def snapshot(self) -> dict:
        return {
            key: value
            for key, value in (self._provider.snapshot() or {}).items()
            if _numeric(value) is None
        }


class ConsensusMacroProvider:
    """Reconcile macro snapshots across several publishers.

    Every numeric key present in any source is resolved through `SourceRegistry`;
    a key only one source publishes keeps its value and is recorded at
    confidence 0.5, which is exactly what "we have this from one place" is worth.
    """

    name = "consensus-macro"

    def __init__(self, sources: list["MacroProvider"],
                 threshold: float = _MACRO_THRESHOLD) -> None:
        if not sources:
            raise ValueError("ConsensusMacroProvider needs at least one source")
        self._adapters = [
            SnapshotSignalSource(s, _source_name(s, i)) for i, s in enumerate(sources)
        ]
        self._resolver = ConsensusResolver(threshold=threshold)

    @property
    def source_names(self) -> list[str]:
        return [a.name for a in self._adapters]

    @property
    def sources(self) -> list["MacroProvider"]:
        """The wrapped publishers, in registration order."""
        return [a.provider for a in self._adapters]

    def snapshot(self) -> dict:
        for adapter in self._adapters:
            adapter.refresh()

        registry = SourceRegistry(resolver=self._resolver)
        for adapter in self._adapters:
            registry.register(adapter)

        snapshots = [a.snapshot() for a in self._adapters]

        # Numeric signals: median across whoever publishes them.
        numeric_keys: list[str] = []
        for snap in snapshots:
            for key, value in snap.items():
                if _numeric(value) is not None and key not in numeric_keys:
                    numeric_keys.append(key)

        out: dict = {}
        consensus: dict[str, dict] = {}
        disagreements: list[str] = []
        for key in numeric_keys:
            result = registry.resolve(key)
            out[key] = result.value
            consensus[key] = _summarise(result)
            if result.disagreement:
                disagreements.append(key)

        # Qualitative fields: first publisher wins, in registration order. These
        # are labels and dates, not measurements — there is no median of
        # "cautious", and picking one publisher's is more honest than blending.
        for snap in snapshots:
            for key, value in snap.items():
                if key not in out and _numeric(value) is None:
                    out[key] = value

        out["consensus"] = consensus
        out["sources"] = self.source_names
        if disagreements:
            out["disagreement"] = disagreements
        return out
