"""Multi-source consensus reconciliation for market signal readings.

Mirrors shopscout's consensus.py logic — adapted to the finance domain.

Resolution rules (same algorithm as shopscout):
- Single source   → confidence capped at 0.5 (cannot be corroborated)
- All values equal → spread=0 → full confidence (handles all-zero edge case)
- Median=0 but values differ → spread=inf → confidence=0, disagreement=True
- Otherwise       → spread = (max-min)/|median|; confidence = clamp(1-spread)
                    disagreement = spread > threshold
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from wealthwise.utils import clamp


@dataclass
class Reading:
    """A single numeric signal reading from one source."""
    source: str
    value: float


@dataclass
class ConsensusResult:
    """Reconciled value + meta from ConsensusResolver.resolve()."""
    value: float
    confidence: float          # 0..1
    disagreement: bool         # True when spread exceeds threshold
    readings: list[Reading] = field(default_factory=list)


class ConsensusResolver:
    """Reconcile multiple sources' readings for one signal.

    Median is the reconciled value (robust to a single outlier source).
    Confidence falls as sources disagree; a lone source is capped at 0.5
    because it cannot be corroborated.
    """

    def __init__(self, threshold: float = 0.2) -> None:
        self._threshold = threshold

    def resolve(self, readings: list[Reading]) -> ConsensusResult:
        if not readings:
            raise ValueError("resolve() needs at least one reading")

        if len(readings) == 1:
            return ConsensusResult(
                value=readings[0].value,
                confidence=0.5,
                disagreement=False,
                readings=readings,
            )

        values = [r.value for r in readings]
        mid = median(values)

        if max(values) == min(values):
            spread = 0.0                    # sources agree exactly, even if all 0
        elif mid == 0:
            spread = float("inf")           # values differ but median is 0
        else:
            spread = (max(values) - min(values)) / abs(mid)

        return ConsensusResult(
            value=mid,
            confidence=clamp(1.0 - spread),
            disagreement=spread > self._threshold,
            readings=readings,
        )
