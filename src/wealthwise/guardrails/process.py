"""Process guardrail — clean and bound the asset-candidate list.

Three steps (applied in order):
1. Drop malformed candidates: empty symbol, invalid R-level, invalid market.
   Real provider payloads can be messy; garbage in must never reach the
   optimizer or compliance gate.
2. Deduplicate by symbol (keep first occurrence — the one with the highest
   confidence score, since providers typically sort by relevance).
3. Truncate to max_candidates to bound downstream compute cost.
"""
from __future__ import annotations

from wealthwise.agents.state import AssetCandidate

VALID_R_LEVELS = {"R1", "R2", "R3", "R4", "R5"}
VALID_MARKETS = {"A", "HK", "US"}


def cap_candidates(
    candidates: list[AssetCandidate],
    max_candidates: int = 50,
) -> list[AssetCandidate]:
    """Dedupe, clean, and truncate *candidates*.

    Parameters
    ----------
    candidates:
        Raw candidate list from one or more asset providers.
    max_candidates:
        Hard upper bound on the returned list length.

    Returns
    -------
    list[AssetCandidate]
        Cleaned list of at most *max_candidates* items.
    """
    cleaned: list[AssetCandidate] = []
    seen: set[str] = set()

    for c in candidates:
        # drop dirty candidates
        if not c.symbol:
            continue
        if c.r_level not in VALID_R_LEVELS:
            continue
        if c.market not in VALID_MARKETS:
            continue
        # deduplicate: keep first occurrence of each symbol
        if c.symbol in seen:
            continue
        seen.add(c.symbol)
        cleaned.append(c)

    # truncate to max width
    return cleaned[:max_candidates]
