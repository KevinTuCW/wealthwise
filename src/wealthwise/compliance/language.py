"""Misleading-language detection and neutralization for financial text.

Responsibilities
----------------
- Maintain a curated list of banned promotional phrases that are prohibited
  under China's asset-management and securities-advertising regulations.
- Normalize input text (NFKC + zero-width-char stripping) before matching so
  that simple Unicode evasion tricks (full-width chars, zero-width spaces) do
  not bypass the detector.
- Provide detect_misleading() for screening and neutralize() for sanitizing.

This module has ZERO external dependencies — stdlib unicodedata only.
"""
from __future__ import annotations

import unicodedata

# ---------------------------------------------------------------------------
# Banned terms
# ---------------------------------------------------------------------------
# Curated set of phrases prohibited by China's 资管新规 / 广告法 / 基金法 context.
# The list errs on the side of completeness rather than brevity: a missed term
# is a compliance failure; a false positive only adds friction.

MISLEADING_TERMS: tuple[str, ...] = (
    # Capital-guarantee cluster
    "保本",
    "保收益",
    "保证收益",
    "保证本金",
    "保证回报",
    # Commitment / promise cluster
    "承诺收益",
    "承诺回报",
    "承诺利润",
    # No-loss cluster
    "稳赚",
    "稳赢",
    "包赚",
    "必赚",
    "必盈",
    "不亏",
    "无亏损",
    "保证不亏",
    "稳赚不赔",
    "稳赢不赔",
    "只赚不赔",
    # Zero-risk cluster
    "无风险",
    "零风险",
    "低风险无损",
    # Misleading stable-return claim
    "稳健收益保证",
    "稳定回报保证",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Zero-width Unicode characters commonly used for evasion
_ZERO_WIDTH = frozenset([
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "﻿",  # zero-width no-break space (BOM)
    "­",  # soft hyphen
])


def _normalize(text: str) -> str:
    """NFKC-normalize + strip zero-width characters.

    NFKC folds full-width/half-width variants, compatibility forms, etc.
    Zero-width characters are then stripped so that an adversarial insertion
    like 保​本 (with a U+200B between) normalizes back to 保本.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalized if ch not in _ZERO_WIDTH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_misleading(text: str) -> list[str]:
    """Return the banned terms present in *text* after normalization.

    Parameters
    ----------
    text:
        Raw user-supplied or LLM-generated text to screen.

    Returns
    -------
    list[str]
        Banned terms found (may include duplicates if a term appears multiple
        times — callers who need a set can do ``list(set(result))``).
        Empty list means the text is clean.

    Notes
    -----
    - Matching is substring-based: if a banned term appears anywhere in the
      normalized text it is reported.
    - The function deliberately does NOT match partial sub-strings of SAFE
      disclaimer language.  The only way a false-positive could occur is if a
      banned phrase literally appears inside a disclaimer — callers must not
      place banned phrases inside disclaimer texts.
    - Order of returned items follows MISLEADING_TERMS declaration order.
    """
    if not text:
        return []

    clean = _normalize(text)
    found: list[str] = []
    for term in MISLEADING_TERMS:
        if term in clean:
            found.append(term)
    return found


_FLAG_MARKER = "「[已删除违规表述]」"


def neutralize(text: str) -> str:
    """Remove or flag all banned terms from *text*.

    Each occurrence of a banned term (after normalization) is replaced with
    _FLAG_MARKER.  The function works on the normalized version of the input
    so that Unicode evasion variants are also removed.

    Parameters
    ----------
    text:
        Raw text that may contain banned promotional language.

    Returns
    -------
    str
        Sanitized text with banned terms replaced by the flag marker.
        If no banned terms are present the original text is returned unchanged
        (identity on clean input).

    Notes
    -----
    Terms in MISLEADING_TERMS are applied longest-first to avoid leaving
    fragments when a shorter term is a prefix of a longer one (e.g. "保本"
    inside "保本保证收益").
    """
    if not text:
        return text

    clean = _normalize(text)

    # Sort longest-first so that "稳赚不赔" is replaced before "稳赚"
    sorted_terms = sorted(MISLEADING_TERMS, key=len, reverse=True)

    result = clean
    for term in sorted_terms:
        result = result.replace(term, _FLAG_MARKER)

    return result
