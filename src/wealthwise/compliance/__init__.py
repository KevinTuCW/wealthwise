"""Compliance sub-package: suitability matching and language screening.

Modules
-------
suitability
    Investor-suitability C-R hard gate.  Pure rule-based, deterministic.
language
    Misleading-language detection and neutralization.  Regex/substring-based,
    no ML dependencies.
"""
from wealthwise.compliance.suitability import check_suitability, is_over_level
from wealthwise.compliance.language import detect_misleading, neutralize

__all__ = [
    "check_suitability",
    "is_over_level",
    "detect_misleading",
    "neutralize",
]
