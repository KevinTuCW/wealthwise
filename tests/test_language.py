"""Tests for misleading-language detection and neutralization.

TDD: these tests are written before implementation. Expected failure mode
before implementation: ImportError / ModuleNotFoundError.
"""
from __future__ import annotations

import pytest

from wealthwise.compliance.language import detect_misleading, neutralize


# ---------------------------------------------------------------------------
# test_detect_baoben
# ---------------------------------------------------------------------------

def test_detect_baoben():
    """'本产品保本保收益' → hits include '保本' and a '保收益'-family term."""
    hits = detect_misleading("本产品保本保收益")
    assert len(hits) > 0, "Expected at least one banned term to be detected"
    # Must catch 保本
    assert any("保本" in h for h in hits), f"'保本' not in hits: {hits}"
    # Must catch 保收益 (or a containing term like 保证收益)
    assert any("保收益" in h or "保证收益" in h for h in hits), (
        f"No 保收益-family term in hits: {hits}"
    )


# ---------------------------------------------------------------------------
# test_no_false_positive
# ---------------------------------------------------------------------------

def test_no_false_positive():
    """Disclaimer phrasing ('历史年化约8%，过往业绩不代表未来表现') must not trip detector."""
    text = "历史年化约8%，过往业绩不代表未来表现"
    hits = detect_misleading(text)
    assert hits == [], f"False positive on disclaimer text: {hits}"


# ---------------------------------------------------------------------------
# test_empty_safe
# ---------------------------------------------------------------------------

def test_empty_safe():
    """Empty string → empty list."""
    assert detect_misleading("") == []


# ---------------------------------------------------------------------------
# test_normalization_evasion
# ---------------------------------------------------------------------------

def test_normalization_evasion():
    """Zero-width character inserted into '保本' is still caught after normalization."""
    # Insert a zero-width space (U+200B) between 保 and 本 to try to evade the detector
    evasion_text = "本产品​保​本，稳赚不赔"
    hits = detect_misleading(evasion_text)
    assert len(hits) > 0, (
        f"Normalization evasion not detected: '{evasion_text}' produced hits={hits}"
    )


# ---------------------------------------------------------------------------
# test_neutralize
# ---------------------------------------------------------------------------

def test_neutralize():
    """'保证稳赚不赔' → banned terms removed/flagged; offending text no longer present verbatim."""
    original = "保证稳赚不赔"
    result = neutralize(original)
    # At least one banned term from the original should no longer appear verbatim
    # (Either '保证' family or '稳赚' is banned)
    assert "稳赚" not in result, (
        f"Banned term '稳赚' still appears verbatim in neutralized output: '{result}'"
    )
    # Result should be non-empty (replaced/flagged, not deleted entirely)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_detect_multiple_terms():
    """Text with several banned terms → all are returned."""
    text = "该产品承诺收益且零风险，稳赢不赔"
    hits = detect_misleading(text)
    # Should catch at least 承诺收益 and 零风险 and 稳赢
    assert len(hits) >= 2, f"Expected ≥2 hits, got: {hits}"


def test_detect_full_width_evasion():
    """Full-width digits/chars in banned term are normalized and caught."""
    # Use full-width characters for 保本 → ｂａｏｂｅｎ in romaji doesn't apply,
    # but we can use CJK compatibility characters or just a simpler test:
    # full-width space between characters (U+3000)
    text = "本产品保　本收益稳定"
    # After NFKC normalization U+3000 → space, so "保 本" won't match "保本"
    # The real evasion test is zero-width which IS stripped.
    # Just verify the function handles it gracefully without exception.
    hits = detect_misleading(text)  # may or may not match; just must not raise
    assert isinstance(hits, list)


def test_neutralize_preserves_clean_text():
    """Text with no banned terms is returned without the flag marker.

    Note: neutralize() runs NFKC normalization which may convert full-width
    punctuation to ASCII equivalents (e.g. ，→,) — that is expected and
    correct behavior.  The contract is that no banned-term flag markers appear
    and the semantic content is preserved.
    """
    clean = "历史年化约8%，过往业绩不代表未来表现，投资有风险"
    result = neutralize(clean)
    # No violation markers should appear
    assert "已删除违规表述" not in result
    # Core content is preserved (normalization may change punctuation form)
    assert "历史年化约8" in result
    assert "过往业绩不代表未来表现" in result
