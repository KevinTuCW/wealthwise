"""Tests for ConsensusResolver — Task 1.2.

Mirrors shopscout's consensus test suite, adapted to market signal domain.
"""
from __future__ import annotations

import pytest


class TestConsensusResolver:
    def _make_reading(self, source: str, value: float):
        from wealthwise.providers.consensus import Reading
        return Reading(source=source, value=value)

    def test_three_agreeing_sources_high_confidence_not_flagged(self):
        from wealthwise.providers.consensus import ConsensusResolver

        r = ConsensusResolver(threshold=0.2)
        readings = [
            self._make_reading("src_a", 0.18),
            self._make_reading("src_b", 0.18),
            self._make_reading("src_c", 0.18),
        ]
        result = r.resolve(readings)
        assert result.confidence >= 0.9
        assert result.disagreement is False
        assert result.value == pytest.approx(0.18)

    def test_single_source_confidence_capped_at_0_5(self):
        from wealthwise.providers.consensus import ConsensusResolver

        r = ConsensusResolver(threshold=0.2)
        readings = [self._make_reading("only_src", 0.25)]
        result = r.resolve(readings)
        assert result.confidence == pytest.approx(0.5)
        assert result.disagreement is False
        assert result.value == pytest.approx(0.25)

    def test_disagreement_beyond_threshold_is_flagged_and_discounted(self):
        from wealthwise.providers.consensus import ConsensusResolver

        r = ConsensusResolver(threshold=0.2)
        # large spread relative to median
        readings = [
            self._make_reading("src_a", 0.10),
            self._make_reading("src_b", 0.50),  # big outlier
            self._make_reading("src_c", 0.12),
        ]
        result = r.resolve(readings)
        assert result.disagreement is True
        assert result.confidence < 0.8  # discounted

    def test_all_zero_values_full_agreement(self):
        """All-zero edge case — spread should be 0 (full agreement), not division-by-zero."""
        from wealthwise.providers.consensus import ConsensusResolver

        r = ConsensusResolver(threshold=0.2)
        readings = [
            self._make_reading("a", 0.0),
            self._make_reading("b", 0.0),
            self._make_reading("c", 0.0),
        ]
        result = r.resolve(readings)
        assert result.value == pytest.approx(0.0)
        assert result.disagreement is False
        assert result.confidence == pytest.approx(1.0)

    def test_all_equal_non_zero_full_agreement(self):
        from wealthwise.providers.consensus import ConsensusResolver

        r = ConsensusResolver(threshold=0.2)
        readings = [
            self._make_reading("a", 0.15),
            self._make_reading("b", 0.15),
        ]
        result = r.resolve(readings)
        assert result.confidence == pytest.approx(1.0)
        assert result.disagreement is False

    def test_median_zero_with_differing_values_infinite_spread_flagged(self):
        """Median=0 but values differ → spread=inf → disagreement flagged, confidence=0."""
        from wealthwise.providers.consensus import ConsensusResolver

        r = ConsensusResolver(threshold=0.2)
        readings = [
            self._make_reading("a", -0.1),
            self._make_reading("b", 0.0),
            self._make_reading("c", 0.1),
        ]
        result = r.resolve(readings)
        assert result.disagreement is True
        assert result.confidence == pytest.approx(0.0)

    def test_empty_readings_raises(self):
        from wealthwise.providers.consensus import ConsensusResolver

        r = ConsensusResolver(threshold=0.2)
        with pytest.raises(ValueError, match="at least one"):
            r.resolve([])

    def test_two_sources_moderate_spread(self):
        from wealthwise.providers.consensus import ConsensusResolver

        r = ConsensusResolver(threshold=0.3)
        readings = [
            self._make_reading("a", 0.20),
            self._make_reading("b", 0.22),
        ]
        result = r.resolve(readings)
        # small spread → not flagged, reasonably high confidence
        assert result.disagreement is False
        assert result.confidence > 0.8


# ---------------------------------------------------------------------------
# SourceRegistry integration
# ---------------------------------------------------------------------------

class TestSourceRegistry:
    def test_registry_aggregates_readings_across_sources(self):
        from wealthwise.providers.registry import SourceRegistry
        from wealthwise.providers.consensus import ConsensusResolver, Reading

        class FakeProvider:
            def __init__(self, name, readings_map):
                self.name = name
                self._map = readings_map

            def signal_readings(self, signal: str):
                return [Reading(source=self.name, value=v) for v in self._map.get(signal, [])]

        src_a = FakeProvider("a", {"vol": [0.20]})
        src_b = FakeProvider("b", {"vol": [0.22]})

        registry = SourceRegistry(resolver=ConsensusResolver(threshold=0.3))
        registry.register(src_a)
        registry.register(src_b)

        result = registry.resolve("vol")
        assert result.value == pytest.approx(0.21)
        assert result.confidence > 0.8

    def test_registry_single_provider_confidence_capped(self):
        from wealthwise.providers.registry import SourceRegistry
        from wealthwise.providers.consensus import ConsensusResolver, Reading

        class FakeProvider:
            name = "only"
            def signal_readings(self, signal):
                return [Reading(source=self.name, value=0.15)]

        registry = SourceRegistry(resolver=ConsensusResolver(threshold=0.2))
        registry.register(FakeProvider())

        result = registry.resolve("vol")
        assert result.confidence == pytest.approx(0.5)
