"""Tests for the metrics subsystem.

Covers :class:`~metrics.collector.MetricsCollector` and
:class:`~metrics.types.MetricsSummary`.
"""
from __future__ import annotations

import pytest

from metrics.collector import MetricsCollector
from metrics.types import MetricsSummary


class TestMetricsCollector:
    """MetricsCollector tests."""

    def test_initial_snapshot_zeros(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Fresh collector snapshot should have all-zero counters."""
        s = metrics_collector.snapshot()
        assert s.cache_hits == 0
        assert s.cache_misses == 0
        assert s.evictions == 0
        assert s.transfers_completed == 0

    def test_record_cache_hit(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """record_cache_hit should increment cache_hits."""
        metrics_collector.record_cache_hit()
        metrics_collector.record_cache_hit()
        assert metrics_collector.snapshot().cache_hits == 2

    def test_record_cache_miss(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """record_cache_miss should increment cache_misses."""
        metrics_collector.record_cache_miss()
        assert metrics_collector.snapshot().cache_misses == 1

    def test_record_eviction(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """record_eviction should increment evictions."""
        metrics_collector.record_eviction()
        metrics_collector.record_eviction()
        metrics_collector.record_eviction()
        assert metrics_collector.snapshot().evictions == 3

    def test_hit_rate_calculation(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """hit_rate should equal hits / (hits + misses)."""
        metrics_collector.record_cache_hit()
        metrics_collector.record_cache_hit()
        metrics_collector.record_cache_hit()
        metrics_collector.record_cache_miss()
        assert metrics_collector.snapshot().hit_rate == pytest.approx(0.75)

    def test_reset_clears_counters(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """reset() should zero all accumulated metrics."""
        metrics_collector.record_cache_hit()
        metrics_collector.record_cache_miss()
        metrics_collector.reset()
        s = metrics_collector.snapshot()
        assert s.cache_hits == 0
        assert s.cache_misses == 0

    def test_record_ttft_and_snapshot(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """record_ttft should populate ttft_ms_avg in snapshot."""
        metrics_collector.record_ttft(120.0)
        metrics_collector.record_ttft(80.0)
        s = metrics_collector.snapshot()
        assert s.ttft_ms_avg == pytest.approx(100.0)
