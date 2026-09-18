"""Metrics collector for MoEInfra.

:class:`MetricsCollector` is a thread-safe, in-process metrics store that
accumulates counters and latency samples during inference.  Call
:meth:`snapshot` to get a point-in-time :class:`~metrics.types.MetricsSummary`
and :meth:`reset` to clear all accumulators.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from metrics.types import MetricsSummary


class MetricsCollector:
    """Thread-safe in-process metrics accumulator.

    All ``record_*`` methods are safe to call from multiple threads
    concurrently.

    Attributes:
        enabled: When ``False`` all ``record_*`` calls are no-ops.
    """

    def __init__(
        self,
        enabled: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the collector.

        Args:
            enabled: Set to ``False`` to disable all metric recording (useful
                in latency-sensitive benchmarks).
            logger: Optional pre-configured logger.
        """
        self.enabled: bool = enabled
        self._logger: logging.Logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._reset_state()

    # ------------------------------------------------------------------ #
    # Internal state helpers                                               #
    # ------------------------------------------------------------------ #

    def _reset_state(self) -> None:
        """Zero-initialise all accumulator fields."""
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._evictions: int = 0
        self._transfers_completed: int = 0
        self._total_bytes_transferred: int = 0
        self._ttft_samples: list[float] = []
        self._tps_samples: list[float] = []
        self._transfer_bandwidth_gbps: float = 0.0
        self._gpu_slots_used: int = 0
        self._cpu_slots_used: int = 0

    # ------------------------------------------------------------------ #
    # Record methods                                                       #
    # ------------------------------------------------------------------ #

    def record_cache_hit(self) -> None:
        """Increment the cache-hit counter.

        Returns:
            None
        """
        if not self.enabled:
            return
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        """Increment the cache-miss counter.

        Returns:
            None
        """
        if not self.enabled:
            return
        with self._lock:
            self._cache_misses += 1

    def record_eviction(self) -> None:
        """Increment the eviction counter.

        Returns:
            None
        """
        if not self.enabled:
            return
        with self._lock:
            self._evictions += 1

    def record_transfer(self, bytes_transferred: int, bandwidth_gbps: float) -> None:
        """Record a completed PCIe transfer.

        Args:
            bytes_transferred: Number of bytes moved in this transfer.
            bandwidth_gbps: Observed throughput for this transfer in GB/s.

        Returns:
            None
        """
        if not self.enabled:
            return
        with self._lock:
            self._transfers_completed += 1
            self._total_bytes_transferred += bytes_transferred
            self._transfer_bandwidth_gbps = bandwidth_gbps

    def record_ttft(self, ttft_ms: float) -> None:
        """Record a time-to-first-token measurement.

        Args:
            ttft_ms: TTFT in milliseconds for a single request.

        Returns:
            None
        """
        if not self.enabled:
            return
        with self._lock:
            self._ttft_samples.append(ttft_ms)

    def record_tps(self, tps: float) -> None:
        """Record a tokens-per-second measurement.

        Args:
            tps: Decode throughput for a single request.

        Returns:
            None
        """
        if not self.enabled:
            return
        with self._lock:
            self._tps_samples.append(tps)

    def record_slot_usage(self, gpu_slots: int, cpu_slots: int) -> None:
        """Update the current GPU and CPU slot occupancy counts.

        Args:
            gpu_slots: Number of GPU cache slots currently in use.
            cpu_slots: Number of CPU cache slots currently in use.

        Returns:
            None
        """
        if not self.enabled:
            return
        with self._lock:
            self._gpu_slots_used = gpu_slots
            self._cpu_slots_used = cpu_slots

    # ------------------------------------------------------------------ #
    # Snapshot and reset                                                   #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> MetricsSummary:
        """Return a point-in-time snapshot of all accumulated metrics.

        Returns:
            A :class:`~metrics.types.MetricsSummary` dataclass populated with
            the current counter values and computed averages.
        """
        with self._lock:
            ttft_avg = (
                sum(self._ttft_samples) / len(self._ttft_samples)
                if self._ttft_samples
                else None
            )
            tps_avg = (
                sum(self._tps_samples) / len(self._tps_samples)
                if self._tps_samples
                else None
            )
            return MetricsSummary(
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                evictions=self._evictions,
                transfers_completed=self._transfers_completed,
                total_bytes_transferred=self._total_bytes_transferred,
                ttft_ms_avg=ttft_avg,
                tps_avg=tps_avg,
                transfer_bandwidth_gbps=self._transfer_bandwidth_gbps,
                gpu_slots_used=self._gpu_slots_used,
                cpu_slots_used=self._cpu_slots_used,
            )

    def reset(self) -> None:
        """Clear all accumulated metrics.

        Returns:
            None
        """
        with self._lock:
            self._reset_state()
        self._logger.debug("MetricsCollector reset.")
