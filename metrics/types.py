"""Type definitions for the MoEInfra metrics subsystem.

Defines :class:`MetricsSummary`, a snapshot dataclass produced by
:meth:`~metrics.collector.MetricsCollector.snapshot`.
"""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class MetricsSummary:
    """A point-in-time snapshot of all collected metrics.

    Attributes:
        cache_hits: Total number of successful cache lookups.
        cache_misses: Total number of cache misses.
        evictions: Total number of cache evictions.
        transfers_completed: Total number of finished PCIe transfers.
        total_bytes_transferred: Total bytes moved across the PCIe bus.
        ttft_ms_avg: Average time-to-first-token in milliseconds, or
            ``None`` if no requests have been processed.
        tps_avg: Average tokens-per-second, or ``None`` if no decode
            phases have completed.
        transfer_bandwidth_gbps: Most recent rolling PCIe bandwidth in GB/s.
        gpu_slots_used: Current number of GPU cache slots occupied.
        cpu_slots_used: Current number of CPU cache slots occupied.
    """

    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    transfers_completed: int = 0
    total_bytes_transferred: int = 0
    ttft_ms_avg: Optional[float] = None
    tps_avg: Optional[float] = None
    transfer_bandwidth_gbps: float = 0.0
    gpu_slots_used: int = 0
    cpu_slots_used: int = 0

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a value in ``[0.0, 1.0]``.

        Returns:
            ``cache_hits / (cache_hits + cache_misses)``, or ``0.0`` if no
            requests have been made.
        """
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
