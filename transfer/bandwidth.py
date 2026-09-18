"""Bandwidth monitoring for the MoEInfra transfer subsystem.

:class:`BandwidthMonitor` tracks rolling PCIe throughput over a configurable
time window and exposes helpers for current and peak observed GB/s.
"""
from __future__ import annotations

import collections
import time
from typing import Deque, Tuple


class BandwidthMonitor:
    """Rolling-window PCIe bandwidth tracker.

    Keeps a deque of ``(timestamp, bytes)`` samples from the last
    *window_s* seconds and computes throughput on demand.

    Attributes:
        window_s: Duration (seconds) of the rolling measurement window.
    """

    def __init__(self, window_s: float = 5.0) -> None:
        """Initialise the monitor.

        Args:
            window_s: Width of the rolling window in seconds. Samples older
                than this are pruned automatically on each :meth:`record` call.
        """
        self.window_s: float = window_s
        self._samples: Deque[Tuple[float, int]] = collections.deque()
        self._peak_gbps: float = 0.0

    def record(self, bytes_transferred: int) -> None:
        """Append a new bandwidth sample and prune expired entries.

        Args:
            bytes_transferred: Number of bytes transferred in this sample.

        Returns:
            None
        """
        now: float = time.monotonic()
        self._samples.append((now, bytes_transferred))

        # Prune samples outside the window
        cutoff: float = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        # Update peak
        current = self.current_gbps()
        if current > self._peak_gbps:
            self._peak_gbps = current

    def current_gbps(self) -> float:
        """Return the rolling throughput over the current window.

        Computes total bytes in the window divided by the elapsed window
        duration, converted to GB/s.

        Returns:
            Throughput in GB/s as a float, or ``0.0`` if fewer than two
            samples are available.
        """
        if len(self._samples) < 2:
            return 0.0

        now: float = time.monotonic()
        cutoff: float = now - self.window_s
        window_samples = [s for s in self._samples if s[0] >= cutoff]

        if not window_samples:
            return 0.0

        total_bytes: int = sum(s[1] for s in window_samples)
        elapsed: float = now - window_samples[0][0]
        if elapsed <= 0:
            return 0.0

        return (total_bytes / elapsed) / (1024 ** 3)

    def peak_gbps(self) -> float:
        """Return the peak observed throughput since last :meth:`reset`.

        Returns:
            Peak throughput in GB/s.
        """
        return self._peak_gbps

    def reset(self) -> None:
        """Clear all samples and reset the peak counter.

        Returns:
            None
        """
        self._samples.clear()
        self._peak_gbps = 0.0
